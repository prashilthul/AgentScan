"""Model providers for the lab workloads.

Workload code speaks one shape (Anthropic-style messages, tools and content blocks). Each
provider turns that into a real API call and returns a response with the same shape:
    resp.content      -> blocks with .type == "text" | "tool_use" (.text / .id .name .input)
    resp.stop_reason  -> "tool_use" | "end_turn"
    resp.usage        -> .input_tokens / .output_tokens

  gemini     Google AI Studio free tier (default). The SDK has no native OTel, so we emit
             the OTel GenAI `chat` span ourselves, the way OpenLLMetry-style instrumentation does.
  anthropic  Anthropic API. The SDK emits its own `anthropic.messages.create` spans.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel


class Block(BaseModel):
    type: Literal["text", "tool_use"]
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = {}


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Response(BaseModel):
    model: str
    content: list[Block]
    stop_reason: str
    usage: Usage


def _as_dict(b) -> dict | str:
    return b.model_dump() if hasattr(b, "model_dump") else b


# --------------------------------------------------------------------------- gemini


class _RateLimiter:
    """Free tier allows a handful of requests per minute: space calls out, process-wide."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self) -> None:
        with self.lock:
            delay = self.last + self.min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


_LIMITER = _RateLimiter(float(os.environ.get("LAB_MIN_INTERVAL", "0")))


class GeminiProvider:
    name = "gemini"

    def __init__(self, tracer, model: str | None = None, httpx_client=None):
        from google import genai
        from google.genai import types

        self.types = types
        self.model = model or os.environ.get("LAB_GEMINI_MODEL", "gemini-2.5-flash")
        opts = types.HttpOptions(httpx_client=httpx_client) if httpx_client is not None else None
        self.client = genai.Client(http_options=opts)
        self.tracer = tracer
        _LIMITER.min_interval = float(os.environ.get("LAB_MIN_INTERVAL", "0"))

    def _contents(self, messages: list[dict]):
        t = self.types
        names: dict[str, str] = {}  # tool_use id -> tool name (Gemini answers by name)
        out = []
        for m in messages:
            content = m["content"]
            blocks = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else [_as_dict(b) for b in content]
            )
            parts = []
            for b in blocks:
                if b["type"] == "text" and b.get("text"):
                    parts.append(t.Part(text=b["text"]))
                elif b["type"] == "tool_use":
                    names[b["id"]] = b["name"]
                    parts.append(
                        t.Part(
                            function_call=t.FunctionCall(
                                id=b["id"], name=b["name"], args=b["input"]
                            )
                        )
                    )
                elif b["type"] == "tool_result":
                    parts.append(
                        t.Part(
                            function_response=t.FunctionResponse(
                                id=b["tool_use_id"],
                                name=names.get(b["tool_use_id"], "tool"),
                                response={"error" if b.get("is_error") else "result": b["content"]},
                            )
                        )
                    )
            out.append(t.Content(role="model" if m["role"] == "assistant" else "user", parts=parts))
        return out

    def create(self, *, messages, system=None, tools=None, max_tokens=2048):
        t = self.types
        config = t.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=[
                t.Tool(
                    function_declarations=[
                        t.FunctionDeclaration(
                            name=x["name"],
                            description=x.get("description", ""),
                            parameters_json_schema=x["input_schema"],
                        )
                        for x in tools
                    ]
                )
            ]
            if tools
            else None,
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "gcp.gemini",
            "gen_ai.request.model": self.model,
        }
        with self.tracer.start_as_current_span(f"chat {self.model}", attributes=attrs) as span:
            resp = self._call(contents=self._contents(messages), config=config)
            usage = resp.usage_metadata
            out = Response(
                model=self.model,
                content=self._blocks(resp),
                stop_reason="end_turn",
                usage=Usage(
                    input_tokens=(usage and usage.prompt_token_count) or 0,
                    output_tokens=(usage and usage.candidates_token_count) or 0,
                ),
            )
            if any(b.type == "tool_use" for b in out.content):
                out.stop_reason = "tool_use"
            span.set_attribute("gen_ai.usage.input_tokens", out.usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", out.usage.output_tokens)
            span.set_attribute("gen_ai.response.finish_reasons", [out.stop_reason])
        return out

    def _call(self, **kw):
        from google.genai import errors

        for attempt in range(6):
            _LIMITER.wait()
            try:
                return self.client.models.generate_content(model=self.model, **kw)
            except errors.APIError as e:
                if e.code not in (429, 500, 503) or attempt == 5:
                    raise
                time.sleep(min(60, 5 * 2**attempt))  # free-tier quota: back off and retry
        raise RuntimeError("unreachable")

    @staticmethod
    def _blocks(resp) -> list[Block]:
        blocks: list[Block] = []
        cand = (resp.candidates or [None])[0]
        for p in cand.content.parts if cand and cand.content and cand.content.parts else []:
            if p.function_call:
                fc = p.function_call
                blocks.append(
                    Block(
                        type="tool_use",
                        id=fc.id or f"call_{uuid.uuid4().hex[:12]}",
                        name=fc.name,
                        input=dict(fc.args or {}),
                    )
                )
            elif p.text and not getattr(p, "thought", False):
                blocks.append(Block(type="text", text=p.text))
        return blocks


# --------------------------------------------------------------------------- anthropic


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, tracer_provider=None, model: str | None = None, http_client=None):
        import anthropic

        self.model = model or os.environ.get("LAB_ANTHROPIC_MODEL", "claude-haiku-5-5")
        kw: dict = {
            "open_telemetry": {"tracer_provider": tracer_provider} if tracer_provider else False
        }
        if http_client is not None:
            kw["http_client"] = http_client
        self.client = anthropic.Anthropic(**kw)

    def create(self, **request):
        return self.client.messages.create(model=self.model, **request)
