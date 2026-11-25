"""Test doubles. Used ONLY by the test suite - real lab runs call the real API.

ScriptedModel stands in for the Messages API at the HTTP transport level, so the real
Anthropic SDK still runs end to end (its own OTel spans, its own httpx2 request logs).
It behaves like a tool-using model: while tools are offered it calls them for two rounds,
then answers in text.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict

import httpx
import httpx2
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from agentscan.collectors.traces import Span


def _fill(schema: dict) -> dict:
    out = {}
    for name in schema.get("required", []):
        prop = schema.get("properties", {}).get(name, {})
        if "enum" in prop:
            out[name] = prop["enum"][0]
        elif prop.get("type") in ("number", "integer"):
            out[name] = 1
        else:
            out[name] = "A-1001"
    return out


def scripted_model(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    tools = body.get("tools") or []
    rounds = sum(
        1
        for m in body["messages"]
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
    )
    if tools and rounds < 2:
        t = tools[rounds % len(tools)]
        content = [
            {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:12]}",
                "name": t["name"],
                "input": _fill(t["input_schema"]),
            }
        ]
        stop = "tool_use"
    else:
        content = [{"type": "text", "text": "ok"}]
        stop = "end_turn"
    return httpx2.Response(
        200,
        json={
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 7},
        },
    )


def scripted_gemini(request: httpx.Request) -> httpx.Response:
    """Same behaviour as scripted_model, speaking the Gemini generateContent wire format."""
    body = json.loads(request.content)
    # Like Gemini 3+: a function call sent back without its thought signature is rejected.
    for c in body["contents"]:
        if any("functionCall" in p and not p.get("thoughtSignature") for p in c.get("parts", [])):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": 400,
                        "status": "INVALID_ARGUMENT",
                        "message": "missing thought_signature",
                    }
                },
            )
    decls = [d for t in body.get("tools") or [] for d in t.get("functionDeclarations", [])]
    rounds = sum(
        1
        for c in body["contents"]
        if c.get("role") == "user" and any("functionResponse" in p for p in c.get("parts", []))
    )
    if decls and rounds < 2:
        d = decls[rounds % len(decls)]
        schema = d.get("parametersJsonSchema") or d.get("parameters") or {}
        parts = [
            {"functionCall": {"name": d["name"], "args": _fill(schema)}, "thoughtSignature": "c2ln"}
        ]
    else:
        parts = [{"text": "ok"}]
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 7,
                "totalTokenCount": 19,
            },
            "modelVersion": "gemini-3.5-flash",
        },
    )


class KeepSpans(SpanExporter):
    """In-memory exporter that survives provider shutdown (each workload shuts its own)."""

    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class InMemoryTraceSource:
    """TraceSource over exported spans - mirrors what the Jaeger API returns."""

    def __init__(self, exporter: KeepSpans):
        self.exporter = exporter

    def _converted(self) -> list[Span]:
        out = []
        for s in self.exporter.spans:
            res = dict(s.resource.attributes)
            out.append(
                Span(
                    trace_id=format(s.context.trace_id, "032x"),
                    span_id=format(s.context.span_id, "016x"),
                    name=s.name,
                    start_us=s.start_time // 1000,
                    end_us=s.end_time // 1000,
                    service=str(res.get("service.name", "")),
                    attrs=dict(s.attributes or {}),
                    resource=res,
                )
            )
        return out

    def services(self) -> list[str]:
        return sorted({s.service for s in self._converted()})

    def traces(self, service: str, lookback_hours: int, limit: int) -> list[list[Span]]:
        by_trace = defaultdict(list)
        for s in self._converted():
            by_trace[s.trace_id].append(s)
        return [spans for spans in by_trace.values() if any(s.service == service for s in spans)][
            :limit
        ]
