"""Gemini free-tier failover: next model on a spent daily quota, without breaking a tool loop."""

import base64
import json

import httpx
from opentelemetry import trace

from lab.shim.providers import GeminiProvider
from lab.workloads._loop import run_agent, tool
from tests.fakes import scripted_gemini

DAILY_QUOTA = {
    "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "message": "quota exceeded",
        "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}],
    }
}


def test_spent_model_hands_over_mid_loop_with_placeholder_signature(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LAB_MIN_INTERVAL", "0")
    seen = []

    def transport(request: httpx.Request) -> httpx.Response:
        model = request.url.path.split("/models/")[1].split(":")[0]
        seen.append((model, json.loads(request.content)))
        if model == "model-a" and len(seen) > 1:  # signs the first tool call, then runs dry
            return httpx.Response(429, json=DAILY_QUOTA)
        return scripted_gemini(request)

    provider = GeminiProvider(
        trace.get_tracer("test"),
        model="model-a,model-b",
        httpx_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    rt = type(
        "Rt",
        (),
        {
            "llm": lambda self, **kw: provider.create(**kw),
            "agent_span": lambda self, name: _null(),
            "tool_span": lambda self, name: _null(),
        },
    )()
    tools = [tool("lookup", "Look something up.", {"q": {"type": "string"}}, ["q"])]
    run_agent(rt, "tester", "be brief", "find it", tools, {"lookup": lambda q: "found"})

    assert [m for m, _ in seen][:3] == ["model-a", "model-a", "model-b"]
    assert provider.models == ["model-b"]
    sent = [p for c in seen[2][1]["contents"] for p in c["parts"] if "functionCall" in p]
    assert base64.b64decode(sent[0]["thoughtSignature"]) == b"skip_thought_signature_validator"


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
