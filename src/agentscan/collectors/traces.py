"""Trace collector: the only source that shows HOW a workload uses a model.

Spans are recognised by the OpenTelemetry GenAI semantic conventions
(gen_ai.operation.name = chat / execute_tool / invoke_agent) and by the span names the
Anthropic SDK emits natively (anthropic.messages.create, anthropic.custom_tool_use <tool>, ...).
Anything following either convention is understood; the backend is pluggable.

Backends:
  JaegerSource  - Jaeger query API (used locally; any OTLP pipeline can feed it)
  X-Ray         - not implemented here; the TraceSource protocol is the seam for it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import httpx

from agentscan.models import AccountSnapshot, TraceStats

LLM_OPS = {"chat", "text_completion", "generate_content", "embeddings"}
LLM_SPAN_NAMES = ("anthropic.messages.create", "anthropic.messages.stream")
TOOL_SPAN_PREFIXES = (
    "anthropic.custom_tool_use",
    "anthropic.tool_use",
    "anthropic.mcp_tool_use",
    "execute_tool",
)
AGENT_OPS = {"invoke_agent", "create_agent"}
AGENT_SPAN_PREFIXES = ("invoke_agent", "anthropic.messages.tool_runner", "anthropic.session.turn")


@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    start_us: int
    end_us: int
    service: str
    attrs: dict = field(default_factory=dict)
    resource: dict = field(default_factory=dict)

    @property
    def op(self) -> str | None:
        return self.attrs.get("gen_ai.operation.name")

    def is_llm(self) -> bool:
        return self.op in LLM_OPS or self.name.startswith(LLM_SPAN_NAMES)

    def is_tool(self) -> bool:
        return self.op == "execute_tool" or self.name.startswith(TOOL_SPAN_PREFIXES)

    def is_agent(self) -> bool:
        return self.op in AGENT_OPS or self.name.startswith(AGENT_SPAN_PREFIXES)


class TraceSource(Protocol):
    def services(self) -> list[str]: ...

    def traces(self, service: str, lookback_hours: int, limit: int) -> list[list[Span]]: ...


class JaegerSource:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def services(self) -> list[str]:
        r = self.http.get("/api/services")
        r.raise_for_status()
        return [s for s in (r.json().get("data") or []) if s != "jaeger-all-in-one"]

    def traces(self, service: str, lookback_hours: int, limit: int) -> list[list[Span]]:
        r = self.http.get(
            "/api/traces",
            params={"service": service, "lookback": f"{lookback_hours}h", "limit": limit},
        )
        r.raise_for_status()
        out = []
        for t in r.json().get("data") or []:
            procs = {
                pid: (p["serviceName"], {tag["key"]: tag["value"] for tag in p.get("tags", [])})
                for pid, p in t.get("processes", {}).items()
            }
            spans = []
            for s in t.get("spans", []):
                svc, res = procs.get(s.get("processID"), ("", {}))
                spans.append(
                    Span(
                        trace_id=s["traceID"],
                        span_id=s["spanID"],
                        name=s["operationName"],
                        start_us=s["startTime"],
                        end_us=s["startTime"] + s["duration"],
                        service=svc,
                        attrs={tag["key"]: tag["value"] for tag in s.get("tags", [])},
                        resource=res,
                    )
                )
            out.append(spans)
        return out


def summarise(service: str, traces: list[list[Span]]) -> TraceStats:
    st = TraceStats(service=service)
    agents, tools, models = set(), set(), set()
    for spans in traces:
        own = [s for s in spans if s.service == service]
        if not own:
            continue
        if not st.resource_attrs:
            st.resource_attrs = {k: str(v) for k, v in own[0].resource.items()}
        st.traces += 1
        llm = [s for s in own if s.is_llm()]
        tool = [s for s in own if s.is_tool()]
        agent = [s for s in own if s.is_agent()]
        if llm:
            st.traces_with_llm += 1
            if len(st.sample_trace_ids) < 5:
                st.sample_trace_ids.append(own[0].trace_id)
        if len(llm) == 1:
            st.single_llm_call_traces += 1
        st.max_llm_calls_per_trace = max(st.max_llm_calls_per_trace, len(llm))
        if tool:
            st.traces_with_tools += 1
        if agent:
            st.traces_with_agent_span += 1
        # A loop: some model call starts after a tool call finished (the result was fed back).
        if tool and any(c.start_us >= min(t.end_us for t in tool) for c in llm):
            st.traces_with_loop += 1
        for s in agent:
            if name := s.attrs.get("gen_ai.agent.name"):
                agents.add(str(name))
        for s in tool:
            tools.add(str(s.attrs.get("gen_ai.tool.name") or s.name.split(" ", 1)[-1]))
        for s in llm:
            if m := s.attrs.get("gen_ai.request.model"):
                models.add(str(m))
    st.agent_names, st.tool_names, st.models = sorted(agents), sorted(tools), sorted(models)
    return st


def collect_traces(source: TraceSource, snap: AccountSnapshot, limit_per_service: int = 50) -> str:
    services = source.services()
    seen: set[str] = set()
    for svc in services:
        traces = source.traces(svc, snap.lookback_hours, limit_per_service)
        snap.trace_stats[svc] = summarise(svc, traces)
        for spans in traces:
            for s in spans:
                if s.service == svc and s.span_id not in seen:
                    seen.add(s.span_id)
                    snap.spans.append(asdict(s))
    return f"{len(services)} traced service(s)"
