"""Merge every agent-activity source into one audit trail.

Half the lab's agents push OpenTelemetry spans to a collector; the others only leave what AWS
records out of the box (CloudTrail, Bedrock invocation logs, CloudWatch app logs). Each
source has its own shape. This module turns all of them into AuditEvent rows and works out
which workload each row belongs to, using the same links the classifier uses: service.name
for spans, the execution role (and Lambda session name) for AWS records, the log group for
app logs, and the invoked agent for InvokeAgent / InvokeAgentRuntime.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from agentscan.collectors.traces import Span
from agentscan.models import AGENT_TYPES, AccountSnapshot, AuditEvent, Workload
from agentscan.signals import MODEL_CALL_EVENTS, _role_name, invoked_agent

# Span attributes worth keeping. Everything else (and all message content) is dropped.
_SPAN_KEEP = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.response.finish_reasons",
    "gen_ai.provider.name",
)


def _id(source: str, native: str) -> str:
    return hashlib.sha256(f"{source}:{native}".encode()).hexdigest()[:24]


class _Owners:
    def __init__(self, workloads: list[Workload]):
        self.by_service, self.by_role, self.by_group, self.by_agent = {}, {}, {}, {}
        for wl in workloads:
            for svc in wl.trace_services:
                self.by_service[svc] = wl
            for res in [wl.primary, *wl.components]:
                if role := _role_name(res.role_arn):
                    self.by_role.setdefault(role, []).append(wl)
                for g in res.log_groups:
                    self.by_group[g] = wl
            if wl.primary.type in AGENT_TYPES:
                self.by_agent[wl.primary.arn] = wl
                if agent_id := wl.primary.attributes.get("agent_id"):
                    self.by_agent[agent_id] = wl

    def by_identity(self, principal_arn: str | None) -> Workload | None:
        """assumed-role/<role>/<session> -> the workload running as that role."""
        cands = self.by_role.get(_role_name(principal_arn) or "", [])
        if len(cands) == 1:
            return cands[0]
        session = (principal_arn or "").rsplit("/", 1)[-1]
        named = [w for w in cands if w.name == session]  # Lambda sessions carry the name
        return named[0] if len(named) == 1 else None  # shared role, can't tell: unattributed


def _event(snap: AccountSnapshot, wl: Workload | None, **kw) -> AuditEvent:
    return AuditEvent(
        account_id=snap.account_id,
        workload_id=wl.id if wl else None,
        workload=wl.name if wl else None,
        **kw,
    )


def _from_spans(snap: AccountSnapshot, own: _Owners) -> list[AuditEvent]:
    out = []
    for raw in snap.spans:
        s = Span(**raw)
        if s.is_llm():
            action, target = "model_call", s.attrs.get("gen_ai.request.model") or s.name
        elif s.is_tool():
            action, target = "tool_call", s.attrs.get("gen_ai.tool.name") or s.name
        elif s.is_agent():
            action, target = "agent_step", s.attrs.get("gen_ai.agent.name") or s.name
        else:
            continue  # db queries, HTTP handlers: not AI activity
        detail = {k.removeprefix("gen_ai."): s.attrs[k] for k in _SPAN_KEEP if k in s.attrs}
        detail["duration_ms"] = round((s.end_us - s.start_us) / 1000, 1)
        out.append(
            _event(
                snap,
                own.by_service.get(s.service),
                id=_id("otel", s.span_id),
                occurred_at=datetime.fromtimestamp(s.start_us / 1e6, UTC),
                source="otel",
                action=action,
                target=str(target),
                actor=s.service,
                correlation_id=s.trace_id,
                call_id=s.attrs.get("aws.request_id"),
                detail=detail,
            )
        )
    return out


def _from_cloudtrail(snap: AccountSnapshot, own: _Owners) -> list[AuditEvent]:
    out = []
    for ev in snap.trail_events:
        if agent := invoked_agent(ev):
            wl, action, target = own.by_agent.get(agent), "agent_invoked", agent
            corr = ev.request_parameters.get("runtimeSessionId") or ev.request_parameters.get(
                "sessionId"
            )
        elif ev.event_name in MODEL_CALL_EVENTS:
            wl, action, corr = own.by_identity(ev.principal_arn), "model_call", None
            target = ev.request_parameters.get("modelId", "?")
        else:
            continue
        out.append(
            _event(
                snap,
                wl,
                id=_id("cloudtrail", ev.event_id),
                occurred_at=ev.event_time,
                source="cloudtrail",
                action=action,
                target=str(target),
                actor=ev.principal_arn,
                correlation_id=corr,
                call_id=ev.request_id if action == "model_call" else None,
                detail={"event_name": ev.event_name, "event_source": ev.event_source},
            )
        )
    return out


def _from_invocations(snap: AccountSnapshot, own: _Owners) -> list[AuditEvent]:
    out = []
    for r in snap.invocations:
        wl = own.by_identity(r.identity_arn)
        common = {"occurred_at": r.timestamp, "actor": r.identity_arn, "call_id": r.request_id}
        out.append(
            _event(
                snap,
                wl,
                id=_id("bedrock-invocation-log", r.request_id),
                source="bedrock-invocation-log",
                action="model_call",
                target=r.model_id,
                detail={
                    "operation": r.operation,
                    "tools_offered": r.tools_offered,
                    "tool_results_in_input": r.tool_results_in_input,
                    "stop_reason": r.stop_reason,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                },
                **common,
            )
        )
        # The log shows the model asked for a tool, not that it ran (that shows up as a tool
        # result in the next call), so this is "requested", not "tool_call".
        for i, name in enumerate(r.tools_requested):
            out.append(
                _event(
                    snap,
                    wl,
                    id=_id("bedrock-invocation-log", f"{r.request_id}:tool:{i}"),
                    source="bedrock-invocation-log",
                    action="tool_requested",
                    target=name,
                    detail={},
                    **common,
                )
            )
    return out


def _from_app_logs(snap: AccountSnapshot, own: _Owners) -> list[AuditEvent]:
    return [
        _event(
            snap,
            own.by_group.get(c.log_group),
            id=_id("app-log", f"{c.log_group}:{c.stream}:{c.timestamp.isoformat()}:{i}"),
            occurred_at=c.timestamp,
            source="app-log",
            action="model_call",
            target=c.endpoint,
            actor=c.log_group,
            correlation_id=c.invocation_id,
            detail={"direct_to_provider": True},
        )
        for i, c in enumerate(snap.log_calls)
    ]


def normalise(snap: AccountSnapshot, workloads: list[Workload]) -> list[AuditEvent]:
    own = _Owners(workloads)
    events = (
        _from_spans(snap, own)
        + _from_cloudtrail(snap, own)
        + _from_invocations(snap, own)
        + _from_app_logs(snap, own)
    )
    return sorted(events, key=lambda e: e.occurred_at)
