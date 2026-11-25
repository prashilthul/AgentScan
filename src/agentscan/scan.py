"""Scan orchestration: collect -> correlate -> extract evidence -> classify -> report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime

from botocore.exceptions import BotoCoreError, ClientError
from httpx import HTTPError

from agentscan import __version__
from agentscan.access import OPTIONAL_SOURCES, requests
from agentscan.audit import normalise
from agentscan.aws import Aws
from agentscan.collectors.cloudtrail import collect_cloudtrail
from agentscan.collectors.discovery import UNCOVERED_COMPUTE, collect_discovery
from agentscan.collectors.iam import collect_iam
from agentscan.collectors.inventory import COLLECTORS
from agentscan.collectors.logs import collect_app_logs, collect_invocation_logs
from agentscan.collectors.traces import TraceSource, collect_traces
from agentscan.correlate import correlate
from agentscan.judge import Judge
from agentscan.models import AccountSnapshot, Report, SourceStatus, WorkloadResult
from agentscan.risk import assess
from agentscan.scoring import classify
from agentscan.signals import extract

__all__ = ["OPTIONAL_SOURCES", "analyse", "collect", "diff"]


def _run(snap: AccountSnapshot, name: str, fn: Callable[[], str | None]) -> None:
    try:
        detail = fn() or ""
        snap.sources.append(SourceStatus(name=name, ok=True, detail=detail))
    except (ClientError, BotoCoreError, HTTPError, OSError) as e:
        snap.sources.append(SourceStatus(name=name, ok=False, detail=f"{type(e).__name__}: {e}"))


def collect(
    aws: Aws,
    regions: list[str],
    lookback_hours: int,
    trace_source: TraceSource | None,
    disabled: set[str],
) -> AccountSnapshot:
    snap = AccountSnapshot(
        account_id=aws.account_id(regions[0]),
        regions=regions,
        collected_at=datetime.now(UTC),
        lookback_hours=lookback_hours,
    )
    for region in regions:
        _run(
            snap, f"discovery@{region}", lambda region=region: collect_discovery(aws, region, snap)
        )
        for name, fn in COLLECTORS.items():
            _run(snap, f"{name}@{region}", lambda fn=fn, region=region: fn(aws, region, snap))
    if "iam" not in disabled:
        _run(snap, "iam", lambda: collect_iam(aws, regions[0], snap))
    for region in regions:
        if "cloudtrail" not in disabled:
            _run(snap, "cloudtrail", lambda region=region: collect_cloudtrail(aws, region, snap))
        if "invocation-logs" not in disabled:
            _run(
                snap,
                "invocation-logs",
                lambda region=region: collect_invocation_logs(aws, region, snap),
            )
        if "app-logs" not in disabled:
            _run(snap, "app-logs", lambda region=region: collect_app_logs(aws, region, snap))
    if trace_source is not None and "traces" not in disabled:
        _run(snap, "traces", lambda: collect_traces(trace_source, snap))
    return snap


def analyse(snap: AccountSnapshot, judge: Judge | None = None) -> Report:
    workloads, unattributed = correlate(snap)
    role_count = Counter(w.primary.role_arn for w in workloads if w.primary.role_arn)
    events = normalise(snap, workloads)
    by_wl: dict[str, list] = {}
    for e in events:
        by_wl.setdefault(e.workload_id or "", []).append(e)
    results = []
    for wl in sorted(workloads, key=lambda w: w.name):
        evidence = extract(wl, snap, role_shared=role_count[wl.primary.role_arn] > 1)
        cls = classify(evidence)
        if judge is not None:
            cls = judge.review(wl, evidence, cls)
        mine = by_wl.get(wl.id, [])
        results.append(
            WorkloadResult(
                workload=wl,
                classification=cls,
                evidence=evidence,
                risk=assess(wl, cls, evidence, mine, snap),
                activity=activity(mine),
            )
        )
    summary = Counter(r.classification.category.value for r in results)
    classified = {res.arn for r in results for res in [r.workload.primary, *r.workload.components]}
    return Report(
        version=__version__,
        generated_at=datetime.now(UTC),
        account_id=snap.account_id,
        regions=snap.regions,
        lookback_hours=snap.lookback_hours,
        sources=snap.sources,
        summary={k: summary.get(k, 0) for k in ("agent", "llm", "service")},
        results=results,
        unattributed_trace_services=unattributed,
        inventory=dict(sorted(Counter(d.type for d in snap.discovered).items())),
        unclassified=[
            d for d in snap.discovered if d.type in UNCOVERED_COMPUTE and d.arn not in classified
        ],
        access_requests=requests(results, snap.sources),
        audit_events=events,
    )


def activity(events: list) -> dict[str, int]:
    """Distinct actions by type. Spans are the fullest record, so when a workload has them
    they're what's counted; otherwise the AWS records for one call are merged by request id."""
    out = {}
    for action in sorted({e.action for e in events}):
        same = [e for e in events if e.action == action]
        same = [e for e in same if e.source == "otel"] or same
        out[action] = len({e.call_id or e.id for e in same})
    if "tool_call" in out:  # spans show the tools running; "requested" would count them twice
        out.pop("tool_requested", None)
    return out


def diff(prev: Report, cur: Report) -> list[str]:
    """What this scan learned compared with the previous one."""
    before = {r.workload.id: r for r in prev.results}
    out = []
    for r in cur.results:
        c, old = r.classification, before.pop(r.workload.id, None)
        if old is None:
            out.append(f"{r.workload.name}: new, {c.category} ({c.confidence:.2f})")
            continue
        o = old.classification
        if o.category != c.category:
            out.append(
                f"{r.workload.name}: {o.category} -> {c.category} "
                f"({o.confidence:.2f} -> {c.confidence:.2f})"
            )
        elif abs(o.confidence - c.confidence) >= 0.05:
            out.append(f"{r.workload.name}: confidence {o.confidence:.2f} -> {c.confidence:.2f}")
    out += [f"{r.workload.name}: no longer found" for r in before.values()]
    granted = [
        f"access granted since last scan: {s}" for s in sorted(set(cur.grants) - set(prev.grants))
    ]
    return granted + out
