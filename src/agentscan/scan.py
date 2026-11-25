"""Scan orchestration: collect -> correlate -> extract evidence -> classify -> report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime

from botocore.exceptions import BotoCoreError, ClientError
from httpx import HTTPError

from agentscan import __version__
from agentscan.aws import Aws
from agentscan.collectors.cloudtrail import collect_cloudtrail
from agentscan.collectors.iam import collect_iam
from agentscan.collectors.inventory import COLLECTORS
from agentscan.collectors.logs import collect_app_logs, collect_invocation_logs
from agentscan.collectors.traces import TraceSource, collect_traces
from agentscan.correlate import correlate
from agentscan.judge import Judge
from agentscan.models import AccountSnapshot, Report, SourceStatus, WorkloadResult
from agentscan.scoring import classify
from agentscan.signals import extract

# Sources that can be switched off, e.g. to measure what each one contributes.
OPTIONAL_SOURCES = ("iam", "cloudtrail", "invocation-logs", "app-logs", "traces")


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
    results = []
    for wl in sorted(workloads, key=lambda w: w.name):
        evidence = extract(wl, snap, role_shared=role_count[wl.primary.role_arn] > 1)
        cls = classify(evidence)
        if judge is not None:
            cls = judge.review(wl, evidence, cls)
        results.append(WorkloadResult(workload=wl, classification=cls, evidence=evidence))
    summary = Counter(r.classification.category.value for r in results)
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
    )
