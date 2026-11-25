"""CloudWatch Logs collectors.

1. Bedrock model-invocation logs: if the account enabled them, each model call is logged with
   its request and response. We keep STRUCTURE only - how many tools were offered, whether
   tool results were fed back, the stop reason - and drop all prompt and completion text.

2. Application logs: scan each workload's log group for HTTP calls to known LLM endpoints.
   This catches workloads that call a provider directly (not via Bedrock), which CloudTrail
   never sees. Lambda START/END markers let us count calls per invocation.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from botocore.exceptions import ClientError

from agentscan.aws import Aws
from agentscan.models import AccountSnapshot, InvocationRecord, LogHit

LLM_ENDPOINTS = {
    "api.anthropic.com": "anthropic",
    "api.openai.com": "openai",
    "generativelanguage.googleapis.com": "google",
    "api.mistral.ai": "mistral",
    "api.cohere.com": "cohere",
    "api.groq.com": "groq",
    "openai.azure.com": "azure-openai",
    "bedrock-runtime": "bedrock",
}
_ENDPOINT_RE = re.compile("|".join(re.escape(h) for h in LLM_ENDPOINTS))
_START_RE = re.compile(r"^START RequestId: (\S+)")
_END_RE = re.compile(r"^END RequestId: (\S+)")


def _filter_events(logs, group: str, start_ms: int):
    paginator = logs.get_paginator("filter_log_events")
    for page in paginator.paginate(logGroupName=group, startTime=start_ms):
        yield from page.get("events", [])


def _start_ms(snap: AccountSnapshot) -> int:
    return int((datetime.now(UTC) - timedelta(hours=snap.lookback_hours)).timestamp() * 1000)


# --------------------------------------------------------------------- invocation logs


def _count_tool_results(messages: list) -> int:
    n = 0
    for m in messages or []:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if "toolResult" in block or block.get("type") == "tool_result":
                n += 1
    return n


def _parse_invocation(entry: dict) -> InvocationRecord:
    inp = (entry.get("input") or {}).get("inputBodyJson") or {}
    out = (entry.get("output") or {}).get("outputBodyJson") or {}
    tools = (inp.get("toolConfig") or {}).get("tools") or inp.get("tools") or []
    return InvocationRecord(
        request_id=entry.get("requestId", ""),
        timestamp=datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")),
        identity_arn=(entry.get("identity") or {}).get("arn", ""),
        operation=entry.get("operation", ""),
        model_id=entry.get("modelId", ""),
        tools_offered=len(tools),
        tool_results_in_input=_count_tool_results(inp.get("messages")),
        stop_reason=out.get("stopReason") or out.get("stop_reason"),
    )


def collect_invocation_logs(aws: Aws, region: str, snap: AccountSnapshot) -> str:
    try:
        cfg = aws.client("bedrock", region).get_model_invocation_logging_configuration()
    except ClientError as e:
        return f"invocation logging config unreadable: {e.response['Error']['Code']}"
    group = ((cfg.get("loggingConfig") or {}).get("cloudWatchConfig") or {}).get("logGroupName")
    if not group:
        return "model invocation logging not enabled"
    logs = aws.client("logs", region)
    n = 0
    for ev in _filter_events(logs, group, _start_ms(snap)):
        try:
            snap.invocations.append(_parse_invocation(json.loads(ev["message"])))
            n += 1
        except (ValueError, KeyError):
            continue
    return f"{n} invocation record(s) from {group}"


# --------------------------------------------------------------------- app logs


def collect_app_logs(aws: Aws, region: str, snap: AccountSnapshot) -> str:
    logs = aws.client("logs", region)
    groups = sorted({g for r in snap.resources if r.region == region for g in r.log_groups})
    scanned = 0
    for group in groups:
        try:
            events = list(_filter_events(logs, group, _start_ms(snap)))
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                continue
            raise
        scanned += 1
        per_endpoint: dict[str, int] = defaultdict(int)
        per_invocation: dict[str, int] = defaultdict(int)
        current: dict[str, str | None] = {}  # log stream -> open Lambda request id
        invocations = 0
        for ev in sorted(events, key=lambda e: e["timestamp"]):
            msg, stream = ev["message"], ev.get("logStreamName", "")
            if m := _START_RE.match(msg):
                current[stream] = m.group(1)
                invocations += 1
                continue
            if _END_RE.match(msg):
                current[stream] = None
                continue
            if m := _ENDPOINT_RE.search(msg):
                per_endpoint[m.group(0)] += 1
                if req := current.get(stream):
                    per_invocation[req] += 1
        for host, count in per_endpoint.items():
            snap.log_hits.append(
                LogHit(
                    log_group=group,
                    endpoint=host,
                    count=count,
                    invocations_seen=invocations,
                    max_calls_per_invocation=max(per_invocation.values(), default=0),
                )
            )
    return f"{scanned} log group(s) scanned"
