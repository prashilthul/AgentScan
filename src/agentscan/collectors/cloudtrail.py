"""Read CloudTrail records for AI service calls.

Preferred path: the trail's S3 delivery (AWSLogs/<acct>/CloudTrail/<region>/YYYY/MM/DD/*.json.gz),
which includes data events such as bedrock InvokeAgent. Fallback: LookupEvents (management
events only, 90 days).

Only the call metadata is kept: who called which API on which model, and when.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta

from botocore.exceptions import BotoCoreError, ClientError

from agentscan.aws import Aws
from agentscan.models import AccountSnapshot, TrailEvent

AI_EVENT_SOURCES = ("bedrock", "sagemaker")


def _to_event(rec: dict) -> TrailEvent | None:
    source = rec.get("eventSource", "")
    if not source.startswith(AI_EVENT_SOURCES):
        return None
    event_id, event_time = rec.get("eventID"), rec.get("eventTime")
    if not event_id or not event_time:
        return None
    try:
        when = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    ident = rec.get("userIdentity") or {}
    issuer = ((ident.get("sessionContext") or {}).get("sessionIssuer") or {}).get("arn")
    principal = ident.get("arn", "")
    session = principal.rsplit("/", 1)[-1] if ident.get("type") == "AssumedRole" else None
    return TrailEvent(
        event_id=event_id,
        event_time=when,
        event_source=source,
        event_name=rec.get("eventName", ""),
        principal_arn=principal,
        role_arn=issuer,
        session_name=session,
        request_parameters=rec.get("requestParameters") or {},
        request_id=rec.get("requestID"),
    )


def _days(lookback_hours: int) -> list[datetime]:
    now = datetime.now(UTC)
    n = lookback_hours // 24 + 1
    return [now - timedelta(days=i) for i in range(n)]


def collect_cloudtrail(aws: Aws, region: str, snap: AccountSnapshot) -> str:
    ct = aws.client("cloudtrail", region)
    s3 = aws.client("s3", region)
    cutoff = datetime.now(UTC) - timedelta(hours=snap.lookback_hours)
    trails = ct.describe_trails().get("trailList", [])
    objects = 0
    trail_error: str | None = None
    try:
        for trail in trails:
            bucket = trail.get("S3BucketName")
            if not bucket:
                continue
            prefix = f"{trail['S3KeyPrefix']}/" if trail.get("S3KeyPrefix") else ""
            for day in _days(snap.lookback_hours):
                key_prefix = (
                    f"{prefix}AWSLogs/{snap.account_id}/CloudTrail/{region}/{day:%Y/%m/%d}/"
                )
                for page in s3.get_paginator("list_objects_v2").paginate(
                    Bucket=bucket, Prefix=key_prefix
                ):
                    for obj in page.get("Contents", []):
                        body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                        objects += 1
                        for rec in json.loads(gzip.decompress(body)).get("Records", []):
                            ev = _to_event(rec)
                            if ev and ev.event_time >= cutoff:
                                snap.trail_events.append(ev)
    except (BotoCoreError, ClientError, OSError, ValueError) as e:
        trail_error = f"{type(e).__name__}: {e}"[:300]

    if trails and objects and not trail_error:
        if region not in snap.cloudtrail_trail_regions:
            snap.cloudtrail_trail_regions.append(region)
        return f"{len(trails)} trail(s), {objects} log file(s) read"

    # Fall back to the 90-day management event history. It can show that a call happened, but
    # not that one did not: Bedrock InvokeModel/Converse are data events and never appear here.
    for page in ct.get_paginator("lookup_events").paginate(
        LookupAttributes=[
            {"AttributeKey": "EventSource", "AttributeValue": "bedrock.amazonaws.com"}
        ],
        StartTime=cutoff,
    ):
        for e in page.get("Events", []):
            ev = _to_event(json.loads(e["CloudTrailEvent"]))
            if ev:
                snap.trail_events.append(ev)
    if trails:
        why = trail_error or "no readable log files in the window"
        return (
            f"{len(trails)} trail(s) but {why}; used LookupEvents "
            "(management events only, no Bedrock data events)"
        )
    return "no trail configured; used LookupEvents (management events only)"
