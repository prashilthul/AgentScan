"""Access the client has to approve, and the requests the crawler raises for it.

The baseline (discovery and resource configuration) comes with the read-only role the client
deploys. Every other source is off until someone on the client side approves it with
`agentscan approve`, which records who approved it and when. After each scan the crawler
looks at what it still can't settle and asks only for the sources that would help, naming
the workloads and the reason, so each extra bit of access is justified before it is granted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agentscan.models import AccessRequest, SourceStatus, Tier, WorkloadResult

# A workload at or above this confidence is settled: no more access is requested for it.
SETTLED = 0.85


@dataclass(frozen=True)
class Access:
    permissions: list[str]
    reads: str
    never_reads: str
    why: str  # the reason given to the client in a request


CATALOG: dict[str, Access] = {
    "iam": Access(
        [
            "iam:ListRolePolicies",
            "iam:GetRolePolicy",
            "iam:ListAttachedRolePolicies",
            "iam:GetPolicy",
            "iam:GetPolicyVersion",
        ],
        "the policies attached to each workload's execution role",
        "users, groups, credentials",
        "To see what these workloads' roles are allowed to call (Bedrock, model endpoints, "
        "LLM API key secrets). Permissions only point at where to look; they never decide a label.",
    ),
    "cloudtrail": Access(
        [
            "cloudtrail:DescribeTrails",
            "cloudtrail:LookupEvents",
            "s3:ListBucket + s3:GetObject (trail bucket only)",
        ],
        "who called which Bedrock API on which model, and when",
        "request or response bodies",
        "Configuration says these workloads could use a model. CloudTrail shows whether they "
        "actually call Bedrock, since a permission is not usage.",
    ),
    "invocation-logs": Access(
        [
            "bedrock:GetModelInvocationLoggingConfiguration",
            "logs:FilterLogEvents (invocation log group only)",
        ],
        "the structure of each Bedrock model call: tools offered, tool results fed back",
        "prompt and completion text (dropped on read)",
        "These workloads use Bedrock but it's unclear whether they run a tool loop (agent) or "
        "make one call per request (LLM app). Invocation logs show if tool results go back "
        "to the model.",
    ),
    "app-logs": Access(
        ["logs:FilterLogEvents (these workloads' log groups only)"],
        "HTTP calls to known LLM provider hosts and calls per invocation",
        "any other log content",
        "These workloads show signs of model use that CloudTrail can't see, e.g. calling "
        "OpenAI, Anthropic or Gemini directly. Their logs show those calls.",
    ),
    "traces": Access(
        ["xray:GetTraceSummaries + xray:BatchGetTraces (or the team's trace backend)"],
        "span names and GenAI attributes: model calls, tool calls, agent spans",
        "span payloads, prompts",
        "Traces are the only source that shows how a model is called inside one request: "
        "once, or in a loop with tools.",
    ),
}
OPTIONAL_SOURCES = tuple(CATALOG)


def load_grants(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    grants = json.loads(path.read_text())
    return {k: v for k, v in grants.items() if k in CATALOG}


def approve(path: Path, source: str, by: str, note: str = "") -> dict[str, str]:
    if source not in CATALOG:
        raise ValueError(f"unknown source {source!r}; one of {', '.join(OPTIONAL_SOURCES)}")
    if not by.strip():
        raise ValueError("an approval must name the person who approved it")
    grants = load_grants(path)
    grants[source] = {
        "approved_by": by.strip(),
        "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": note,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(grants, indent=2))
    return grants[source]


def _wants(source: str, r: WorkloadResult) -> bool:
    c, ev = r.classification, r.evidence
    if c.confidence >= SETTLED:
        return False
    ai_hint = any(e.ai_weight > 0 for e in ev)
    if source == "iam":
        return bool(r.workload.primary.role_arn)
    if source == "cloudtrail":
        return ai_hint and bool(r.workload.primary.role_arn)
    if source == "invocation-logs":
        bedrock = any(
            e.signal.startswith(("iam.", "cloudtrail.model", "cloudtrail.calls_agent"))
            and e.ai_weight > 0
            for e in ev
        )
        decided = any(e.agentic_weight and e.tier is not Tier.DECLARED for e in ev)
        return bedrock and not decided
    if source == "app-logs":
        via_bedrock = any(e.signal == "cloudtrail.model_calls" for e in ev)
        return ai_hint and not via_bedrock and bool(r.workload.primary.log_groups)
    if source == "traces":
        return ai_hint and c.basis is not Tier.BEHAVIORAL
    return False


def requests(results: list[WorkloadResult], sources: list[SourceStatus]) -> list[AccessRequest]:
    """One request per source the scan didn't get, listing the unsettled workloads it would help."""
    out = []
    for name, acc in CATALOG.items():
        statuses = [s for s in sources if s.name == name]
        if any(s.ok for s in statuses):
            continue
        wanted = [r.workload.name for r in results if _wants(name, r)]
        if not wanted:
            continue
        status = f"approved but failed: {statuses[0].detail}" if statuses else "not approved"
        out.append(
            AccessRequest(
                source=name,
                status=status,
                permissions=acc.permissions,
                why=acc.why,
                workloads=wanted,
            )
        )
    return out
