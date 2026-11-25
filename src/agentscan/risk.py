"""Risk tier for each AI workload, with every factor tied to OWASP and MITRE ATLAS.

The tier comes from what a workload can do and what it was seen doing, not from its label
alone: an agent that only reads is less risky than an LLM app that can write to a database.
Each factor adds points, names the reason, and cites the OWASP Top 10 for LLM Applications
(2026 edition) entry and the MITRE ATLAS techniques it corresponds to. The mapping is ours,
following the published OWASP-to-ATLAS crosswalks, and the points are a judgement call
written down so they can be argued with.

    high >= 6     medium >= 3     low otherwise     (services aren't tiered)
"""

from __future__ import annotations

import re

from agentscan.models import (
    AccountSnapshot,
    AuditEvent,
    Category,
    Classification,
    Evidence,
    Risk,
    RiskFactor,
    Workload,
)

_NOT_WRITES = ("logs:", "xray:", "bedrock:invoke", "bedrock:converse", "bedrock-agentcore:invoke")
_WRITE_VERB = re.compile(
    r"^(put|update|delete|create|send|start|stop|reboot|terminate|publish|modify|attach|"
    r"detach|run|execute|approve|invoke)|\*$"
)
_READ_DATA = re.compile(r"^(s3:get|dynamodb:(get|query|scan|batchget)|secretsmanager:get|rds)")
# Tool names that start with an action verb (issue_refund, restart_service), not ones that
# merely mention one (refund_policy only reads).
_ACTION_TOOL = re.compile(
    r"^(issue|approve|reject|flag|restart|delete|update|create|send|pay|transfer|write|post|"
    r"cancel|execute|run|terminate|submit|set)(_|$)",
    re.I,
)
_DESTRUCTIVE = re.compile(r"(delete|terminate|restart|refund|transfer|pay|drop|cancel)", re.I)
_STEP_LIMIT_ENV = re.compile(r"MAX_(STEPS|ITERATIONS|TURNS)", re.I)

TIERS = ((6, "high"), (3, "medium"), (0, "low"))


def _actions(snap: AccountSnapshot, role_arn: str | None) -> list[str]:
    pol = snap.role_policies.get(role_arn or "")
    if not pol:
        return []
    return [a for st in pol.statements if st["effect"] == "Allow" for a in st["actions"]]


def assess(
    wl: Workload,
    cls: Classification,
    evidence: list[Evidence],
    events: list[AuditEvent],
    snap: AccountSnapshot,
) -> Risk | None:
    if cls.category is Category.SERVICE:
        return None
    agent = cls.category is Category.AGENT
    actions = [a for r in [wl.primary, *wl.components] for a in _actions(snap, r.role_arn)]
    writes = sorted(
        {
            a
            for a in actions
            if not a.startswith(_NOT_WRITES) and _WRITE_VERB.search(a.split(":")[-1])
        }
    )
    reads = sorted({a for a in actions if _READ_DATA.search(a)})
    # shortcut: "does this tool act?" is read from its name; a tool called `decide` that
    # approves invoices is missed. Tool schemas from the invocation logs would do better.
    tools = sorted({e.target for e in events if e.action in ("tool_call", "tool_requested")})
    acting_tools = [t for t in tools if _ACTION_TOOL.search(t)]
    factors: list[RiskFactor] = []

    def add(key, points, why, owasp, atlas):
        factors.append(RiskFactor(key=key, points=points, why=why, owasp=owasp, atlas=atlas))

    if agent:
        add(
            "autonomy",
            2,
            "The model decides which actions to take and in what order.",
            ["LLM03:2026 Excessive Agency"],
            ["AML.T0053 AI Agent Tool Invocation"],
        )
    if writes or acting_tools:
        seen = f" Tools seen acting: {', '.join(acting_tools)}." if acting_tools else ""
        perms = f" Role can: {', '.join(writes[:6])}." if writes else ""
        destructive = any(_DESTRUCTIVE.search(t) for t in acting_tools) or any(
            _DESTRUCTIVE.search(a) for a in writes
        )
        add(
            "can_act",
            3 if agent else 2,
            "Can change things outside itself." + perms + seen,
            ["LLM03:2026 Excessive Agency"],
            ["AML.T0053 AI Agent Tool Invocation"]
            + (["AML.T0101 Data Destruction via AI Agent Tool Invocation"] if destructive else []),
        )
    if reads:
        add(
            "sensitive_data",
            2,
            f"Can read data stores or secrets: {', '.join(reads[:6])}.",
            ["LLM02:2026 Sensitive Information Disclosure"],
            ["AML.T0057 LLM Data Leakage"]
            + (["AML.T0086 Exfiltration via AI Agent Tool Invocation"] if agent else []),
        )
    if agent and (reads or tools):
        add(
            "untrusted_input",
            1,
            "Feeds tool results or retrieved data back into the model, so instructions hidden "
            "in that data can steer it.",
            ["LLM01:2026 Prompt Injection"],
            ["AML.T0051.001 LLM Prompt Injection: Indirect"],
        )
    direct = [
        e
        for e in evidence
        if e.signal in ("applog.llm_endpoint", "config.llm_secret_ref", "iam.llm_secret_access")
    ]
    if direct:
        add(
            "external_provider",
            1,
            "Calls a model provider outside the AWS account directly; data leaves the account "
            "and the dependency isn't governed by Bedrock controls.",
            ["LLM04:2026 Supply Chain", "LLM02:2026 Sensitive Information Disclosure"],
            ["AML.T0057 LLM Data Leakage"],
        )
    env = [n for r in [wl.primary, *wl.components] for n in r.env_var_names]
    if agent and not any(_STEP_LIMIT_ENV.search(n) for n in env):
        add(
            "no_step_limit",
            1,
            "No step or iteration limit is configured, so a loop can run up cost unchecked.",
            ["LLM06:2026 Unbounded Consumption"],
            ["AML.T0034 Cost Harvesting"],
        )
    if agent and not any(e.source == "otel" for e in events):
        add(
            "audit_gap",
            1,
            "Not instrumented: its individual tool actions are only partly visible in AWS logs.",
            [],
            [],
        )
    score = sum(f.points for f in factors)
    tier = next(name for floor, name in TIERS if score >= floor)
    return Risk(tier=tier, score=score, factors=factors)
