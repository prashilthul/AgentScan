"""Data model shared by collectors, the correlator, the classifier and the report.

Flow:  collectors -> AccountSnapshot -> correlate() -> list[Workload]
       signals.extract(workload) -> list[Evidence] -> scoring.classify() -> Classification
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Category(StrEnum):
    AGENT = "agent"  # multi-step: LLM calls + tool use in a loop, or agents calling agents
    LLM = "llm"  # uses a model, one call per request (input -> model -> output)
    SERVICE = "service"  # no LLM usage (classical ML included)


class Tier(StrEnum):
    """How strong a piece of evidence is, by where it came from.

    declared   - configuration says the workload *could* use AI (IAM, env var names, images)
    observed   - audit/app logs show it *did* call a model (CloudTrail, invocation logs, app logs)
    behavioral - traces show *how* it called the model (one call vs. a tool loop)
    """

    DECLARED = "declared"
    OBSERVED = "observed"
    BEHAVIORAL = "behavioral"


TIER_RANK = {Tier.DECLARED: 1, Tier.OBSERVED: 2, Tier.BEHAVIORAL: 3}


# --------------------------------------------------------------------------- inventory


class Resource(BaseModel):
    arn: str
    type: str  # lambda.function | ecs.service | sagemaker.endpoint | bedrock.agent
    name: str
    region: str
    role_arn: str | None = None
    # Metadata only. Env var VALUES are never stored, only names.
    env_var_names: list[str] = Field(default_factory=list)
    secret_refs: list[str] = Field(default_factory=list)  # secret ARNs/names referenced
    images: list[str] = Field(default_factory=list)  # container images / model images
    layers: list[str] = Field(default_factory=list)
    log_groups: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class RolePolicy(BaseModel):
    role_arn: str
    statements: list[dict[str, Any]]  # normalised: {"effect", "actions": [...], "resources": [...]}


class TrailEvent(BaseModel):
    """The subset of a CloudTrail record the classifier needs."""

    event_id: str
    event_time: datetime
    event_source: str
    event_name: str
    principal_arn: str  # assumed-role ARN or user ARN
    role_arn: str | None  # issuing role, when the principal is an assumed role
    session_name: str | None
    request_parameters: dict[str, Any] = Field(default_factory=dict)


class InvocationRecord(BaseModel):
    """Structure-only view of one Bedrock model-invocation log entry (no prompt text kept)."""

    request_id: str
    timestamp: datetime
    identity_arn: str
    operation: str
    model_id: str
    tools_offered: int  # number of tools in toolConfig
    tool_results_in_input: int  # toolResult blocks fed back = continuation of a tool loop
    stop_reason: str | None


class LogHit(BaseModel):
    log_group: str
    endpoint: str  # e.g. api.anthropic.com
    count: int
    invocations_seen: int  # Lambda START lines seen in the window
    max_calls_per_invocation: int


class TraceStats(BaseModel):
    """Per-service aggregate over the traces the trace backend returned."""

    service: str
    resource_attrs: dict[str, str] = Field(default_factory=dict)
    traces: int = 0
    traces_with_llm: int = 0
    traces_with_tools: int = 0
    traces_with_loop: int = 0  # an LLM call starts after a tool call ended, same trace
    traces_with_agent_span: int = 0
    max_llm_calls_per_trace: int = 0
    single_llm_call_traces: int = 0
    agent_names: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    sample_trace_ids: list[str] = Field(default_factory=list)


class SourceStatus(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class AccountSnapshot(BaseModel):
    account_id: str
    regions: list[str]
    collected_at: datetime
    lookback_hours: int
    resources: list[Resource] = Field(default_factory=list)
    role_policies: dict[str, RolePolicy] = Field(default_factory=dict)
    trail_events: list[TrailEvent] = Field(default_factory=list)
    invocations: list[InvocationRecord] = Field(default_factory=list)
    log_hits: list[LogHit] = Field(default_factory=list)
    trace_stats: dict[str, TraceStats] = Field(default_factory=dict)  # keyed by service.name
    secret_names: list[str] = Field(default_factory=list)
    # Bedrock agent id -> Lambda ARNs that execute its action groups
    agent_action_lambdas: dict[str, list[str]] = Field(default_factory=dict)
    sources: list[SourceStatus] = Field(default_factory=list)

    def source_ok(self, name: str) -> bool:
        return any(s.name == name and s.ok for s in self.sources)


# --------------------------------------------------------------------------- analysis


class Evidence(BaseModel):
    id: str
    source: str  # where it came from, e.g. "cloudtrail", "iam", "traces"
    signal: str  # stable machine key, e.g. "trace.tool_loop"
    tier: Tier
    detail: str  # one human-readable sentence
    # Two-stage model (see scoring.py):
    #   ai_weight      > 0 pushes towards "uses an LLM", < 0 towards "plain service"
    #   agentic_weight > 0 pushes towards "agent",     < 0 towards "single-call LLM"
    ai_weight: float = 0.0
    agentic_weight: float = 0.0
    refs: dict[str, Any] = Field(default_factory=dict)  # pointers an auditor can follow

    @property
    def supports(self) -> Category:
        if self.agentic_weight > 0:
            return Category.AGENT
        if self.ai_weight < 0:
            return Category.SERVICE
        return Category.LLM


class Workload(BaseModel):
    id: str  # primary resource ARN
    name: str
    primary: Resource
    components: list[Resource] = Field(default_factory=list)
    trace_services: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    category: Category
    confidence: float
    cited_evidence: list[str]
    rationale: str
    model: str


class Classification(BaseModel):
    category: Category
    confidence: float
    probabilities: dict[Category, float]
    p_ai: float
    p_agentic: float
    basis: Tier | None  # strongest tier among evidence backing the decision
    method: str  # "rules" | "rules+judge"
    rationale: str
    rules_category: Category
    judge: JudgeVerdict | None = None


class WorkloadResult(BaseModel):
    workload: Workload
    classification: Classification
    evidence: list[Evidence]


class Report(BaseModel):
    tool: str = "agentscan"
    version: str
    generated_at: datetime
    account_id: str
    regions: list[str]
    lookback_hours: int
    sources: list[SourceStatus]
    summary: dict[str, int]
    results: list[WorkloadResult]
    unattributed_trace_services: list[str] = Field(default_factory=list)
