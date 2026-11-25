"""Turn raw snapshot facts into weighted Evidence for one workload.

Every weight lives in WEIGHTS so the whole policy is auditable in one place.
  ai       > 0 -> uses an LLM        < 0 -> plain service
  agentic  > 0 -> agent (tool loop)  < 0 -> single-call LLM app

Deliberately NOT evidence: resource names and descriptions. Names lie ("ai-helper-svc"
can be a CRUD API); only configuration and behaviour count.
"""

from __future__ import annotations

import re
from collections import Counter

from agentscan.models import AccountSnapshot, Evidence, Tier, Workload

# (tier, ai_weight, agentic_weight)
WEIGHTS: dict[str, tuple[Tier, float, float]] = {
    # declared: configuration says it could
    "resource.bedrock_agent": (Tier.DECLARED, 4.0, 4.0),
    "iam.model_invoke": (Tier.DECLARED, 1.0, 0.0),
    "iam.agent_invoke": (Tier.DECLARED, 1.0, 0.75),
    "iam.wildcard_ai": (Tier.DECLARED, 0.4, 0.0),
    "iam.sagemaker_invoke": (Tier.DECLARED, 0.3, 0.0),
    "iam.llm_secret_access": (Tier.DECLARED, 1.2, 0.0),
    "config.llm_env": (Tier.DECLARED, 1.2, 0.0),
    "config.agent_env": (Tier.DECLARED, 0.3, 0.6),
    "config.llm_secret_ref": (Tier.DECLARED, 1.2, 0.0),
    "config.llm_sdk": (Tier.DECLARED, 1.0, 0.0),
    "config.agent_framework": (Tier.DECLARED, 1.2, 1.0),
    "config.llm_serving_image": (Tier.DECLARED, 2.0, -0.5),
    "config.classical_ml_image": (Tier.DECLARED, -1.0, 0.0),
    # observed: audit / app logs say it did
    "cloudtrail.model_calls": (Tier.OBSERVED, 3.0, 0.0),
    "cloudtrail.agent_invocations": (Tier.OBSERVED, 3.0, 3.0),
    "cloudtrail.calls_agent": (Tier.OBSERVED, 2.0, 0.5),
    "cloudtrail.permission_unused": (Tier.OBSERVED, -2.0, 0.0),
    "invocation.tool_loop": (Tier.OBSERVED, 3.0, 3.0),
    "invocation.tools_no_loop": (Tier.OBSERVED, 2.0, -1.5),
    "invocation.single_shot": (Tier.OBSERVED, 2.0, -1.5),
    "applog.llm_endpoint": (Tier.OBSERVED, 2.5, 0.0),
    "applog.multi_call_invocation": (Tier.OBSERVED, 0.0, 1.5),
    "applog.single_call_invocation": (Tier.OBSERVED, 0.0, -1.0),
    # behavioral: traces show how
    "trace.llm_calls": (Tier.BEHAVIORAL, 3.0, 0.0),
    "trace.no_llm": (Tier.BEHAVIORAL, -3.0, 0.0),
    "trace.tool_loop": (Tier.BEHAVIORAL, 0.0, 3.5),
    "trace.agent_spans": (Tier.BEHAVIORAL, 0.0, 2.5),
    "trace.multi_agent": (Tier.BEHAVIORAL, 0.0, 1.0),
    "trace.single_call": (Tier.BEHAVIORAL, 0.0, -3.0),
}

LLM_ENV_RE = re.compile(
    r"(OPENAI|ANTHROPIC|CLAUDE|GEMINI|MISTRAL|COHERE|GROQ|TOGETHER|HUGGINGFACE|HF)_?(API_?KEY|TOKEN)"
    r"|^(LLM|BEDROCK|GENAI)_|MODEL_ID$|^MODEL_NAME$|_MODEL_ID$|^HF_MODEL_ID$",
    re.I,
)
AGENT_ENV_RE = re.compile(
    r"^AGENT_|_AGENT_|MAX_(STEPS|ITERATIONS|TURNS)|TOOLS?_(ENABLED|CONFIG)", re.I
)
LLM_SECRET_RE = re.compile(r"(openai|anthropic|claude|gemini|mistral|cohere|llm|genai)", re.I)
LLM_SDK_RE = re.compile(r"(anthropic|openai|bedrock-runtime|litellm|google-genai|mistralai)", re.I)
AGENT_FW_RE = re.compile(
    r"(langgraph|langchain|crewai|autogen|strands|llama-?index|semantic-kernel|agentcore|pydantic-ai)",
    re.I,
)
LLM_SERVING_RE = re.compile(
    r"(huggingface-pytorch-tgi|tgi-inference|djl-(inference|lmi)|vllm|lmi)", re.I
)
CLASSICAL_ML_RE = re.compile(
    r"(xgboost|sklearn|scikit|linear-learner|kmeans|randomcutforest|lightgbm|catboost)", re.I
)
MODEL_CALL_EVENTS = {
    "InvokeModel",
    "InvokeModelWithResponseStream",
    "Converse",
    "ConverseStream",
}
AGENT_CALL_EVENTS = {"InvokeAgent", "InvokeInlineAgent", "InvokeAgentRuntime"}


def _role_name(arn: str | None) -> str | None:
    """Role name from a role ARN or an assumed-role principal ARN."""
    if not arn:
        return None
    if ":assumed-role/" in arn:
        return arn.split(":assumed-role/", 1)[1].split("/", 1)[0]
    return arn.rsplit("/", 1)[-1]


class _Builder:
    def __init__(self, wl: Workload):
        self.wl = wl
        self.items: list[Evidence] = []

    def add(self, signal: str, source: str, detail: str, scale: float = 1.0, **refs) -> None:
        tier, ai, agentic = WEIGHTS[signal]
        self.items.append(
            Evidence(
                id=f"E{len(self.items) + 1}",
                source=source,
                signal=signal,
                tier=tier,
                detail=detail,
                ai_weight=round(ai * scale, 3),
                agentic_weight=round(agentic * scale, 3),
                refs=refs,
            )
        )


def _ai_actions(snap: AccountSnapshot, role_arn: str | None) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    pol = snap.role_policies.get(role_arn or "")
    if not pol:
        return found
    for st in pol.statements:
        if st["effect"] != "Allow":
            continue
        for a in st["actions"]:
            if a in ("*", "bedrock:*"):
                found.setdefault("iam.wildcard_ai", []).append(a)
            elif a.startswith(("bedrock:invokemodel", "bedrock:converse")):
                found.setdefault("iam.model_invoke", []).append(a)
            elif a.startswith(
                ("bedrock:invokeagent", "bedrock:invokeinlineagent", "bedrock-agentcore:invoke")
            ):
                found.setdefault("iam.agent_invoke", []).append(a)
            elif a == "sagemaker:invokeendpoint":
                found.setdefault("iam.sagemaker_invoke", []).append(a)
            elif a in ("secretsmanager:getsecretvalue", "secretsmanager:*"):
                llm_secrets = [r for r in st["resources"] if LLM_SECRET_RE.search(r)]
                if llm_secrets:
                    found.setdefault("iam.llm_secret_access", []).extend(llm_secrets)
    return found


def _declared(b: _Builder, snap: AccountSnapshot) -> None:
    wl = b.wl
    res = wl.primary
    if res.type == "bedrock.agent":
        b.add(
            "resource.bedrock_agent",
            "bedrock-agent",
            f"Resource is a Bedrock Agent (model {res.attributes.get('foundation_model')}).",
            arn=res.arn,
        )

    for sig, actions in _ai_actions(snap, res.role_arn).items():
        b.add(
            sig,
            "iam",
            f"Execution role allows {', '.join(sorted(set(actions)))}.",
            role=res.role_arn,
        )

    llm_env = [n for n in res.env_var_names if LLM_ENV_RE.search(n)]
    if llm_env:
        b.add(
            "config.llm_env",
            "config",
            f"Env var names suggest model access: {', '.join(llm_env)}.",
            names=llm_env,
        )
    agent_env = [n for n in res.env_var_names if AGENT_ENV_RE.search(n)]
    if agent_env:
        b.add(
            "config.agent_env",
            "config",
            f"Env var names suggest agent settings: {', '.join(agent_env)}.",
            names=agent_env,
        )
    llm_refs = [s for s in res.secret_refs if LLM_SECRET_RE.search(s)]
    if llm_refs:
        b.add(
            "config.llm_secret_ref",
            "config",
            "Task injects an LLM-provider secret.",
            secrets=llm_refs,
        )

    packages = res.layers + res.images
    if fw := sorted({m.group(1).lower() for p in packages if (m := AGENT_FW_RE.search(p))}):
        b.add(
            "config.agent_framework",
            "config",
            f"Packages an agent framework: {', '.join(fw)}.",
            packages=[p for p in packages if AGENT_FW_RE.search(p)],
        )
    if sdk := sorted({m.group(1).lower() for p in packages if (m := LLM_SDK_RE.search(p))}):
        b.add(
            "config.llm_sdk",
            "config",
            f"Packages an LLM SDK: {', '.join(sdk)}.",
            packages=[p for p in packages if LLM_SDK_RE.search(p)],
        )
    if res.type == "sagemaker.endpoint":
        if any(LLM_SERVING_RE.search(i) for i in res.images) or "HF_MODEL_ID" in res.env_var_names:
            b.add(
                "config.llm_serving_image",
                "sagemaker",
                "Endpoint serves a generative model (LLM serving container).",
                images=res.images,
            )
        elif any(CLASSICAL_ML_RE.search(i) for i in res.images):
            b.add(
                "config.classical_ml_image",
                "sagemaker",
                "Endpoint serves a classical ML model (not an LLM).",
                images=res.images,
            )


def _observed(b: _Builder, snap: AccountSnapshot, role_shared: bool) -> None:
    res = b.wl.primary
    trail_ok = snap.source_ok("cloudtrail")

    # CloudTrail: calls made BY this workload's role.
    mine = []
    for ev in snap.trail_events:
        if not res.role_arn or ev.role_arn != res.role_arn:
            continue
        if (
            role_shared
            and res.type == "lambda.function"
            and ev.session_name not in (None, res.name)
        ):
            continue  # Lambda sessions are named after the function: attribute precisely
        mine.append(ev)
    ambiguous = role_shared and not (
        res.type == "lambda.function" and any(e.session_name == res.name for e in mine)
    )
    scale = 0.5 if ambiguous else 1.0
    model_calls = [e for e in mine if e.event_name in MODEL_CALL_EVENTS]
    if model_calls:
        models = Counter(e.request_parameters.get("modelId", "?") for e in model_calls)
        b.add(
            "cloudtrail.model_calls",
            "cloudtrail",
            f"{len(model_calls)} Bedrock model call(s) by this role "
            f"({', '.join(f'{m} x{n}' for m, n in models.most_common(3))})"
            + (" - role is shared, attribution halved." if ambiguous else "."),
            scale=scale,
            event_ids=[e.event_id for e in model_calls[:10]],
        )
    agent_calls = [e for e in mine if e.event_name in AGENT_CALL_EVENTS]
    if agent_calls:
        b.add(
            "cloudtrail.calls_agent",
            "cloudtrail",
            f"{len(agent_calls)} InvokeAgent call(s) made by this role.",
            scale=scale,
            event_ids=[e.event_id for e in agent_calls[:10]],
        )

    # CloudTrail: calls made TO this workload (Bedrock agents).
    if res.type == "bedrock.agent":
        agent_id = res.attributes.get("agent_id")
        to_me = [
            e
            for e in snap.trail_events
            if e.event_name in AGENT_CALL_EVENTS and e.request_parameters.get("agentId") == agent_id
        ]
        if to_me:
            b.add(
                "cloudtrail.agent_invocations",
                "cloudtrail",
                f"Agent was invoked {len(to_me)} time(s) in the window.",
                event_ids=[e.event_id for e in to_me[:10]],
            )

    # Permission granted but never used (only meaningful when CloudTrail coverage exists).
    bedrock_perm = any(
        s in _ai_actions(snap, res.role_arn)
        for s in ("iam.model_invoke", "iam.wildcard_ai", "iam.agent_invoke")
    )
    if (
        trail_ok
        and bedrock_perm
        and not model_calls
        and not agent_calls
        and res.type != "bedrock.agent"
    ):
        b.add(
            "cloudtrail.permission_unused",
            "cloudtrail",
            f"Role may call Bedrock but made no Bedrock calls in the last {snap.lookback_hours}h.",
            role=res.role_arn,
        )

    # Bedrock model-invocation logs (structure only).
    role = _role_name(res.role_arn)
    recs = [
        r
        for r in snap.invocations
        if role
        and _role_name(r.identity_arn) == role
        and (
            not role_shared
            or res.type != "lambda.function"
            or r.identity_arn.rsplit("/", 1)[-1] == res.name
        )
    ]
    if recs:
        looped = [r for r in recs if r.tool_results_in_input > 0]
        with_tools = [r for r in recs if r.tools_offered > 0]
        if looped:
            b.add(
                "invocation.tool_loop",
                "bedrock-invocation-logs",
                f"{len(looped)}/{len(recs)} model call(s) fed tool results back to the model "
                "(a tool-use loop).",
                request_ids=[r.request_id for r in looped[:10]],
            )
        elif with_tools and len(recs) >= 3:
            b.add(
                "invocation.tools_no_loop",
                "bedrock-invocation-logs",
                f"{len(with_tools)}/{len(recs)} call(s) offered tools but no tool result was ever "
                "fed back (structured-output pattern, not a loop).",
                request_ids=[r.request_id for r in with_tools[:10]],
            )
        elif len(recs) >= 3:
            b.add(
                "invocation.single_shot",
                "bedrock-invocation-logs",
                f"{len(recs)} model call(s), none offered tools (single-shot prompts).",
                request_ids=[r.request_id for r in recs[:10]],
            )

    # Application logs: direct calls to LLM provider endpoints.
    hits = [h for h in snap.log_hits if h.log_group in res.log_groups]
    if hits:
        b.add(
            "applog.llm_endpoint",
            "cloudwatch-logs",
            "App logs show HTTP calls to "
            + ", ".join(f"{h.endpoint} x{h.count}" for h in hits)
            + ".",
            log_groups=sorted({h.log_group for h in hits}),
        )
        inv = max(h.invocations_seen for h in hits)
        per_inv = max(h.max_calls_per_invocation for h in hits)
        if per_inv >= 2:
            b.add(
                "applog.multi_call_invocation",
                "cloudwatch-logs",
                f"Up to {per_inv} LLM calls within a single invocation.",
                invocations=inv,
            )
        elif per_inv == 1 and inv >= 3:
            b.add(
                "applog.single_call_invocation",
                "cloudwatch-logs",
                f"Exactly one LLM call per invocation across {inv} invocations.",
                invocations=inv,
            )


def _behavioral(b: _Builder, snap: AccountSnapshot) -> None:
    for svc in b.wl.trace_services:
        st = snap.trace_stats[svc]
        if st.traces == 0:
            continue
        refs = {"service": svc, "trace_ids": st.sample_trace_ids}
        if st.traces_with_llm:
            b.add(
                "trace.llm_calls",
                "traces",
                f"{st.traces_with_llm}/{st.traces} trace(s) contain model calls"
                + (f" ({', '.join(st.models)})" if st.models else "")
                + ".",
                **refs,
            )
        elif st.traces >= 3:
            b.add(
                "trace.no_llm",
                "traces",
                f"{st.traces} trace(s) examined, none contain a model call.",
                service=svc,
            )
        if st.traces_with_loop:
            b.add(
                "trace.tool_loop",
                "traces",
                f"{st.traces_with_loop} trace(s) call the model again after a tool ran "
                f"(tools: {', '.join(st.tool_names[:6])}); up to {st.max_llm_calls_per_trace} "
                "model calls per request.",
                **refs,
            )
        if st.traces_with_agent_span:
            b.add(
                "trace.agent_spans",
                "traces",
                f"{st.traces_with_agent_span} trace(s) contain agent spans"
                + (f" ({', '.join(st.agent_names)})" if st.agent_names else "")
                + ".",
                **refs,
            )
            if len(st.agent_names) >= 2:
                b.add(
                    "trace.multi_agent",
                    "traces",
                    f"{len(st.agent_names)} distinct agents cooperate within requests.",
                    **refs,
                )
        if (
            st.traces_with_llm >= 3
            and st.max_llm_calls_per_trace == 1
            and not st.traces_with_tools
            and not st.traces_with_agent_span
        ):
            b.add(
                "trace.single_call",
                "traces",
                f"Every one of {st.traces_with_llm} LLM trace(s) makes exactly one model call, no tools.",
                **refs,
            )


def extract(wl: Workload, snap: AccountSnapshot, role_shared: bool = False) -> list[Evidence]:
    b = _Builder(wl)
    _declared(b, snap)
    _observed(b, snap, role_shared)
    _behavioral(b, snap)
    return reconcile(b.items)


def reconcile(evidence: list[Evidence]) -> list[Evidence]:
    """Resolve evidence that only means something in context.

    "Bedrock permission unused" only says Bedrock is unused. If LLM usage is seen through
    another channel (direct API calls in logs, traces), it must not count against "uses AI".
    It stays in the trail, neutralised, so an auditor still sees the stale permission.
    """
    used_elsewhere = any(e.ai_weight > 0 and e.tier is not Tier.DECLARED for e in evidence)
    for e in evidence:
        if e.signal == "cloudtrail.permission_unused" and used_elsewhere and e.ai_weight:
            e.ai_weight = 0.0
            e.detail += " Not counted: LLM usage observed through another channel."
    return evidence
