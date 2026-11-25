"""Unit tests for the scoring policy and the judge's guardrails."""

from types import SimpleNamespace

from agentscan.judge import Judge
from agentscan.models import Category, Evidence, Resource, Tier, Workload
from agentscan.scoring import classify
from agentscan.signals import WEIGHTS, reconcile


def ev(signal: str, i: int = 1) -> Evidence:
    tier, ai, agentic = WEIGHTS[signal]
    return Evidence(
        id=f"E{i}",
        source="test",
        signal=signal,
        tier=tier,
        detail=signal,
        ai_weight=ai,
        agentic_weight=agentic,
    )


def test_no_evidence_is_a_low_confidence_service():
    c = classify([])
    assert c.category is Category.SERVICE
    assert c.basis is None
    assert c.confidence <= 0.60


def test_permission_alone_is_not_usage():
    # bedrock:* on the role but nothing observed: stays a service
    c = classify([ev("iam.wildcard_ai")])
    assert c.category is Category.SERVICE


def test_declared_evidence_is_capped():
    c = classify(
        [ev("config.agent_framework", 1), ev("config.llm_env", 2), ev("iam.model_invoke", 3)]
    )
    assert c.basis is Tier.DECLARED
    assert c.confidence <= 0.75


def test_tool_schema_without_loop_is_llm_not_agent():
    c = classify([ev("cloudtrail.model_calls", 1), ev("invocation.tools_no_loop", 2)])
    assert c.category is Category.LLM


def test_trace_loop_beats_declared_single_call_hints():
    c = classify(
        [ev("config.llm_serving_image", 1), ev("trace.llm_calls", 2), ev("trace.tool_loop", 3)]
    )
    assert c.category is Category.AGENT
    assert c.basis is Tier.BEHAVIORAL


def test_unused_permission_is_overridden_by_direct_api_calls():
    evidence = reconcile(
        [
            ev("iam.wildcard_ai", 1),
            ev("cloudtrail.permission_unused", 2),
            ev("applog.llm_endpoint", 3),
        ]
    )
    assert classify(evidence).category is Category.LLM
    assert "Not counted" in evidence[1].detail  # kept in the audit trail, neutralised


# ---------------------------------------------------------------- judge guardrails


class _FakeClient:
    def __init__(self, text: str):
        self.text = text
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **_):
        return SimpleNamespace(
            stop_reason="end_turn",
            model="fake",
            content=[SimpleNamespace(type="text", text=self.text)],
        )


def _judge(text: str) -> Judge:
    j = Judge.__new__(Judge)
    j.model, j.threshold, j.client, j.provider = "fake", 0.7, _FakeClient(text), "anthropic"
    return j


WL = Workload(
    id="arn:x",
    name="x",
    primary=Resource(arn="arn:x", type="lambda.function", name="x", region="us-east-1"),
)


def test_judge_verdict_without_valid_citations_is_discarded():
    evidence = [ev("config.llm_env", 1)]
    rules = classify(evidence)
    out = _judge(
        '{"category":"agent","confidence":0.99,"cited_evidence":["E9"],"rationale":"x"}'
    ).review(WL, evidence, rules)
    assert out.category == rules.category
    assert out.method == "rules"


def test_judge_confidence_is_capped_by_evidence_tier():
    evidence = [ev("config.llm_env", 1), ev("config.agent_env", 2)]
    rules = classify(evidence)
    assert rules.confidence < 0.7
    out = _judge(
        '{"category":"llm","confidence":0.99,"cited_evidence":["E1"],"rationale":"x"}'
    ).review(WL, evidence, rules)
    assert out.method == "rules+judge"
    assert out.category is Category.LLM
    assert out.confidence <= 0.75  # declared-only evidence


def test_judge_skipped_when_rules_are_confident():
    evidence = [ev("trace.llm_calls", 1), ev("trace.single_call", 2)]
    rules = classify(evidence)
    out = _judge("not json").review(WL, evidence, rules)
    assert out is rules
