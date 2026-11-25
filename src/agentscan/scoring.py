"""Rule-based classifier: two logistic stages over weighted evidence.

Agent is a subset of AI, so the classes are not independent. We score two questions:
  stage 1  p_ai      = sigmoid(AI_BIAS      + sum(ai_weight))       does it use an LLM at all?
  stage 2  p_agentic = sigmoid(AGENTIC_BIAS + sum(agentic_weight))  if so, does it loop with tools?
  P(agent) = p_ai * p_agentic,  P(llm) = p_ai * (1 - p_agentic),  P(service) = 1 - p_ai

Confidence is then capped by the strongest tier of evidence behind the winner: a label
resting on configuration alone can never be reported as near-certain.
"""

from __future__ import annotations

import math

from agentscan.models import TIER_RANK, Category, Classification, Evidence, Tier

AI_BIAS = -1.5  # with no evidence a workload is most likely a plain service
AGENTIC_BIAS = -1.0  # an AI workload with no loop evidence is most likely single-call

CONFIDENCE_CAP: dict[Tier | None, float] = {
    Tier.BEHAVIORAL: 0.97,
    Tier.OBSERVED: 0.90,
    Tier.DECLARED: 0.75,
    None: 0.60,  # decided by absence of evidence
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _supporting(category: Category, evidence: list[Evidence]) -> list[Evidence]:
    if category is Category.AGENT:
        return [e for e in evidence if e.agentic_weight > 0]
    if category is Category.LLM:
        return [e for e in evidence if e.ai_weight > 0 or e.agentic_weight < 0]
    return [e for e in evidence if e.ai_weight < 0]


def best_tier(evidence: list[Evidence]) -> Tier | None:
    return max((e.tier for e in evidence), key=TIER_RANK.__getitem__, default=None)


def classify(evidence: list[Evidence]) -> Classification:
    p_ai = _sigmoid(AI_BIAS + sum(e.ai_weight for e in evidence))
    p_agentic = _sigmoid(AGENTIC_BIAS + sum(e.agentic_weight for e in evidence))
    probs = {
        Category.AGENT: p_ai * p_agentic,
        Category.LLM: p_ai * (1 - p_agentic),
        Category.SERVICE: 1 - p_ai,
    }
    category = max(probs, key=probs.__getitem__)
    support = _supporting(category, evidence)
    basis = best_tier(support)
    confidence = min(probs[category], CONFIDENCE_CAP[basis])

    top = sorted(support, key=lambda e: abs(e.ai_weight) + abs(e.agentic_weight), reverse=True)[:3]
    if top:
        rationale = "; ".join(f"[{e.id}] {e.detail}" for e in top)
    else:
        rationale = "No AI-related configuration, audit-log or trace evidence found."
    return Classification(
        category=category,
        confidence=round(confidence, 3),
        probabilities={k: round(v, 3) for k, v in probs.items()},
        p_ai=round(p_ai, 3),
        p_agentic=round(p_agentic, 3),
        basis=basis,
        method="rules",
        rationale=rationale,
        rules_category=category,
    )
