"""LLM judge for workloads the rules cannot decide confidently.

The judge is a reviewer, not the classifier:
  * it only runs when rule confidence is below a threshold;
  * it sees ONLY the collected evidence bundle (no raw logs, no prompts, no secrets);
  * it must cite evidence IDs, and a verdict citing none that exist is discarded;
  * its confidence is capped by the same evidence-tier rule as the rules engine.
"""

from __future__ import annotations

import json

import anthropic

from agentscan.models import Category, Classification, Evidence, JudgeVerdict, Workload
from agentscan.scoring import CONFIDENCE_CAP, _supporting, best_tier

SYSTEM = """You audit cloud workloads for an AI-governance inventory.
Classify ONE workload into exactly one category:
- "agent": uses an LLM in a multi-step loop - the model chooses tools, tool results are fed
  back to the model, or several agents cooperate. One request => several model calls.
- "llm": uses an LLM, but one model call per request (prompt in, answer out). Offering a tool
  schema only to get structured output, with no tool result ever fed back, is still "llm".
- "service": no LLM usage. Classical ML (xgboost, sklearn, ...) counts as "service".

Rules:
- Decide ONLY from the evidence list. Resource names are not evidence.
- "declared" evidence (IAM, env var names, packages) shows capability, not usage.
  "observed" (audit/app logs) shows usage. "behavioral" (traces) shows how it is used.
  Prefer stronger tiers when they conflict.
- Cite the IDs of the evidence items your decision rests on.
- Lower your confidence when evidence is thin or conflicting."""

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["agent", "llm", "service"]},
        "confidence": {"type": "number"},
        "cited_evidence": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["category", "confidence", "cited_evidence", "rationale"],
    "additionalProperties": False,
}


def _bundle(wl: Workload, evidence: list[Evidence], rules: Classification) -> str:
    return json.dumps(
        {
            "workload": {"type": wl.primary.type, "components": [c.type for c in wl.components]},
            "evidence": [
                {
                    "id": e.id,
                    "tier": e.tier,
                    "source": e.source,
                    "signal": e.signal,
                    "detail": e.detail,
                }
                for e in evidence
            ],
            "rules_engine": {
                "category": rules.category,
                "probabilities": rules.probabilities,
                "note": "rules were not confident enough; give an independent judgement",
            },
        },
        indent=2,
    )


class Judge:
    def __init__(self, model: str, threshold: float = 0.7):
        self.model = model
        self.threshold = threshold
        self.client = anthropic.Anthropic()

    def review(
        self, wl: Workload, evidence: list[Evidence], rules: Classification
    ) -> Classification:
        if rules.confidence >= self.threshold or not evidence:
            return rules
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "effort": "medium",
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },
                system=SYSTEM,
                messages=[{"role": "user", "content": _bundle(wl, evidence, rules)}],
            )
        except anthropic.APIError as e:
            rules.rationale += f" (LLM judge unavailable: {type(e).__name__})"
            return rules
        if resp.stop_reason != "end_turn":
            rules.rationale += f" (LLM judge stopped: {resp.stop_reason})"
            return rules
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return rules

        known = {e.id for e in evidence}
        cited = [i for i in raw.get("cited_evidence", []) if i in known]
        if not cited:  # an uncited verdict is not auditable: keep the rules result
            rules.rationale += " (LLM judge verdict discarded: no valid evidence cited)"
            return rules

        category = Category(raw["category"])
        basis = best_tier(_supporting(category, evidence))
        cap = CONFIDENCE_CAP[basis]
        verdict = JudgeVerdict(
            category=category,
            confidence=round(min(float(raw["confidence"]), cap), 3),
            cited_evidence=cited,
            rationale=raw["rationale"],
            model=resp.model,
        )
        return rules.model_copy(
            update={
                "category": verdict.category,
                "confidence": verdict.confidence,
                "basis": basis,
                "method": "rules+judge",
                "rationale": f"{verdict.rationale} [cites {', '.join(cited)}]",
                "judge": verdict,
            }
        )
