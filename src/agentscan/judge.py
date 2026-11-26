"""LLM judge for workloads the rules cannot decide confidently.

The judge is a reviewer, not the classifier:
  * it only runs when rule confidence is below a threshold;
  * it sees ONLY the collected evidence bundle (no raw logs, no prompts, no secrets);
  * it must cite evidence IDs, and a verdict citing none that exist is discarded;
  * its confidence is capped by the same evidence-tier rule as the rules engine.
"""

from __future__ import annotations

import json
import math
import os
import time

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


class JudgeUnavailable(Exception):
    def __init__(self, msg: str, retry: bool = False):
        super().__init__(msg)
        self.retry = retry  # rate limit / overloaded: worth asking again after a pause


RETRY_STATUS = {429, 500, 502, 503, 504}
BACKOFF = (5, 15, 30)  # seconds; free tiers and shared-capacity APIs need a slow retry


def _json_text(text: str) -> str:
    """Models without schema enforcement sometimes wrap JSON in a ``` fence."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return t


class Judge:
    """provider: "gemini" (Google AI Studio, free tier), "anthropic", or "openai" (any
    OpenAI-compatible endpoint: JUDGE_BASE_URL + JUDGE_API_KEY)."""

    def __init__(self, model: str, threshold: float = 0.8, provider: str = "gemini"):
        self.model = model
        self.threshold = threshold
        self.provider = provider
        self.client: object | None = None

    def _connect(self):
        """Build the SDK client on first use; missing credentials degrade to rules."""
        if self.client is not None:
            return self.client
        try:
            if self.provider == "openai":
                import httpx

                self.client = httpx.Client(
                    base_url=os.environ["JUDGE_BASE_URL"],
                    headers={"Authorization": f"Bearer {os.environ['JUDGE_API_KEY']}"},
                    timeout=120,
                )
            elif self.provider == "anthropic":
                import anthropic

                self.client = anthropic.Anthropic()
            else:
                from google import genai

                self.client = genai.Client()
        except Exception as e:  # noqa: BLE001 - a missing key/URL must not stop a scan
            raise JudgeUnavailable(f"client unavailable: {type(e).__name__}") from e
        return self.client

    def _ask(self, prompt: str) -> tuple[str, str]:
        """Return (json text, model that answered)."""
        client = self._connect()
        if self.provider == "openai":
            return self._ask_openai(client, prompt)
        if self.provider == "anthropic":
            import anthropic

            try:
                resp = client.beta.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    output_config={
                        "effort": "medium",
                        "format": {"type": "json_schema", "schema": SCHEMA},
                    },
                    system=SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
            except anthropic.APIStatusError as e:
                raise JudgeUnavailable(
                    f"{type(e).__name__} {e.status_code}", e.status_code in RETRY_STATUS
                ) from e
            except anthropic.APIError as e:
                raise JudgeUnavailable(type(e).__name__) from e
            except Exception as e:  # noqa: BLE001 - auth/network errors degrade to rules
                raise JudgeUnavailable(type(e).__name__) from e
            if resp.stop_reason != "end_turn":
                raise JudgeUnavailable(f"stopped: {resp.stop_reason}")
            return next((b.text for b in resp.content if b.type == "text"), ""), resp.model

        from google.genai import errors, types

        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    response_mime_type="application/json",
                    response_json_schema=SCHEMA,
                    max_output_tokens=4000,
                ),
            )
        except errors.APIError as e:
            raise JudgeUnavailable(f"{type(e).__name__} {e.code}", e.code in RETRY_STATUS) from e
        except Exception as e:  # noqa: BLE001 - auth/network errors degrade to rules
            raise JudgeUnavailable(type(e).__name__) from e
        return resp.text or "", self.model

    def _ask_openai(self, client, prompt: str) -> tuple[str, str]:
        import httpx

        body = {
            "model": self.model,
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "system",
                    "content": f"{SYSTEM}\n\nReply with one JSON object only, matching this "
                    f"schema:\n{json.dumps(SCHEMA)}",
                },
                {"role": "user", "content": prompt},
            ],
        }
        try:
            r = client.post("/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise JudgeUnavailable(type(e).__name__, retry=True) from e
        if r.status_code != 200:
            raise JudgeUnavailable(f"HTTP {r.status_code}", r.status_code in RETRY_STATUS)
        choice = r.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise JudgeUnavailable("stopped: output limit")
        return _json_text(choice["message"].get("content") or ""), r.json().get("model", self.model)

    def _ask_with_retry(self, prompt: str) -> tuple[str, str]:
        for wait in (*BACKOFF, None):
            try:
                return self._ask(prompt)
            except JudgeUnavailable as e:
                if not e.retry or wait is None:
                    raise
                time.sleep(wait)
        raise AssertionError("unreachable")

    def review(
        self, wl: Workload, evidence: list[Evidence], rules: Classification
    ) -> Classification:
        if rules.confidence >= self.threshold or not evidence:
            return rules
        try:
            text, model = self._ask_with_retry(_bundle(wl, evidence, rules))
            raw = json.loads(text)
        except JudgeUnavailable as e:
            rules.rationale += f" (LLM judge unavailable: {e})"
            return rules
        except json.JSONDecodeError:
            rules.rationale += " (LLM judge returned invalid JSON)"
            return rules

        if not isinstance(raw, dict):
            rules.rationale += " (LLM judge verdict discarded: not a JSON object)"
            return rules

        known = {e.id for e in evidence}
        raw_cited = raw.get("cited_evidence")
        cited = (
            [i for i in raw_cited if isinstance(i, str) and i in known]
            if isinstance(raw_cited, list)
            else []
        )
        if not cited:  # an uncited verdict is not auditable: keep the rules result
            rules.rationale += " (LLM judge verdict discarded: no valid evidence cited)"
            return rules

        try:
            category = Category(raw["category"])
            confidence = float(raw["confidence"])
            rationale = str(raw.get("rationale") or "").strip()
        except (KeyError, TypeError, ValueError):
            rules.rationale += " (LLM judge verdict discarded: malformed verdict)"
            return rules
        if not math.isfinite(confidence) or not rationale:
            rules.rationale += " (LLM judge verdict discarded: malformed verdict)"
            return rules

        basis = best_tier(_supporting(category, evidence))
        cap = CONFIDENCE_CAP[basis]
        verdict = JudgeVerdict(
            category=category,
            confidence=round(max(0.0, min(confidence, cap)), 3),
            cited_evidence=cited,
            rationale=rationale,
            model=model,
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
