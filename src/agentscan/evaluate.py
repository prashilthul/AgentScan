"""Score a report against a ground-truth file: accuracy, confusion matrix, mistakes.

Ground truth format: {"workloads": [{"arn": ..., "name": ..., "label": "agent|llm|service",
"why": ...}]}. Matching is by ARN, so names never influence scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from agentscan.models import Category, Report

LABELS = [c.value for c in Category]


class Mistake(BaseModel):
    name: str
    expected: str
    predicted: str
    confidence: float
    basis: str | None


class EvalResult(BaseModel):
    total: int
    correct: int
    accuracy: float
    missing: list[str]  # ground-truth workloads the scan did not find
    confusion: dict[str, dict[str, int]]  # expected -> predicted -> count
    per_class: dict[str, dict[str, float]]  # precision / recall per label
    mistakes: list[Mistake]
    mean_confidence_correct: float
    mean_confidence_wrong: float


def evaluate(report: Report, truth_path: Path) -> EvalResult:
    truth = json.loads(truth_path.read_text())["workloads"]
    by_arn = {r.workload.id: r for r in report.results}
    for r in report.results:  # components (e.g. action-group Lambdas) count as their workload
        for c in r.workload.components:
            by_arn.setdefault(c.arn, r)
    confusion = {e: {p: 0 for p in LABELS} for e in LABELS}
    mistakes, missing, conf_ok, conf_bad = [], [], [], []
    for t in truth:
        r = by_arn.get(t["arn"])
        if r is None:
            missing.append(t["name"])
            continue
        pred = r.classification.category.value
        confusion[t["label"]][pred] += 1
        if pred == t["label"]:
            conf_ok.append(r.classification.confidence)
        else:
            conf_bad.append(r.classification.confidence)
            mistakes.append(
                Mistake(
                    name=t["name"],
                    expected=t["label"],
                    predicted=pred,
                    confidence=r.classification.confidence,
                    basis=r.classification.basis,
                )
            )
    per_class = {}
    for lab in LABELS:
        tp = confusion[lab][lab]
        predicted = sum(confusion[e][lab] for e in LABELS)
        actual = sum(confusion[lab].values())
        per_class[lab] = {
            "precision": round(tp / predicted, 3) if predicted else 0.0,
            "recall": round(tp / actual, 3) if actual else 0.0,
        }
    correct = sum(confusion[label][label] for label in LABELS)
    total = len(truth)
    return EvalResult(
        total=total,
        correct=correct,
        accuracy=round(correct / total, 3) if total else 0.0,
        missing=missing,
        confusion=confusion,
        per_class=per_class,
        mistakes=mistakes,
        mean_confidence_correct=round(sum(conf_ok) / len(conf_ok), 3) if conf_ok else 0.0,
        mean_confidence_wrong=round(sum(conf_bad) / len(conf_bad), 3) if conf_bad else 0.0,
    )
