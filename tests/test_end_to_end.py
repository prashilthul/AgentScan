"""Full pipeline, offline: seed the fake account, run every lab workload through the real
SDK against a scripted model, crawl, classify, and score against ground truth."""

import anthropic
import httpx
import httpx2
import pytest

from agentscan.aws import Aws
from agentscan.evaluate import evaluate
from agentscan.scan import analyse, collect
from lab.run import run_workload
from lab.seed import seed
from lab.shim.mirror import AwsMirror
from lab.spec import WORKLOADS
from tests.fakes import InMemoryTraceSource, KeepSpans, scripted_gemini, scripted_model


@pytest.fixture(params=["gemini", "anthropic"])
def provider(request, monkeypatch):
    monkeypatch.setenv("LAB_PROVIDER", request.param)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LAB_MIN_INTERVAL", "0")
    return request.param


def _http(provider):
    if provider == "anthropic":
        return anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(scripted_model))
    return httpx.Client(transport=httpx.MockTransport(scripted_gemini))


def _run_lab(tmp_path, provider):
    state = seed(tmp_path)
    mirror = AwsMirror(state["account_id"], state["region"])
    exporter = KeepSpans()
    http = _http(provider)
    for spec in WORKLOADS:
        if spec.module:
            run_workload(spec, state, mirror, span_exporter=exporter, http_client=http)
    return state, exporter


def _scan(state, exporter, disabled=()):
    snap = collect(Aws(), [state["region"]], 24, InMemoryTraceSource(exporter), set(disabled))
    return analyse(snap)


def test_all_workloads_classified_correctly(aws, provider, tmp_path):
    state, exporter = _run_lab(tmp_path, provider)
    report = _scan(state, exporter)
    assert all(s.ok for s in report.sources), [s for s in report.sources if not s.ok]
    res = evaluate(report, tmp_path / "ground_truth.json")
    assert not res.missing
    assert res.accuracy == 1.0, [m.model_dump() for m in res.mistakes]


def test_every_label_has_evidence_and_bounded_confidence(aws, provider, tmp_path):
    state, exporter = _run_lab(tmp_path, provider)
    report = _scan(state, exporter)
    for r in report.results:
        c = r.classification
        assert 0 < c.confidence <= 0.97
        if c.category != "service":
            assert r.evidence, r.workload.name
        for e in r.evidence:
            assert e.id and e.detail and e.tier


def test_without_traces_confidence_drops_but_labels_hold_where_logs_exist(aws, provider, tmp_path):
    state, exporter = _run_lab(tmp_path, provider)
    full = {r.workload.name: r.classification for r in _scan(state, exporter).results}
    blind = {
        r.workload.name: r.classification
        for r in _scan(state, exporter, disabled=["traces"]).results
    }
    # Bedrock invocation logs still reveal the tool loop of a traced Bedrock agent.
    assert blind["support-agent"].category == "agent"
    assert blind["support-agent"].confidence < full["support-agent"].confidence
