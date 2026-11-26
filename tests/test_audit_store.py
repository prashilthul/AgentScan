"""Merged audit trail, append-only store, risk tiers and the dashboard API, offline."""

import sqlite3
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from agentscan.api import create_app
from agentscan.models import AuditEvent
from agentscan.store import Store
from tests.test_end_to_end import _run_lab, _scan

OTEL_AGENTS = {"support-agent", "research-crew"}
AWS_NATIVE_AGENTS = {"ops-copilot", "invoice-agent"}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for k, v in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
        "LAB_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-key-not-real",
        "LAB_MIN_INTERVAL": "0",
    }.items():
        mp.setenv(k, v)
    mp.delenv("AWS_ENDPOINT_URL", raising=False)
    from moto import mock_aws

    with mock_aws():
        lab = _run_lab(tmp_path_factory.mktemp("lab"), "gemini")
        yield _scan(*lab)
    mp.undo()


def _by(report, name):
    return next(r for r in report.results if r.workload.name == name)


def test_half_the_agents_report_via_otel_and_half_via_aws(report):
    sources = {}
    for e in report.audit_events:
        sources.setdefault(e.workload, set()).add(e.source)
    assert all("otel" in sources[a] for a in OTEL_AGENTS)
    assert all("otel" not in sources[a] for a in AWS_NATIVE_AGENTS)
    assert all(sources[a] & {"cloudtrail", "bedrock-invocation-log"} for a in AWS_NATIVE_AGENTS)
    assert all(e.workload for e in report.audit_events)  # every event attributed


def test_one_bedrock_call_seen_three_ways_counts_once(report):
    calls = [
        e for e in report.audit_events if e.workload == "support-agent" and e.action == "model_call"
    ]
    by_call = {}
    for e in calls:
        by_call.setdefault(e.call_id, set()).add(e.source)
    assert all(s == {"otel", "cloudtrail", "bedrock-invocation-log"} for s in by_call.values())
    assert _by(report, "support-agent").activity["model_call"] == len(by_call)


def test_no_message_content_in_the_audit_trail(report):
    blob = " ".join(str(e.detail) + e.target for e in report.audit_events).lower()
    for text in ("refund policy", "arrived damaged", "invoice inv-1", "churn"):
        assert text not in blob


def test_risk_tiers(report):
    tier = {r.workload.name: r.risk and r.risk.tier for r in report.results}
    assert tier["support-agent"] == tier["ops-copilot"] == "high"  # agents that act
    assert {tier[n] for n in ("orders-api", "nightly-report", "ai-helper-svc", "fraud-score")} == {
        None
    }
    acting = next(f for f in _by(report, "ops-copilot").risk.factors if f.key == "can_act")
    assert "ecs:updateservice" in acting.why and "LLM03:2026 Excessive Agency" in acting.owasp
    # The scripted model only asks for lookup tools; refund_policy must not count as acting.
    support = {f.key: f for f in _by(report, "support-agent").risk.factors}
    assert "refund_policy" not in support["can_act"].why
    for r in report.results:
        for f in r.risk.factors if r.risk else []:
            assert all(o.startswith("LLM") and ":2026 " in o for o in f.owasp)
            assert all(a.startswith("AML.T") for a in f.atlas)


def test_store_is_append_only_and_idempotent(report, tmp_path):
    db = tmp_path / "a.db"
    store = Store(str(db))
    _, first = store.save(report)
    _, second = store.save(report)
    assert first == len(report.audit_events) and second == 0
    assert len(store.scans()) == 2
    with sqlite3.connect(db) as raw:
        for sql in ("UPDATE audit_events SET target = 'x'", "DELETE FROM audit_events"):
            with pytest.raises(sqlite3.IntegrityError):
                raw.execute(sql)


def test_tool_call_is_not_merged_into_its_model_call(report, tmp_path):
    now = datetime.now(UTC)
    model = AuditEvent(
        id="model",
        account_id="1",
        workload_id=None,
        workload=None,
        occurred_at=now,
        source="otel",
        action="model_call",
        target="m",
        actor="svc",
        call_id="req-1",
    )
    tool = model.model_copy(update={"id": "tool", "action": "tool_call", "target": "t"})
    store = Store(str(tmp_path / "calls.db"))
    store.save(report.model_copy(update={"audit_events": [model, tool]}))
    # The tool span reused the model's request id, but it is still a second event.
    assert store.distinct_event_count() == 2


def test_api_overview_and_workload(report, tmp_path):
    store = Store(str(tmp_path / "api.db"))
    client = TestClient(create_app(store))
    assert client.get("/api/overview").status_code == 404  # nothing scanned yet
    store.save(report)
    ov = client.get("/api/overview").json()
    assert ov["total"] == 12 and ov["summary"] == {"agent": 4, "llm": 4, "service": 4}
    assert ov["workloads"][0]["risk"]["tier"] == "high"
    assert "LLM03:2026 Excessive Agency" in ov["owasp"]
    # Headline counts calls, not source rows: a Bedrock call seen three ways is one event.
    rows = sum(r["n"] for r in ov["events_by_source"])
    expected = len(
        {e.call_id if e.action == "model_call" and e.call_id else e.id for e in report.audit_events}
    )
    assert ov["distinct_events"] == expected < rows
    wl = client.get("/api/workloads/invoice-agent").json()
    assert {e["source"] for e in wl["events"]} == {"cloudtrail", "bedrock-invocation-log"}
    assert client.get("/api/workloads/nope").status_code == 404
    # A negative limit must not turn into "no limit" (SQLite) or a 500 (Postgres).
    assert client.get("/api/events?limit=-1").json() == []
