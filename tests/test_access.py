"""Discovery from zero, access requests, client approval, and what each new grant changes."""

import json

import pytest

from agentscan.access import OPTIONAL_SOURCES, approve, load_grants, requests
from agentscan.evaluate import evaluate
from agentscan.models import SourceStatus
from agentscan.scan import diff
from tests.test_end_to_end import _run_lab, _scan


@pytest.fixture
def lab(aws, monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LAB_MIN_INTERVAL", "0")
    return _run_lab(tmp_path, "gemini")


def _day(lab, granted):
    report = _scan(*lab, disabled=set(OPTIONAL_SOURCES) - set(granted))
    report.grants = {s: {"approved_by": "cio"} for s in granted}
    return report


def test_discovery_surfaces_compute_with_no_collector(lab):
    report = _scan(*lab)
    assert {d.type for d in report.unclassified} == {"ec2.instance", "states.stateMachine"}
    assert report.inventory["lambda.function"] >= 7
    assert any(s.name.startswith("discovery@") and s.ok for s in report.sources)


def test_day_one_asks_only_for_what_would_help(lab):
    reqs = {r.source: r for r in _day(lab, granted=[]).access_requests}
    assert "iam" in reqs and "cloudtrail" in reqs
    assert all(r.status == "not approved" and r.why and r.workloads for r in reqs.values())
    # Nothing about a plain CRUD API hints at AI, so nobody asks to read its audit trail yet.
    assert "orders-api" in reqs["iam"].workloads
    assert "orders-api" not in reqs["cloudtrail"].workloads


def test_each_grant_settles_more_until_nothing_is_left_to_ask(lab, tmp_path):
    days, granted = [_day(lab, [])], []
    for source in OPTIONAL_SOURCES:
        granted.append(source)
        days.append(_day(lab, granted))
    first = evaluate(days[0], tmp_path / "ground_truth.json")
    last = evaluate(days[-1], tmp_path / "ground_truth.json")
    assert last.accuracy == 1.0 > first.accuracy
    assert not days[-1].access_requests
    changes = diff(days[0], days[-1])
    assert "access granted since last scan: traces" in changes
    assert any(" -> " in c and "confidence" not in c for c in changes)  # some label flipped


def test_approved_source_that_fails_is_raised_again(lab):
    report = _day(lab, ["iam"])
    sources = [*report.sources, SourceStatus(name="cloudtrail", ok=False, detail="AccessDenied")]
    reqs = {r.source: r for r in requests(report.results, sources)}
    assert reqs["cloudtrail"].status == "approved but failed: AccessDenied"
    assert reqs["traces"].status == "not approved"


def test_partial_regional_failure_is_asked_again(lab):
    report = _day(lab, ["iam"])
    sources = [
        *report.sources,
        SourceStatus(name="cloudtrail", ok=True, detail="us-east-1 ok"),
        SourceStatus(name="cloudtrail", ok=False, detail="AccessDenied in eu-west-1"),
    ]
    reqs = {r.source: r for r in requests(report.results, sources)}
    assert reqs["cloudtrail"].status == "approved but failed: AccessDenied in eu-west-1"


def test_approval_is_recorded_with_a_name(tmp_path):
    path = tmp_path / "grants.json"
    approve(path, "cloudtrail", by="Asha (CIO)", note="CHG-1042")
    g = load_grants(path)
    assert g["cloudtrail"]["approved_by"] == "Asha (CIO)" and g["cloudtrail"]["approved_at"]
    with pytest.raises(ValueError):
        approve(path, "cloudtrail", by="  ")
    with pytest.raises(ValueError):
        approve(path, "source-code", by="Asha")
    path.write_text(json.dumps({**g, "everything": {}}))
    assert set(load_grants(path)) == {"cloudtrail"}  # unknown keys in the file grant nothing
