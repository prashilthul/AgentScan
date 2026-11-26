"""Read-only HTTP API for the governance dashboard, plus the built front end (web/out).

Everything comes from the store: the latest scan's report and the merged audit trail.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agentscan.store import Store

WEB = Path(__file__).resolve().parents[2] / "web" / "out"
TIER_ORDER = {"high": 0, "medium": 1, "low": 2, None: 3}


def overview(
    report: dict[str, Any], counts: list[dict], scans: list[dict], distinct_events: int
) -> dict[str, Any]:
    """The executive view, derived from one report."""
    results = report["results"]
    ai = [r for r in results if r["classification"]["category"] != "service"]
    tiers = Counter((r.get("risk") or {}).get("tier") for r in ai)
    owasp, atlas = Counter(), Counter()
    for r in ai:
        for f in (r.get("risk") or {}).get("factors", []):
            owasp.update(set(f["owasp"]))
            atlas.update(set(f["atlas"]))
    activity = Counter()
    for r in results:
        activity.update(r.get("activity") or {})
    sources = report["sources"]
    optional = {"iam", "cloudtrail", "invocation-logs", "app-logs", "traces"}
    granted = set(report.get("grants") or {})
    workloads = sorted(
        (
            {
                "id": r["workload"]["id"],
                "name": r["workload"]["name"],
                "type": r["workload"]["primary"]["type"],
                "label": r["classification"]["category"],
                "confidence": r["classification"]["confidence"],
                "basis": r["classification"]["basis"],
                "risk": r.get("risk"),
                "activity": r.get("activity") or {},
            }
            for r in results
        ),
        key=lambda w: (
            TIER_ORDER[(w["risk"] or {}).get("tier")],
            -(w["risk"] or {}).get("score", 0),
            w["name"],
        ),
    )
    return {
        "account_id": report["account_id"],
        "generated_at": report["generated_at"],
        "regions": report["regions"],
        "summary": report["summary"],
        "total": len(results),
        "risk_tiers": {t: tiers.get(t, 0) for t in ("high", "medium", "low")},
        "owasp": dict(owasp.most_common()),
        "atlas": dict(atlas.most_common()),
        "activity": dict(activity),
        "events_by_source": counts,
        "distinct_events": distinct_events,
        "coverage": {
            "granted": sorted(granted & optional),
            "missing": sorted(optional - granted),
            "failed": [s["name"] for s in sources if not s["ok"]],
        },
        "access_requests": report.get("access_requests", []),
        "unclassified": report.get("unclassified", []),
        "changes": report.get("changes", []),
        "workloads": workloads,
        "scans": scans,
    }


def create_app(store: Store | None = None) -> FastAPI:
    store = store or Store()
    app = FastAPI(title="agentscan", docs_url="/api/docs", openapi_url="/api/openapi.json")
    # Only for `pnpm dev` on :3000; the built dashboard is served from this same origin.
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["GET"]
    )

    def latest() -> dict[str, Any]:
        report = store.latest_report()
        if report is None:
            raise HTTPException(404, "no scans yet: run `agentscan scan` first")
        return report

    @app.get("/api/overview")
    def get_overview():
        return overview(
            latest(), store.event_counts(), store.scans(20), store.distinct_event_count()
        )

    @app.get("/api/workloads/{name}")
    def get_workload(name: str):
        for r in latest()["results"]:
            if name in (r["workload"]["name"], r["workload"]["id"]):
                return r | {"events": store.events(r["workload"]["id"], 200)}
        raise HTTPException(404, f"no workload {name!r}")

    @app.get("/api/events")
    def get_events(limit: int = 200):
        return store.events(limit=max(0, min(limit, 2000)))

    @app.get("/api/scans")
    def get_scans():
        return store.scans()

    if WEB.is_dir():
        app.mount("/_next", StaticFiles(directory=WEB / "_next"), name="next")

        @app.get("/{path:path}", include_in_schema=False)
        def page(path: str):
            for candidate in (WEB / path, WEB / f"{path}.html", WEB / path / "index.html"):
                if candidate.is_file() and WEB in candidate.resolve().parents:
                    return FileResponse(candidate)
            return FileResponse(WEB / "index.html")

    return app
