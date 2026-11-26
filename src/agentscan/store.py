"""Where scans and the merged audit trail are kept.

Two tables, both append-only (a trigger rejects UPDATE and DELETE):
  scans         one row per scan, with the full report as JSON
  audit_events  one row per AuditEvent, whatever source it came from

An event's id is a hash of its source and the source's own id, so re-scanning the same window
inserts nothing new and the first scan that saw an event stays recorded against it.

AGENTSCAN_DB picks the backend: a postgresql:// URL (the docker-compose Postgres) or a file
path for SQLite (the default, out/agentscan.db, also what the tests use).
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import closing
from typing import Any

from agentscan.models import Report

DEFAULT_DB = "out/agentscan.db"

_COLUMNS = (
    "id, scan_id, account_id, workload_id, workload, occurred_at, source, action, target, "
    "actor, correlation_id, call_id, detail"
)

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS scans (
  id TEXT PRIMARY KEY, account_id TEXT NOT NULL, generated_at TEXT NOT NULL,
  report TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_events (
  id TEXT PRIMARY KEY, scan_id TEXT NOT NULL REFERENCES scans(id), account_id TEXT NOT NULL,
  workload_id TEXT, workload TEXT, occurred_at TEXT NOT NULL, source TEXT NOT NULL,
  action TEXT NOT NULL, target TEXT NOT NULL, actor TEXT NOT NULL, correlation_id TEXT,
  call_id TEXT, detail TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS audit_by_workload ON audit_events (account_id, workload_id, occurred_at);
CREATE TRIGGER IF NOT EXISTS scans_no_update BEFORE UPDATE ON scans
  BEGIN SELECT RAISE(ABORT, 'scans is append-only'); END;
CREATE TRIGGER IF NOT EXISTS scans_no_delete BEFORE DELETE ON scans
  BEGIN SELECT RAISE(ABORT, 'scans is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_events
  BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_events
  BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
"""

_PG_DDL = """
CREATE TABLE IF NOT EXISTS scans (
  id text PRIMARY KEY, account_id text NOT NULL, generated_at timestamptz NOT NULL,
  report jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS audit_events (
  id text PRIMARY KEY, scan_id text NOT NULL REFERENCES scans(id), account_id text NOT NULL,
  workload_id text, workload text, occurred_at timestamptz NOT NULL, source text NOT NULL,
  action text NOT NULL, target text NOT NULL, actor text NOT NULL, correlation_id text,
  call_id text, detail jsonb NOT NULL);
CREATE INDEX IF NOT EXISTS audit_by_workload ON audit_events (account_id, workload_id, occurred_at);
CREATE OR REPLACE FUNCTION agentscan_append_only() RETURNS trigger AS $$
  BEGIN RAISE EXCEPTION '% is append-only', TG_TABLE_NAME; END $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS scans_append_only ON scans;
CREATE TRIGGER scans_append_only BEFORE UPDATE OR DELETE ON scans
  FOR EACH ROW EXECUTE FUNCTION agentscan_append_only();
DROP TRIGGER IF EXISTS audit_append_only ON audit_events;
CREATE TRIGGER audit_append_only BEFORE UPDATE OR DELETE ON audit_events
  FOR EACH ROW EXECUTE FUNCTION agentscan_append_only();
"""


class Store:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ.get("AGENTSCAN_DB") or DEFAULT_DB
        self.pg = self.dsn.startswith(("postgresql://", "postgres://"))
        self.p = "%s" if self.pg else "?"
        self.j = "%s::jsonb" if self.pg else "?"  # JSON parameter
        with closing(self._connect()) as db:
            if self.pg:
                with db.cursor() as cur:
                    cur.execute(_PG_DDL)
            else:
                db.executescript(_SQLITE_DDL)
            db.commit()

    def _connect(self):
        if self.pg:
            import psycopg

            return psycopg.connect(self.dsn)
        os.makedirs(os.path.dirname(self.dsn) or ".", exist_ok=True)
        return sqlite3.connect(self.dsn)

    def _query(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with closing(self._connect()) as db:
            cur = db.cursor()
            cur.execute(sql, args)
            names = [d[0] for d in cur.description]
            rows = [dict(zip(names, r, strict=True)) for r in cur.fetchall()]
        for r in rows:  # same shape from both backends
            for k in ("report", "detail"):
                if isinstance(r.get(k), str):
                    r[k] = json.loads(r[k])
            for k in ("generated_at", "occurred_at"):
                if r.get(k) is not None and not isinstance(r[k], str):
                    r[k] = r[k].isoformat()
        return rows

    def save(self, report: Report) -> tuple[str, int]:
        """Record a scan and its events. Returns (scan id, events not seen before)."""
        scan_id = str(uuid.uuid4())
        p, j = self.p, self.j
        rows = [
            (
                e.id,
                scan_id,
                e.account_id,
                e.workload_id,
                e.workload,
                e.occurred_at.isoformat(),
                e.source,
                e.action,
                e.target,
                e.actor,
                e.correlation_id,
                e.call_id,
                json.dumps(e.detail),
            )
            for e in report.audit_events
        ]
        with closing(self._connect()) as db:
            cur = db.cursor()
            cur.execute(
                f"INSERT INTO scans (id, account_id, generated_at, report) VALUES ({p},{p},{p},{j})",
                (
                    scan_id,
                    report.account_id,
                    report.generated_at.isoformat(),
                    report.model_dump_json(),
                ),
            )
            new = 0
            for row in rows:
                cur.execute(
                    f"INSERT INTO audit_events ({_COLUMNS}) VALUES "
                    f"({','.join([p] * 12)},{j}) ON CONFLICT (id) DO NOTHING",
                    row,
                )
                new += cur.rowcount
            db.commit()
        return scan_id, new

    def scans(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._query(
            f"SELECT id, account_id, generated_at, report FROM scans "
            f"ORDER BY generated_at DESC LIMIT {self.p}",
            (limit,),
        )
        return [
            {k: r[k] for k in ("id", "account_id", "generated_at")}
            | {"summary": r["report"]["summary"], "grants": sorted(r["report"].get("grants", {}))}
            for r in rows
        ]

    def latest_report(self) -> dict[str, Any] | None:
        rows = self._query("SELECT report FROM scans ORDER BY generated_at DESC LIMIT 1")
        return rows[0]["report"] if rows else None

    def events(self, workload_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        where, args = ("WHERE workload_id = " + self.p, (workload_id,)) if workload_id else ("", ())
        return self._query(
            f"SELECT {_COLUMNS} FROM audit_events {where} ORDER BY occurred_at DESC LIMIT {self.p}",
            (*args, limit),
        )

    def event_counts(self) -> list[dict[str, Any]]:
        return self._query(
            "SELECT source, action, COUNT(*) AS n FROM audit_events GROUP BY source, action "
            "ORDER BY source, action"
        )

    def distinct_event_count(self) -> int:
        """Calls, not source rows: one Bedrock call seen by three sources counts once.

        Only model calls share a call_id across sources. A tool or agent step keeps its own
        row id even if a framework happens to copy the model request id onto it.
        """
        rows = self._query(
            "SELECT COUNT(DISTINCT CASE "
            "WHEN action = 'model_call' AND call_id IS NOT NULL AND call_id <> '' "
            "THEN call_id ELSE id END) AS n FROM audit_events"
        )
        return int(rows[0]["n"]) if rows else 0
