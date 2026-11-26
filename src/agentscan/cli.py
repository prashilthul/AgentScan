"""agentscan CLI.

agentscan scan     crawl the account with the access approved so far, write a JSON report
agentscan approve  record that the client approved one more source
agentscan eval     score a report against ground truth
agentscan explain  show the evidence trail for one workload
agentscan serve    governance dashboard + API over the stored scans and audit trail
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from agentscan import access
from agentscan.collectors.traces import JaegerSource
from agentscan.evaluate import LABELS, evaluate
from agentscan.judge import Judge
from agentscan.models import Report
from agentscan.scan import OPTIONAL_SOURCES, analyse, collect, diff

GRANTS = Path("out/grants.json")

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()


def _summary_table(report: Report) -> Table:
    t = Table(title=f"agentscan - account {report.account_id}")
    for col in ("workload", "type", "label", "conf", "basis", "method"):
        t.add_column(col)
    colour = {"agent": "magenta", "llm": "cyan", "service": "white"}
    for r in report.results:
        c = r.classification
        t.add_row(
            r.workload.name,
            r.workload.primary.type,
            f"[{colour[c.category]}]{c.category}[/]",
            f"{c.confidence:.2f}",
            c.basis or "-",
            c.method,
        )
    return t


@app.command()
def scan(
    out: Annotated[Path, typer.Option(help="Where to write the JSON report")] = Path(
        "out/report.json"
    ),
    region: Annotated[list[str] | None, typer.Option(help="Region(s) to scan; repeatable")] = None,
    profile: Annotated[str | None, typer.Option(help="AWS profile (read-only role)")] = None,
    lookback_hours: Annotated[int, typer.Option(help="Audit/trace window")] = 24,
    trace_backend: Annotated[str, typer.Option(help="jaeger | none")] = "jaeger",
    jaeger_url: Annotated[str | None, typer.Option(help="Jaeger query URL")] = None,
    disable: Annotated[
        list[str] | None,
        typer.Option(help=f"Skip a source ({', '.join(OPTIONAL_SOURCES)}); repeatable"),
    ] = None,
    judge: Annotated[bool, typer.Option(help="Ask an LLM to review low-confidence results")] = True,
    judge_model: Annotated[str | None, typer.Option(help="Model for the judge")] = None,
    judge_provider: Annotated[str | None, typer.Option(help="gemini | anthropic")] = None,
    judge_threshold: Annotated[
        float, typer.Option(help="Judge results below this confidence")
    ] = 0.8,
    grants: Annotated[
        Path, typer.Option(help="Approved sources (see `agentscan approve`)")
    ] = GRANTS,
    db: Annotated[
        str | None, typer.Option(help="postgresql:// URL or SQLite path (default $AGENTSCAN_DB)")
    ] = None,
    store: Annotated[bool, typer.Option(help="Save the scan and audit trail to the store")] = True,
) -> None:
    """Crawl the account (read-only) with the approved sources and classify every workload."""
    load_dotenv()
    from agentscan.aws import Aws

    disabled = set(disable or [])
    if bad := disabled - set(OPTIONAL_SOURCES):
        raise typer.BadParameter(f"unknown source(s): {', '.join(sorted(bad))}")
    approved = access.load_grants(grants)
    disabled |= set(OPTIONAL_SOURCES) - set(approved)
    regions = region or [os.environ.get("AWS_DEFAULT_REGION", "us-east-1")]
    source = None
    if trace_backend == "jaeger":
        source = JaegerSource(jaeger_url or os.environ.get("JAEGER_URL", "http://localhost:16686"))

    reviewer = None
    if judge:
        provider = judge_provider or os.environ.get("JUDGE_PROVIDER", "gemini")
        default_model = "gemini-3.5-flash" if provider == "gemini" else "claude-sonnet-5-5"
        reviewer = Judge(
            judge_model or os.environ.get("JUDGE_MODEL", default_model),
            provider=provider,
            threshold=judge_threshold,
        )
    with console.status("collecting..."):
        snap = collect(Aws(profile), regions, lookback_hours, source, disabled)
    with console.status("classifying..."):
        report = analyse(snap, reviewer)
    report.grants = approved
    try:
        report.changes = diff(Report.model_validate_json(out.read_text()), report)
    except (OSError, ValidationError):
        pass  # first scan, or an unreadable previous report: nothing to compare against

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2))
    saved = None
    if store:
        from agentscan.store import Store

        saved = Store(db).save(report)
    console.print(_summary_table(report))
    for s in report.sources:
        mark = "[green]ok[/]" if s.ok else "[red]FAILED[/]"
        console.print(f"  source {s.name:<22} {mark} {s.detail}")
    if report.unattributed_trace_services:
        console.print(
            f"  traced services with no matching resource: "
            f"{', '.join(report.unattributed_trace_services)}"
        )
    for d in report.unclassified:
        console.print(f"  [yellow]found, not classified yet[/] {d.type} {d.arn}")
    for change in report.changes:
        console.print(f"  [cyan]changed[/] {change}")
    for req in report.access_requests:
        console.print(
            f"\n[bold]access request: {req.source}[/] ({req.status})\n  why: {req.why}\n"
            f"  would help: {', '.join(req.workloads)}\n"
            f"  needs: {', '.join(req.permissions)}\n"
            f"  approve with: agentscan approve {req.source} --by <name>"
        )
    console.print(f"report written to {out}")
    if saved:
        console.print(
            f"stored scan {saved[0]}: {len(report.audit_events)} audit event(s), {saved[1]} new"
        )


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    db: Annotated[str | None, typer.Option(help="postgresql:// URL or SQLite path")] = None,
) -> None:
    """Serve the governance dashboard and its API."""
    load_dotenv()
    import uvicorn

    from agentscan.api import WEB, create_app
    from agentscan.store import Store

    if not WEB.is_dir():
        console.print("[yellow]web/out not built: API only (run `make web` for the dashboard)[/]")
    console.print(f"dashboard on http://{host}:{port}")
    uvicorn.run(create_app(Store(db)), host=host, port=port, log_level="warning")


@app.command()
def approve(
    source: str,
    by: Annotated[str, typer.Option(help="Who on the client side approved this")],
    note: Annotated[str, typer.Option(help="Ticket, scope or conditions")] = "",
    grants: Annotated[Path, typer.Option(help="Grants file the scan reads")] = GRANTS,
    report_path: Annotated[
        Path, typer.Option("--report", help="Report holding the request")
    ] = Path("out/report.json"),
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt")] = False,
) -> None:
    """Record the client's approval for one more data source."""
    if source not in access.CATALOG:
        raise typer.BadParameter(f"unknown source {source!r}; one of {', '.join(OPTIONAL_SOURCES)}")
    acc = access.CATALOG[source]
    console.print(f"[bold]{source}[/]\n  reads: {acc.reads}\n  never stores: {acc.not_kept}")
    console.print(f"  permissions: {', '.join(acc.permissions)}")
    try:
        pending = Report.model_validate_json(report_path.read_text()).access_requests
    except (OSError, ValidationError):
        pending = []
    for req in pending:
        if req.source == source:
            console.print(f"  requested because: {req.why}\n  for: {', '.join(req.workloads)}")
    if not yes:
        typer.confirm(f"Record that {by} approved {source}?", abort=True)
    try:
        g = access.approve(grants, source, by, note)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    console.print(f"approved by {g['approved_by']} at {g['approved_at']} -> {grants}")


@app.command("eval")
def eval_cmd(
    report_path: Path,
    truth_path: Path,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable result")] = False,
) -> None:
    """Score a report against ground truth."""
    report = Report.model_validate_json(report_path.read_text())
    res = evaluate(report, truth_path)
    if as_json:
        print(res.model_dump_json(indent=2))
        return
    console.print(
        f"[bold]accuracy {res.correct}/{res.total} = {res.accuracy:.1%}[/]  "
        f"(mean confidence: correct {res.mean_confidence_correct:.2f}, "
        f"wrong {res.mean_confidence_wrong:.2f})"
    )
    t = Table(title="confusion (rows = expected, cols = predicted)")
    t.add_column("")
    for lab in LABELS:
        t.add_column(lab)
    t.add_column("recall")
    for e in LABELS:
        t.add_row(
            e, *[str(res.confusion[e][p]) for p in LABELS], f"{res.per_class[e]['recall']:.2f}"
        )
    t.add_row("precision", *[f"{res.per_class[p]['precision']:.2f}" for p in LABELS], "")
    console.print(t)
    for m in res.mistakes:
        console.print(
            f"  [red]x[/] {m.name}: expected {m.expected}, got {m.predicted} "
            f"(conf {m.confidence:.2f}, basis {m.basis})"
        )
    if res.missing:
        console.print(f"  [yellow]not found by scan:[/] {', '.join(res.missing)}")


@app.command()
def explain(report_path: Path, workload: str) -> None:
    """Print the evidence trail behind one workload's label."""
    report = Report.model_validate_json(report_path.read_text())
    match = [r for r in report.results if workload in (r.workload.name, r.workload.id)]
    if not match:
        raise typer.BadParameter(f"no workload named {workload!r}")
    r = match[0]
    c = r.classification
    console.print(f"[bold]{r.workload.name}[/] ({r.workload.primary.type})  {r.workload.id}")
    console.print(
        f"label [bold]{c.category}[/]  confidence {c.confidence:.2f}  basis {c.basis}  "
        f"method {c.method}"
    )
    console.print(f"P = {json.dumps(c.probabilities)}   p_ai={c.p_ai}  p_agentic={c.p_agentic}")
    console.print(f"why: {c.rationale}")
    for note in r.workload.notes:
        console.print(f"note: {note}")
    t = Table()
    for col in ("id", "tier", "source", "ai", "agentic", "detail"):
        t.add_column(col)
    for e in r.evidence:
        t.add_row(
            e.id, e.tier, e.source, f"{e.ai_weight:+.1f}", f"{e.agentic_weight:+.1f}", e.detail
        )
    console.print(t)


if __name__ == "__main__":
    app()
