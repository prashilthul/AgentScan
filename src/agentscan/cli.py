"""agentscan CLI.

agentscan scan     crawl the account and write a JSON report
agentscan eval     score a report against ground truth
agentscan explain  show the evidence trail for one workload
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agentscan.collectors.traces import JaegerSource
from agentscan.evaluate import LABELS, evaluate
from agentscan.judge import Judge
from agentscan.models import Report
from agentscan.scan import OPTIONAL_SOURCES, analyse, collect

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
    ] = 0.7,
) -> None:
    """Crawl the account (read-only) and classify every workload."""
    load_dotenv()
    from agentscan.aws import Aws

    disabled = set(disable or [])
    if bad := disabled - set(OPTIONAL_SOURCES):
        raise typer.BadParameter(f"unknown source(s): {', '.join(sorted(bad))}")
    regions = region or [os.environ.get("AWS_DEFAULT_REGION", "us-east-1")]
    source = None
    if trace_backend == "jaeger":
        source = JaegerSource(jaeger_url or os.environ.get("JAEGER_URL", "http://localhost:16686"))

    reviewer = None
    if judge:
        provider = judge_provider or os.environ.get("JUDGE_PROVIDER", "gemini")
        default_model = "gemini-2.5-flash" if provider == "gemini" else "claude-sonnet-5-5"
        reviewer = Judge(
            judge_model or os.environ.get("JUDGE_MODEL", default_model),
            provider=provider,
            threshold=judge_threshold,
        )
    with console.status("collecting..."):
        snap = collect(Aws(profile), regions, lookback_hours, source, disabled)
    with console.status("classifying..."):
        report = analyse(snap, reviewer)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2))
    console.print(_summary_table(report))
    for s in report.sources:
        mark = "[green]ok[/]" if s.ok else "[red]FAILED[/]"
        console.print(f"  source {s.name:<22} {mark} {s.detail}")
    if report.unattributed_trace_services:
        console.print(
            f"  traced services with no matching resource: "
            f"{', '.join(report.unattributed_trace_services)}"
        )
    console.print(f"report written to {out}")


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
