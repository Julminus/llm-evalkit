"""Command-line interface: ``evalkit validate | check | list-checks``.

Exit codes: 0 = success, 1 = a suite/responses file is invalid, or (with
``--strict``) at least one case failed. Typer itself uses 2 for bad usage.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from evalkit import __version__
from evalkit.checks import CHECK_TYPES, Status
from evalkit.evaluate import CheckSummary, ResponsesError, check_suite, load_responses
from evalkit.schema import Suite, SuiteError, load_suite

app = typer.Typer(
    name="evalkit",
    help="Bilingual (EN/ID) LLM evaluation harness.",
    no_args_is_help=True,
    add_completion=False,
)
# soft_wrap: long lines (error messages, paths) are not hard-wrapped at the
# terminal width, so they stay greppable and copy-pasteable.
console = Console(soft_wrap=True)
err_console = Console(stderr=True, soft_wrap=True)

_STATUS_STYLE = {Status.PASS: "green", Status.FAIL: "red", Status.SKIP: "yellow"}

SuitePath = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Suite YAML file.")
]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"evalkit {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Bilingual (EN/ID) LLM evaluation harness."""


def _load_or_exit(path: Path) -> Suite:
    try:
        return load_suite(path)
    except SuiteError as exc:
        err_console.print(f"[red]error:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc


@app.command()
def validate(
    suites: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Suite YAML file(s)."),
    ],
) -> None:
    """Validate suite files and print a summary of each."""
    failed = False
    for path in suites:
        try:
            suite = load_suite(path)
        except SuiteError as exc:
            err_console.print(f"[red]invalid[/] {escape(str(exc))}")
            failed = True
            continue
        langs = Counter(c.language for c in suite.cases)
        checks = Counter(c.type for case in suite.cases for c in case.effective_constraints())
        tags = Counter(t for case in suite.cases for t in case.tags)
        console.print(
            f"[green]valid[/] {escape(str(path))}: suite [bold]{escape(suite.name)}[/] "
            f"v{escape(suite.version)}, {len(suite.cases)} cases "
            f"({', '.join(f'{n} {lang}' for lang, n in sorted(langs.items()))})"
        )
        console.print("  checks: " + ", ".join(f"{t} ({n})" for t, n in sorted(checks.items())))
        if tags:
            console.print("  tags:   " + ", ".join(f"{t} ({n})" for t, n in tags.most_common()))
    if failed:
        raise typer.Exit(1)


def _summary_json(summary: CheckSummary) -> str:
    payload = {
        "pass_rate": round(summary.pass_rate, 4),
        "pass_rate_by_language": {
            k: round(v, 4) for k, v in summary.pass_rate_by_language().items()
        },
        "counts": {s.value: summary.counts[s] for s in Status},
        "by_check": {
            check: {s.value: c[s] for s in Status} for check, c in summary.by_check().items()
        },
        "failure_modes": {m.value: n for m, n in summary.failure_modes().most_common()},
        "missing": summary.missing,
        "unknown": summary.unknown,
        "cases": [r.model_dump(mode="json") for r in summary.reports],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _print_summary(summary: CheckSummary, *, verbose: bool) -> None:
    table = Table(title="Deterministic checks", show_lines=False)
    table.add_column("case")
    table.add_column("lang")
    table.add_column("status")
    table.add_column("details", overflow="fold")
    for report in summary.reports:
        shown = report.results if verbose else report.failures
        details = "\n".join(
            f"[{_STATUS_STYLE[r.status]}]{r.check}[/]: {escape(r.message)}" for r in shown
        )
        style = _STATUS_STYLE[report.status]
        table.add_row(report.case_id, report.language, f"[{style}]{report.status}[/]", details)
    console.print(table)

    counts = summary.counts
    by_lang = ", ".join(f"{k} {v:.0%}" for k, v in summary.pass_rate_by_language().items())
    console.print(
        f"cases: {counts[Status.PASS]} pass, {counts[Status.FAIL]} fail, "
        f"{counts[Status.SKIP]} skip; pass rate {summary.pass_rate:.0%} ({by_lang})"
    )
    modes = summary.failure_modes()
    if modes:
        console.print(
            "failure modes: " + ", ".join(f"{m.value} ({n})" for m, n in modes.most_common())
        )
    if summary.missing:
        console.print(f"[yellow]no response for:[/] {', '.join(summary.missing)}")
    if summary.unknown:
        console.print(f"[yellow]responses with no matching case:[/] {', '.join(summary.unknown)}")


@app.command()
def check(
    suite_path: SuitePath,
    responses_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help='JSONL with one {"id": ..., "response": ...} per line.',
        ),
    ],
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table.")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show passing checks too.")
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit with code 1 if any case fails.")
    ] = False,
) -> None:
    """Run the suite's deterministic checks on a file of responses."""
    suite = _load_or_exit(suite_path)
    try:
        responses = load_responses(responses_path)
    except ResponsesError as exc:
        err_console.print(f"[red]error:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    summary = check_suite(suite, responses)
    if as_json:
        typer.echo(_summary_json(summary))
    else:
        _print_summary(summary, verbose=verbose)
    if strict and summary.counts[Status.FAIL]:
        raise typer.Exit(1)


@app.command("list-checks")
def list_checks() -> None:
    """List the available constraint types."""
    table = Table(title="Constraint types")
    table.add_column("type")
    table.add_column("fails as")
    table.add_column("what it checks")
    for cls in CHECK_TYPES:
        type_name = cls.model_fields["type"].default
        table.add_row(type_name, cls.default_failure_mode.value, cls.summary)
    console.print(table)
