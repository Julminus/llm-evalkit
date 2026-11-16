"""Command-line interface: ``evalkit validate | check | run | list-checks``.

Exit codes: 0 = success; 1 = a suite/responses file is invalid, a provider
can't be set up, a case got no response (``run``), or (with ``--strict``) at
least one case failed its checks or, for ``check``, had no response, or a
response matched no case; 130 = interrupted. Typer itself uses 2 for bad
usage.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from evalkit import __version__
from evalkit.cache import ResponseCache
from evalkit.checks import CHECK_TYPES, Status
from evalkit.evaluate import CheckSummary, ResponsesError, check_suite, load_responses
from evalkit.providers import (
    GenerationParams,
    MockProvider,
    Provider,
    ProviderConfigError,
    ProviderError,
    make_provider,
    parse_spec,
)
from evalkit.runner import RetryPolicy, RunRecord, RunResult, run_suite
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


def _never_crash_on_output() -> None:
    """Keep each stream's encoding, but escape what it can't encode.

    On Windows a redirected stream uses the ANSI code page (cp1252), so
    ``evalkit check ... > out.txt`` crashed on the first response quoting
    "→", an emoji or CJK text. Switching the stream to UTF-8 is no fix:
    Windows PowerShell 5.1 decodes a program's output with the OEM code page,
    so Rich's box drawing came out garbled. So the encoding stays (Rich then
    draws ASCII boxes), and characters it can't hold are written as
    ``\\u2192``-style escapes. ``--json`` escapes non-ASCII itself on such a
    stream, so JSON output stays exact. Terminals are unaffected: Python
    writes to the Windows console as UTF-16.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.errors == "strict":
            stream.reconfigure(errors="backslashreplace")


def _stdout_is_utf8() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    return encoding.lower().replace("_", "-") in {"utf-8", "utf8"}


def _dumps(payload: Any) -> str:
    """JSON for stdout: readable on a UTF-8 stream, pure ASCII on any other."""
    return json.dumps(payload, ensure_ascii=not _stdout_is_utf8(), indent=2)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Bilingual (EN/ID) LLM evaluation harness."""
    _never_crash_on_output()


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
            shown_tags = ", ".join(f"{t} ({n})" for t, n in tags.most_common())
            console.print("  tags:   " + escape(shown_tags))
    if failed:
        raise typer.Exit(1)


def _summary_payload(summary: CheckSummary) -> dict[str, Any]:
    return {
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


def _summary_json(summary: CheckSummary) -> str:
    return _dumps(_summary_payload(summary))


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
        console.print(f"[yellow]no response for:[/] {escape(', '.join(summary.missing))}")
    if summary.unknown:
        unknown = escape(", ".join(summary.unknown))
        console.print(f"[yellow]responses with no matching case:[/] {unknown}")


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
        bool,
        typer.Option(
            "--strict",
            help="Exit with code 1 if any case fails or has no response, "
            "or a response matches no case.",
        ),
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
    # As a CI gate, "nothing failed" must not pass when the responses file was
    # the wrong one or incomplete: an ungraded case is not a passing case.
    if strict and (summary.counts[Status.FAIL] or summary.missing or summary.unknown):
        raise typer.Exit(1)


def _exit_error(message: str, code: int = 1) -> typer.Exit:
    err_console.print(f"[red]error:[/] {escape(message)}")
    return typer.Exit(code)


def _build_provider(
    spec: str, *, suite: Suite, mock_responses: Path | None, timeout: float, base_url: str | None
) -> Provider:
    try:
        if mock_responses is None:
            return make_provider(spec, timeout=timeout, base_url=base_url)
        name, model = parse_spec(spec)
        if name != "mock":
            raise ProviderConfigError("--mock-responses only works with --provider mock")
        return MockProvider.for_suite(suite, load_responses(mock_responses), model=model)
    except (ProviderConfigError, ResponsesError) as exc:
        raise _exit_error(str(exc)) from exc


def _print_run(result: RunResult, provider: Provider) -> None:
    c = result.counts()
    console.print(
        f"run [bold]{escape(result.run_id)}[/] ({escape(provider.spec)}): "
        f"{c['cases']} cases in {result.duration_s:.1f} s"
    )
    console.print(
        f"  {c['generated']} generated, {c['cached']} from cache, {c['errors']} failed; "
        f"{c['input_tokens']} input / {c['output_tokens']} output tokens"
    )
    if result.truncated:
        console.print(
            f"[yellow]hit the token limit (may fail checks for that reason):[/] "
            f"{escape(', '.join(result.truncated))}"
        )
    if result.refused:
        console.print(f"[yellow]refused:[/] {escape(', '.join(result.refused))}")
    for warning in result.warnings:
        err_console.print(f"[yellow]warning:[/] {escape(warning)}")
    for record in result.errors:
        err_console.print(f"[red]failed[/] {escape(record.id)}: {escape(record.error or '')}")
    if result.run_dir is not None:
        console.print(f"log: {escape(str(result.run_dir / 'responses.jsonl'))}")


@app.command()
def run(
    suite_path: SuitePath,
    provider_spec: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="mock | ollama:MODEL | openai:MODEL | anthropic:MODEL (e.g. ollama:qwen2.5:7b).",
        ),
    ],
    temperature: Annotated[
        float | None,
        typer.Option(
            min=0.0, max=2.0, help="Sampling temperature. Default: 0 where the provider accepts it."
        ),
    ] = None,
    seed: Annotated[
        int | None, typer.Option(help="Sampling seed. Default: 0 where the provider accepts it.")
    ] = None,
    max_tokens: Annotated[
        int | None, typer.Option(min=1, help="Output token limit. Default: the provider's.")
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", min=1, help="Requests in flight at once.")
    ] = 4,
    retries: Annotated[int, typer.Option(min=0, help="Retries per case for transient errors.")] = 3,
    timeout: Annotated[float, typer.Option(min=1.0, help="Seconds per request.")] = 300.0,
    base_url: Annotated[
        str | None,
        typer.Option(
            help="Server URL; overrides OLLAMA_HOST / OPENAI_BASE_URL / ANTHROPIC_BASE_URL."
        ),
    ] = None,
    cache_dir: Annotated[Path, typer.Option(help="Response cache directory.")] = Path(
        ".evalkit-cache"
    ),
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Neither read nor write the cache.")
    ] = False,
    out: Annotated[Path, typer.Option(help="Directory for run logs.")] = Path("runs"),
    run_id: Annotated[
        str | None, typer.Option(help="Run directory name. Default: time + model.")
    ] = None,
    case: Annotated[
        list[str] | None, typer.Option("--case", help="Only run this case id (repeatable).")
    ] = None,
    mock_responses: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help='With --provider mock: JSONL of {"id", "response"} to answer with.',
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of tables.")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show passing checks too.")
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit with code 1 if any case fails its checks.")
    ] = False,
) -> None:
    """Generate responses for a suite with a model, log them, and check them."""
    suite = _load_or_exit(suite_path)
    selected = [c for c in suite.cases if not case or c.id in case]
    unknown = sorted(set(case or []) - {c.id for c in suite.cases})
    if unknown:
        raise _exit_error(f"no such case id(s) in {suite_path}: {', '.join(unknown)}")

    provider = _build_provider(
        provider_spec,
        suite=suite,
        mock_responses=mock_responses,
        timeout=timeout,
        base_url=base_url,
    )
    # Deterministic by default, but only where the provider takes the setting,
    # so the Anthropic provider is not "told" about settings nobody asked for.
    accepts = provider.supported_params
    if temperature is None and "temperature" in accepts:
        temperature = 0.0
    if seed is None and "seed" in accepts:
        seed = 0
    params = GenerationParams(temperature=temperature, seed=seed, max_tokens=max_tokens)
    ignored = provider.ignored_params(params)
    if ignored:
        err_console.print(
            f"[yellow]note:[/] {provider.name} does not accept {', '.join(ignored)}; "
            "those settings are not sent"
        )
    cache = None if no_cache else ResponseCache(cache_dir)

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err_console,
        transient=True,
        disable=not err_console.is_terminal,
    )
    task = progress.add_task(escape(provider.spec), total=len(selected))

    def advance(_: RunRecord) -> None:
        progress.advance(task)

    async def main() -> RunResult:
        async with provider:
            await provider.preflight()
            return await run_suite(
                suite,
                provider,
                params=params,
                concurrency=concurrency,
                retry=RetryPolicy(max_retries=retries),
                cache=cache,
                out_dir=out,
                run_id=run_id,
                case_ids=[c.id for c in selected] if case else None,
                suite_path=suite_path,
                on_record=advance,
            )

    try:
        with progress:
            result = asyncio.run(main())
    except ProviderError as exc:
        raise _exit_error(str(exc)) from exc
    except KeyboardInterrupt as exc:
        err_console.print(
            "[yellow]interrupted.[/] Completed responses are in the run log and the cache; "
            "run the same command again to continue."
        )
        raise typer.Exit(130) from exc

    summary = check_suite(suite.model_copy(update={"cases": selected}), result.responses())
    if as_json:
        payload = {
            "run": {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir) if result.run_dir else None,
                "provider": provider.spec,
                "duration_s": result.duration_s,
                "counts": result.counts(),
                "errors": {r.id: r.error for r in result.errors},
                "truncated": result.truncated,
                "warnings": result.warnings,
            },
            "checks": _summary_payload(summary),
        }
        typer.echo(_dumps(payload))
    else:
        _print_summary(summary, verbose=verbose)
        _print_run(result, provider)
    if result.errors or (strict and summary.counts[Status.FAIL]):
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
        # Summaries contain "[min, max]", which Rich would otherwise eat as markup.
        table.add_row(type_name, cls.default_failure_mode.value, escape(cls.summary))
    console.print(table)
