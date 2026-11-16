"""Render the README screenshots from real CLI output, saved as SVG.

Rich records exactly what the CLI prints and exports it, so each image is the
actual output, not a mock-up.

    uv run python scripts/render_screenshot.py check
        `evalkit check` on the example suite's hand-written responses
    uv run python scripts/render_screenshot.py run --provider ollama:qwen2.5:7b
        `evalkit run` on the example suite with a real model (needs Ollama)
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress

from evalkit import cli
from evalkit.evaluate import check_suite, load_responses
from evalkit.schema import load_suite

ROOT = Path(__file__).resolve().parents[1]
SUITE = "examples/suites/smoke.yaml"
RESPONSES = "examples/responses/smoke.jsonl"
IMG = ROOT / "docs" / "img"


def _recording() -> Console:
    return Console(record=True, width=100, force_terminal=True, file=io.StringIO(), soft_wrap=True)


def _save(console: Console, title: str, out: Path) -> None:
    svg = console.export_svg(title=title)
    # LF endings and no trailing spaces, so the file passes the pre-commit hooks
    # unchanged no matter which OS renders it.
    clean = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(clean, encoding="utf-8", newline="\n")
    print(f"wrote {out}")


def render_check() -> None:
    cli.console = _recording()
    summary = check_suite(load_suite(ROOT / SUITE), load_responses(ROOT / RESPONSES))
    cli._print_summary(summary, verbose=False)
    _save(cli.console, f"evalkit check {SUITE} {RESPONSES}", IMG / "check-output.svg")


class _NoProgress(Progress):
    """The live progress bar is transient in a terminal, but a recording
    console keeps every frame, which left a stray bar at the top of the image."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def render_run(provider: str, out_dir: Path, *, image: Path = IMG / "run-output.svg") -> None:
    recorder = _recording()
    cli.console = recorder
    cli.err_console = recorder  # failures and notes go to stderr; show them too
    cli.Progress = _NoProgress  # type: ignore[misc]
    # --no-cache: the image should show a real generation, not a cache replay.
    args = ["run", SUITE, "-p", provider, "--out", str(out_dir), "--no-cache"]
    command = typer.main.get_command(cli.app)
    try:
        command.main(args=args, prog_name="evalkit", standalone_mode=False)
    except typer.Exit as exc:  # pragma: no cover - a failing case is still worth showing
        if exc.exit_code not in (0, 1):
            raise
    # --out is relative by default, so the log path in the image is not this PC's.
    _save(recorder, f"evalkit run {SUITE} -p {provider}", image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="target", required=True)
    sub.add_parser("check")
    run = sub.add_parser("run")
    run.add_argument("--provider", default="ollama:qwen2.5:7b")
    run.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if args.target == "check":
        render_check()
    else:
        render_run(args.provider, args.out)


if __name__ == "__main__":
    main()
