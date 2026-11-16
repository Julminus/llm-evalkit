"""Render the README screenshot: real `evalkit check` output saved as SVG.

Rich records exactly what the CLI prints and exports it, so the image is the
actual output of the example suite, not a mock-up.

    uv run python scripts/render_screenshot.py
"""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from evalkit import cli
from evalkit.evaluate import check_suite, load_responses
from evalkit.schema import load_suite

ROOT = Path(__file__).resolve().parents[1]
SUITE = "examples/suites/smoke.yaml"
RESPONSES = "examples/responses/smoke.jsonl"
OUT = ROOT / "docs" / "img" / "check-output.svg"


def main() -> None:
    cli.console = Console(record=True, width=100, force_terminal=True, file=io.StringIO())
    summary = check_suite(load_suite(ROOT / SUITE), load_responses(ROOT / RESPONSES))
    cli._print_summary(summary, verbose=False)
    svg = cli.console.export_svg(title=f"evalkit check {SUITE} {RESPONSES}")
    # LF endings and no trailing spaces, so the file passes the pre-commit hooks
    # unchanged no matter which OS renders it.
    clean = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(clean, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
