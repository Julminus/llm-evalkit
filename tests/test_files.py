from pathlib import Path
from typing import Any

import pytest

from evalkit import files


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []
    monkeypatch.setattr(files.time, "sleep", waits.append)
    return waits


def flaky_replace(monkeypatch: pytest.MonkeyPatch, failures: int) -> list[int]:
    calls: list[int] = []
    real = files.os.replace

    def replace(src: Any, dst: Any) -> None:
        calls.append(1)
        if len(calls) <= failures:
            raise PermissionError(13, "The process cannot access the file")
        real(src, dst)

    monkeypatch.setattr(files.os, "replace", replace)
    return calls


def test_replace_file_retries_a_brief_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    src, dst = tmp_path / "a.tmp", tmp_path / "a"
    src.write_text("new", encoding="utf-8")
    calls = flaky_replace(monkeypatch, failures=2)
    files.replace_file(src, dst)
    assert dst.read_text(encoding="utf-8") == "new"
    assert len(calls) == 3
    assert no_sleep == [0.02, 0.04]


def test_replace_file_gives_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    src = tmp_path / "a.tmp"
    src.write_text("new", encoding="utf-8")
    flaky_replace(monkeypatch, failures=99)
    with pytest.raises(PermissionError):
        files.replace_file(src, tmp_path / "a")
    assert sum(no_sleep) == pytest.approx(0.3)  # bounded: this blocks the event loop


def test_read_utf8_strips_a_bom(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_bytes(b"\xef\xbb\xbfhalo")
    assert files.read_utf8(path) == "halo"
