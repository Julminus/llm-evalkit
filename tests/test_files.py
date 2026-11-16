import sys
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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_atomic_write_uses_the_normal_file_mode(tmp_path: Path) -> None:
    # mkstemp creates 0600; run logs and manifests must stay readable to
    # other users (e.g. a host user reading a Docker bind mount).
    files.write_text_atomic(tmp_path / "manifest.json", "{}\n")
    plain = tmp_path / "plain.txt"
    plain.write_text("x", encoding="utf-8")
    mode = (tmp_path / "manifest.json").stat().st_mode & 0o777
    assert mode == plain.stat().st_mode & 0o777
    assert mode & 0o044  # group/other can read under a typical umask


def test_atomic_write_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "x.json"
    files.write_text_atomic(target, "satu\ndua\n")
    files.write_text_atomic(target, "tiga\n")
    assert target.read_bytes() == b"tiga\n"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_read_utf8_strips_a_bom(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_bytes(b"\xef\xbb\xbfhalo")
    assert files.read_utf8(path) == "halo"
