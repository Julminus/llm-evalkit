import os
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
    # The umask is read at write time, not once at import.
    old = os.umask(0o027)
    try:
        files.write_text_atomic(tmp_path / "strict.json", "{}\n")
    finally:
        os.umask(old)
    assert (tmp_path / "strict.json").stat().st_mode & 0o777 == 0o640


def test_temp_name_is_shorter_than_a_cache_entry_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On Windows without long paths, a longer temp name than the target's
    # could fail where the target itself fits.
    seen: list[str] = []
    real = files.replace_file
    monkeypatch.setattr(files, "replace_file", lambda s, d: (seen.append(Path(s).name), real(s, d)))
    target = tmp_path / ("ab" * 32 + ".json")  # a cache entry name
    files.write_text_atomic(target, "{}")
    assert len(seen[0]) < len(target.name)


def test_an_existing_temp_name_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the exclusive create fails, the file that is already there belongs
    # to someone else and must survive.
    monkeypatch.setattr(files.uuid, "uuid4", lambda: type("U", (), {"hex": "0" * 32})())
    other = tmp_path / ".tmp-000000000000"
    other.write_text("not mine", encoding="utf-8")
    with pytest.raises(FileExistsError):
        files.write_text_atomic(tmp_path / "x.json", "{}")
    assert other.read_text(encoding="utf-8") == "not mine"


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
