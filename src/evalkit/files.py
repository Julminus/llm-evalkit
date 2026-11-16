"""File helpers: reading input files, writing output files atomically.

Inputs (suites, responses) are UTF-8. A UTF-8 byte-order mark is accepted,
since Windows editors add one. Anything else gets a message that says what the
file actually is, because the usual cause is Windows PowerShell 5.1, whose
``>`` writes UTF-16.

Outputs (cache entries, run logs, manifests) are written to a temp file next
to the target and renamed into place, so a crash or Ctrl-C never leaves half
a file, and two runs writing the same cache entry can't interleave.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_REPLACE_ATTEMPTS = 5
_REPLACE_FIRST_DELAY = 0.02  # seconds; doubles each attempt, ~0.3 s in total


def replace_file(src: str | Path, dst: str | Path) -> None:
    """``os.replace``, retried briefly on ``PermissionError``.

    On Windows, replacing a file fails while another process holds it open
    without delete sharing: ``Get-Content -Wait`` on a run log, an editor, a
    virus scan, or a second evalkit run reading the same cache entry. Those
    holds are usually brief. The waits block, so they are kept short.
    """
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_FIRST_DELAY * 2**attempt)


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` as UTF-8 with LF endings, all or nothing.

    The temp file is created with a plain exclusive ``open()``, so it gets
    the usual umask-based mode. ``mkstemp`` would give 0600, and the rename
    keeps the mode, so a run log written in the Docker image (uid 1000) to a
    bind mount would be unreadable to the host user. Its name is short
    (shorter than a cache entry's), so it never pushes a path past Windows'
    260-character limit when the target itself fits.
    """
    tmp = path.with_name(f".tmp-{uuid.uuid4().hex[:12]}")
    fh = tmp.open("x", encoding="utf-8", newline="\n")  # fails before creating anything
    try:
        with fh:
            fh.write(text)
        replace_file(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class NotUtf8Error(ValueError):
    """The file is not UTF-8 text. The message is user-facing."""


def read_utf8(path: Path) -> str:
    """Return the file's text. ``OSError`` propagates; bad encodings raise :class:`NotUtf8Error`."""
    data = path.read_bytes()
    if data.startswith(_UTF16_BOMS):
        raise NotUtf8Error(
            "file is UTF-16, not UTF-8 (Windows PowerShell 5.1 writes UTF-16 with '>'); "
            "re-save it as UTF-8, e.g. with Out-File -Encoding utf8"
        )
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise NotUtf8Error(
            f"file is not UTF-8 (byte 0x{data[exc.start]:02x} at offset {exc.start}); "
            "re-save it as UTF-8"
        ) from exc
