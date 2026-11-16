"""Reading the text files evalkit takes as input (suites, responses).

Both are UTF-8. A UTF-8 byte-order mark is accepted, since Windows editors
add one. Anything else gets a message that says what the file actually is,
because the usual cause is Windows PowerShell 5.1, whose ``>`` writes UTF-16.
"""

from __future__ import annotations

from pathlib import Path

_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


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
