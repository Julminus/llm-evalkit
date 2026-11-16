"""Text segmentation shared by the deterministic checks.

These helpers decide what counts as a "word", "sentence" or "paragraph", so
every length check in the harness agrees on the same definitions. The rules
are deliberately simple and documented, because an evaluation metric that
nobody can predict is hard to trust.
"""

from __future__ import annotations

import re
import unicodedata

# A number with internal separators ("3.14", "1,000", "50.000") is one word.
# Otherwise a word is a run of letters/digits that may be joined by hyphens or
# apostrophes, so Indonesian reduplication ("anak-anak", "kupu-kupu") and
# English contractions ("don't") each count once, as a human would count them.
_WORD_RE = re.compile(r"\d+(?:[.,]\d+)+|\w+(?:[-'’]\w+)*")

_FENCED_CODE_RE = re.compile(r"^\s*(```|~~~).*?^\s*\1[^\n]*$", re.MULTILINE | re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
# Trailing punctuation is left outside the URL: in "see https://x.org/a." the
# final dot ends the sentence.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+?(?=[.,;:!?)\]\"']*(?:\s|$))")

# Leading list markers and Markdown headings are structure, not sentence text.
# Markers: bullets, "1." / "1)", lettered options "a." / "B)", and short roman
# numerals "iv." / "IV)". A letter or numeral is a marker only when text
# follows on the line: a line that is just "B." is the answer itself.
_LINE_PREFIX_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|>\s*|[-*+•]\s+|\d+[.)]\s+"
    r"|(?:[A-Za-z]|[ivx]{1,4}|[IVX]{1,4})[.)]\s+(?=\S))"
)

# Candidate sentence end: terminal punctuation, optional closing quotes or
# brackets, then whitespace or end of line. "3.14" has no space after the dot,
# so it is never a candidate.
_BOUNDARY_RE = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")

# Abbreviations that end with a dot but do not end a sentence. Lowercase,
# without the final dot. EN and ID together, because responses mix them.
# fmt: off
_ABBREVIATIONS = frozenset({
    # English
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e",
    "cf", "al", "fig", "approx", "dept", "est", "inc", "ltd", "co", "etc",
    # Indonesian
    "dll", "dsb", "dst", "yth", "bpk", "sdr", "sdri", "jl", "tgl", "hlm",
    "kab", "kec", "prov", "drs", "ir", "pt", "rp", "no", "s.kom", "s.h",
})
# fmt: on
# Of those, the ones that often close a sentence ("..., apples, etc. Then we").
# They end a sentence only when the next word starts with a capital letter.
_MAY_END_SENTENCE = frozenset({"etc", "dll", "dsb", "dst"})
# Of those, the ones that abbreviate only before a number ("No. 27",
# "Rp. 50.000"). Anywhere else "No." is the answer word and ends a sentence.
_BEFORE_NUMBER_ONLY = frozenset({"no", "rp"})

_OPENING_PUNCT = "(\"'“‘["

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

# NFKC leaves typographic quotes alone, but models write "don’t" as often as
# "don't", and a phrase check should treat the two as the same text.
_QUOTE_FOLD = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"'})


_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def scrub_surrogates(text: str) -> str:
    """Replace lone UTF-16 surrogates with U+FFFD.

    JSON can spell one (``"\\ud83d"`` with no low half), and Python decodes it
    into a ``str`` that can't be written as UTF-8, so a single one in a
    response or a judge reply would crash the run log. Valid characters,
    emoji included, are never surrogates in a Python ``str``.
    """
    return _LONE_SURROGATE_RE.sub("�", text)


def normalize(text: str) -> str:
    """NFKC-normalize and collapse runs of whitespace to single spaces."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def fold_quotes(text: str) -> str:
    """Map typographic quotes and apostrophes to their ASCII forms."""
    return text.translate(_QUOTE_FOLD)


def strip_code(text: str) -> str:
    """Replace fenced code blocks, inline code spans and URLs with placeholders.

    Used before sentence counting and language detection: code and URLs are
    not prose in any language. Inline code and URLs become the neutral words
    ``code`` and ``url`` so the sentence around them stays intact.
    """
    text = _FENCED_CODE_RE.sub("\n", text)
    text = _INLINE_CODE_RE.sub("code", text)
    return _URL_RE.sub("url", text)


def words(text: str) -> list[str]:
    """Split text into words (see module notes for the rules)."""
    return _WORD_RE.findall(unicodedata.normalize("NFKC", text))


def _has_word_char(segment: str) -> bool:
    return any(ch.isalnum() for ch in segment)


def _is_initial_token(token: str) -> bool:
    """True for a dotted capital initial such as ``J.``."""
    token = token.lstrip(_OPENING_PUNCT)
    return len(token) == 2 and token[0].isupper() and token[1] == "."


def _is_initial(before: list[str], rest: str) -> bool:
    """True if ``before[-1]``, a single character before a dot, is a name initial.

    One predictable rule rather than a guess: a capital letter is an initial
    at the start of the sentence ("H. Agus Salim wrote") or next to another
    initial ("by J. K. Rowling"). Anywhere else it ends the sentence, which is
    right for "the answer is C. It ...", "Vitamin C. It ..." and "World War
    I. The ...", and wrong for a lone middle initial ("John F. Kennedy" counts
    as two sentences; see Known limits in the README). A lowercase letter is
    never an initial ("pilihan a.").
    """
    letter = before[-1].lstrip(_OPENING_PUNCT)
    if not letter.isupper():
        return False
    if len(before) == 1 or _is_initial_token(before[-2]):
        return True
    following = rest.split(maxsplit=1)
    return bool(following) and _is_initial_token(following[0])


def _is_abbreviation_stop(line: str, start: int, boundary: re.Match[str]) -> bool:
    """True if the dot at ``boundary`` belongs to an abbreviation, not a sentence end."""
    if not boundary.group().startswith(".") or len(boundary.group().rstrip("\"'”’)]")) > 1:
        return False  # "!", "?", or an ellipsis
    before = line[start : boundary.start()].split()
    if not before:
        return False
    rest = line[boundary.end() :].lstrip()
    token = before[-1].lstrip(_OPENING_PUNCT).lower()
    if len(token) == 1:
        return _is_initial(before, rest)
    if token not in _ABBREVIATIONS:
        return False
    if token in _BEFORE_NUMBER_ONLY:
        return rest[:1].isdigit()
    if token in _MAY_END_SENTENCE:
        return not (rest[:1].isupper())
    return True


def sentences(text: str) -> list[str]:
    """Split text into sentences.

    Rules: code blocks and URLs are ignored; every line break ends a sentence
    (so each list item or heading is its own sentence); within a line a
    sentence ends at ``.``, ``!``, ``?`` or ``…`` followed by whitespace,
    except after a known abbreviation or a name initial (see ``_is_initial``).
    "No." and "Rp." count as abbreviations only before a number. Segments
    without any letter or digit (e.g. ``---``) are dropped.
    """
    result: list[str] = []
    for raw_line in strip_code(text).splitlines():
        line = _LINE_PREFIX_RE.sub("", raw_line).strip()
        if not line:
            continue
        start = 0
        for boundary in _BOUNDARY_RE.finditer(line):
            if _is_abbreviation_stop(line, start, boundary):
                continue
            segment = line[start : boundary.end()].strip()
            if _has_word_char(segment):
                result.append(segment)
            start = boundary.end()
        tail = line[start:].strip()
        if _has_word_char(tail):
            result.append(tail)
    return result


def paragraphs(text: str) -> list[str]:
    """Split text into paragraphs: blocks separated by one or more blank lines."""
    blocks = _PARAGRAPH_SPLIT_RE.split(text.strip())
    return [b.strip() for b in blocks if _has_word_char(b)]
