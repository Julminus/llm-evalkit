import pytest
from pydantic import ValidationError

from evalkit.checks import RegexCheck


def test_match_found() -> None:
    result = RegexCheck(pattern=r"Pasal \d+").check("Lihat Pasal 20 dan Pasal 21.")
    assert result.passed
    assert result.observed == {"matches": 2}


def test_no_match_fails() -> None:
    result = RegexCheck(pattern=r"Pasal \d+").check("Tidak ada rujukan.")
    assert result.failed
    assert "at least 1" in result.message


def test_expect_no_match() -> None:
    check = RegexCheck(pattern=r"https?://", expect="no_match")
    assert check.check("no links here").passed
    result = check.check("see http://x.org")
    assert result.failed
    assert result.observed == {"matches": 1}


def test_flags() -> None:
    check = RegexCheck(pattern="^ringkasan:", flags=("ignorecase", "multiline"))
    assert check.check("Intro\nRINGKASAN: ok").passed
    assert RegexCheck(pattern="^ringkasan:").check("Intro\nRINGKASAN: ok").failed


def test_dotall_flag() -> None:
    assert RegexCheck(pattern="a.b", flags=("dotall",)).check("a\nb").passed


def test_fullmatch_integer_answer() -> None:
    check = RegexCheck(pattern=r"-?\d+", fullmatch=True)
    assert check.check("  42\n").passed
    result = check.check("The answer is 42.")
    assert result.failed
    assert "does not fully match" in result.message


def test_fullmatch_no_match() -> None:
    check = RegexCheck(pattern=r"\d+", fullmatch=True, expect="no_match")
    assert check.check("forty-two").passed
    assert "fully matches" in check.check("42").message


def test_match_count_range() -> None:
    check = RegexCheck(pattern=r"\[Pasal \d+\]", min_matches=2, max_matches=3)
    assert check.check("[Pasal 1] [Pasal 2]").passed
    assert check.check("[Pasal 1]").failed
    result = check.check("[Pasal 1] [Pasal 2] [Pasal 3] [Pasal 4]")
    assert result.failed
    assert "at most 3" in result.message


def test_invalid_pattern_rejected_at_load() -> None:
    with pytest.raises(ValidationError, match="invalid regex"):
        RegexCheck(pattern="(unclosed")


def test_counts_with_no_match_rejected() -> None:
    with pytest.raises(ValidationError, match="only apply"):
        RegexCheck(pattern="x", expect="no_match", max_matches=2)


def test_counts_with_fullmatch_rejected() -> None:
    with pytest.raises(ValidationError, match="only apply"):
        RegexCheck(pattern="x", fullmatch=True, min_matches=2)


def test_max_below_min_rejected() -> None:
    with pytest.raises(ValidationError, match="less than"):
        RegexCheck(pattern="x", min_matches=3, max_matches=2)


def test_unknown_flag_rejected(parse) -> None:
    with pytest.raises(ValidationError):
        parse({"type": "regex", "pattern": "x", "flags": ["verbose"]})
