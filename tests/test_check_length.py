import pytest
from pydantic import ValidationError

from evalkit.checks import ParagraphCount, SentenceCount, Status, WordCount
from evalkit.taxonomy import FailureMode


class TestWordCount:
    def test_within_max(self) -> None:
        result = WordCount(max=5).check("one two three")
        assert result.status is Status.PASS
        assert result.observed == {"words": 3}

    def test_over_max(self) -> None:
        result = WordCount(max=2).check("one two three")
        assert result.status is Status.FAIL
        assert result.failure_mode is FailureMode.CONSTRAINT_VIOLATION
        assert result.message == "3 words, expected at most 2"

    def test_under_min(self) -> None:
        result = WordCount(min=4).check("one two three")
        assert result.failed
        assert "at least 4" in result.message

    def test_exact_bounds(self) -> None:
        check = WordCount(min=3, max=3)
        assert check.check("satu dua tiga").passed
        assert "exactly 3" in check.check("satu dua").message

    def test_range_message(self) -> None:
        assert "(expected 1-10)" in WordCount(min=1, max=10).check("hi there").message

    def test_boundaries_are_inclusive(self) -> None:
        assert WordCount(min=2, max=2).check("a b").passed

    def test_reduplication_counts_once(self) -> None:
        assert WordCount(max=3).check("Anak-anak bermain kupu-kupu").passed

    def test_needs_a_bound(self) -> None:
        with pytest.raises(ValidationError, match="at least one of"):
            WordCount()

    def test_min_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than"):
            WordCount(min=5, max=2)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WordCount(max=-1)


class TestSentenceCount:
    def test_counts_sentences(self) -> None:
        result = SentenceCount(max=2).check("First. Second. Third.")
        assert result.failed
        assert result.observed == {"sentences": 3}

    def test_abbreviations_do_not_split(self) -> None:
        assert SentenceCount(max=1).check("Dr. Andi tiba pukul 10.30 di Jl. Sudirman.").passed

    def test_each_bullet_counts(self) -> None:
        assert SentenceCount(min=3, max=3).check("- one\n- two\n- three").passed


class TestParagraphCount:
    def test_two_paragraphs(self) -> None:
        source = "Paragraf pertama.\n\nParagraf kedua, masih pendek."
        assert ParagraphCount(min=2, max=2).check(source).passed

    def test_too_many(self) -> None:
        result = ParagraphCount(max=1).check("a\n\nb\n\nc")
        assert result.failed
        assert result.observed == {"paragraphs": 3}


def test_parsed_from_yaml_dict(parse) -> None:
    check = parse({"type": "word_count", "max": 50})
    assert isinstance(check, WordCount)
    assert check.max == 50


def test_unknown_key_rejected(parse) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        parse({"type": "word_count", "max_words": 50})


def test_unknown_type_rejected(parse) -> None:
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        parse({"type": "char_count", "max": 50})
