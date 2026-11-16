import pytest
from pydantic import ValidationError

from evalkit.checks import ForbiddenPhrases, RequiredPhrases, Status


class TestRequiredPhrases:
    def test_all_present(self) -> None:
        result = RequiredPhrases(phrases=["deadline", "Jumat"]).check(
            "Deadline-nya hari Jumat, jangan telat."
        )
        assert result.passed

    def test_missing_reported(self) -> None:
        result = RequiredPhrases(phrases=["deadline", "budget"]).check("The deadline is Friday.")
        assert result.failed
        assert result.observed == {"missing": ["budget"]}

    def test_case_insensitive_by_default(self) -> None:
        assert RequiredPhrases(phrases=["JAKARTA"]).check("di jakarta").passed

    def test_case_sensitive(self) -> None:
        assert RequiredPhrases(phrases=["Jakarta"], case_sensitive=True).check("jakarta").failed

    def test_whole_word_by_default(self) -> None:
        # "AI" must not match inside "said".
        assert RequiredPhrases(phrases=["AI"]).check("She said hello.").failed

    def test_substring_mode(self) -> None:
        assert RequiredPhrases(phrases=["data"], whole_word=False).check("database").passed

    def test_phrase_spanning_a_line_break(self) -> None:
        assert (
            RequiredPhrases(phrases=["pelindungan data pribadi"])
            .check("UU tentang pelindungan\ndata   pribadi")
            .passed
        )

    def test_phrase_with_punctuation_edges(self) -> None:
        assert RequiredPhrases(phrases=["C++"]).check("I like C++ a lot").passed
        assert RequiredPhrases(phrases=["C++"]).check("I like C a lot").failed

    def test_any_mode(self) -> None:
        check = RequiredPhrases(phrases=["terima kasih", "makasih"], match="any")
        assert check.check("Makasih ya!").passed
        result = check.check("Oke.")
        assert result.failed
        assert "none of" in result.message

    def test_blank_phrase_rejected(self) -> None:
        with pytest.raises(ValidationError, match="blank"):
            RequiredPhrases(phrases=["ok", "  "])

    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RequiredPhrases(phrases=[])


class TestForbiddenPhrases:
    def test_clean(self) -> None:
        result = ForbiddenPhrases(phrases=["as an AI"]).check("Here is the summary.")
        assert result.status is Status.PASS

    def test_found(self) -> None:
        result = ForbiddenPhrases(phrases=["as an AI", "sebagai AI"]).check(
            "Sebagai AI, saya tidak punya pendapat."
        )
        assert result.failed
        assert result.observed == {"found": ["sebagai AI"]}

    def test_whole_word_avoids_false_positive(self) -> None:
        assert ForbiddenPhrases(phrases=["hell"]).check("Say hello").passed

    def test_unicode_normalization(self) -> None:
        # A fullwidth "ＡＩ" in the response is the same phrase after NFKC.
        assert ForbiddenPhrases(phrases=["AI"]).check("Saya ＡＩ.").failed


def test_parse(parse) -> None:
    check = parse({"type": "forbidden_phrases", "phrases": ["lorem"]})
    assert isinstance(check, ForbiddenPhrases)
