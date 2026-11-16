import pytest
from pydantic import ValidationError

from evalkit.checks import LanguageCheck, Status
from evalkit.taxonomy import FailureMode

ID_TEXT = (
    "Pengujian unit membantu kita menemukan bug lebih awal dan membuat kode lebih mudah dirawat."
)
EN_TEXT = "Unit tests help you find bugs early and make the code easier to maintain."


def test_expected_language_passes() -> None:
    result = LanguageCheck(expected="id").check(ID_TEXT)
    assert result.passed
    assert result.observed["detected"] == "id"
    assert result.observed["confidence"] == 1.0


def test_wrong_language_fails() -> None:
    result = LanguageCheck(expected="id").check(EN_TEXT)
    assert result.failed
    assert result.failure_mode is FailureMode.LANGUAGE_MISMATCH
    assert result.message == "expected id, detected en"


def test_short_answer_is_skipped_not_failed() -> None:
    result = LanguageCheck(expected="id").check("42")
    assert result.status is Status.SKIP
    assert "too little text" in result.message


def test_heavily_mixed_response_fails() -> None:
    # 6 Indonesian vs 4 English function words: Indonesian wins at 60%, below the 70% bar.
    mixed = "Saya rasa poin utamanya adalah kita harus test it before deploy ke production karena the risk is high."
    result = LanguageCheck(expected="id").check(mixed)
    assert result.failed
    assert result.observed["hits"] == {"en": 4, "id": 6}
    assert "mostly id but mixed (60% < 70% confidence)" in result.message


def test_even_mix_reports_no_winner() -> None:
    result = LanguageCheck(expected="en").check("the yang of dan")
    assert result.failed
    assert "even EN/ID mix" in result.message


def test_thresholds_are_configurable() -> None:
    mixed = "Saya rasa poin utamanya adalah kita harus test it before deploy ke production karena the risk is high."
    assert LanguageCheck(expected="id", min_confidence=0.6).check(mixed).passed
    assert LanguageCheck(expected="id", min_signal_words=50).check(ID_TEXT).status is Status.SKIP


def test_confidence_bounds_validated() -> None:
    with pytest.raises(ValidationError):
        LanguageCheck(expected="en", min_confidence=0.3)


def test_only_en_and_id_supported(parse) -> None:
    with pytest.raises(ValidationError):
        parse({"type": "language", "expected": "ms"})
