import pytest

from evalkit import langdetect
from evalkit.langdetect import detect_language


def test_word_lists_are_disjoint() -> None:
    # A word in both lists would count for both languages and blur the signal.
    assert not (langdetect._EN_WORDS & langdetect._ID_WORDS)


def test_word_lists_are_lowercase_letters_only() -> None:
    for word in langdetect._EN_WORDS | langdetect._ID_WORDS:
        assert word.isalpha()
        assert word == word.lower()


@pytest.mark.parametrize(
    "source",
    [
        "The quick brown fox jumps over the lazy dog because it is bored.",
        "Unit tests give you fast feedback and make refactoring safer.",
        "Jakarta is the capital of Indonesia and its largest city.",
    ],
)
def test_english(source: str) -> None:
    guess = detect_language(source)
    assert guess.language == "en"
    assert guess.confidence == 1.0


@pytest.mark.parametrize(
    "source",
    [
        "Saya tidak bisa datang ke rapat karena sedang sakit.",
        "Pengujian unit memberikan umpan balik yang cepat dan membuat refaktor lebih aman.",
        # Code-mixed Indonesian: English nouns, Indonesian grammar.
        "Meeting-nya di-reschedule ke besok karena deadline project kita mundur.",
        "Untuk deploy ke production, kita perlu approval dari tim QA dulu.",
    ],
)
def test_indonesian(source: str) -> None:
    guess = detect_language(source)
    assert guess.language == "id"
    assert guess.confidence >= 0.8


def test_no_signal_is_unknown() -> None:
    guess = detect_language("42")
    assert guess.language is None
    assert guess.signal_words == 0
    assert guess.confidence == 0.0


def test_exact_tie_is_unknown() -> None:
    guess = detect_language("the yang")
    assert guess.language is None
    assert guess.confidence == 0.5
    assert guess.hits == {"en": 1, "id": 1}


def test_mixed_text_has_lower_confidence() -> None:
    guess = detect_language("Saya suka the book karena it is very good dan murah.")
    assert 0.5 < guess.confidence < 0.8


def test_code_and_urls_are_ignored() -> None:
    source = (
        "Jalankan perintah ini untuk instalasi:\n```\nif the file is in the path then run it\n```"
    )
    assert detect_language(source).language == "id"


def test_legal_references_do_not_count_as_english() -> None:
    # "No." and the Roman numeral "I" are deliberately not English signal words.
    guess = detect_language("Lihat UU No. 27 Tahun 2022, Bab I, tentang pelindungan data pribadi.")
    assert guess.language == "id"
    assert guess.hits["en"] == 0
