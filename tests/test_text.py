import pytest

from evalkit import text


class TestWords:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("one two  three", 3),
            ("", 0),
            ("   \n\t ", 0),
            # Indonesian reduplication is one word, as Indonesian orthography counts it.
            ("Anak-anak bermain kupu-kupu", 3),
            ("don't stop", 2),
            ("Rp 50.000 or 1,250 or 3.14", 6),
            ("COVID-19 cases", 2),
            ("**bold** and _italic_", 3),
            ("Hello, world! Apa kabar?", 4),
        ],
    )
    def test_counts(self, source: str, expected: int) -> None:
        assert len(text.words(source)) == expected

    def test_numbers_with_separators_stay_whole(self) -> None:
        assert text.words("Harganya Rp 50.000.") == ["Harganya", "Rp", "50.000"]

    def test_fullwidth_characters_are_normalized(self) -> None:
        assert text.words("ＡＢＣ ｄｅｆ") == ["ABC", "def"]


class TestSentences:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("One. Two! Three?", 3),
            ("No terminal punctuation", 1),
            ("", 0),
            ("Pi is about 3.14 here.", 1),
            ("Dr. Smith met Mr. Lee today.", 1),
            ("Kami membeli apel, jeruk, dll. untuk acara.", 1),
            # "etc."/"dll." end the sentence when a capitalized word follows.
            ("We bought apples, pears, etc. Then we left.", 2),
            ("Kami membeli apel, jeruk, dll. Lalu kami pulang.", 2),
            ("J. K. Rowling wrote it.", 1),
            ("The book is by J. K. Rowling. It sold well.", 2),
            ("H. Agus Salim lahir di Koto Gadang.", 1),
            # A lone letter after an ordinary word is an answer, not an initial.
            ("The correct answer is C. Photosynthesis needs light.", 2),
            ("Jawabannya B. Karena itu benar.", 2),
            ("Pilihan a. Lalu pilihan b.", 2),
            # "No." abbreviates only before a number; otherwise it is the answer word.
            ("No. The capital is Canberra.", 2),
            ("Lihat UU No. 27 Tahun 2022.", 1),
            ("Harganya Rp. 50.000 saja.", 1),
            ("Wait... what?", 2),
            ('He said "stop." Then he left.', 2),
            ("Is it? (Yes.) Good.", 3),
        ],
    )
    def test_counts(self, source: str, expected: int) -> None:
        assert len(text.sentences(source)) == expected

    def test_each_line_is_a_sentence(self) -> None:
        source = "# Title\n- first item\n- second item\n\n1. numbered item"
        assert text.sentences(source) == ["Title", "first item", "second item", "numbered item"]

    def test_list_marker_number_is_not_a_sentence(self) -> None:
        assert text.sentences("1. Satu.\n2. Dua.") == ["Satu.", "Dua."]

    def test_code_and_urls_are_ignored(self) -> None:
        source = "Run this.\n```python\nprint('x. y. z.')\n```\nSee https://example.com/a.b. Done."
        assert text.sentences(source) == ["Run this.", "See url.", "Done."]

    def test_line_starting_with_a_dot(self) -> None:
        assert text.sentences(". Hello.") == ["Hello."]

    def test_rules_and_punctuation_only_lines_are_dropped(self) -> None:
        assert text.sentences("Intro.\n---\n...\nOutro.") == ["Intro.", "Outro."]


class TestParagraphs:
    def test_blank_lines_separate_paragraphs(self) -> None:
        assert len(text.paragraphs("A b c.\n\nD e f.\n   \nG h.")) == 3

    def test_single_newlines_do_not(self) -> None:
        assert len(text.paragraphs("line one\nline two\n- item")) == 1

    def test_empty(self) -> None:
        assert text.paragraphs("  \n\n ") == []


def test_normalize_collapses_whitespace() -> None:
    assert text.normalize("  a\n\tb   c ") == "a b c"


def test_fold_quotes_maps_typographic_quotes_to_ascii() -> None:
    assert text.fold_quotes("don’t ‘a’ “b”") == "don't 'a' \"b\""


def test_strip_code_removes_fences_inline_code_and_urls() -> None:
    stripped = text.strip_code("Use `pip`.\n~~~\ncode here\n~~~\nwww.example.com end")
    assert "pip" not in stripped
    assert "code here" not in stripped
    assert "example" not in stripped
    assert stripped.startswith("Use code.")
    assert stripped.endswith("url end")
