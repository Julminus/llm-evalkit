import pytest
from pydantic import ValidationError

from evalkit.checks import BulletListCheck, EndsWith, StartsWith
from evalkit.taxonomy import FailureMode

THREE_BULLETS = "- Feedback cepat\n- Refaktor lebih aman\n- Dokumentasi hidup"


class TestBulletList:
    def test_exact_count(self) -> None:
        result = BulletListCheck(min_items=3, max_items=3).check(THREE_BULLETS)
        assert result.passed
        assert result.observed == {"items": 3}

    def test_too_few(self) -> None:
        result = BulletListCheck(min_items=4).check(THREE_BULLETS)
        assert result.failed
        assert result.failure_mode is FailureMode.FORMAT_ERROR
        assert "at least 4" in result.message

    def test_too_many(self) -> None:
        assert "at most 2" in BulletListCheck(max_items=2).check(THREE_BULLETS).message

    def test_no_list(self) -> None:
        result = BulletListCheck(min_items=1).check("Just a paragraph.")
        assert result.failed
        assert result.message == "no list items found"

    @pytest.mark.parametrize("marker", ["-", "*", "+", "•"])
    def test_unordered_markers(self, marker: str) -> None:
        source = f"{marker} satu\n{marker} dua"
        assert BulletListCheck(min_items=2, max_items=2).check(source).passed

    def test_ordered_markers(self) -> None:
        source = "1. satu\n2) dua\n3. tiga"
        assert BulletListCheck(min_items=3, style="ordered").check(source).passed

    def test_style_mismatch(self) -> None:
        result = BulletListCheck(style="ordered").check(THREE_BULLETS)
        assert result.failed
        assert "3 of 3 items are not ordered" in result.message

    def test_nested_items_are_not_counted(self) -> None:
        source = "- Kelebihan\n  - cepat\n  - murah\n- Kekurangan\n    - rumit"
        assert BulletListCheck(min_items=2, max_items=2).check(source).passed

    def test_bold_and_rules_are_not_items(self) -> None:
        source = "**Catatan:** penting\n---\n- satu"
        result = BulletListCheck(min_items=1).check(source)
        assert result.observed == {"items": 1}

    def test_preamble_allowed_by_default(self) -> None:
        assert BulletListCheck(min_items=3).check("Here are three:\n" + THREE_BULLETS).passed

    def test_only_list_rejects_preamble(self) -> None:
        result = BulletListCheck(min_items=3, only_list=True).check(
            "Here are three:\n" + THREE_BULLETS
        )
        assert result.failed
        assert "Here are three" in result.message

    def test_only_list_allows_continuation_lines_and_blanks(self) -> None:
        source = "- first item\n  wraps onto a second line\n\n- second item"
        assert BulletListCheck(min_items=2, only_list=True).check(source).passed

    def test_min_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BulletListCheck(min_items=3, max_items=1)


class TestStartsWith:
    def test_starts(self) -> None:
        assert StartsWith(text="Ringkasan:").check("Ringkasan: semuanya aman.").passed

    def test_case_insensitive_and_whitespace(self) -> None:
        assert StartsWith(text="ringkasan:").check("\n  RINGKASAN:   ok").passed

    def test_loose_ignores_markdown(self) -> None:
        assert StartsWith(text="Summary").check("## **Summary**\nAll good.").passed

    def test_strict_mode(self) -> None:
        check = StartsWith(text="Summary", loose=False, case_sensitive=True)
        assert check.check("**Summary**").failed
        assert check.check("summary").failed
        assert check.check("Summary: ok").passed

    def test_failure_shows_actual_start(self) -> None:
        result = StartsWith(text="Ya").check("Tidak, karena...")
        assert result.failed
        assert result.failure_mode is FailureMode.FORMAT_ERROR
        assert "'Tidak, karena...'" in result.message


class TestEndsWith:
    def test_ends(self) -> None:
        assert EndsWith(text="Ada pertanyaan lain?").check("Begitu. Ada pertanyaan lain?").passed

    def test_loose_ignores_trailing_markdown(self) -> None:
        assert (
            EndsWith(text="Is there anything else?")
            .check("Done.\n\n**Is there anything else?**")
            .passed
        )

    def test_trailing_whitespace_ignored(self) -> None:
        assert EndsWith(text="Selesai.").check("Selesai.  \n\n").passed

    def test_strict_mode(self) -> None:
        check = EndsWith(text="Done.", loose=False)
        assert check.check("All **Done.**").failed
        assert check.check("All Done.").passed

    def test_fails(self) -> None:
        result = EndsWith(text="Terima kasih.").check("Sekian dan sampai jumpa.")
        assert result.failed
        assert "sampai jumpa." in result.message

    def test_blank_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EndsWith(text="")


def test_parse(parse) -> None:
    assert isinstance(parse({"type": "bullet_list", "min_items": 2}), BulletListCheck)
    assert isinstance(parse({"type": "starts_with", "text": "Ya"}), StartsWith)
    assert isinstance(parse({"type": "ends_with", "text": "."}), EndsWith)
