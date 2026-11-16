import json
import os
from pathlib import Path

import pytest

from evalkit.cache import CACHE_VERSION, ResponseCache, cache_key
from evalkit.prompts import Prompt
from evalkit.providers import Completion

IDENT = {"provider": "ollama", "model": "qwen2.5:7b"}
PROMPT = Prompt(user="Halo", system="Singkat.")
PARAMS = {"temperature": 0, "seed": 0}
COMPLETION = Completion(
    text="Halo juga!", model="qwen2.5:7b", finish_reason="stop", output_tokens=3
)


class TestKey:
    def test_is_sha256_hex(self) -> None:
        key = cache_key(IDENT, PROMPT, PARAMS)
        assert len(key) == 64
        int(key, 16)

    def test_dict_order_does_not_matter(self) -> None:
        a = cache_key({"provider": "ollama", "model": "m"}, PROMPT, {"seed": 0, "temperature": 0})
        b = cache_key({"model": "m", "provider": "ollama"}, PROMPT, {"temperature": 0, "seed": 0})
        assert a == b

    @pytest.mark.parametrize(
        "change",
        [
            {"identity": {"provider": "ollama", "model": "llama3.1:8b"}},
            {"identity": {**IDENT, "base_url": "http://other"}},
            {"prompt": Prompt(user="Halo")},
            {"prompt": Prompt(user="Halo!", system="Singkat.")},
            {"params": {"temperature": 0.7, "seed": 0}},
            {"params": {}},
            {"sample": 1},
        ],
    )
    def test_any_input_change_changes_key(self, change: dict) -> None:
        base = {"identity": IDENT, "prompt": PROMPT, "params": PARAMS, "sample": 0}
        args = {**base, **change}
        changed = cache_key(args["identity"], args["prompt"], args["params"], sample=args["sample"])
        assert changed != cache_key(IDENT, PROMPT, PARAMS)

    def test_version_is_part_of_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = cache_key(IDENT, PROMPT, PARAMS)
        monkeypatch.setattr("evalkit.cache.CACHE_VERSION", CACHE_VERSION + 1)
        assert cache_key(IDENT, PROMPT, PARAMS) != before


class TestStore:
    def test_round_trip_and_layout(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        key = cache_key(IDENT, PROMPT, PARAMS)
        assert cache.get(key) is None
        cache.put(key, COMPLETION, latency_s=1.25, request={"prompt": PROMPT.model_dump()})
        path = tmp_path / key[:2] / f"{key}.json"
        assert path.is_file()
        entry = cache.get(key)
        assert entry is not None
        assert entry.completion == COMPLETION
        assert entry.latency_s == 1.25
        # Human-readable on disk, LF line endings, no stray temp files.
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert json.loads(raw)["request"]["prompt"]["user"] == "Halo"
        assert [p.name for p in path.parent.iterdir()] == [path.name]

    def test_overwrite(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cache.put("ab" * 32, COMPLETION, latency_s=1, request={})
        other = COMPLETION.model_copy(update={"text": "baru"})
        cache.put("ab" * 32, other, latency_s=2, request={})
        entry = cache.get("ab" * 32)
        assert entry is not None
        assert entry.completion.text == "baru"

    @pytest.mark.parametrize("content", ["{not json", '{"key": "x"}', ""])
    def test_corrupt_entry_is_a_miss(self, tmp_path: Path, content: str) -> None:
        cache = ResponseCache(tmp_path)
        key = "cd" * 32
        cache.path(key).parent.mkdir(parents=True)
        cache.path(key).write_text(content, encoding="utf-8")
        assert cache.get(key) is None

    def test_entry_for_another_key_is_a_miss(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cache.put("ef" * 32, COMPLETION, latency_s=1, request={})
        moved = cache.path("e0" + "f" * 62)
        moved.parent.mkdir(parents=True)
        os.replace(cache.path("ef" * 32), moved)
        assert cache.get("e0" + "f" * 62) is None

    def test_failed_write_leaves_no_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = ResponseCache(tmp_path)

        def boom(src: str, dst: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("evalkit.files.os.replace", boom)
        with pytest.raises(OSError, match="disk full"):
            cache.put("12" * 32, COMPLETION, latency_s=1, request={})
        assert list((tmp_path / "12").iterdir()) == []
