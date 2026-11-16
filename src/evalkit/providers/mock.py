"""A deterministic, offline provider for tests and CI.

The mock's output is a pure function of the prompt: either a canned response
looked up by prompt, or a fixed placeholder derived from the prompt's hash.
Because it never depends on anything else, caching and re-running it behave
exactly like they would for a real model at temperature 0.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from evalkit.prompts import Prompt, build_prompt
from evalkit.providers.base import (
    Completion,
    GenerationParams,
    Provider,
    ProviderConfigError,
)
from evalkit.schema import Suite

__all__ = ["MockProvider"]


def _prompt_digest(prompt: Prompt) -> str:
    blob = json.dumps(prompt.model_dump(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class MockProvider(Provider):
    """Returns canned responses; anything else gets ``mock-<hash>``.

    The placeholder contains no words, so the automatic language check skips
    it instead of failing it. ``max_tokens`` is honoured by cutting the text to
    that many whitespace-separated words (finish reason ``length``), so the
    truncation path can be exercised offline. Other settings are accepted and
    have no effect.
    """

    name = "mock"
    supported_params = frozenset({"temperature", "top_p", "max_tokens", "seed", "stop"})

    def __init__(self, model: str = "default", responses: Mapping[Prompt, str] | None = None):
        super().__init__(model)
        self._responses = dict(responses or {})
        table = sorted((_prompt_digest(p), r) for p, r in self._responses.items())
        self._table_digest = hashlib.sha256(json.dumps(table).encode()).hexdigest()[:16]

    @classmethod
    def for_suite(
        cls, suite: Suite, responses: Mapping[str, str], model: str = "default"
    ) -> MockProvider:
        """Answer each case of ``suite`` with ``responses[case.id]``.

        Raises ProviderConfigError when two cases render to the same prompt but
        have different responses: a function of the prompt can't return both.
        """
        table: dict[Prompt, str] = {}
        owner: dict[Prompt, str] = {}
        for case in suite.cases:
            if case.id not in responses:
                continue
            prompt = build_prompt(case)
            if prompt in table and table[prompt] != responses[case.id]:
                raise ProviderConfigError(
                    f"mock: cases {owner[prompt]!r} and {case.id!r} have the same prompt "
                    "but different responses"
                )
            table[prompt] = responses[case.id]
            owner.setdefault(prompt, case.id)
        return cls(model, table)

    def identity(self) -> dict[str, str]:
        # A different response table is a different "model": never reuse its cache.
        return {**super().identity(), "responses": self._table_digest}

    async def generate(self, prompt: Prompt, params: GenerationParams) -> Completion:
        text = self._responses.get(prompt)
        if text is None:
            text = f"mock-{_prompt_digest(prompt)[:12]}"
        words = text.split()
        finish = "stop"
        if params.max_tokens is not None and len(words) > params.max_tokens:
            words = words[: params.max_tokens]
            text = " ".join(words)
            finish = "length"
        return Completion(
            text=text,
            model=self.model,
            finish_reason=finish,
            raw_finish_reason=finish,
            input_tokens=len(prompt.user.split()) + len((prompt.system or "").split()),
            output_tokens=len(words),
        )
