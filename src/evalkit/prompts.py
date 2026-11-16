"""The prompt a model sees, and how a suite case becomes one."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from evalkit.schema import EvalCase

__all__ = ["Prompt", "build_prompt"]


class Prompt(BaseModel):
    """A single-turn prompt: an optional system message and one user message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    user: str
    system: str | None = None

    def messages(self) -> list[dict[str, str]]:
        """Chat messages in the OpenAI / Ollama shape."""
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        msgs.append({"role": "user", "content": self.user})
        return msgs


def build_prompt(case: EvalCase) -> Prompt:
    """Render ``case`` as a single-turn prompt.

    A case's ``context`` goes before the question, inside ``<context>`` tags.
    The tags carry no language: an English label such as "Context:" in front
    of an Indonesian question would be a nudge towards answering in English,
    and language mismatch is exactly what the suite measures.
    """
    user = case.prompt
    if case.context is not None:
        user = f"<context>\n{case.context.strip()}\n</context>\n\n{case.prompt}"
    return Prompt(user=user, system=case.system)
