"""llm-evalkit: a bilingual (EN/ID) harness for evaluating LLM outputs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-evalkit")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
