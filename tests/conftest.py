from typing import Any

import pytest
from pydantic import TypeAdapter

from evalkit.checks import Constraint

_ADAPTER: TypeAdapter[Any] = TypeAdapter(Constraint)


@pytest.fixture
def parse():
    """Parse a constraint dict exactly as it would appear in a YAML suite."""

    def _parse(data: dict[str, Any]) -> Any:
        return _ADAPTER.validate_python(data)

    return _parse
