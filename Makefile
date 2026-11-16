# Recipes stay single-line uv calls so they run the same under sh (Linux/CI)
# and cmd.exe (GNU make on Windows without Git Bash on PATH).

SUITE ?= examples/suites/smoke.yaml
RESPONSES ?= examples/responses/smoke.jsonl

.PHONY: help setup lint format typecheck test cov run docker clean

help:
	@echo "setup      install deps + pre-commit hooks"
	@echo "lint       ruff check + format check"
	@echo "format     apply ruff fixes and formatting"
	@echo "typecheck  mypy (strict) on src/"
	@echo "test       pytest with coverage"
	@echo "run        validate the example suite and check example responses"
	@echo "docker     build the CLI image"

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

run:
	uv run evalkit validate $(SUITE)
	uv run evalkit check $(SUITE) $(RESPONSES)

docker:
	docker build -t llm-evalkit:dev .

clean:
	uv run python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in (\".pytest_cache\", \".mypy_cache\", \".ruff_cache\", \"htmlcov\", \"dist\")]"
