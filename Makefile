# Recipes stay single-line uv calls so they run the same under sh (Linux/CI)
# and cmd.exe (GNU make on Windows without Git Bash on PATH).

SUITE ?= examples/suites/smoke.yaml
RESPONSES ?= examples/responses/smoke.jsonl
JUDGMENTS ?= examples/judgments/smoke.jsonl
RESPONSES_B ?= examples/responses/smoke-b.jsonl
MODEL ?= qwen2.5:7b
JUDGE ?= qwen2.5:14b

.PHONY: help setup lint format typecheck test cov run run-ollama judge judge-ollama pairwise consistency docker clean

help:
	@echo "setup       install deps + pre-commit hooks"
	@echo "lint        ruff check + format check"
	@echo "format      apply ruff fixes and formatting"
	@echo "typecheck   mypy (strict) on src/"
	@echo "test        pytest with coverage"
	@echo "run         validate the example suite, check example responses, replay them via the mock"
	@echo "run-ollama  run SUITE on a local Ollama model (MODEL=qwen2.5:7b)"
	@echo "judge       judge the example responses with the mock judge (canned replies)"
	@echo "judge-ollama  judge RESPONSES with a local Ollama judge (JUDGE=qwen2.5:14b)"
	@echo "pairwise    compare RESPONSES with RESPONSES_B in both orders (mock judge)"
	@echo "consistency judge each example response 3 times (mock judge)"
	@echo "docker      build the CLI image"

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
	uv run evalkit run $(SUITE) -p mock --mock-responses $(RESPONSES)

run-ollama:
	uv run evalkit run $(SUITE) -p ollama:$(MODEL)

judge:
	uv run evalkit judge $(SUITE) $(RESPONSES) -j mock --mock-judgments $(JUDGMENTS)

judge-ollama:
	uv run evalkit judge $(SUITE) $(RESPONSES) -j ollama:$(JUDGE)

pairwise:
	uv run evalkit pairwise $(SUITE) $(RESPONSES) $(RESPONSES_B) -j mock --mock-judgments examples/judgments/smoke-pairwise.jsonl

consistency:
	uv run evalkit consistency $(SUITE) $(RESPONSES) -j mock --mock-judgments $(JUDGMENTS) -n 3

docker:
	docker build -t llm-evalkit:dev .

clean:
	uv run python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in (\".pytest_cache\", \".mypy_cache\", \".ruff_cache\", \"htmlcov\", \"dist\")]"
