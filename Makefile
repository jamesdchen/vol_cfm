.PHONY: lint format format-check typecheck test check install-hooks clean

lint:
	ruff check cfm/ scripts/ tests/

format:
	ruff format cfm/ scripts/ tests/

format-check:
	ruff format --check cfm/ scripts/ tests/

typecheck:
	mypy cfm/

test:
	pytest tests/ -v

check: lint format-check typecheck test

install-hooks:
	pre-commit install

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache .pytest_cache
