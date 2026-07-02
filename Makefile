PORT ?= 8000

.DEFAULT_GOAL := help

.PHONY: help install run lint format typecheck test

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (includes dev extras: pytest, ruff, mypy)
	uv sync --extra dev

run: ## Start uvicorn with hot-reload (PORT default: 8000)
	PYTHONPATH=src uvicorn vitrina_cv.main:app --reload --port $(PORT)

lint: ## Check code style with ruff (check + format)
	ruff check . && ruff format --check .

format: ## Auto-fix style issues with ruff
	ruff check --fix . && ruff format .

typecheck: ## Run mypy static type checks
	PYTHONPATH=src mypy src/

test: ## Run test suite with coverage
	PYTHONPATH=src pytest --cov=src/vitrina_cv --cov-report=term-missing
