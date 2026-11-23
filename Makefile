.PHONY: install test live lint check image
.DEFAULT_GOAL := check
NAME ?= qwen3-coder-30b

install:
	uv sync --all-extras

test:
	uv run pytest

live:  ## end-to-end tests against a deployment: make live NAME=<deployment>
	BYOM_LIVE=$(NAME) uv run pytest tests/test_live.py -v

lint:
	uv run ruff check byom tests examples
	uv run mypy

check: lint test

image:  ## build the proxy image from this checkout
	docker build -f docker/proxy.Dockerfile -t byom:local .
