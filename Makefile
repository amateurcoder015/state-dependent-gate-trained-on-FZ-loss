.PHONY: all download prepare base test

all: prepare base

download:
	uv run python scripts/01_download.py

prepare:
	uv run python scripts/02_prepare.py

base:
	uv run python scripts/03_base_forecasts.py

test:
	uv run pytest -q
