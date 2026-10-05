.PHONY: all download prepare base combos gate table test

all: prepare base combos gate table

download:
	uv run python scripts/01_download.py

prepare:
	uv run python scripts/02_prepare.py

base:
	uv run python scripts/03_base_forecasts.py

combos:
	uv run python scripts/04_combinations.py

gate:
	uv run python scripts/05_gate.py

table:
	uv run python scripts/06_fz0_table.py

test:
	uv run pytest -q
