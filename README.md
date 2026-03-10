# jaguar-geo-assign

Lightweight `uv` bootstrap for a leakage-safe transfer-learning pipeline focused on jaguar geographic assignment.

## Quickstart

- `uv sync`
- `uv run pytest`
- `uv run python -m jaguar_geo_assign.cli --help`

## Repository layout

- `src/jaguar_geo_assign/`: package source and CLI entry points
- `configs/examples/`: versioned bootstrap configs that do not require private data
- `tests/`: unit tests for config validation and CLI behavior
- `scripts/`: placeholder location for developer helpers
- `data/`: ignored raw/processed data placeholders
- `artifacts/`: ignored checkpoints and run outputs
- `reports/generated/`: ignored generated report outputs

## Bootstrap guarantees

- Python `3.11` is pinned via `.python-version` and `pyproject.toml`
- jaguar metadata contract is limited to `sample_id`, `individual_id`, `locality_id`, biome/population label, `latitude`, and `longitude`
- legacy baseline support is intentionally deferred to a shared extension point