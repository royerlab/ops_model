# Running Scripts and Tests

All commands must be run from the **monorepo root** (`ops_monorepo/`) using `uv run`,
which ensures the full workspace environment (all subpackages installed in editable mode) is active.

## Prerequisites

```bash
module load uv
uv sync   # first time, or after pulling new changes
```

## Running Python scripts

```bash
uv run python ops_model/my_script.py
```

## Running entry-point CLI tools

Entry points defined in `pyproject.toml` (e.g. `run_eval`) are available via `uv run`:

```bash
uv run run_eval --guide_embedding path/to/guide.h5ad --output results.csv
```

## Running tests

```bash
# All ops_model tests
uv run pytest ops_model/tests/

# A specific test file or directory
uv run pytest ops_model/tests/eval/

# With verbose output
uv run pytest ops_model/tests/eval/ -v
```

## Notes

- Do **not** activate a conda/venv manually — `uv run` handles the environment.
- Do **not** use `python` or `pytest` directly — use `uv run python` / `uv run pytest`.
- All commands must be run from `ops_monorepo/`, not from within a subpackage directory.
