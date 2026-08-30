# QUANTA justfile

default:
    @just --list

# Run test suite with pytest
test:
    python -m pytest -v

# Run linter (ruff check)
lint:
    python -m ruff check src tests

# Format codebase (ruff format)
format:
    python -m ruff format src tests
    python -m ruff check --fix src tests

# Regenerate canonical slot layouts
export-slots:
    python src/scripts/export_slots.py

# Remove cache and build artifacts
clean:
    python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('.pytest_cache')]; [p.unlink() for p in pathlib.Path('.').glob('.coverage*') if p.is_file()]"
