# QUANTA Makefile
.PHONY: help test lint format export-slots clean

PYTHON ?= python

help:
	@echo "QUANTA Development Targets:"
	@echo "  make test         Run test suite with pytest"
	@echo "  make lint         Run linter (ruff check)"
	@echo "  make format       Format codebase (ruff format)"
	@echo "  make export-slots Regenerate canonical slot layouts"
	@echo "  make clean        Remove cache and build artifacts"

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

export-slots:
	$(PYTHON) src/scripts/export_slots.py

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('.pytest_cache')]; [p.unlink() for p in pathlib.Path('.').glob('.coverage*') if p.is_file()]"
