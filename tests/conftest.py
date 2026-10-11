"""Pytest setup for QUANTA tests."""

import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src and repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest

SLOW_OR_BENCHMARK_MODULES = {
    # Live LLM inference / local GPU server benchmarks
    "test_cold_start_expansion",
    "test_context_expansion_server",
    "test_mcp_reverse_proxy_parity",
    "test_proxy_session_pool",
    "test_llm_mcp_eval",
    "test_compact_prompt_sweep",
    "test_positional_prompt_sweep",
    "test_phase_six_lora_pipeline",
    "test_unsloth_manager",
    "test_unsloth_transducer",
    # Benchmarks and heavy scaling runs
    "test_paired_benchmarks",
    "test_long_context_bench",
    "test_benchmark_musique_representation",
    "test_performance_improvement",
    "test_page_table_scaling",
    "test_multihop_reasoning_scale",
    "test_dimension_sweep",
    "test_kev_ablation",
    "test_end_to_end_suite",
    "test_mmap_grounder_speed",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow, live-LLM, and heavy benchmark tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow/benchmark/live-LLM to exclude by default"
    )


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow", default=False)
    marker_expr = config.getoption("-m", default="")
    if "slow" in marker_expr or run_slow:
        return

    # Check if a specific test file or method was directly targeted on the CLI
    cli_files = {Path(arg).stem for arg in config.args if "test_" in arg}

    skip_slow = pytest.mark.skip(
        reason="Skipped by default (slow/benchmark/live-LLM). Run with --run-slow or target test specifically."
    )

    for item in items:
        file_stem = Path(item.fspath).stem if hasattr(item, "fspath") else ""
        if file_stem in cli_files:
            continue

        if "slow" in item.keywords or file_stem in SLOW_OR_BENCHMARK_MODULES:
            item.add_marker(pytest.mark.slow)
            item.add_marker(skip_slow)

