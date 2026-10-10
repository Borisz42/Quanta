"""Unit and contract tests for parallel slot concurrency optimization (§Section 4).

Verifies:
1. MultiScaleConfig slot parameter registry defaults, profile loading, and env overrides.
2. CognitivePipeline batch workers initialization, runtime update, and passage ingestion.
3. benchmark_parallel_slots sweep execution in mock mode with telemetry and metric collection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from config.multi_scale_config import MultiScaleConfig
from pipeline.cognitive_pipeline import CognitivePipeline
from scripts.benchmark_parallel_slots import (
    parse_slots_arg,
    run_parallel_slot_sweep,
)


def test_multi_scale_config_slot_defaults(tmp_path: Path):
    """Verifies default values of slot configuration parameters."""
    empty_profile = tmp_path / "empty_profile.json"
    cfg = MultiScaleConfig(profile_path=empty_profile)

    assert cfg.server_max_parallel_slots == 8
    assert cfg.server_batch_workers == 8
    assert not cfg.is_calibrated("server.max_parallel_slots")
    assert not cfg.is_calibrated("server.batch_workers")


def test_multi_scale_config_slot_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verifies environment variable overrides for slot configuration."""
    empty_profile = tmp_path / "empty_profile.json"

    monkeypatch.setenv("QUANTA_MAX_SLOTS", "12")
    cfg = MultiScaleConfig(profile_path=empty_profile)
    assert cfg.server_max_parallel_slots == 12
    assert cfg.server_batch_workers == 12

    monkeypatch.setenv("QUANTA_BATCH_WORKERS", "4")
    cfg2 = MultiScaleConfig(profile_path=empty_profile)
    assert cfg2.server_batch_workers == 4
    assert cfg2.server_max_parallel_slots == 12


def test_multi_scale_config_slot_profile_loading(tmp_path: Path):
    """Verifies loading calibrated slot parameters from profile JSON."""
    profile_path = tmp_path / "multi_scale_profile.json"
    data = {
        "server.max_parallel_slots": {
            "value": 16,
            "source_exp": "exp-032a",
            "source_commit": "abcdef1",
            "calibrated_on": "musique/dev_paragraphs",
            "notes": "Optimal slot setting",
        },
        "server.batch_workers": {
            "value": 16,
            "source_exp": "exp-032a",
            "source_commit": "abcdef1",
            "calibrated_on": "musique/dev_paragraphs",
            "notes": "Optimal batch worker count",
        },
    }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    cfg = MultiScaleConfig(profile_path=profile_path)
    assert cfg.server_max_parallel_slots == 16
    assert cfg.server_batch_workers == 16
    assert cfg.is_calibrated("server.max_parallel_slots")
    assert cfg.is_calibrated("server.batch_workers")


def test_cognitive_pipeline_batch_workers_configuration():
    """Verifies CognitivePipeline batch workers initialization and dynamic adjustment."""
    pipe = CognitivePipeline(transducer_backend="mock_sexpr", max_workers=4)
    assert pipe.max_workers == 4

    pipe.set_batch_workers(8)
    assert pipe.max_workers == 8
    assert pipe.multi_scale_config.server_batch_workers == 8

    pipe.close()


def test_cognitive_pipeline_batch_ingestion_with_slots():
    """Verifies CognitivePipeline.ingest_passages_batch across mock workers."""
    passages = [
        ("P1", "D1", "Charles Babbage designed the Analytical Engine in London."),
        ("P2", "D1", "The University of Cambridge was founded in 1209."),
        ("P3", "D1", "Alan Turing worked at Bletchley Park during the war."),
    ]
    pipe = CognitivePipeline(
        transducer_backend="mock_sexpr",
        skeleton_format="sexpr_compact",
        page_table_path=":memory:",
        max_workers=2,
    )
    graphs = pipe.ingest_passages_batch(passages, max_workers=2, validate=False)
    assert len(graphs) == 3
    assert len(pipe.passage_store) == 3
    assert len(pipe.binary_table) > 0
    pipe.close()


def test_parse_slots_arg():
    """Verifies parsing of command-line slot strings."""
    assert parse_slots_arg("1,2,4,8,12,16") == [1, 2, 4, 8, 12, 16]
    assert parse_slots_arg(" 4 , 8 , 2 , 4 ") == [2, 4, 8]


def test_parallel_slots_sweep_mock_execution(tmp_path: Path):
    """Verifies end-to-end execution of benchmark_parallel_slots in mock mode."""
    out_file = tmp_path / "test_report.md"
    res = run_parallel_slot_sweep(
        slot_counts=[1, 2],
        corpus="paragraphs",
        mode="mock",
        skeleton_format="sexpr_compact",
        co_decoded=False,
        save_profile=False,
        output_path=out_file,
    )

    assert "optimal_slots" in res
    assert res["optimal_slots"] in (1, 2)
    assert len(res["results"]) == 2
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "QUANTA Parallel Slot Concurrency Optimization Report" in content
    assert "Concurrency Scaling Matrix" in content
