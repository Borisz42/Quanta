"""Unit and Contract Tests for Calibrated Parameter Registry (§1).

Verifies:
1. Pass-through default resolution with no profile loaded.
2. Precedence hierarchy: defaults -> profile JSON -> env vars -> runtime overrides.
3. Strict metadata enforcement: rejection of entries without source_exp.
4. Provenance report markdown generation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from config.multi_scale_config import (
    CalibratedParam,
    MultiScaleConfig,
    REGISTERED_PARAMS,
)


def test_uncalibrated_defaults_resolve_to_passthrough(tmp_path: Path):
    """With no profile loaded, every registered parameter resolves to pass-through."""
    empty_profile = tmp_path / "nonexistent.json"
    cfg = MultiScaleConfig(profile_path=empty_profile)

    assert cfg.query_extractor_strategy == "passthrough"
    assert cfg.filter_strategy == "passthrough"
    assert cfg.fast_path_mode == "passthrough"
    assert cfg.background_enabled is False
    assert cfg.chunker_macro_target_tokens is None
    assert cfg.chunker_micro_target_words is None
    assert cfg.filter_keep_threshold is None

    # Verify every registered parameter has an entry matching default_passthrough
    for name, (default_val, _) in REGISTERED_PARAMS.items():
        param = cfg.get_param(name)
        assert param is not None
        assert param.value == default_val
        assert param.default_passthrough == default_val
        assert not cfg.is_calibrated(name)


def test_profile_json_loading(tmp_path: Path):
    """Loads calibrated parameters from valid JSON profile."""
    profile_file = tmp_path / "multi_scale_profile.json"
    profile_data = {
        "fast_path.mode": {
            "value": "hot_transduce",
            "source_exp": "exp-029a",
            "source_commit": "a1b2c3d",
            "calibrated_on": "musique_long/dev",
            "notes": "Optimal Pareto frontier at 4k",
        },
        "chunker.macro_target_tokens": {
            "value": 1000,
            "source_exp": "exp-028a",
            "source_commit": "e5f6g7h",
            "calibrated_on": "niah/dev",
            "notes": "Minimizes mid-block miss",
        },
    }
    with open(profile_file, "w", encoding="utf-8") as f:
        json.dump(profile_data, f)

    cfg = MultiScaleConfig(profile_path=profile_file)

    assert cfg.fast_path_mode == "hot_transduce"
    assert cfg.chunker_macro_target_tokens == 1000
    # Other parameters remain uncalibrated pass-through
    assert cfg.filter_strategy == "passthrough"
    assert cfg.is_calibrated("fast_path.mode")
    assert cfg.is_calibrated("chunker.macro_target_tokens")
    assert not cfg.is_calibrated("filter.strategy")

    p = cfg.get_param("fast_path.mode")
    assert p.source_exp == "exp-029a"
    assert p.source_commit == "a1b2c3d"
    assert p.calibrated_on == "musique_long/dev"


def test_rejection_of_missing_source_exp(tmp_path: Path):
    """Rejects profile entries that lack source_exp."""
    profile_file = tmp_path / "invalid_profile.json"
    invalid_data = {
        "filter.strategy": {
            "value": "kev_macro",
            # missing source_exp!
        }
    }
    with open(profile_file, "w", encoding="utf-8") as f:
        json.dump(invalid_data, f)

    with pytest.raises(ValueError, match="missing required 'source_exp'"):
        MultiScaleConfig(profile_path=profile_file)


def test_write_calibrated_enforces_source_exp(tmp_path: Path):
    """write_calibrated raises ValueError if source_exp is missing or empty."""
    profile_file = tmp_path / "profile.json"

    with pytest.raises(ValueError, match="Cannot calibrate parameter"):
        MultiScaleConfig.write_calibrated(
            name="filter.strategy",
            value="lexical",
            source_exp="",
            profile_path=profile_file,
        )

    with pytest.raises(ValueError, match="Cannot calibrate parameter"):
        MultiScaleConfig.write_calibrated(
            name="filter.strategy",
            value="lexical",
            source_exp=None,  # type: ignore
            profile_path=profile_file,
        )

    # Valid write succeeds and round-trips
    MultiScaleConfig.write_calibrated(
        name="filter.strategy",
        value="lexical",
        source_exp="exp-028b",
        source_commit="commit123",
        calibrated_on="musique_long/dev",
        notes="Fast BM25 filter",
        profile_path=profile_file,
    )

    cfg = MultiScaleConfig(profile_path=profile_file)
    assert cfg.filter_strategy == "lexical"
    p = cfg.get_param("filter.strategy")
    assert p.source_exp == "exp-028b"
    assert p.source_commit == "commit123"


def test_precedence_hierarchy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies strict 4-tier precedence: default -> profile -> env -> runtime override."""
    profile_file = tmp_path / "profile.json"

    # Step 1: Base profile has fast_path.mode = hot_transduce
    MultiScaleConfig.write_calibrated(
        name="fast_path.mode",
        value="hot_transduce",
        source_exp="exp-029a",
        profile_path=profile_file,
    )
    MultiScaleConfig.write_calibrated(
        name="chunker.macro_target_tokens",
        value=800,
        source_exp="exp-028a",
        profile_path=profile_file,
    )

    # 1. Profile beats default
    cfg1 = MultiScaleConfig(profile_path=profile_file)
    assert cfg1.fast_path_mode == "hot_transduce"
    assert cfg1.chunker_macro_target_tokens == 800

    # 2. Env var beats profile
    monkeypatch.setenv("QUANTA_MS_FAST_PATH_MODE", "raw_only")
    monkeypatch.setenv("QUANTA_MS_CHUNKER_MACRO_TARGET_TOKENS", "1200")
    cfg2 = MultiScaleConfig(profile_path=profile_file)
    assert cfg2.fast_path_mode == "raw_only"
    assert cfg2.chunker_macro_target_tokens == 1200

    # 3. Runtime override beats env var
    cfg3 = cfg2.with_overrides({
        "fast_path.mode": "full",
        "chunker.macro_target_tokens": 1500,
    })
    assert cfg3.fast_path_mode == "full"
    assert cfg3.chunker_macro_target_tokens == 1500


def test_provenance_report_markdown(tmp_path: Path):
    """Outputs a valid Markdown table with parameter provenance."""
    profile_file = tmp_path / "profile.json"
    MultiScaleConfig.write_calibrated(
        name="query_extractor.strategy",
        value="qe_b",
        source_exp="exp-027a",
        source_commit="commitabc",
        calibrated_on="musique_long/dev",
        notes="Intent extraction",
        profile_path=profile_file,
    )

    cfg = MultiScaleConfig(profile_path=profile_file)
    report = cfg.provenance_report()

    assert "| Parameter | Active Value | Default (Pass-through) |" in report
    assert "`query_extractor.strategy`" in report
    assert "`qe_b`" in report
    assert "`exp-027a`" in report
    assert "`commitabc`" in report
