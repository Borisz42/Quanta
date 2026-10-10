"""Unit tests for RelevanceFilter and Keep-Policy semantics (§2.2)."""

import json
from pathlib import Path
import pytest

from config.multi_scale_config import MultiScaleConfig
from models.kev_engine import KevDecisionEngine, MockKevEngine, _normalize_option_logprobs_to_probs
from parser.chunker import DiscourseChunk
from parser.multi_scale_chunker import MacroBlock
from retrieval.relevance_filter import RelevanceFilter, ScoredUnit


def _create_sample_chunks() -> list[DiscourseChunk]:
    return [
        DiscourseChunk(
            chunk_id="chunk_0001",
            text="The quantum circuit operates at cryogenic temperatures. Superconducting qubits maintain coherence.",
            global_offset=0,
            global_end_offset=102,
            word_count=12,
            token_count_estimate=16,
        ),
        DiscourseChunk(
            chunk_id="chunk_0002",
            text="Ancient Roman aqueducts transported mountain spring water across long stone archways into public baths.",
            global_offset=103,
            global_end_offset=210,
            word_count=14,
            token_count_estimate=19,
        ),
        DiscourseChunk(
            chunk_id="chunk_0003",
            text="Cryogenic pulse-tube refrigerators cool the beryllium shields of quantum processors down to 4 Kelvin.",
            global_offset=211,
            global_end_offset=316,
            word_count=14,
            token_count_estimate=19,
        ),
    ]


def test_passthrough_mode_keeps_everything():
    chunks = _create_sample_chunks()
    filter_engine = RelevanceFilter(strategy="passthrough")

    scored = filter_engine.score("What cools the quantum processor?", chunks)
    assert len(scored) == len(chunks)
    assert all(s.score == 1.0 for s in scored)

    kept, discarded = filter_engine.apply_keep_policy(scored)
    assert len(kept) == len(chunks)
    assert len(discarded) == 0


def test_keep_policy_threshold():
    chunks = _create_sample_chunks()
    filter_engine = RelevanceFilter(
        strategy="F-A",
        keep_threshold=0.5,
    )

    scored = filter_engine.score("cryogenic quantum processors", chunks)
    kept, discarded = filter_engine.apply_keep_policy(scored)

    # chunks 1 and 3 mention cryogenic / quantum, chunk 2 is Roman aqueducts
    kept_ids = {u.unit_id for u in kept}
    assert "chunk_0001" in kept_ids or "chunk_0003" in kept_ids
    assert "chunk_0002" in {u.unit_id for u in discarded}


def test_keep_policy_top_k():
    chunks = _create_sample_chunks()
    filter_engine = RelevanceFilter(
        strategy="F-A",
        keep_top_k=1,
    )

    scored = filter_engine.score("superconducting qubits and cryogenics", chunks)
    kept, discarded = filter_engine.apply_keep_policy(scored)

    assert len(kept) == 1
    assert len(discarded) == 2
    assert kept[0].unit_id in ("chunk_0001", "chunk_0003")


def test_keep_policy_token_budget():
    chunks = _create_sample_chunks()
    # Each chunk is ~16-19 tokens. With a budget of 25 tokens, only 1 chunk can fit.
    filter_engine = RelevanceFilter(
        strategy="F-A",
        keep_budget_tokens=25,
    )

    scored = filter_engine.score("quantum coherence", chunks)
    kept, discarded = filter_engine.apply_keep_policy(scored)

    total_tokens = sum(u.tokens for u in kept)
    assert len(kept) == 1
    assert total_tokens <= 25


def test_variants_scoring_execution():
    chunks = _create_sample_chunks()
    query = "Where are quantum processors cooled?"

    # Test F-A (BM25)
    f_a = RelevanceFilter(strategy="F-A")
    s_a = f_a.score(query, chunks)
    assert len(s_a) == 3
    assert f_a.last_gpu_calls == 0

    # Test F-B (Concept)
    f_b = RelevanceFilter(strategy="F-B")
    s_b = f_b.score(query, chunks)
    assert len(s_b) == 3
    assert f_b.last_gpu_calls == 0

    # Test F-C (Kev Micro)
    f_c = RelevanceFilter(strategy="F-C", kev_engine=MockKevEngine())
    s_c = f_c.score(query, chunks)
    assert len(s_c) == 3
    assert f_c.last_gpu_calls == 3

    # Test Macro variants F-D and F-E
    macro = MacroBlock(
        macro_id="macro_0001",
        char_span=(0, 200),
        text="First sentence describes quantum cooling. Second sentence describes Roman history.",
        micro_chunks=chunks[:2],
        concept_codes=[10, 20],
        surface_entities=["Quantum Cooling"],
        token_count_estimate=30,
    )
    f_d = RelevanceFilter(strategy="F-D", kev_engine=MockKevEngine())
    s_d = f_d.score(query, [macro])
    assert len(s_d) == 1

    f_e = RelevanceFilter(strategy="F-E", kev_engine=MockKevEngine())
    s_e = f_e.score(query, [macro])
    assert len(s_e) == 1

    # Test F-F (Cascade)
    f_f = RelevanceFilter(strategy="F-F", kev_engine=MockKevEngine())
    s_f = f_f.score(query, chunks)
    assert len(s_f) == 3


def test_calibration_round_trips_through_profile(tmp_path: Path):
    profile_file = tmp_path / "multi_scale_profile.json"
    calib_params = {"method": "platt", "a": 1.45, "b": -0.25}

    MultiScaleConfig.write_calibrated(
        name="filter.calibration",
        value=calib_params,
        source_exp="exp-028a",
        source_commit="test_commit",
        notes="Calibrated Platt scaling for F-C",
        profile_path=profile_file,
    )

    loaded_cfg = MultiScaleConfig(profile_path=profile_file)
    assert loaded_cfg.get("filter.calibration") == calib_params

    filter_engine = RelevanceFilter(
        strategy="F-C",
        calibration=loaded_cfg.get("filter.calibration"),
        kev_engine=MockKevEngine(),
    )
    # Raw prob 0.8 should be mapped by Platt scaling
    calibrated_val = filter_engine._apply_calibration(prob=0.8, raw_score=0.0)
    assert 0.0 <= calibrated_val <= 1.0


def test_logprob_parsing_on_simulated_n_probs_response():
    # Simulated llama-server /completion response with completion_probabilities
    simulated_resp = {
        "content": "A",
        "completion_probabilities": [
            {
                "probs": [
                    {"tok_str": "A", "prob": 0.88, "logprob": -0.1278},
                    {"tok_str": "B", "prob": 0.12, "logprob": -2.1202},
                ]
            }
        ],
    }

    engine = KevDecisionEngine(fallback_to_mock=True)
    lps_dict = engine._parse_logprobs_from_response(simulated_resp)
    assert "A" in lps_dict
    assert "B" in lps_dict
    assert abs(lps_dict["A"] - (-0.1278)) < 1e-3

    # Normalize option letters
    top_label, top_prob, label_probs, option_raw = _normalize_option_logprobs_to_probs(
        token_logprobs=lps_dict,
        target_labels=["RELEVANT", "IRRELEVANT"],
    )
    assert top_label == "RELEVANT"
    assert top_prob > 0.80
    assert "RELEVANT" in label_probs
    assert "IRRELEVANT" in label_probs
    assert option_raw["A"] == lps_dict["A"]
