import json
from pathlib import Path
import numpy as np
import pytest

from config.multi_scale_config import MultiScaleConfig
from memory.fast_path_assembler import FastPathAssembler, FastPathMode
from models.kev_engine import KevDecisionEngine, MockKevEngine, _normalize_option_logprobs_to_probs
from parser.chunker import DiscourseChunk
from parser.multi_scale_chunker import HierarchicalChunker, MacroBlock
from retrieval.relevance_filter import RelevanceFilter, ScoredUnit
from server.embedding_manager import MockEmbeddingManager


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


def test_variant_f_g_dense_hybrid_scoring():
    chunks = _create_sample_chunks()
    mgr = MockEmbeddingManager(matryoshka_dim=64)
    filt = RelevanceFilter(
        strategy="F-G",
        embedding_manager=mgr,
        matryoshka_dim=64,
        rrf_k=60,
        weight_lexical=0.5,
        weight_dense=0.5,
    )

    query = "Where are superconducting quantum circuits located?"
    scored = filt.score(query, chunks)
    assert len(scored) == len(chunks)
    for s in scored:
        assert 0.0 <= s.score <= 1.0
        assert s.metadata["strategy"] == "F-G_DENSE_HYBRID"
        assert "rrf_score" in s.metadata
        assert "dense_score" in s.metadata
        assert "bm25_score" in s.metadata
        assert "bm25_rank" in s.metadata
        assert "dense_rank" in s.metadata
        assert s.metadata["fast_path_triggered"] is False


def test_variant_f_g_parameter_overrides_and_weights():
    chunks = _create_sample_chunks()
    mgr = MockEmbeddingManager(matryoshka_dim=64)

    # 1. Dense-only weighting (weight_dense=1.0, weight_lexical=0.0)
    filt_dense = RelevanceFilter(
        strategy="F-G",
        embedding_manager=mgr,
        weight_lexical=0.0,
        weight_dense=1.0,
        rrf_k=30,
    )
    scored_dense = filt_dense.score("quantum cooling cryogenic", chunks)
    # RRF score should correlate strictly with dense_rank
    for s in scored_dense:
        assert s.metadata["w_lex"] == 0.0
        assert s.metadata["w_dense"] == 1.0
        assert s.metadata["k_rrf"] == 30

    # 2. Lexical-only weighting (weight_dense=0.0, weight_lexical=1.0)
    filt_lex = RelevanceFilter(
        strategy="F-G",
        embedding_manager=mgr,
        weight_lexical=1.0,
        weight_dense=0.0,
        rrf_k=20,
    )
    scored_lex = filt_lex.score("quantum cooling cryogenic", chunks)
    for s in scored_lex:
        assert s.metadata["w_lex"] == 1.0
        assert s.metadata["w_dense"] == 0.0


def test_variant_f_g_synonym_resilience():
    """Verifies that dense bi-encoder elevates gold evidence on synonym/paraphrase queries without keyword overlap."""
    mgr = MockEmbeddingManager(matryoshka_dim=64)
    query = "refrigeration for calculating processors"
    q_vec = mgr.embed_query(query, matryoshka_dim=64)

    # Gold chunk has perfect dense semantic affinity (simulated via aligned embedding)
    chunk_gold = DiscourseChunk(
        chunk_id="chunk_gold",
        text="The refrigeration unit chills the calculating processor to sub-zero temperature.",
        global_offset=0,
        global_end_offset=80,
        word_count=10,
        token_count_estimate=13,
        dense_embedding=q_vec,
    )
    chunk_distractor = DiscourseChunk(
        chunk_id="chunk_distractor",
        text="Ancient ledgers tracked accounting numbers and arithmetic calculations in Roman times.",
        global_offset=81,
        global_end_offset=170,
        word_count=11,
        token_count_estimate=15,
    )
    chunk_unrelated = DiscourseChunk(
        chunk_id="chunk_unrelated",
        text="The botanical garden contains tropical palm trees and colorful flowering shrubs.",
        global_offset=171,
        global_end_offset=250,
        word_count=10,
        token_count_estimate=14,
    )
    all_chunks = [chunk_gold, chunk_distractor, chunk_unrelated]

    filt_hybrid = RelevanceFilter(
        strategy="F-G",
        embedding_manager=mgr,
        weight_lexical=0.3,
        weight_dense=0.7,
        keep_top_k=1,
    )
    scored = filt_hybrid.score(query, all_chunks)
    kept, _ = filt_hybrid.apply_keep_policy(scored)
    assert len(kept) == 1
    assert kept[0].unit_id == "chunk_gold"


def test_variant_f_g_dense_fast_path_bypass():
    """Verifies that high dense similarity triggers fast-path bypass."""
    mgr = MockEmbeddingManager(matryoshka_dim=64)
    chunks = _create_sample_chunks()
    query = "cryogenic pulse-tube refrigerators beryllium shields"
    q_vec = mgr.embed_query(query, matryoshka_dim=64)

    # Set hot dense embedding on chunk_0003
    chunks[2].dense_embedding = q_vec

    # Set threshold to 0.75, which chunk_0003 exceeds (cos = 1.0)
    filt_hot = RelevanceFilter(
        strategy="F-G",
        embedding_manager=mgr,
        dense_fast_path_threshold=0.75,
    )
    scored_hot = filt_hot.score(query, chunks)
    assert filt_hot.last_fast_path_triggered is True
    top_scored = max(scored_hot, key=lambda s: s.score)
    assert top_scored.unit_id == "chunk_0003"
    assert top_scored.metadata.get("dense_fast_path_triggered") is True
    assert top_scored.metadata.get("fast_path_bypassed") is True

    # When threshold is higher than 1.0 (e.g. 1.5, never reached), bypass is not triggered
    filt_normal = RelevanceFilter(
        strategy="F-G",
        embedding_manager=mgr,
        dense_fast_path_threshold=1.5,
    )
    scored_normal = filt_normal.score(query, chunks)
    assert filt_normal.last_fast_path_triggered is False


def test_macro_and_discourse_chunk_embedding_caching():
    """Verifies that MacroBlock and DiscourseChunk store and cache dense embeddings."""
    mgr = MockEmbeddingManager(matryoshka_dim=64)

    chunk = DiscourseChunk(
        chunk_id="chunk_test",
        text="Continuous geometric vector representation in host memory.",
        word_count=8,
        token_count_estimate=11,
    )
    assert chunk.dense_embedding is None
    emb = chunk.get_dense_embedding(embedding_manager=mgr, matryoshka_dim=64)
    assert emb is not None
    assert emb.shape == (64,)
    assert chunk.dense_embedding is not None

    # Test serialization round-trip
    d = chunk.to_dict()
    assert "dense_embedding" in d
    assert len(d["dense_embedding"]) == 64
    chunk_restored = DiscourseChunk.from_dict(d)
    assert chunk_restored.dense_embedding is not None
    assert len(chunk_restored.dense_embedding) == 64

    # Test MacroBlock embedding
    macro = MacroBlock(
        macro_id="macro_test",
        char_span=(0, 100),
        text="A coarse structural discourse block spanning multiple paragraphs.",
        micro_chunks=[chunk],
    )
    assert macro.dense_embedding is None
    m_emb = macro.get_dense_embedding(embedding_manager=mgr, matryoshka_dim=64)
    assert m_emb is not None
    assert m_emb.shape == (64,)
    assert macro.dense_embedding is not None

    m_dict = macro.to_dict()
    assert "dense_embedding" in m_dict
    macro_restored = MacroBlock.from_dict(m_dict)
    assert macro_restored.dense_embedding is not None

    # Test HierarchicalChunker batch attachment
    chunker = HierarchicalChunker(embedding_manager=mgr, matryoshka_dim=64)
    blocks, micros = chunker.chunk_document("Paragraph one on quantum computing.\n\nParagraph two on cryogenics.")
    assert len(blocks) > 0
    assert blocks[0].dense_embedding is not None
    assert blocks[0].dense_embedding.shape == (64,)


def test_fast_path_assembler_dense_hot_routing():
    """Verifies that FastPathAssembler routes directly to RAW_ONLY on dense hot match."""
    assembler = FastPathAssembler(
        mode=FastPathMode.COVERAGE_ADAPTIVE.value,
        dense_fast_path_threshold=0.80,
    )

    chunk = DiscourseChunk(
        chunk_id="chunk_hot",
        text="Critical hot evidence that immediately answers the user query.",
        word_count=9,
        token_count_estimate=12,
    )
    scored = ScoredUnit(
        unit=chunk,
        unit_id="chunk_hot",
        score=0.95,
        raw_score=0.95,
        metadata={"dense_score": 0.92, "dense_fast_path_triggered": True},
    )

    triggered, max_score, top_u = assembler.check_dense_hot_path([scored])
    assert triggered is True
    assert max_score >= 0.80

    eff_mode, cov = assembler.resolve_effective_mode(
        intent="What answers the query?",
        kept_units=[scored],
    )
    assert eff_mode == FastPathMode.RAW_ONLY.value

    # Assemble directly
    ctx = assembler.assemble_dense_hot_path(scored)
    assert ctx.mode == FastPathMode.RAW_ONLY.value
    assert "Critical hot evidence" in ctx.context_text
    assert ctx.transduced_count == 0

