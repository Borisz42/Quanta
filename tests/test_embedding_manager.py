"""Unit tests for Auxiliary Neural Vector Coprocessor (ANVC) Embedding Manager.

Verifies:
1. Task prefix formatting (asymmetric query, document, code).
2. Arbitrary Matryoshka truncation D in [64..768] with L2 normalization.
3. Multi-backend dispatch and MockEmbeddingManager deterministic behavior.
4. Cosine similarity and batch similarity computations.
5. MultiScaleConfig parameter resolution and runtime overrides.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.multi_scale_config import MultiScaleConfig
from server.embedding_manager import (
    EmbeddingManager,
    MockEmbeddingManager,
    format_code_prefix,
    format_document_prefix,
    format_query_prefix,
    truncate_and_normalize,
)


def test_task_prefix_formatters():
    """Verifies standard task prefix formatting rules."""
    # Query prefix
    q_pref = format_query_prefix("What caused the heatshield failure?")
    assert q_pref == "task: question answering | query: What caused the heatshield failure?"

    q_search = format_query_prefix("Apollo 13 telemetry", task="search result")
    assert q_search == "task: search result | query: Apollo 13 telemetry"

    # Document prefix
    doc_pref_with_title = format_document_prefix("The cryogenic tank exploded.", title="Section 4")
    assert doc_pref_with_title == "title: Section 4 | text: The cryogenic tank exploded."

    doc_pref_no_title = format_document_prefix("The cryogenic tank exploded.")
    assert doc_pref_no_title == "title: none | text: The cryogenic tank exploded."

    # Code prefix
    code_pref = format_code_prefix("def quicksort(arr): return arr")
    assert code_pref == "task: code retrieval | query: def quicksort(arr): return arr"


def test_matryoshka_truncation_and_l2_normalization():
    """Verifies truncation across all target dimensions and strict L2 normalization."""
    rng = np.random.RandomState(42)
    # Generate 5 random 768-D vectors
    raw_vecs = rng.randn(5, 768).astype(np.float32)

    for dim in [64, 128, 256, 512, 768]:
        norm_vecs = truncate_and_normalize(raw_vecs, dim)
        assert norm_vecs.shape == (5, dim)

        # Verify unit L2 norm for each row
        row_norms = np.linalg.norm(norm_vecs, axis=-1)
        np.testing.assert_allclose(row_norms, 1.0, atol=1e-5)

    # Test single 1D vector
    single_raw = rng.randn(768).astype(np.float32)
    norm_128 = truncate_and_normalize(single_raw, 128)
    assert norm_128.shape == (128,)
    np.testing.assert_allclose(np.linalg.norm(norm_128), 1.0, atol=1e-5)


def test_mock_embedding_manager_determinism():
    """Verifies that MockEmbeddingManager produces deterministic, repeatable vectors."""
    mgr = MockEmbeddingManager(matryoshka_dim=256)

    # Same query produces identical vector
    v1 = mgr.embed_query("Who invented the Analytical Engine?")
    v2 = mgr.embed_query("Who invented the Analytical Engine?")
    assert v1.shape == (256,)
    assert v2.shape == (256,)
    np.testing.assert_array_equal(v1, v2)
    assert mgr.compute_similarity(v1, v2) == pytest.approx(1.0, abs=1e-5)

    # Different text produces distinct vector
    v3 = mgr.embed_query("The quick brown fox jumps over the lazy dog.")
    assert not np.array_equal(v1, v3)
    sim = mgr.compute_similarity(v1, v3)
    assert -1.0 <= sim <= 1.0


def test_dynamic_matryoshka_override_per_call():
    """Verifies per-call Matryoshka dimension override on EmbeddingManager."""
    mgr = MockEmbeddingManager(matryoshka_dim=768)

    # Default dim = 768
    v_full = mgr.embed_query("test query")
    assert v_full.shape == (768,)

    # Per-call overrides
    for d in [64, 128, 256, 512]:
        v_d = mgr.embed_query("test query", matryoshka_dim=d)
        assert v_d.shape == (d,)
        np.testing.assert_allclose(np.linalg.norm(v_d), 1.0, atol=1e-5)


def test_embed_documents_and_batch_similarity():
    """Verifies batched document embedding and matrix similarity computation."""
    mgr = MockEmbeddingManager(matryoshka_dim=128)

    docs = [
        "Ada Lovelace was an English mathematician and writer.",
        "Charles Babbage designed the Analytical Engine.",
        "Quantum computing utilizes superposition and entanglement.",
    ]
    titles = ["Ada Lovelace", "Charles Babbage", "Quantum Computing"]

    doc_embeddings = mgr.embed_documents(docs, titles=titles)
    assert doc_embeddings.shape == (3, 128)

    query = "Who was Ada Lovelace?"
    q_vec = mgr.embed_query(query)
    assert q_vec.shape == (128,)

    # Batch similarity
    sims = mgr.batch_similarity(q_vec, doc_embeddings)
    assert sims.shape == (3,)
    assert all(-1.0 <= s <= 1.0 for s in sims)

    # Due to token-correlation in MockEmbeddingBackend, doc 0 should have the highest similarity
    best_idx = int(np.argmax(sims))
    assert best_idx == 0


def test_embed_code():
    """Verifies code embedding functionality."""
    mgr = MockEmbeddingManager(matryoshka_dim=64)
    code = "def binary_search(arr, target): ..."
    v_code = mgr.embed_code(code)
    assert v_code.shape == (64,)
    np.testing.assert_allclose(np.linalg.norm(v_code), 1.0, atol=1e-5)


def test_embedding_config_integration():
    """Verifies seamless integration with MultiScaleConfig and runtime overrides."""
    config = MultiScaleConfig(
        runtime_overrides={
            "embedding.matryoshka_dim": 256,
            "embedding.backend": "mock",
            "embedding.batch_size": 32,
        }
    )
    assert config.embedding_matryoshka_dim == 256
    assert config.embedding_backend == "mock"
    assert config.embedding_batch_size == 32

    mgr = EmbeddingManager(config=config)
    assert mgr.matryoshka_dim == 256
    assert mgr.active_backend_name == "mock"

    v = mgr.embed_document("Some text")
    assert v.shape == (256,)
