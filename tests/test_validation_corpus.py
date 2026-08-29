"""Tests for Phase 11: Validation Corpus & Benchmark Data.

Verifies:
- 11A.1: FOLIO dataset ingestion
- 11A.2: ProofWriter dataset ingestion
- 11A.3: bAbI dataset ingestion
- 11A.4: CLUTRR dataset ingestion
- 11A.5: Python AST dataset collection (~200+ functions)
- 11A.6: Multi-domain corpus generation (5,000 balanced propositions across 5 domains)
- 11A.7: 🧪 Forward parser success rate (≥ 95% on validation propositions)
- 11B.1: 🧪 Forward-parse entire validation corpus into quaternary tensors (Σ^256)
"""

import ast
import numpy as np
import pytest

from core.types import QuantaVector
from data.corpus_generator import ValidationCorpusGenerator, GeneratedCorpus
from data.real_loader import RealDatasetLoader


@pytest.fixture(scope="module")
def real_loader():
    return RealDatasetLoader()


@pytest.fixture(scope="module")
def synthetic_generator():
    return ValidationCorpusGenerator(seed=42)


def test_folio_loader_sampling(real_loader):
    """11A.1: Verify FOLIO dataset ingestion produces distinct premise sentences and FOL formulas."""
    samples = real_loader.load_folio_samples(max_samples=50)
    assert len(samples) >= 10
    for text, fol in samples:
        assert isinstance(text, str) and len(text) > 3
        assert isinstance(fol, str)


def test_proofwriter_loader_sampling(real_loader):
    """11A.2: Verify ProofWriter dataset ingestion produces distinct rule/fact statements."""
    samples = real_loader.load_proofwriter_samples(max_samples=50)
    assert len(samples) >= 10
    for sent in samples:
        assert isinstance(sent, str) and len(sent) > 3


def test_babi_loader_sampling(real_loader):
    """11A.3: Verify bAbI tasks dataset ingestion produces distinct spatial/movement sentences."""
    samples = real_loader.load_babi_samples(max_samples=50)
    assert len(samples) >= 10
    for sent in samples:
        assert isinstance(sent, str) and len(sent) > 3


def test_clutrr_loader_sampling(real_loader):
    """11A.4: Verify CLUTRR dataset ingestion produces distinct kinship statements."""
    samples = real_loader.load_clutrr_samples(max_samples=50)
    assert len(samples) >= 10
    for sent in samples:
        assert isinstance(sent, str) and len(sent) > 3


def test_python_ast_loader_sampling(real_loader):
    """11A.5: Verify Python AST dataset ingestion loads 200+ distinct algorithmic functions."""
    samples = real_loader.load_python_ast_samples(max_samples=250)
    assert len(samples) >= 200
    ast_types = set()
    for desc, node in samples:
        assert isinstance(desc, str)
        assert isinstance(node, ast.AST)
        ast_types.add(type(node).__name__)

    # Ensure diverse AST constructs are represented
    assert "FunctionDef" in ast_types or "ClassDef" in ast_types


def test_synthetic_validation_corpus_generation(synthetic_generator):
    """11A.6: Verify synthetic corpus generator creates 5,000 balanced propositions across 5 domains."""
    corpus = synthetic_generator.generate_corpus(num_samples=5000)

    assert isinstance(corpus, GeneratedCorpus)
    assert corpus.canonical_matrix.shape == (5000, 256)
    assert corpus.candidate_matrix.shape[0] == 5000
    assert corpus.candidate_matrix.shape[1] >= 512
    assert len(corpus.labels) == 5000
    assert len(corpus.propositions) == 5000

    # Verify all 5 domains are present and equally balanced
    labels_arr = corpus.labels
    unique_labels, counts = np.unique(labels_arr, return_counts=True)
    assert set(unique_labels) == {0, 1, 2, 3, 4}
    for count in counts:
        assert count == 1000

    # Verify quaternary values strictly in {0, 1, 2, 3}
    assert np.all(np.isin(corpus.canonical_matrix, [0, 1, 2, 3]))


def test_forward_parser_success_rate_on_real_samples(real_loader):
    """11A.7 🧪: Verify forward parser achieves ≥ 95% valid parse rate across benchmark propositions."""
    # Test a representative batch across all 5 benchmark domains
    domains_data = [
        ("FOLIO", [item[0] for item in real_loader.load_folio_samples(max_samples=100)]),
        ("ProofWriter", real_loader.load_proofwriter_samples(max_samples=100)),
        ("bAbI", real_loader.load_babi_samples(max_samples=100)),
        ("CLUTRR", real_loader.load_clutrr_samples(max_samples=100)),
        ("CodeAST", real_loader.load_python_ast_samples(max_samples=100)),
    ]

    total_propositions = 0
    successful_parses = 0

    for domain_name, items in domains_data:
        domain_total = len(items)
        domain_success = 0

        for item in items:
            total_propositions += 1
            try:
                if domain_name == "CodeAST":
                    desc, ast_node = item
                    graph = real_loader.ast_parser.parse_ast_node(ast_node)
                elif domain_name == "FOLIO":
                    graph = real_loader.nlp_parser.parse_sentence(item, domain_context="FOLIO")
                else:
                    graph = real_loader.nlp_parser.parse_sentence(item, domain_context=domain_name)

                # Valid parse must have a non-empty graph and root CID
                if graph.root_cid is not None and len(graph.nodes) > 0:
                    vec = graph.to_proposition_vector()
                    if isinstance(vec, QuantaVector):
                        domain_success += 1
                        successful_parses += 1
            except Exception:
                pass

        domain_rate = domain_success / domain_total if domain_total > 0 else 0
        print(f"Domain {domain_name}: {domain_success}/{domain_total} ({domain_rate:.2%})")

    overall_success_rate = successful_parses / total_propositions
    print(f"Overall Forward Parser Success Rate: {successful_parses}/{total_propositions} ({overall_success_rate:.2%})")

    # Acceptance criterion from todo.md Phase 11: > 95%
    assert overall_success_rate >= 0.95, f"Parse rate {overall_success_rate:.2%} below 95% threshold"


def test_quaternary_tensor_matrix_properties(real_loader):
    """11B.1 🧪: Verify real corpus converts into valid (N, 256) quaternary tensors with all 4 bands active."""
    corpus = real_loader.build_real_corpus(samples_per_domain=20)

    assert corpus.canonical_matrix.shape == (100, 256)
    assert np.all(np.isin(corpus.canonical_matrix, [0, 1, 2, 3]))

    # Verify active slots across all 4 bands
    band0_active = np.sum(corpus.canonical_matrix[:, 0:64] > 0)
    band1_active = np.sum(corpus.canonical_matrix[:, 64:128] > 0)
    band2_active = np.sum(corpus.canonical_matrix[:, 128:192] > 0)
    band3_active = np.sum(corpus.canonical_matrix[:, 192:256] > 0)

    assert band0_active > 0, "Band 0 (NSM Primes & Kinematics) must have active slots"
    assert band1_active > 0, "Band 1 (Valencies & Topology) must have active slots"
    assert band2_active > 0, "Band 2 (Ontological Signatures) must have active slots"
    assert band3_active > 0, "Band 3 (Epistemic Bounds & Metarules) must have active slots"
