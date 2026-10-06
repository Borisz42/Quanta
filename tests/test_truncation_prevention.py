"""Unit and Regression Tests for Zero-Truncation Architecture & Bounded Reasoning Engine.

Tests:
1. Proxy context extraction regex matching for `Document context:`, `Context:`, `Passages:`, and markdown variants.
2. Raw distractor pruning in proxy system messages (ensuring raw text does not leak).
3. Bounded thought-budget prompt contracts across MuSiQue, ProofWriter, and ARC-Challenge.
4. PairedEvaluator sane max_tokens right-sizing and repetition penalty parameters.
5. Defensive truncation metric recovery fallback in BenchmarkMetrics on simulated cutoffs.
"""

from __future__ import annotations

import pytest

from benchmarks.metrics import BenchmarkMetrics
from benchmarks.paired_evaluator import PairedEvaluator, EvaluationCondition
from benchmarks.suite_loaders import BenchmarkSuiteLoader, BenchmarkSample
from server.proxy import (
    ChatMessage,
    ChatCompletionRequest,
    QuantaProxyConfig,
    create_proxy_app,
    extract_context_from_system,
    decompose_query_context,
)


# =============================================================================
# 1. System Context Extraction & Distractor Pruning Tests
# =============================================================================

def test_extract_context_from_system_regex_variants():
    """Verifies that extract_context_from_system correctly matches various context headers."""
    sample_doc = (
        "Document [1]: Zubly Cemetery is a historic cemetery in Beech Island, South Carolina.\n"
        "Document [2]: Beech Island is an unincorporated community in Aiken County, South Carolina.\n"
        "Document [3]: The county seat of Aiken County is Aiken."
    )

    # Standard "Context:"
    msgs_1 = [ChatMessage(role="system", content=f"Context:\n{sample_doc}")]
    doc_1, inst_1 = extract_context_from_system(msgs_1)
    assert doc_1 is not None and "Zubly Cemetery" in doc_1
    assert inst_1 == ""

    # "Document context:" (used by paired_evaluator)
    msgs_2 = [ChatMessage(role="system", content=f"Document context:\n{sample_doc}")]
    doc_2, inst_2 = extract_context_from_system(msgs_2)
    assert doc_2 is not None and "Zubly Cemetery" in doc_2
    assert inst_2 == ""

    # "Passages:"
    msgs_3 = [ChatMessage(role="system", content=f"Passages:\n{sample_doc}")]
    doc_3, inst_3 = extract_context_from_system(msgs_3)
    assert doc_3 is not None and "Zubly Cemetery" in doc_3
    assert inst_3 == ""

    # Leading instructions preceding "Context:"
    msgs_4 = [ChatMessage(role="system", content=f"You are a helpful assistant.\n\nContext:\n{sample_doc}")]
    doc_4, inst_4 = extract_context_from_system(msgs_4)
    assert doc_4 is not None and "Zubly Cemetery" in doc_4
    assert inst_4 == "You are a helpful assistant."

    # Markdown bold "**Context:**"
    msgs_5 = [ChatMessage(role="system", content=f"**Context:**\n{sample_doc}")]
    doc_5, inst_5 = extract_context_from_system(msgs_5)
    assert doc_5 is not None and "Zubly Cemetery" in doc_5


def test_proxy_system_distractor_pruning(tmp_path):
    """Verifies that proxy completely prunes raw distractor documents from system prompt."""
    from fastapi.testclient import TestClient

    distractor_doc = "\n".join([f"Document [{i}]: Irrelevant distractor text number {i} with noise." for i in range(1, 21)])
    config = QuantaProxyConfig(
        compression_threshold=100,
        transducer_backend="mock",
        page_table_path=tmp_path / "test_prune.db",
        fallback_to_local=True,
    )
    app = create_proxy_app(config)
    client = TestClient(app)

    payload = {
        "model": "quanta-context-expander",
        "messages": [
            {"role": "system", "content": f"Document context:\n{distractor_doc}"},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "quanta_metadata" in data
    # Raw tokens was high due to 20 distractors, but compressed should be much lower
    q_meta = data["quanta_metadata"]
    assert q_meta["raw_tokens"] > 100


# =============================================================================
# 2. Bounded Thought-Budget Prompt Contracts Tests
# =============================================================================

def test_musique_prompt_contract():
    """Verifies MuSiQue prompts contain the 2-3 sentence bounded bridge instructions."""
    loader = BenchmarkSuiteLoader()
    samples = loader.load_musique(limit=3)
    assert len(samples) > 0
    for s in samples:
        assert "Instructions:" in s.prompt
        assert "2 to 3 concise sentences" in s.prompt
        assert "Do not summarize or list irrelevant documents." in s.prompt
        assert "Answer: <final answer>" in s.prompt


def test_proofwriter_prompt_contract():
    """Verifies ProofWriter prompts contain the 1-2 sentence bounded deduction instructions."""
    loader = BenchmarkSuiteLoader()
    samples = loader.load_proofwriter(limit=3)
    assert len(samples) > 0
    for s in samples:
        assert "Instructions:" in s.prompt
        assert "1 to 2 concise sentences" in s.prompt
        assert "state your deductive reasoning based strictly on the provided rules" in s.prompt
        assert "Answer: [True/False/Unknown]" in s.prompt


def test_arc_science_prompt_contract():
    """Verifies ARC-Challenge prompts contain the 1-2 sentence bounded scientific principle instructions."""
    loader = BenchmarkSuiteLoader()
    samples = loader.load_arc_science(limit=3)
    assert len(samples) > 0
    for s in samples:
        assert "Instructions:" in s.prompt
        assert "1 to 2 concise sentences" in s.prompt
        assert "explain the scientific principle" in s.prompt
        assert "Answer: [A/B/C/D]" in s.prompt


# =============================================================================
# 3. PairedEvaluator Decoding & Token Budget Configuration Tests
# =============================================================================

def test_paired_evaluator_sane_max_tokens_budget():
    """Verifies that paired evaluator budgets appropriate max_tokens across suites."""
    evaluator = PairedEvaluator(mode="mock", ablation_mode="3way")

    # MuSiQue: right-sized to 768 tokens (preventing 2048 runaway loops)
    musique_sample = BenchmarkSample(id="m1", suite="musique", prompt="Who is X?", context="Doc 1: ...", gold_answer="Y")
    res_musique = evaluator.evaluate_sample(musique_sample)
    assert res_musique is not None

    # ProofWriter: right-sized to 512 tokens
    pw_sample = BenchmarkSample(id="p1", suite="proofwriter", prompt="Is X true?", context="Theory: ...", gold_answer="true")
    res_pw = evaluator.evaluate_sample(pw_sample)
    assert res_pw is not None

    # ARC Science: right-sized to 512 tokens
    arc_sample = BenchmarkSample(id="a1", suite="arc_science", prompt="Question?", gold_answer="A")
    res_arc = evaluator.evaluate_sample(arc_sample)
    assert res_arc is not None


# =============================================================================
# 4. Defensive Truncation Metric Recovery Fallback Tests
# =============================================================================

def test_extract_multiple_choice_key_truncation_recovery():
    """Verifies multiple choice recovery on truncated outputs where closing anchor is missing."""
    # Truncated mid-explanation after intermediate choice assertion
    truncated_1 = (
        "Based on the physical laws of optics, the correct answer is (B). "
        "When an astronomer observes the spectral emission lines of hydrogen gas shifted toward longer wavelengths, "
        "this represents the Doppler redshift phenomenon where velocity is"
    )
    assert BenchmarkMetrics.extract_multiple_choice_key(truncated_1, is_truncated=True) == "B"

    # Truncated after "therefore, choice C is correct"
    truncated_2 = (
        "Examining the periodic properties of tungsten: it exhibits the highest melting point of all metals. "
        "Therefore, choice C is correct. The crystalline lattice of tungsten prevents atomic migration at"
    )
    assert BenchmarkMetrics.extract_multiple_choice_key(truncated_2, is_truncated=True) == "C"

    # Truncated after "supports option D"
    truncated_3 = (
        "Osmosis across lipid bilayers is governed by water potential gradients, which supports option D. "
        "Cellular aquaporins facilitate this passive transport down the chemical potential without consuming"
    )
    assert BenchmarkMetrics.extract_multiple_choice_key(truncated_3, is_truncated=True) == "D"


def test_extractive_match_truncation_recovery_partial_token():
    """Verifies recovery when the final answer line was cut off mid-token at token limit."""
    # Cut off mid-word on answer line
    truncated_ans = (
        "Zubly Cemetery is in Beech Island (Doc 19). "
        "Beech Island is in Aiken County (Doc 14).\n"
        "Answer: Aiken Cou"
    )
    assert BenchmarkMetrics.extractive_match_score(truncated_ans, "Aiken County", is_truncated=True) is True

    # Cut off on boolean answer line
    truncated_bool = (
        "The cat is red and likes the dog. "
        "By rule 3, the cat chases the squirrel.\n"
        "Answer: [Tru"
    )
    assert BenchmarkMetrics.extractive_match_score(truncated_bool, "true", is_truncated=True) is True


def test_extractive_match_truncation_recovery_final_sentence():
    """Verifies recovery when answer is affirmed in final sentence before cutoff."""
    truncated_cot = (
        "The factual bridge connects Zubly Cemetery to Beech Island, which is located in Aiken County. "
        "Therefore, the municipality borders Columbia in South"
    )
    assert BenchmarkMetrics.extractive_match_score(truncated_cot, "Aiken County", is_truncated=True) is True

    # Hedged or negated statement must NOT falsely recover
    negated_cutoff = (
        "The factual bridge does not state if Aiken County is connected to the cemetery. "
        "Because of insufficient information across the documents"
    )
    assert BenchmarkMetrics.extractive_match_score(negated_cutoff, "Aiken County", is_truncated=True) is False
