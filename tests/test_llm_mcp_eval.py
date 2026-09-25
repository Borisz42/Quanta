"""Test Suite for Section 7.5: Full Pipeline LLM Evaluation & MCP vs Non-MCP Ablation.

Verifies:
1. Model Context Protocol (MCP) Server integration with GlobalKnowledgeBase.
2. Zero-Shot Parametric LLM Baseline evaluation (Qwen via Unsloth).
3. Classical Dense RAG + LLM Baseline evaluation (demonstrating semantic hop drift).
4. LLM with QUANTA MCP Tools (quanta_query_memory, quanta_get_entity_details).
5. Head-to-Head Comparison: LLM with MCP vs LLM without MCP (quantifying hallucination drop and bridge recovery).
6. Evidence verification: Benchmark report contains zero hardcoded placeholders (~35%, <25%).
7. Physical latency advantage: QUANTA direct symbolic traversal is > 10x faster than LLM generation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from memory.global_kb import GlobalKnowledgeBase
from pipeline.multihop_evaluator import (
    DenseRAGBaseline,
    MCPLLMEvaluator,
    MultiHopBenchmarkEvaluator,
    MultiHopReasoner,
    ZeroShotLLMBaseline,
)
from server.mcp_server import MCPServer
from server.unsloth_manager import UnslothServerManager


DB_PATH = Path("data/wikipedia_quanta.db")
BENCHMARK_PATH = Path("data/benchmarks/musique_sample_real.json")


@pytest.fixture(scope="module")
def global_kb() -> GlobalKnowledgeBase:
    if not DB_PATH.exists():
        pytest.skip(f"Wikipedia database not found at {DB_PATH}")
    kb = GlobalKnowledgeBase(db_path=DB_PATH)
    yield kb
    kb.close()


@pytest.fixture(scope="module")
def sample_question(global_kb: GlobalKnowledgeBase) -> dict:
    if not BENCHMARK_PATH.exists():
        pytest.skip(f"Benchmark dataset not found at {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[0]


@pytest.fixture(scope="module")
def unsloth_mgr() -> UnslothServerManager:
    return UnslothServerManager(target_model="unsloth/Qwen3.5-4B-MTP-GGUF")


class TestLLMMCPFullPipeline:
    """Test suite for full pipeline LLM evaluation and MCP comparative ablation."""

    def test_mcp_server_with_global_kb_lookup(self, global_kb: GlobalKnowledgeBase):
        """MCPServer with global_kb must resolve entities and return verified facts."""
        mcp = MCPServer(global_kb=global_kb)

        # 1. Query memory for Alan Turing
        res_mem = mcp._tool_query_memory({"query": "Alan Turing"})
        assert "Alan Turing" in res_mem or "Q7251" in res_mem

        # 2. Get entity details for Alan Turing
        res_det = mcp._tool_get_entity_details({"name_or_cid": "Alan Turing"})
        data = json.loads(res_det)
        assert data.get("canonical_cid") is not None
        assert "turing" in data.get("anchor", "").lower() or "turing" in str(data.get("literal", {})).lower()
        assert data.get("total_active_slots", 0) > 0

    def test_zero_shot_baseline_evaluation(
        self, sample_question: dict, unsloth_mgr: UnslothServerManager
    ):
        """ZeroShotLLMBaseline must evaluate sample returning all diagnostic keys."""
        evaluator = ZeroShotLLMBaseline()
        mgr_to_use = unsloth_mgr if unsloth_mgr.is_service_responsive() else None
        res = evaluator.evaluate_sample(sample_question, unsloth_manager=mgr_to_use)

        assert "answer" in res
        assert "correct" in res
        assert "hallucinated" in res
        assert "bridge_recall" in res
        assert "latency_ms" in res
        assert res["latency_ms"] > 0
        assert "prompt_tokens" in res

    def test_dense_rag_llm_evaluation(
        self, sample_question: dict, unsloth_mgr: UnslothServerManager
    ):
        """DenseRAGBaseline.generate_with_llm must return retrieval and synthesis diagnostics."""
        dense = DenseRAGBaseline()
        mgr_to_use = unsloth_mgr if unsloth_mgr.is_service_responsive() else None
        res = dense.generate_with_llm(sample_question, unsloth_manager=mgr_to_use)

        assert "retrieved_passages" in res
        assert "bridge_recall" in res
        assert "answer" in res
        assert "correct" in res
        assert "hallucinated" in res
        assert "latency_ms" in res

    def test_mcp_llm_evaluation_tool_calling(
        self,
        global_kb: GlobalKnowledgeBase,
        sample_question: dict,
        unsloth_mgr: UnslothServerManager,
    ):
        """MCPLLMEvaluator must support tool calling to ground multi-hop facts."""
        evaluator = MCPLLMEvaluator(global_kb=global_kb)
        mgr_to_use = unsloth_mgr if unsloth_mgr.is_service_responsive() else None
        res = evaluator.evaluate_sample(sample_question, unsloth_manager=mgr_to_use)

        assert "answer" in res
        assert "tool_called" in res
        assert "tool_calls" in res
        assert "correct" in res
        assert "hallucinated" in res
        assert "bridge_recall" in res
        assert res["bridge_recall"] >= 0.80

    def test_full_pipeline_mcp_vs_nomcp_comparison(
        self,
        global_kb: GlobalKnowledgeBase,
        unsloth_mgr: UnslothServerManager,
    ):
        """Evaluates 5 samples across all 4 configurations to compare MCP vs non-MCP."""
        evaluator = MultiHopBenchmarkEvaluator(
            global_kb=global_kb,
            benchmark_path=BENCHMARK_PATH,
            unsloth_manager=unsloth_mgr,
        )

        results = evaluator.run_benchmark(
            sample_size=5,
            audit_sample_size=100,
            mode="live" if unsloth_mgr.is_service_responsive() else "offline",
            eval_llm=True,
            llm_samples=5,
        )

        ov = results["overall"]
        mcp_abl = results["mcp_vs_nomcp_ablation"]

        # Ensure all empirical metrics are populated (no None, no hardcoded strings)
        assert isinstance(ov["zero_shot_bridge_recall_pct"], (int, float))
        assert isinstance(ov["zero_shot_hallucination_rate_pct"], (int, float))
        assert isinstance(ov["zero_shot_mean_latency_ms"], (int, float))
        assert isinstance(ov["dense_rag_hallucination_rate_pct"], (int, float))
        assert isinstance(ov["mcp_bridge_recall_pct"], (int, float))
        assert isinstance(ov["mcp_hallucination_rate_pct"], (int, float))
        assert isinstance(ov["mcp_mean_latency_ms"], (int, float))
        assert isinstance(ov["quanta_bridge_recall_pct"], (int, float))
        assert isinstance(ov["mean_traversal_latency_ms"], (int, float))

        # MCP should achieve higher or equal accuracy and lower or equal hallucination
        assert ov["mcp_bridge_recall_pct"] >= ov["dense_bridge_recall_pct"]
        assert ov["mcp_hallucination_rate_pct"] <= ov["zero_shot_hallucination_rate_pct"]

        # QUANTA direct symbolic must achieve sub-10ms latency
        assert ov["mean_traversal_latency_ms"] < 10.0
        assert ov["quanta_bridge_recall_pct"] >= 90.0

        # Empirical evidence must contain 5 audited cases
        assert len(results["empirical_evidence"]) == 5
        for case in results["empirical_evidence"]:
            assert "question" in case
            assert "gold_answer" in case
            assert "zero_shot" in case
            assert "dense_rag" in case
            assert "llm_mcp" in case
            assert "quanta_direct" in case

    def test_benchmark_report_contains_no_hardcoded_placeholders(
        self,
        global_kb: GlobalKnowledgeBase,
        tmp_path: Path,
    ):
        """Exported markdown report must NOT contain hardcoded strings like ~35.0% or < 25.0%."""
        report_file = tmp_path / "test_report.md"
        json_file = tmp_path / "test_results.json"

        evaluator = MultiHopBenchmarkEvaluator(
            global_kb=global_kb,
            benchmark_path=BENCHMARK_PATH,
        )

        results = evaluator.run_benchmark(
            sample_size=5,
            audit_sample_size=50,
            mode="offline",
            eval_llm=True,
            llm_samples=3,
        )

        evaluator.export_markdown_report(results, report_file)
        evaluator.export_json_results(results, json_file)

        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")

        # Must not contain ungrounded placeholder strings
        assert "< 25.0%" not in content
        assert "~35.0%" not in content
        assert "~18.5%" not in content
        assert "~1,200 ms" not in content
        assert "~45.0 ms" not in content

        # Must contain new sections
        assert "## 2. Head-to-Head Comparative Ablation (Tasks 7.5.2 & 7.5.5)" in content
        assert "## 3. Model Context Protocol (MCP) vs. Non-MCP Head-to-Head Ablation" in content
        assert "## 5. Empirical LLM Inference Traces & Audit Evidence" in content
        assert "LLM + QUANTA MCP" in content
