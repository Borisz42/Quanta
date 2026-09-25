"""Unit tests for PipelineExecutionTracer and Mermaid diagram generation."""

import json
from pathlib import Path
import pytest

from pipeline.tracer import PipelineExecutionTracer


class TestPipelineExecutionTracer:
    """Test suite for pipeline tracer, diagram emission, and export formats."""

    @pytest.fixture
    def tracer(self):
        t = PipelineExecutionTracer()
        t.reset()
        return t

    def test_record_all_stages(self, tracer: PipelineExecutionTracer):
        """Records events across all 7 pipeline stages plus ablation probe."""
        tracer.record_lexical_grounding("telescope", "hash_123", latency_us=2.5)
        tracer.record_interning("cid_456", is_hit=True, reuse_rate=75.0, total_nodes=100)
        tracer.record_pagetable_canvas("put_node", "cid_456", canvas_size=120, capacity=512)
        tracer.record_world_state("Order_1042", "STATUS_SLOT", "status:FULFILLED", t_start=17.0)
        tracer.record_spreading_activation("telescope optics", "Retrieved JWST Optics", latency_ms=1.8)
        tracer.record_reverse_proxy(raw_tokens=2000, compressed_tokens=400, latency_ms=12.0, turns_pruned=8)
        tracer.record_backend_call("POST", "http://127.0.0.1:8888/v1/chat/completions", status_code=200, latency_s=0.45)
        tracer.record_ablation_probe(
            probe_name="Private Token Probe",
            question="What is txn ref?",
            without_graph_response="I don't know",
            with_graph_response="txn_9941",
            grounding_verdict="PASS",
            evidence_token="txn_9941",
        )

        assert len(tracer.events) == 8
        assert len(tracer.ablation_results) == 1

    def test_mermaid_generation(self, tracer: PipelineExecutionTracer):
        """Validates that Mermaid diagrams render with expected syntax and tokens."""
        jwst_mermaid = tracer.generate_jwst_mermaid()
        assert "graph TD" in jwst_mermaid
        assert "NIRCam" in jwst_mermaid
        assert "WASP-96b" in jwst_mermaid

        saga_mermaid = tracer.generate_saga_mermaid()
        assert "stateDiagram-v2" in saga_mermaid
        assert "PENDING" in saga_mermaid
        assert "FULFILLED" in saga_mermaid
        assert "CANCELLED" in saga_mermaid

        flow_mermaid = tracer.generate_pipeline_dataflow_mermaid()
        assert "flowchart LR" in flow_mermaid
        assert "Mmap Lexical Grounder" in flow_mermaid
        assert "Unsloth GPU Server" in flow_mermaid

    def test_export_json_and_markdown(self, tracer: PipelineExecutionTracer, tmp_path: Path):
        """Tests file export functionality."""
        tracer.record_lexical_grounding("beryllium", "hash_ber", latency_us=1.9)
        tracer.record_ablation_probe(
            probe_name="Synthetic Alloy Probe",
            question="What alloy?",
            without_graph_response="Gold",
            with_graph_response="QUANTA-ALLOY-X99",
            grounding_verdict="PASS: 100% Fidelity",
            evidence_token="QUANTA-ALLOY-X99",
        )

        md_path = tmp_path / "test_trace.md"
        json_path = tmp_path / "test_trace.json"

        md_content = tracer.export_markdown(md_path)
        json_content_path = tracer.export_json(json_path)

        assert md_path.exists()
        assert json_path.exists()
        assert "## 1. End-to-End Pipeline Dataflow Architecture" in md_content
        assert "QUANTA-ALLOY-X99" in md_content

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_events"] == 2
        assert len(data["ablation_results"]) == 1

    def test_record_comparative_eval(self, tracer: PipelineExecutionTracer, tmp_path: Path):
        """Tests comparative baseline vs QUANTA recording and markdown output."""
        tracer.record_comparative_eval(
            task="JWST Exoplanet",
            query="What did NIRCam observe on WASP-96b?",
            baseline_prompt_tokens=420,
            quanta_prompt_tokens=85,
            baseline_latency_s=1.20,
            quanta_latency_s=0.28,
            baseline_tps=50.0,
            quanta_tps=55.0,
            baseline_answer="NIRCam observed water vapor absorption signatures.",
            quanta_answer="Water vapor absorption was observed on WASP-96b.",
            factual_token="water vapor",
            is_baseline_correct=True,
            is_quanta_correct=True,
            retrieval_latency_ms=1.45,
        )
        assert len(tracer.events) == 1
        assert len(tracer.comparative_results) == 1
        assert tracer.comparative_results[0]["token_savings_pct"] > 70.0

        md_path = tmp_path / "comp_trace.md"
        content = tracer.export_markdown(md_path)
        assert "## 4. Head-to-Head Comparative Evaluation: Baseline Raw Text vs. QUANTA Subgraph" in content
        assert "JWST Exoplanet" in content
        assert "water vapor" in content

