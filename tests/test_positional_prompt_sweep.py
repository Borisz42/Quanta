"""Unit tests for QUANTA Positional S-Expression System Prompt Sweep (exp-034a).

Verifies:
1. All 5 prompt variants (P0-P4) compile and format cleanly in POSITIONAL_PROMPT_VARIANTS.
2. SkeletonTransducer and MockSkeletonTransducer accept prompt_variant parameter and resolve system_prompt correctly.
3. CognitivePipeline forwards prompt_variant to SkeletonTransducer during initialization and ingestion.
4. Benchmark sweep runner executes cleanly in mock mode producing all required metrics.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from parser.skeleton_transducer import (
    CO_DECODED_POSITIONAL_ENHANCED_PROMPT,
    DEFAULT_POSITIONAL_SEXPR_SYSTEM_PROMPT,
    POSITIONAL_BASELINE_TERSE_PROMPT,
    POSITIONAL_JURISDICTION_HIERARCHY_PROMPT,
    POSITIONAL_PROMPT_VARIANTS,
    POSITIONAL_RELATIONAL_GUIDANCE_PROMPT,
    POSITIONAL_SCHEMA_EXPLICIT_PROMPT,
    MockSkeletonTransducer,
    SkeletonTransducer,
)
from pipeline.cognitive_pipeline import CognitivePipeline
from scripts.benchmark_positional_prompt_sweep import run_prompt_sweep


class TestPositionalPromptVariants:
    """Verifies prompt compilation and dictionary registration."""

    def test_prompt_dictionary_registration(self):
        assert "p0" in POSITIONAL_PROMPT_VARIANTS
        assert "p1" in POSITIONAL_PROMPT_VARIANTS
        assert "p2" in POSITIONAL_PROMPT_VARIANTS
        assert "p3" in POSITIONAL_PROMPT_VARIANTS
        assert "p4" in POSITIONAL_PROMPT_VARIANTS

        assert POSITIONAL_PROMPT_VARIANTS["p0"] == DEFAULT_POSITIONAL_SEXPR_SYSTEM_PROMPT
        assert POSITIONAL_PROMPT_VARIANTS["p1"] == POSITIONAL_SCHEMA_EXPLICIT_PROMPT
        assert POSITIONAL_PROMPT_VARIANTS["p2"] == POSITIONAL_RELATIONAL_GUIDANCE_PROMPT
        assert POSITIONAL_PROMPT_VARIANTS["p3"] == POSITIONAL_JURISDICTION_HIERARCHY_PROMPT
        assert POSITIONAL_PROMPT_VARIANTS["p4"] == CO_DECODED_POSITIONAL_ENHANCED_PROMPT

    def test_prompt_content_validity(self):
        for key, prompt in POSITIONAL_PROMPT_VARIANTS.items():
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 30
            assert "QUANTA" in prompt
            assert "Positional" in prompt or "S-expression" in prompt or "S-Expression" in prompt


class TestSkeletonTransducerPromptResolution:
    """Verifies that SkeletonTransducer and MockSkeletonTransducer resolve prompt variants."""

    def test_mock_transducer_variant_resolution(self):
        t0 = MockSkeletonTransducer(skeleton_format="sexpr_positional", prompt_variant="p0")
        assert t0.system_prompt == POSITIONAL_BASELINE_TERSE_PROMPT

        t1 = MockSkeletonTransducer(skeleton_format="sexpr_positional", prompt_variant="p1")
        assert t1.system_prompt == POSITIONAL_SCHEMA_EXPLICIT_PROMPT

        t2 = MockSkeletonTransducer(skeleton_format="sexpr_positional", prompt_variant="p2")
        assert t2.system_prompt == POSITIONAL_RELATIONAL_GUIDANCE_PROMPT

        t3 = MockSkeletonTransducer(skeleton_format="sexpr_positional", prompt_variant="p3")
        assert t3.system_prompt == POSITIONAL_JURISDICTION_HIERARCHY_PROMPT

        t4 = MockSkeletonTransducer(skeleton_format="sexpr_positional", prompt_variant="p4")
        assert t4.system_prompt == CO_DECODED_POSITIONAL_ENHANCED_PROMPT

    def test_skeleton_transducer_variant_resolution(self):
        st2 = SkeletonTransducer(
            fallback_to_mock=True,
            skeleton_format="sexpr_positional",
            prompt_variant="p2",
        )
        assert st2.prompt_variant == "p2"
        assert st2.system_prompt == POSITIONAL_RELATIONAL_GUIDANCE_PROMPT

        st3 = SkeletonTransducer(
            fallback_to_mock=True,
            skeleton_format="sexpr_positional",
            prompt_variant="p3",
        )
        assert st3.prompt_variant == "p3"
        assert st3.system_prompt == POSITIONAL_JURISDICTION_HIERARCHY_PROMPT

    def test_custom_system_prompt_overrides_variant(self):
        custom = "Custom system prompt override."
        st = SkeletonTransducer(
            fallback_to_mock=True,
            skeleton_format="sexpr_positional",
            system_prompt=custom,
            prompt_variant="p2",
        )
        assert st.system_prompt == custom


class TestCognitivePipelinePromptVariantIntegration:
    """Verifies CognitivePipeline integration with prompt variants."""

    def test_pipeline_initialization_with_variant(self):
        pipe = CognitivePipeline(
            transducer_backend="mock",
            skeleton_format="sexpr_positional",
            prompt_variant="p2",
            page_table_path=":memory:",
            max_workers=1,
        )
        assert pipe.prompt_variant == "p2"
        assert pipe.skeleton_transducer.system_prompt == POSITIONAL_RELATIONAL_GUIDANCE_PROMPT
        pipe.close()

    def test_pipeline_ingest_document_with_variant(self):
        pipe = CognitivePipeline(
            transducer_backend="mock",
            skeleton_format="sexpr_positional",
            prompt_variant="p3",
            page_table_path=":memory:",
            max_workers=1,
        )
        graph = pipe.ingest_document(
            text="Alan Turing studied at Cambridge in the United Kingdom.",
            doc_id="doc_turing",
            passage_id="P_turing_01",
            validate=False,
        )
        assert len(graph) > 0
        sk = getattr(graph, "skeleton_result", None)
        assert sk is not None
        assert sk.metadata.get("prompt_variant") == "p3"
        pipe.close()


class TestBenchmarkPromptSweepRunner:
    """Verifies that benchmark runner executes cleanly in mock mode."""

    def test_mock_prompt_sweep_execution(self, tmp_path):
        out_file = tmp_path / "mock_sweep_benchmark.md"
        res = run_prompt_sweep(
            backend_mode="mock",
            slots=2,
            output_path=str(out_file),
            target_prompts=["p0", "p1", "p2", "p3", "p4"],
            corpus="paragraphs",
            num_samples=2,
        )
        assert res["active_backend"] == "mock"
        results = res["results"]
        assert len(results) == 5

        # Verify all variants have valid telemetry
        for r in results:
            assert r["throughput_wps"] > 0
            assert r["mean_latency_ms"] > 0
            assert r["mean_prefill_ms"] >= 0
            assert r["mean_decoding_ms"] >= 0
            assert "reader_ans" in r
            assert 0.0 <= r["gold_recall_pct"] <= 100.0

        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "QUANTA Positional S-Expression System Prompt Optimization Report" in content
        assert "P0: Baseline Terse" in content
        assert "P2: Relational & Transitive Guidance" in content
        assert "P3: Geographic & Jurisdiction Hierarchy" in content
