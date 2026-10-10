"""Unit tests for QUANTA Compact Keyword S-Expression System Prompt Variants (exp-035a).

Verifies:
1. All 5 compact prompt variants (C0-C4) compile cleanly in COMPACT_PROMPT_VARIANTS and ALL_PROMPT_VARIANTS.
2. SkeletonTransducer and MockSkeletonTransducer accept prompt_variant parameter and resolve system_prompt correctly.
3. CognitivePipeline forwards prompt_variant to SkeletonTransducer for compact format.
"""

from __future__ import annotations

import pytest

from parser.skeleton_transducer import (
    ALL_PROMPT_VARIANTS,
    CO_DECODED_COMPACT_ENHANCED_PROMPT,
    COMPACT_BASELINE_PROMPT,
    COMPACT_JURISDICTION_HIERARCHY_PROMPT,
    COMPACT_MINIMALIST_PROMPT,
    COMPACT_PROMPT_VARIANTS,
    COMPACT_RELATIONAL_GUIDANCE_PROMPT,
    DEFAULT_COMPACT_SEXPR_SYSTEM_PROMPT,
    MockSkeletonTransducer,
    SkeletonTransducer,
)
from pipeline.cognitive_pipeline import CognitivePipeline


class TestCompactPromptVariants:
    """Verifies compact prompt dictionary registration and formatting."""

    def test_prompt_dictionary_registration(self):
        assert "c0" in COMPACT_PROMPT_VARIANTS
        assert "c1" in COMPACT_PROMPT_VARIANTS
        assert "c2" in COMPACT_PROMPT_VARIANTS
        assert "c3" in COMPACT_PROMPT_VARIANTS
        assert "c4" in COMPACT_PROMPT_VARIANTS

        assert COMPACT_PROMPT_VARIANTS["c0"] == COMPACT_BASELINE_PROMPT
        assert COMPACT_PROMPT_VARIANTS["c1"] == DEFAULT_COMPACT_SEXPR_SYSTEM_PROMPT
        assert COMPACT_PROMPT_VARIANTS["c2"] == COMPACT_RELATIONAL_GUIDANCE_PROMPT
        assert COMPACT_PROMPT_VARIANTS["c3"] == COMPACT_JURISDICTION_HIERARCHY_PROMPT
        assert COMPACT_PROMPT_VARIANTS["c4"] == CO_DECODED_COMPACT_ENHANCED_PROMPT

        assert "c0" in ALL_PROMPT_VARIANTS
        assert "p0" in ALL_PROMPT_VARIANTS

    def test_prompt_content_validity(self):
        for key, prompt in COMPACT_PROMPT_VARIANTS.items():
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 30
            assert "QUANTA" in prompt
            assert "Compact" in prompt or "S-expression" in prompt or "S-Expression" in prompt


class TestSkeletonTransducerCompactResolution:
    """Verifies that SkeletonTransducer and MockSkeletonTransducer resolve compact prompt variants."""

    def test_mock_transducer_variant_resolution(self):
        t0 = MockSkeletonTransducer(skeleton_format="sexpr_compact", prompt_variant="c0")
        assert t0.system_prompt == COMPACT_BASELINE_PROMPT

        t1 = MockSkeletonTransducer(skeleton_format="sexpr_compact", prompt_variant="c1")
        assert t1.system_prompt == COMPACT_MINIMALIST_PROMPT

        t2 = MockSkeletonTransducer(skeleton_format="sexpr_compact", prompt_variant="c2")
        assert t2.system_prompt == COMPACT_RELATIONAL_GUIDANCE_PROMPT

        t3 = MockSkeletonTransducer(skeleton_format="sexpr_compact", prompt_variant="c3")
        assert t3.system_prompt == COMPACT_JURISDICTION_HIERARCHY_PROMPT

        t4 = MockSkeletonTransducer(skeleton_format="sexpr_compact", prompt_variant="c4")
        assert t4.system_prompt == CO_DECODED_COMPACT_ENHANCED_PROMPT

    def test_skeleton_transducer_variant_resolution(self):
        st1 = SkeletonTransducer(
            fallback_to_mock=True,
            skeleton_format="sexpr_compact",
            prompt_variant="c1",
        )
        assert st1.prompt_variant == "c1"
        assert st1.system_prompt == COMPACT_MINIMALIST_PROMPT

        st3 = SkeletonTransducer(
            fallback_to_mock=True,
            skeleton_format="sexpr_compact",
            prompt_variant="c3",
        )
        assert st3.prompt_variant == "c3"
        assert st3.system_prompt == COMPACT_JURISDICTION_HIERARCHY_PROMPT


class TestPipelineCompactPromptIntegration:
    """Verifies that CognitivePipeline cleanly accepts and forwards prompt_variant for compact S-expressions."""

    def test_pipeline_initialization_with_compact_variant(self):
        pipeline = CognitivePipeline(
            skeleton_format="sexpr_compact",
            prompt_variant="c3",
            co_decoded_kev=False,
        )
        assert pipeline.prompt_variant == "c3"
        assert pipeline.skeleton_transducer.prompt_variant == "c3"
        assert pipeline.skeleton_transducer.system_prompt == COMPACT_JURISDICTION_HIERARCHY_PROMPT
        pipeline.close()
