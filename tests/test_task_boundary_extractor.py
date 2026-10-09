"""Unit tests for TaskBoundaryExtractor and query isolation strategies (§Phase 1, exp-027*)."""

from __future__ import annotations

import pytest

from config.multi_scale_config import MultiScaleConfig
from parser.task_boundary_extractor import (
    ExtractedTaskIntent,
    TaskBoundaryExtractor,
    decompose_query_context,
    extract_context_from_system,
    extract_target_entities,
)


SAMPLE_DOC_LONG = (
    "The celestial dynamics of the outer Magellanic Cloud exhibit intricate gravitational tidal perturbations "
    "induced by interaction with the Milky Way halo. Deep spectroscopic surveys reveal anomalous stellar kinematics "
    "across the leading arm, indicating that ancient dwarf galaxy mergers deposited substantial metal-poor globular "
    "clusters in the galactic periphery. Cryogenic engineering in superconducting systems mandates thermal management."
)

SAMPLE_DOC_HUNGARIAN = (
    "A kvantumoptikai kísérletek során a nemlineáris kristályok parametrikus lefelé konverziót hoznak létre. "
    "A koherens fotonpárok összefonódottságát Bell-egyenlőtlenségek mérésével hitelesítik a szegedi lézerközpontban. "
    "Az analízis kimutatta a fáziskoherencia megőrzését szobahőmérsékleten is."
)


class TestTaskBoundaryExtractor:
    """Suite verifying task boundary extraction, legacy parity, and variant behaviors."""

    def test_refactor_parity_qe_a(self):
        """QE-A must be identical in output to legacy decompose_query_context."""
        test_inputs = [
            # Explicit
            f"Context:\n{SAMPLE_DOC_LONG}\n\nQuestion: What causes the anomalous stellar kinematics?",
            # Tail paragraph
            f"{SAMPLE_DOC_LONG}\n\nWhat causes the anomalous stellar kinematics?",
            # Tail sentence
            f"{SAMPLE_DOC_LONG} What causes the kinematics?",
            # Direct question without context
            "What is the capital of France?",
            # Empty / whitespace
            "",
            "   ",
            None,
        ]

        extractor_qe_a = TaskBoundaryExtractor(strategy="QE-A")

        for inp in test_inputs:
            legacy_doc, legacy_q = decompose_query_context(inp)
            new_doc, new_q = extractor_qe_a.decompose(inp)

            assert legacy_doc == new_doc, f"Doc mismatch on: {inp!r}"
            assert legacy_q == new_q, f"Query mismatch on: {inp!r}"

    def test_explicit_delimiters_standard_and_inverted(self):
        """Explicit delimiters must be recognized in both Context->Question and Question->Context orders."""
        extractor = TaskBoundaryExtractor(strategy="QE-B")

        # Standard order
        text_standard = f"Context:\n{SAMPLE_DOC_LONG}\n\nQuestion: What was deposited in the periphery?"
        intent1 = extractor.extract(text_standard)
        assert intent1.boundary_location == "EXPLICIT"
        assert intent1.query_text == "What was deposited in the periphery?"
        assert intent1.context_text == SAMPLE_DOC_LONG
        assert intent1.char_span is not None
        assert text_standard[intent1.char_span[0]:intent1.char_span[1]] == intent1.query_text

        # Inverted order (Question first, then Context)
        text_inverted = f"Question: What was deposited in the periphery?\n\nContext:\n{SAMPLE_DOC_LONG}"
        intent2 = extractor.extract(text_inverted)
        assert intent2.boundary_location == "EXPLICIT"
        assert intent2.query_text == "What was deposited in the periphery?"
        assert intent2.context_text == SAMPLE_DOC_LONG

        # Hungarian explicit delimiter
        text_hu = f"Eredeti forrásdokumentumok:\n{SAMPLE_DOC_HUNGARIAN}\n\nKérdés: Hol végezték a méréseket?"
        intent3 = extractor.extract(text_hu)
        assert intent3.boundary_location == "EXPLICIT"
        assert intent3.query_text == "Hol végezték a méréseket?"
        assert intent3.context_text == SAMPLE_DOC_HUNGARIAN

    def test_head_boundary_interrogative_and_directives(self):
        """QE-B must reliably isolate head questions and task directives preceding document blocks."""
        extractor_b = TaskBoundaryExtractor(strategy="QE-B")
        extractor_a = TaskBoundaryExtractor(strategy="QE-A")

        prompt = (
            "Where is the apple?\n\n"
            "Instructions: Track the dynamic locations and entity movements in the narrative. "
            "Conclude with: 'Answer: <final answer>'.\n\n"
            f"Narrative Block [1]: {SAMPLE_DOC_LONG}\n\n"
            "Narrative Block [2]: Sandra journeyed to the garden."
        )

        # QE-A fails on head questions and treats the whole document as query or falls back
        intent_a = extractor_a.extract(prompt)
        assert intent_a.boundary_location == "FALLBACK"
        assert intent_a.context_text is None

        # QE-B correctly isolates head question and instructions
        intent_b = extractor_b.extract(prompt)
        assert intent_b.boundary_location == "HEAD"
        assert "Where is the apple?" in intent_b.query_text
        assert "Instructions:" in intent_b.query_text
        assert intent_b.context_text is not None
        assert "Narrative Block [1]" in intent_b.context_text
        assert "Sandra journeyed" in intent_b.context_text

    def test_head_directive_you_are(self):
        """Head directives starting with 'You are...' or 'Your task is...' must be extracted as query/directive."""
        extractor = TaskBoundaryExtractor(strategy="QE-B")

        prompt = (
            "You are an expert astrophysicist. Your task is to identify anomalous kinematics.\n\n"
            f"Section [1]: {SAMPLE_DOC_LONG}"
        )

        intent = extractor.extract(prompt)
        assert intent.boundary_location == "HEAD"
        assert "Your task is to identify anomalous kinematics." in intent.query_text
        assert intent.context_text is not None
        assert "Section [1]" in intent.context_text

    def test_tail_boundary(self):
        """Tail interrogatives must be properly extracted by both QE-A and QE-B."""
        extractor = TaskBoundaryExtractor(strategy="QE-B")

        prompt = (
            f"Paragraph [1]: {SAMPLE_DOC_LONG}\n\n"
            "Which galaxy mergers deposited globular clusters?"
        )

        intent = extractor.extract(prompt)
        assert intent.boundary_location == "TAIL"
        assert intent.query_text == "Which galaxy mergers deposited globular clusters?"
        assert intent.context_text == f"Paragraph [1]: {SAMPLE_DOC_LONG}"

    def test_system_and_chat_message_extraction_qe_c(self):
        """QE-C must handle structured chat messages and system instructions."""
        extractor = TaskBoundaryExtractor(strategy="QE-C")

        # 1. System context
        messages_sys = [
            {"role": "system", "content": f"Context:\n{SAMPLE_DOC_LONG}"},
            {"role": "user", "content": "What causes the perturbations?"},
        ]
        intent_sys = extractor.extract(messages=messages_sys)
        assert intent_sys.boundary_location == "SYSTEM"
        assert intent_sys.query_text == "What causes the perturbations?"
        assert intent_sys.context_text == SAMPLE_DOC_LONG

        # 2. Multi-turn conversation where previous turn holds bulky document
        messages_multi = [
            {"role": "user", "content": f"Please analyze this text:\n\n{SAMPLE_DOC_LONG}"},
            {"role": "assistant", "content": "I have reviewed the document. What would you like to know?"},
            {"role": "user", "content": "Where did the mergers deposit globular clusters?"},
        ]
        intent_multi = extractor.extract(messages=messages_multi)
        assert intent_multi.boundary_location == "CHAT_ROLES"
        assert intent_multi.query_text == "Where did the mergers deposit globular clusters?"
        assert intent_multi.context_text is not None
        assert "Magellanic Cloud" in intent_multi.context_text

    def test_target_entity_extraction(self):
        """Target entities must include alphanumeric codes, proper nouns, and quoted terms without stopwords."""
        q1 = "What is the designated mission access code for Vault 81 and Project DELTA-X99?"
        entities1 = extract_target_entities(q1)
        assert "Vault 81" in entities1
        assert "DELTA-X99" in entities1

        q2 = "What university did John von Neumann and Albert Einstein attend in Berlin?"
        entities2 = extract_target_entities(q2)
        assert "John von Neumann" in entities2
        assert "Albert Einstein" in entities2
        assert "Berlin" in entities2

        # Verify stopwords like 'What', 'Where', 'Answer' are excluded
        assert "What" not in entities2
        assert "Where" not in entities2

    def test_config_integration(self, tmp_path):
        """MultiScaleConfig strategy resolving into TaskBoundaryExtractor."""
        # Uncalibrated empty profile defaults to passthrough (QE-A)
        empty_profile = tmp_path / "empty_profile.json"
        cfg_uncalibrated = MultiScaleConfig(profile_path=empty_profile)
        ext_uncalibrated = TaskBoundaryExtractor(config=cfg_uncalibrated)
        assert ext_uncalibrated.strategy == "QE-A"

        # Explicit override to QE-B
        cfg_qeb = cfg_uncalibrated.with_overrides({"query_extractor.strategy": "QE-B"})
        ext_qeb = TaskBoundaryExtractor(config=cfg_qeb)
        assert ext_qeb.strategy == "QE-B"

        # Explicit override to QE-C
        cfg_qec = cfg_uncalibrated.with_overrides({"query_extractor.strategy": "QE-C"})
        ext_qec = TaskBoundaryExtractor(config=cfg_qec)
        assert ext_qec.strategy == "QE-C"
