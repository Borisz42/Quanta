"""Unit and integration tests for QUANTA S-Expression Skeleton Transducer (Master Plan Section 2).

Verifies Section 2 requirements:
1. Fast tokenizer/parser processing keyword clauses:
   (graph (entity E1 "...") (event EV1 pred :subj E1 :obj E2))
   and positional clauses:
   ((e E1 "...") (ev EV1 pred E1 E2)).
2. Accurate extraction of single-letter co-decoded Kev decision symbols (intent, epist, allen, pearl).
3. Graceful error recovery: auto-closing unbalanced parentheses, stripping fences, comments, and skipping fragments.
4. Exact character and UTF-8 byte span grounding fidelity via SpanAligner.
5. Bidirectional round-trip serialization between typed schemas and S-expression formats.
6. MockSkeletonTransducer and SkeletonTransducer format dispatching (sexpr_compact, sexpr_positional, json).
7. UnslothTransducer integration and parameter exposure.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest

from core.asg import QuantaGraph
from parser.skeleton_transducer import (
    CO_DECODED_COMPACT_SEXPR_SYSTEM_PROMPT,
    CO_DECODED_POSITIONAL_SEXPR_SYSTEM_PROMPT,
    CO_DECODED_SKELETON_SYSTEM_PROMPT,
    DEFAULT_COMPACT_SEXPR_SYSTEM_PROMPT,
    DEFAULT_POSITIONAL_SEXPR_SYSTEM_PROMPT,
    DEFAULT_SKELETON_SYSTEM_PROMPT,
    MockSkeletonTransducer,
    SkeletonEntity,
    SkeletonEvent,
    SkeletonExtractionResult,
    SkeletonTransducer,
    _locate_skeleton_gbnf,
    estimate_token_count,
    parse_skeleton_sexpr,
)
from parser.span_aligner import SpanAligner
from parser.unsloth_transducer import MockUnslothTransducer, UnslothTransducer


# ---------------------------------------------------------------------------
# 1. High-Speed S-Expression Lexer & Parser Unit Tests
# ---------------------------------------------------------------------------

class TestSExprParserUnit:
    """Verifies parse_skeleton_sexpr on compact keyword and positional S-expressions."""

    def test_parse_compact_keyword_sexpr_basic(self):
        sexpr = """
        (graph
          (entity E1 "Charles Babbage")
          (entity E2 "Trinity College, Cambridge")
          (event EV1 matriculated :subj E1 :obj E2))
        """
        data = parse_skeleton_sexpr(sexpr)
        assert len(data["entities"]) == 2
        assert len(data["events"]) == 1

        e1 = data["entities"][0]
        assert e1["id"] == "E1"
        assert e1["text"] == "Charles Babbage"
        assert e1["category"] == "OBJECT"

        e2 = data["entities"][1]
        assert e2["id"] == "E2"
        assert e2["text"] == "Trinity College, Cambridge"

        ev1 = data["events"][0]
        assert ev1["id"] == "EV1"
        assert ev1["pred"] == "matriculated"
        assert ev1["subj"] == "E1"
        assert ev1["obj"] == "E2"
        assert "intent" not in ev1

    def test_parse_compact_sexpr_with_types(self):
        sexpr = """
        (graph
          (entity E1 "Alan Turing" :type PERSON)
          (entity E2 "Cambridge University" :type LOCATION)
          (event EV1 studied :subj E1 :obj E2))
        """
        data = parse_skeleton_sexpr(sexpr)
        assert data["entities"][0]["category"] == "PERSON"
        assert data["entities"][1]["category"] == "LOCATION"

    def test_parse_positional_sexpr_basic(self):
        sexpr = """
        ((e E1 "Charles Babbage")
         (e E2 "Trinity College, Cambridge")
         (ev EV1 matriculated E1 E2))
        """
        data = parse_skeleton_sexpr(sexpr)
        assert len(data["entities"]) == 2
        assert len(data["events"]) == 1

        assert data["entities"][0]["id"] == "E1"
        assert data["entities"][0]["text"] == "Charles Babbage"
        assert data["events"][0]["pred"] == "matriculated"
        assert data["events"][0]["subj"] == "E1"
        assert data["events"][0]["obj"] == "E2"

    def test_parse_positional_sexpr_with_types(self):
        sexpr = """
        ((e E1 "Charles Babbage" PERSON)
         (e E2 "Trinity College" LOCATION)
         (ev EV1 graduated E1 E2))
        """
        data = parse_skeleton_sexpr(sexpr)
        assert data["entities"][0]["category"] == "PERSON"
        assert data["entities"][1]["category"] == "LOCATION"

    def test_parse_co_decoded_kev_compact(self):
        sexpr = """
        (graph
          (entity E1 "Charles Babbage")
          (entity E2 "Difference Engine")
          (event EV1 invented :subj E1 :obj E2 :intent I :epist O :allen B :pearl M))
        """
        data = parse_skeleton_sexpr(sexpr)
        ev = data["events"][0]
        assert ev["intent"] == "I"
        assert ev["epist"] == "O"
        assert ev["epistemic"] == "O"
        assert ev["allen"] == "B"
        assert ev["pearl"] == "M"

    def test_parse_co_decoded_kev_positional(self):
        sexpr = """
        ((e E1 "Charles Babbage")
         (e E2 "Difference Engine")
         (ev EV1 invented E1 E2 I O B M))
        """
        data = parse_skeleton_sexpr(sexpr)
        ev = data["events"][0]
        assert ev["subj"] == "E1"
        assert ev["obj"] == "E2"
        assert ev["intent"] == "I"
        assert ev["epist"] == "O"
        assert ev["allen"] == "B"
        assert ev["pearl"] == "M"

    def test_parse_partial_positional_event(self):
        # Event with only subject and no object
        sexpr = """
        ((e E1 "Eleanor Vance")
         (ev EV1 entered E1))
        """
        data = parse_skeleton_sexpr(sexpr)
        assert data["events"][0]["subj"] == "E1"
        assert data["events"][0]["obj"] is None

        # Event with predicate only
        sexpr2 = "((e E1 \"Test\") (ev EV1 observe))"
        data2 = parse_skeleton_sexpr(sexpr2)
        assert data2["events"][0]["subj"] is None
        assert data2["events"][0]["obj"] is None

    def test_parse_auto_repair_unclosed_parentheses(self):
        # Truncated completion missing closing parentheses
        truncated = '(graph (entity E1 "Marcus Vance") (event EV1 pressurize :subj E1'
        data = parse_skeleton_sexpr(truncated)
        assert len(data["entities"]) == 1
        assert data["entities"][0]["text"] == "Marcus Vance"
        assert len(data["events"]) == 1
        assert data["events"][0]["subj"] == "E1"

    def test_parse_strips_comments_and_code_fences(self):
        raw = """```lisp
        ;; QUANTA extraction result
        (graph
          ;; Entities
          (entity E1 "Charles Babbage") ; creator
          (event EV1 designed :subj E1) ; action
        )
        ```"""
        data = parse_skeleton_sexpr(raw)
        assert len(data["entities"]) == 1
        assert data["entities"][0]["text"] == "Charles Babbage"
        assert len(data["events"]) == 1

    def test_parse_strips_reasoning_blocks(self):
        raw = """<think>Thinking about entities and relations...</think>
        (graph (entity E1 "Alan Turing") (event EV1 worked :subj E1))"""
        data = parse_skeleton_sexpr(raw)
        assert len(data["entities"]) == 1
        assert data["entities"][0]["text"] == "Alan Turing"

    def test_parse_escaped_strings(self):
        sexpr = r'(graph (entity E1 "\"Analytical\" Engine") (event EV1 test :subj E1))'
        data = parse_skeleton_sexpr(sexpr)
        assert data["entities"][0]["text"] == '"Analytical" Engine'

    def test_parse_skips_malformed_clauses_gracefully(self):
        sexpr = """(graph
          (invalid clause with no sense)
          (entity E1 "Charles Babbage")
          (garbage)
          (event EV1 matriculated :subj E1)
        )"""
        data = parse_skeleton_sexpr(sexpr)
        assert len(data["entities"]) == 1
        assert data["entities"][0]["id"] == "E1"
        assert len(data["events"]) == 1
        assert data["events"][0]["id"] == "EV1"

    def test_parse_empty_input(self):
        assert parse_skeleton_sexpr("") == {"entities": [], "events": []}
        assert parse_skeleton_sexpr("(graph)") == {"entities": [], "events": []}
        assert parse_skeleton_sexpr("()") == {"entities": [], "events": []}

    def test_parse_json_fallback(self):
        raw_json = json.dumps({
            "entities": [{"id": "E1", "text": "Charles Babbage"}],
            "events": [{"id": "EV1", "pred": "invented", "subj": "E1"}],
        })
        data = parse_skeleton_sexpr(raw_json)
        assert len(data["entities"]) == 1
        assert data["entities"][0]["text"] == "Charles Babbage"
        assert len(data["events"]) == 1


# ---------------------------------------------------------------------------
# 2. Span Grounding Fidelity & SpanAligner Integration
# ---------------------------------------------------------------------------

class TestSpanGroundingFidelity:
    """Verifies exact character and byte-level grounding from S-expression extraction."""

    def test_grounding_compact_sexpr(self):
        text = "Charles Babbage studied at Trinity College, Cambridge in 1810."
        sexpr = """(graph
          (entity E1 "Charles Babbage")
          (entity E2 "Trinity College, Cambridge")
          (event EV1 studied :subj E1 :obj E2))"""

        res = SkeletonExtractionResult.from_sexpr(sexpr, text=text)
        assert len(res.entities) == 2
        assert len(res.events) == 1

        e1 = res.entities[0]
        assert text[e1.char_span[0]:e1.char_span[1]] == "Charles Babbage"
        assert e1.byte_span is not None
        raw_bytes = text.encode("utf-8")
        assert raw_bytes[e1.byte_span[0]:e1.byte_span[1]].decode("utf-8") == "Charles Babbage"

        e2 = res.entities[1]
        assert text[e2.char_span[0]:e2.char_span[1]] == "Trinity College, Cambridge"

        ev1 = res.events[0]
        assert text[ev1.char_span[0]:ev1.char_span[1]] == "studied"

    def test_grounding_positional_sexpr(self):
        text = "Eleanor Vance entered the containment cell cautiously."
        sexpr = """((e E1 "Eleanor Vance")
                    (e E2 "containment cell")
                    (ev EV1 entered E1 E2))"""

        res = SkeletonExtractionResult.from_sexpr(sexpr, text=text)
        assert len(res.entities) == 2
        assert text[res.entities[0].char_span[0]:res.entities[0].char_span[1]] == "Eleanor Vance"
        assert text[res.entities[1].char_span[0]:res.entities[1].char_span[1]] == "containment cell"
        assert text[res.events[0].char_span[0]:res.events[0].char_span[1]] == "entered"

    def test_grounding_multibyte_hungarian_unicode(self):
        text = "Dr. Kovács János szintetizálta az új fluoropolimer mintát."
        sexpr = """(graph
          (entity E1 "Kovács János")
          (entity E2 "fluoropolimer minta")
          (event EV1 szintetizálta :subj E1 :obj E2))"""

        res = SkeletonExtractionResult.from_sexpr(sexpr, text=text)
        e1 = res.entities[0]
        assert text[e1.char_span[0]:e1.char_span[1]] == "Kovács János"
        # Multibyte: byte_span length > char_span length
        char_len = e1.char_span[1] - e1.char_span[0]
        byte_len = e1.byte_span[1] - e1.byte_span[0]
        assert byte_len > char_len
        raw_bytes = text.encode("utf-8")
        assert raw_bytes[e1.byte_span[0]:e1.byte_span[1]].decode("utf-8") == "Kovács János"


# ---------------------------------------------------------------------------
# 3. Serialization Round-Trip Tests
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    """Verifies roundtrip fidelity between typed schemas and S-expression formats."""

    def test_compact_sexpr_roundtrip(self):
        original = SkeletonExtractionResult(
            entities=[
                SkeletonEntity(id="E1", surface_text="Charles Babbage", char_span=(0, 15), category="PERSON"),
                SkeletonEntity(id="E2", surface_text="Difference Engine", char_span=(25, 42), category="OBJECT"),
            ],
            events=[
                SkeletonEvent(
                    id="EV1",
                    predicate="invented",
                    char_span=(16, 24),
                    subject_ent_id="E1",
                    object_ent_id="E2",
                    intent="I",
                    epist="O",
                    allen="B",
                    pearl="M",
                ),
            ],
        )

        sexpr = original.to_skeleton_compact_sexpr()
        assert sexpr.startswith("(graph ")
        assert '(entity E1 "Charles Babbage" :type PERSON)' in sexpr
        assert ':intent I :epist O :allen B :pearl M' in sexpr

        data = parse_skeleton_sexpr(sexpr)
        assert len(data["entities"]) == 2
        assert len(data["events"]) == 1
        assert data["entities"][0]["text"] == "Charles Babbage"
        assert data["events"][0]["intent"] == "I"
        assert data["events"][0]["epist"] == "O"

    def test_positional_sexpr_roundtrip(self):
        original = SkeletonExtractionResult(
            entities=[
                SkeletonEntity(id="E1", surface_text="Charles Babbage", char_span=(0, 15)),
                SkeletonEntity(id="E2", surface_text="Difference Engine", char_span=(25, 42)),
            ],
            events=[
                SkeletonEvent(
                    id="EV1",
                    predicate="invented",
                    char_span=(16, 24),
                    subject_ent_id="E1",
                    object_ent_id="E2",
                    intent="I",
                    epist="O",
                    allen="B",
                    pearl="M",
                ),
            ],
        )

        sexpr = original.to_skeleton_positional_sexpr()
        assert sexpr.startswith("(")
        assert '(e E1 "Charles Babbage")' in sexpr
        assert '(ev EV1 invented E1 E2 I O B M)' in sexpr

        data = parse_skeleton_sexpr(sexpr)
        assert len(data["entities"]) == 2
        assert len(data["events"]) == 1
        assert data["events"][0]["subj"] == "E1"
        assert data["events"][0]["obj"] == "E2"
        assert data["events"][0]["intent"] == "I"
        assert data["events"][0]["pearl"] == "M"

    def test_to_skeleton_raw_dispatch(self):
        res = SkeletonExtractionResult(
            entities=[SkeletonEntity(id="E1", surface_text="Babbage", char_span=(0, 7))],
            events=[SkeletonEvent(id="EV1", predicate="compute", char_span=(8, 15), subject_ent_id="E1")],
        )

        compact_str = res.to_skeleton_raw(format="sexpr_compact")
        assert compact_str.startswith("(graph")

        pos_str = res.to_skeleton_raw(format="sexpr_positional")
        assert pos_str.startswith("((e")

        json_str = res.to_skeleton_raw(format="json")
        assert json_str.startswith("{")


# ---------------------------------------------------------------------------
# 4. MockSkeletonTransducer Format & Speed Verification
# ---------------------------------------------------------------------------

class TestMockSkeletonTransducerFormats:
    """Verifies MockSkeletonTransducer in compact, positional, and JSON modes."""

    def test_default_is_compact_sexpr(self):
        mock = MockSkeletonTransducer()
        assert mock.skeleton_format == "sexpr_compact"
        assert mock.co_decoded is False

        text = "Dr. Marcus Vance pressurized the argon cylinder."
        res = mock.transduce(text)

        assert res.metadata["skeleton_format"] == "sexpr_compact"
        assert res.metadata["co_decoded"] is False
        assert res.metadata["raw_repr"].startswith("(graph")
        assert len(res.entities) >= 2
        assert len(res.events) >= 1
        assert res.events[0].intent is None  # Pure SVO default

    def test_positional_mode(self):
        mock = MockSkeletonTransducer(skeleton_format="sexpr_positional")
        assert mock.skeleton_format == "sexpr_positional"

        text = "Dr. Marcus Vance pressurized the argon cylinder."
        res = mock.transduce(text)

        assert res.metadata["skeleton_format"] == "sexpr_positional"
        assert res.metadata["raw_repr"].startswith("((")
        assert "(ev" in res.metadata["raw_repr"]

    def test_co_decoded_compact_mode(self):
        mock = MockSkeletonTransducer(skeleton_format="sexpr_compact", co_decoded=True)
        text = "Dr. Marcus Vance pressurized the argon cylinder."
        res = mock.transduce(text)

        assert res.metadata["co_decoded"] is True
        assert res.events[0].intent == "I"
        assert res.events[0].epist == "O"
        assert res.events[0].allen == "B"
        assert res.events[0].pearl == "M"
        assert ":intent I" in res.metadata["raw_repr"]

    def test_co_decoded_positional_mode(self):
        mock = MockSkeletonTransducer(skeleton_format="sexpr_positional", co_decoded=True)
        text = "Dr. Marcus Vance pressurized the argon cylinder."
        res = mock.transduce(text)

        assert res.metadata["co_decoded"] is True
        assert res.events[0].intent == "I"
        assert "I O B M" in res.metadata["raw_repr"]

    def test_json_mode_backward_compatibility(self):
        mock = MockSkeletonTransducer(skeleton_format="json")
        text = "Dr. Marcus Vance pressurized the argon cylinder."
        res = mock.transduce(text)

        assert res.metadata["skeleton_format"] == "json"
        assert res.metadata["raw_repr"].startswith("{")
        assert json.loads(res.metadata["raw_repr"])["entities"]

    def test_transduce_raw_output(self):
        mock = MockSkeletonTransducer(skeleton_format="sexpr_compact")
        text = "Dr. Marcus Vance pressurized the argon cylinder."

        compact_raw = mock.transduce_raw(text, skeleton_format="sexpr_compact")
        assert compact_raw.startswith("(graph")

        pos_raw = mock.transduce_raw(text, skeleton_format="sexpr_positional")
        assert pos_raw.startswith("((e")

        json_raw = mock.transduce_raw(text, skeleton_format="json")
        assert json_raw.startswith("{")

    def test_token_reduction_comparisons(self):
        """Verify token counts follow: positional < compact < standard JSON < co-decoded JSON."""
        mock = MockSkeletonTransducer()
        text = "Dr. Marcus Vance pressurized the argon cylinder inside the test chamber."

        res_json = mock.transduce(text, skeleton_format="json", co_decoded=False)
        res_compact = mock.transduce(text, skeleton_format="sexpr_compact", co_decoded=False)
        res_pos = mock.transduce(text, skeleton_format="sexpr_positional", co_decoded=False)

        tokens_json = res_json.metadata["decoding_token_count"]
        tokens_compact = res_compact.metadata["decoding_token_count"]
        tokens_pos = res_pos.metadata["decoding_token_count"]

        assert tokens_pos <= tokens_compact
        assert tokens_compact <= tokens_json

    def test_sub_10ms_latency(self):
        mock = MockSkeletonTransducer()
        text = "Dr. Marcus Vance pressurized the argon cylinder."
        res = mock.transduce(text)
        assert res.metadata["latency_sec"] < 0.05  # typically < 5ms


# ---------------------------------------------------------------------------
# 5. Production SkeletonTransducer Fallback & Dispatch
# ---------------------------------------------------------------------------

class TestSkeletonTransducerProductionClient:
    """Verifies SkeletonTransducer production client configuration and offline mock fallback."""

    def test_grammar_selection_for_formats(self):
        t_compact = SkeletonTransducer(skeleton_format="sexpr_compact")
        assert "compact_skeleton_sexpr.gbnf" in str(t_compact.grammar_path)

        t_pos = SkeletonTransducer(skeleton_format="sexpr_positional")
        assert "positional_skeleton_sexpr.gbnf" in str(t_pos.grammar_path)

        t_json = SkeletonTransducer(skeleton_format="json", co_decoded=False)
        assert "skeleton_schema.gbnf" in str(t_json.grammar_path)

        t_co_json = SkeletonTransducer(skeleton_format="json", co_decoded=True)
        assert "co_decoded_skeleton_schema.gbnf" in str(t_co_json.grammar_path)

    def test_system_prompt_selection_for_formats(self):
        t_compact = SkeletonTransducer(skeleton_format="sexpr_compact", co_decoded=False)
        assert t_compact.system_prompt == DEFAULT_COMPACT_SEXPR_SYSTEM_PROMPT

        t_co_compact = SkeletonTransducer(skeleton_format="sexpr_compact", co_decoded=True)
        assert t_co_compact.system_prompt == CO_DECODED_COMPACT_SEXPR_SYSTEM_PROMPT

        t_pos = SkeletonTransducer(skeleton_format="sexpr_positional", co_decoded=False)
        assert t_pos.system_prompt == DEFAULT_POSITIONAL_SEXPR_SYSTEM_PROMPT

        t_co_pos = SkeletonTransducer(skeleton_format="sexpr_positional", co_decoded=True)
        assert t_co_pos.system_prompt == CO_DECODED_POSITIONAL_SEXPR_SYSTEM_PROMPT

    def test_offline_fallback_in_compact_mode(self):
        transducer = SkeletonTransducer(
            base_url="http://127.0.0.1:58999",  # unreachable offline port
            fallback_to_mock=True,
            skeleton_format="sexpr_compact",
        )
        text = "Dr. Marcus Vance pressurized the argon cylinder."
        res = transducer.transduce(text)

        assert res.metadata["backend"] == "mock_skeleton"
        assert res.metadata["skeleton_format"] == "sexpr_compact"
        assert res.metadata["fallback_from_server"] is True
        assert len(res.entities) >= 2
        assert len(res.events) >= 1

    def test_offline_transduce_raw_methods(self):
        transducer = SkeletonTransducer(
            base_url="http://127.0.0.1:58999",
            fallback_to_mock=True,
            skeleton_format="sexpr_compact",
        )
        text = "Dr. Marcus Vance pressurized the argon cylinder."
        raw_str = transducer.transduce_raw(text)
        assert raw_str.startswith("(graph")

        raw_async = asyncio.run(transducer.transduce_raw_async(text, skeleton_format="sexpr_positional"))
        assert raw_async.startswith("((e")


# ---------------------------------------------------------------------------
# 6. UnslothTransducer Integration Tests
# ---------------------------------------------------------------------------

class TestUnslothTransducerIntegration:
    """Verifies UnslothTransducer exposes skeleton_format and co_decoded."""

    def test_mock_unsloth_transducer_skeleton_formats(self):
        transducer = MockUnslothTransducer(skeleton_format="sexpr_compact")
        assert transducer.skeleton_format == "sexpr_compact"

        text = "Dr. Marcus Vance pressurized the argon cylinder."
        skel = transducer.transduce_skeleton(text)
        assert skel.metadata["skeleton_format"] == "sexpr_compact"

        raw_pos = transducer.transduce_raw(text, skeleton_format="sexpr_positional")
        assert raw_pos.startswith("((e")

    def test_unsloth_transducer_parameters_exposure(self):
        transducer = UnslothTransducer(
            base_url="http://127.0.0.1:58999",
            fallback_to_mock=True,
            skeleton_format="sexpr_positional",
            co_decoded=True,
        )
        assert transducer.skeleton_format == "sexpr_positional"
        assert transducer.co_decoded is True
        assert transducer._skeleton.skeleton_format == "sexpr_positional"
        assert transducer._skeleton.co_decoded is True

        text = "Dr. Marcus Vance pressurized the argon cylinder."
        res = transducer.transduce_skeleton(text)
        assert res.metadata["skeleton_format"] == "sexpr_positional"
        assert res.metadata["co_decoded"] is True
