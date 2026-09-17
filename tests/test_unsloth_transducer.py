"""Unit and integration tests for QUANTA UnslothTransducer and MockUnslothTransducer.

Verifies:
1. GBNF grammar specification auto-detection, loading, and injection.
2. OpenAI-compatible request payload construction with extra_body={"grammar": ...}.
3. Active entity manifest injection into prompt context and influencing entity IDs.
4. MockUnslothTransducer deterministic execution, Eleanor Vance benchmark, and sub-5ms latency.
5. Dynamic heuristic extraction for unknown text emitting valid S-expressions.
6. Graceful fallback to mock when Unsloth endpoint is offline without unhandled socket errors.
7. Strict error raising when fallback_to_mock=False and server is offline.
8. Simulated live HTTP response handling and S-expression parsing.
9. Async API wrappers (transduce_async, transduce_raw_async).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from unittest.mock import MagicMock

import pytest
import requests

from core.asg import QuantaGraph
from parser.entity_manifest import EntityRecord
from parser.schema import DiscourseExtractionResult
from parser.sexpr_parser import parse_sexpr, parse_to_asg
from parser.transducer import (
    CANONICAL_ELEANOR_VANCE_FIXTURE,
    CANONICAL_ELEANOR_VANCE_TEXT,
    CANONICAL_STRESS_1_TEXT,
)
from parser.unsloth_transducer import (
    DEFAULT_UNSLOTH_SYSTEM_PROMPT,
    MockUnslothTransducer,
    UnslothTransducer,
    _format_entity_manifest,
    _locate_gbnf_grammar,
)


# ---------------------------------------------------------------------------
# 1. GBNF Grammar Loading & Payload Structure Tests
# ---------------------------------------------------------------------------

def test_locate_gbnf_grammar_and_loading():
    """Verify that GBNF grammar is automatically found and loaded."""
    path = _locate_gbnf_grammar()
    assert path.is_file()
    assert path.name == "quanta_asg.gbnf"

    transducer = UnslothTransducer(fallback_to_mock=True)
    assert hasattr(transducer, "grammar_content")
    assert "root ::=" in transducer.grammar_content
    assert "entity_clause ::=" in transducer.grammar_content
    assert "event_clause ::=" in transducer.grammar_content
    assert "relation_clause ::=" in transducer.grammar_content
    assert "prop_clause ::=" in transducer.grammar_content


def test_locate_gbnf_grammar_invalid_path():
    """Verify FileNotFoundError if invalid grammar path is passed."""
    with pytest.raises(FileNotFoundError):
        _locate_gbnf_grammar("non_existent_path.gbnf")


def test_payload_structure_and_grammar_injection():
    """Verify request payload contains OpenAI-compatible format with extra_body grammar."""
    transducer = UnslothTransducer(
        base_url="http://localhost:8888/v1",
        model="qwen3.5-4b",
        fallback_to_mock=True,
    )
    payload = transducer._build_payload(
        text="Alice observed the experiment.",
        chunk_id="chunk_test_01",
        temperature=0.0,
        max_tokens=1024,
    )

    assert payload["model"] == "qwen3.5-4b"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 1024
    assert "extra_body" in payload
    assert "grammar" in payload["extra_body"]
    assert payload["extra_body"]["grammar"] == transducer.grammar_content

    messages = payload["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "QUANTA Neural Discourse Transducer" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "CHUNK TEXT:\nAlice observed the experiment." in messages[1]["content"]


# ---------------------------------------------------------------------------
# 2. Active Entity Manifest Injection Tests
# ---------------------------------------------------------------------------

def test_active_entity_manifest_formatting():
    """Verify active entity records format into compact ACTIVE ENTITIES prompt block."""
    entities = [
        EntityRecord(
            canonical_id="E1",
            canonical_name="Dr. Eleanor Vance",
            category="PERSON",
            surface_aliases=["Eleanor", "Vance"],
        ),
        EntityRecord(
            canonical_id="E2",
            canonical_name="synthetic compound",
            category="SUBSTANCE",
            surface_aliases=["specimen"],
        ),
    ]

    manifest = _format_entity_manifest(entities)
    assert manifest is not None
    assert "ACTIVE ENTITIES:" in manifest
    assert "- E1: Dr. Eleanor Vance (aliases: Eleanor, Vance)" in manifest
    assert "- E2: synthetic compound (aliases: specimen)" in manifest


def test_payload_includes_active_entities():
    """Verify active entities are injected into system and user prompts."""
    transducer = UnslothTransducer(fallback_to_mock=True)
    active_entities = [
        EntityRecord(
            canonical_id="E5",
            canonical_name="Dr. Vance",
            category="PERSON",
            surface_aliases=["Eleanor"],
        )
    ]

    payload = transducer._build_payload(
        text="She replicated the experiment.",
        active_entities=active_entities,
    )

    # In user prompt
    user_content = payload["messages"][1]["content"]
    assert "ACTIVE ENTITIES:" in user_content
    assert "- E5: Dr. Vance (aliases: Eleanor)" in user_content
    assert "CHUNK TEXT:\nShe replicated the experiment." in user_content

    # In system prompt context
    system_content = payload["messages"][0]["content"]
    assert "CURRENT CONTEXT:" in system_content
    assert "- E5: Dr. Vance (aliases: Eleanor)" in system_content


# ---------------------------------------------------------------------------
# 3. MockUnslothTransducer Offline Tests
# ---------------------------------------------------------------------------

def test_mock_transducer_eleanor_vance_fixture():
    """Verify MockUnslothTransducer returns complete Eleanor Vance extraction."""
    mock = MockUnslothTransducer()
    result = mock.transduce(CANONICAL_ELEANOR_VANCE_TEXT, chunk_id="test_ev")

    assert isinstance(result, DiscourseExtractionResult)
    assert result.chunk_id == "test_ev"
    assert len(result.entities) == 5
    assert len(result.events) == 6
    assert len(result.relations) == 5
    assert len(result.propositions) == 5

    # Check specific entity IDs and names
    names = [e.canonical_name for e in result.entities]
    assert "Dr. Eleanor Vance" in names
    assert "synthetic compound" in names
    assert "cryogenic containment cell" in names

    # Compile to QuantaGraph
    sexpr_raw = mock.transduce_raw(CANONICAL_ELEANOR_VANCE_TEXT)
    graph = parse_to_asg(sexpr_raw, validate=True)
    assert isinstance(graph, QuantaGraph)
    assert len(graph.nodes) == 11
    assert graph.validation.is_valid is True


def test_mock_transducer_stress_fixture():
    """Verify MockUnslothTransducer recognizes canonical stress fixtures."""
    mock = MockUnslothTransducer()
    res = mock.transduce(CANONICAL_STRESS_1_TEXT)
    assert len(res.entities) == 5
    assert len(res.events) == 5
    assert any(e.canonical_name == "Alice" for e in res.entities)


def test_mock_transducer_active_entities_influence():
    """Verify active entities override entity IDs and wire into event arguments."""
    mock = MockUnslothTransducer()
    active_entities = [
        EntityRecord(
            canonical_id="E99",
            canonical_name="Dr. Eleanor Vance",
            category="PERSON",
            surface_aliases=["Eleanor", "Vance"],
        )
    ]

    result = mock.transduce(CANONICAL_ELEANOR_VANCE_TEXT, active_entities=active_entities)
    vance_ent = next(e for e in result.entities if e.canonical_name == "Dr. Eleanor Vance")
    assert vance_ent.id == "E99"

    # Event 1 agent was Vance; should now reference E99
    ev1 = next(ev for ev in result.events if ev.id == "Ev1")
    assert ev1.agent_id == "E99"


def test_mock_transducer_unknown_text_dynamic():
    """Verify dynamic heuristic extraction on novel text produces valid S-expressions."""
    mock = MockUnslothTransducer()
    text = "The rover surveyed the crater rim."
    raw = mock.transduce_raw(text)

    assert raw.startswith("(graph")
    parsed = parse_sexpr(raw)
    assert len(parsed.entities) == 1
    assert len(parsed.events) == 1
    assert parsed.events[0].agent_id == parsed.entities[0].id


def test_mock_transducer_latency():
    """Verify MockUnslothTransducer runs in < 5ms."""
    mock = MockUnslothTransducer()
    text = "Alice observed the reaction."

    # Warm-up
    mock.transduce(text)

    iterations = 20
    t0 = time.perf_counter()
    for _ in range(iterations):
        res = mock.transduce(text)
        assert res is not None
    avg_latency_ms = ((time.perf_counter() - t0) / iterations) * 1000.0

    assert avg_latency_ms < 5.0, f"Average latency {avg_latency_ms:.2f}ms exceeded 5ms limit."


# ---------------------------------------------------------------------------
# 4. UnslothTransducer Fallback & Resilience Tests
# ---------------------------------------------------------------------------

def test_unsloth_graceful_fallback_when_offline():
    """Verify UnslothTransducer delegates to mock without socket crash when server is offline."""
    # Point to an unassigned local port
    transducer = UnslothTransducer(
        base_url="http://127.0.0.1:58999/v1",
        timeout=0.2,
        max_retries=1,
        fallback_to_mock=True,
    )

    assert transducer.check_health() is False
    assert transducer.is_available() is False

    # Calling transduce should fall back to mock cleanly
    result = transducer.transduce(
        CANONICAL_ELEANOR_VANCE_TEXT,
        chunk_id="chunk_offline_test",
    )
    assert isinstance(result, DiscourseExtractionResult)
    assert result.chunk_id == "chunk_offline_test"
    assert len(result.entities) == 5
    assert result.metadata.get("fallback_from_unsloth") is True

    # transduce_raw should also return a valid S-expression
    raw_sexpr = transducer.transduce_raw("Alice ran quickly.")
    assert raw_sexpr.startswith("(graph")
    parsed = parse_sexpr(raw_sexpr)
    assert len(parsed.entities) >= 1


def test_unsloth_strict_mode_raises_when_offline():
    """Verify UnslothTransducer raises RuntimeError when fallback_to_mock is False and server is offline."""
    transducer = UnslothTransducer(
        base_url="http://127.0.0.1:58999/v1",
        timeout=0.2,
        max_retries=1,
        fallback_to_mock=False,
    )

    with pytest.raises(RuntimeError) as exc_info:
        transducer.transduce("Some discourse text.")
    assert "Failed to extract S-expression from Unsloth server" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Simulated Live HTTP Response Tests
# ---------------------------------------------------------------------------

def test_unsloth_simulated_http_200():
    """Verify UnslothTransducer parses valid 200 JSON chat completion from server."""
    mock_session = MagicMock(spec=requests.Session)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '(graph :chunk-id "simulated_chunk"\n'
                        '  (entity :id E1 :type PERSON :label "Scientist" :surface "Scientist")\n'
                        '  (event :id Ev1 :pred test :agent E1 :polarity TRUE)\n'
                        ')'
                    )
                }
            }
        ]
    }
    mock_session.post.return_value = mock_resp

    transducer = UnslothTransducer(
        base_url="http://localhost:8888/v1",
        session=mock_session,
        fallback_to_mock=False,
    )

    result = transducer.transduce("The scientist tested the hypothesis.")
    assert result.chunk_id == "simulated_chunk"
    assert len(result.entities) == 1
    assert result.entities[0].canonical_name == "Scientist"
    assert len(result.events) == 1
    assert result.events[0].predicate == "test"
    assert result.metadata["backend"] == "unsloth"


def test_unsloth_simulated_http_with_code_fences():
    """Verify UnslothTransducer cleans markdown code fences in model output."""
    mock_session = MagicMock(spec=requests.Session)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        "```lisp\n"
                        '(graph (entity :id E1 :type PERSON :label "Bob" :surface "Bob"))\n'
                        "```"
                    )
                }
            }
        ]
    }
    mock_session.post.return_value = mock_resp

    transducer = UnslothTransducer(
        base_url="http://localhost:8888/v1",
        session=mock_session,
        fallback_to_mock=False,
    )

    raw = transducer.transduce_raw("Bob arrived.")
    assert raw.startswith("(graph")
    parsed = parse_sexpr(raw)
    assert parsed.entities[0].canonical_name == "Bob"


# ---------------------------------------------------------------------------
# 6. Asynchronous API Wrapper Tests
# ---------------------------------------------------------------------------

def test_async_transduction():
    """Verify transduce_async and transduce_raw_async execute cleanly with asyncio."""
    mock = MockUnslothTransducer()

    async def run_async():
        res = await mock.transduce_async("Alice observed the reaction.", chunk_id="async_01")
        raw = await mock.transduce_raw_async("Alice observed the reaction.")
        return res, raw

    result, raw_sexpr = asyncio.run(run_async())
    assert isinstance(result, DiscourseExtractionResult)
    assert result.chunk_id == "async_01"
    assert raw_sexpr.startswith("(graph")
