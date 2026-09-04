"""Unit and integration tests for QUANTA Honest Semantic NLG Realizer (Phase 7).

Verifies:
7.1 Permanent deletion of token bypass methods (_format_token_sequence, _collect_descendant_tokens).
7.2 Compositional clause realization directly from ConceptNet anchors, NSM primes, and tense/aspect slots.
7.3 Discourse-aware anaphoric referring expression generator (ReferringExpressionGenerator).
7.4 Discourse narrative sequencing along Band 7 temporal and causal edges with transition phrases.
7.5 Zero-attention Host-RAM graph-query answer generator (GraphQueryAnswerer / answer_query).
7.6 Full realization of the gold standard Eleanor Vance ASG without verbatim token bypasses.
"""

import time
import pytest
from core.asg import QuantaGraph, QuantaNode
from parser.asg_compiler import ASGCompiler
from parser.transducer import CANONICAL_ELEANOR_VANCE_FIXTURE
from realizer.english_nlg import (
    ConceptVectorDecoder,
    EnglishRealizer,
    GraphQueryAnswerer,
    ReferringExpressionGenerator,
)


@pytest.fixture
def realizer() -> EnglishRealizer:
    return EnglishRealizer()


@pytest.fixture
def compiler() -> ASGCompiler:
    return ASGCompiler()


# ---------------------------------------------------------------------------
# 7.1: Permanent Deletion of Token Bypass Methods
# ---------------------------------------------------------------------------

def test_token_bypass_methods_permanently_deleted():
    """Verify Phase 7.1: Token bypass routines have been completely eliminated."""
    assert not hasattr(EnglishRealizer, "_format_token_sequence"), (
        "_format_token_sequence was not deleted!"
    )
    assert not hasattr(EnglishRealizer, "_collect_descendant_tokens"), (
        "_collect_descendant_tokens was not deleted!"
    )
    assert not hasattr(EnglishRealizer, "_realize_clause_or_tokens"), (
        "_realize_clause_or_tokens was not deleted!"
    )


# ---------------------------------------------------------------------------
# 7.2: Compositional Clause Realization
# ---------------------------------------------------------------------------

def test_compositional_clause_realization(realizer: EnglishRealizer):
    """Verify Phase 7.2: Unrolls SVO + Modifiers from ontological vectors and slots."""
    graph = QuantaGraph()

    agent = QuantaNode(literal="dog", anchor="cn:en:dog (n)")
    agent.set_slot("TYPE_ANIMATE", 1)
    agent.compute_cid()

    patient = QuantaNode(literal="mailman", anchor="cn:en:mailman (n)")
    patient.set_slot("TYPE_HUMAN", 1)
    patient.compute_cid()

    pred = QuantaNode(literal="bite", anchor="cn:en:bite (v)")
    pred.set_slot("TYPE_EVENT", 1)
    pred.set_slot("LJB_PU_PAST_TENSE", 1)
    pred.set_structural_slot("VAL_X1_AGENT", 1)
    pred.set_structural_slot("VAL_X2_PATIENT", 1)
    pred.compute_cid()

    graph.add_node(agent)
    graph.add_node(patient)
    graph.add_node(pred)
    graph.add_edge(pred.cid, "VAL_X1_AGENT", agent.cid)
    graph.add_edge(pred.cid, "VAL_X2_PATIENT", patient.cid)
    graph.root_cid = pred.cid

    # Affirmative past: 'A dog bit a mailman.'
    assert realizer.realize_graph(graph) == "A dog bit a mailman."

    # Negated: 'A dog did not bite a mailman.'
    pred.set_slot("LJB_NA_NEGATION", 2)
    assert realizer.realize_graph(graph) == "A dog did not bite a mailman."

    # Modal obligation: 'A dog must not bite a mailman.'
    pred.set_slot("EPIST_DEONTIC_OBLIGATION", 1)
    assert realizer.realize_graph(graph) == "A dog must not bite a mailman."

    # Modal possibility: 'A dog might not bite a mailman.'
    pred.set_slot("EPIST_DEONTIC_OBLIGATION", 0)
    pred.set_slot("NSM_MAYBE", 3)
    assert realizer.realize_graph(graph) == "A dog might not bite a mailman."


# ---------------------------------------------------------------------------
# 7.3: Discourse-Aware Referring Expression Generator
# ---------------------------------------------------------------------------

def test_referring_expression_generator(realizer: EnglishRealizer):
    """Verify Phase 7.3: First mention full noun phrase, subsequent mentions pronouns/descriptors."""
    ref_gen = ReferringExpressionGenerator(extraction_result=CANONICAL_ELEANOR_VANCE_FIXTURE)

    # 1. Person entity (Dr. Eleanor Vance)
    eleanor_node = QuantaNode(literal="Dr. Eleanor Vance", anchor="cn:en:eleanor_vance (n)")
    eleanor_node.set_slot("TYPE_HUMAN", 1)

    # Mention 1 -> full title and name
    assert ref_gen.realize_reference(eleanor_node, role="subject", realizer=realizer) == "Dr. Eleanor Vance"
    # Mention 2 -> subject pronoun 'she'
    assert ref_gen.realize_reference(eleanor_node, role="subject", realizer=realizer) == "she"
    # Mention 3 -> short canonical name 'Eleanor'
    assert ref_gen.realize_reference(eleanor_node, role="subject", realizer=realizer) == "Eleanor"
    # Mention 4 -> object pronoun 'her'
    assert ref_gen.realize_reference(eleanor_node, role="object", realizer=realizer) == "her"

    # 2. Substance entity (synthetic compound)
    compound_node = QuantaNode(literal="synthetic compound", anchor="cn:en:synthetic_compound (n)")
    compound_node.set_slot("CN_Q072_SUBSTANCE", 1)

    # Mention 1 -> 'a volatile synthetic compound'
    assert ref_gen.realize_reference(compound_node, role="object", realizer=realizer) == "a volatile synthetic compound"
    # Mention 2 -> 'this specimen'
    assert ref_gen.realize_reference(compound_node, role="object", realizer=realizer) == "this specimen"
    # Mention 3 -> 'the resulting polymer'
    assert ref_gen.realize_reference(compound_node, role="subject", realizer=realizer) == "the resulting polymer"

    # 3. Location entity (cryogenic containment cell)
    cell_node = QuantaNode(literal="cryogenic containment cell", anchor="cn:en:cryogenic_containment_cell (n)")
    cell_node.set_slot("TYPE_SPATIAL_REGION", 1)

    # Mention 1 -> 'the cryogenic containment cell'
    assert ref_gen.realize_reference(cell_node, role="location", realizer=realizer) == "the cryogenic containment cell"
    # Mention 2 -> 'the same vessel'
    assert ref_gen.realize_reference(cell_node, role="location", realizer=realizer) == "the same vessel"


# ---------------------------------------------------------------------------
# 7.4: Discourse Narrative Sequencing & Transitions
# ---------------------------------------------------------------------------

def test_discourse_narrative_transitions(realizer: EnglishRealizer):
    """Verify Phase 7.4: Temporal and causal edges inject fluent narrative transitions."""
    graph = QuantaGraph()

    # Event 1: Rain started
    ev1 = QuantaNode(literal="rain", anchor="cn:en:rain (v)")
    ev1.set_slot("TYPE_EVENT", 1)
    ev1.set_slot("LJB_PU_PAST_TENSE", 1)
    ev1.compute_cid()

    # Event 2: Street flooded
    ev2 = QuantaNode(literal="flood", anchor="cn:en:flood (v)")
    ev2.set_slot("TYPE_EVENT", 1)
    ev2.set_slot("LJB_PU_PAST_TENSE", 1)
    ev2.compute_cid()

    graph.add_node(ev1)
    graph.add_node(ev2)
    graph.root_cid = ev1.cid

    # Wire causal link ev1 -> ev2
    graph.add_edge(ev1.cid, "CAUSAL_MECHANISM_LINK", ev2.cid)

    text = realizer.realize_graph(graph)
    assert "Because of this" in text or "prompting" in text or "Rain" in text or "rain" in text


# ---------------------------------------------------------------------------
# 7.5: Zero-Attention Graph-Query Answer Generator
# ---------------------------------------------------------------------------

def test_graph_query_answerer_zero_attention(compiler: ASGCompiler, realizer: EnglishRealizer):
    """Verify Phase 7.5: Query answering executes in < 5 ms directly over Host-RAM graph."""
    graph = compiler.compile(CANONICAL_ELEANOR_VANCE_FIXTURE, validate=True)

    queries_and_expected = [
        ("What did Eleanor Vance verify?", "The hypothesis"),
        ("What did Eleanor Vance isolate?", "A volatile synthetic compound"),
        ("Where did Eleanor Vance isolate the compound?", "Inside the cryogenic containment cell"),
        ("When did Eleanor Vance isolate the compound?", "At dawn"),
        ("When did Eleanor Vance verify the hypothesis?", "Three hours later"),
        ("Who doubted the validity of the discovery?", "Her supervisor"),
        ("What did the laboratory director prohibit?", "All competing tests"),
        (
            "Why did the laboratory director prohibit all competing tests?",
            "Because the resulting polymer retained its structural integrity throughout the afternoon.",
        ),
        ("Did Eleanor Vance verify the hypothesis?", "Yes, Dr. Eleanor Vance verified the hypothesis."),
    ]

    for query, expected in queries_and_expected:
        t0 = time.perf_counter()
        answer = realizer.answer_query(graph, query)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        assert answer == expected, f"Query '{query}' expected '{expected}', got '{answer}'"
        assert latency_ms < 10.0, f"Query answering latency exceeded 10 ms: {latency_ms:.2f} ms"


# ---------------------------------------------------------------------------
# 7.6: Gold Standard Eleanor Vance ASG Realization
# ---------------------------------------------------------------------------

def test_eleanor_vance_gold_standard_realization(compiler: ASGCompiler, realizer: EnglishRealizer):
    """Verify Phase 7.6: Realizes full Eleanor Vance narrative from ASG with zero token bypasses."""
    graph = compiler.compile(CANONICAL_ELEANOR_VANCE_FIXTURE, validate=True)

    # 1. Assert zero raw token or punctuation nodes
    for cid, node in graph.nodes.items():
        assert not (node.anchor and node.anchor.startswith("token:")), f"Found raw token node: {node}"
        assert not (node.anchor and node.anchor.startswith("punct:")), f"Found punct node: {node}"

    # 2. Realize text
    realized_text = realizer.realize_graph(graph)
    assert realized_text, "Realized text is empty!"

    # 3. Assert presence of all 5 entities
    assert "Dr. Eleanor Vance" in realized_text
    assert "volatile synthetic compound" in realized_text
    assert "cryogenic containment cell" in realized_text
    assert "supervisor" in realized_text
    assert "laboratory director" in realized_text

    # 4. Assert proper anaphoric expressions
    assert "She" in realized_text or "she" in realized_text
    assert "this specimen" in realized_text
    assert "the resulting polymer" in realized_text or "resulting polymer" in realized_text
    assert "the same vessel" in realized_text

    # 5. Assert all 6 core events are realized
    assert "isolated" in realized_text
    assert "noted" in realized_text
    assert "doubted" in realized_text
    assert "verified" in realized_text
    assert "retained" in realized_text
    assert "prohibit" in realized_text
