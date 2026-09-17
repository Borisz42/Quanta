"""Unit and integration tests for QUANTA S-Expression Lexer, Parser, AST Converter, and ASG Compiler Bridge.

Verifies:
1. SExprLexer tokenization, string escapes, number formats, booleans, and comment stripping.
2. SExprParser recursive-descent structure, unclosed parenthesis diagnostics, and error coordinates.
3. SExprASTConverter lossless round-trip fidelity with DiscourseExtractionResult.
4. Gold-standard Eleanor Vance fixture round-trip (parse_sexpr(to_sexpr(res)) == res).
5. parse_to_asg direct compilation bridge producing 1024-D QuantaGraph with passing Clingo validation.
6. Formal GBNF grammar specification presence and structure.
"""

from pathlib import Path
import pytest

from core.asg import QuantaGraph
from parser.asg_compiler import ASGCompilationError, ASGCompiler
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
    ExtractedTimeInterval,
)
from parser.sexpr_parser import (
    SExprASTConverter,
    SExprAtom,
    SExprLexer,
    SExprList,
    SExprParser,
    SExprSyntaxError,
    SExprToken,
    SExprTokenType,
    parse_sexpr,
    parse_to_asg,
    to_sexpr,
)
from parser.transducer import CANONICAL_ELEANOR_VANCE_FIXTURE


# ---------------------------------------------------------------------------
# 1. Lexer Unit Tests
# ---------------------------------------------------------------------------

def test_lexer_basic_tokens():
    """Verify lexer extracts parens, keywords, symbols, strings, numbers, and booleans."""
    source = '(graph :id "test_01" :count 42 :ratio -3.14 :flag TRUE :neg FALSE sym)'
    lexer = SExprLexer(source)
    tokens = lexer.tokenize()

    types = [t.type for t in tokens]
    expected_types = [
        SExprTokenType.LPAREN,
        SExprTokenType.SYMBOL,
        SExprTokenType.KEYWORD,
        SExprTokenType.STRING,
        SExprTokenType.KEYWORD,
        SExprTokenType.NUMBER,
        SExprTokenType.KEYWORD,
        SExprTokenType.NUMBER,
        SExprTokenType.KEYWORD,
        SExprTokenType.BOOLEAN,
        SExprTokenType.KEYWORD,
        SExprTokenType.BOOLEAN,
        SExprTokenType.SYMBOL,
        SExprTokenType.RPAREN,
        SExprTokenType.EOF,
    ]
    assert types == expected_types
    assert tokens[3].value == "test_01"
    assert tokens[5].value == 42
    assert abs(tokens[7].value - (-3.14)) < 1e-5
    assert tokens[9].value is True
    assert tokens[11].value is False
    assert tokens[12].value == "sym"


def test_lexer_scientific_notation():
    """Verify lexer parses numbers in scientific notation with or without decimals."""
    source = "(vals 1e5 -2E-3 3.14e+2 -4.5E-6 0)"
    tokens = SExprLexer(source).tokenize()
    numbers = [t.value for t in tokens if t.type == SExprTokenType.NUMBER]
    assert numbers == [100000.0, -0.002, 314.0, -4.5e-6, 0]
    assert isinstance(numbers[0], float)
    assert isinstance(numbers[4], int)


def test_lexer_string_escapes():
    """Verify string literals correctly handle escapes including unicode."""
    source = r'("hello\nworld" "tab\tseparated" "quote\"inner\"" "escaped\\slash" "unicode \u0041")'
    lexer = SExprLexer(source)
    tokens = lexer.tokenize()

    str_tokens = [t for t in tokens if t.type == SExprTokenType.STRING]
    assert len(str_tokens) == 5
    assert str_tokens[0].value == "hello\nworld"
    assert str_tokens[1].value == "tab\tseparated"
    assert str_tokens[2].value == 'quote"inner"'
    assert str_tokens[3].value == "escaped\\slash"
    assert str_tokens[4].value == "unicode A"


def test_lexer_comments_and_whitespace():
    """Verify comments starting with ; or ;; and diverse whitespace are ignored."""
    source = """
    ;; Top-level header comment
    (graph
      ; Inline clause comment
      :id "c1" ; trailing comment
      ;; another comment
      :val 100
    )
    """
    lexer = SExprLexer(source)
    tokens = lexer.tokenize()

    symbols_and_kws = [t.value for t in tokens if t.type in (SExprTokenType.KEYWORD, SExprTokenType.SYMBOL)]
    assert "graph" in symbols_and_kws
    assert ":id" in symbols_and_kws
    assert ":val" in symbols_and_kws


def test_lexer_unterminated_string():
    """Verify lexer raises SExprSyntaxError on unterminated strings with line/col coordinates."""
    source = '(entity :label "Dr. Eleanor Vance)'
    lexer = SExprLexer(source)
    with pytest.raises(SExprSyntaxError) as exc_info:
        lexer.tokenize()
    assert "Unterminated string literal" in str(exc_info.value)
    assert exc_info.value.line == 1
    assert exc_info.value.column == 16


# ---------------------------------------------------------------------------
# 2. Parser Unit Tests & Syntax Diagnostics
# ---------------------------------------------------------------------------

def test_parser_nested_lists():
    """Verify recursive-descent parser correctly nests SExprList trees."""
    source = "(a (b (c 1) (d 2)))"
    lexer = SExprLexer(source)
    parser = SExprParser(lexer.tokenize())
    ast = parser.parse()

    assert isinstance(ast, SExprList)
    assert ast.head_symbol() == "a"
    assert len(ast.elements) == 2
    sub = ast.elements[1]
    assert isinstance(sub, SExprList)
    assert sub.head_symbol() == "b"


def test_parser_unclosed_parenthesis_error():
    """Verify informative syntax error when an opening parenthesis is never closed."""
    source = "(graph\n  (entity :id E1"
    lexer = SExprLexer(source)
    parser = SExprParser(lexer.tokenize())
    with pytest.raises(SExprSyntaxError) as exc_info:
        parser.parse()
    assert "Unclosed parenthesis" in str(exc_info.value)
    # The innermost unclosed parenthesis was opened at line 2, col 3
    assert exc_info.value.line == 2
    assert exc_info.value.column == 3


def test_parser_unexpected_closing_paren():
    """Verify error on stray closing parenthesis."""
    source = "(graph :id test))"
    lexer = SExprLexer(source)
    parser = SExprParser(lexer.tokenize())
    with pytest.raises(SExprSyntaxError) as exc_info:
        parser.parse()
    assert "Unexpected token ')'" in str(exc_info.value) or "closing parenthesis" in str(exc_info.value)


def test_parser_empty_input():
    """Verify error on empty input."""
    lexer = SExprLexer("   \n ;; only comments\n  ")
    parser = SExprParser(lexer.tokenize())
    with pytest.raises(SExprSyntaxError) as exc_info:
        parser.parse()
    assert "Empty S-expression input" in str(exc_info.value)


def test_parser_trailing_tokens():
    """Verify error when extra tokens appear after the root S-expression."""
    source = "(graph :id test) extra"
    lexer = SExprLexer(source)
    parser = SExprParser(lexer.tokenize())
    with pytest.raises(SExprSyntaxError) as exc_info:
        parser.parse()
    assert "Unexpected token 'extra' after root" in str(exc_info.value)


def test_parser_non_list_root():
    """Verify error when root expression is an atom instead of a list."""
    source = '"just a string"'
    lexer = SExprLexer(source)
    parser = SExprParser(lexer.tokenize())
    with pytest.raises(SExprSyntaxError) as exc_info:
        parser.parse()
    assert "Root S-expression must be a parenthesized list" in str(exc_info.value)


def test_event_val_polarity_alias():
    """Verify :val FALSE correctly sets polarity=False on extracted event."""
    source = '(graph (entity :id E1 :type PERSON :label "Bob" :surface "Bob") (event :id Ev1 :pred sleep :agent E1 :val FALSE))'
    res = parse_sexpr(source)
    assert len(res.events) == 1
    assert res.events[0].polarity is False
    assert res.events[0].val == "FALSE"


def test_structured_temporal_interval_parsing():
    """Verify structured :time (interval :start ... :end ... :duration ...) parses into ExtractedTimeInterval."""
    source = """
    (graph
      (entity :id e1 :type HUMAN :label "Eleanor Vance" :surface "Dr. Eleanor Vance")
      (event :id ev1 :pred PROFESSOR_OF :agent e1
             :time (interval :start 2018 :end nil :duration 6) :val TRUE)
    )
    """
    res = parse_sexpr(source)
    assert len(res.events) == 1
    ev = res.events[0]
    assert ev.time_interval is not None
    assert isinstance(ev.time_interval, ExtractedTimeInterval)
    assert ev.time_interval.start == 2018
    assert ev.time_interval.end is None
    assert ev.time_interval.duration == 6
    assert ev.val == "TRUE"
    assert ev.polarity is True


def test_belnap_logic_states_parsing():
    """Verify Belnap 4-valued logic states (TRUE, FALSE, UNKNOWN, IRRELEVANT) in :val."""
    source = """
    (graph
      (entity :id e1 :type HUMAN :label "Alice" :surface "Alice")
      (event :id ev1 :pred hypothesize :agent e1 :val UNKNOWN)
      (event :id ev2 :pred ignore :agent e1 :val IRRELEVANT)
    )
    """
    res = parse_sexpr(source)
    assert len(res.events) == 2
    assert res.events[0].val == "UNKNOWN"
    assert res.events[0].polarity is True
    assert res.events[1].val == "IRRELEVANT"
    assert res.events[1].polarity is True


def test_strict_keyword_validation_error():
    """Verify strict=True raises SExprSyntaxError on unrecognized keywords with line/col info."""
    source = '(graph (entity :id e1 :type PERSON :label "Alice" :surface "Alice" :bogus_prop 123))'
    with pytest.raises(SExprSyntaxError) as exc_info:
        parse_sexpr(source, strict=True)
    assert "Unknown keyword ':bogus_prop' in entity clause" in str(exc_info.value)
    assert exc_info.value.line == 1

    # In non-strict mode, it ignores unrecognized keywords without error
    res = parse_sexpr(source, strict=False)
    assert len(res.entities) == 1


def test_strict_clause_head_validation_error():
    """Verify strict=True raises SExprSyntaxError on unrecognized clause heads."""
    source = '(graph (invalid_head :id e1))'
    with pytest.raises(SExprSyntaxError) as exc_info:
        parse_sexpr(source, strict=True)
    assert "Unknown clause head 'invalid_head' in graph" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. AST Converter & Round-Trip Tests
# ---------------------------------------------------------------------------

def test_eleanor_vance_gold_standard_roundtrip():
    """Verify full round-trip fidelity: parse_sexpr(to_sexpr(res)) == res on gold standard."""
    original = CANONICAL_ELEANOR_VANCE_FIXTURE

    # 1. Serialize to S-expression
    sexpr_text = to_sexpr(original)
    assert sexpr_text.startswith("(graph")
    assert '(entity :id E1 :type PERSON :label "Dr. Eleanor Vance"' in sexpr_text
    assert '(event :id Ev1 :pred isolate' in sexpr_text
    assert '(relation :type TEMP_ALLEN_MEETS' in sexpr_text

    # 2. Parse back to DiscourseExtractionResult
    roundtrip = parse_sexpr(sexpr_text)

    # 3. Assert exact equality
    assert roundtrip.chunk_id == original.chunk_id
    assert len(roundtrip.entities) == len(original.entities)
    assert len(roundtrip.events) == len(original.events)
    assert len(roundtrip.relations) == len(original.relations)
    assert len(roundtrip.propositions) == len(original.propositions)

    for r_ent, o_ent in zip(roundtrip.entities, original.entities):
        assert r_ent.id == o_ent.id
        assert r_ent.canonical_name == o_ent.canonical_name
        assert r_ent.category == o_ent.category
        assert r_ent.surface_aliases == o_ent.surface_aliases
        assert r_ent.properties == o_ent.properties

    for r_ev, o_ev in zip(roundtrip.events, original.events):
        assert r_ev.id == o_ev.id
        assert r_ev.predicate == o_ev.predicate
        assert r_ev.agent_id == o_ev.agent_id
        assert r_ev.patient_id == o_ev.patient_id
        assert r_ev.location_id == o_ev.location_id
        assert r_ev.temporal_anchor == o_ev.temporal_anchor
        assert r_ev.tense == o_ev.tense
        assert r_ev.polarity == o_ev.polarity
        assert r_ev.raw_text == o_ev.raw_text

    for r_rel, o_rel in zip(roundtrip.relations, original.relations):
        assert r_rel.relation_type == o_rel.relation_type
        assert r_rel.source_id == o_rel.source_id
        assert r_rel.target_id == o_rel.target_id
        assert r_rel.mechanism == o_rel.mechanism
        assert abs(r_rel.confidence - o_rel.confidence) < 1e-4

    for r_prop, o_prop in zip(roundtrip.propositions, original.propositions):
        assert r_prop.id == o_prop.id
        assert r_prop.claim_text == o_prop.claim_text
        assert r_prop.epistemic_status == o_prop.epistemic_status
        assert r_prop.source_agent_id == o_prop.source_agent_id

    # Strict object equality
    assert roundtrip == original


def test_empty_graph_roundtrip():
    """Verify empty extraction result roundtrips cleanly."""
    empty = DiscourseExtractionResult(chunk_id="chunk_empty")
    sexpr = to_sexpr(empty)
    parsed = parse_sexpr(sexpr)
    assert parsed == empty


def test_custom_extraction_with_props_and_instruments():
    """Verify entities with complex properties and events with instruments."""
    res = DiscourseExtractionResult(
        chunk_id="chunk_custom",
        entities=[
            ExtractedEntity(
                id="E1",
                canonical_name="Scientist",
                category="PERSON",
                surface_aliases=["The researcher"],
                properties={"clearance": "level_5", "verified": True},
            ),
            ExtractedEntity(
                id="E2",
                canonical_name="spectrometer",
                category="INSTRUMENT",
                surface_aliases=["mass spec"],
                properties={"model": "Q-Exactive"},
            ),
        ],
        events=[
            ExtractedEvent(
                id="Ev1",
                predicate="calibrate",
                agent_id="E1",
                instrument_id="E2",
                temporal_anchor="08:00 AM",
                tense="PAST",
                polarity=True,
                raw_text="Scientist calibrated spectrometer at 08:00 AM.",
            )
        ],
        relations=[
            ExtractedRelation(
                relation_type="CAUSAL_MECHANISM_LINK",
                source_id="Ev1",
                target_id="Ev1",
                mechanism="auto-diagnostic sequence",
                confidence=0.98,
            )
        ],
        propositions=[
            ExtractedProposition(
                id="P1",
                claim_text="Lattice stability verified",
                epistemic_status="FACT",
                source_agent_id="E1",
            )
        ],
    )

    sexpr = to_sexpr(res)
    parsed = parse_sexpr(sexpr)
    assert parsed == res


def test_nil_foreign_keys_handled_cleanly():
    """Verify that events with explicit nil or omitted foreign keys parse to None."""
    sexpr = """
    (graph
      (entity :id E1 :type PERSON :label "Alice" :surface "Alice")
      (event :id Ev1 :pred think :agent E1 :patient nil :location nil :theme nil :polarity TRUE)
    )
    """
    res = parse_sexpr(sexpr)
    assert len(res.events) == 1
    ev = res.events[0]
    assert ev.agent_id == "E1"
    assert ev.patient_id is None
    assert ev.location_id is None
    assert ev.theme_id is None


# ---------------------------------------------------------------------------
# 4. Direct ASG Compilation Bridge Tests
# ---------------------------------------------------------------------------

def test_parse_to_asg_eleanor_vance():
    """Verify parse_to_asg compiles Eleanor Vance S-expression to valid QuantaGraph."""
    sexpr_text = to_sexpr(CANONICAL_ELEANOR_VANCE_FIXTURE)
    compiler = ASGCompiler()
    graph = parse_to_asg(sexpr_text, compiler=compiler, validate=True)

    assert isinstance(graph, QuantaGraph)
    # Exactly 5 entity nodes + 6 event nodes = 11 nodes
    assert len(graph.nodes) == 11

    # Verify BLAKE3 CIDs
    for cid, node in graph.nodes.items():
        assert len(cid) == 64
        assert cid == node.compute_cid()

    # Verify Clingo ASP validation
    assert hasattr(graph, "validation")
    assert graph.validation.is_valid is True
    assert len(graph.validation.errors) == 0


def test_parse_to_asg_foreign_key_error():
    """Verify that an S-expression referencing a non-existent entity raises ASGCompilationError."""
    sexpr = """
    (graph
      (entity :id E1 :type PERSON :label "Alice" :surface "Alice")
      (event :id Ev1 :pred observe :agent E999 :tense PAST :polarity TRUE)
    )
    """
    with pytest.raises(ASGCompilationError) as exc_info:
        parse_to_asg(sexpr, validate=False)
    assert "Foreign-key validation failed" in str(exc_info.value)
    assert "E999" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. GBNF Grammar Specification Verification
# ---------------------------------------------------------------------------

def test_gbnf_grammar_file_exists_and_contains_rules():
    """Verify data/grammar/quanta_asg.gbnf exists and defines formal rules."""
    gbnf_path = Path(__file__).parent.parent / "data" / "grammar" / "quanta_asg.gbnf"
    assert gbnf_path.exists(), f"GBNF grammar missing at {gbnf_path}"

    content = gbnf_path.read_text(encoding="utf-8")
    assert "root ::=" in content
    assert "entity_clause ::=" in content
    assert "event_clause ::=" in content
    assert "relation_clause ::=" in content
    assert "prop_clause ::=" in content
    assert "entity_type ::=" in content
    assert "rel_type ::=" in content
    assert "time_interval ::=" in content
    assert "time_bound ::=" in content
    assert "belnap_value ::=" in content
    assert "string ::=" in content
