"""Unit tests for QUANTA Reverse Realizers (English NLG, Hungarian Morph, FOLEmitter, CodeEmitter)."""

import pytest
from core.asg import QuantaGraph, QuantaNode
from realizer.english_nlg import EnglishRealizer
from realizer.hungarian_morph import HungarianRealizer
from realizer.fol_emitter import FOLEmitter
from realizer.code_emitter import CodeEmitter


@pytest.fixture
def english_realizer():
    return EnglishRealizer()


@pytest.fixture
def hungarian_realizer():
    return HungarianRealizer()


@pytest.fixture
def fol_emitter():
    return FOLEmitter()


@pytest.fixture
def code_emitter():
    return CodeEmitter()


def test_english_svo_realization(english_realizer):
    # "A dog chased a cat."
    event = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1, "LJB_PU_PAST_TENSE": 1},
        anchor="wn:chase.v.01",
    )
    dog = QuantaNode(
        vector={"TYPE_ANIMATE": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="wn:dog.n.01",
    )
    cat = QuantaNode(
        vector={"TYPE_ANIMATE": 1},
        anchor="wn:cat.n.01",
    )

    graph = QuantaGraph()
    e_cid = graph.add_node(event, set_as_root=True)
    d_cid = graph.add_node(dog)
    c_cid = graph.add_node(cat)

    graph.add_edge(e_cid, "VAL_X1_AGENT", d_cid)
    graph.add_edge(e_cid, "VAL_X2_PATIENT", c_cid)

    text = english_realizer.realize_graph(graph)
    assert "dog" in text.lower()
    assert "chased" in text.lower()
    assert "cat" in text.lower()


def test_english_modal_and_negation(english_realizer):
    # "A person must not touch a rock."
    event = QuantaNode(
        vector={
            "NSM_DO": 1,
            "TYPE_EVENT": 1,
            "EPIST_DEONTIC_OBLIGATION": 1,
            "LJB_NA_NEGATION": 2,
        },
        anchor="wn:touch.v.01",
    )
    person = QuantaNode(
        vector={"TYPE_HUMAN": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="wn:person.n.01",
    )
    rock = QuantaNode(
        vector={"TYPE_INANIMATE_PHYSICAL": 1},
        anchor="wn:rock.n.01",
    )

    graph = QuantaGraph()
    e_cid = graph.add_node(event, set_as_root=True)
    p_cid = graph.add_node(person)
    r_cid = graph.add_node(rock)

    graph.add_edge(e_cid, "VAL_X1_AGENT", p_cid)
    graph.add_edge(e_cid, "VAL_X2_PATIENT", r_cid)

    text = english_realizer.realize_graph(graph)
    assert "must not" in text.lower()
def test_english_canonical_example_a(english_realizer):
    """Verify Phase 8 Item 8A.6: Example A graph -> 'A golden retriever bit the mailman in the garden.'"""
    from parser.nlp_forward import NLPForwardParser
    parser = NLPForwardParser()
    graph = parser.parse_sentence("A golden retriever bit the mailman in the garden.")
    realized = english_realizer.realize_graph(graph)
    assert realized == "A golden retriever bit the mailman in the garden."


def test_english_canonical_example_b(english_realizer):
    """Verify Phase 8 Item 8A.7: Example B graph -> 'The dog did not bite the mailman.'"""
    from parser.nlp_forward import NLPForwardParser
    parser = NLPForwardParser()
    graph = parser.parse_sentence("The dog did not bite the mailman.")
    realized = english_realizer.realize_graph(graph)
    assert realized == "The dog did not bite the mailman."


def test_english_canonical_example_c(english_realizer):
    """Verify Phase 8 Item 8A.8: Example C graph -> 'Did the dog perhaps bite a mailman?'"""
    from parser.nlp_forward import NLPForwardParser
    parser = NLPForwardParser()
    graph = parser.parse_sentence("Did the dog perhaps bite a mailman?")
    realized = english_realizer.realize_graph(graph)
    assert realized == "Did the dog perhaps bite a mailman?"


def test_hungarian_vowel_harmony_and_cases(hungarian_realizer):
    # Back vowels: kutya (dog), ház (house)
    assert hungarian_realizer.get_vowel_harmony("kutya") == "back"
    assert hungarian_realizer.get_vowel_harmony("ház") == "back"

    # Front unrounded: kert (garden), ember (person)
    assert hungarian_realizer.get_vowel_harmony("kert") == "front_unrounded"
    assert hungarian_realizer.get_vowel_harmony("ember") == "front_unrounded"

    # Case suffixes
    assert hungarian_realizer.apply_case_suffix("kutya", "acc") == "kutyát"
    assert hungarian_realizer.apply_case_suffix("ház", "ine") == "házban"
    assert hungarian_realizer.apply_case_suffix("kert", "ine") == "kertben"
    assert hungarian_realizer.apply_case_suffix("ház", "ill") == "házba"
    assert hungarian_realizer.apply_case_suffix("kert", "ill") == "kertbe"


def test_hungarian_sentence_realization(hungarian_realizer):
    # "A kutya kergetett a macskát a kertben."
    event = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1, "LJB_PU_PAST_TENSE": 1},
        anchor="wn:chase.v.01",
    )
    dog = QuantaNode(
        vector={"TYPE_ANIMATE": 1},
        anchor="wn:dog.n.01",
    )
    cat = QuantaNode(
        vector={"TYPE_ANIMATE": 1},
        anchor="wn:cat.n.01",
    )
    garden = QuantaNode(
        vector={"TYPE_SPATIAL_REGION": 1},
        anchor="wn:garden.n.01",
    )

    graph = QuantaGraph()
    e_cid = graph.add_node(event, set_as_root=True)
    d_cid = graph.add_node(dog)
    c_cid = graph.add_node(cat)
    g_cid = graph.add_node(garden)

    graph.add_edge(e_cid, "VAL_X1_AGENT", d_cid)
    graph.add_edge(e_cid, "VAL_X2_PATIENT", c_cid)
    graph.add_edge(e_cid, "VAL_LOCATION_SLOT", g_cid)

    hu_text = hungarian_realizer.realize_graph(graph)
    assert "kutya" in hu_text.lower()
    assert "macskát" in hu_text.lower()
    assert "kertben" in hu_text.lower()
    assert "kergetett" in hu_text.lower()


def test_fol_emitter(fol_emitter):
    # \forall x (Dog(x) -> Animal(x))
    root = QuantaNode(
        vector={"NSM_ALL": 1, "LJB_RO_ALL_QUANT": 1, "LJB_GANAI_IF_THEN": 1},
    )
    pred1 = QuantaNode(
        vector={"TYPE_RELATION_ROLE": 1},
        anchor="wn:dog.n.01",
        literal="Dog",
    )
    arg1 = QuantaNode(literal="x", vector={"GRAPH_VARIABLE_BIND": 1})
    pred2 = QuantaNode(
        vector={"TYPE_RELATION_ROLE": 1},
        anchor="wn:animal.n.01",
        literal="Animal",
    )
    arg2 = QuantaNode(literal="x", vector={"GRAPH_VARIABLE_BIND": 1})

    graph = QuantaGraph()
    r_cid = graph.add_node(root, set_as_root=True)
    p1_cid = graph.add_node(pred1)
    a1_cid = graph.add_node(arg1)
    p2_cid = graph.add_node(pred2)
    a2_cid = graph.add_node(arg2)

    graph.add_edge(p1_cid, "VAL_X1_AGENT", a1_cid)
    graph.add_edge(p2_cid, "VAL_X1_AGENT", a2_cid)
    graph.add_edge(r_cid, "GRAPH_IS_SUB_EXP", p1_cid)
    graph.add_edge(r_cid, "GRAPH_IS_SUB_EXP", p2_cid)

    formula = fol_emitter.emit_formula(graph)
    assert "\\forall x" in formula
    assert "Dog(x)" in formula
    assert "\\rightarrow" in formula
    assert "Animal(x)" in formula


def test_code_emitter(code_emitter):
    # def factorial(n):
    #     if condition:
    #         return True
    root = QuantaNode(
        vector={"GRAPH_FUNCTION_DEF": 1, "TYPE_PROCESS": 1},
        anchor="func:factorial",
    )
    arg_n = QuantaNode(vector={"GRAPH_VARIABLE_BIND": 1}, literal="arg:n")
    if_node = QuantaNode(
        vector={"GRAPH_BRANCH_COND": 3, "GRAPH_BRANCH_THEN": 1},
        edges={},
    )
    ret_node = QuantaNode(
        vector={"GRAPH_RETURN_VALUE": 1},
        literal="1",
    )

    graph = QuantaGraph()
    r_cid = graph.add_node(root, set_as_root=True)
    a_cid = graph.add_node(arg_n)
    if_cid = graph.add_node(if_node)
    ret_cid = graph.add_node(ret_node)

    graph.add_edge(r_cid, "VAL_X1_AGENT", a_cid)
    graph.add_edge(r_cid, "GRAPH_IS_SUB_EXP", if_cid)
    graph.add_edge(if_cid, "GRAPH_BRANCH_THEN", ret_cid)

    code = code_emitter.emit_code(graph)
    assert "def factorial(n):" in code
    assert "if " in code
    assert "return 1" in code
