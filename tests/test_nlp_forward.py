"""Tests for NLPForwardParser and First-Order Logic formula parser."""

import pytest
from parser.nlp_forward import NLPForwardParser
from parser.fol_parser import FOLParser
from core.types import QuaternaryValue


@pytest.fixture(scope="module")
def nlp_parser():
    return NLPForwardParser()


@pytest.fixture(scope="module")
def fol_parser():
    return FOLParser()


def test_action_sentence_parsing(nlp_parser):
    # "A brown dog chased the mailman into the garden."
    graph = nlp_parser.parse_sentence("A brown dog chased the mailman into the garden.")
    
    assert graph.root is not None
    root = graph.root
    
    # Root event slots
    assert root.get_slot("TYPE_EVENT") == QuaternaryValue.TRUE
    assert root.get_slot("NSM_MOVE") == QuaternaryValue.TRUE
    assert root.get_slot("LJB_PU_PAST_TENSE") == QuaternaryValue.TRUE
    
    # Graph edges
    assert "VAL_X1_AGENT" in root.edges
    assert "VAL_X2_PATIENT" in root.edges
    assert "VAL_X3_DESTINATION" in root.edges
    
    # Agent node (dog)
    agent_cid = root.edges["VAL_X1_AGENT"][0]
    agent_node = graph.get_node(agent_cid)
    assert agent_node is not None
    assert agent_node.get_slot("TYPE_ANIMATE") == QuaternaryValue.TRUE
    assert agent_node.get_slot("ROLE_AGENT_CAPABLE") == QuaternaryValue.TRUE


def test_mental_predicate_parsing(nlp_parser):
    # "Alice believes that the earth is round."
    graph = nlp_parser.parse_sentence("Alice thinks about astronomy.")
    assert graph.root is not None
    assert graph.root.get_slot("NSM_THINK") == QuaternaryValue.TRUE


def test_fol_implication_parsing(fol_parser):
    # "\forall x (Human(x) \rightarrow Mortal(x))"
    graph = fol_parser.parse_formula(r"\forall x (Human(x) \rightarrow Mortal(x))")
    assert graph.root is not None
    root = graph.root

    assert root.get_slot("NSM_ALL") == QuaternaryValue.TRUE
    assert root.get_slot("LJB_RO_ALL_QUANT") == QuaternaryValue.TRUE
    assert root.get_slot("LJB_GANAI_IF_THEN") == QuaternaryValue.TRUE
    assert root.get_slot("TYPE_PROPOSITION") == QuaternaryValue.TRUE
    assert root.get_slot("EPIST_DEDUCTIVE_INFERENCE") == QuaternaryValue.TRUE


def test_negated_past_sentence_parsing(nlp_parser):
    # "A person did not see a cat."
    graph = nlp_parser.parse_sentence("A person did not see a cat.")
    assert graph.root is not None
    root = graph.root

    assert root.literal == "did not see"
    assert root.get_slot("NSM_SEE") == QuaternaryValue.FALSE
    assert root.get_slot("LJB_NA_NEGATION") == QuaternaryValue.FALSE
    assert root.get_slot("LJB_PU_PAST_TENSE") == QuaternaryValue.TRUE
    assert root.get_slot("NSM_DIE") == QuaternaryValue.IRRELEVANT
    assert "VAL_X1_AGENT" in root.edges
    assert "VAL_X2_PATIENT" in root.edges


def test_example_a_forward_parsing(nlp_parser):
    """Verify Phase 6 Item 6B.9: Full forward parse test for Example A."""
    sentence = "A golden retriever bit the mailman in the garden."
    graph = nlp_parser.parse_sentence(sentence)
    assert graph.root is not None
    root = graph.root

    # Root node verification
    assert root.get_slot("NSM_DO") == 1
    assert root.get_slot("NSM_TOUCH") == 1
    assert root.get_slot("NSM_TRUE") == 1
    assert root.get_slot("VAL_X1_AGENT") == 1
    assert root.get_slot("VAL_X2_PATIENT") == 1
    assert root.get_slot("VAL_LOCATION_SLOT") == 1
    assert root.get_slot("LJB_PU_PAST_TENSE") == 1
    assert root.get_slot("GRAPH_ROOT_NODE") == 1
    assert root.get_slot("TYPE_EVENT") == 1
    assert root.get_slot("MODALITY_LITERAL") == 1
    assert root.get_slot("WN_ACT_ACTION") == 1
    assert root.get_slot("EPIST_DIRECT_OBSERVATION") == 1

    # Child nodes verification
    agent_children = graph.get_children(root.cid, relation="VAL_X1_AGENT")
    assert len(agent_children) == 1
    agent = agent_children[0]
    assert agent.get_slot("NSM_ONE") == 1
    assert agent.get_slot("VAL_X1_AGENT") == 1
    assert agent.get_slot("GRAPH_LEAF") == 1
    assert agent.get_slot("TYPE_ANIMATE") == 1
    assert agent.get_slot("ROLE_AGENT_CAPABLE") == 1
    assert agent.get_slot("ROLE_MOVEABLE") == 1

    patient_children = graph.get_children(root.cid, relation="VAL_X2_PATIENT")
    assert len(patient_children) == 1
    patient = patient_children[0]
    assert patient.get_slot("NSM_THIS") == 1
    assert patient.get_slot("VAL_X2_PATIENT") == 1
    assert patient.get_slot("GRAPH_LEAF") == 1
    assert patient.get_slot("TYPE_HUMAN") == 1
    assert patient.get_slot("ROLE_COMMUNICATOR") == 1

    location_children = graph.get_children(root.cid, relation="VAL_LOCATION_SLOT")
    assert len(location_children) == 1
    loc = location_children[0]
    assert loc.get_slot("NSM_THIS") == 1
    assert loc.get_slot("NSM_INSIDE") == 1
    assert loc.get_slot("VAL_LOCATION_SLOT") == 1
    assert loc.get_slot("TYPE_SPATIAL_REGION") == 1
    assert loc.get_slot("SPATIAL_RCC_NON_TANG_PART") == 1

    # Whole-tree proposition vector verification
    v_tree = graph.to_proposition_vector()
    assert v_tree["NSM_DO"] == 1
    assert v_tree["NSM_TOUCH"] == 1
    assert v_tree["NSM_ONE"] == 1
    assert v_tree["NSM_THIS"] == 1
    assert v_tree["VAL_X1_AGENT"] == 1
    assert v_tree["VAL_X2_PATIENT"] == 1
    assert v_tree["VAL_LOCATION_SLOT"] == 1
    assert v_tree["TYPE_EVENT"] == 1
    assert v_tree["TYPE_ANIMATE"] == 1
    assert v_tree["TYPE_HUMAN"] == 1
    assert v_tree["SPATIAL_RCC_NON_TANG_PART"] == 1


def test_example_b_forward_parsing(nlp_parser):
    """Verify Phase 6 Item 6B.10: Full forward parse test for Example B."""
    sentence = "The dog did not bite the mailman."
    graph = nlp_parser.parse_sentence(sentence)
    assert graph.root is not None
    root = graph.root

    # Explicit negation on root node
    assert root.get_slot("NSM_DO") == 2
    assert root.get_slot("NSM_TOUCH") == 2
    assert root.get_slot("LJB_NA_NEGATION") == 2
    assert root.get_slot("LJB_PU_PAST_TENSE") == 1
    assert root.get_slot("GRAPH_ROOT_NODE") == 1

    # Child nodes
    agent_children = graph.get_children(root.cid, relation="VAL_X1_AGENT")
    assert len(agent_children) == 1
    assert agent_children[0].get_slot("NSM_THIS") == 1
    assert agent_children[0].get_slot("GRAPH_LEAF") == 1

    patient_children = graph.get_children(root.cid, relation="VAL_X2_PATIENT")
    assert len(patient_children) == 1
    assert patient_children[0].get_slot("NSM_THIS") == 1
    assert patient_children[0].get_slot("GRAPH_LEAF") == 1

    # Whole-tree proposition vector
    v_tree = graph.to_proposition_vector()
    assert v_tree["NSM_DO"] == 2
    assert v_tree["NSM_TOUCH"] == 2
    assert v_tree["LJB_NA_NEGATION"] == 2
    assert v_tree["NSM_THIS"] == 1


def test_example_c_uncertainty_question_parsing(nlp_parser):
    """Verify Phase 6 Item 6B.11: Canonical Example C forward parse: 'Did the dog perhaps bite a mailman?'."""
    graph = nlp_parser.parse_sentence("Did the dog perhaps bite a mailman?")
    assert graph.root is not None
    root = graph.root

    # Verifies epistemic uncertainty / query slot values on root node
    assert root.get_slot("GRAPH_QUERY_TARGET") == 3
    assert root.get_slot("NSM_DO") == 3
    assert root.get_slot("NSM_TOUCH") == 3
    assert root.get_slot("NSM_MAYBE") == 3
    assert root.get_slot("MODALITY_HYPOTHETICAL") == 3
    assert root.get_slot("TYPE_PROPOSITION") == 3
    assert root.get_slot("EPIST_PROB_MARGINAL") == 3
    assert root.get_slot("EPIST_FUZZY_PLAUSIBILITY") == 3

    # Child nodes
    agent_children = graph.get_children(root.cid, relation="VAL_X1_AGENT")
    assert len(agent_children) == 1
    assert agent_children[0].get_slot("NSM_THIS") == 1
    assert agent_children[0].get_slot("GRAPH_LEAF") == 1

    patient_children = graph.get_children(root.cid, relation="VAL_X2_PATIENT")
    assert len(patient_children) == 1
    assert patient_children[0].get_slot("NSM_ONE") == 1
    assert patient_children[0].get_slot("GRAPH_LEAF") == 1

    # Aggregate proposition vector check
    prop_vec = graph.to_proposition_vector()
    assert prop_vec["GRAPH_QUERY_TARGET"] == 3
    assert prop_vec["NSM_DO"] == 3
    assert prop_vec["NSM_TOUCH"] == 3
    assert prop_vec["NSM_MAYBE"] == 3
    assert prop_vec["NSM_THIS"] == 1
    assert prop_vec["NSM_ONE"] == 1
    assert prop_vec["TYPE_PROPOSITION"] == 3
    assert prop_vec["MODALITY_HYPOTHETICAL"] == 3



def test_quanta_graph_get_children(nlp_parser):
    """Verify QuantaGraph get_children helper method."""
    graph = nlp_parser.parse_sentence("A golden retriever bit the mailman in the garden.")
    assert graph.root is not None
    root = graph.root

    # Get all children of root
    all_children = graph.get_children(root.cid)
    assert len(all_children) >= 2

    # Get children filtered by relation
    agent_children = graph.get_children(root.cid, relation="VAL_X1_AGENT")
    assert len(agent_children) == 1
    assert agent_children[0].anchor in ("cn:en:golden retriever (n)", "wn:golden_retriever.n.01")

    patient_children = graph.get_children(root.cid, relation="VAL_X2_PATIENT")
    assert len(patient_children) == 1
    assert patient_children[0].anchor in ("cn:en:mailman (n)", "wn:mailman.n.01")


def test_parse_dependency_tree_and_extract_svo(nlp_parser):
    """Verify Phase 6 Items 6A.2, 6A.3, 6A.4: SVO and location extraction."""
    sentence = "A golden retriever bit the mailman in the garden."
    doc = nlp_parser.parse_dependency_tree(sentence)
    assert doc is not None
    assert len(doc) > 0

    svo = nlp_parser.extract_subject_verb_object(doc)
    assert svo.subject == "golden retriever"
    assert svo.verb == "bit"
    assert svo.object == "mailman"
    assert len(svo.modifiers) == 1
    assert svo.modifiers[0]["prep"] == "in"
    assert svo.modifiers[0]["pobj"] == "garden"
    assert "garden" in svo.modifiers[0]["text"]



