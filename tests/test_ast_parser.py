"""Unit tests for ASTForwardParser (Phase 6D: 6D.1-6D.9)."""

import pytest
import numpy as np
from parser.ast_parser import ASTForwardParser
from parser.scene_graph_grounder import (
    SceneGraphGrounder,
    VisualCrop,
    compute_rcc8_relation,
    RCC8_PREDICATES,
)
from realizer.code_emitter import CodeEmitter
from core.types import QuaternaryValue
from server.embedding_manager import MockEmbeddingManager
from config.multi_scale_config import MultiScaleConfig


@pytest.fixture(scope="module")
def ast_parser():
    return ASTForwardParser()


@pytest.fixture(scope="module")
def code_emitter():
    return CodeEmitter()


def test_example_f_recursive_factorial_asg(ast_parser):
    """Verify Phase 6 Item 6D.6, 6D.9: Canonical Example F (AST Code Topology for Recursive Factorial)."""
    source_code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
""".strip()

    graph = ast_parser.parse_ast_node(source_code)
    assert graph.root is not None
    root = graph.root

    # 1. Root Function Definition (func:factorial, literal: "def factorial(n)")
    assert root.get_slot("GRAPH_ROOT_NODE") == 1
    assert root.get_slot("GRAPH_FUNCTION_DEF") == 1
    assert root.get_slot("GRAPH_SCOPED_CONTEXT") == 1
    assert root.get_slot("TYPE_PROCESS") == 1
    assert root.get_slot("ROLE_COMMUNICATOR") == 1
    assert root.anchor == "func:factorial"
    assert "factorial" in root.literal

    # 2. Formal Parameter (var:n, literal: "n")
    assert "VAL_X1_AGENT" in root.edges
    assert "GRAPH_ARGUMENT_LIST" in root.edges
    arg_cid = root.edges["VAL_X1_AGENT"][0]
    arg_node = graph.get_node(arg_cid)
    assert arg_node is not None
    assert arg_node.anchor == "var:n"
    assert arg_node.get_slot("GRAPH_VARIABLE_BIND") == 1
    assert arg_node.get_slot("GRAPH_ARGUMENT_LIST") == 1
    assert arg_node.get_slot("VAL_X1_AGENT") == 1
    assert arg_node.get_slot("GRAPH_LEAF") == 1
    assert arg_node.get_slot("TYPE_NUMERIC_VALUE") == 1
    assert arg_node.get_slot("TYPE_MEASURE_SCALAR") == 1

    # 3. Sub-expressions of function body
    assert "GRAPH_IS_SUB_EXP" in root.edges
    sub_cids = root.edges["GRAPH_IS_SUB_EXP"]
    assert len(sub_cids) >= 2

    # Find the If branch and the Recursive return statement
    if_node = None
    rec_return_node = None
    for cid in sub_cids:
        node = graph.get_node(cid)
        if node and node.get_slot("GRAPH_BRANCH_COND") != 0:
            if_node = node
        elif node and node.get_slot("GRAPH_RECURSIVE_REF") == 1:
            rec_return_node = node

    assert if_node is not None, "If branch node not found"
    assert rec_return_node is not None, "Recursive return node not found"

    # Verify If Branch Node
    assert if_node.get_slot("GRAPH_BRANCH_COND") == 3
    assert if_node.get_slot("GRAPH_BRANCH_THEN") == 1
    assert if_node.get_slot("LJB_GANAI_IF_THEN") == 3
    assert if_node.get_slot("GRAPH_IS_SUB_EXP") == 1
    assert if_node.get_slot("TYPE_PROPOSITION") == 3

    # Verify Comparison Node (n == 0)
    assert "GRAPH_BRANCH_COND" in if_node.edges
    cmp_cid = if_node.edges["GRAPH_BRANCH_COND"][0]
    cmp_node = graph.get_node(cmp_cid)
    assert cmp_node is not None
    assert cmp_node.get_slot("NSM_SAME") == 1
    assert cmp_node.get_slot("LJB_DU_IDENTITY") == 1
    assert cmp_node.get_slot("LJB_NO_NONE_QUANT") == 1
    assert cmp_node.get_slot("TYPE_PROPOSITION") == 3

    # Verify Base Return Node (return 1)
    assert "GRAPH_BRANCH_THEN" in if_node.edges
    base_ret_cid = if_node.edges["GRAPH_BRANCH_THEN"][0]
    base_ret_node = graph.get_node(base_ret_cid)
    assert base_ret_node is not None
    assert base_ret_node.get_slot("NSM_ONE") == 1
    assert base_ret_node.get_slot("GRAPH_RETURN_VALUE") == 1
    assert base_ret_node.get_slot("GRAPH_SCOPED_CONTEXT") == 1
    assert base_ret_node.get_slot("TYPE_NUMERIC_VALUE") == 1
    assert base_ret_node.get_slot("TYPE_MEASURE_SCALAR") == 1

    # Verify Recursive Return Node (return n * factorial(n - 1))
    assert rec_return_node.get_slot("NSM_PART") == 1
    assert rec_return_node.get_slot("NSM_MUCH") == 1
    assert rec_return_node.get_slot("GRAPH_RETURN_VALUE") == 1
    assert rec_return_node.get_slot("GRAPH_IS_SUB_EXP") == 1
    assert rec_return_node.get_slot("GRAPH_RECURSIVE_REF") == 1
    assert rec_return_node.get_slot("TYPE_PROCESS") == 1

    # Verify Recursive Backlink points to Root Function
    assert "GRAPH_RECURSIVE_REF" in rec_return_node.edges
    target_cid = rec_return_node.edges["GRAPH_RECURSIVE_REF"][0]
    target_node = graph.get_node(target_cid)
    assert target_node is not None
    assert target_node.anchor == "func:factorial"
    assert target_node.get_slot("GRAPH_FUNCTION_DEF") == 1


def test_ast_exception_handling_mapping(ast_parser):
    """Verify Phase 6 Item 6D.8: Exception handling mapping."""
    source_code = """
try:
    x = 1 / 0
except ZeroDivisionError:
    raise ValueError
""".strip()

    graph = ast_parser.parse_ast_node(source_code)
    assert graph.root is not None
    root = graph.root

    # Try node
    assert root.get_slot("GRAPH_EXCEPTION_HANDLE") == 1
    assert root.get_slot("GRAPH_SCOPED_CONTEXT") == 1

    # Except handler node
    assert "GRAPH_BRANCH_ELSE" in root.edges
    handler_cid = root.edges["GRAPH_BRANCH_ELSE"][0]
    handler_node = graph.get_node(handler_cid)
    assert handler_node is not None
    assert handler_node.get_slot("GRAPH_EXCEPTION_HANDLE") == 1
    assert handler_node.get_slot("GRAPH_BRANCH_ELSE") == 1

    # Raise node
    assert "GRAPH_IS_SUB_EXP" in handler_node.edges
    raise_cid = handler_node.edges["GRAPH_IS_SUB_EXP"][0]
    raise_node = graph.get_node(raise_cid)
    assert raise_node is not None
    assert raise_node.get_slot("GRAPH_EXCEPTION_HANDLE") == 1
    assert raise_node.get_slot("TYPE_EVENT") == 1


def test_factorial_code_execution(ast_parser, code_emitter):
    """Verify emitted code from factorial ASG executes correctly and factorial(5) == 120."""
    source_code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
""".strip()

    graph = ast_parser.parse_ast_node(source_code)
    emitted = code_emitter.emit_code(graph)

    env = {}
    exec(emitted, env)
    assert "factorial" in env
    fact_fn = env["factorial"]
    assert fact_fn(0) == 1
    assert fact_fn(1) == 1
    assert fact_fn(5) == 120


def test_ast_loop_topologies(ast_parser):
    """Verify Phase 6 Item 6D.7: Loop mapping (For / While -> GRAPH_CONTROL_LOOP=1)."""
    # For loop
    for_code = """
for i in range(10):
    total = total + i
""".strip()
    g_for = ast_parser.parse_ast_node(for_code)
    assert g_for.root is not None
    root_for = g_for.root
    assert root_for.get_slot("GRAPH_CONTROL_LOOP") == 1
    assert root_for.get_slot("GRAPH_CYCLIC_BACKLINK") == 1
    assert root_for.get_slot("TYPE_PROCESS") == 1
    assert root_for.get_slot("GRAPH_SCOPED_CONTEXT") == 1

    # While loop
    while_code = """
while n > 0:
    n = n - 1
""".strip()
    g_while = ast_parser.parse_ast_node(while_code)
    assert g_while.root is not None
    root_while = g_while.root
    assert root_while.get_slot("GRAPH_CONTROL_LOOP") == 1
    assert root_while.get_slot("GRAPH_BRANCH_COND") == 3


def test_ast_multi_argument_function(ast_parser):
    """Verify Phase 6 Item 6D.5: Multi-argument function signatures."""
    code = """
def add(a, b, c):
    return a + b + c
""".strip()
    graph = ast_parser.parse_ast_node(code)
    assert graph.root is not None
    root = graph.root

    assert root.get_slot("GRAPH_FUNCTION_DEF") == 1
    assert root.literal == "def add(a, b, c)"

    arg_children = graph.get_children(root.cid, relation="GRAPH_ARGUMENT_LIST")
    assert len(arg_children) == 3
    arg_literals = [c.literal for c in arg_children]
    assert arg_literals == ["a", "b", "c"]
    for arg_node in arg_children:
        assert arg_node.get_slot("GRAPH_VARIABLE_BIND") == 1
        assert arg_node.get_slot("GRAPH_ARGUMENT_LIST") == 1
        assert arg_node.get_slot("GRAPH_LEAF") == 1


def test_ast_class_definition_inheritance(ast_parser):
    """Verify class definition and base inheritance mapping."""
    code = """
class GoldenRetriever(Dog):
    pass
""".strip()
    graph = ast_parser.parse_ast_node(code)
    assert graph.root is not None
    root = graph.root

    assert root.get_slot("TYPE_ABSTRACT_CONCEPT") == 1
    assert root.get_slot("GRAPH_ROOT_NODE") == 1
    assert root.anchor == "class:GoldenRetriever"

    base_children = graph.get_children(root.cid, relation="VAL_X4_SOURCE")
    assert len(base_children) == 1
    assert base_children[0].literal == "base:Dog"


# -----------------------------------------------------------------------------
# ANVC Code AST & Multimodal Alignment Unit Tests (§Session 5, exp-037f)
# -----------------------------------------------------------------------------

def test_ast_dense_vector_embedding_and_docstring():
    """Verify function root nodes attach dense vectors and docstrings with configured dimension."""
    mock_mgr = MockEmbeddingManager(matryoshka_dim=64)
    parser = ASTForwardParser(embedding_manager=mock_mgr)

    code = """
def quick_sort(arr):
    \"\"\"Sort an array using the divide-and-conquer quicksort algorithm.\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
""".strip()

    graph = parser.parse_ast_node(code)
    root = graph.root
    assert root is not None
    assert root.anchor == "func:quick_sort"
    assert root.get_slot("GRAPH_FUNCTION_DEF") == 1
    assert root.docstring is not None
    assert "quicksort" in root.docstring

    # Dense vector attached with correct Matryoshka dimension D=64
    assert root.dense_vector is not None
    assert root.dense_vector.shape == (64,)
    # Assert L2 normalized
    norm = float(np.linalg.norm(root.dense_vector))
    assert abs(norm - 1.0) < 1e-3


def test_ast_code_retrieval_semantic_grounding():
    """Verify cross-modal natural language queries search and retrieve corresponding AST functions."""
    mock_mgr = MockEmbeddingManager(matryoshka_dim=64)
    parser = ASTForwardParser(embedding_manager=mock_mgr, code_similarity_threshold=0.20)

    code_suite = """
def bubble_sort(elements):
    \"\"\"Sort a sequence of items by repeatedly stepping through the list and swapping adjacent elements.\"\"\"
    return elements

def binary_search(array, target):
    \"\"\"Search for a target value within a sorted array using divide and conquer interval halving.\"\"\"
    return -1

def calculate_factorial(n):
    \"\"\"Compute the factorial product of all positive integers less than or equal to n.\"\"\"
    return 1
""".strip()

    graph = parser.parse_ast_node(code_suite)
    # Search 1: sorting query
    matches_sort = parser.search_functions("algorithm that swaps adjacent elements to sort", graph)
    assert len(matches_sort) > 0
    top_sort_node, top_sort_score = matches_sort[0]
    assert top_sort_node.anchor == "func:bubble_sort"

    # Search 2: binary search query
    matches_search = parser.search_functions("divide and conquer halving search in sorted array", graph)
    assert len(matches_search) > 0
    top_search_node, top_search_score = matches_search[0]
    assert top_search_node.anchor == "func:binary_search"

    # Search 3: factorial query
    matches_fact = parser.search_functions("compute the factorial product of positive integers", graph)
    assert len(matches_fact) > 0
    top_fact_node, top_fact_score = matches_fact[0]
    assert top_fact_node.anchor == "func:calculate_factorial"


def test_ast_identifier_independence():
    """Verify semantic retrieval grounds to correct AST even when function identifiers are anonymized."""
    mock_mgr = MockEmbeddingManager(matryoshka_dim=64)
    parser = ASTForwardParser(embedding_manager=mock_mgr, code_similarity_threshold=0.20)

    # Function names obfuscated to solution_a, solution_b
    obfuscated_code = """
def solution_a(items):
    \"\"\"Sort elements by repeatedly swapping adjacent items.\"\"\"
    return items

def solution_b(items, key):
    \"\"\"Find target value in sorted list using interval halving.\"\"\"
    return key
""".strip()

    graph = parser.parse_ast_node(obfuscated_code)

    # Query matches based on semantics/docstring without relying on function name
    matches_sort = parser.rank_functions("swap adjacent items to sort", graph)
    assert len(matches_sort) >= 2
    assert matches_sort[0][0].anchor == "func:solution_a"

    matches_search = parser.rank_functions("find target in sorted list interval halving", graph)
    assert len(matches_search) >= 2
    assert matches_search[0][0].anchor == "func:solution_b"


def test_ast_configurable_threshold():
    """Verify threshold filtering honors ast.code_similarity_threshold."""
    mock_mgr = MockEmbeddingManager(matryoshka_dim=64)
    cfg = MultiScaleConfig(runtime_overrides={"ast.code_similarity_threshold": 0.99})
    parser = ASTForwardParser(config=cfg, embedding_manager=mock_mgr)

    code = "def f(x):\n    \"\"\"Do something.\"\"\"\n    return x"
    graph = parser.parse_ast_node(code)

    # High threshold 0.99 should filter out low similarity query
    matches = parser.search_functions("completely unrelated cosmological query", graph)
    assert len(matches) == 0

    # Explicit threshold override 0.01 admits the candidate
    admitted = parser.search_functions("query", graph, threshold=0.01)
    assert len(admitted) == 1


def test_scene_graph_grounder_archetype_mapping():
    """Verify visual perception crops map to correct WordNet noun synsets with >= 85% accuracy."""
    mock_mgr = MockEmbeddingManager(matryoshka_dim=64)
    cfg = MultiScaleConfig()
    grounder = SceneGraphGrounder(config=cfg, embedding_manager=mock_mgr)

    test_crops = [
        VisualCrop("c1", "dog", (0.1, 0.1, 0.4, 0.4), description="golden retriever dog"),
        VisualCrop("c2", "cat", (0.5, 0.5, 0.8, 0.8), description="domestic feline cat"),
        VisualCrop("c3", "car", (0.2, 0.2, 0.6, 0.6), description="motor automobile vehicle"),
        VisualCrop("c4", "apple", (0.3, 0.3, 0.5, 0.5), description="fresh red fruit apple"),
        VisualCrop("c5", "rose", (0.4, 0.4, 0.6, 0.6), description="red garden flower rose"),
        VisualCrop("c6", "person", (0.0, 0.0, 0.3, 0.8), description="adult human individual"),
    ]

    expected_categories = {
        "c1": "WN_ANIMAL_FAUNA",
        "c2": "WN_ANIMAL_FAUNA",
        "c3": "WN_ARTIFACT_OBJECT",
        "c4": "WN_FOOD_NUTRITION",
        "c5": "WN_PLANT_FLORA",
        "c6": "WN_PERSON_HUMAN",
    }

    correct_count = 0
    for crop in test_crops:
        g_obj = grounder.ground_crop(crop)
        expected_slot = expected_categories[crop.crop_id]
        if g_obj.wn_root_slot == expected_slot:
            correct_count += 1

    accuracy = float(correct_count) / float(len(test_crops))
    assert accuracy >= 0.85, f"WordNet mapping accuracy {accuracy:.2f} < 0.85"


def test_scene_graph_grounder_rcc8_spatial():
    """Verify all 8 RCC-8 spatial predicates are correctly computed and wired into QuantaGraph."""
    # 1. Congruent Equal (EQ)
    assert compute_rcc8_relation((0.2, 0.2, 0.6, 0.6), (0.2, 0.2, 0.6, 0.6)) == "SPATIAL_RCC_CONGRUENT_EQ"

    # 2. Disconnected (DC)
    assert compute_rcc8_relation((0.0, 0.0, 0.2, 0.2), (0.6, 0.6, 0.8, 0.8)) == "SPATIAL_RCC_DISCONNECTED"

    # 3. Externally Connected (EC) - touching boundaries
    assert compute_rcc8_relation((0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 1.0, 0.5)) == "SPATIAL_RCC_EXT_CONNECTED"

    # 4. Partial Overlap (PO)
    assert compute_rcc8_relation((0.1, 0.1, 0.6, 0.6), (0.4, 0.4, 0.9, 0.9)) == "SPATIAL_RCC_PARTIAL_OVERLAP"

    # 5. Tangential Proper Part (TPP) - A inside B, touching boundary
    assert compute_rcc8_relation((0.2, 0.2, 0.5, 0.5), (0.2, 0.2, 0.8, 0.8)) == "SPATIAL_RCC_TANGENTIAL_PART"

    # 6. Non-Tangential Proper Part (NTPP) - A strictly interior inside B
    assert compute_rcc8_relation((0.3, 0.3, 0.5, 0.5), (0.1, 0.1, 0.9, 0.9)) == "SPATIAL_RCC_NON_TANG_PART"

    # 7. Tangential Proper Part Inverse (TPPi) - B inside A, touching boundary
    assert compute_rcc8_relation((0.2, 0.2, 0.8, 0.8), (0.2, 0.2, 0.5, 0.5)) == "SPATIAL_RCC_TANG_PART_INV"

    # 8. Non-Tangential Proper Part Inverse (NTPPi) - B strictly interior inside A
    assert compute_rcc8_relation((0.1, 0.1, 0.9, 0.9), (0.3, 0.3, 0.5, 0.5)) == "SPATIAL_RCC_NON_TANG_PART_INV"

    # Verify scene graph building
    mock_mgr = MockEmbeddingManager(matryoshka_dim=64)
    grounder = SceneGraphGrounder(embedding_manager=mock_mgr)

    crops = [
        VisualCrop("obj_sky", "sky", (0.0, 0.0, 1.0, 0.4)),
        VisualCrop("obj_car", "car", (0.1, 0.5, 0.5, 0.9)),
        VisualCrop("obj_wheel", "wheel", (0.15, 0.7, 0.3, 0.90)),
    ]
    graph = grounder.build_scene_graph(crops)

    assert graph.root is not None
    assert graph.root.anchor == "scene:root"
    assert graph.root.get_slot("GRAPH_ROOT_NODE") == 1

    # Verify RCC-8 relations in edges: wheel inside car
    car_node = None
    wheel_node = None
    for n in graph.nodes.values():
        if n.anchor == "vis:obj_car":
            car_node = n
        elif n.anchor == "vis:obj_wheel":
            wheel_node = n

    assert car_node is not None
    assert wheel_node is not None

    # wheel is TPP of car
    assert "SPATIAL_RCC_TANGENTIAL_PART" in wheel_node.edges
    assert car_node.cid in wheel_node.edges["SPATIAL_RCC_TANGENTIAL_PART"]

