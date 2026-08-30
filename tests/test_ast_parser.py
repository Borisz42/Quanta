"""Unit tests for ASTForwardParser (Phase 6D: 6D.1-6D.9)."""

import pytest
from parser.ast_parser import ASTForwardParser
from realizer.code_emitter import CodeEmitter
from core.types import QuaternaryValue


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

