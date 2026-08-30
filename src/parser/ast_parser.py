"""Recursive AST Forward Parser for converting Python Abstract Syntax Trees into Quanta ASGs."""

from __future__ import annotations
import ast
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.types import (
    QuantaVector,
    QuaternaryValue,
    EpistemicValue,
    StructuralValue,
    RoutingValue,
    RegisterValue,
    BandContract,
)
from parser.lexical_grounder import WordNetLexicalGrounder


class ASTForwardParser:
    """Recursively converts Python AST nodes into rich, content-addressed QuantaGraph ASGs."""

    def __init__(self, offline_cache_path: Optional[str] = None):
        self.grounder = WordNetLexicalGrounder(offline_cache_path=offline_cache_path)
        self._current_function_name: Optional[str] = None
        self._current_function_node: Optional[QuantaNode] = None

    def parse_ast_node(self, node: Union[ast.AST, str], source_text: Optional[str] = None) -> QuantaGraph:
        """Parses a Python AST node or source string into a full QuantaGraph."""
        if isinstance(node, str):
            node = ast.parse(node)

        graph = QuantaGraph()
        self._current_function_name = None
        self._current_function_node = None
        root_node = self._build_node_recursive(node, graph)
        if root_node:
            graph.root_cid = root_node.cid
        return graph

    def ast_to_vector(self, node: Union[ast.AST, str]) -> QuantaVector:
        """Parses AST node and returns aggregated proposition vector across the entire subtree."""
        graph = self.parse_ast_node(node)
        return graph.to_proposition_vector()

    def _build_node_recursive(self, node: ast.AST, graph: QuantaGraph) -> QuantaNode:
        """Recursively translates an AST node and attaches child nodes/edges."""
        node_type = type(node).__name__
        q_node = QuantaNode(literal=f"AST:{node_type}")

        # Default modality and basic types
        q_node.set_slot("MODALITY_LITERAL", 1)

        # 0. Module Root
        if isinstance(node, ast.Module):
            if not node.body:
                graph.add_node(q_node)
                return q_node
            if len(node.body) == 1:
                return self._build_node_recursive(node.body[0], graph)
            q_node.set_slot("GRAPH_ROOT_NODE", 1)
            q_node.set_slot("GRAPH_SCOPED_CONTEXT", 1)
            graph.add_node(q_node)
            for stmt in node.body:
                stmt_qnode = self._build_node_recursive(stmt, graph)
                graph.add_edge(q_node, "GRAPH_IS_SUB_EXP", stmt_qnode)
            return q_node

        # 1. Function / Method Definitions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            q_node.set_slot("GRAPH_FUNCTION_DEF", 1)
            q_node.set_slot("GRAPH_ROOT_NODE", 1)
            q_node.set_slot("GRAPH_SCOPED_CONTEXT", 1)
            q_node.set_slot("TYPE_PROCESS", 1)
            q_node.set_slot("ROLE_COMMUNICATOR", 1)
            q_node.anchor = f"func:{node.name}"
            args_repr = ", ".join(a.arg for a in node.args.args)
            q_node.literal = f"def {node.name}({args_repr})"

            if isinstance(node, ast.AsyncFunctionDef):
                q_node.set_slot("LJB_ASYNC_CONCURRENT", 1)

            graph.add_node(q_node)

            prev_func_name = self._current_function_name
            prev_func_node = self._current_function_node
            self._current_function_name = node.name
            self._current_function_node = q_node

            # Arguments
            for arg in node.args.args:
                arg_qnode = QuantaNode(literal=arg.arg)
                arg_qnode.set_slot("GRAPH_VARIABLE_BIND", 1)
                arg_qnode.set_slot("VAL_X1_AGENT", 1)
                arg_qnode.set_slot("GRAPH_ARGUMENT_LIST", 1)
                arg_qnode.set_slot("GRAPH_LEAF", 1)
                arg_qnode.set_slot("TYPE_NUMERIC_VALUE", 1)
                arg_qnode.set_slot("TYPE_MEASURE_SCALAR", 1)
                arg_qnode.anchor = f"var:{arg.arg}"
                graph.add_node(arg_qnode)
                graph.add_edge(q_node, "VAL_X1_AGENT", arg_qnode)
                graph.add_edge(q_node, "GRAPH_ARGUMENT_LIST", arg_qnode)

            # Body statements
            for stmt in node.body:
                stmt_qnode = self._build_node_recursive(stmt, graph)
                graph.add_edge(q_node, "GRAPH_IS_SUB_EXP", stmt_qnode)

            self._current_function_name = prev_func_name
            self._current_function_node = prev_func_node
            return q_node

        # 2. Class Definitions
        elif isinstance(node, ast.ClassDef):
            q_node.set_slot("TYPE_ABSTRACT_CONCEPT", 1)
            q_node.set_slot("WN_GROUP_SOCIAL", 1)
            q_node.set_slot("GRAPH_ROOT_NODE", 1)
            q_node.set_slot("GRAPH_SCOPED_CONTEXT", 1)
            q_node.anchor = f"class:{node.name}"
            q_node.literal = f"class {node.name}"

            graph.add_node(q_node)

            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_qnode = QuantaNode(literal=f"base:{base.id}")
                    base_qnode.set_slot("TYPE_ABSTRACT_CONCEPT", 1)
                    base_qnode.set_slot("VAL_X4_SOURCE", 1)
                    graph.add_node(base_qnode)
                    graph.add_edge(q_node, "VAL_X4_SOURCE", base_qnode)
            return q_node

        # 3. Loops (For / While)
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            q_node.set_slot("GRAPH_CONTROL_LOOP", 1)
            q_node.set_slot("GRAPH_CYCLIC_BACKLINK", 1)
            q_node.set_slot("NSM_CONTINUOUS_RATE", 1)
            q_node.set_slot("TYPE_PROCESS", 1)
            q_node.set_slot("GRAPH_SCOPED_CONTEXT", 1)

            if isinstance(node, ast.While):
                q_node.set_slot("GRAPH_BRANCH_COND", 3)

            if isinstance(node, ast.AsyncFor):
                q_node.set_slot("LJB_ASYNC_CONCURRENT", 1)

            graph.add_node(q_node)

            for stmt in node.body[:3]:
                stmt_qnode = self._build_node_recursive(stmt, graph)
                graph.add_edge(q_node, "GRAPH_IS_SUB_EXP", stmt_qnode)
            return q_node

        # 4. Conditionals (If / IfExp)
        elif isinstance(node, (ast.If, ast.IfExp)):
            test_str = ast.unparse(node.test) if hasattr(ast, "unparse") else "condition"
            q_node.literal = f"if {test_str}"
            q_node.anchor = "branch:if"
            q_node.set_slot("GRAPH_BRANCH_COND", 3)
            q_node.set_slot("GRAPH_BRANCH_THEN", 1)
            q_node.set_slot("LJB_GANAI_IF_THEN", 3)
            q_node.set_slot("GRAPH_IS_SUB_EXP", 1)
            q_node.set_slot("TYPE_PROPOSITION", 3)

            if getattr(node, "orelse", None):
                q_node.set_slot("GRAPH_BRANCH_ELSE", 3)

            graph.add_node(q_node)

            test_qnode = self._build_node_recursive(node.test, graph)
            graph.add_edge(q_node, "GRAPH_BRANCH_COND", test_qnode)

            for stmt in getattr(node, "body", []):
                then_qnode = self._build_node_recursive(stmt, graph)
                graph.add_edge(q_node, "GRAPH_BRANCH_THEN", then_qnode)

            for stmt in getattr(node, "orelse", []):
                else_qnode = self._build_node_recursive(stmt, graph)
                graph.add_edge(q_node, "GRAPH_BRANCH_ELSE", else_qnode)

            return q_node

        # 5. Assignments (Assign / AugAssign / AnnAssign)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            q_node.set_slot("GRAPH_VARIABLE_BIND", 1)
            q_node.set_slot("LJB_DU_IDENTITY", 1)
            q_node.set_slot("TYPE_EVENT", 1)

            if isinstance(node, ast.AnnAssign):
                q_node.set_slot("GRAPH_TYPE_SIGNATURE", 1)

            if isinstance(node, ast.AugAssign):
                q_node.set_slot("NSM_MORE", 1)

            graph.add_node(q_node)

            val_node = getattr(node, "value", None)
            if val_node:
                val_qnode = self._build_node_recursive(val_node, graph)
                graph.add_edge(q_node, "VAL_X2_PATIENT", val_qnode)
            return q_node

        # 6. Returns & Yields
        elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
            val_str = ast.unparse(node.value) if (getattr(node, "value", None) and hasattr(ast, "unparse")) else "None"
            q_node.literal = f"return {val_str}"
            q_node.set_slot("GRAPH_RETURN_VALUE", 1)

            # Check if this return statement contains a recursive call to enclosing function
            has_recursive_call = False
            if self._current_function_name and getattr(node, "value", None):
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Call):
                        call_id = getattr(sub.func, "id", None) if isinstance(sub.func, ast.Name) else getattr(sub.func, "attr", None)
                        if call_id == self._current_function_name:
                            has_recursive_call = True
                            break

            if has_recursive_call:
                q_node.set_slot("GRAPH_IS_SUB_EXP", 1)
                q_node.set_slot("GRAPH_RECURSIVE_REF", 1)
                q_node.set_slot("TYPE_PROCESS", 1)

                if getattr(node, "value", None):
                    for sub in ast.walk(node.value):
                        if isinstance(sub, ast.BinOp):
                            if isinstance(sub.op, (ast.Mult, ast.MatMult)):
                                q_node.set_slot("NSM_MUCH", 1)
                            elif isinstance(sub.op, ast.Sub):
                                q_node.set_slot("NSM_PART", 1)

                graph.add_node(q_node)
                if self._current_function_node:
                    graph.add_edge(q_node, "GRAPH_RECURSIVE_REF", self._current_function_node)
                return q_node

            q_node.set_slot("GRAPH_SCOPED_CONTEXT", 1)
            val_node = getattr(node, "value", None)
            if isinstance(val_node, ast.Constant):
                if val_node.value == 1:
                    q_node.set_slot("NSM_ONE", 1)
                    q_node.set_slot("TYPE_NUMERIC_VALUE", 1)
                    q_node.set_slot("TYPE_MEASURE_SCALAR", 1)
                elif val_node.value == 0:
                    q_node.set_slot("LJB_NO_NONE_QUANT", 1)

            graph.add_node(q_node)

            if val_node and not isinstance(val_node, ast.Constant):
                val_qnode = self._build_node_recursive(val_node, graph)
                graph.add_edge(q_node, "VAL_X2_PATIENT", val_qnode)
            return q_node

        # 7. Function Calls
        elif isinstance(node, ast.Call):
            q_node.set_slot("GRAPH_INVOCATION_CALL", 1)
            q_node.set_slot("TYPE_EVENT", 1)

            func_name = "func"
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                q_node.anchor = f"call:{func_name}"
                q_node.literal = f"call {func_name}"
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
                q_node.anchor = f"method:{func_name}"
                q_node.literal = f"call {func_name}"

            if self._current_function_name and func_name == self._current_function_name:
                q_node.set_slot("GRAPH_RECURSIVE_REF", 1)
                if self._current_function_node:
                    graph.add_edge(q_node, "GRAPH_RECURSIVE_REF", self._current_function_node)

            arg_count = len(node.args)
            if arg_count == 1:
                q_node.set_slot("NSM_ONE", 1)
            elif arg_count == 2:
                q_node.set_slot("NSM_TWO", 1)
            elif arg_count > 2:
                q_node.set_slot("NSM_MUCH", 1)

        # 8. Exception Handling (Try / Except / Raise)
        elif isinstance(node, ast.Try):
            q_node.literal = "try ... except"
            q_node.anchor = "control:try"
            q_node.set_slot("GRAPH_EXCEPTION_HANDLE", 1)
            q_node.set_slot("GRAPH_SCOPED_CONTEXT", 1)
            q_node.set_slot("TYPE_PROCESS", 1)

            graph.add_node(q_node)

            for stmt in node.body:
                stmt_qnode = self._build_node_recursive(stmt, graph)
                graph.add_edge(q_node, "GRAPH_IS_SUB_EXP", stmt_qnode)

            for handler in node.handlers:
                h_qnode = self._build_node_recursive(handler, graph)
                graph.add_edge(q_node, "GRAPH_BRANCH_ELSE", h_qnode)

            for stmt in getattr(node, "finalbody", []):
                f_qnode = self._build_node_recursive(stmt, graph)
                graph.add_edge(q_node, "GRAPH_ORDERED_SEQ", f_qnode)

            return q_node

        elif isinstance(node, ast.ExceptHandler):
            exc_name = ast.unparse(node.type) if (node.type and hasattr(ast, "unparse")) else "Exception"
            q_node.literal = f"except {exc_name}"
            q_node.anchor = f"except:{exc_name}"
            q_node.set_slot("GRAPH_EXCEPTION_HANDLE", 1)
            q_node.set_slot("GRAPH_BRANCH_ELSE", 1)
            q_node.set_slot("TYPE_EVENT", 1)

            graph.add_node(q_node)

            for stmt in node.body:
                stmt_qnode = self._build_node_recursive(stmt, graph)
                graph.add_edge(q_node, "GRAPH_IS_SUB_EXP", stmt_qnode)

            return q_node

        elif isinstance(node, ast.Raise):
            exc_str = ast.unparse(node.exc) if (node.exc and hasattr(ast, "unparse")) else "Exception"
            q_node.literal = f"raise {exc_str}"
            q_node.set_slot("GRAPH_EXCEPTION_HANDLE", 1)
            q_node.set_slot("TYPE_EVENT", 1)
            graph.add_node(q_node)
            return q_node

        # 9. Binary Operations
        elif isinstance(node, ast.BinOp):
            q_node.set_slot("TYPE_PROCESS", 1)
            op = node.op

            if isinstance(op, ast.Add):
                q_node.set_slot("LJB_JE_AND", 1)
                q_node.set_slot("NSM_MORE", 1)
            elif isinstance(op, ast.Sub):
                q_node.set_slot("NSM_PART", 1)
                q_node.set_slot("NSM_LITTLE", 1)
            elif isinstance(op, (ast.Mult, ast.MatMult)):
                q_node.set_slot("NSM_MUCH", 1)
                q_node.set_slot("NSM_BIG", 1)
            elif isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)):
                q_node.set_slot("NSM_PART", 1)
                q_node.set_slot("NSM_SMALL", 1)
            elif isinstance(op, ast.Pow):
                q_node.set_slot("NSM_ACCELERATING_RATE", 1)
            elif isinstance(op, ast.BitAnd):
                q_node.set_slot("LJB_JE_AND", 1)
            elif isinstance(op, ast.BitOr):
                q_node.set_slot("LJB_JA_OR", 1)
            elif isinstance(op, ast.BitXor):
                q_node.set_slot("LJB_JON_XOR", 1)

        # 10. Comparisons
        elif isinstance(node, ast.Compare):
            cmp_str = ast.unparse(node) if hasattr(ast, "unparse") else "comparison"
            q_node.literal = cmp_str
            q_node.set_slot("TYPE_PROPOSITION", 3)
            if len(node.ops) > 0:
                cmp_op = node.ops[0]
                if isinstance(cmp_op, (ast.Eq, ast.Is)):
                    q_node.set_slot("NSM_SAME", 1)
                    q_node.set_slot("LJB_DU_IDENTITY", 1)
                    if node.comparators and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 0:
                        q_node.set_slot("LJB_NO_NONE_QUANT", 1)
                elif isinstance(cmp_op, (ast.NotEq, ast.IsNot)):
                    q_node.set_slot("NSM_SAME", 2)
                    q_node.set_slot("NSM_OTHER", 1)
                    q_node.set_slot("LJB_NA_NEGATION", 2)
                elif isinstance(cmp_op, (ast.Lt, ast.LtE)):
                    q_node.set_slot("NSM_SMALL", 1)
                    q_node.set_slot("NSM_LITTLE", 1)
                elif isinstance(cmp_op, (ast.Gt, ast.GtE)):
                    q_node.set_slot("NSM_BIG", 1)
                    q_node.set_slot("NSM_MORE", 1)
                elif isinstance(cmp_op, ast.In):
                    q_node.set_slot("NSM_INSIDE", 1)
                elif isinstance(cmp_op, ast.NotIn):
                    q_node.set_slot("NSM_INSIDE", 2)
                    q_node.set_slot("NSM_OUTSIDE", 1)
                    q_node.set_slot("LJB_NA_NEGATION", 2)

        # 11. Boolean Operations (And / Or)
        elif isinstance(node, ast.BoolOp):
            q_node.set_slot("TYPE_PROPOSITION", 1)
            if isinstance(node.op, ast.And):
                q_node.set_slot("LJB_JE_AND", 1)
            elif isinstance(node.op, ast.Or):
                q_node.set_slot("LJB_JA_OR", 1)

        # 12. Unary Operations (Not / USub)
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                q_node.set_slot("LJB_NA_NEGATION", 2)
            elif isinstance(node.op, ast.USub):
                q_node.set_slot("NSM_PART", 1)

        # 13. Identifiers (Name)
        elif isinstance(node, ast.Name):
            q_node.set_slot("GRAPH_VARIABLE_BIND", 1)
            q_node.anchor = f"var:{node.id}"
            q_node.literal = node.id
            lemma = node.id.lower().replace("_", "")
            try:
                concept = self.grounder.ground_synset(lemma)
                q_node.vector = concept.vector.copy()
            except Exception:
                q_node.set_slot("TYPE_ABSTRACT_CONCEPT", 1)

        # 14. Literals / Constants
        elif isinstance(node, ast.Constant):
            val = node.value
            q_node.literal = repr(val)
            if isinstance(val, str):
                q_node.set_slot("TYPE_COMMUNICATION_MSG", 1)
                q_node.set_slot("WN_COMMUNICATION_INFO", 1)
            elif isinstance(val, (int, float)):
                q_node.set_slot("TYPE_NUMERIC_VALUE", 1)
                q_node.set_slot("TYPE_MEASURE_SCALAR", 1)
                if val == 0:
                    q_node.set_slot("LJB_NO_NONE_QUANT", 1)
                elif val == 1:
                    q_node.set_slot("NSM_ONE", 1)
                elif val == 2:
                    q_node.set_slot("NSM_TWO", 1)
                elif val > 2:
                    q_node.set_slot("NSM_MUCH", 1)
            elif isinstance(val, bool):
                if val:
                    q_node.set_slot("NSM_TRUE", 1)
                else:
                    q_node.set_slot("LJB_NA_NEGATION", 2)

        graph.add_node(q_node)
        return q_node

