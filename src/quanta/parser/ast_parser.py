"""Recursive AST Forward Parser for converting Python Abstract Syntax Trees into Quanta ASGs."""

from __future__ import annotations
import ast
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from quanta.core.asg import QuantaGraph, QuantaNode
from quanta.core.types import QuantaVector, QuaternaryValue
from quanta.parser.lexical_grounder import WordNetLexicalGrounder


class ASTForwardParser:
    """Recursively converts Python AST nodes into rich, content-addressed QuantaGraph ASGs."""

    def __init__(self, offline_cache_path: Optional[str] = None):
        self.grounder = WordNetLexicalGrounder(offline_cache_path=offline_cache_path)

    def parse_ast_node(self, node: ast.AST, source_text: Optional[str] = None) -> QuantaGraph:
        """Parses a Python AST node into a full QuantaGraph."""
        graph = QuantaGraph()
        root_node = self._build_node_recursive(node, graph)
        if root_node:
            graph.root_cid = root_node.cid
        return graph

    def ast_to_vector(self, node: ast.AST) -> QuantaVector:
        """Parses AST node and returns aggregated proposition vector across the entire subtree."""
        graph = self.parse_ast_node(node)
        return graph.to_proposition_vector()

    def _build_node_recursive(self, node: ast.AST, graph: QuantaGraph) -> QuantaNode:
        """Recursively translates an AST node and attaches child nodes/edges."""
        node_type = type(node).__name__
        q_node = QuantaNode(literal=f"AST:{node_type}")

        # Default modality and basic types
        q_node.set_slot("MODALITY_LITERAL", 1)

        # 1. Function / Method Definitions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            q_node.set_slot("EXT_AST_FUNCTION_DEF", 1)
            q_node.set_slot("GRAPH_ROOT_NODE", 1)
            q_node.set_slot("EXT_AST_SCOPE_ENTER", 1)
            q_node.set_slot("TYPE_PROCESS", 1)
            q_node.set_slot("ROLE_COMMUNICATOR", 1)
            q_node.anchor = f"func:{node.name}"
            q_node.literal = f"def {node.name}(...)"

            if isinstance(node, ast.AsyncFunctionDef):
                q_node.set_slot("LJB_ASYNC_CONCURRENT", 1)

            graph.add_node(q_node)

            # Arguments
            for arg in node.args.args:
                arg_qnode = QuantaNode(literal=f"arg:{arg.arg}")
                arg_qnode.set_slot("EXT_AST_VARIABLE_BINDING", 1)
                arg_qnode.set_slot("VAL_X1_AGENT", 1)
                arg_qnode.set_slot("GRAPH_ARGUMENT_LIST", 1)
                arg_qnode.set_slot("GRAPH_LEAF", 1)
                arg_qnode.anchor = f"var:{arg.arg}"
                arg_cid = graph.add_node(arg_qnode)
                graph.add_edge(q_node.cid, "VAL_X1_AGENT", arg_cid)

            # Body statements (first 5 to capture structure)
            for stmt in node.body[:5]:
                stmt_qnode = self._build_node_recursive(stmt, graph)
                graph.add_edge(q_node.cid, "GRAPH_IS_SUB_EXP", stmt_qnode.cid)
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
                    base_cid = graph.add_node(base_qnode)
                    graph.add_edge(q_node.cid, "VAL_X4_SOURCE", base_cid)
            return q_node

        # 3. Loops (For / While)
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            q_node.set_slot("EXT_AST_CONTROL_LOOP", 1)
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
                graph.add_edge(q_node.cid, "GRAPH_IS_SUB_EXP", stmt_qnode.cid)
            return q_node

        # 4. Conditionals (If / IfExp)
        elif isinstance(node, (ast.If, ast.IfExp)):
            q_node.set_slot("GRAPH_BRANCH_COND", 3)
            q_node.set_slot("GRAPH_BRANCH_THEN", 1)
            q_node.set_slot("LJB_GANAI_IF_THEN", 3)
            q_node.set_slot("TYPE_PROPOSITION", 3)

            if getattr(node, "orelse", None):
                q_node.set_slot("GRAPH_BRANCH_ELSE", 3)

            graph.add_node(q_node)

            test_qnode = self._build_node_recursive(node.test, graph)
            graph.add_edge(q_node.cid, "GRAPH_BRANCH_COND", test_qnode.cid)
            return q_node

        # 5. Assignments (Assign / AugAssign / AnnAssign)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            q_node.set_slot("EXT_AST_VARIABLE_BINDING", 1)
            q_node.set_slot("LJB_DU_IDENTITY", 1)
            q_node.set_slot("TYPE_EVENT", 1)

            if isinstance(node, ast.AnnAssign):
                q_node.set_slot("EXT_AST_TYPE_CHECK", 1)
                q_node.set_slot("GRAPH_TYPE_SIGNATURE", 1)

            if isinstance(node, ast.AugAssign):
                q_node.set_slot("NSM_MORE", 1)

            graph.add_node(q_node)

            val_node = getattr(node, "value", None)
            if val_node:
                val_qnode = self._build_node_recursive(val_node, graph)
                graph.add_edge(q_node.cid, "VAL_X2_PATIENT", val_qnode.cid)
            return q_node

        # 6. Returns & Yields
        elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
            q_node.set_slot("EXT_AST_RETURN", 1)
            q_node.set_slot("EXT_AST_SCOPE_EXIT", 1)
            q_node.set_slot("GRAPH_RETURN_VALUE", 1)

            graph.add_node(q_node)

            val_node = getattr(node, "value", None)
            if val_node:
                val_qnode = self._build_node_recursive(val_node, graph)
                graph.add_edge(q_node.cid, "VAL_X2_PATIENT", val_qnode.cid)
            return q_node

        # 7. Function Calls
        elif isinstance(node, ast.Call):
            q_node.set_slot("EXT_AST_CALL", 1)
            q_node.set_slot("GRAPH_INVOCATION_HEAD", 1)
            q_node.set_slot("TYPE_EVENT", 1)

            if isinstance(node.func, ast.Name):
                q_node.anchor = f"call:{node.func.id}"
                q_node.literal = f"call {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                q_node.anchor = f"method:{node.func.attr}"
                q_node.literal = f"call {node.func.attr}"

            arg_count = len(node.args)
            if arg_count == 1:
                q_node.set_slot("NSM_ONE", 1)
            elif arg_count == 2:
                q_node.set_slot("NSM_TWO", 1)
            elif arg_count > 2:
                q_node.set_slot("NSM_MUCH", 1)

        # 8. Binary Operations
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

        # 9. Comparisons
        elif isinstance(node, ast.Compare):
            q_node.set_slot("TYPE_PROPOSITION", 3)
            if len(node.ops) > 0:
                cmp_op = node.ops[0]
                if isinstance(cmp_op, (ast.Eq, ast.Is)):
                    q_node.set_slot("NSM_SAME", 1)
                    q_node.set_slot("LJB_DU_IDENTITY", 1)
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

        # 10. Boolean Operations (And / Or)
        elif isinstance(node, ast.BoolOp):
            q_node.set_slot("TYPE_PROPOSITION", 1)
            if isinstance(node.op, ast.And):
                q_node.set_slot("LJB_JE_AND", 1)
            elif isinstance(node.op, ast.Or):
                q_node.set_slot("LJB_JA_OR", 1)

        # 11. Unary Operations (Not / USub)
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                q_node.set_slot("LJB_NA_NEGATION", 2)
            elif isinstance(node.op, ast.USub):
                q_node.set_slot("NSM_PART", 1)

        # 12. Identifiers (Name)
        elif isinstance(node, ast.Name):
            q_node.set_slot("EXT_AST_VARIABLE_BINDING", 1)
            q_node.anchor = f"var:{node.id}"
            q_node.literal = node.id
            lemma = node.id.lower().replace("_", "")
            try:
                concept = self.grounder.ground_synset(lemma)
                q_node.vector = concept.vector.copy()
            except Exception:
                q_node.set_slot("TYPE_ABSTRACT_CONCEPT", 1)

        # 13. Literals / Constants
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
