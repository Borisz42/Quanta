"""Python Code Emitter for QUANTA.

Reconstructs valid, executable Python source code from Abstract Syntax Graph (ASG) AST topologies.
"""

from __future__ import annotations
import ast
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, QuaternaryValue


class CodeEmitter:
    """Reconstructs executable Python source code from content-addressed QuantaGraph ASGs."""

    def emit_code(self, graph: QuantaGraph) -> str:
        """Emits formatted Python code from a QuantaGraph."""
        root = graph.root
        if root is None:
            return ""

        return self._emit_node(graph, root, indent_level=0)

    def _emit_node(self, graph: QuantaGraph, node: QuantaNode, indent_level: int = 0) -> str:
        """Recursively converts an ASG node into Python source code statements."""
        indent = "    " * indent_level

        # 1. Function Definition
        if node.get_slot("GRAPH_FUNCTION_DEF") == 1:
            func_name = "func"
            if node.anchor and node.anchor.startswith("func:"):
                func_name = node.anchor[5:]
            elif node.literal and str(node.literal).startswith("def "):
                parts = str(node.literal).split()
                if len(parts) >= 2:
                    func_name = parts[1].split("(")[0]

            # Extract arguments
            args = []
            for rel in ("VAL_X1_AGENT", "GRAPH_ARGUMENT_LIST"):
                if rel in node.edges:
                    for cid in node.edges[rel]:
                        arg_node = graph.get_node(cid)
                        if arg_node:
                            if arg_node.anchor and arg_node.anchor.startswith("var:"):
                                args.append(arg_node.anchor[4:])
                            elif arg_node.literal:
                                lit = str(arg_node.literal)
                                args.append(lit.replace("arg:", ""))

            args_str = ", ".join(args) if args else "n"

            # Function body
            body_stmts = []
            if "GRAPH_IS_SUB_EXP" in node.edges:
                for child_cid in node.edges["GRAPH_IS_SUB_EXP"]:
                    child_node = graph.get_node(child_cid)
                    if child_node:
                        stmt = self._emit_node(graph, child_node, indent_level=indent_level + 1)
                        if stmt:
                            body_stmts.append(stmt)

            if not body_stmts:
                body_stmts.append(f"{indent}    pass")

            body_code = "\n".join(body_stmts)
            return f"{indent}def {func_name}({args_str}):\n{body_code}"

        # 2. Control Loops (For / While)
        elif node.get_slot("GRAPH_CONTROL_LOOP") == 1:
            is_while = node.get_slot("GRAPH_BRANCH_COND") != 0
            header = f"{indent}while condition:" if is_while else f"{indent}for item in iterable:"

            body_stmts = []
            if "GRAPH_IS_SUB_EXP" in node.edges:
                for child_cid in node.edges["GRAPH_IS_SUB_EXP"]:
                    child_node = graph.get_node(child_cid)
                    if child_node:
                        stmt = self._emit_node(graph, child_node, indent_level=indent_level + 1)
                        if stmt:
                            body_stmts.append(stmt)

            if not body_stmts:
                body_stmts.append(f"{indent}    pass")

            return f"{header}\n" + "\n".join(body_stmts)

        # 3. Branching / Conditionals (If)
        elif node.get_slot("GRAPH_BRANCH_COND") != 0 or node.get_slot("LJB_GANAI_IF_THEN") != 0:
            cond_expr = "condition"
            if "GRAPH_BRANCH_COND" in node.edges and node.edges["GRAPH_BRANCH_COND"]:
                cond_node = graph.get_node(node.edges["GRAPH_BRANCH_COND"][0])
                if cond_node and cond_node.literal:
                    cond_expr = str(cond_node.literal)

            then_body = f"{indent}    return True"
            if "GRAPH_BRANCH_THEN" in node.edges and node.edges["GRAPH_BRANCH_THEN"]:
                then_node = graph.get_node(node.edges["GRAPH_BRANCH_THEN"][0])
                if then_node:
                    then_body = self._emit_node(graph, then_node, indent_level=indent_level + 1)

            result = f"{indent}if {cond_expr}:\n{then_body}"
            if "GRAPH_BRANCH_ELSE" in node.edges and node.edges["GRAPH_BRANCH_ELSE"]:
                else_node = graph.get_node(node.edges["GRAPH_BRANCH_ELSE"][0])
                if else_node:
                    else_body = self._emit_node(graph, else_node, indent_level=indent_level + 1)
                    result += f"\n{indent}else:\n{else_body}"

            return result

        # 4. Variable Binding / Assignment
        elif node.get_slot("GRAPH_VARIABLE_BIND") == 1:
            var_name = "x"
            if node.anchor and node.anchor.startswith("var:"):
                var_name = node.anchor[4:]
            elif node.literal:
                var_name = str(node.literal).replace("var:", "")

            val_str = "None"
            if "VAL_X2_PATIENT" in node.edges and node.edges["VAL_X2_PATIENT"]:
                val_node = graph.get_node(node.edges["VAL_X2_PATIENT"][0])
                if val_node and val_node.literal is not None:
                    val_str = str(val_node.literal)

            return f"{indent}{var_name} = {val_str}"

        # 5. Return Statement
        elif node.get_slot("GRAPH_RETURN_VALUE") == 1:
            ret_val = ""
            if "VAL_X2_PATIENT" in node.edges and node.edges["VAL_X2_PATIENT"]:
                val_node = graph.get_node(node.edges["VAL_X2_PATIENT"][0])
                if val_node:
                    if val_node.anchor and val_node.anchor.startswith("var:"):
                        ret_val = val_node.anchor[4:]
                    elif val_node.literal is not None:
                        ret_val = str(val_node.literal)
            elif node.literal:
                ret_val = str(node.literal)

            return f"{indent}return {ret_val}".rstrip()

        # 6. Function / Method Calls
        elif node.get_slot("GRAPH_INVOCATION_CALL") == 1:
            call_target = "call_func"
            if node.anchor and node.anchor.startswith("call:"):
                call_target = node.anchor[5:]
            elif node.literal:
                call_target = str(node.literal).replace("call ", "")

            return f"{indent}{call_target}()"

        # Fallback to literal representation
        if node.literal:
            return f"{indent}{node.literal}"

        return f"{indent}pass"
