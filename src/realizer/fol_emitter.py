"""First-Order Logic (FOL) Formula Emitter for QUANTA.

Reconstructs standard mathematical First-Order Logic formula strings from QuantaGraph ASGs.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, QuaternaryValue


class FOLEmitter:
    """Reconstructs standard First-Order Logic (FOL) formulas from QuantaGraph ASG topologies."""

    def emit_formula(self, graph: QuantaGraph) -> str:
        """Emits a First-Order Logic string from a QuantaGraph."""
        root = graph.root
        if root is None:
            return ""

        # If literal contains the original formula and matches structure
        if root.literal and any(sym in str(root.literal) for sym in ("\\forall", "\\exists", "∀", "∃", "(", "->", "\\rightarrow")):
            # If literal was an intact FOL string
            lit = str(root.literal).strip()
            return lit

        # Reconstruct from ASG sub-expressions and predicates
        sub_nodes = []
        if "GRAPH_IS_SUB_EXP" in root.edges:
            for cid in root.edges["GRAPH_IS_SUB_EXP"]:
                n = graph.get_node(cid)
                if n:
                    sub_nodes.append(n)

        # 1. Detect Quantifiers
        quantifier_prefix = ""
        if root.get_slot("NSM_ALL") == 1 or root.get_slot("LJB_RO_ALL_QUANT") == 1:
            quantifier_prefix = "\\forall x "
        elif root.get_slot("NSM_SOME") == 1 or root.get_slot("LJB_SUO_AT_LEAST_ONE") == 1:
            quantifier_prefix = "\\exists x "

        # 2. Build Sub-Predicates
        pred_strings = []
        for pred_node in sub_nodes:
            pred_str = self._emit_predicate(graph, pred_node)
            if pred_str:
                pred_strings.append(pred_str)

        if not pred_strings:
            # Fallback: single predicate from root
            pred_str = self._emit_predicate(graph, root)
            return f"{quantifier_prefix}{pred_str}".strip()

        # 3. Connect Predicates with Logical Connectives
        if root.get_slot("LJB_GANAI_IF_THEN") != 0 or root.get_slot("GRAPH_BRANCH_COND") == 1:
            if len(pred_strings) >= 2:
                antecedent = pred_strings[0]
                consequent = " \\land ".join(pred_strings[1:]) if len(pred_strings) > 2 else pred_strings[1]
                body = f"({antecedent} \\rightarrow {consequent})"
            else:
                body = pred_strings[0]
        elif root.get_slot("LJB_JA_OR") == 1:
            body = " \\lor ".join(pred_strings)
        elif root.get_slot("LJB_JON_XOR") == 1:
            body = " \\oplus ".join(pred_strings)
        else:
            body = " \\land ".join(pred_strings)

        if root.get_slot("LJB_NA_NEGATION") == 2:
            body = f"\\neg({body})"

        return f"{quantifier_prefix}{body}".strip()

    def _emit_predicate(self, graph: QuantaGraph, node: QuantaNode) -> str:
        """Emits a single atomic predicate: Pred(arg1, arg2...)."""
        pred_name = ""
        if node.literal and isinstance(node.literal, str):
            lit = node.literal.strip()
            if "(" not in lit:
                pred_name = lit
        if not pred_name and node.anchor and node.anchor.startswith("wn:"):
            pred_name = node.anchor[3:].split(".")[0].capitalize()

        if not pred_name:
            pred_name = "P"

        is_negated = node.get_slot("LJB_NA_NEGATION") == 2
        neg_prefix = "\\neg " if is_negated else ""

        # Collect arguments
        args = []
        for rel in sorted(node.edges.keys()):
            if rel.startswith("VAL_X"):
                for t_cid in node.edges[rel]:
                    t_node = graph.get_node(t_cid)
                    if t_node and t_node.literal:
                        args.append(str(t_node.literal).strip())
                    else:
                        args.append("x")

        if not args:
            args = ["x"]

        return f"{neg_prefix}{pred_name}({', '.join(args)})"
