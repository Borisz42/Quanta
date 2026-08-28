"""First-Order Logic (FOL) formula parser for FOLIO dataset integration."""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Union

from quanta.core.asg import QuantaGraph, QuantaNode
from quanta.core.types import QuantaVector, QuaternaryValue
from quanta.parser.lexical_grounder import WordNetLexicalGrounder


class FOLParser:
    """Parses First-Order Logic formulas into canonical QUANTA ASG graphs with 4-valued polarity."""

    def __init__(self, offline_cache_path: Optional[str] = None):
        self.grounder = WordNetLexicalGrounder(offline_cache_path=offline_cache_path)

    def parse_formula(self, fol_str: str, is_query: bool = False) -> QuantaGraph:
        """Parses a First-Order Logic string into a QuantaGraph with full quaternary polarities."""
        fol = fol_str.strip()
        graph = QuantaGraph()

        has_negation = bool(re.search(r"(\\neg|¬|~|\bNOT\b)", fol))
        has_implication = bool(re.search(r"(\\rightarrow|->|→|\bIMPLIES\b)", fol))

        root_node = QuantaNode(literal=fol)
        root_node.set_slot("TYPE_PROPOSITION", 3 if is_query else 1)
        root_node.set_slot("GRAPH_ASSERTION_CLAIM", 1)
        root_node.set_slot("EPIST_DEDUCTIVE_INFERENCE", 1)
        root_node.set_slot("SOLVER_PROOF_VALIDATED", 1)

        if is_query:
            root_node.set_slot("GRAPH_QUERY_TARGET", 3)
            root_node.set_slot("EPIST_PROB_MARGINAL", 3)
            root_node.set_slot("MODALITY_HYPOTHETICAL", 3)

        # 1. Detect Quantifiers
        if "\\forall" in fol or "∀" in fol or "FORALL" in fol or fol.startswith("all "):
            root_node.set_slot("NSM_ALL", 1)
            root_node.set_slot("LJB_RO_ALL_QUANT", 1)
        elif "\\exists" in fol or "∃" in fol or "EXISTS" in fol or fol.startswith("some "):
            root_node.set_slot("NSM_SOME", 1)
            root_node.set_slot("LJB_SUO_AT_LEAST_ONE", 1)

        # 2. Detect Connectives
        if has_implication:
            root_node.set_slot("LJB_GANAI_IF_THEN", 3 if is_query else 1)
            root_node.set_slot("GRAPH_ENTAILMENT_EDGE", 1)
            root_node.set_slot("GRAPH_BRANCH_COND", 1)
            root_node.set_slot("GRAPH_BRANCH_THEN", 1)
        if "\\land" in fol or "&" in fol or "∧" in fol or "AND" in fol:
            root_node.set_slot("LJB_JE_AND", 1)
        if "\\lor" in fol or "|" in fol or "∨" in fol or "OR" in fol:
            root_node.set_slot("LJB_JA_OR", 1)
        if "\\oplus" in fol or "XOR" in fol or "⊕" in fol:
            root_node.set_slot("LJB_JON_XOR", 1)
        if has_negation:
            root_node.set_slot("LJB_NA_NEGATION", 2)
        else:
            root_node.set_slot("NSM_TRUE", 1)

        root_cid = graph.add_node(root_node, set_as_root=True)

        # 3. Extract Predicates and Arguments: (~)?PredName(arg1, arg2...)
        pred_pattern = re.compile(r"(\\neg|¬|~)?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]+)\)")
        matches = pred_pattern.findall(fol)

        for neg_prefix, pred_name, args_str in matches:
            is_pred_negated = bool(neg_prefix)
            pred_polarity = 2 if is_pred_negated else (3 if is_query else 1)
            args = [a.strip() for a in args_str.split(",")]
            pred_lemma = pred_name.lower()

            try:
                concept = self.grounder.ground_synset(pred_lemma)
                pred_node = QuantaNode(vector=concept.vector.copy(), anchor=concept.synset_name, literal=pred_name)
                pred_node.set_slot("GRAPH_IS_SUB_EXP", 1)
                if is_pred_negated:
                    for slot_idx, val in pred_node.vector.active_slots().items():
                        pred_node.vector[slot_idx] = QuaternaryValue.NEGATED
                    pred_node.set_slot("LJB_NA_NEGATION", 2)
            except Exception:
                pred_node = QuantaNode(literal=pred_name)
                pred_node.set_slot("TYPE_RELATION_ROLE", pred_polarity)
                pred_node.set_slot("GRAPH_IS_SUB_EXP", 1)
                if is_pred_negated:
                    pred_node.set_slot("LJB_NA_NEGATION", 2)

            pred_cid = graph.add_node(pred_node)
            graph.add_edge(root_cid, "GRAPH_IS_SUB_EXP", pred_cid)

            # Add arguments as child nodes
            for i, arg in enumerate(args):
                arg_node = QuantaNode(literal=arg)
                arg_node.set_slot("GRAPH_LEAF", 1)
                if i == 0:
                    arg_node.set_slot("VAL_X1_AGENT", 1)
                elif i == 1:
                    arg_node.set_slot("VAL_X2_PATIENT", 1)
                elif i == 2:
                    arg_node.set_slot("VAL_X3_DESTINATION", 1)

                if arg.islower() and len(arg) == 1:  # Variable like x, y
                    arg_node.set_slot("EXT_AST_VARIABLE_BINDING", 1)
                else:  # Constant like 'socrates'
                    arg_node.set_slot("TYPE_HUMAN", 1)
                    arg_node.set_slot("ROLE_AGENT_CAPABLE", 1)
                arg_cid = graph.add_node(arg_node)
                rel_name = f"VAL_X{min(i+1, 5)}_AGENT" if i == 0 else f"VAL_X{min(i+1, 5)}_PATIENT"
                graph.add_edge(pred_cid, rel_name, arg_cid)

        return graph

    def formula_to_vector(self, fol_str: str, is_query: bool = False) -> QuantaVector:
        """Helper returning the 256-d vector of the parsed FOL proposition ASG."""
        graph = self.parse_formula(fol_str, is_query=is_query)
        return graph.to_proposition_vector()
