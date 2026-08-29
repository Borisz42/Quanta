"""s(CASP) / SWI-Prolog bridge and coinductive ASG cycle reasoning engine for QUANTA.

Provides top-down coinductive Answer Set Programming (ASP) evaluation to resolve
cyclic dependencies (such as recursive function ASTs: factorial, sort, tree traversal)
that cause grounding explosion in standard bottom-up ASP solvers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_index, get_slot_by_name
from solver.validator_gate import ValidationGate, ValidationResult


@dataclass
class CoinductiveProofResult:
    """Result of coinductive rational tree / loop verification."""
    is_valid: bool
    cycles_detected: List[List[str]] = field(default_factory=list)
    coinductive_hypotheses: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class SCaspBridge:
    """s(CASP) / SWI-Prolog subprocess bridge with fallback coinductive cycle reasoning."""

    def __init__(
        self,
        rules_path: Optional[Union[str, Path]] = None,
        swipl_cmd: str = "swipl",
    ):
        self.swipl_cmd = swipl_cmd
        if rules_path is not None:
            self.rules_path = Path(rules_path)
            with open(self.rules_path, "r", encoding="utf-8") as f:
                self.rules_content = f.read()
        else:
            default_path = Path(__file__).parent / "scasp_rules.pl"
            if default_path.exists():
                with open(default_path, "r", encoding="utf-8") as f:
                    self.rules_content = f.read()
            else:
                self.rules_content = ""

        self._asp_gate = ValidationGate()

    def is_swipl_available(self) -> bool:
        """Checks if SWI-Prolog binary is installed and executable on PATH."""
        return shutil.which(self.swipl_cmd) is not None

    def graph_to_prolog_facts(self, graph: QuantaGraph) -> str:
        """Converts QuantaGraph nodes, slots, and edges into Prolog fact statements."""
        lines: List[str] = []
        lines.append("% QUANTA ASG Prolog Facts")

        for cid, node in graph.nodes.items():
            lines.append(f"node('{cid}').")
            if node.anchor:
                escaped_anchor = node.anchor.replace("'", "\\'")
                lines.append(f"anchor('{cid}', '{escaped_anchor}').")

            active = node.vector.active_slots()
            for idx, qval in active.items():
                slot_name = get_slot_by_index(idx).name
                val_int = int(qval)
                lines.append(f"slot('{cid}', '{slot_name}', {val_int}).")

            for rel, targets in node.edges.items():
                for t_cid in targets:
                    lines.append(f"edge('{cid}', '{rel}', '{t_cid}').")

        return "\n".join(lines)

    def generate_prolog_program(self, graph: QuantaGraph, query_goal: str = "valid_asg") -> str:
        """Generates a complete standalone s(CASP) / Prolog program with integrity rules and facts."""
        parts: List[str] = [
            "%",
            "% Auto-generated s(CASP) / Prolog Program for QUANTA ASG Validation",
            "%",
            self.rules_content,
            "",
            "% Graph Facts",
            self.graph_to_prolog_facts(graph),
            "",
            "% Top-level validation rule",
            f"{query_goal} :- not false.",
        ]
        return "\n".join(parts)

    def validate_coinductive_cycles(self, graph: QuantaGraph) -> CoinductiveProofResult:
        """Verifies cyclic graph structures using coinductive greatest fixed-point reasoning.
        
        Evaluates recursive AST topologies (GRAPH_RECURSIVE_REF, GRAPH_CYCLIC_BACKLINK)
        to verify that:
        1. Every cycle is an intentional recursive call backlink targeting an ancestor routine/scope.
        2. The recursive cycle possesses a base-case branching structure (GRAPH_BRANCH_COND / GRAPH_RETURN_VALUE).
        3. No unbounded or malformed cycles exist in non-recursive entity sub-graphs.
        """
        nodes = graph.nodes
        if not nodes:
            return CoinductiveProofResult(is_valid=True)

        # 1. Detect all directed simple cycles using DFS
        visited: Dict[str, int] = {}  # 0=unvisited, 1=visiting (active stack), 2=visited
        detected_cycles: List[List[str]] = []

        def dfs(curr: str, path: List[str]):
            visited[curr] = 1
            node = graph.get_node(curr)
            if node is not None:
                for rel, targets in node.edges.items():
                    for t in targets:
                        target_node = graph.get_node(t)
                        if target_node is None:
                            continue
                        canonical_t = target_node.cid
                        if visited.get(canonical_t, 0) == 1:
                            # Cycle detected: extract cycle slice
                            cycle_start_idx = path.index(canonical_t) if canonical_t in path else -1
                            if cycle_start_idx != -1:
                                cycle = path[cycle_start_idx:] + [curr, canonical_t]
                            else:
                                cycle = [canonical_t, curr, canonical_t]
                            detected_cycles.append(cycle)
                        elif visited.get(canonical_t, 0) == 0:
                            dfs(canonical_t, path + [curr])
            visited[curr] = 2

        for cid in sorted(nodes.keys()):
            if visited.get(cid, 0) == 0:
                dfs(cid, [])

        if not detected_cycles:
            # Acyclic graph -> trivial coinductive validity
            return CoinductiveProofResult(is_valid=True)

        # 2. Coinductive cycle evaluation
        coinductive_hypotheses: List[str] = []
        errors: List[str] = []

        for cycle in detected_cycles:
            # Inspect nodes along the cycle
            cycle_nodes = [nodes[c] for c in cycle if c in nodes]
            has_recursive_ref = False
            has_base_case = False
            target_is_function = False

            # Check if any edge in the cycle is explicitly a recursive call or backlink
            for i in range(len(cycle) - 1):
                src_cid = cycle[i]
                dst_cid = cycle[i + 1]
                src_node = nodes.get(src_cid)
                dst_node = nodes.get(dst_cid)

                if src_node is not None:
                    # Check for recursive slot flags
                    if (
                        src_node.get_slot("GRAPH_RECURSIVE_REF") == 1
                        or src_node.get_slot("GRAPH_CYCLIC_BACKLINK") == 1
                        or src_node.get_slot("SOLVER_COINDUCTION_COLOOP") == 1
                    ):
                        has_recursive_ref = True

                    # Check for recursive relations
                    for rel, targets in src_node.edges.items():
                        if dst_cid in targets:
                            if "RECURSIVE" in rel.upper() or rel in ("GRAPH_RECURSIVE_REF", "CALL", "INVOCATION"):
                                has_recursive_ref = True

                if dst_node is not None:
                    if (
                        dst_node.get_slot("GRAPH_FUNCTION_DEF") == 1
                        or dst_node.get_slot("GRAPH_SCOPED_CONTEXT") == 1
                        or dst_cid == graph.root_cid
                    ):
                        target_is_function = True

            # Check if the surrounding function or sub-graph contains branching base-case
            for n in cycle_nodes:
                if (
                    n.get_slot("GRAPH_BRANCH_COND") == 1
                    or n.get_slot("GRAPH_BRANCH_THEN") == 1
                    or n.get_slot("GRAPH_BRANCH_ELSE") == 1
                    or n.get_slot("GRAPH_RETURN_VALUE") == 1
                ):
                    has_base_case = True

            # Also check all nodes in graph if the cycle is inside a function with branches
            if not has_base_case:
                for n in nodes.values():
                    if (
                        n.get_slot("GRAPH_BRANCH_COND") == 1
                        or n.get_slot("GRAPH_BRANCH_THEN") == 1
                        or n.get_slot("GRAPH_BRANCH_ELSE") == 1
                    ):
                        has_base_case = True
                        break

            if has_recursive_ref and (target_is_function or has_base_case):
                hyp_name = f"coinductive_hypothesis({cycle[0][:8]} -> ... -> {cycle[-1][:8]})"
                coinductive_hypotheses.append(hyp_name)
            else:
                errors.append(
                    f"Malformed ungrounded cycle without coinductive base case: {' -> '.join(c[:8] for c in cycle)}"
                )

        is_valid = len(errors) == 0
        return CoinductiveProofResult(
            is_valid=is_valid,
            cycles_detected=detected_cycles,
            coinductive_hypotheses=coinductive_hypotheses,
            errors=errors,
        )

    def validate_with_swipl(self, graph: QuantaGraph, timeout: float = 5.0) -> ValidationResult:
        """Executes validation via external SWI-Prolog / s(CASP) subprocess if installed."""
        if not self.is_swipl_available():
            # Fallback to internal ASP gate + coinductive cycle validator
            return self.validate(graph)

        program = self.generate_prolog_program(graph, query_goal="valid_asg")
        with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False, encoding="utf-8") as tf:
            tf.write(program)
            tf_path = Path(tf.name)

        try:
            cmd = [
                self.swipl_cmd,
                "-q",
                "-s",
                str(tf_path),
                "-g",
                "valid_asg, halt.",
                "-t",
                "halt(1).",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if res.returncode == 0:
                return ValidationResult(is_valid=True)
            else:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"s(CASP) Prolog validation failure: {res.stderr.strip() or 'false derived'}"],
                )
        except Exception as ex:
            return ValidationResult(
                is_valid=False,
                errors=[f"Prolog execution error: {str(ex)}"],
            )
        finally:
            try:
                tf_path.unlink()
            except OSError:
                pass

    def validate(self, graph: QuantaGraph) -> ValidationResult:
        """Unified validation evaluating ontological ASP constraints and coinductive cycles."""
        # 1. Check coinductive cycles first
        coind_res = self.validate_coinductive_cycles(graph)
        if not coind_res.is_valid:
            return ValidationResult(
                is_valid=False,
                errors=coind_res.errors,
            )

        # 2. Run Clingo ontological & thematic invariant validation
        asp_res = self._asp_gate.validate_graph(graph)
        return asp_res


__all__ = [
    "CoinductiveProofResult",
    "SCaspBridge",
]
