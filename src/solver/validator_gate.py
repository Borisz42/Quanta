"""Neuro-symbolic constraint validator gate with Minimal Unsatisfiable Core (MUC) extraction."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import clingo

from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_index, get_slot_by_name


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    muc_slots: List[Tuple[str, str, int]] = field(default_factory=list)  # (node_cid, slot_name, val)
    models: List[List[str]] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.is_valid:
            return "ValidationResult(VALID)"
        return f"ValidationResult(INVALID, errors={self.errors}, muc_slots={self.muc_slots})"


class ValidationGate:
    """Symbolic validation gate enforcing ontological integrity and extracting Minimal Unsatisfiable Cores (MUCs)."""

    def __init__(self, rules_path: Optional[Union[str, Path]] = None):
        if rules_path is not None:
            self.rules_path = Path(rules_path)
            with open(self.rules_path, "r", encoding="utf-8") as f:
                self.rules_content = f.read()
        else:
            default_path = Path(__file__).parent / "scasp_rules.lp"
            if default_path.exists():
                with open(default_path, "r", encoding="utf-8") as f:
                    self.rules_content = f.read()
            else:
                self.rules_content = ""

    def validate_node(self, node: QuantaNode, node_id: str = "node_0") -> ValidationResult:
        """Validates an isolated QuantaNode against ontological integrity rules."""
        temp_graph = QuantaGraph()
        temp_graph.add_node(node)
        return self.validate_graph(temp_graph)

    def validate_graph(self, graph: QuantaGraph) -> ValidationResult:
        """Validates an entire QuantaGraph against ontological and relational constraints."""
        # 1. Structural graph integrity check first
        struct_valid, struct_errors = graph.validate_integrity()
        if not struct_valid:
            return ValidationResult(
                is_valid=False,
                errors=[f"Structural graph integrity error: {e}" for e in struct_errors],
            )

        # 2. Build Clingo Control instance
        ctl = clingo.Control(["--warn=none"])

        candidates: List[str] = [
            "{ slot(N, S, V) } :- candidate_slot(N, S, V).",
            "{ edge(Src, Rel, Dst) } :- candidate_edge(Src, Rel, Dst).",
        ]
        assumptions: List[Tuple[clingo.Symbol, bool]] = []
        slot_map: Dict[str, Tuple[str, str, int]] = {}

        current_nodes = graph.nodes
        for cid, node in current_nodes.items():
            active = node.vector.active_slots()
            for idx, qval in active.items():
                slot_def = get_slot_by_index(idx)
                slot_name = slot_def.name
                val_int = int(qval)

                candidates.append(f'candidate_slot("{cid}", "{slot_name}", {val_int}).')
                sym = clingo.Function(
                    "slot",
                    [clingo.String(cid), clingo.String(slot_name), clingo.Number(val_int)],
                )
                assumptions.append((sym, True))
                slot_map[str(sym)] = (cid, slot_name, val_int)

            for rel, targets in node.edges.items():
                for t_cid in targets:
                    candidates.append(f'candidate_edge("{cid}", "{rel}", "{t_cid}").')
                    edge_sym = clingo.Function(
                        "edge",
                        [clingo.String(cid), clingo.String(rel), clingo.String(t_cid)],
                    )
                    assumptions.append((edge_sym, True))

        # 3. Assemble and ground ASP program
        program = self.rules_content + "\n" + "\n".join(candidates)
        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        # 4. Solve with assumptions and extract MUC if UNSAT
        with ctl.solve(assumptions=assumptions, yield_=True) as handle:
            solve_res = handle.get()

            if solve_res.satisfiable:
                return ValidationResult(is_valid=True)
            else:
                core_lits = set(handle.core())
                abs_core_lits = {abs(lit) for lit in core_lits}
                muc_slots: List[Tuple[str, str, int]] = []
                muc_symbols: List[str] = []

                for atom in ctl.symbolic_atoms:
                    if atom.literal in core_lits or atom.literal in abs_core_lits:
                        sym_str = str(atom.symbol)
                        muc_symbols.append(sym_str)
                        if sym_str in slot_map:
                            muc_slots.append(slot_map[sym_str])

                errors = [
                    f"Ontological contradiction in node '{cid}' for slot '{slot}' (value={val})"
                    for cid, slot, val in muc_slots
                ]
                if not errors and muc_symbols:
                    errors = [f"ASP constraint violation in core: {s}" for s in muc_symbols]

                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    muc_slots=muc_slots,
                )
