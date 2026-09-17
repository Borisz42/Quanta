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
    muc_edges: List[Tuple[str, str, str]] = field(default_factory=list)  # (src_cid, rel, dst_cid)
    muc_intervals: List[Tuple[str, int, int]] = field(default_factory=list)  # (node_cid, start, end)
    models: List[List[str]] = field(default_factory=list)

    @property
    def muc_nodes(self) -> List[str]:
        """Returns the distinct list of node CIDs identified in the Minimal Unsatisfiable Core (MUC)."""
        seen: Set[str] = set()
        nodes: List[str] = []
        for cid, _, _ in self.muc_slots:
            if cid not in seen:
                seen.add(cid)
                nodes.append(cid)
        for src, _, dst in self.muc_edges:
            if src not in seen:
                seen.add(src)
                nodes.append(src)
            if dst not in seen:
                seen.add(dst)
                nodes.append(dst)
        for cid, _, _ in self.muc_intervals:
            if cid not in seen:
                seen.add(cid)
                nodes.append(cid)
        return nodes

    @property
    def muc(self) -> Optional[List[str]]:
        """Minimal Unsatisfiable Core node CIDs if invalid, else None."""
        if not self.is_valid:
            return self.muc_nodes
        return None

    def __iter__(self):
        """Allows unpacking as (is_valid, muc_nodes)."""
        yield self.is_valid
        yield self.muc

    def __repr__(self) -> str:
        if self.is_valid:
            return "ValidationResult(VALID)"
        return f"ValidationResult(INVALID, errors={self.errors}, muc_slots={self.muc_slots}, muc_nodes={self.muc_nodes})"


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

    def validate(self, graph: QuantaGraph) -> Tuple[bool, Optional[List[str]]]:
        """Validates a graph and returns (is_valid: bool, muc: list[node_cid] | None) conforming to 7C.3 API."""
        result = self.validate_graph(graph)
        return result.is_valid, result.muc

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
            "{ event_interval(Ev, Start, End) } :- candidate_interval(Ev, Start, End).",
        ]
        assumptions: List[Tuple[clingo.Symbol, bool]] = []
        slot_map: Dict[str, Tuple[str, str, int]] = {}
        edge_map: Dict[str, Tuple[str, str, str]] = {}
        interval_map: Dict[str, Tuple[str, int, int]] = {}

        from core.slots import LEGACY_ONTOLOGY_ALIASES
        inv_aliases: Dict[str, List[str]] = {}
        for alias_k, canon_v in LEGACY_ONTOLOGY_ALIASES.items():
            inv_aliases.setdefault(canon_v, []).append(alias_k)

        current_nodes = graph.nodes
        for cid, node in current_nodes.items():
            active = node.vector.active_slots()
            for idx, qval in active.items():
                if 0 <= idx < 1024:
                    slot_name = get_slot_by_index(idx).name
                else:
                    slot_name = f"DIM_{idx}"
                val_int = int(qval)

                names_to_emit = [slot_name] + inv_aliases.get(slot_name, [])
                for s_name in names_to_emit:
                    candidates.append(f'candidate_slot("{cid}", "{s_name}", {val_int}).')
                    sym = clingo.Function(
                        "slot",
                        [clingo.String(cid), clingo.String(s_name), clingo.Number(val_int)],
                    )
                    assumptions.append((sym, True))
                    slot_map[str(sym)] = (cid, s_name, val_int)

            for rel, targets in node.edges.items():
                for t_cid in targets:
                    t_node = graph.get_node(t_cid)
                    canonical_t = t_node.cid if t_node is not None else t_cid
                    candidates.append(f'candidate_edge("{cid}", "{rel}", "{canonical_t}").')
                    edge_sym = clingo.Function(
                        "edge",
                        [clingo.String(cid), clingo.String(rel), clingo.String(canonical_t)],
                    )
                    assumptions.append((edge_sym, True))
                    edge_map[str(edge_sym)] = (cid, rel, canonical_t)

            # Temporal intervals
            t_start = getattr(node, "time_start", None)
            t_end = getattr(node, "time_end", None)
            if t_start is not None or t_end is not None:
                s_int = int(t_start) if t_start is not None else -999999
                e_int = int(t_end) if t_end is not None else 999999
                candidates.append(f'candidate_interval("{cid}", {s_int}, {e_int}).')
                int_sym = clingo.Function(
                    "event_interval",
                    [clingo.String(cid), clingo.Number(s_int), clingo.Number(e_int)],
                )
                assumptions.append((int_sym, True))
                interval_map[str(int_sym)] = (cid, s_int, e_int)

        # 3. Assemble and ground ASP program
        program = self.rules_content + "\n" + "\n".join(candidates)
        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        # 4. Solve with assumptions and extract MUC if UNSAT
        solve_res = ctl.solve(assumptions=assumptions)

        if solve_res.satisfiable:
            return ValidationResult(is_valid=True)
        else:
            # Extract Minimal Unsatisfiable Core (MUC) using deletion filter
            current_core = list(assumptions)
            for i in range(len(current_core) - 1, -1, -1):
                test_assumptions = current_core[:i] + current_core[i + 1 :]
                test_res = ctl.solve(assumptions=test_assumptions)
                if not test_res.satisfiable:
                    current_core = test_assumptions

            muc_slots: List[Tuple[str, str, int]] = []
            muc_edges: List[Tuple[str, str, str]] = []
            muc_intervals: List[Tuple[str, int, int]] = []
            muc_symbols: List[str] = []

            for sym, _ in current_core:
                sym_str = str(sym)
                muc_symbols.append(sym_str)
                if sym_str in slot_map:
                    muc_slots.append(slot_map[sym_str])
                elif sym_str in edge_map:
                    muc_edges.append(edge_map[sym_str])
                elif sym_str in interval_map:
                    muc_intervals.append(interval_map[sym_str])

            errors = [
                f"Ontological contradiction in node '{cid}' for slot '{slot}' (value={val})"
                for cid, slot, val in muc_slots
            ]
            for src, rel, dst in muc_edges:
                errors.append(f"Relational contradiction on edge '{rel}' from '{src}' to '{dst}'")
            for cid, s_val, e_val in muc_intervals:
                errors.append(f"Temporal interval contradiction on node '{cid}' ({s_val}..{e_val})")
            if not errors and muc_symbols:
                errors = [f"ASP constraint violation in core: {s}" for s in muc_symbols]

            return ValidationResult(
                is_valid=False,
                errors=errors,
                muc_slots=muc_slots,
                muc_edges=muc_edges,
                muc_intervals=muc_intervals,
            )


# Alias ValidatorGate to ValidationGate for 7C.1 specification compliance
ValidatorGate = ValidationGate

__all__ = [
    "ValidationResult",
    "ValidationGate",
    "ValidatorGate",
]
