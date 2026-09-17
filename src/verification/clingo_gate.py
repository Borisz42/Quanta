"""Formal Semantic Verification Gate and Closed-Loop MUC Repair Manager for QUANTA (Phase 4).

Connects low-level Clingo / s(CASP) Answer Set Programming assumption cores with
the neural transduction layer:
1. ClingoVerificationGate: Analyzes raw Minimal Unsatisfiable Cores (MUCs) and
   translates them into model-actionable [REPAIR REQUEST] diagnostic prompts.
2. MUCRepairManager: Coordinates the closed-loop repair cycle with a strict cap of
   maximum 2 repair attempts (governed by Stacked Bushes protocol).
3. MUCDiagnostic: Typed container storing the violation category, summary, conflicting
   nodes/slots, and formatted repair request block.
4. RepairResult: Typed outcome of the verification and repair process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_index, get_slot_by_name
from parser.asg_compiler import ASGCompilationError, ASGCompiler
from parser.entity_manifest import EntityRecord
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)
from parser.sexpr_parser import parse_sexpr, to_sexpr
from solver.validator_gate import ValidationGate, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class MUCDiagnostic:
    """Actionable diagnostic explanation extracted from a Minimal Unsatisfiable Core (MUC)."""

    category: str  # "ontological", "temporal", "causal", "structural", "epistemic", "unknown"
    summary: str
    conflicting_nodes: List[str] = field(default_factory=list)
    conflicting_slots: List[Tuple[str, str, int]] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    repair_prompt: str = ""

    def format_repair_request(self) -> str:
        """Format an actionable [REPAIR REQUEST] prompt block for the neural transducer."""
        cat_desc = {
            "ontological": "ontological axiom",
            "temporal": "Allen temporal calculus consistency",
            "causal": "causal hierarchy exclusivity",
            "structural": "graph structural topology invariant",
            "epistemic": "epistemic/deontic modal consistency",
            "unknown": "semantic validity constraint",
        }.get(self.category, f"{self.category} axiom")

        lines = [
            "[REPAIR REQUEST]",
            f"Candidate sub-graph violated {cat_desc}:",
            f"  CONFLICT: {self.summary}",
        ]
        if self.details:
            for d in self.details:
                lines.append(f"  DETAIL: {d}")
        lines.append("Regenerate S-expression resolving the conflict.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_repair_request()


@dataclass
class MUCDiagnosticResult:
    """Outcome of ClingoVerificationGate verification."""

    is_valid: bool
    diagnostic: Optional[MUCDiagnostic] = None
    validation_result: Optional[ValidationResult] = None
    errors: List[str] = field(default_factory=list)

    @property
    def muc(self) -> Optional[List[str]]:
        if self.validation_result:
            return self.validation_result.muc
        return None

    @property
    def muc_slots(self) -> List[Tuple[str, str, int]]:
        if self.validation_result:
            return self.validation_result.muc_slots
        return []

    def __iter__(self):
        yield self.is_valid
        yield self.diagnostic


@dataclass
class RepairResult:
    """Outcome of closed-loop MUC repair orchestration."""

    success: bool
    graph: Optional[QuantaGraph] = None
    final_sexpr: Optional[str] = None
    extraction_result: Optional[DiscourseExtractionResult] = None
    attempts: int = 0
    diagnostics: List[MUCDiagnostic] = field(default_factory=list)
    error_message: Optional[str] = None


class ClingoVerificationGate:
    """Formal verification gate analyzing Clingo MUCs into actionable diagnostics."""

    def __init__(
        self,
        validator_gate: Optional[ValidationGate] = None,
        rules_path: Optional[Union[str, Path]] = None,
    ):
        self.validator_gate = validator_gate or ValidationGate(rules_path=rules_path)

    def validate_graph(
        self,
        graph: QuantaGraph,
        extraction_result: Optional[DiscourseExtractionResult] = None,
    ) -> MUCDiagnosticResult:
        """Validates a QuantaGraph and returns a typed diagnostic if UNSAT."""
        val_res = self.validator_gate.validate_graph(graph)
        if val_res.is_valid:
            return MUCDiagnosticResult(
                is_valid=True,
                validation_result=val_res,
            )

        diagnostic = self.diagnose_result(
            val_res=val_res,
            graph=graph,
            extraction_result=extraction_result,
        )
        return MUCDiagnosticResult(
            is_valid=False,
            diagnostic=diagnostic,
            validation_result=val_res,
            errors=val_res.errors,
        )

    def diagnose_result(
        self,
        val_res: ValidationResult,
        graph: Optional[QuantaGraph] = None,
        extraction_result: Optional[DiscourseExtractionResult] = None,
    ) -> MUCDiagnostic:
        """Translates low-level MUC slots and errors into a human/model-actionable diagnostic."""
        muc_slots = val_res.muc_slots
        muc_nodes = val_res.muc_nodes
        errors = val_res.errors

        # Extract node metadata if graph is provided
        node_map: Dict[str, QuantaNode] = graph.nodes if graph else {}
        edges_map: Dict[str, Dict[str, List[str]]] = {cid: n.edges for cid, n in node_map.items()}

        slot_names: Set[str] = {s_name for _, s_name, _ in muc_slots}

        def _get_node_slot(node_obj: Optional[QuantaNode], s_name: str) -> int:
            if node_obj is None:
                return 0
            try:
                return int(node_obj.vector[s_name])
            except (KeyError, IndexError, TypeError):
                return 0

        # -------------------------------------------------------------------
        # 1. Ontological Domain / Range Violations
        # -------------------------------------------------------------------
        # Pattern A: Abstract concept acting as agent (Rule 1 or Rule 6)
        has_abstract = "TYPE_ABSTRACT_CONCEPT" in slot_names or any(
            _get_node_slot(n, "TYPE_ABSTRACT_CONCEPT") == 1 for n in node_map.values()
        )
        has_agent_role = "ROLE_AGENT_CAPABLE" in slot_names or any(
            "VAL_X1_AGENT" in n.edges for n in node_map.values()
        )

        # Check for explicit Rule 6: EventNode --VAL_X1_AGENT--> AgentNode where Agent is abstract
        abstract_agent_found = False
        conflict_ent_label = "abstract entity"
        conflict_ev_label = "event"
        conflict_ev_id = "Ev1"

        if graph:
            for cid, n in node_map.items():
                if "VAL_X1_AGENT" in n.edges:
                    for target_cid in n.edges["VAL_X1_AGENT"]:
                        target_node = node_map.get(target_cid)
                        if target_node and (
                            _get_node_slot(target_node, "TYPE_ABSTRACT_CONCEPT") == 1
                            or target_cid in muc_nodes
                            or "TYPE_ABSTRACT_CONCEPT" in slot_names
                        ):
                            abstract_agent_found = True
                            conflict_ent_label = target_node.literal or target_node.anchor or target_cid
                            conflict_ev_label = n.literal or n.anchor or cid
                            conflict_ev_id = cid
                            break
                if abstract_agent_found:
                    break

        if not abstract_agent_found and has_abstract and (has_agent_role or "ROLE_AGENT_CAPABLE" in slot_names):
            # Rule 1 direct slot contradiction on single node
            abstract_agent_found = True
            for cid, s_name, _ in muc_slots:
                if s_name == "TYPE_ABSTRACT_CONCEPT":
                    node = node_map.get(cid)
                    if node:
                        conflict_ent_label = node.literal or node.anchor or cid
                    break

        # Check extraction result for better human-readable IDs if available
        if extraction_result and abstract_agent_found:
            for ent in extraction_result.entities:
                if (
                    ent.canonical_name.lower() in str(conflict_ent_label).lower()
                    or ent.category.upper() == "OBJECT"
                    or "abstract" in ent.category.lower()
                ):
                    conflict_ent_label = ent.canonical_name
                    break
            for ev in extraction_result.events:
                if ev.predicate.lower() in str(conflict_ev_label).lower():
                    conflict_ev_id = ev.id
                    conflict_ev_label = ev.predicate
                    break

        if abstract_agent_found:
            summary = (
                f"Entity '{conflict_ent_label}' (abstract concept) cannot act as agent "
                f"in event '{conflict_ev_id}' ({conflict_ev_label})."
            )
            diag = MUCDiagnostic(
                category="ontological",
                summary=summary,
                conflicting_nodes=muc_nodes,
                conflicting_slots=muc_slots,
                details=[
                    "Physical and literal actions require an agent-capable physical entity (PERSON, ANIMAL, or ORGANIZATION).",
                    "Abstract concepts cannot exert physical agency unless figurative modality is explicitly declared.",
                ],
            )
            diag.repair_prompt = diag.format_repair_request()
            return diag

        # Pattern B: Inanimate entity as sentient experiencer (Rule 2 / Rule 7)
        if "ROLE_SENTIENT" in slot_names and (
            "TYPE_INANIMATE_PHYSICAL" in slot_names or "TYPE_ARTIFACT" in slot_names
        ):
            summary = "Inanimate physical entity cannot possess sentience or act as sentient experiencer."
            diag = MUCDiagnostic(
                category="ontological",
                summary=summary,
                conflicting_nodes=muc_nodes,
                conflicting_slots=muc_slots,
                details=["Sentient and experiencer roles require animate beings (PERSON or ANIMAL)."],
            )
            diag.repair_prompt = diag.format_repair_request()
            return diag

        # Pattern C: Natural object vs Artifact mutual exclusivity (Rule 3)
        if "TYPE_NATURAL_OBJECT" in slot_names and "TYPE_ARTIFACT" in slot_names:
            summary = "Entity cannot be simultaneously a natural object and a manufactured artifact."
            diag = MUCDiagnostic(
                category="ontological",
                summary=summary,
                conflicting_nodes=muc_nodes,
                conflicting_slots=muc_slots,
                details=["Natural objects and manufactured artifacts are mutually exclusive categories."],
            )
            diag.repair_prompt = diag.format_repair_request()
            return diag

        # -------------------------------------------------------------------
        # 2. Allen Temporal Contradictions (Rule 9)
        # -------------------------------------------------------------------
        temp_slots = [s for s in slot_names if s.startswith("TEMP_ALLEN_")]
        if len(temp_slots) >= 2 or any("TEMP_ALLEN_" in err for err in errors):
            conflicting_names = ", ".join(temp_slots) if temp_slots else "temporal intervals"
            summary = (
                f"Conflicting temporal interval relations ({conflicting_names}). "
                "Events cannot simultaneously satisfy mutually exclusive temporal intervals."
            )
            diag = MUCDiagnostic(
                category="temporal",
                summary=summary,
                conflicting_nodes=muc_nodes,
                conflicting_slots=muc_slots,
                details=[
                    "Allen interval calculus dictates mutual exclusivity among temporal ordering primitives (e.g. BEFORE vs DURING or EQUALS vs MEETS).",
                    "Ensure temporal endpoints and relation types maintain a consistent chronological order.",
                ],
            )
            diag.repair_prompt = diag.format_repair_request()
            return diag

        # -------------------------------------------------------------------
        # 3. Pearl's Causal Hierarchy Contradictions (Rules 11-14)
        # -------------------------------------------------------------------
        causal_slots = [s for s in slot_names if s.startswith("CAUSAL_")]
        if (
            "CAUSAL_DIRECT_MECHANISM" in slot_names and "CAUSAL_PREVENTIVE_BLOCK" in slot_names
        ) or (
            "CAUSAL_ENABLING_CONDITION" in slot_names and "CAUSAL_PREVENTIVE_BLOCK" in slot_names
        ) or (
            "CAUSAL_COMMON_CONFOUNDER" in slot_names and "CAUSAL_COLLIDER_EFFECT" in slot_names
        ):
            summary = (
                f"Simultaneous mutually exclusive causal assertions ({', '.join(causal_slots)}). "
                "Direct causal mechanism and preventive blocking cannot co-occur on the same relation."
            )
            diag = MUCDiagnostic(
                category="causal",
                summary=summary,
                conflicting_nodes=muc_nodes,
                conflicting_slots=muc_slots,
                details=["Separate positive causal mechanisms from preventive inhibiting blockers."],
            )
            diag.repair_prompt = diag.format_repair_request()
            return diag

        # -------------------------------------------------------------------
        # 4. Epistemic & Deontic Modal Consistency (Rules 19-21)
        # -------------------------------------------------------------------
        if "EPIST_DEONTIC_OBLIGATION" in slot_names and "EPIST_DEONTIC_PROHIBITION" in slot_names:
            summary = "Simultaneous deontic obligation and prohibition on the same action."
            diag = MUCDiagnostic(
                category="epistemic",
                summary=summary,
                conflicting_nodes=muc_nodes,
                conflicting_slots=muc_slots,
                details=["An action cannot be simultaneously mandatory and strictly prohibited."],
            )
            diag.repair_prompt = diag.format_repair_request()
            return diag

        if "EPIST_DIRECT_OBSERVATION" in slot_names and "EPIST_HEARSAY_TESTIMONY" in slot_names:
            summary = "Direct first-person observation cannot be marked as hearsay testimony."
            diag = MUCDiagnostic(
                category="epistemic",
                summary=summary,
                conflicting_nodes=muc_nodes,
                conflicting_slots=muc_slots,
                details=["Distinguish direct witness observation from unverified third-party hearsay."],
            )
            diag.repair_prompt = diag.format_repair_request()
            return diag

        # -------------------------------------------------------------------
        # 5. Structural Topology Invariants (Rules 15-18)
        # -------------------------------------------------------------------
        if "GRAPH_ROOT_NODE" in slot_names and "GRAPH_LEAF" in slot_names:
            summary = "Node cannot be simultaneously a root proposition and a terminal leaf."
            diag = MUCDiagnostic(
                category="structural",
                summary=summary,
                conflicting_nodes=muc_nodes,
                conflicting_slots=muc_slots,
                details=["Root nodes initiate evaluation flows; terminal leaves terminate them."],
            )
            diag.repair_prompt = diag.format_repair_request()
            return diag

        # -------------------------------------------------------------------
        # 6. Fallback General Diagnostic
        # -------------------------------------------------------------------
        err_msg = "; ".join(errors) if errors else "Ontological contradiction detected in ASP solver core"
        summary = f"Symbolic constraint violation: {err_msg}"
        diag = MUCDiagnostic(
            category="unknown",
            summary=summary,
            conflicting_nodes=muc_nodes,
            conflicting_slots=muc_slots,
            details=[f"Conflicting slots: {muc_slots}"] if muc_slots else [],
        )
        diag.repair_prompt = diag.format_repair_request()
        return diag


class MUCRepairManager:
    """Closed-loop prompt repair manager enforcing a strict maximum repair cap (<= 2 retries)."""

    def __init__(
        self,
        gate: Optional[ClingoVerificationGate] = None,
        compiler: Optional[ASGCompiler] = None,
        max_repair_attempts: int = 2,
    ):
        self.gate = gate or ClingoVerificationGate()
        self.compiler = compiler or ASGCompiler(validator_gate=self.gate.validator_gate)
        self.max_repair_attempts = max_repair_attempts

    def repair_chunk(
        self,
        text: str,
        transducer: Any,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        initial_sexpr: Optional[str] = None,
        **kwargs,
    ) -> RepairResult:
        """Orchestrates closed-loop verification and repair for a single discourse chunk.

        Cycle:
        1. Compile candidate S-expression to QuantaGraph.
        2. Run ClingoVerificationGate.validate_graph(graph).
        3. If valid: Commit graph and return success.
        4. If invalid:
           - Increment repair counter.
           - Enforce repair cap: If attempts > 2, halt cleanly and report unresolvable conflict.
           - Extract MUC diagnostic and generate [REPAIR REQUEST] prompt block.
           - Re-prompt transducer with repair context.
           - Re-parse and re-verify.
        """
        attempts = 0
        diagnostics: List[MUCDiagnostic] = []

        # 1. Acquire initial candidate S-expression
        if initial_sexpr is not None and initial_sexpr.strip():
            current_sexpr = initial_sexpr.strip()
        else:
            current_sexpr = self._invoke_transducer(
                transducer=transducer,
                text=text,
                active_entities=active_entities,
                chunk_id=chunk_id,
                repair_request=None,
                **kwargs,
            )

        while True:
            # 2. Parse and compile candidate S-expression
            extraction_res: Optional[DiscourseExtractionResult] = None
            graph: Optional[QuantaGraph] = None
            compile_err: Optional[Exception] = None

            try:
                extraction_res = parse_sexpr(current_sexpr)
                # Compile without throwing directly to handle validation through gate
                graph = self.compiler.compile(extraction_res, validate=False)
            except ASGCompilationError as ace:
                compile_err = ace
                if ace.validation_result:
                    diag_res = MUCDiagnosticResult(
                        is_valid=False,
                        diagnostic=self.gate.diagnose_result(
                            val_res=ace.validation_result,
                            graph=graph,
                            extraction_result=extraction_res,
                        ),
                        validation_result=ace.validation_result,
                        errors=ace.validation_result.errors,
                    )
                else:
                    diag = MUCDiagnostic(
                        category="structural",
                        summary=str(ace),
                        details=["Compilation failed prior to symbolic solver grounding."],
                    )
                    diag.repair_prompt = diag.format_repair_request()
                    diag_res = MUCDiagnosticResult(is_valid=False, diagnostic=diag, errors=[str(ace)])
            except Exception as e:
                compile_err = e
                diag = MUCDiagnostic(
                    category="structural",
                    summary=f"S-expression syntax or compilation error: {e}",
                    details=["Ensure valid S-expression syntax conforming to GBNF grammar."],
                )
                diag.repair_prompt = diag.format_repair_request()
                diag_res = MUCDiagnosticResult(is_valid=False, diagnostic=diag, errors=[str(e)])

            # If compilation succeeded, perform Clingo verification
            if compile_err is None and graph is not None:
                diag_res = self.gate.validate_graph(graph, extraction_result=extraction_res)

            # 3. If valid: return success
            if diag_res.is_valid and graph is not None:
                return RepairResult(
                    success=True,
                    graph=graph,
                    final_sexpr=current_sexpr,
                    extraction_result=extraction_res,
                    attempts=attempts,
                    diagnostics=diagnostics,
                )

            # 4. If invalid: record diagnostic
            diagnostic = diag_res.diagnostic or self.gate.diagnose_result(
                val_res=diag_res.validation_result or ValidationResult(is_valid=False, errors=diag_res.errors),
                graph=graph,
                extraction_result=extraction_res,
            )
            diagnostics.append(diagnostic)

            # 5. Enforce Repair Cap (maximum 2 repair attempts)
            if attempts >= self.max_repair_attempts:
                logger.warning(
                    "MUCRepairManager reached maximum repair attempts (%d) without resolution: %s",
                    self.max_repair_attempts,
                    diagnostic.summary,
                )
                return RepairResult(
                    success=False,
                    graph=graph,
                    final_sexpr=current_sexpr,
                    extraction_result=extraction_res,
                    attempts=attempts,
                    diagnostics=diagnostics,
                    error_message=(
                        f"MUC repair cap exceeded ({self.max_repair_attempts} attempts). "
                        f"Unresolvable conflict: {diagnostic.summary}"
                    ),
                )

            # 6. Prepare [REPAIR REQUEST] prompt and increment attempt counter
            attempts += 1
            repair_request = diagnostic.format_repair_request()
            logger.info(
                "MUC repair attempt %d/%d initiated. Diagnostic: %s",
                attempts,
                self.max_repair_attempts,
                diagnostic.summary,
            )

            # 7. Re-prompt transducer with repair context
            current_sexpr = self._invoke_transducer(
                transducer=transducer,
                text=text,
                active_entities=active_entities,
                chunk_id=chunk_id,
                repair_request=repair_request,
                attempt=attempts,
                diagnostic=diagnostic,
                **kwargs,
            )

    def verify_and_repair_sexpr(
        self,
        sexpr: str,
        text: str,
        transducer: Any,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        **kwargs,
    ) -> RepairResult:
        """Alias to verify an already generated S-expression and repair if needed."""
        return self.repair_chunk(
            text=text,
            transducer=transducer,
            active_entities=active_entities,
            chunk_id=chunk_id,
            initial_sexpr=sexpr,
            **kwargs,
        )

    def _invoke_transducer(
        self,
        transducer: Any,
        text: str,
        active_entities: Optional[List[EntityRecord]],
        chunk_id: Optional[str],
        repair_request: Optional[str],
        **kwargs,
    ) -> str:
        """Invoke transducer via duck typing supporting transduce_raw, transduce, or callable."""
        if hasattr(transducer, "transduce_raw"):
            return transducer.transduce_raw(
                text=text,
                active_entities=active_entities,
                chunk_id=chunk_id,
                repair_request=repair_request,
                **kwargs,
            )
        elif hasattr(transducer, "transduce"):
            try:
                res = transducer.transduce(
                    text=text,
                    active_entities=active_entities,
                    chunk_id=chunk_id,
                    repair_request=repair_request,
                    **kwargs,
                )
            except TypeError:
                res = transducer.transduce(
                    chunk_text=text,
                    chunk_id=chunk_id,
                    active_manifest_prompt=kwargs.get("active_manifest_prompt"),
                    **kwargs,
                )
            if isinstance(res, DiscourseExtractionResult):
                return to_sexpr(res)
            return str(res)
        elif callable(transducer):
            res = transducer(
                text=text,
                active_entities=active_entities,
                chunk_id=chunk_id,
                repair_request=repair_request,
                **kwargs,
            )
            if isinstance(res, DiscourseExtractionResult):
                return to_sexpr(res)
            return str(res)
        else:
            raise TypeError(
                f"Transducer must implement 'transduce_raw', 'transduce', or be callable. Got: {type(transducer)}"
            )


__all__ = [
    "MUCDiagnostic",
    "MUCDiagnosticResult",
    "ClingoVerificationGate",
    "MUCRepairManager",
    "RepairResult",
]
