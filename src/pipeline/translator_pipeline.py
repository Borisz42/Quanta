"""Unified Two-Way Translation & Execution Pipeline for QUANTA.

Integrates Forward Parsers, Symbolic Validation Gates, Merkle Address Lookup,
and Reverse Realizers into an end-to-end verifiable pipeline with structured
per-stage telemetry logging and latency tracking.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, QuaternaryValue
from parser.nlp_forward import NLPForwardParser
from parser.fol_parser import FOLParser
from parser.ast_parser import ASTForwardParser
from solver.validator_gate import ValidationGate, ValidationResult
from realizer.english_nlg import EnglishRealizer
from realizer.fol_emitter import FOLEmitter
from realizer.code_emitter import CodeEmitter

logger = logging.getLogger("quanta.pipeline")


@dataclass
class StageLog:
    """Structured telemetry entry for an individual translation or verification stage."""
    stage_name: str
    duration_ms: float
    status: str  # "success" | "failed" | "skipped"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "details": self.details,
        }


@dataclass
class TranslationResult:
    input_text: str
    input_modality: str
    target_modality: str
    output_text: str
    graph: Optional[QuantaGraph] = None
    validation: Optional[ValidationResult] = None
    merkle_root: Optional[str] = None
    is_success: bool = True
    error_message: Optional[str] = None
    stage_timings: Dict[str, float] = field(default_factory=dict)
    stage_logs: List[StageLog] = field(default_factory=list)
    total_duration_ms: float = 0.0


@dataclass
class RoundTripResult:
    original_input: str
    input_modality: str
    realized_output: str
    reparsed_output: Optional[str] = None
    original_vector: Optional[QuantaVector] = None
    reparsed_vector: Optional[QuantaVector] = None
    hamming_distance: int = 0
    slot_preservation_rate: float = 1.0
    validation_pass: bool = True
    muc_errors: List[str] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    stage_logs: List[StageLog] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def is_invariant(self, max_hamming: int = 0) -> bool:
        """Checks if the round-trip is invariant within allowed Hamming tolerance."""
        return self.validation_pass and self.hamming_distance <= max_hamming


class TwoWayTranslationPipeline:
    """End-to-end bidirectional translation and neuro-symbolic verification pipeline."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        offline_cache_path: Optional[str] = None,
        rules_path: Optional[str] = None,
    ):
        self.nlp_parser = NLPForwardParser(spacy_model=spacy_model, offline_cache_path=offline_cache_path)
        self.fol_parser = FOLParser(offline_cache_path=offline_cache_path)
        self.ast_parser = ASTForwardParser(offline_cache_path=offline_cache_path)
        self.validator = ValidationGate(rules_path=rules_path)

        # Realizers
        self.english_realizer = EnglishRealizer()
        self.fol_emitter = FOLEmitter()
        self.code_emitter = CodeEmitter()

    def detect_modality(self, input_data: Union[str, Any]) -> str:
        """Detects whether input is English, First-Order Logic, or Python Code."""
        if not isinstance(input_data, str):
            return "python"

        text = input_data.strip()
        # FOL indicators
        if any(tok in text for tok in ("\\forall", "\\exists", "∀", "∃", "\\rightarrow", "->", "\\land", "\\lor", "\\oplus", "\\neg", "¬")):
            return "fol"

        # Python Code indicators
        if text.startswith(("def ", "class ", "import ", "from ", "for ", "while ", "if ")) or "\n    " in text:
            try:
                import ast
                ast.parse(text)
                return "python"
            except Exception:
                pass

        return "english"

    def translate_forward(
        self,
        input_data: Union[str, Any],
        modality: str = "auto",
        domain_context: Optional[str] = None,
        validate: bool = True,
    ) -> Tuple[QuantaGraph, ValidationResult]:
        """Translates natural text, FOL, or code into a verified QuantaGraph ASG."""
        if modality == "auto":
            modality = self.detect_modality(input_data)

        # 1. Parse Input to ASG
        if modality == "fol":
            graph = self.fol_parser.parse_formula(str(input_data))
        elif modality == "python":
            if isinstance(input_data, str):
                import ast
                ast_root = ast.parse(input_data)
                graph = self.ast_parser.parse_ast_node(ast_root, source_text=input_data)
            else:
                graph = self.ast_parser.parse_ast_node(input_data)
        else:
            graph = self.nlp_parser.parse_sentence(str(input_data), domain_context=domain_context)

        # 2. Structural & Neuro-Symbolic Validation
        if validate:
            val_res = self.validator.validate_graph(graph)
        else:
            val_res = ValidationResult(is_valid=True)

        return graph, val_res

    def translate_reverse(
        self,
        graph: QuantaGraph,
        target_modality: str = "english",
    ) -> str:
        """Reconstructs text, logic, or code from a QuantaGraph ASG."""
        mod = target_modality.lower()
        if mod in ("english", "en"):
            return self.english_realizer.realize_graph(graph)
        elif mod in ("fol", "logic"):
            return self.fol_emitter.emit_formula(graph)
        elif mod in ("python", "code", "py"):
            return self.code_emitter.emit_code(graph)
        else:
            raise ValueError(f"Unsupported target modality '{target_modality}'")

    def execute_translation(
        self,
        input_data: str,
        target_modality: str = "english",
        source_modality: str = "auto",
    ) -> TranslationResult:
        """Executes full translation with per-stage structured telemetry and validation gating."""
        start_total = time.perf_counter()
        stage_timings: Dict[str, float] = {}
        stage_logs: List[StageLog] = []

        def _log_stage(name: str, duration: float, status: str, details: Dict[str, Any]):
            dur_ms = duration * 1000.0
            stage_timings[name] = round(dur_ms, 3)
            s_log = StageLog(stage_name=name, duration_ms=dur_ms, status=status, details=details)
            stage_logs.append(s_log)
            logger.info("Pipeline stage completed: %s", json.dumps(s_log.to_dict()))

        # Stage 1: Modality Detection
        t0 = time.perf_counter()
        if source_modality == "auto":
            detected_modality = self.detect_modality(input_data)
        else:
            detected_modality = source_modality
        _log_stage("modality_detection", time.perf_counter() - t0, "success", {
            "source_modality": detected_modality,
            "target_modality": target_modality,
        })

        # Stage 2: Forward Parsing
        t0 = time.perf_counter()
        try:
            if detected_modality == "fol":
                graph = self.fol_parser.parse_formula(str(input_data))
            elif detected_modality == "python":
                if isinstance(input_data, str):
                    import ast
                    ast_root = ast.parse(input_data)
                    graph = self.ast_parser.parse_ast_node(ast_root, source_text=input_data)
                else:
                    graph = self.ast_parser.parse_ast_node(input_data)
            else:
                graph = self.nlp_parser.parse_sentence(str(input_data))

            node_count = len(graph.nodes) if graph else 0
            _log_stage("forward_parse", time.perf_counter() - t0, "success", {
                "node_count": node_count,
                "root_cid": graph.root_cid if graph else None,
            })
        except Exception as e:
            _log_stage("forward_parse", time.perf_counter() - t0, "failed", {"error": str(e)})
            total_duration = (time.perf_counter() - start_total) * 1000.0
            return TranslationResult(
                input_text=str(input_data),
                input_modality=detected_modality,
                target_modality=target_modality,
                output_text="",
                is_success=False,
                error_message=f"Forward parser failure: {e}",
                stage_timings=stage_timings,
                stage_logs=stage_logs,
                total_duration_ms=round(total_duration, 3),
            )

        # Stage 3: Symbolic Validation Gate
        t0 = time.perf_counter()
        val_res = self.validator.validate_graph(graph)
        val_status = "success" if val_res.is_valid else "failed"
        _log_stage("validation_gate", time.perf_counter() - t0, val_status, {
            "is_valid": val_res.is_valid,
            "error_count": len(val_res.errors),
            "errors": val_res.errors,
            "muc_nodes": val_res.muc_nodes,
        })

        if not val_res.is_valid:
            total_duration = (time.perf_counter() - start_total) * 1000.0
            return TranslationResult(
                input_text=str(input_data),
                input_modality=detected_modality,
                target_modality=target_modality,
                output_text="",
                graph=graph,
                validation=val_res,
                is_success=False,
                error_message=f"Validation failed with MUC errors: {val_res.errors}",
                stage_timings=stage_timings,
                stage_logs=stage_logs,
                total_duration_ms=round(total_duration, 3),
            )

        # Stage 4: Merkle Addressing
        t0 = time.perf_counter()
        merkle_root = graph.compute_merkle_root() if graph else None
        _log_stage("merkle_addressing", time.perf_counter() - t0, "success", {
            "merkle_root": merkle_root,
        })

        # Stage 5: Reverse Realization
        t0 = time.perf_counter()
        output_text = self.translate_reverse(graph, target_modality=target_modality)
        _log_stage("reverse_realization", time.perf_counter() - t0, "success", {
            "target_modality": target_modality,
            "output_length": len(output_text),
        })

        total_duration = (time.perf_counter() - start_total) * 1000.0
        return TranslationResult(
            input_text=str(input_data),
            input_modality=detected_modality,
            target_modality=target_modality,
            output_text=output_text,
            graph=graph,
            validation=val_res,
            merkle_root=merkle_root,
            is_success=True,
            stage_timings=stage_timings,
            stage_logs=stage_logs,
            total_duration_ms=round(total_duration, 3),
        )

    def round_trip(
        self,
        input_data: str,
        modality: str = "auto",
    ) -> RoundTripResult:
        """Executes full Input -> Quanta ASG -> Output -> Reparsed ASG invariance check with telemetry."""
        start_total = time.perf_counter()
        stage_timings: Dict[str, float] = {}
        stage_logs: List[StageLog] = []

        def _log_stage(name: str, duration: float, status: str, details: Dict[str, Any]):
            dur_ms = duration * 1000.0
            stage_timings[name] = round(dur_ms, 3)
            s_log = StageLog(stage_name=name, duration_ms=dur_ms, status=status, details=details)
            stage_logs.append(s_log)
            logger.info("Pipeline round-trip stage: %s", json.dumps(s_log.to_dict()))

        # Stage 1: Forward Pass 1
        t0 = time.perf_counter()
        if modality == "auto":
            detected_modality = self.detect_modality(input_data)
        else:
            detected_modality = modality

        g1, v1 = self.translate_forward(input_data, modality=detected_modality)
        _log_stage("forward_pass_1", time.perf_counter() - t0, "success" if v1.is_valid else "failed", {
            "is_valid": v1.is_valid,
            "node_count": len(g1.nodes) if g1 else 0,
            "errors": v1.errors,
        })

        if not v1.is_valid:
            total_duration = (time.perf_counter() - start_total) * 1000.0
            return RoundTripResult(
                original_input=input_data,
                input_modality=detected_modality,
                realized_output="",
                validation_pass=False,
                muc_errors=v1.errors,
                stage_timings=stage_timings,
                stage_logs=stage_logs,
                total_duration_ms=round(total_duration, 3),
            )

        # Stage 2: Reverse Realization
        t0 = time.perf_counter()
        out_text = self.translate_reverse(g1, target_modality=detected_modality)
        _log_stage("reverse_realization", time.perf_counter() - t0, "success", {
            "out_text": out_text,
        })

        # Stage 3: Forward Pass 2 (Reparse)
        t0 = time.perf_counter()
        g2, v2 = self.translate_forward(out_text, modality=detected_modality)
        _log_stage("forward_pass_2_reparse", time.perf_counter() - t0, "success" if v2.is_valid else "failed", {
            "is_valid": v2.is_valid,
            "node_count": len(g2.nodes) if g2 else 0,
        })

        # Stage 4: Compare Vectors & Calculate Invariance
        t0 = time.perf_counter()
        vec1 = g1.to_proposition_vector()
        vec2 = g2.to_proposition_vector()

        hamming = vec1.hamming_distance(vec2)
        active1 = set(vec1.active_slots().keys())
        active2 = set(vec2.active_slots().keys())

        total_slots = len(active1 | active2)
        matching_slots = sum(1 for s in (active1 | active2) if vec1[s] == vec2[s])
        preservation = matching_slots / max(total_slots, 1)
        _log_stage("invariance_audit", time.perf_counter() - t0, "success", {
            "hamming_distance": hamming,
            "preservation_rate": preservation,
        })

        total_duration = (time.perf_counter() - start_total) * 1000.0
        return RoundTripResult(
            original_input=input_data,
            input_modality=detected_modality,
            realized_output=out_text,
            reparsed_output=out_text,
            original_vector=vec1,
            reparsed_vector=vec2,
            hamming_distance=hamming,
            slot_preservation_rate=preservation,
            validation_pass=v2.is_valid,
            muc_errors=v2.errors if not v2.is_valid else [],
            stage_timings=stage_timings,
            stage_logs=stage_logs,
            total_duration_ms=round(total_duration, 3),
        )


# Canonical Alias for TwoWayTranslationPipeline
TranslatorPipeline = TwoWayTranslationPipeline
