"""Unified Two-Way Translation & Execution Pipeline for QUANTA.

Integrates Forward Parsers, Symbolic Validation Gates, Merkle Address Lookup,
and Reverse Realizers into an end-to-end verifiable pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
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
        """Executes full translation with validation gating."""
        if source_modality == "auto":
            source_modality = self.detect_modality(input_data)

        graph, val_res = self.translate_forward(input_data, modality=source_modality)

        if not val_res.is_valid:
            return TranslationResult(
                input_text=str(input_data),
                input_modality=source_modality,
                target_modality=target_modality,
                output_text="",
                graph=graph,
                validation=val_res,
                is_success=False,
                error_message=f"Validation failed with MUC errors: {val_res.errors}",
            )

        output_text = self.translate_reverse(graph, target_modality=target_modality)
        return TranslationResult(
            input_text=str(input_data),
            input_modality=source_modality,
            target_modality=target_modality,
            output_text=output_text,
            graph=graph,
            validation=val_res,
            merkle_root=graph.compute_merkle_root(),
            is_success=True,
        )

    def round_trip(
        self,
        input_data: str,
        modality: str = "auto",
    ) -> RoundTripResult:
        """Executes full Input -> Quanta ASG -> Output -> Reparsed ASG invariance check."""
        if modality == "auto":
            modality = self.detect_modality(input_data)

        # 1. Forward Pass 1
        g1, v1 = self.translate_forward(input_data, modality=modality)
        if not v1.is_valid:
            return RoundTripResult(
                original_input=input_data,
                input_modality=modality,
                realized_output="",
                validation_pass=False,
                muc_errors=v1.errors,
            )

        # 2. Reverse Realization
        out_text = self.translate_reverse(g1, target_modality=modality)

        # 3. Forward Pass 2
        g2, v2 = self.translate_forward(out_text, modality=modality)

        # 4. Compare Vectors
        vec1 = g1.to_proposition_vector()
        vec2 = g2.to_proposition_vector()

        hamming = vec1.hamming_distance(vec2)
        active1 = set(vec1.active_slots().keys())
        active2 = set(vec2.active_slots().keys())

        total_slots = len(active1 | active2)
        matching_slots = sum(1 for s in (active1 | active2) if vec1[s] == vec2[s])
        preservation = matching_slots / max(total_slots, 1)

        return RoundTripResult(
            original_input=input_data,
            input_modality=modality,
            realized_output=out_text,
            reparsed_output=out_text,
            original_vector=vec1,
            reparsed_vector=vec2,
            hamming_distance=hamming,
            slot_preservation_rate=preservation,
            validation_pass=v2.is_valid,
            muc_errors=v2.errors if not v2.is_valid else [],
        )
