"""Symbolic Abstract Syntax Graph (ASG) Compiler for QUANTA (Phase 4).

Transforms normalized JSON intermediate representations produced by the Neural
Discourse Transducer (Phase 3) into deterministic, content-addressed 1024-dimensional
quaternary Abstract Syntax Graphs (QuantaGraph / QuantaNode) with:
1. ConceptNet 5.7.0 (403,503 concepts) and WordNet lexical grounding (Bands 3 & 4).
2. 1024-dimensional quaternary vector synthesis (\u03a3^1024, 256 bytes per node).
3. Band 1 thematic valencies (VAL_X1_AGENT, VAL_X2_PATIENT, VAL_LOCATION_SLOT, VAL_X5_INSTRUMENT).
4. Band 2 variable scoping registers (VAR_SLOT_X0..X7) with RegisterValue.BOUND_LOCAL.
5. Band 5 & 6 Theory of Mind and Epistemic/Deontic logic constraints.
6. Band 7 spatio-temporal Allen intervals and Pearl causal DAG links.
7. Bottom-up 256-bit BLAKE3 cryptographic CID computation.
8. Clingo ASP ValidatorGate constraint satisfaction and Minimal Unsatisfiable Core (MUC) extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.slots import (
    SLOT_NAME_TO_INDEX,
    get_slot_by_name,
)
from core.types import (
    QuantaVector,
    RegisterValue,
    StructuralValue,
)
from parser.lexical_grounder import (
    ConceptNetLexicalGrounder,
    WordNetLexicalGrounder,
)
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)
from solver.validator_gate import ValidationGate, ValidationResult, ValidatorGate


class ASGCompilationError(Exception):
    """Raised when ASG compilation or Clingo validation fails."""

    def __init__(self, message: str, validation_result: Optional[ValidationResult] = None):
        super().__init__(message)
        self.validation_result = validation_result


class ASGCompiler:
    """Compiles normalized extraction schemas into cryptographic, type-checked Quanta ASGs."""

    # Core verb lemma classification maps for Band 0 NSM primes
    COGNITION_VERBS: Set[str] = {
        "think", "believe", "know", "doubt", "verify", "suppose", "assume",
        "infer", "deduce", "judge", "consider", "reckon", "reflect", "ponder",
        "comprehend", "recognize", "replicate", "hypothesize",
    }
    PERCEPTION_VERBS: Set[str] = {
        "see", "look", "watch", "observe", "note", "notice", "view", "glance",
        "spot", "perceive", "witness", "detect", "discover",
    }
    COMMUNICATION_VERBS: Set[str] = {
        "say", "tell", "speak", "talk", "prohibit", "warn", "command", "declare",
        "state", "report", "describe", "explain", "argue", "mention", "assert",
        "suggest", "propose", "instruct", "order", "forbid", "audit",
    }
    MOTION_VERBS: Set[str] = {
        "move", "go", "walk", "run", "travel", "fly", "jump", "enter", "leave",
        "cross", "drive", "ride", "step", "pass", "shift", "head", "crawl",
        "swim", "chase", "pursue", "arrive", "depart", "reach", "approach",
        "flee", "escape", "return", "navigate",
    }
    ACTION_VERBS: Set[str] = {
        "isolate", "retain", "pressurize", "make", "build", "create", "test",
        "synthesize", "extract", "heat", "cool", "mix", "separate", "measure",
        "modify", "apply", "execute", "perform", "touch", "hit", "grab", "push",
        "pull", "carry", "place", "put", "transform",
    }
    LIFE_VERBS: Set[str] = {
        "live", "die", "born", "grow", "breathe", "perish", "survive", "decay", "sprout",
    }
    POSSESSION_VERBS: Set[str] = {
        "have", "own", "possess", "hold", "keep", "acquire", "get", "give",
        "receive", "buy", "sell", "obtain", "gain", "retain",
    }

    def __init__(
        self,
        conceptnet_grounder: Optional[ConceptNetLexicalGrounder] = None,
        wordnet_grounder: Optional[WordNetLexicalGrounder] = None,
        validator_gate: Optional[ValidationGate] = None,
    ):
        """Initialize the compiler with grounding backends and symbolic validation gate."""
        self.conceptnet_grounder = conceptnet_grounder or ConceptNetLexicalGrounder.get_default()
        self.wordnet_grounder = wordnet_grounder or WordNetLexicalGrounder.get_default()
        self.validator_gate = validator_gate or ValidatorGate()

    def compile(
        self,
        extraction: Union[DiscourseExtractionResult, str, Dict[str, Any]],
        validate: bool = True,
    ) -> QuantaGraph:
        """Compiles an extraction result into a validated QuantaGraph.

        Args:
            extraction: DiscourseExtractionResult instance, JSON string, or dict.
            validate: Whether to run Clingo ASP validation on the resulting graph.

        Returns:
            Fully compiled, content-addressed QuantaGraph.

        Raises:
            ASGCompilationError: If foreign keys are missing or Clingo validation fails.
        """
        # 1. Normalize input to DiscourseExtractionResult
        if isinstance(extraction, str):
            extraction_result = DiscourseExtractionResult.from_json(extraction)
        elif isinstance(extraction, dict):
            extraction_result = DiscourseExtractionResult.from_dict(extraction)
        elif isinstance(extraction, DiscourseExtractionResult):
            extraction_result = extraction
        else:
            raise TypeError(f"Unsupported extraction input type: {type(extraction)}")

        # 2. Foreign-key cross-reference validation
        fk_errors = extraction_result.validate_foreign_keys()
        if fk_errors:
            raise ASGCompilationError(
                f"Foreign-key validation failed before compilation: {fk_errors}"
            )

        graph = QuantaGraph()
        entity_nodes: Dict[str, QuantaNode] = {}
        event_nodes: Dict[str, QuantaNode] = {}

        # 3. Entity compilation (Phase 4.2)
        for i, ent in enumerate(extraction_result.entities):
            node = self.compile_entity(ent, register_index=i)
            cid = graph.add_node(node)
            entity_nodes[ent.id] = node

        # 4. Event compilation (Phase 4.3)
        entity_cids = {ent_id: node.cid for ent_id, node in entity_nodes.items()}
        for i, ev in enumerate(extraction_result.events):
            node = self.compile_event(
                ev,
                entity_cids=entity_cids,
                propositions=extraction_result.propositions,
            )
            if i == 0:
                node.set_slot("GRAPH_ROOT_NODE", 1)

            # Set thematic valency structural routing slots BEFORE computing CID
            if ev.agent_id and ev.agent_id in entity_nodes:
                node.set_structural_slot("VAL_X1_AGENT", StructuralValue.ACTIVE_LOCAL)
            if ev.patient_id and ev.patient_id in entity_nodes:
                node.set_structural_slot("VAL_X2_PATIENT", StructuralValue.ACTIVE_LOCAL)
            elif ev.theme_id and ev.theme_id in entity_nodes:
                node.set_structural_slot("VAL_X2_PATIENT", StructuralValue.ACTIVE_LOCAL)
            if ev.location_id and ev.location_id in entity_nodes:
                node.set_structural_slot("VAL_LOCATION_SLOT", StructuralValue.ACTIVE_LOCAL)
            if ev.instrument_id and ev.instrument_id in entity_nodes:
                node.set_structural_slot("VAL_X5_INSTRUMENT", StructuralValue.ACTIVE_LOCAL)

            # Set self-relations and causal slots BEFORE computing CID
            for rel in extraction_result.relations:
                rel_type = rel.relation_type
                if rel_type == "CAUSAL_MECHANISM_LINK":
                    slot_name = "CAUSAL_DIRECT_MECHANISM"
                elif rel_type == "CAUSAL_PREVENTIVE_BLOCK":
                    slot_name = "CAUSAL_PREVENTIVE_BLOCK"
                else:
                    slot_name = rel_type

                if rel.source_id == rel.target_id == ev.id:
                    if slot_name in SLOT_NAME_TO_INDEX:
                        node.set_slot(slot_name, 1)
                elif rel.source_id == ev.id and slot_name in SLOT_NAME_TO_INDEX and slot_name.startswith("CAUSAL_"):
                    node.set_slot(slot_name, 1)

            node.compute_cid()
            graph.add_node(node)
            event_nodes[ev.id] = node

        # 5. Wire Thematic Valencies from events to entities
        for ev in extraction_result.events:
            ev_node = event_nodes[ev.id]
            if ev.agent_id and ev.agent_id in entity_nodes:
                ent_node = entity_nodes[ev.agent_id]
                graph.add_edge(ev_node.cid, "VAL_X1_AGENT", ent_node.cid)

            if ev.patient_id and ev.patient_id in entity_nodes:
                ent_node = entity_nodes[ev.patient_id]
                graph.add_edge(ev_node.cid, "VAL_X2_PATIENT", ent_node.cid)
            elif ev.theme_id and ev.theme_id in entity_nodes:
                ent_node = entity_nodes[ev.theme_id]
                graph.add_edge(ev_node.cid, "VAL_X2_PATIENT", ent_node.cid)

            if ev.location_id and ev.location_id in entity_nodes:
                ent_node = entity_nodes[ev.location_id]
                graph.add_edge(ev_node.cid, "VAL_LOCATION_SLOT", ent_node.cid)

            if ev.instrument_id and ev.instrument_id in entity_nodes:
                ent_node = entity_nodes[ev.instrument_id]
                graph.add_edge(ev_node.cid, "VAL_X5_INSTRUMENT", ent_node.cid)

        # 6. Spatio-temporal & causal edge wiring (Phase 4.4)
        for rel in extraction_result.relations:
            if rel.source_id != rel.target_id:
                self._wire_relation(rel, graph, entity_nodes, event_nodes)

        # 7. Designate Root Event Node
        if extraction_result.events:
            root_ev_id = extraction_result.events[0].id
            graph.root_cid = event_nodes[root_ev_id].cid

        # 8. Structural integrity validation
        struct_valid, struct_errors = graph.validate_integrity()
        if not struct_valid:
            raise ASGCompilationError(
                f"Graph structural integrity verification failed: {struct_errors}"
            )

        # 9. Symbolic ValidatorGate Integration (Phase 4.5)
        if validate:
            val_res = self.validator_gate.validate_graph(graph)
            # Attach validation result to graph for downstream consumers
            setattr(graph, "validation", val_res)
            if not val_res.is_valid:
                raise ASGCompilationError(
                    f"Clingo ASP ontological validation failed with {len(val_res.errors)} error(s): {val_res.errors}",
                    validation_result=val_res,
                )

        # 10. Attach extraction result & entity/event maps to graph for honest NLG realization
        setattr(graph, "extraction_result", extraction_result)
        setattr(graph, "entity_nodes", entity_nodes)
        setattr(graph, "event_nodes", event_nodes)

        return graph

    def compile_entity(self, entity: ExtractedEntity, register_index: int = 0) -> QuantaNode:
        """Compiles an ExtractedEntity into a canonical QuantaNode with ConceptNet grounding.

        - Band 2: Assigns bound local variable registers (VAR_SLOT_X0..X7)
        - Band 3 & 4: ConceptNet taxonomy and affordance vectors
        - Band 1: Marks GRAPH_VARIABLE_BIND and GRAPH_LEAF
        """
        node = QuantaNode(literal=entity.canonical_name)

        # 1. Band 2: Variable Register Scoping
        reg_slot = f"VAR_SLOT_X{register_index % 8}"
        node.set_register_slot(reg_slot, RegisterValue.BOUND_LOCAL)
        node.set_slot("GRAPH_VARIABLE_BIND", 1)
        node.set_slot("GRAPH_LEAF", 1)

        # 2. ConceptNet 5.7.0 Lexical Grounding
        clean_name = entity.canonical_name.strip()
        # Remove common prefixes like 'Dr. '
        lookup_name = clean_name
        for prefix in ("dr. ", "dr ", "mr. ", "mr ", "ms. ", "ms ", "prof. ", "prof "):
            if lookup_name.lower().startswith(prefix):
                lookup_name = lookup_name[len(prefix):].strip()
                break

        res = self.conceptnet_grounder.resolve_concept(lookup_name, pos="n")
        if not res and " " in lookup_name:
            # Fallback: try head noun / individual tokens in reverse order
            tokens = lookup_name.split()
            for token in reversed(tokens):
                res = self.conceptnet_grounder.resolve_concept(token, pos="n")
                if res:
                    break

        if not res and self.wordnet_grounder:
            # Fallback to WordNet
            wn_syn = self.wordnet_grounder.resolve_synset(lookup_name, pos="n")
            if wn_syn:
                res = self.conceptnet_grounder.resolve_concept(wn_syn, pos="n")

        if res:
            node.anchor = res.synset_name
            for slot_name, slot_val in res.active_slots.items():
                if slot_name in SLOT_NAME_TO_INDEX:
                    node.set_slot(slot_name, slot_val)
        else:
            anchor_slug = clean_name.lower().replace(" ", "_").replace(".", "")
            node.anchor = f"cn:en:{anchor_slug} (n)"

        # 3. Categorical Ontological Enforcement
        cat = entity.category.upper()
        if cat in ("ABSTRACT", "ABSTRACT_CONCEPT") or (entity.properties and entity.properties.get("abstract")):
            node.set_slot("TYPE_ABSTRACT_CONCEPT", 1)
            node.set_slot("TYPE_INANIMATE_PHYSICAL", 0)
            node.set_slot("TYPE_NATURAL_OBJECT", 0)
            node.set_slot("TYPE_ARTIFACT", 0)
            node.set_slot("ROLE_SENTIENT", 0)
            node.set_slot("TYPE_ANIMATE", 0)
            node.set_slot("TYPE_HUMAN", 0)
            node.set_slot("GRAPH_VARIABLE_BIND", 0)
            if entity.properties and entity.properties.get("agent_capable"):
                node.set_slot("ROLE_AGENT_CAPABLE", 1)
        elif cat == "PERSON":
            node.set_slot("TYPE_HUMAN", 1)
            node.set_slot("TYPE_ANIMATE", 1)
            node.set_slot("ROLE_AGENT_CAPABLE", 1)
            node.set_slot("ROLE_SENTIENT", 1)
            node.set_slot("TYPE_NATURAL_OBJECT", 1)
            node.set_slot("TYPE_ARTIFACT", 0)
            node.set_slot("TYPE_INANIMATE_PHYSICAL", 0)
        elif cat == "SUBSTANCE":
            node.set_slot("CN_Q072_SUBSTANCE", 1)
            node.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
            node.set_slot("ROLE_SENTIENT", 0)
            node.set_slot("TYPE_ANIMATE", 0)
            node.set_slot("TYPE_HUMAN", 0)
        elif cat == "LOCATION":
            node.set_slot("TYPE_SPATIAL_REGION", 1)
            node.set_slot("WN_LOCATION_PLACE", 1)
            node.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
            node.set_slot("ROLE_SENTIENT", 0)
            node.set_slot("TYPE_ANIMATE", 0)
            node.set_slot("TYPE_HUMAN", 0)
        elif cat in ("OBJECT", "ARTIFACT", "INSTRUMENT"):
            node.set_slot("TYPE_ARTIFACT", 1)
            node.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
            node.set_slot("TYPE_NATURAL_OBJECT", 0)
            node.set_slot("ROLE_SENTIENT", 0)
            node.set_slot("TYPE_ANIMATE", 0)
            node.set_slot("TYPE_HUMAN", 0)
            if node.vector["TYPE_ABSTRACT_CONCEPT"] == 1:
                node.set_slot("GRAPH_VARIABLE_BIND", 0)
        elif cat == "NATURAL_OBJECT":
            node.set_slot("TYPE_NATURAL_OBJECT", 1)
            node.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
            node.set_slot("TYPE_ARTIFACT", 0)
            node.set_slot("ROLE_SENTIENT", 0)
            node.set_slot("TYPE_ANIMATE", 0)
            node.set_slot("TYPE_HUMAN", 0)
        elif cat == "ORGANIZATION":
            node.set_slot("TYPE_ORGANIZATION", 1)
            node.set_slot("ROLE_AGENT_CAPABLE", 1)
        elif cat == "ANIMAL":
            node.set_slot("TYPE_ANIMATE", 1)
            node.set_slot("ROLE_AGENT_CAPABLE", 1)
            node.set_slot("ROLE_SENTIENT", 1)

        node.compute_cid()
        return node

    def compile_event(
        self,
        event: ExtractedEvent,
        entity_cids: Dict[str, str],
        propositions: List[ExtractedProposition],
    ) -> QuantaNode:
        """Compiles an ExtractedEvent into a predicate QuantaNode.

        - Band 0: Synthesizes NSM primes (NSM_DO, NSM_MOVE, NSM_THINK, NSM_TRUE)
        - Band 1: Sets Tense & Aspect slots
        - Band 5 & 6: Epistemic & Deontic slots derived from linked ExtractedProposition
        """
        node = QuantaNode(literal=event.raw_text or event.predicate)
        node.set_slot("TYPE_EVENT", 1)
        node.set_slot("WN_ACT_ACTION", 1)

        pred_clean = event.predicate.strip().lower()

        # 1. ConceptNet 5.7.0 Grounding for Verb
        res = self.conceptnet_grounder.resolve_concept(pred_clean, pos="v")
        if not res and self.wordnet_grounder:
            wn_syn = self.wordnet_grounder.resolve_synset(pred_clean, pos="v")
            if wn_syn:
                res = self.conceptnet_grounder.resolve_concept(wn_syn, pos="v")

        if res:
            node.anchor = res.synset_name
            for slot_name, slot_val in res.active_slots.items():
                if slot_name in SLOT_NAME_TO_INDEX:
                    node.set_slot(slot_name, slot_val)
        else:
            node.anchor = f"cn:en:{pred_clean} (v)"

        # 2. Band 0: Map verb lemmas to universal NSM primes
        if pred_clean in self.COGNITION_VERBS:
            node.set_slot("NSM_THINK", 1)
            if pred_clean == "verify":
                node.set_slot("NSM_TRUE", 1)
                node.set_slot("SOLVER_PROOF_VALIDATED", 1)
            elif pred_clean == "doubt":
                node.set_slot("NSM_MAYBE", 3)
        elif pred_clean in self.PERCEPTION_VERBS:
            node.set_slot("NSM_SEE", 1)
            node.set_slot("EPIST_DIRECT_OBSERVATION", 1)
        elif pred_clean in self.COMMUNICATION_VERBS:
            node.set_slot("NSM_SAY", 1)
            if pred_clean in ("prohibit", "forbid"):
                node.set_slot("EPIST_DEONTIC_PROHIBITION", 1)
        elif pred_clean in self.MOTION_VERBS:
            node.set_slot("NSM_MOVE", 1)
        elif pred_clean in self.LIFE_VERBS:
            node.set_slot("NSM_LIVE", 1)
        elif pred_clean in self.POSSESSION_VERBS:
            node.set_slot("NSM_HAVE", 1)
            node.set_slot("NSM_DO", 1)
        else:
            # Default active action prime
            node.set_slot("NSM_DO", 1)

        # 3. Polarity & Modality
        if event.polarity:
            node.set_slot("NSM_TRUE", 1)
        else:
            node.set_slot("LJB_NA_NEGATION", 2)

        mod = (event.modality or "LITERAL").upper()
        if mod == "FIGURATIVE":
            node.set_slot("MODALITY_FIGURATIVE", 1)
        else:
            node.set_slot("MODALITY_LITERAL", 1)

        # 4. Band 1: Grammatical Tense & Aspect
        tense = (event.tense or "PAST").upper()
        if tense in ("PAST", "PAST_PERFECT"):
            node.set_slot("LJB_PU_PAST_TENSE", 1)
        elif tense in ("PRESENT", "PRESENT_PERFECT"):
            node.set_slot("LJB_CA_PRESENT_TENSE", 1)
        elif tense in ("FUTURE", "FUTURE_PERFECT"):
            node.set_slot("LJB_BA_FUTURE_TENSE", 1)

        aspect = (event.aspect or "SIMPLE").upper()
        if aspect == "PERFECT":
            node.set_slot("ASPECT_PERFECTIVE", 1)
        elif aspect == "PROGRESSIVE":
            node.set_slot("ASPECT_PROGRESSIVE", 1)
        elif aspect == "HABITUAL":
            node.set_slot("ASPECT_HABITUAL", 1)
        elif aspect == "ITERATIVE":
            node.set_slot("ASPECT_ITERATIVE", 1)

        # 5. Band 5 & 6: Epistemic & Deontic slots from linked propositions
        for prop in propositions:
            if prop.event_id == event.id:
                st = (prop.epistemic_status or "FACT").upper()
                if st == "OBSERVATION":
                    node.set_slot("EPIST_DIRECT_OBSERVATION", 1)
                elif st == "HYPOTHESIS":
                    node.set_slot("MODALITY_HYPOTHETICAL", 3)
                    node.set_slot("NSM_MAYBE", 3)
                elif st == "DOUBTED":
                    node.set_slot("NSM_MAYBE", 3)
                    node.set_slot("TOM_FIRST_ORDER_BELIEF", 1)
                elif st == "FACT":
                    node.set_slot("NSM_TRUE", 1)
                    node.set_slot("SOLVER_PROOF_VALIDATED", 1)
                elif st == "PROHIBITED":
                    node.set_slot("EPIST_DEONTIC_PROHIBITION", 1)
                elif st == "BELIEF":
                    node.set_slot("TOM_FIRST_ORDER_BELIEF", 1)

        node.compute_cid()
        return node

    def _wire_relation(
        self,
        rel: ExtractedRelation,
        graph: QuantaGraph,
        entity_nodes: Dict[str, QuantaNode],
        event_nodes: Dict[str, QuantaNode],
    ):
        """Wires an ExtractedRelation into the ASG.

        - If source_id == target_id: compiles aspectual duration as an intrinsic node slot.
        - If source_id != target_id: compiles a directed relation edge in QuantaGraph.
        """
        src_id = rel.source_id
        tgt_id = rel.target_id
        rel_type = rel.relation_type

        # Canonicalize causal relation naming
        if rel_type == "CAUSAL_MECHANISM_LINK":
            edge_type = "CAUSAL_MECHANISM_LINK"
            slot_name = "CAUSAL_DIRECT_MECHANISM"
        elif rel_type == "CAUSAL_PREVENTIVE_BLOCK":
            edge_type = "CAUSAL_PREVENTIVE_BLOCK"
            slot_name = "CAUSAL_PREVENTIVE_BLOCK"
        else:
            edge_type = rel_type
            slot_name = rel_type

        # Case A: Self-relation (handled during event compilation)
        if src_id == tgt_id:
            return

        # Case B: Directed edge between distinct nodes
        src_node = event_nodes.get(src_id) or entity_nodes.get(src_id)
        tgt_node = event_nodes.get(tgt_id) or entity_nodes.get(tgt_id)

        if src_node and tgt_node:
            graph.add_edge(src_node.cid, edge_type, tgt_node.cid)


__all__ = [
    "ASGCompiler",
    "ASGCompilationError",
]
