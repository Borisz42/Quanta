"""Clean, over-complete candidate dimension pool (K = 512) for multi-objective dimension selection."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from core.slots import CANONICAL_SLOTS, SLOT_NAME_TO_INDEX


@dataclass(frozen=True)
class CandidateDimension:
    id: int
    name: str
    source: str
    category: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "category": self.category,
            "description": self.description,
        }


def build_candidate_pool() -> List[CandidateDimension]:
    """Compiles a clean, domain-general over-complete pool of 512 semantic/syntactic primitives,
    free from benchmark-specific token pollution.
    """
    candidates: List[CandidateDimension] = []
    curr_id = 0

    def add(name: str, source: str, category: str, description: str):
        nonlocal curr_id
        candidates.append(CandidateDimension(
            id=curr_id,
            name=name,
            source=source,
            category=category,
            description=description,
        ))
        curr_id += 1

    # 1. Canonical Slots (1024 Slots across Bands 0-7)
    for slot in CANONICAL_SLOTS:
        if slot.name.startswith("CN_Q"):
            source = "ConceptNet 5.7.0"
        else:
            source = f"Band{int(slot.band)}_{slot.band.name}"
        add(
            name=slot.name,
            source=source,
            category=slot.category,
            description=slot.description,
        )

    # 2. FrameNet Extended Roles & Frames (60 Candidates)
    framenet_roles = [
        ("FN_EXPERIENCER_PASSIVE", "Core Roles", "fn:Perceiver experiencing sensory/emotional input"),
        ("FN_STIMULUS_EVOKING", "Core Roles", "fn:Entity evoking perception, emotion, or reaction"),
        ("FN_MEANS_METHOD", "Secondary Roles", "fn:Method or action sequence accomplishing goal"),
        ("FN_REASON_EXPLANATION", "Secondary Roles", "fn:Causal explanation for state or event"),
        ("FN_DIRECTION_HEADING", "Spatial Paths", "fn:Orientational heading or vector"),
        ("FN_DISTANCE_SCALAR", "Spatial Paths", "fn:Scalar metric of spatial displacement"),
        ("FN_COMMERCE_BUYING", "Commerce", "fn:Commercial exchange acquiring assets"),
        ("FN_COMMERCE_SELLING", "Commerce", "fn:Commercial exchange offering assets"),
        ("FN_TRANSFER_POSSESSION", "Possession", "fn:Change of ownership or physical holding"),
        ("FN_CO_AGENT_PARTNER", "Core Roles", "fn:Secondary collaborative agent"),
        ("FN_CO_PATIENT_SECONDARY", "Core Roles", "fn:Secondary affected undergoer participant"),
        ("FN_DESCRIPTIVE_PROP", "Modifiers", "fn:Qualifying property or modifier descriptor"),
        ("FN_DEGREE_INTENSITY", "Modifiers", "fn:Scalar intensity or degree modifier"),
        ("FN_CIRCUMSTANTIAL_COND", "Circumstantial", "fn:Situational background context conditions"),
        ("FN_CAUSAL_EFFECT", "Secondary Roles", "fn:Consequence or outcome of causal event"),
        ("FN_COMM_TOPIC", "Communication", "fn:Subject matter or discourse theme"),
        ("FN_COMM_MEDIUM", "Communication", "fn:Physical or symbolic transmission medium"),
        ("FN_COMM_MESSAGE", "Communication", "fn:Informational payload content"),
        ("FN_DONOR_ORIGIN", "Possession", "fn:Giving party in physical/abstract transfer"),
        ("FN_TRANSFERRED_ITEM", "Possession", "fn:Transferred physical or abstract entity"),
        ("FN_OWNER_POSSESSOR", "Possession", "fn:Possessing or holding entity"),
        ("FN_DURATION_INTERVAL", "Circumstantial", "fn:Temporal length of event or state"),
        ("FN_REPETITION_FREQ", "Circumstantial", "fn:Temporal repetition frequency count"),
        ("FN_MEREOLOGY_PART_WHOLE", "Mereology", "fn:Structural composition part-whole link"),
        ("FN_CONTAINER_VESSEL", "Spatial", "fn:Enclosing physical or abstract container"),
        ("FN_ENCLOSED_CONTENTS", "Spatial", "fn:Enclosed or contained entities"),
        ("FN_COGNITION_ACTIVITY", "Mind", "fn:Mental deliberation or cognitive activity"),
        ("FN_PERCEPTION_ACTIVITY", "Perception", "fn:Active sensory gathering (look, listen)"),
        ("FN_STATEMENT_ACT", "Communication", "fn:Explicit communicative assertion act"),
        ("FN_MOTION_TRANSLATION", "Kinematics", "fn:Physical translation across spatial coordinates"),
    ]
    for name, cat, desc in framenet_roles:
        add(name, "FrameNet", cat, desc)

    # 3. Abstract Meaning Representation (AMR) Primitives (30 Candidates)
    amr_relations = [
        ("AMR_PROTO_AGENT", "Core Arguments", "amr:ARG0 - proto-agent / subject"),
        ("AMR_PROTO_PATIENT", "Core Arguments", "amr:ARG1 - proto-patient / object"),
        ("AMR_PROTO_RECIPIENT", "Core Arguments", "amr:ARG2 - proto-beneficiary / destination"),
        ("AMR_PROTO_INSTRUMENT", "Core Arguments", "amr:ARG3 - start point / instrument"),
        ("AMR_PROTO_RESULT", "Core Arguments", "amr:ARG4 - end point / result state"),
        ("AMR_LOCATIVE_COORD", "Spatial / Temporal", "amr:location - spatial coordinate"),
        ("AMR_DESTINATION_TARGET", "Spatial / Temporal", "amr:destination - spatial target"),
        ("AMR_TRAJECTORY_PATH", "Spatial / Temporal", "amr:path - spatial trajectory"),
        ("AMR_TEMPORAL_ANCHOR", "Spatial / Temporal", "amr:time - temporal anchor point"),
        ("AMR_DURATION_SPAN", "Spatial / Temporal", "amr:duration - temporal interval span"),
        ("AMR_ANTECEDENT_COND", "Modifiers", "amr:condition - antecedent premise"),
        ("AMR_CAUSAL_DETERMINANT", "Modifiers", "amr:cause - causal determinant"),
        ("AMR_TELEOLOGICAL_GOAL", "Modifiers", "amr:purpose - teleological goal"),
        ("AMR_CONTRASTIVE_CONCESSION", "Modifiers", "amr:concession - contrastive adversative"),
        ("AMR_QUALITATIVE_MANNER", "Modifiers", "amr:manner - qualitative mode"),
        ("AMR_POLARITY_NEGATION", "Modifiers", "amr:polarity - logical negation"),
        ("AMR_EXPRESSIVE_MODE", "Modality", "amr:mode - expressive illocution"),
        ("AMR_ALIENABLE_POSS", "Relations", "amr:poss - alienable/inalienable possession"),
        ("AMR_MEREOLOGICAL_PART", "Relations", "amr:part - mereological inclusion"),
        ("AMR_TEMPORAL_SUBEVENT", "Relations", "amr:subevent - temporal part of event"),
        ("AMR_ACCOMPANIER_PARTICIPANT", "Modifiers", "amr:accompanier - associate participant"),
        ("AMR_DISCOURSE_TOPIC", "Relations", "amr:topic - discourse theme"),
        ("AMR_CHANNEL_MEDIUM", "Relations", "amr:medium - communication channel"),
        ("AMR_ORIGIN_SOURCE", "Relations", "amr:source - informational or spatial origin"),
        ("AMR_REPETITION_RATE", "Modifiers", "amr:frequency - repetition frequency"),
        ("AMR_SCALAR_INTENSITY", "Modifiers", "amr:degree - scalar intensity"),
        ("AMR_CONCEPTUAL_DOMAIN", "Relations", "amr:domain - conceptual domain"),
        ("AMR_GENERIC_MODIFIER", "Modifiers", "amr:mod - generic modifier"),
        ("AMR_ORDINAL_RANK", "Modifiers", "amr:ord - ordinal rank"),
        ("AMR_NUMERIC_CARDINALITY", "Quantifiers", "amr:quant - numeric cardinality"),
    ]
    for name, cat, desc in amr_relations:
        add(name, "AMR", cat, desc)

    # 4. WordNet Extended Taxonomic Roots (40 Candidates)
    wn_classes = [
        ("WN_LIVING_ORGANISM", "Biology", "wn:living_thing - animate organism"),
        ("WN_MAMMAL_SPECIES", "Zoology", "wn:mammal - warm-blooded milk-producing animal"),
        ("WN_AVIAN_BIRD", "Zoology", "wn:bird - avian feathered species"),
        ("WN_BOTANICAL_PLANT", "Biology", "wn:plant - kingdom plantae botanical organism"),
        ("WN_EDIBLE_NUTRIENT", "Substance", "wn:food - edible solid or liquid matter"),
        ("WN_PHYSICAL_SUBSTANCE", "Physics", "wn:substance - physical material constituent"),
        ("WN_NATURAL_PHENOMENON", "Nature", "wn:phenomenon - natural physical event or occurrence"),
        ("WN_AFFECTIVE_FEELING", "Mind", "wn:feeling - affective emotional internal state"),
        ("WN_VOLITIONAL_ACT", "Events", "wn:act - intentional volitional human deed"),
        ("WN_SOCIAL_COLLECTIVE", "Social", "wn:group - collection of interacting entities"),
        ("WN_ECONOMIC_POSSESSION", "Economics", "wn:possession - owned asset or holding"),
        ("WN_INTRINSIC_ATTRIBUTE", "Qualities", "wn:attribute - intrinsic or extrinsic property"),
        ("WN_MEASURABLE_QUANTITY", "Metrics", "wn:quantity - measurable extent or count"),
        ("WN_TEMPORAL_PERIOD", "Temporal", "wn:time_period - temporal duration or epoch"),
        ("WN_GEOMETRIC_SHAPE", "Spatial", "wn:shape - spatial configuration or geometry"),
        ("WN_ANATOMICAL_PART", "Anatomy", "wn:body_part - anatomical organ or constituent"),
        ("WN_STATE_TRANSITION", "Events", "wn:change - transition from one state to another"),
        ("WN_SPATIAL_MOTION", "Kinematics", "wn:motion - physical movement through space"),
        ("WN_PHYSICAL_CONTACT", "Kinematics", "wn:contact - physical impingement or adjacency"),
        ("WN_SENSORY_PERCEPTION", "Perception", "wn:perception - sensory intake and signal processing"),
        ("WN_NUTRIENT_CONSUMPTION", "Biology", "wn:consumption - intake of nutrients or energy"),
        ("WN_SYNTHESIS_CREATION", "Production", "wn:creation - fabrication or synthesis of new entity"),
        ("WN_ADVERSARIAL_CONTEST", "Social", "wn:competition - adversarial struggle or contest"),
        ("WN_METEOROLOGY_WEATHER", "Nature", "wn:weather - atmospheric condition or meteorological state"),
        ("WN_IMPLEMENT_TOOL", "Artifacts", "wn:tool - implement or device used to perform task"),
        ("WN_PRECISION_INSTRUMENT", "Artifacts", "wn:instrument - precision device or musical artifact"),
        ("WN_SYMBOLIC_GLYPH", "Language", "wn:symbol - sign or glyph representing concept"),
        ("WN_STORAGE_CONTAINER", "Artifacts", "wn:container - receptacle designed to hold matter"),
        ("WN_WEARABLE_GARMENT", "Artifacts", "wn:clothing - wearable garment or textile covering"),
        ("WN_ARCHITECTURAL_BUILDING", "Artifacts", "wn:building - architectural structure"),
    ]
    for name, cat, desc in wn_classes:
        add(name, "WordNet", cat, desc)

    # 5. Logic, Calculi & Formal Primitives (60 Candidates)
    logic_and_ast = [
        ("LOGIC_CONDITIONAL_IMPLICATION", "Formal Logic", "Conditional implication P -> Q"),
        ("LOGIC_CONJUNCTIVE_AND", "Formal Logic", "Conjunctive AND operator"),
        ("LOGIC_DISJUNCTIVE_OR", "Formal Logic", "Disjunctive OR operator"),
        ("LOGIC_NEGATION_NOT", "Formal Logic", "Negation NOT operator"),
        ("LOGIC_UNIVERSAL_QUANT", "Formal Logic", "Universal quantifier ∀"),
        ("LOGIC_EXISTENTIAL_QUANT", "Formal Logic", "Existential quantifier ∃"),
        ("LOGIC_AXIOMATIC_PREMISE", "Formal Logic", "Axiomatic baseline premise"),
        ("LOGIC_BICONDITIONAL_EQ", "Formal Logic", "Biconditional equivalence P <-> Q"),
        ("LOGIC_TAUTOLOGY_TRUE", "Formal Logic", "Universally valid tautological sentence"),
        ("LOGIC_CONTRADICTION_FALSUM", "Formal Logic", "Explicit inconsistency or falsum"),
        ("ALLEN_STRICT_BEFORE", "Allen Temporal", "Interval relation: A strictly before B"),
        ("ALLEN_STRICT_AFTER", "Allen Temporal", "Interval relation: A strictly after B"),
        ("ALLEN_MEETS_BOUNDARY", "Allen Temporal", "Interval relation: A meets B at boundary"),
        ("ALLEN_OVERLAPS_INTERVAL", "Allen Temporal", "Interval relation: A overlaps B"),
        ("ALLEN_DURING_INTERVAL", "Allen Temporal", "Interval relation: A strictly during B"),
        ("ALLEN_CONTAINS_INTERVAL", "Allen Temporal", "Interval relation: A contains B"),
        ("ALLEN_STARTS_BOUNDARY", "Allen Temporal", "Interval relation: A starts B"),
        ("ALLEN_FINISHES_BOUNDARY", "Allen Temporal", "Interval relation: A finishes B"),
        ("ALLEN_EQUALS_CONGRUENT", "Allen Temporal", "Interval relation: A temporally equals B"),
        ("RCC8_DISCONNECTED_DC", "RCC-8 Spatial", "Disconnected: exterior spatial disjointness"),
        ("RCC8_EXT_CONNECTED_EC", "RCC-8 Spatial", "Externally Connected: touching at boundary"),
        ("RCC8_PARTIAL_OVERLAP_PO", "RCC-8 Spatial", "Partial Overlap: sharing interior points"),
        ("RCC8_CONGRUENT_EQ", "RCC-8 Spatial", "Equal: identical spatial regions"),
        ("RCC8_TANGENTIAL_PART_TPP", "RCC-8 Spatial", "Tangential Proper Part: interior subset touching boundary"),
        ("RCC8_NON_TANG_PART_NTPP", "RCC-8 Spatial", "Non-Tangential Proper Part: strictly interior subset"),
        ("PEARL_ASSOC_L1", "Pearl Causal", "Level 1 Association: observational correlation P(y|x)"),
        ("PEARL_INTERV_L2", "Pearl Causal", "Level 2 Intervention: do-calculus action P(y|do(x))"),
        ("PEARL_COUNTER_L3", "Pearl Causal", "Level 3 Counterfactual: retrospective reasoning P(y_x|x',y')"),
        ("MODAL_NECESSITY_BOX", "Modal Logic", "Box necessity operator □P"),
        ("MODAL_POSSIBILITY_DIAMOND", "Modal Logic", "Diamond possibility operator ◇P"),
        ("MODAL_DEONTIC_OBLIGATION", "Deontic Logic", "Deontic obligation O(P)"),
        ("MODAL_DEONTIC_PERMISSION", "Deontic Logic", "Deontic permission P(P)"),
        ("LTL_NEXT_OPERATOR", "Temporal Logic", "LTL Next-time temporal operator X(P)"),
        ("LTL_ALWAYS_OPERATOR", "Temporal Logic", "LTL Globally/Always temporal operator G(P)"),
        ("LTL_EVENTUALLY_OPERATOR", "Temporal Logic", "LTL Finally/Eventually temporal operator F(P)"),
        ("LTL_UNTIL_OPERATOR", "Temporal Logic", "LTL Until temporal operator P U Q"),
        ("AST_FUNCTION_HEADER", "Code", "Function definition header"),
        ("AST_LOOP_EXECUTION", "Code", "Loop execution block (for/while)"),
        ("AST_CONDITIONAL_BRANCH", "Code", "Conditional branching statement (if/else)"),
        ("AST_VARIABLE_ASSIGN", "Code", "Variable assignment binding (x = expr)"),
        ("AST_RETURN_STATEMENT", "Code", "Return statement expression"),
        ("AST_INVOCATION_CALL", "Code", "Function invocation call"),
        ("AST_CLASS_STRUCTURE", "Code", "Class structure declaration"),
        ("AST_TRY_EXCEPT_BLOCK", "Code", "Exception handler try-catch block"),
        ("AST_RECURSIVE_INVOCATION", "Code", "Self-referential recursive invocation"),
        ("AST_ASYNC_SUSPEND", "Code", "Asynchronous coroutine suspend point"),
        ("AST_LAMBDA_CLOSURE", "Code", "Anonymous lambda expression closure"),
    ]
    for name, cat, desc in logic_and_ast:
        add(name, "Logic/Calculi", cat, desc)

    # Pad with structured auxiliary formal dimensions if needed to reach target over-complete pool
    target_pool_size = max(len(CANONICAL_SLOTS), 2048)
    while len(candidates) < target_pool_size:
        idx = len(candidates)
        add(
            name=f"EXT_FORMAL_DIM_{idx}",
            source="ExtendedFormal",
            category="Formal Reasoning",
            description=f"Extended formal semantic dimension index {idx}",
        )

    return candidates


def project_canonical_to_candidates(canonical_matrix: np.ndarray, candidates: List[CandidateDimension]) -> np.ndarray:
    """Projects an (N, D) canonical quaternary matrix onto an (N, K) candidate matrix."""
    num_samples = canonical_matrix.shape[0]
    num_canon = canonical_matrix.shape[1]
    num_candidates = len(candidates)
    candidate_matrix = np.zeros((num_samples, num_candidates), dtype=np.uint8)

    # 1. Directly copy available canonical slots
    k_copy = min(num_canon, num_candidates)
    candidate_matrix[:, :k_copy] = canonical_matrix[:, :k_copy]

    def get_c(name: str) -> np.ndarray:
        idx = SLOT_NAME_TO_INDEX.get(name)
        if idx is not None:
            return canonical_matrix[:, idx]
        return np.zeros(num_samples, dtype=np.uint8)

    cand_name_to_idx = {c.name: c.id for c in candidates}

    # Projections for extended clean candidates
    for cand_name, slot_map in [
        ("FN_EXPERIENCER_PASSIVE", "VAL_EXPERIENCER"),
        ("FN_STIMULUS_EVOKING", "ROLE_AFFECTIVE_TARGET"),
        ("FN_MEANS_METHOD", "VAL_X5_INSTRUMENT"),
        ("FN_REASON_EXPLANATION", "WN_MOTIVE_REASON"),
        ("FN_DIRECTION_HEADING", "NSM_SIDE"),
        ("FN_COMMERCE_BUYING", "WN_POSSESSION_ASSET"),
        ("FN_COMMERCE_SELLING", "WN_POSSESSION_ASSET"),
        ("FN_TRANSFER_POSSESSION", "WN_POSSESSION_ASSET"),
        ("FN_DESCRIPTIVE_PROP", "TYPE_ATTRIBUTE_PROPERTY"),
        ("FN_CIRCUMSTANTIAL_COND", "GRAPH_SCOPED_CONTEXT"),
        ("FN_COMM_TOPIC", "TYPE_ABSTRACT_CONCEPT"),
        ("FN_COMM_MEDIUM", "TYPE_COMMUNICATION_MSG"),
        ("FN_COMM_MESSAGE", "TYPE_COMMUNICATION_MSG"),
        ("FN_TRANSFERRED_ITEM", "TYPE_ARTIFACT"),
        ("FN_OWNER_POSSESSOR", "ROLE_AGENT_CAPABLE"),
        ("FN_DURATION_INTERVAL", "TYPE_TEMPORAL_INTERVAL"),
        ("FN_MEREOLOGY_PART_WHOLE", "MEREOLOGY_MERONYM_PART"),
        ("FN_CONTAINER_VESSEL", "TYPE_ARTIFACT"),
        ("FN_ENCLOSED_CONTENTS", "TYPE_INANIMATE_PHYSICAL"),
        ("FN_COGNITION_ACTIVITY", "NSM_THINK"),
        ("FN_PERCEPTION_ACTIVITY", "NSM_SEE"),
        ("FN_STATEMENT_ACT", "NSM_SAY"),
        ("FN_MOTION_TRANSLATION", "NSM_MOVE"),
        ("AMR_PROTO_AGENT", "VAL_X1_AGENT"),
        ("AMR_PROTO_PATIENT", "VAL_X2_PATIENT"),
        ("AMR_PROTO_RECIPIENT", "VAL_X3_DESTINATION"),
        ("AMR_PROTO_INSTRUMENT", "VAL_X5_INSTRUMENT"),
        ("AMR_PROTO_RESULT", "GRAPH_RETURN_VALUE"),
        ("AMR_LOCATIVE_COORD", "VAL_LOCATION_SLOT"),
        ("AMR_DESTINATION_TARGET", "VAL_X3_DESTINATION"),
        ("AMR_TRAJECTORY_PATH", "NSM_MOVE"),
        ("AMR_TEMPORAL_ANCHOR", "VAL_TIME_SLOT"),
        ("AMR_ANTECEDENT_COND", "LJB_GANAI_IF_THEN"),
        ("AMR_CAUSAL_DETERMINANT", "CAUSAL_DIRECT_MECHANISM"),
        ("AMR_TELEOLOGICAL_GOAL", "VAL_PURPOSE_SLOT"),
        ("AMR_POLARITY_NEGATION", "LJB_NA_NEGATION"),
        ("AMR_MEREOLOGICAL_PART", "NSM_PART"),
        ("AMR_TEMPORAL_SUBEVENT", "GRAPH_IS_SUB_EXP"),
        ("LOGIC_CONDITIONAL_IMPLICATION", "LJB_GANAI_IF_THEN"),
        ("LOGIC_CONJUNCTIVE_AND", "LJB_JE_AND"),
        ("LOGIC_DISJUNCTIVE_OR", "LJB_JA_OR"),
        ("LOGIC_NEGATION_NOT", "LJB_NA_NEGATION"),
        ("LOGIC_UNIVERSAL_QUANT", "LJB_RO_ALL_QUANT"),
        ("LOGIC_EXISTENTIAL_QUANT", "LJB_SUO_AT_LEAST_ONE"),
        ("LOGIC_AXIOMATIC_PREMISE", "EPIST_AXIOMATIC_PREMISE"),
        ("LOGIC_BICONDITIONAL_EQ", "LJB_DU_IDENTITY"),
        ("LOGIC_TAUTOLOGY_TRUE", "SOLVER_PROOF_VALIDATED"),
        ("LOGIC_CONTRADICTION_FALSUM", "SOLVER_CONTRADICTION_FLAG"),
        ("ALLEN_STRICT_BEFORE", "TEMP_ALLEN_BEFORE"),
        ("ALLEN_STRICT_AFTER", "TEMP_ALLEN_AFTER"),
        ("ALLEN_MEETS_BOUNDARY", "TEMP_ALLEN_MEETS"),
        ("ALLEN_OVERLAPS_INTERVAL", "TEMP_ALLEN_OVERLAPS"),
        ("ALLEN_DURING_INTERVAL", "TEMP_ALLEN_DURING"),
        ("RCC8_DISCONNECTED_DC", "SPATIAL_RCC_DISCONNECTED"),
        ("RCC8_EXT_CONNECTED_EC", "SPATIAL_RCC_EXT_CONNECTED"),
        ("RCC8_PARTIAL_OVERLAP_PO", "SPATIAL_RCC_PARTIAL_OVERLAP"),
        ("RCC8_NON_TANG_PART_NTPP", "SPATIAL_RCC_NON_TANG_PART"),
        ("PEARL_ASSOC_L1", "CAUSAL_DIRECT_MECHANISM"),
        ("MODAL_NECESSITY_BOX", "LOGIC_NECESSITY_BOX"),
        ("MODAL_POSSIBILITY_DIAMOND", "LOGIC_POSSIBILITY_DIAMOND"),
        ("MODAL_DEONTIC_OBLIGATION", "EPIST_DEONTIC_OBLIGATION"),
        ("MODAL_DEONTIC_PERMISSION", "EPIST_DEONTIC_PERMISSION"),
        ("LTL_ALWAYS_OPERATOR", "LOGIC_TEMPORAL_ALWAYS_G"),
        ("LTL_EVENTUALLY_OPERATOR", "LOGIC_TEMPORAL_EVENTUALLY_F"),
        ("AST_FUNCTION_HEADER", "GRAPH_FUNCTION_DEF"),
        ("AST_LOOP_EXECUTION", "GRAPH_CONTROL_LOOP"),
        ("AST_CONDITIONAL_BRANCH", "GRAPH_BRANCH_COND"),
        ("AST_VARIABLE_ASSIGN", "GRAPH_VARIABLE_BIND"),
        ("AST_RETURN_STATEMENT", "GRAPH_RETURN_VALUE"),
        ("AST_INVOCATION_CALL", "GRAPH_INVOCATION_CALL"),
        ("AST_CLASS_STRUCTURE", "GRAPH_SCOPED_CONTEXT"),
        ("AST_TRY_EXCEPT_BLOCK", "GRAPH_EXCEPTION_HANDLE"),
        ("AST_RECURSIVE_INVOCATION", "GRAPH_RECURSIVE_REF"),
    ]:
        if cand_name in cand_name_to_idx:
            candidate_matrix[:, cand_name_to_idx[cand_name]] = get_c(slot_map)

    return candidate_matrix


def export_candidate_pool(
    candidates: Optional[List[CandidateDimension]] = None,
    output_path: Union[str, Path] = "output/candidate_pool.json",
) -> Path:
    """Exports the clean candidate pool to a structured JSON file."""
    if candidates is None:
        candidates = build_candidate_pool()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data = [c.to_dict() for c in candidates]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out_file
