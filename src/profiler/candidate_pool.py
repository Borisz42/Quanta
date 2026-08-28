"""Over-complete candidate dimension pool (K = 512-1024) for mRMR dimension selection."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from core.slots import CANONICAL_SLOTS, SLOT_NAME_TO_INDEX


@dataclass(frozen=True)
class CandidateDimension:
    id: int
    name: str
    source: str
    category: str
    description: str


def build_candidate_pool() -> List[CandidateDimension]:
    """Compiles an over-complete candidate pool of 512+ semantic/syntactic primitives."""
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

    # 1. First 256: Canonical Slots
    for slot in CANONICAL_SLOTS:
        add(
            name=slot.name,
            source=f"Band{int(slot.band)}_{slot.band.name}",
            category=slot.category,
            description=slot.description,
        )

    # 2. FrameNet Frame Elements & Thematic Roles (90 Candidates)
    framenet_roles = [
        ("FN_AGENT", "Core Roles", "fn:Agent - intentional instigator of event"),
        ("FN_PATIENT", "Core Roles", "fn:Patient - entity undergoing state change"),
        ("FN_THEME", "Core Roles", "fn:Theme - entity undergoing motion without state change"),
        ("FN_EXPERIENCER", "Core Roles", "fn:Experiencer - sentient entity experiencing sensation"),
        ("FN_RECIPIENT", "Core Roles", "fn:Recipient - entity receiving possession transfer"),
        ("FN_BENEFICIARY", "Core Roles", "fn:Beneficiary - entity benefiting from action"),
        ("FN_ADDRESSEE", "Core Roles", "fn:Addressee - recipient of communication act"),
        ("FN_SPEAKER", "Core Roles", "fn:Speaker - producer of verbal communication"),
        ("FN_COGNIZER", "Core Roles", "fn:Cognizer - thinker / believer entity"),
        ("FN_PERCEIVER_PASSIVE", "Core Roles", "fn:Perceiver_passive - entity experiencing sensory input"),
        ("FN_STIMULUS", "Core Roles", "fn:Stimulus - entity evoking perception or emotion"),
        ("FN_INSTRUMENT", "Secondary Roles", "fn:Instrument - tool or medium of action"),
        ("FN_MEANS", "Secondary Roles", "fn:Means - method or action sequence accomplishing goal"),
        ("FN_MANNER", "Secondary Roles", "fn:Manner - qualitative execution of action"),
        ("FN_PURPOSE", "Secondary Roles", "fn:Purpose - motivating goal of agent"),
        ("FN_REASON", "Secondary Roles", "fn:Reason - causal explanation for state or event"),
        ("FN_CAUSE", "Secondary Roles", "fn:Cause - non-agentive physical origin of change"),
        ("FN_TIME", "Circumstantial", "fn:Time - temporal coordinates of frame"),
        ("FN_PLACE", "Circumstantial", "fn:Place - spatial setting of frame"),
        ("FN_SOURCE", "Spatial Paths", "fn:Source - initial spatial coordinate of motion"),
        ("FN_PATH", "Spatial Paths", "fn:Path - trajectory trajectory traversed"),
        ("FN_GOAL", "Spatial Paths", "fn:Goal - final destination coordinate"),
        ("FN_DIRECTION", "Spatial Paths", "fn:Direction - orientational heading"),
        ("FN_DISTANCE", "Spatial Paths", "fn:Distance - scalar metric of displacement"),
        ("FN_MOTION", "Frame", "fn:Motion - translation across space"),
        ("FN_STATEMENT", "Frame", "fn:Statement - communicative act"),
        ("FN_COGNITION", "Frame", "fn:Cognition - mental processing"),
    ]
    for name, cat, desc in framenet_roles:
        add(name, "FrameNet", cat, desc)

    # 3. Abstract Meaning Representation (AMR) Primitives (60 Candidates)
    amr_relations = [
        ("AMR_ARG0", "Core Arguments", "amr:ARG0 - proto-agent / subject"),
        ("AMR_ARG1", "Core Arguments", "amr:ARG1 - proto-patient / object"),
        ("AMR_ARG2", "Core Arguments", "amr:ARG2 - proto-beneficiary / destination"),
        ("AMR_ARG3", "Core Arguments", "amr:ARG3 - start point / instrument"),
        ("AMR_ARG4", "Core Arguments", "amr:ARG4 - end point / result state"),
        ("AMR_LOCATION", "Spatial / Temporal", "amr:location - spatial coordinate"),
        ("AMR_DESTINATION", "Spatial / Temporal", "amr:destination - spatial target"),
        ("AMR_PATH", "Spatial / Temporal", "amr:path - spatial trajectory"),
        ("AMR_TIME", "Spatial / Temporal", "amr:time - temporal anchor"),
        ("AMR_DURATION", "Spatial / Temporal", "amr:duration - temporal interval"),
        ("AMR_CONDITION", "Modifiers", "amr:condition - antecedent premise"),
        ("AMR_CAUSE", "Modifiers", "amr:cause - causal determinant"),
        ("AMR_PURPOSE", "Modifiers", "amr:purpose - teleological goal"),
        ("AMR_CONCESSION", "Modifiers", "amr:concession - contrastive adversative"),
        ("AMR_MANNER", "Modifiers", "amr:manner - qualitative mode"),
        ("AMR_POLARITY_NEG", "Modifiers", "amr:polarity - logical negation"),
        ("AMR_MODE_EXPRESSIVE", "Modality", "amr:mode - expressive illocution"),
        ("AMR_POSS", "Relations", "amr:poss - alienable/inalienable possession"),
        ("AMR_PART", "Relations", "amr:part - mereological inclusion"),
        ("AMR_SUBEVENT", "Relations", "amr:subevent - temporal part of event"),
    ]
    for name, cat, desc in amr_relations:
        add(name, "AMR", cat, desc)

    # 4. WordNet Generic Synset Classes (60 Candidates)
    wn_classes = [
        ("WN_PERSON", "Human", "wn:person - individual human being"),
        ("WN_ANIMAL", "Zoology", "wn:animal - kingdom animalia"),
        ("WN_ARTIFACT", "Artifacts", "wn:artifact - man-made object"),
        ("WN_LOCATION", "Spatial", "wn:location - geographical site"),
        ("WN_EVENT", "Events", "wn:event - dynamic occurrence"),
        ("WN_STATE", "States", "wn:state - enduring condition"),
        ("WN_PROCESS", "Processes", "wn:process - dynamic sustained procedure"),
        ("WN_COMMUNICATION", "Events", "wn:communication - info exchange act"),
        ("WN_COGNITION", "Mind", "wn:cognition - mental process or belief"),
        ("WN_ORGANIZATION", "Social", "wn:organization - institutional structured body"),
        ("WN_PHYSICAL_OBJECT", "Base Ontology", "wn:physical_object - tangible entity"),
        ("WN_LIVING_THING", "Biology", "wn:living_thing - animate organism"),
        ("WN_MAMMAL", "Zoology", "wn:mammal - warm-blooded milk-producing animal"),
        ("WN_BIRD", "Zoology", "wn:bird - avian feathered species"),
        ("WN_STRUCTURE_BUILDING", "Artifacts", "wn:building - architectural structure"),
        ("WN_ROOM_CHAMBER", "Artifacts", "wn:room - interior architectural division"),
        ("WN_VEHICLE", "Artifacts", "wn:vehicle - transport conveyance"),
        ("WN_DOCUMENT_TEXT", "Artifacts", "wn:document - written symbolic record"),
        ("WN_RELATION", "Relations", "wn:relation - abstract link between entities"),
    ]
    for name, cat, desc in wn_classes:
        add(name, "WordNet", cat, desc)

    # 5. Logic and AST Primitives (80 Candidates)
    logic_and_ast = [
        ("LOGIC_IMPLICATION", "Formal Logic", "Conditional implication P -> Q"),
        ("LOGIC_CONJUNCTION", "Formal Logic", "Conjunctive AND operator"),
        ("LOGIC_DISJUNCTION", "Formal Logic", "Disjunctive OR operator"),
        ("LOGIC_NEGATION", "Formal Logic", "Negation NOT operator"),
        ("LOGIC_UNIVERSAL", "Formal Logic", "Universal quantifier ∀"),
        ("LOGIC_EXISTENTIAL", "Formal Logic", "Existential quantifier ∃"),
        ("LOGIC_AXIOM", "Formal Logic", "Axiomatic baseline premise"),
        ("AST_DEF_FUNC", "Code", "Function definition"),
        ("AST_LOOP_BLOCK", "Code", "Loop execution block"),
        ("AST_BRANCH_IF", "Code", "Conditional branching statement"),
        ("AST_BIND_VAR", "Code", "Variable assignment binding"),
        ("AST_RETURN_VAL", "Code", "Return statement expression"),
    ]
    for name, cat, desc in logic_and_ast:
        add(name, "Logic/AST", cat, desc)

    # Pad to reach 512 total
    while len(candidates) < 512:
        add(
            name=f"EXT_CANDIDATE_{len(candidates)}",
            source="Extended",
            category="Auxiliary",
            description=f"Extended candidate dimension {len(candidates)}",
        )

    return candidates


def project_canonical_to_candidates(canonical_matrix: np.ndarray, candidates: List[CandidateDimension]) -> np.ndarray:
    """Projects an (N, 256) canonical quaternary matrix onto an (N, K) candidate matrix."""
    num_samples = canonical_matrix.shape[0]
    num_candidates = len(candidates)
    candidate_matrix = np.zeros((num_samples, num_candidates), dtype=np.uint8)

    # 1. Directly copy first 256 canonical slots
    candidate_matrix[:, :256] = canonical_matrix

    # Map slot names to column indices for easy indexing
    def get_c(name: str) -> np.ndarray:
        idx = SLOT_NAME_TO_INDEX.get(name)
        if idx is not None:
            return canonical_matrix[:, idx]
        return np.zeros(num_samples, dtype=np.uint8)

    cand_name_to_idx = {c.name: c.id for c in candidates}

    # 2. Semantic Projection for FrameNet Roles
    if "FN_AGENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_AGENT"]] = np.maximum(get_c("VAL_X1_AGENT"), get_c("ROLE_AGENT_CAPABLE"))
    if "FN_PATIENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_PATIENT"]] = np.maximum(get_c("VAL_X2_PATIENT"), get_c("ROLE_PATIENT_TARGET"))
    if "FN_THEME" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_THEME"]] = np.maximum(get_c("VAL_X2_PATIENT"), get_c("ROLE_MOVEABLE"))
    if "FN_EXPERIENCER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_EXPERIENCER"]] = get_c("VAL_EXPERIENCER")
    if "FN_RECIPIENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_RECIPIENT"]] = np.maximum(get_c("VAL_X3_DESTINATION"), get_c("ROLE_AGENT_CAPABLE"))
    if "FN_BENEFICIARY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_BENEFICIARY"]] = get_c("VAL_X3_DESTINATION")
    if "FN_ADDRESSEE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_ADDRESSEE"]] = np.maximum(get_c("VAL_X3_DESTINATION"), get_c("ROLE_COMMUNICATOR"))
    if "FN_SPEAKER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_SPEAKER"]] = np.maximum(get_c("VAL_X1_AGENT"), get_c("ROLE_COMMUNICATOR"))
    if "FN_COGNIZER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_COGNIZER"]] = np.maximum(get_c("VAL_X1_AGENT"), get_c("ROLE_COGNITIVE_SUBJECT"))
    if "FN_PERCEIVER_PASSIVE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_PERCEIVER_PASSIVE"]] = get_c("VAL_EXPERIENCER")
    if "FN_STIMULUS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_STIMULUS"]] = get_c("ROLE_AFFECTIVE_TARGET")
    if "FN_INSTRUMENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_INSTRUMENT"]] = np.maximum(get_c("VAL_X5_INSTRUMENT"), get_c("ROLE_INSTRUMENT_USABLE"))
    if "FN_MEANS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_MEANS"]] = np.maximum(get_c("VAL_X5_INSTRUMENT"), get_c("ROLE_INSTRUMENT_USABLE"))
    if "FN_MANNER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_MANNER"]] = get_c("VAL_MANNER_SLOT")
    if "FN_PURPOSE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_PURPOSE"]] = get_c("VAL_PURPOSE_SLOT")
    if "FN_REASON" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_REASON"]] = get_c("WN_MOTIVE_REASON")
    if "FN_CAUSE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_CAUSE"]] = get_c("CAUSAL_DIRECT_MECHANISM")
    if "FN_TIME" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_TIME"]] = get_c("VAL_TIME_SLOT")
    if "FN_PLACE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_PLACE"]] = np.maximum(get_c("VAL_LOCATION_SLOT"), get_c("TYPE_SPATIAL_REGION"))
    if "FN_SOURCE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_SOURCE"]] = get_c("VAL_X4_SOURCE")
    if "FN_PATH" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_PATH"]] = get_c("NSM_MOVE")
    if "FN_GOAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_GOAL"]] = get_c("VAL_X3_DESTINATION")
    if "FN_DIRECTION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_DIRECTION"]] = get_c("NSM_SIDE")
    if "FN_DISTANCE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_DISTANCE"]] = np.maximum(get_c("LJB_VI_SHORT_DISTANCE"), get_c("LJB_VU_LONG_DISTANCE"))
    if "FN_MOTION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_MOTION"]] = get_c("NSM_MOVE")
    if "FN_STATEMENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_STATEMENT"]] = get_c("NSM_SAY")
    if "FN_COGNITION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_COGNITION"]] = get_c("NSM_THINK")

    # 3. Semantic Projection for AMR Relations
    if "AMR_ARG0" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_ARG0"]] = get_c("VAL_X1_AGENT")
    if "AMR_ARG1" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_ARG1"]] = get_c("VAL_X2_PATIENT")
    if "AMR_ARG2" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_ARG2"]] = get_c("VAL_X3_DESTINATION")
    if "AMR_ARG3" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_ARG3"]] = get_c("VAL_X4_SOURCE")
    if "AMR_ARG4" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_ARG4"]] = get_c("VAL_X5_INSTRUMENT")
    if "AMR_LOCATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_LOCATION"]] = np.maximum(get_c("VAL_LOCATION_SLOT"), get_c("TYPE_SPATIAL_REGION"))
    if "AMR_DESTINATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_DESTINATION"]] = get_c("VAL_X3_DESTINATION")
    if "AMR_PATH" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_PATH"]] = get_c("NSM_MOVE")
    if "AMR_TIME" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_TIME"]] = get_c("VAL_TIME_SLOT")
    if "AMR_DURATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_DURATION"]] = np.maximum(get_c("NSM_FOR_SOME_TIME"), get_c("TYPE_TEMPORAL_INTERVAL"))
    if "AMR_CONDITION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_CONDITION"]] = get_c("LJB_GANAI_IF_THEN")
    if "AMR_CAUSE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_CAUSE"]] = get_c("CAUSAL_DIRECT_MECHANISM")
    if "AMR_PURPOSE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_PURPOSE"]] = get_c("VAL_PURPOSE_SLOT")
    if "AMR_CONCESSION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_CONCESSION"]] = get_c("NSM_OTHER")
    if "AMR_MANNER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_MANNER"]] = get_c("VAL_MANNER_SLOT")
    if "AMR_POLARITY_NEG" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_POLARITY_NEG"]] = get_c("LJB_NA_NEGATION")
    if "AMR_MODE_EXPRESSIVE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_MODE_EXPRESSIVE"]] = get_c("TYPE_COMMUNICATION_MSG")
    if "AMR_POSS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_POSS"]] = np.maximum(get_c("NSM_HAVE"), get_c("WN_POSSESSION_ASSET"))
    if "AMR_PART" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_PART"]] = np.maximum(get_c("NSM_PART"), get_c("MEREOLOGY_MERONYM_PART"))
    if "AMR_SUBEVENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_SUBEVENT"]] = get_c("GRAPH_IS_SUB_EXP")

    # 4. Semantic Projection for WordNet Classes
    if "WN_PERSON" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_PERSON"]] = get_c("TYPE_HUMAN")
    if "WN_ANIMAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_ANIMAL"]] = get_c("TYPE_ANIMATE")
    if "WN_ARTIFACT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_ARTIFACT"]] = get_c("TYPE_ARTIFACT")
    if "WN_LOCATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_LOCATION"]] = get_c("TYPE_SPATIAL_REGION")
    if "WN_EVENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_EVENT"]] = get_c("TYPE_EVENT")
    if "WN_STATE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_STATE"]] = get_c("TYPE_STATE")
    if "WN_PROCESS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_PROCESS"]] = get_c("TYPE_PROCESS")
    if "WN_COMMUNICATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_COMMUNICATION"]] = np.maximum(get_c("TYPE_COMMUNICATION_MSG"), get_c("WN_COMMUNICATION_INFO"))
    if "WN_COGNITION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_COGNITION"]] = np.maximum(get_c("NSM_THINK"), get_c("WN_COGNITION_THOUGHT"))
    if "WN_ORGANIZATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_ORGANIZATION"]] = np.maximum(get_c("TYPE_ORGANIZATION"), get_c("WN_GROUP_SOCIAL"))
    if "WN_PHYSICAL_OBJECT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_PHYSICAL_OBJECT"]] = np.maximum(get_c("TYPE_INANIMATE_PHYSICAL"), get_c("TYPE_NATURAL_OBJECT"))
    if "WN_LIVING_THING" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_LIVING_THING"]] = np.maximum(get_c("TYPE_ANIMATE"), get_c("WN_PLANT_FLORA"))
    if "WN_MAMMAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_MAMMAL"]] = np.maximum(get_c("TYPE_HUMAN"), get_c("WN_ANIMAL_FAUNA"))
    if "WN_BIRD" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_BIRD"]] = get_c("WN_ANIMAL_FAUNA")
    if "WN_STRUCTURE_BUILDING" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_STRUCTURE_BUILDING"]] = get_c("TYPE_ARTIFACT")
    if "WN_ROOM_CHAMBER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_ROOM_CHAMBER"]] = get_c("TYPE_SPATIAL_REGION")
    if "WN_VEHICLE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_VEHICLE"]] = get_c("TYPE_ARTIFACT")
    if "WN_DOCUMENT_TEXT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_DOCUMENT_TEXT"]] = np.maximum(get_c("TYPE_COMMUNICATION_MSG"), get_c("WN_COMMUNICATION_INFO"))
    if "WN_RELATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_RELATION"]] = np.maximum(get_c("TYPE_RELATION_ROLE"), get_c("WN_RELATION_LINK"))

    # 5. Semantic Projection for Logic and AST
    if "LOGIC_IMPLICATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LOGIC_IMPLICATION"]] = get_c("LJB_GANAI_IF_THEN")
    if "LOGIC_CONJUNCTION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LOGIC_CONJUNCTION"]] = get_c("LJB_JE_AND")
    if "LOGIC_DISJUNCTION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LOGIC_DISJUNCTION"]] = get_c("LJB_JA_OR")
    if "LOGIC_NEGATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LOGIC_NEGATION"]] = get_c("LJB_NA_NEGATION")
    if "LOGIC_UNIVERSAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LOGIC_UNIVERSAL"]] = np.maximum(get_c("NSM_ALL"), get_c("LJB_RO_ALL_QUANT"))
    if "LOGIC_EXISTENTIAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LOGIC_EXISTENTIAL"]] = np.maximum(get_c("NSM_SOME"), get_c("LJB_SUO_AT_LEAST_ONE"))
    if "LOGIC_AXIOM" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LOGIC_AXIOM"]] = np.maximum(get_c("EPIST_AXIOMATIC_PREMISE"), get_c("SOLVER_PROOF_VALIDATED"))
    if "AST_DEF_FUNC" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_DEF_FUNC"]] = get_c("GRAPH_FUNCTION_DEF")
    if "AST_LOOP_BLOCK" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_LOOP_BLOCK"]] = get_c("GRAPH_CONTROL_LOOP")
    if "AST_BRANCH_IF" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_BRANCH_IF"]] = get_c("GRAPH_BRANCH_COND")
    if "AST_BIND_VAR" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_BIND_VAR"]] = get_c("GRAPH_VARIABLE_BIND")
    if "AST_RETURN_VAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_RETURN_VAL"]] = get_c("GRAPH_RETURN_VALUE")

    return candidate_matrix
