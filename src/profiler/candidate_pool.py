"""Over-complete candidate dimension pool (K = 512-1024) for mRMR dimension selection."""

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

    # 1. First 256: Canonical Slots (Bands 0-3)
    for slot in CANONICAL_SLOTS:
        add(
            name=slot.name,
            source=f"Band{int(slot.band)}_{slot.band.name}",
            category=slot.category,
            description=slot.description,
        )

    # 2. FrameNet Frame Elements & Thematic Roles (50 Candidates)
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
        ("FN_PERCEPTION", "Frame", "fn:Perception - sensory gathering"),
        ("FN_COMMERCE_BUY", "Commerce", "fn:Commerce_buy - commercial exchange acquiring"),
        ("FN_COMMERCE_SELL", "Commerce", "fn:Commerce_sell - commercial exchange offering"),
        ("FN_POSSESSION_TRANSFER", "Possession", "fn:Transfer - change of ownership or holding"),
        ("FN_CO_AGENT", "Core Roles", "fn:Co_agent - secondary collaborative agent"),
        ("FN_CO_PATIENT", "Core Roles", "fn:Co_patient - secondary affected participant"),
        ("FN_DESCRIPTIVE_PROPERTY", "Modifiers", "fn:Descriptor - qualifying attribute"),
        ("FN_DEGREE", "Modifiers", "fn:Degree - intensity scale"),
        ("FN_RESULT", "Secondary Roles", "fn:Result - resulting output state"),
        ("FN_STATE", "Core Roles", "fn:State - background state of affairs"),
        ("FN_CIRCUMSTANCES", "Circumstantial", "fn:Circumstances - situational conditions"),
        ("FN_EFFECT", "Secondary Roles", "fn:Effect - consequence of causal event"),
        ("FN_TOPIC", "Communication", "fn:Topic - subject matter of communication"),
        ("FN_MEDIUM", "Communication", "fn:Medium - channel of transmission"),
        ("FN_MESSAGE", "Communication", "fn:Message - informational content"),
        ("FN_DONOR", "Possession", "fn:Donor - giving party in transfer"),
        ("FN_ITEM", "Possession", "fn:Item - transferred physical or abstract entity"),
        ("FN_OWNER", "Possession", "fn:Owner - possessing entity"),
        ("FN_DURATION", "Circumstantial", "fn:Duration - temporal length of event"),
        ("FN_FREQUENCY", "Circumstantial", "fn:Frequency - temporal repetition count"),
        ("FN_PART_WHOLE", "Mereology", "fn:Part_whole - structural composition"),
        ("FN_CONTAINER", "Spatial", "fn:Container - enclosing entity"),
        ("FN_CONTENTS", "Spatial", "fn:Contents - enclosed entities"),
    ]
    for name, cat, desc in framenet_roles:
        add(name, "FrameNet", cat, desc)

    # 3. Abstract Meaning Representation (AMR) Primitives (30 Candidates)
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
        ("AMR_ACCOMPANIER", "Modifiers", "amr:accompanier - associate participant"),
        ("AMR_TOPIC", "Relations", "amr:topic - discourse theme"),
        ("AMR_MEDIUM", "Relations", "amr:medium - communication channel"),
        ("AMR_SOURCE", "Relations", "amr:source - informational or spatial origin"),
        ("AMR_FREQUENCY", "Modifiers", "amr:frequency - repetition frequency"),
        ("AMR_DEGREE", "Modifiers", "amr:degree - scalar intensity"),
        ("AMR_DOMAIN", "Relations", "amr:domain - conceptual domain"),
        ("AMR_MOD", "Modifiers", "amr:mod - generic modifier"),
        ("AMR_ORDINAL", "Modifiers", "amr:ord - ordinal rank"),
        ("AMR_QUANT", "Quantifiers", "amr:quant - numeric cardinality"),
    ]
    for name, cat, desc in amr_relations:
        add(name, "AMR", cat, desc)

    # 4. WordNet Generic Synset Classes & Hypernym Roots (45 Candidates)
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
        ("WN_PLANT_FLORA_EXT", "Biology", "wn:plant - kingdom plantae botanical organism"),
        ("WN_FOOD_NUTRIENT", "Substance", "wn:food - edible solid or liquid matter"),
        ("WN_SUBSTANCE_MATTER", "Physics", "wn:substance - physical material constituent"),
        ("WN_PHENOMENON_NATURAL", "Nature", "wn:phenomenon - natural physical event or occurrence"),
        ("WN_PSYCH_FEELING", "Mind", "wn:feeling - affective emotional internal state"),
        ("WN_ACT_ACTION_EXT", "Events", "wn:act - intentional volitional human deed"),
        ("WN_GROUP_COLLECTIVE", "Social", "wn:group - collection of interacting entities"),
        ("WN_POSSESSION_PROPERTY", "Economics", "wn:possession - owned asset or holding"),
        ("WN_ATTRIBUTE_QUALITY", "Qualities", "wn:attribute - intrinsic or extrinsic property"),
        ("WN_QUANTITY_MEASURE", "Metrics", "wn:quantity - measurable extent or count"),
        ("WN_TIME_PERIOD", "Temporal", "wn:time_period - temporal duration or epoch"),
        ("WN_SHAPE_FORM", "Spatial", "wn:shape - spatial configuration or geometry"),
        ("WN_BODY_PART_EXT", "Anatomy", "wn:body_part - anatomical organ or constituent"),
        ("WN_CHANGE_TRANSITION", "Events", "wn:change - transition from one state to another"),
        ("WN_MOTION_MOVEMENT", "Kinematics", "wn:motion - physical movement through space"),
        ("WN_CONTACT_TOUCH", "Kinematics", "wn:contact - physical impingement or adjacency"),
        ("WN_PERCEPTION_SENSE", "Perception", "wn:perception - sensory intake and signal processing"),
        ("WN_CONSUMPTION_INGEST", "Biology", "wn:consumption - intake of nutrients or energy"),
        ("WN_CREATION_MAKE", "Production", "wn:creation - fabrication or synthesis of new entity"),
        ("WN_COMPETITION_CONTEST", "Social", "wn:competition - adversarial struggle or contest"),
        ("WN_WEATHER_METEO", "Nature", "wn:weather - atmospheric condition or meteorological state"),
        ("WN_TOOL_DEVICE", "Artifacts", "wn:tool - implement or device used to perform task"),
        ("WN_INSTRUMENT_DEVICE", "Artifacts", "wn:instrument - precision device or musical artifact"),
        ("WN_SYMBOLIC_NOTATION", "Language", "wn:symbol - sign or glyph representing concept"),
        ("WN_CONTAINER_VESSEL", "Artifacts", "wn:container - receptacle designed to hold matter"),
        ("WN_GARMENT_CLOTHING", "Artifacts", "wn:clothing - wearable garment or textile covering"),
    ]
    for name, cat, desc in wn_classes:
        add(name, "WordNet", cat, desc)

    # 5. Logic, Calculi & Formal Primitives (50 Candidates)
    logic_and_ast = [
        ("LOGIC_IMPLICATION", "Formal Logic", "Conditional implication P -> Q"),
        ("LOGIC_CONJUNCTION", "Formal Logic", "Conjunctive AND operator"),
        ("LOGIC_DISJUNCTION", "Formal Logic", "Disjunctive OR operator"),
        ("LOGIC_NEGATION", "Formal Logic", "Negation NOT operator"),
        ("LOGIC_UNIVERSAL", "Formal Logic", "Universal quantifier ∀"),
        ("LOGIC_EXISTENTIAL", "Formal Logic", "Existential quantifier ∃"),
        ("LOGIC_AXIOM", "Formal Logic", "Axiomatic baseline premise"),
        ("LOGIC_EQUIVALENCE", "Formal Logic", "Biconditional equivalence P <-> Q"),
        ("LOGIC_TAUTOLOGY", "Formal Logic", "Universally valid tautological sentence"),
        ("LOGIC_CONTRADICTION", "Formal Logic", "Explicit inconsistency or falsum"),
        ("ALLEN_BEFORE", "Allen Temporal", "Interval relation: A strictly before B"),
        ("ALLEN_AFTER", "Allen Temporal", "Interval relation: A strictly after B"),
        ("ALLEN_MEETS", "Allen Temporal", "Interval relation: A meets B at boundary"),
        ("ALLEN_MET_BY", "Allen Temporal", "Interval relation: A met by B"),
        ("ALLEN_OVERLAPS", "Allen Temporal", "Interval relation: A overlaps B"),
        ("ALLEN_DURING", "Allen Temporal", "Interval relation: A strictly during B"),
        ("ALLEN_CONTAINS", "Allen Temporal", "Interval relation: A contains B"),
        ("ALLEN_STARTS", "Allen Temporal", "Interval relation: A starts B"),
        ("ALLEN_FINISHES", "Allen Temporal", "Interval relation: A finishes B"),
        ("ALLEN_EQUALS", "Allen Temporal", "Interval relation: A temporally equals B"),
        ("RCC8_DC", "RCC-8 Spatial", "Disconnected: exterior spatial disjointness"),
        ("RCC8_EC", "RCC-8 Spatial", "Externally Connected: touching at boundary"),
        ("RCC8_PO", "RCC-8 Spatial", "Partial Overlap: sharing interior points"),
        ("RCC8_EQ", "RCC-8 Spatial", "Equal: identical spatial regions"),
        ("RCC8_TPP", "RCC-8 Spatial", "Tangential Proper Part: interior subset touching boundary"),
        ("RCC8_NTPP", "RCC-8 Spatial", "Non-Tangential Proper Part: strictly interior subset"),
        ("RCC8_TPPi", "RCC-8 Spatial", "Tangential Proper Part Inverse: containing touching part"),
        ("RCC8_NTPPi", "RCC-8 Spatial", "Non-Tangential Proper Part Inverse: containing interior"),
        ("PEARL_L1_ASSOC", "Pearl Causal", "Level 1 Association: observational correlation P(y|x)"),
        ("PEARL_L2_INTERV", "Pearl Causal", "Level 2 Intervention: do-calculus action P(y|do(x))"),
        ("PEARL_L3_COUNTER", "Pearl Causal", "Level 3 Counterfactual: retrospective reasoning P(y_x|x',y')"),
        ("MODAL_NECESSARY", "Modal Logic", "Box necessity operator □P"),
        ("MODAL_POSSIBLE", "Modal Logic", "Diamond possibility operator ◇P"),
        ("MODAL_OBLIGATORY", "Deontic Logic", "Deontic obligation O(P)"),
        ("MODAL_PERMISSIBLE", "Deontic Logic", "Deontic permission P(P)"),
        ("LTL_NEXT", "Temporal Logic", "LTL Next-time temporal operator X(P)"),
        ("LTL_ALWAYS", "Temporal Logic", "LTL Globally/Always temporal operator G(P)"),
        ("LTL_EVENTUALLY", "Temporal Logic", "LTL Finally/Eventually temporal operator F(P)"),
        ("LTL_UNTIL", "Temporal Logic", "LTL Until temporal operator P U Q"),
        ("AST_DEF_FUNC", "Code", "Function definition header"),
        ("AST_LOOP_BLOCK", "Code", "Loop execution block (for/while)"),
        ("AST_BRANCH_IF", "Code", "Conditional branching statement (if/else)"),
        ("AST_BIND_VAR", "Code", "Variable assignment binding (x = expr)"),
        ("AST_RETURN_VAL", "Code", "Return statement expression"),
        ("AST_CALL_INVOKE", "Code", "Function invocation call"),
        ("AST_CLASS_DEF", "Code", "Class structure declaration"),
        ("AST_EXCEPTION_TRY", "Code", "Exception handler try-catch block"),
        ("AST_RECURSIVE_CALL", "Code", "Self-referential recursive invocation"),
        ("AST_ASYNC_AWAIT", "Code", "Asynchronous coroutine suspend point"),
        ("AST_LAMBDA_EXPR", "Code", "Anonymous lambda expression closure"),
    ]
    for name, cat, desc in logic_and_ast:
        add(name, "Logic/Calculi", cat, desc)

    # 6. NSM Primes & Lojban Grammatical Primitives (40 Candidates)
    nsm_lojban = [
        ("NSM_I_ME", "NSM Primes", "nsm:I - 1st person singular subjective substantive"),
        ("NSM_YOU_PRON", "NSM Primes", "nsm:YOU - 2nd person singular subjective substantive"),
        ("NSM_SOMEONE_INDEF", "NSM Primes", "nsm:SOMEONE - indefinite personal substantive"),
        ("NSM_SOMETHING_THING", "NSM Primes", "nsm:SOMETHING - indefinite inanimate substantive"),
        ("NSM_PEOPLE_COLL", "NSM Primes", "nsm:PEOPLE - plural human collective"),
        ("NSM_BODY_CORP", "NSM Primes", "nsm:BODY - corporeal physical manifestation"),
        ("NSM_KIND_OF", "NSM Primes", "nsm:KIND - taxonomic relational category"),
        ("NSM_PART_OF", "NSM Primes", "nsm:PART - mereological structural component"),
        ("NSM_THIS_PROX", "NSM Primes", "nsm:THIS - proximal spatial/deictic determiner"),
        ("NSM_THE_SAME", "NSM Primes", "nsm:SAME - identity relational descriptor"),
        ("NSM_OTHER_ELSE", "NSM Primes", "nsm:OTHER - alterity non-identity descriptor"),
        ("NSM_ONE_CARD", "NSM Primes", "nsm:ONE - singular cardinality baseline"),
        ("NSM_TWO_CARD", "NSM Primes", "nsm:TWO - dual cardinality baseline"),
        ("NSM_MUCH_MANY", "NSM Primes", "nsm:MUCH - large quantity measure"),
        ("NSM_LITTLE_FEW", "NSM Primes", "nsm:FEW - small quantity measure"),
        ("NSM_GOOD_EVAL", "NSM Primes", "nsm:GOOD - positive axiological evaluation"),
        ("NSM_BAD_EVAL", "NSM Primes", "nsm:BAD - negative axiological evaluation"),
        ("NSM_BIG_SIZE", "NSM Primes", "nsm:BIG - positive magnitude metric"),
        ("NSM_SMALL_SIZE", "NSM Primes", "nsm:SMALL - negative magnitude metric"),
        ("NSM_KNOW_EPIST", "NSM Primes", "nsm:KNOW - epistemic confirmed state"),
        ("NSM_WANT_VOLIT", "NSM Primes", "nsm:WANT - intentional volitional desiderative"),
        ("NSM_FEEL_AFFECT", "NSM Primes", "nsm:FEEL - affective sensory internal perception"),
        ("NSM_SEE_VISUAL", "NSM Primes", "nsm:SEE - visual sensory gathering"),
        ("NSM_HEAR_AUDIO", "NSM Primes", "nsm:HEAR - auditory sensory gathering"),
        ("NSM_LIVE_ALIVE", "NSM Primes", "nsm:LIVE - biological living vitality"),
        ("NSM_DIE_MORTAL", "NSM Primes", "nsm:DIE - biological cessation of life"),
        ("NSM_WHEN_TIME", "NSM Primes", "nsm:WHEN - temporal coordinate anchor"),
        ("NSM_NOW_PRESENT", "NSM Primes", "nsm:NOW - current temporal instant"),
        ("NSM_BEFORE_PAST", "NSM Primes", "nsm:BEFORE - prior temporal coordinate"),
        ("NSM_AFTER_FUT", "NSM Primes", "nsm:AFTER - posterior temporal coordinate"),
        ("NSM_WHERE_SPACE", "NSM Primes", "nsm:WHERE - spatial coordinate locus"),
        ("NSM_HERE_LOC", "NSM Primes", "nsm:HERE - proximal spatial locus"),
        ("NSM_INSIDE_INT", "NSM Primes", "nsm:INSIDE - topological interior location"),
        ("NSM_FAR_DIST", "NSM Primes", "nsm:FAR - distal spatial separation"),
        ("NSM_NEAR_PROX", "NSM Primes", "nsm:NEAR - proximal spatial separation"),
        ("LJB_POI_RESTRICT", "Lojban Grammar", "poi: restrictive relative clause marker"),
        ("LJB_NOI_INCIDENT", "Lojban Grammar", "noi: non-restrictive incidental clause marker"),
        ("LJB_BA_FUTURE_TENSE_EXT", "Lojban Tense", "ba: future tense temporal displacement"),
        ("LJB_PU_PAST_TENSE_EXT", "Lojban Tense", "pu: past tense temporal displacement"),
        ("LJB_CA_PRESENT_TENSE_EXT", "Lojban Tense", "ca: present tense temporal anchor"),
    ]
    for name, cat, desc in nsm_lojban:
        add(name, "NSM/Lojban", cat, desc)

    # 7. Benchmark Specific Reasoning Operators (to bring total to 512+)
    benchmark_ops = [
        ("FOLIO_RULE_PREMISE", "FOLIO", "First-order logic conditional rule premise"),
        ("FOLIO_FACT_ASSERTION", "FOLIO", "First-order logic atomic ground fact assertion"),
        ("FOLIO_CONCLUSION_TARGET", "FOLIO", "Target deductive conclusion for proof verification"),
        ("FOLIO_SYLLOGISM_STEP", "FOLIO", "Categorical syllogism inference intermediate step"),
        ("PROOFWRITER_CLOSED_WORLD", "ProofWriter", "Closed world negation assumption marker"),
        ("PROOFWRITER_RULE_CHAIN", "ProofWriter", "Multi-hop deductive rule chaining link"),
        ("PROOFWRITER_CONTRADICTION", "ProofWriter", "Direct contradiction flag across rule set"),
        ("PROOFWRITER_QUERY_PROVE", "ProofWriter", "Proof verification target query statement"),
        ("BABI_MOVE_EVENT", "bAbI", "Agent spatial translation between named locations"),
        ("BABI_PICKUP_EVENT", "bAbI", "Agent acquires physical possession of artifact"),
        ("BABI_DROP_EVENT", "bAbI", "Agent relinquishes physical possession of artifact"),
        ("BABI_WHERE_QUERY", "bAbI", "Interrogative spatial locus query for object/agent"),
        ("CLUTRR_KIN_PARENT", "CLUTRR", "Kinship relation: direct parental lineage"),
        ("CLUTRR_KIN_CHILD", "CLUTRR", "Kinship relation: direct child descendant"),
        ("CLUTRR_KIN_SIBLING", "CLUTRR", "Kinship relation: shared parental descent"),
        ("CLUTRR_KIN_SPOUSE", "CLUTRR", "Kinship relation: marital / domestic union"),
        ("CLUTRR_KIN_GRANDPARENT", "CLUTRR", "Kinship relation: 2-hop parental ancestor"),
        ("CLUTRR_KIN_INLAW", "CLUTRR", "Kinship relation: affinity via marriage link"),
        ("CODE_CONTROL_FLOW_GRAPH", "Code", "Executable control flow graph basic block"),
        ("CODE_DATA_DEPENDENCY", "Code", "Data dependency edge between variable definitions"),
    ]
    for name, cat, desc in benchmark_ops:
        add(name, "Benchmark", cat, desc)

    # Pad with structured auxiliary dimensions if needed to reach exactly 512 total
    while len(candidates) < 512:
        idx = len(candidates)
        add(
            name=f"AUX_REASONING_DIM_{idx}",
            source="Auxiliary",
            category="Extended Reasoning",
            description=f"Auxiliary discrete reasoning dimension index {idx}",
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
    if "FN_PERCEPTION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_PERCEPTION"]] = np.maximum(get_c("NSM_SEE"), get_c("NSM_HEAR"))
    if "FN_COMMERCE_BUY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_COMMERCE_BUY"]] = np.maximum(get_c("NSM_HAVE"), get_c("WN_POSSESSION_ASSET"))
    if "FN_COMMERCE_SELL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_COMMERCE_SELL"]] = np.maximum(get_c("NSM_GIVE"), get_c("WN_POSSESSION_ASSET"))
    if "FN_POSSESSION_TRANSFER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_POSSESSION_TRANSFER"]] = np.maximum(get_c("NSM_GIVE"), get_c("WN_POSSESSION_ASSET"))
    if "FN_CO_AGENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_CO_AGENT"]] = np.maximum(get_c("VAL_X1_AGENT"), get_c("LJB_JE_AND"))
    if "FN_CO_PATIENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_CO_PATIENT"]] = np.maximum(get_c("VAL_X2_PATIENT"), get_c("LJB_JE_AND"))
    if "FN_DESCRIPTIVE_PROPERTY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_DESCRIPTIVE_PROPERTY"]] = get_c("TYPE_ATTRIBUTE_PROPERTY")
    if "FN_DEGREE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_DEGREE"]] = np.maximum(get_c("NSM_MUCH"), get_c("NSM_MORE"))
    if "FN_RESULT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_RESULT"]] = np.maximum(get_c("GRAPH_RETURN_VALUE"), get_c("CAUSAL_DIRECT_MECHANISM"))
    if "FN_STATE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_STATE"]] = get_c("TYPE_STATE")
    if "FN_CIRCUMSTANCES" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_CIRCUMSTANCES"]] = get_c("GRAPH_SCOPED_CONTEXT")
    if "FN_EFFECT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_EFFECT"]] = get_c("CAUSAL_DIRECT_MECHANISM")
    if "FN_TOPIC" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_TOPIC"]] = get_c("TYPE_ABSTRACT_CONCEPT")
    if "FN_MEDIUM" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_MEDIUM"]] = get_c("TYPE_COMMUNICATION_MSG")
    if "FN_MESSAGE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_MESSAGE"]] = np.maximum(get_c("TYPE_COMMUNICATION_MSG"), get_c("WN_COMMUNICATION_INFO"))
    if "FN_DONOR" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_DONOR"]] = np.maximum(get_c("VAL_X1_AGENT"), get_c("VAL_X4_SOURCE"))
    if "FN_ITEM" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_ITEM"]] = np.maximum(get_c("VAL_X2_PATIENT"), get_c("TYPE_ARTIFACT"))
    if "FN_OWNER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_OWNER"]] = np.maximum(get_c("VAL_X1_AGENT"), get_c("WN_POSSESSION_ASSET"))
    if "FN_DURATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_DURATION"]] = np.maximum(get_c("NSM_FOR_SOME_TIME"), get_c("TYPE_TEMPORAL_INTERVAL"))
    if "FN_FREQUENCY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_FREQUENCY"]] = get_c("NSM_CONTINUOUS_RATE")
    if "FN_PART_WHOLE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_PART_WHOLE"]] = np.maximum(get_c("NSM_PART"), get_c("MEREOLOGY_MERONYM_PART"))
    if "FN_CONTAINER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_CONTAINER"]] = np.maximum(get_c("TYPE_ARTIFACT"), get_c("TYPE_SPATIAL_REGION"))
    if "FN_CONTENTS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FN_CONTENTS"]] = get_c("TYPE_INANIMATE_PHYSICAL")

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
    if "AMR_ACCOMPANIER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_ACCOMPANIER"]] = get_c("LJB_JOI_ASSOCIATION")
    if "AMR_TOPIC" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_TOPIC"]] = get_c("TYPE_ABSTRACT_CONCEPT")
    if "AMR_MEDIUM" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_MEDIUM"]] = get_c("TYPE_COMMUNICATION_MSG")
    if "AMR_SOURCE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_SOURCE"]] = get_c("VAL_X4_SOURCE")
    if "AMR_FREQUENCY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_FREQUENCY"]] = get_c("NSM_CONTINUOUS_RATE")
    if "AMR_DEGREE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_DEGREE"]] = get_c("NSM_MUCH")
    if "AMR_DOMAIN" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_DOMAIN"]] = get_c("GRAPH_SCOPED_CONTEXT")
    if "AMR_MOD" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_MOD"]] = get_c("TYPE_ATTRIBUTE_PROPERTY")
    if "AMR_ORDINAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_ORDINAL"]] = get_c("TYPE_NUMERIC_VALUE")
    if "AMR_QUANT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AMR_QUANT"]] = np.maximum(get_c("NSM_ONE"), get_c("NSM_ALL"))

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
    if "WN_PLANT_FLORA_EXT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_PLANT_FLORA_EXT"]] = get_c("WN_PLANT_FLORA")
    if "WN_FOOD_NUTRIENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_FOOD_NUTRIENT"]] = get_c("TYPE_NATURAL_OBJECT")
    if "WN_SUBSTANCE_MATTER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_SUBSTANCE_MATTER"]] = np.maximum(get_c("TYPE_INANIMATE_PHYSICAL"), get_c("WN_SUBSTANCE_MATTER"))
    if "WN_PHENOMENON_NATURAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_PHENOMENON_NATURAL"]] = np.maximum(get_c("TYPE_EVENT"), get_c("WN_PHENOMENON_EVENT"))
    if "WN_PSYCH_FEELING" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_PSYCH_FEELING"]] = np.maximum(get_c("NSM_FEEL"), get_c("WN_FEELING_EMOTION"))
    if "WN_ACT_ACTION_EXT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_ACT_ACTION_EXT"]] = np.maximum(get_c("NSM_DO"), get_c("WN_ACT_ACTION"))
    if "WN_GROUP_COLLECTIVE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_GROUP_COLLECTIVE"]] = np.maximum(get_c("TYPE_ORGANIZATION"), get_c("WN_GROUP_SOCIAL"))
    if "WN_POSSESSION_PROPERTY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_POSSESSION_PROPERTY"]] = np.maximum(get_c("NSM_HAVE"), get_c("WN_POSSESSION_ASSET"))
    if "WN_ATTRIBUTE_QUALITY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_ATTRIBUTE_QUALITY"]] = np.maximum(get_c("TYPE_ATTRIBUTE_PROPERTY"), get_c("WN_ATTRIBUTE_PROP"))
    if "WN_QUANTITY_MEASURE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_QUANTITY_MEASURE"]] = np.maximum(get_c("TYPE_NUMERIC_VALUE"), get_c("WN_QUANTITY_METRIC"))
    if "WN_TIME_PERIOD" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_TIME_PERIOD"]] = np.maximum(get_c("TYPE_TEMPORAL_INTERVAL"), get_c("WN_TIME_TEMPORAL"))
    if "WN_SHAPE_FORM" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_SHAPE_FORM"]] = np.maximum(get_c("TYPE_SPATIAL_REGION"), get_c("WN_SHAPE_FORM"))
    if "WN_BODY_PART_EXT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_BODY_PART_EXT"]] = np.maximum(get_c("NSM_PART"), get_c("WN_BODY_ANATOMY"))
    if "WN_CHANGE_TRANSITION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_CHANGE_TRANSITION"]] = np.maximum(get_c("TYPE_PROCESS"), get_c("WN_CHANGE_TRANSITION"))
    if "WN_MOTION_MOVEMENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_MOTION_MOVEMENT"]] = np.maximum(get_c("NSM_MOVE"), get_c("WN_MOTION_MOVEMENT"))
    if "WN_CONTACT_TOUCH" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_CONTACT_TOUCH"]] = np.maximum(get_c("NSM_TOUCH"), get_c("WN_CONTACT_TOUCH"))
    if "WN_PERCEPTION_SENSE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_PERCEPTION_SENSE"]] = np.maximum(get_c("NSM_SEE"), get_c("WN_PERCEPTION_SENSE"))
    if "WN_CONSUMPTION_INGEST" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_CONSUMPTION_INGEST"]] = np.maximum(get_c("NSM_DO"), get_c("WN_CONSUMPTION_INGEST"))
    if "WN_CREATION_MAKE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_CREATION_MAKE"]] = np.maximum(get_c("NSM_DO"), get_c("WN_CREATION_MAKE"))
    if "WN_COMPETITION_CONTEST" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_COMPETITION_CONTEST"]] = np.maximum(get_c("TYPE_EVENT"), get_c("WN_COMPETITION_CONTEST"))
    if "WN_WEATHER_METEO" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_WEATHER_METEO"]] = np.maximum(get_c("TYPE_EVENT"), get_c("WN_PHENOMENON_EVENT"))
    if "WN_TOOL_DEVICE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_TOOL_DEVICE"]] = np.maximum(get_c("TYPE_ARTIFACT"), get_c("ROLE_INSTRUMENT_USABLE"))
    if "WN_INSTRUMENT_DEVICE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_INSTRUMENT_DEVICE"]] = np.maximum(get_c("TYPE_ARTIFACT"), get_c("VAL_X5_INSTRUMENT"))
    if "WN_SYMBOLIC_NOTATION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_SYMBOLIC_NOTATION"]] = np.maximum(get_c("TYPE_COMMUNICATION_MSG"), get_c("WN_COMMUNICATION_INFO"))
    if "WN_CONTAINER_VESSEL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_CONTAINER_VESSEL"]] = get_c("TYPE_ARTIFACT")
    if "WN_GARMENT_CLOTHING" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["WN_GARMENT_CLOTHING"]] = get_c("TYPE_ARTIFACT")

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
    if "LOGIC_EQUIVALENCE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LOGIC_EQUIVALENCE"]] = get_c("LJB_DU_IDENTITY")
    if "LOGIC_TAUTOLOGY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LOGIC_TAUTOLOGY"]] = get_c("SOLVER_PROOF_VALIDATED")
    if "LOGIC_CONTRADICTION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LOGIC_CONTRADICTION"]] = get_c("SOLVER_CONTRADICTION_FLAG")
    if "ALLEN_BEFORE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["ALLEN_BEFORE"]] = get_c("TEMPORAL_ALLEN_BEFORE")
    if "ALLEN_AFTER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["ALLEN_AFTER"]] = get_c("TEMPORAL_ALLEN_AFTER")
    if "ALLEN_MEETS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["ALLEN_MEETS"]] = get_c("TEMPORAL_ALLEN_MEETS")
    if "ALLEN_MET_BY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["ALLEN_MET_BY"]] = get_c("TEMPORAL_ALLEN_MEETS")
    if "ALLEN_OVERLAPS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["ALLEN_OVERLAPS"]] = get_c("TEMPORAL_ALLEN_OVERLAPS")
    if "ALLEN_DURING" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["ALLEN_DURING"]] = get_c("TEMPORAL_ALLEN_DURING")
    if "ALLEN_CONTAINS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["ALLEN_CONTAINS"]] = get_c("TEMPORAL_ALLEN_STARTS")
    if "ALLEN_STARTS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["ALLEN_STARTS"]] = get_c("TEMPORAL_ALLEN_STARTS")
    if "ALLEN_FINISHES" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["ALLEN_FINISHES"]] = get_c("TEMPORAL_ALLEN_FINISHES")
    if "ALLEN_EQUALS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["ALLEN_EQUALS"]] = get_c("TEMPORAL_ALLEN_EQUALS")
    if "RCC8_DC" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["RCC8_DC"]] = get_c("SPATIAL_RCC_DISCONNECTED")
    if "RCC8_EC" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["RCC8_EC"]] = get_c("SPATIAL_RCC_EXT_CONNECTED")
    if "RCC8_PO" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["RCC8_PO"]] = get_c("SPATIAL_RCC_PARTIAL_OVERLAP")
    if "RCC8_EQ" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["RCC8_EQ"]] = get_c("SPATIAL_RCC_TANG_PROPER_PART")
    if "RCC8_TPP" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["RCC8_TPP"]] = get_c("SPATIAL_RCC_TANG_PROPER_PART")
    if "RCC8_NTPP" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["RCC8_NTPP"]] = get_c("SPATIAL_RCC_NON_TANG_PART")
    if "RCC8_TPPi" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["RCC8_TPPi"]] = get_c("SPATIAL_RCC_TANG_PROPER_PART")
    if "RCC8_NTPPi" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["RCC8_NTPPi"]] = get_c("SPATIAL_RCC_NON_TANG_PART")
    if "PEARL_L1_ASSOC" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["PEARL_L1_ASSOC"]] = get_c("CAUSAL_L1_ASSOCIATIONAL")
    if "PEARL_L2_INTERV" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["PEARL_L2_INTERV"]] = get_c("CAUSAL_L2_INTERVENTIONAL")
    if "PEARL_L3_COUNTER" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["PEARL_L3_COUNTER"]] = get_c("CAUSAL_L3_COUNTERFACTUAL")
    if "MODAL_NECESSARY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["MODAL_NECESSARY"]] = get_c("EPIST_PROB_CERTAIN")
    if "MODAL_POSSIBLE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["MODAL_POSSIBLE"]] = np.maximum(get_c("NSM_MAYBE"), get_c("MODALITY_HYPOTHETICAL"))
    if "MODAL_OBLIGATORY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["MODAL_OBLIGATORY"]] = get_c("EPIST_AXIOMATIC_PREMISE")
    if "MODAL_PERMISSIBLE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["MODAL_PERMISSIBLE"]] = get_c("MODALITY_LITERAL")
    if "LTL_NEXT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LTL_NEXT"]] = get_c("TEMPORAL_ALLEN_AFTER")
    if "LTL_ALWAYS" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LTL_ALWAYS"]] = get_c("NSM_ALL")
    if "LTL_EVENTUALLY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LTL_EVENTUALLY"]] = get_c("NSM_SOME")
    if "LTL_UNTIL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LTL_UNTIL"]] = get_c("NSM_FOR_SOME_TIME")
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
    if "AST_CALL_INVOKE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_CALL_INVOKE"]] = get_c("GRAPH_INVOCATION_HEAD")
    if "AST_CLASS_DEF" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_CLASS_DEF"]] = get_c("GRAPH_SCOPED_CONTEXT")
    if "AST_EXCEPTION_TRY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_EXCEPTION_TRY"]] = get_c("GRAPH_EXCEPTION_HANDLE")
    if "AST_RECURSIVE_CALL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_RECURSIVE_CALL"]] = get_c("GRAPH_RECURSIVE_REF")
    if "AST_ASYNC_AWAIT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_ASYNC_AWAIT"]] = get_c("LJB_ASYNC_CONCURRENT")
    if "AST_LAMBDA_EXPR" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["AST_LAMBDA_EXPR"]] = get_c("GRAPH_FUNCTION_DEF")

    # 6. Semantic Projection for NSM & Lojban
    if "NSM_I_ME" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_I_ME"]] = get_c("NSM_I")
    if "NSM_YOU_PRON" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_YOU_PRON"]] = get_c("NSM_YOU")
    if "NSM_SOMEONE_INDEF" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_SOMEONE_INDEF"]] = get_c("NSM_SOMEONE")
    if "NSM_SOMETHING_THING" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_SOMETHING_THING"]] = get_c("NSM_SOMETHING")
    if "NSM_PEOPLE_COLL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_PEOPLE_COLL"]] = get_c("NSM_PEOPLE")
    if "NSM_BODY_CORP" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_BODY_CORP"]] = get_c("NSM_BODY")
    if "NSM_KIND_OF" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_KIND_OF"]] = get_c("NSM_KIND")
    if "NSM_PART_OF" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_PART_OF"]] = get_c("NSM_PART")
    if "NSM_THIS_PROX" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_THIS_PROX"]] = get_c("NSM_THIS")
    if "NSM_THE_SAME" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_THE_SAME"]] = get_c("NSM_SAME")
    if "NSM_OTHER_ELSE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_OTHER_ELSE"]] = get_c("NSM_OTHER")
    if "NSM_ONE_CARD" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_ONE_CARD"]] = get_c("NSM_ONE")
    if "NSM_TWO_CARD" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_TWO_CARD"]] = get_c("NSM_TWO")
    if "NSM_MUCH_MANY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_MUCH_MANY"]] = get_c("NSM_MUCH")
    if "NSM_LITTLE_FEW" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_LITTLE_FEW"]] = get_c("NSM_LITTLE")
    if "NSM_GOOD_EVAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_GOOD_EVAL"]] = get_c("NSM_GOOD")
    if "NSM_BAD_EVAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_BAD_EVAL"]] = get_c("NSM_BAD")
    if "NSM_BIG_SIZE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_BIG_SIZE"]] = get_c("NSM_BIG")
    if "NSM_SMALL_SIZE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_SMALL_SIZE"]] = get_c("NSM_SMALL")
    if "NSM_KNOW_EPIST" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_KNOW_EPIST"]] = get_c("NSM_KNOW")
    if "NSM_WANT_VOLIT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_WANT_VOLIT"]] = get_c("NSM_WANT")
    if "NSM_FEEL_AFFECT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_FEEL_AFFECT"]] = get_c("NSM_FEEL")
    if "NSM_SEE_VISUAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_SEE_VISUAL"]] = get_c("NSM_SEE")
    if "NSM_HEAR_AUDIO" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_HEAR_AUDIO"]] = get_c("NSM_HEAR")
    if "NSM_LIVE_ALIVE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_LIVE_ALIVE"]] = get_c("NSM_LIVE")
    if "NSM_DIE_MORTAL" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_DIE_MORTAL"]] = get_c("NSM_DIE")
    if "NSM_WHEN_TIME" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_WHEN_TIME"]] = get_c("NSM_WHEN")
    if "NSM_NOW_PRESENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_NOW_PRESENT"]] = get_c("NSM_NOW")
    if "NSM_BEFORE_PAST" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_BEFORE_PAST"]] = get_c("NSM_BEFORE")
    if "NSM_AFTER_FUT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_AFTER_FUT"]] = get_c("NSM_AFTER")
    if "NSM_WHERE_SPACE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_WHERE_SPACE"]] = get_c("NSM_WHERE")
    if "NSM_HERE_LOC" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_HERE_LOC"]] = get_c("NSM_HERE")
    if "NSM_INSIDE_INT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_INSIDE_INT"]] = get_c("NSM_INSIDE")
    if "NSM_FAR_DIST" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_FAR_DIST"]] = get_c("NSM_FAR")
    if "NSM_NEAR_PROX" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["NSM_NEAR_PROX"]] = get_c("NSM_NEAR")
    if "LJB_POI_RESTRICT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LJB_POI_RESTRICT"]] = get_c("LJB_POI_RESTRICTIVE")
    if "LJB_NOI_INCIDENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LJB_NOI_INCIDENT"]] = get_c("LJB_NOI_INCIDENTAL")
    if "LJB_BA_FUTURE_TENSE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LJB_BA_FUTURE_TENSE"]] = get_c("LJB_BA_FUTURE_TENSE")
    if "LJB_PU_PAST_TENSE_EXT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LJB_PU_PAST_TENSE_EXT"]] = get_c("LJB_PU_PAST_TENSE")
    if "LJB_CA_PRESENT_TENSE_EXT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["LJB_CA_PRESENT_TENSE_EXT"]] = get_c("LJB_CA_PRESENT_TENSE")

    # 7. Semantic Projection for Benchmark Operators
    if "FOLIO_RULE_PREMISE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FOLIO_RULE_PREMISE"]] = get_c("LJB_GANAI_IF_THEN")
    if "FOLIO_FACT_ASSERTION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FOLIO_FACT_ASSERTION"]] = get_c("GRAPH_ASSERTION_CLAIM")
    if "FOLIO_CONCLUSION_TARGET" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FOLIO_CONCLUSION_TARGET"]] = get_c("SOLVER_PROOF_VALIDATED")
    if "FOLIO_SYLLOGISM_STEP" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["FOLIO_SYLLOGISM_STEP"]] = get_c("EPIST_DEDUCTIVE_INFERENCE")
    if "PROOFWRITER_CLOSED_WORLD" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["PROOFWRITER_CLOSED_WORLD"]] = get_c("SOLVER_CWA_CLOSED_WORLD")
    if "PROOFWRITER_RULE_CHAIN" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["PROOFWRITER_RULE_CHAIN"]] = get_c("EPIST_DEDUCTIVE_INFERENCE")
    if "PROOFWRITER_CONTRADICTION" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["PROOFWRITER_CONTRADICTION"]] = get_c("SOLVER_CONTRADICTION_FLAG")
    if "PROOFWRITER_QUERY_PROVE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["PROOFWRITER_QUERY_PROVE"]] = get_c("EPIST_AXIOMATIC_PREMISE")
    if "BABI_MOVE_EVENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["BABI_MOVE_EVENT"]] = get_c("NSM_MOVE")
    if "BABI_PICKUP_EVENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["BABI_PICKUP_EVENT"]] = np.maximum(get_c("NSM_TOUCH"), get_c("NSM_HAVE"))
    if "BABI_DROP_EVENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["BABI_DROP_EVENT"]] = get_c("NSM_DO")
    if "BABI_WHERE_QUERY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["BABI_WHERE_QUERY"]] = np.maximum(get_c("GRAPH_QUERY_TARGET"), get_c("NSM_WHERE"))
    if "CLUTRR_KIN_PARENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["CLUTRR_KIN_PARENT"]] = get_c("NSM_BORN")
    if "CLUTRR_KIN_CHILD" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["CLUTRR_KIN_CHILD"]] = get_c("NSM_BORN")
    if "CLUTRR_KIN_SIBLING" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["CLUTRR_KIN_SIBLING"]] = get_c("NSM_SAME")
    if "CLUTRR_KIN_SPOUSE" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["CLUTRR_KIN_SPOUSE"]] = get_c("LJB_SOI_RECIPROCAL")
    if "CLUTRR_KIN_GRANDPARENT" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["CLUTRR_KIN_GRANDPARENT"]] = get_c("NSM_BORN")
    if "CLUTRR_KIN_INLAW" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["CLUTRR_KIN_INLAW"]] = get_c("LJB_SOI_RECIPROCAL")
    if "CODE_CONTROL_FLOW_GRAPH" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["CODE_CONTROL_FLOW_GRAPH"]] = get_c("GRAPH_SCOPED_CONTEXT")
    if "CODE_DATA_DEPENDENCY" in cand_name_to_idx:
        candidate_matrix[:, cand_name_to_idx["CODE_DATA_DEPENDENCY"]] = get_c("GRAPH_VARIABLE_BIND")

    return candidate_matrix


def export_candidate_pool(
    candidates: Optional[List[CandidateDimension]] = None,
    output_path: Union[str, Path] = "output/candidate_pool.json",
) -> Path:
    """Exports the candidate pool to a structured JSON file."""
    if candidates is None:
        candidates = build_candidate_pool()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data = [c.to_dict() for c in candidates]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out_file

