"""Canonical 256-dimension slot registry for QUANTA.

Partitions the 256 dimensions into four isolated 64-slot bands:
- Band 0 (0-63): Universal NSM Primes & Kinematics
- Band 1 (64-127): Structural Valencies & Concurrency Topology
- Band 2 (128-191): Ontological Signatures & Theory of Mind Modalities
- Band 3 (192-255): Epistemic Bounds, Probability & Logical Metarules
"""

from __future__ import annotations
from dataclasses import dataclass
import enum
from typing import Dict, List, Optional


class SlotBand(enum.IntEnum):
    BAND_0_NSM_KINEMATICS = 0
    BAND_1_VALENCIES_TOPOLOGY = 1
    BAND_2_ONTOLOGY_MODALITY = 2
    BAND_3_EPISTEMIC_METARULES = 3


@dataclass(frozen=True)
class SlotDefinition:
    index: int
    name: str
    band: SlotBand
    category: str
    description: str


# ==============================================================================
# BAND 0: Universal NSM Primes & Kinematics (0-63)
# ==============================================================================
BAND_0_SLOTS = [
    # Substantives (00-05)
    SlotDefinition(0, "NSM_I", SlotBand.BAND_0_NSM_KINEMATICS, "Substantives", "First-person speaker prime"),
    SlotDefinition(1, "NSM_YOU", SlotBand.BAND_0_NSM_KINEMATICS, "Substantives", "Second-person addressee prime"),
    SlotDefinition(2, "NSM_SOMEONE", SlotBand.BAND_0_NSM_KINEMATICS, "Substantives", "Indefinite person/agent prime"),
    SlotDefinition(3, "NSM_SOMETHING", SlotBand.BAND_0_NSM_KINEMATICS, "Substantives", "Inanimate entity or thing prime"),
    SlotDefinition(4, "NSM_PEOPLE", SlotBand.BAND_0_NSM_KINEMATICS, "Substantives", "Plural humans prime"),
    SlotDefinition(5, "NSM_BODY", SlotBand.BAND_0_NSM_KINEMATICS, "Substantives", "Physical organism body prime"),
    
    # Quantifiers & Determiners (06-17)
    SlotDefinition(6, "NSM_THIS", SlotBand.BAND_0_NSM_KINEMATICS, "Determiners", "Proximal deictic determiner"),
    SlotDefinition(7, "NSM_SAME", SlotBand.BAND_0_NSM_KINEMATICS, "Determiners", "Identity relation prime"),
    SlotDefinition(8, "NSM_OTHER", SlotBand.BAND_0_NSM_KINEMATICS, "Determiners", "Alterior / difference prime"),
    SlotDefinition(9, "NSM_ONE", SlotBand.BAND_0_NSM_KINEMATICS, "Quantifiers", "Singular count prime"),
    SlotDefinition(10, "NSM_TWO", SlotBand.BAND_0_NSM_KINEMATICS, "Quantifiers", "Dual count prime"),
    SlotDefinition(11, "NSM_MUCH", SlotBand.BAND_0_NSM_KINEMATICS, "Quantifiers", "Large quantity / mass prime"),
    SlotDefinition(12, "NSM_LITTLE", SlotBand.BAND_0_NSM_KINEMATICS, "Quantifiers", "Small quantity / mass prime"),
    SlotDefinition(13, "NSM_SOME", SlotBand.BAND_0_NSM_KINEMATICS, "Quantifiers", "Existential partition prime"),
    SlotDefinition(14, "NSM_ALL", SlotBand.BAND_0_NSM_KINEMATICS, "Quantifiers", "Universal quantifier prime"),
    SlotDefinition(15, "NSM_MORE", SlotBand.BAND_0_NSM_KINEMATICS, "Quantifiers", "Additive / comparative excess prime"),
    SlotDefinition(16, "NSM_FEW", SlotBand.BAND_0_NSM_KINEMATICS, "Quantifiers", "Paucal quantifier prime"),
    SlotDefinition(17, "NSM_PART", SlotBand.BAND_0_NSM_KINEMATICS, "Quantifiers", "Mereological part prime"),

    # Evaluators, Descriptors, & Mental (18-29)
    SlotDefinition(18, "NSM_GOOD", SlotBand.BAND_0_NSM_KINEMATICS, "Evaluators", "Positive evaluation prime"),
    SlotDefinition(19, "NSM_BAD", SlotBand.BAND_0_NSM_KINEMATICS, "Evaluators", "Negative evaluation prime"),
    SlotDefinition(20, "NSM_BIG", SlotBand.BAND_0_NSM_KINEMATICS, "Descriptors", "Large magnitude prime"),
    SlotDefinition(21, "NSM_SMALL", SlotBand.BAND_0_NSM_KINEMATICS, "Descriptors", "Small magnitude prime"),
    SlotDefinition(22, "NSM_VERY", SlotBand.BAND_0_NSM_KINEMATICS, "Augmentatives", "Intensifier prime"),
    SlotDefinition(23, "NSM_TRUE", SlotBand.BAND_0_NSM_KINEMATICS, "Evaluators", "Truthfulness / verity prime"),
    SlotDefinition(24, "NSM_THINK", SlotBand.BAND_0_NSM_KINEMATICS, "Mental Predicates", "Cognition prime"),
    SlotDefinition(25, "NSM_KNOW", SlotBand.BAND_0_NSM_KINEMATICS, "Mental Predicates", "Knowledge prime"),
    SlotDefinition(26, "NSM_WANT", SlotBand.BAND_0_NSM_KINEMATICS, "Mental Predicates", "Volition / desire prime"),
    SlotDefinition(27, "NSM_FEEL", SlotBand.BAND_0_NSM_KINEMATICS, "Mental Predicates", "Sensation / affect prime"),
    SlotDefinition(28, "NSM_SEE", SlotBand.BAND_0_NSM_KINEMATICS, "Perception", "Visual perception prime"),
    SlotDefinition(29, "NSM_HEAR", SlotBand.BAND_0_NSM_KINEMATICS, "Perception", "Auditory perception prime"),

    # Actions, Events, & Vitality (30-42)
    SlotDefinition(30, "NSM_SAY", SlotBand.BAND_0_NSM_KINEMATICS, "Communication", "Speech act prime"),
    SlotDefinition(31, "NSM_WORDS", SlotBand.BAND_0_NSM_KINEMATICS, "Communication", "Lexical / linguistic payload prime"),
    SlotDefinition(32, "NSM_DO", SlotBand.BAND_0_NSM_KINEMATICS, "Action", "Active agentive action prime"),
    SlotDefinition(33, "NSM_HAPPEN", SlotBand.BAND_0_NSM_KINEMATICS, "Events", "Spontaneous occurrence prime"),
    SlotDefinition(34, "NSM_MOVE", SlotBand.BAND_0_NSM_KINEMATICS, "Action", "Kinematic motion prime"),
    SlotDefinition(35, "NSM_TOUCH", SlotBand.BAND_0_NSM_KINEMATICS, "Action", "Physical contact prime"),
    SlotDefinition(36, "NSM_BE_SOMEWHERE", SlotBand.BAND_0_NSM_KINEMATICS, "Location", "Locative presence prime"),
    SlotDefinition(37, "NSM_THERE_IS", SlotBand.BAND_0_NSM_KINEMATICS, "Existence", "Existential ontic presence prime"),
    SlotDefinition(38, "NSM_HAVE", SlotBand.BAND_0_NSM_KINEMATICS, "Possession", "Possessive association prime"),
    SlotDefinition(39, "NSM_LIVE", SlotBand.BAND_0_NSM_KINEMATICS, "Life", "Biological animation prime"),
    SlotDefinition(40, "NSM_DIE", SlotBand.BAND_0_NSM_KINEMATICS, "Life", "Cessation of life prime"),
    SlotDefinition(41, "NSM_BORN", SlotBand.BAND_0_NSM_KINEMATICS, "Life", "Inception of life prime"),
    SlotDefinition(42, "NSM_GROW", SlotBand.BAND_0_NSM_KINEMATICS, "Life", "Developmental expansion prime"),

    # Time & Space (43-59)
    SlotDefinition(43, "NSM_NOW", SlotBand.BAND_0_NSM_KINEMATICS, "Time", "Temporal present prime"),
    SlotDefinition(44, "NSM_BEFORE", SlotBand.BAND_0_NSM_KINEMATICS, "Time", "Temporal precedence prime"),
    SlotDefinition(45, "NSM_AFTER", SlotBand.BAND_0_NSM_KINEMATICS, "Time", "Temporal succession prime"),
    SlotDefinition(46, "NSM_A_LONG_TIME", SlotBand.BAND_0_NSM_KINEMATICS, "Time", "Extended temporal duration"),
    SlotDefinition(47, "NSM_A_SHORT_TIME", SlotBand.BAND_0_NSM_KINEMATICS, "Time", "Brief temporal duration"),
    SlotDefinition(48, "NSM_FOR_SOME_TIME", SlotBand.BAND_0_NSM_KINEMATICS, "Time", "Bounded temporal interval"),
    SlotDefinition(49, "NSM_MOMENT", SlotBand.BAND_0_NSM_KINEMATICS, "Time", "Instantaneous point in time"),
    SlotDefinition(50, "NSM_HERE", SlotBand.BAND_0_NSM_KINEMATICS, "Space", "Proximal spatial location"),
    SlotDefinition(51, "NSM_ABOVE", SlotBand.BAND_0_NSM_KINEMATICS, "Space", "Superior vertical spatial relation"),
    SlotDefinition(52, "NSM_BELOW", SlotBand.BAND_0_NSM_KINEMATICS, "Space", "Inferior vertical spatial relation"),
    SlotDefinition(53, "NSM_FAR", SlotBand.BAND_0_NSM_KINEMATICS, "Space", "Distal spatial relation"),
    SlotDefinition(54, "NSM_NEAR", SlotBand.BAND_0_NSM_KINEMATICS, "Space", "Proximal spatial relation"),
    SlotDefinition(55, "NSM_SIDE", SlotBand.BAND_0_NSM_KINEMATICS, "Space", "Lateral spatial relation"),
    SlotDefinition(56, "NSM_INSIDE", SlotBand.BAND_0_NSM_KINEMATICS, "Space", "Interior topological containment"),
    SlotDefinition(57, "NSM_OUTSIDE", SlotBand.BAND_0_NSM_KINEMATICS, "Space", "Exterior spatial exclusion"),
    SlotDefinition(58, "NSM_TOUCHING", SlotBand.BAND_0_NSM_KINEMATICS, "Space", "Contiguous boundary contact"),
    SlotDefinition(59, "NSM_BETWEEN", SlotBand.BAND_0_NSM_KINEMATICS, "Space", "Intermediate spatial placement"),

    # Continuous Kinematics & Logical Modality (60-63)
    SlotDefinition(60, "NSM_CONTINUOUS_RATE", SlotBand.BAND_0_NSM_KINEMATICS, "Kinematics", "Constant velocity / rate of change"),
    SlotDefinition(61, "NSM_ACCELERATING_RATE", SlotBand.BAND_0_NSM_KINEMATICS, "Kinematics", "Second-order accelerating rate"),
    SlotDefinition(62, "NSM_CAN", SlotBand.BAND_0_NSM_KINEMATICS, "Logical Modality", "Possibility / dynamic ability prime"),
    SlotDefinition(63, "NSM_MAYBE", SlotBand.BAND_0_NSM_KINEMATICS, "Logical Modality", "Epistemic possibility prime"),
]

# ==============================================================================
# BAND 1: Structural Valencies & Concurrency Topology (64-127)
# ==============================================================================
BAND_1_SLOTS = [
    # Predicate Place Structures (64-74)
    SlotDefinition(64, "VAL_X1_AGENT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Actor / initiator place x1"),
    SlotDefinition(65, "VAL_X2_PATIENT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Patient / theme / undergoer place x2"),
    SlotDefinition(66, "VAL_X3_DESTINATION", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Goal / recipient / destination place x3"),
    SlotDefinition(67, "VAL_X4_SOURCE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Origin / source place x4"),
    SlotDefinition(68, "VAL_X5_INSTRUMENT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Instrument / medium place x5"),
    SlotDefinition(69, "VAL_EXPERIENCER", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Experiencer / sentient recipient"),
    SlotDefinition(70, "VAL_TIME_SLOT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Temporal anchor argument"),
    SlotDefinition(71, "VAL_LOCATION_SLOT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Spatial frame argument"),
    SlotDefinition(72, "VAL_MANNER_SLOT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Manner / modification argument"),
    SlotDefinition(73, "VAL_PURPOSE_SLOT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Teleological goal argument"),
    SlotDefinition(74, "VAL_RESULT_SLOT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Valency", "Result / output state argument"),

    # Logical Connectives / cmavo (75-95)
    SlotDefinition(75, "LJB_NA_NEGATION", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Connectives", "Brute truth-functional negation"),
    SlotDefinition(76, "LJB_JE_AND", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Connectives", "Conjunction / logical AND"),
    SlotDefinition(77, "LJB_JA_OR", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Connectives", "Inclusive disjunction / logical OR"),
    SlotDefinition(78, "LJB_JON_XOR", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Connectives", "Exclusive disjunction / logical XOR"),
    SlotDefinition(79, "LJB_GANAI_IF_THEN", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Connectives", "Material implication / conditional"),
    SlotDefinition(80, "LJB_DU_IDENTITY", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Connectives", "Strict logical identity (=)"),
    SlotDefinition(81, "LJB_PU_PAST_TENSE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Tense", "Past tense grammatical marker"),
    SlotDefinition(82, "LJB_CA_PRESENT_TENSE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Tense", "Present tense grammatical marker"),
    SlotDefinition(83, "LJB_BA_FUTURE_TENSE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Tense", "Future tense grammatical marker"),
    SlotDefinition(84, "LJB_VI_SHORT_DISTANCE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Spatial Tense", "Near distance spatial marker"),
    SlotDefinition(85, "LJB_VA_MEDIUM_DISTANCE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Spatial Tense", "Medium distance spatial marker"),
    SlotDefinition(86, "LJB_VU_LONG_DISTANCE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Spatial Tense", "Far distance spatial marker"),
    SlotDefinition(87, "LJB_ZI_SHORT_PAST", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Temporal Interval", "Recent past marker"),
    SlotDefinition(88, "LJB_ZA_MEDIUM_PAST", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Temporal Interval", "Intermediate past marker"),
    SlotDefinition(89, "LJB_ZU_LONG_PAST", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Temporal Interval", "Ancient/distant past marker"),
    SlotDefinition(90, "LJB_RO_ALL_QUANT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Quantifiers", "All / every formal quantifier"),
    SlotDefinition(91, "LJB_SUO_AT_LEAST_ONE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Quantifiers", "At least one existential quantifier"),
    SlotDefinition(92, "LJB_NO_NONE_QUANT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Quantifiers", "Zero / none formal quantifier"),
    SlotDefinition(93, "LJB_MOI_ORDINAL", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Quantifiers", "Ordinal modifier"),
    SlotDefinition(94, "LJB_MEI_CARDINAL", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Quantifiers", "Cardinal group converter"),
    SlotDefinition(95, "LJB_SOI_RECIPROCAL", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Connectives", "Reciprocal / mutual relation marker"),

    # Asynchronous Concurrency markers (96-98)
    SlotDefinition(96, "LJB_ASYNC_CONCURRENT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Concurrency", "Concurrent non-blocking execution"),
    SlotDefinition(97, "LJB_MUTEX_DEPENDENCY", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Concurrency", "Mutual exclusion synchronization dependency"),
    SlotDefinition(98, "LJB_RACE_CONDITION", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Concurrency", "Non-deterministic race order flag"),

    # Graph Structural Topology (99-127)
    SlotDefinition(99, "GRAPH_ROOT_NODE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Root proposition node of ASG"),
    SlotDefinition(100, "GRAPH_LEAF", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Terminal leaf entity in ASG"),
    SlotDefinition(101, "GRAPH_RECURSIVE_REF", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Self-referential recursive backlink"),
    SlotDefinition(102, "GRAPH_IS_SUB_EXP", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Nested sub-expression / sub-graph"),
    SlotDefinition(103, "GRAPH_CYCLIC_BACKLINK", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Cyclic co-inductive edge pointer"),
    SlotDefinition(104, "GRAPH_ORDERED_SEQ", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Ordered sequence child relation"),
    SlotDefinition(105, "GRAPH_BRANCH_COND", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Branch condition evaluation head"),
    SlotDefinition(106, "GRAPH_BRANCH_THEN", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Branch true evaluation branch"),
    SlotDefinition(107, "GRAPH_BRANCH_ELSE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Branch false evaluation branch"),
    SlotDefinition(108, "GRAPH_MERKLE_FOLD_POINT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Merkle tree folding boundary"),
    SlotDefinition(109, "GRAPH_EXT_REFERENCE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "External CID reference link"),
    SlotDefinition(110, "GRAPH_ANAPHORA_TARGET", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Pronoun / anaphora referent link"),
    SlotDefinition(111, "GRAPH_COREF_BUNDLE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Co-reference equivalence cluster"),
    SlotDefinition(112, "GRAPH_METADATA_HEADER", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Metadata / provenance envelope"),
    SlotDefinition(113, "GRAPH_SCOPED_CONTEXT", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Lexical / operational scope container"),
    SlotDefinition(114, "GRAPH_CLOSURE_CAPTURE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Captured environment variable closure"),
    SlotDefinition(115, "GRAPH_TYPE_SIGNATURE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Static type constraint descriptor"),
    SlotDefinition(116, "GRAPH_SCHEMA_SPEC", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Imported library schema definition"),
    SlotDefinition(117, "GRAPH_INVOCATION_HEAD", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Function / predicate call operator"),
    SlotDefinition(118, "GRAPH_ARGUMENT_LIST", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Variadic argument sequence head"),
    SlotDefinition(119, "GRAPH_RETURN_VALUE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Evaluation outcome sink"),
    SlotDefinition(120, "GRAPH_EXCEPTION_HANDLE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Error / exceptional exit branch"),
    SlotDefinition(121, "GRAPH_ASSERTION_CLAIM", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Logical assertion proposition head"),
    SlotDefinition(122, "GRAPH_QUERY_TARGET", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Goal / query variable target"),
    SlotDefinition(123, "GRAPH_ENTAILMENT_EDGE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Deductive derivation edge"),
    SlotDefinition(124, "GRAPH_CONTRADICTION_EDGE", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Mutual exclusivity constraint edge"),
    SlotDefinition(125, "GRAPH_PROBABILISTIC_PRIOR", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Prior Bayesian probability attachment"),
    SlotDefinition(126, "GRAPH_VIRTUAL_PAGE_LINK", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Host RAM virtual page-table pointer"),
    SlotDefinition(127, "GRAPH_IMMUTABLE_HASH_LOCK", SlotBand.BAND_1_VALENCIES_TOPOLOGY, "Graph Topology", "Cryptographically locked sub-graph CID"),
]

# ==============================================================================
# BAND 2: Ontological Signatures & Theory of Mind Modalities (128-191)
# ==============================================================================
BAND_2_SLOTS = [
    # Entity Types (128-147)
    SlotDefinition(128, "TYPE_ANIMATE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Biological living being"),
    SlotDefinition(129, "TYPE_HUMAN", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Human person"),
    SlotDefinition(130, "TYPE_INANIMATE_PHYSICAL", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Non-living physical entity"),
    SlotDefinition(131, "TYPE_ABSTRACT_CONCEPT", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Intangible / mathematical / theoretical concept"),
    SlotDefinition(132, "TYPE_PROPOSITION", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Declarative statement with truth value"),
    SlotDefinition(133, "TYPE_EVENT", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Bounded temporal transition / occurrence"),
    SlotDefinition(134, "TYPE_STATE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Static condition or enduring property"),
    SlotDefinition(135, "TYPE_PROCESS", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Continuous ongoing activity"),
    SlotDefinition(136, "TYPE_TEMPORAL_INTERVAL", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Time span / duration / era"),
    SlotDefinition(137, "TYPE_SPATIAL_REGION", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Geometric location or zone"),
    SlotDefinition(138, "TYPE_MEASURE_SCALAR", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Quantitative magnitude with unit"),
    SlotDefinition(139, "TYPE_COLLECTION_SET", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Plural set or aggregated group"),
    SlotDefinition(140, "TYPE_SUBSTANCE_MASS", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Continuous uncounted material / matter"),
    SlotDefinition(141, "TYPE_ARTIFACT", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Manufactured tool or object"),
    SlotDefinition(142, "TYPE_NATURAL_OBJECT", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Non-manufactured physical object (rock, star)"),
    SlotDefinition(143, "TYPE_ORGANIZATION", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Institutional or corporate body"),
    SlotDefinition(144, "TYPE_COMMUNICATION_MSG", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Message, utterance, or document"),
    SlotDefinition(145, "TYPE_ATTRIBUTE_PROPERTY", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Inherent trait, quality, or descriptor"),
    SlotDefinition(146, "TYPE_RELATION_ROLE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Relational association or bridge"),
    SlotDefinition(147, "TYPE_NUMERIC_VALUE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Entity Types", "Pure mathematical number or constant"),

    # Semantic Roles & Capabilities (148-154)
    SlotDefinition(148, "ROLE_AGENT_CAPABLE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Capabilities", "Capable of intentional volition / action"),
    SlotDefinition(149, "ROLE_SENTIENT", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Capabilities", "Capable of subjective perception and feeling"),
    SlotDefinition(150, "ROLE_MOVEABLE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Capabilities", "Capable of physical translation in space"),
    SlotDefinition(151, "ROLE_COMMUNICATOR", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Capabilities", "Capable of transmitting linguistic symbols"),
    SlotDefinition(152, "ROLE_CONSUMABLE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Capabilities", "Can be ingested, depleted, or absorbed"),
    SlotDefinition(153, "ROLE_CONTAINER", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Capabilities", "Can enclose other physical/abstract entities"),
    SlotDefinition(154, "ROLE_INSTRUMENT_USABLE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Capabilities", "Can be utilized by an agent as an instrument"),

    # Theory of Mind & Deception (155-156)
    SlotDefinition(155, "ROLE_DECEPTIVE_PROJECTION", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Theory of Mind", "Deceptive or manipulative communicative intent"),
    SlotDefinition(156, "ROLE_SARCASM_IRONY", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Theory of Mind", "Pragmatic irony or sarcastic reversal"),

    # Modality Overlays (157-158)
    SlotDefinition(157, "MODALITY_LITERAL", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Modality", "Strict literal compositional interpretation"),
    SlotDefinition(158, "MODALITY_FIGURATIVE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Modality", "Metaphorical, allegorical, or poetic interpretation"),

    # Extended Theory of Mind & Semantic Attributes (159-170)
    SlotDefinition(159, "TOM_BELIEF_FIRST_ORDER", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Theory of Mind", "Agent belief about world state (A believes X)"),
    SlotDefinition(160, "TOM_BELIEF_SECOND_ORDER", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Theory of Mind", "Agent belief about other belief (A believes B believes X)"),
    SlotDefinition(161, "TOM_INTENTION", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Theory of Mind", "Agent teleological commitment to an action"),
    SlotDefinition(162, "TOM_DESIRE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Theory of Mind", "Agent appetitive / goal preference"),
    SlotDefinition(163, "TOM_SHARED_ATTENTION", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Theory of Mind", "Intersubjective mutual focus"),
    SlotDefinition(164, "MODALITY_HYPOTHETICAL", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Modality", "Conditional / hypothetical scenario premise"),
    SlotDefinition(165, "MODALITY_COUNTERFACTUAL", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Modality", "Counterfactual world branch (contrary to fact)"),
    SlotDefinition(166, "MODALITY_METAPHORICAL", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Modality", "Cross-domain mapping metaphor"),
    SlotDefinition(167, "ROLE_COGNITIVE_SUBJECT", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Thematic Roles", "Subject of internal cognition or belief"),
    SlotDefinition(168, "ROLE_AFFECTIVE_TARGET", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Thematic Roles", "Object of emotional valence"),
    SlotDefinition(169, "ROLE_VOLITIONAL_SOURCE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Thematic Roles", "Origin of willful command or purpose"),
    SlotDefinition(170, "ROLE_EPISTEMIC_AUTHORITY", SlotBand.BAND_2_ONTOLOGY_MODALITY, "Thematic Roles", "Source of truth / authoritative claim"),

    # WordNet Root Categories (171-191)
    SlotDefinition(171, "WN_ACT_ACTION", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:act - actions, deeds"),
    SlotDefinition(172, "WN_ANIMAL_FAUNA", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:animal - zoological fauna"),
    SlotDefinition(173, "WN_ARTIFACT_OBJECT", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:artifact - manufactured artifacts"),
    SlotDefinition(174, "WN_ATTRIBUTE_PROP", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:attribute - traits and characteristics"),
    SlotDefinition(175, "WN_BODY_PART", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:body - anatomical parts and organs"),
    SlotDefinition(176, "WN_COGNITION_THOUGHT", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:cognition - cognitive concepts and ideas"),
    SlotDefinition(177, "WN_COMMUNICATION_INFO", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:communication - linguistic exchanges"),
    SlotDefinition(178, "WN_EVENT_OCCURRENCE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:event - phenomena and incidents"),
    SlotDefinition(179, "WN_FEELING_EMOTION", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:feeling - affective feelings and states"),
    SlotDefinition(180, "WN_FOOD_NUTRITION", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:food - edible items and nutrient substance"),
    SlotDefinition(181, "WN_GROUP_SOCIAL", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:group - social collectives and groupings"),
    SlotDefinition(182, "WN_LOCATION_PLACE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:location - geographical sites"),
    SlotDefinition(183, "WN_MOTIVE_REASON", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:motive - reasons and motivational causes"),
    SlotDefinition(184, "WN_OBJECT_NATURAL", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:object - natural inanimate physical objects"),
    SlotDefinition(185, "WN_PERSON_HUMAN", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:person - individual humans"),
    SlotDefinition(186, "WN_PHENOMENON_NATURE", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:phenomenon - natural atmospheric/physical events"),
    SlotDefinition(187, "WN_PLANT_FLORA", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:plant - botanical flora and vegetation"),
    SlotDefinition(188, "WN_POSSESSION_ASSET", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:possession - financial assets and holdings"),
    SlotDefinition(189, "WN_PROCESS_SERIES", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:process - dynamic multi-step procedures"),
    SlotDefinition(190, "WN_QUANTITY_NUMBER", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:quantity - amounts and quantitative measures"),
    SlotDefinition(191, "WN_RELATION_LINK", SlotBand.BAND_2_ONTOLOGY_MODALITY, "WordNet Roots", "wn:relation - ontological and conceptual relations"),
]

# ==============================================================================
# BAND 3: Epistemic Bounds, Probability & Logical Metarules (192-255)
# ==============================================================================
BAND_3_SLOTS = [
    # Epistemic Context (192-200)
    SlotDefinition(192, "EPIST_DIRECT_OBSERVATION", SlotBand.BAND_3_EPISTEMIC_METARULES, "Epistemic Context", "Empirically verified direct sensory observation"),
    SlotDefinition(193, "EPIST_DEDUCTIVE_INFERENCE", SlotBand.BAND_3_EPISTEMIC_METARULES, "Epistemic Context", "Sound deductive entailment from known axioms"),
    SlotDefinition(194, "EPIST_INDUCTIVE_GENERAL", SlotBand.BAND_3_EPISTEMIC_METARULES, "Epistemic Context", "Inductive statistical generalization"),
    SlotDefinition(195, "EPIST_ABDUCTIVE_BEST_EXPL", SlotBand.BAND_3_EPISTEMIC_METARULES, "Epistemic Context", "Abductive inference to best explanation"),
    SlotDefinition(196, "EPIST_HEARSAY_TESTIMONY", SlotBand.BAND_3_EPISTEMIC_METARULES, "Epistemic Context", "Indirect hearsay or second-hand testimony"),
    SlotDefinition(197, "EPIST_AXIOMATIC_PREMISE", SlotBand.BAND_3_EPISTEMIC_METARULES, "Epistemic Context", "Given ground truth or assumed premise"),
    SlotDefinition(198, "EPIST_DEONTIC_OBLIGATION", SlotBand.BAND_3_EPISTEMIC_METARULES, "Deontic Modality", "Strict normative duty / requirement (Must)"),
    SlotDefinition(199, "EPIST_DEONTIC_PERMISSION", SlotBand.BAND_3_EPISTEMIC_METARULES, "Deontic Modality", "Normative permission (May)"),
    SlotDefinition(200, "EPIST_DEONTIC_PROHIBITION", SlotBand.BAND_3_EPISTEMIC_METARULES, "Deontic Modality", "Normative prohibition (Must Not)"),

    # Probabilistic Weights (201-207)
    SlotDefinition(201, "EPIST_PROB_CERTAIN", SlotBand.BAND_3_EPISTEMIC_METARULES, "Probability", "P = 1.0 (Deterministic certainty)"),
    SlotDefinition(202, "EPIST_PROB_HIGH", SlotBand.BAND_3_EPISTEMIC_METARULES, "Probability", "P >= 0.8 (Strong probabilistic confidence)"),
    SlotDefinition(203, "EPIST_PROB_MARGINAL", SlotBand.BAND_3_EPISTEMIC_METARULES, "Probability", "P ~ 0.5 (Equi-probable or marginal)"),
    SlotDefinition(204, "EPIST_PROB_DISTRIBUTED", SlotBand.BAND_3_EPISTEMIC_METARULES, "Probability", "Multi-modal probability distribution"),
    SlotDefinition(205, "EPIST_STATISTICAL_EDGE", SlotBand.BAND_3_EPISTEMIC_METARULES, "Probability", "Plausible correlation with statistical edge"),
    SlotDefinition(206, "EPIST_FUZZY_PLAUSIBILITY", SlotBand.BAND_3_EPISTEMIC_METARULES, "Probability", "Possibilistic fuzzy truth evaluation"),
    SlotDefinition(207, "EPIST_DEFAULT_HEURISTIC", SlotBand.BAND_3_EPISTEMIC_METARULES, "Probability", "Non-monotonic default presumption"),

    # s(CASP) & Proof Solver Flags (208-223)
    SlotDefinition(208, "SOLVER_CWA_CLOSED_WORLD", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Closed World Assumption flag for s(CASP)"),
    SlotDefinition(209, "SOLVER_MUC_TARGETED", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Minimal Unsatisfiable Core candidate target"),
    SlotDefinition(210, "SOLVER_PROOF_VALIDATED", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Verified stable model theorem"),
    SlotDefinition(211, "SOLVER_CONTRADICTION_FLAG", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Integrity constraint contradiction identified"),
    SlotDefinition(212, "SOLVER_ABDUCIBLE_PREDICATE", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Abducible open predicate flag"),
    SlotDefinition(213, "SOLVER_COINDUCTION_COLOOP", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Co-inductive greatest fixed point hypothesis"),
    SlotDefinition(214, "SOLVER_EVEN_LOOP_OVER_NEG", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Even loop over negation choice branching"),
    SlotDefinition(215, "SOLVER_GLOBAL_CONSTRAINT", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Global headless integrity constraint"),
    SlotDefinition(216, "SOLVER_DUAL_RULE_APPLIED", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Dual program rule negation verification"),
    SlotDefinition(217, "SOLVER_INCONSISTENCY_CORE", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Active membership in inconsistency core"),
    SlotDefinition(218, "SOLVER_EXPLANATION_REQUIRED", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Proof justification tree requested"),
    SlotDefinition(219, "SOLVER_RE_DENOISE_REQUIRED", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "MUC triggers diffusion re-denoising cycle"),
    SlotDefinition(220, "SOLVER_ENTAILMENT_VERIFIED", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Formally proven entailed conclusion"),
    SlotDefinition(221, "SOLVER_COUNTEREXAMPLE_FOUND", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Counterexample model constructed"),
    SlotDefinition(222, "SOLVER_STABLE_MODEL_MEMBER", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "Atom belongs to active answer set"),
    SlotDefinition(223, "SOLVER_PARTIAL_INTERPRETATION", SlotBand.BAND_3_EPISTEMIC_METARULES, "Solver Flags", "3-valued partial model assignment"),

    # Extension Registers & AST / Domain Primitives (224-255)
    SlotDefinition(224, "EXT_AST_FUNCTION_DEF", SlotBand.BAND_3_EPISTEMIC_METARULES, "AST Primitives", "AST: Function or method definition"),
    SlotDefinition(225, "EXT_AST_RETURN", SlotBand.BAND_3_EPISTEMIC_METARULES, "AST Primitives", "AST: Return expression"),
    SlotDefinition(226, "EXT_AST_CALL", SlotBand.BAND_3_EPISTEMIC_METARULES, "AST Primitives", "AST: Function or operator call"),
    SlotDefinition(227, "EXT_AST_VARIABLE_BINDING", SlotBand.BAND_3_EPISTEMIC_METARULES, "AST Primitives", "AST: Variable assignment / binding"),
    SlotDefinition(228, "EXT_AST_SCOPE_ENTER", SlotBand.BAND_3_EPISTEMIC_METARULES, "AST Primitives", "AST: Lexical block entry"),
    SlotDefinition(229, "EXT_AST_SCOPE_EXIT", SlotBand.BAND_3_EPISTEMIC_METARULES, "AST Primitives", "AST: Lexical block exit"),
    SlotDefinition(230, "EXT_AST_CONTROL_LOOP", SlotBand.BAND_3_EPISTEMIC_METARULES, "AST Primitives", "AST: Iteration loop (while/for)"),
    SlotDefinition(231, "EXT_AST_TYPE_CHECK", SlotBand.BAND_3_EPISTEMIC_METARULES, "AST Primitives", "AST: Type cast / assertion check"),
    SlotDefinition(232, "EXT_DOM_TEMPORAL_ORDER", SlotBand.BAND_3_EPISTEMIC_METARULES, "Domain Primitives", "Temporal ordering constraint (Allen interval)"),
    SlotDefinition(233, "EXT_DOM_SPATIAL_ADJACENT", SlotBand.BAND_3_EPISTEMIC_METARULES, "Domain Primitives", "Spatial adjacency topology"),
    SlotDefinition(234, "EXT_DOM_CAUSAL_MECHANISM", SlotBand.BAND_3_EPISTEMIC_METARULES, "Domain Primitives", "Direct causal dependency"),
    SlotDefinition(235, "EXT_DOM_MEREOLOGICAL_PART", SlotBand.BAND_3_EPISTEMIC_METARULES, "Domain Primitives", "Mereological whole-part relationship"),
    SlotDefinition(236, "EXT_DOM_AGENT_INTENT", SlotBand.BAND_3_EPISTEMIC_METARULES, "Domain Primitives", "Strategic intentional objective"),
    SlotDefinition(237, "EXT_DOM_RESOURCE_BOUND", SlotBand.BAND_3_EPISTEMIC_METARULES, "Domain Primitives", "Resource constraint (memory, time, energy)"),
    SlotDefinition(238, "EXT_DOM_STATE_TRANSITION", SlotBand.BAND_3_EPISTEMIC_METARULES, "Domain Primitives", "Automata state machine transition"),
    SlotDefinition(239, "EXT_DOM_EVIDENCE_WEIGHT", SlotBand.BAND_3_EPISTEMIC_METARULES, "Domain Primitives", "Evidential support weight"),
    SlotDefinition(240, "EXT_REG_240", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 240"),
    SlotDefinition(241, "EXT_REG_241", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 241"),
    SlotDefinition(242, "EXT_REG_242", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 242"),
    SlotDefinition(243, "EXT_REG_243", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 243"),
    SlotDefinition(244, "EXT_REG_244", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 244"),
    SlotDefinition(245, "EXT_REG_245", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 245"),
    SlotDefinition(246, "EXT_REG_246", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 246"),
    SlotDefinition(247, "EXT_REG_247", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 247"),
    SlotDefinition(248, "EXT_REG_248", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 248"),
    SlotDefinition(249, "EXT_REG_249", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 249"),
    SlotDefinition(250, "EXT_REG_250", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 250"),
    SlotDefinition(251, "EXT_REG_251", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 251"),
    SlotDefinition(252, "EXT_REG_252", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 252"),
    SlotDefinition(253, "EXT_REG_253", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 253"),
    SlotDefinition(254, "EXT_REG_254", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 254"),
    SlotDefinition(255, "EXT_REG_255", SlotBand.BAND_3_EPISTEMIC_METARULES, "Extension Registers", "General purpose extension slot 255"),
]

# Combined canonical list of all 256 slots
CANONICAL_SLOTS: List[SlotDefinition] = BAND_0_SLOTS + BAND_1_SLOTS + BAND_2_SLOTS + BAND_3_SLOTS

assert len(CANONICAL_SLOTS) == 256, f"Expected 256 slots, got {len(CANONICAL_SLOTS)}"
for i, slot in enumerate(CANONICAL_SLOTS):
    assert slot.index == i, f"Slot {slot.name} index mismatch: expected {i}, got {slot.index}"

SLOT_NAME_TO_INDEX: Dict[str, int] = {slot.name: slot.index for slot in CANONICAL_SLOTS}
SLOT_INDEX_TO_NAME: Dict[int, str] = {slot.index: slot.name for slot in CANONICAL_SLOTS}


def get_slot_by_name(name: str) -> Optional[SlotDefinition]:
    """Retrieves slot definition by exact name."""
    idx = SLOT_NAME_TO_INDEX.get(name)
    if idx is not None:
        return CANONICAL_SLOTS[idx]
    return None


def get_slot_by_index(index: int) -> SlotDefinition:
    """Retrieves slot definition by index [0, 255]."""
    if not (0 <= index < 256):
        raise IndexError(f"Slot index {index} out of range [0, 255]")
    return CANONICAL_SLOTS[index]


def get_slot_names() -> List[str]:
    """Returns an ordered list of all 256 canonical slot names."""
    return [slot.name for slot in CANONICAL_SLOTS]
