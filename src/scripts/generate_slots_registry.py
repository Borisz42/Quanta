import json
from pathlib import Path
import re
import sys

slots_py_text = Path("src/core/slots.py").read_text(encoding="utf-8")

def extract_band_block(code: str, band_name: str) -> list:
    pattern = rf"(# =+\s*\n# {band_name}\s*\n# =+\s*\n{band_name} = \[.*?^\s*\])"
    m = re.search(pattern, code, re.MULTILINE | re.DOTALL)
    if not m:
        raise ValueError(f"Could not find {band_name} in slots.py")
    return [m.group(1), ""]

cn_slots = json.load(open('data/conceptnet_slots.json', 'r', encoding='utf-8'))
b3_cn = [s for s in cn_slots if s['band'] == 3]
b4_cn = [s for s in cn_slots if s['band'] == 4]

lines = []
lines.append('"""Canonical 1024-dimension slot registry for QUANTA.')
lines.append('')
lines.append('Partitions the 1024 dimensions into eight isolated 128-slot bands:')
lines.append('- Band 0 (000–127): Universal NSM Primes, Classical Kinematics & Continuous Physics')
lines.append('- Band 1 (128–255): Structural Valencies, Grammatical Tense/Aspect, Code AST & Concurrency Topologies')
lines.append('- Band 2 (256–383): Logic Quantifiers (∀, ∃, ∃!), Variable Binding Registers & Query Unification Heads')
lines.append('- Band 3 (384–511): 128 Data-Driven ConceptNet Taxonomies, Scientific Domains & Entity Types')
lines.append('- Band 4 (512–639): 128 Data-Driven ConceptNet Actions, Functional Affordances & Cyber-Physical Properties')
lines.append('- Band 5 (640–767): Theory of Mind, 3-Tier Nested Beliefs, Teleological Goals & Pragmatic Intent')
lines.append('- Band 6 (768–895): Epistemic Proof Solvers, s(CASP) Invariants & Deontic Normative Logic')
lines.append('- Band 7 (896–1023): Spatio-Temporal Mereotopology (Allen, RCC-8) & Pearl Causal Counterfactual DAGs')
lines.append('"""')
lines.append('')
lines.append('from __future__ import annotations')
lines.append('from dataclasses import dataclass')
lines.append('import enum')
lines.append('import json')
lines.append('from pathlib import Path')
lines.append('from typing import Dict, List, Optional, Union')
lines.append('')
lines.append('class BandContract(str, enum.Enum):')
lines.append('    """Polymorphic contract applied to dimensions depending on their Band."""')
lines.append('    EPISTEMIC = "EPISTEMIC"      # Bands 0, 3, 4, 5, 6, 7: Truth & Uncertainty (Belnap FOUR)')
lines.append('    STRUCTURAL = "STRUCTURAL"    # Band 1: Valencies, AST Topology, Concurrency Routing')
lines.append('    REGISTER = "REGISTER"        # Band 2: Formal Logic Quantifiers & Variable Scoping')
lines.append('')
lines.append('')
lines.append('class SlotBand(enum.IntEnum):')
lines.append('    BAND_0_NSM_KINEMATICS = 0')
lines.append('    BAND_1_VALENCIES_TOPOLOGY = 1')
lines.append('    BAND_2_LOGIC_VARIABLES = 2')
lines.append('    BAND_3_ONTOLOGY_STRUCTURES = 3')
lines.append('    BAND_4_AFFORDANCES_OPERATIONS = 4')
lines.append('    BAND_5_TOM_PRAGMATICS = 5')
lines.append('    BAND_6_PROOF_DEONTICS = 6')
lines.append('    BAND_7_SPATIOTEMPORAL_CAUSAL = 7')
lines.append('    # Backward compatibility aliases')
lines.append('    BAND_2_ONTOLOGY_MODALITY = 3')
lines.append('    BAND_3_EPISTEMIC_METARULES = 6')
lines.append('')
lines.append('')
lines.append('BAND_CONTRACTS: Dict[SlotBand, BandContract] = {')
lines.append('    SlotBand.BAND_0_NSM_KINEMATICS: BandContract.EPISTEMIC,')
lines.append('    SlotBand.BAND_1_VALENCIES_TOPOLOGY: BandContract.STRUCTURAL,')
lines.append('    SlotBand.BAND_2_LOGIC_VARIABLES: BandContract.REGISTER,')
lines.append('    SlotBand.BAND_3_ONTOLOGY_STRUCTURES: BandContract.EPISTEMIC,')
lines.append('    SlotBand.BAND_4_AFFORDANCES_OPERATIONS: BandContract.EPISTEMIC,')
lines.append('    SlotBand.BAND_5_TOM_PRAGMATICS: BandContract.EPISTEMIC,')
lines.append('    SlotBand.BAND_6_PROOF_DEONTICS: BandContract.EPISTEMIC,')
lines.append('    SlotBand.BAND_7_SPATIOTEMPORAL_CAUSAL: BandContract.EPISTEMIC,')
lines.append('}')
lines.append('')
lines.append('')
lines.append('@dataclass(frozen=True)')
lines.append('class SlotDefinition:')
lines.append('    index: int')
lines.append('    name: str')
lines.append('    band: SlotBand')
lines.append('    category: str')
lines.append('    description: str')
lines.append('')
lines.append('    @property')
lines.append('    def contract(self) -> BandContract:')
lines.append('        return BAND_CONTRACTS.get(self.band, BandContract.EPISTEMIC)')
lines.append('')
lines.extend(extract_band_block(slots_py_text, 'BAND_0_SLOTS'))
lines.extend(extract_band_block(slots_py_text, 'BAND_1_SLOTS'))
lines.extend(extract_band_block(slots_py_text, 'BAND_2_SLOTS'))

# Format Band 3
res_b3 = ['# ==============================================================================']
res_b3.append('# BAND 3: BAND_3_ONTOLOGY_STRUCTURES (384–511) - ConceptNet Taxonomies & Domains')
res_b3.append('# ==============================================================================')
res_b3.append('BAND_3_SLOTS = [')
for s in b3_cn:
    desc_escaped = s.get('description', '').replace('"', '\\"')
    cat_escaped = s.get('category', 'ConceptNet Taxonomy & Domain').replace('"', '\\"')
    res_b3.append(f'    SlotDefinition({s["index"]}, "{s["name"]}", SlotBand.BAND_3_ONTOLOGY_STRUCTURES, "{cat_escaped}", "{desc_escaped}"),')
res_b3.append(']')
res_b3.append('')
lines.extend(res_b3)

# Format Band 4
res_b4 = ['# ==============================================================================']
res_b4.append('# BAND 4: BAND_4_AFFORDANCES_OPERATIONS (512–639) - ConceptNet Affordances & Actions')
res_b4.append('# ==============================================================================')
res_b4.append('BAND_4_SLOTS = [')
for s in b4_cn:
    desc_escaped = s.get('description', '').replace('"', '\\"')
    cat_escaped = s.get('category', 'ConceptNet Affordance & Function').replace('"', '\\"')
    res_b4.append(f'    SlotDefinition({s["index"]}, "{s["name"]}", SlotBand.BAND_4_AFFORDANCES_OPERATIONS, "{cat_escaped}", "{desc_escaped}"),')
res_b4.append(']')
res_b4.append('')
lines.extend(res_b4)

lines.extend(extract_band_block(slots_py_text, 'BAND_5_SLOTS'))
lines.extend(extract_band_block(slots_py_text, 'BAND_6_SLOTS'))
lines.extend(extract_band_block(slots_py_text, 'BAND_7_SLOTS'))

lines.append('# Combined canonical list of all 1024 slots')
lines.append('CANONICAL_SLOTS: List[SlotDefinition] = (')
lines.append('    BAND_0_SLOTS + BAND_1_SLOTS + BAND_2_SLOTS + BAND_3_SLOTS +')
lines.append('    BAND_4_SLOTS + BAND_5_SLOTS + BAND_6_SLOTS + BAND_7_SLOTS')
lines.append(')')
lines.append('')
lines.append('assert len(CANONICAL_SLOTS) == 1024, f"Expected 1024 slots, got {len(CANONICAL_SLOTS)}"')
lines.append('for i, slot in enumerate(CANONICAL_SLOTS):')
lines.append('    assert slot.index == i, f"Slot {slot.name} index mismatch: expected {i}, got {slot.index}"')
lines.append('')
lines.append('SLOT_NAME_TO_INDEX: Dict[str, int] = {slot.name: slot.index for slot in CANONICAL_SLOTS}')
lines.append('SLOT_INDEX_TO_NAME: Dict[int, str] = {slot.index: slot.name for slot in CANONICAL_SLOTS}')
lines.append('')

slot_by_slug = {}
for s in cn_slots:
    raw_target = s.get("target", "")
    slug = re.sub(r"[^\w\s]", "", raw_target).strip().upper().replace(" ", "_")
    slot_by_slug.setdefault(slug, s["name"])
    name_slug = s["name"].split("_", 2)[-1]
    slot_by_slug.setdefault(name_slug, s["name"])
    slot_by_slug.setdefault(s["name"], s["name"])

def resolve_target_alias(hint: str) -> str:
    h = hint.split("_", 2)[-1] if hint.startswith("CN_Q") else hint
    if h in slot_by_slug:
        return slot_by_slug[h]
    for k, v in slot_by_slug.items():
        if h in k or k in h:
            return v
    return cn_slots[0]["name"]

RAW_LEGACY_ONTOLOGY_ALIASES = [
    # Entity Types
    ("TYPE_ANIMATE", "ANIMAL"),
    ("TYPE_HUMAN", "PERSON"),
    ("TYPE_INANIMATE_PHYSICAL", "TANGIBLE_THING"),
    ("TYPE_NATURAL_OBJECT", "ORGANISM"),
    ("TYPE_ARTIFACT", "DEVICE"),
    ("TYPE_SUBSTANCE_MASS", "MASS"),
    ("TYPE_COLLECTION_SET", "SET"),
    ("TYPE_ABSTRACT_CONCEPT", "LOGIC"),
    ("TYPE_PROPOSITION", "INFORMATION"),
    ("TYPE_EVENT", "EVENT"),
    ("TYPE_STATE", "STATE"),
    ("TYPE_PROCESS", "ACTIVITY"),
    ("TYPE_TEMPORAL_INTERVAL", "TIME"),
    ("TYPE_SPATIAL_REGION", "AREA"),
    ("TYPE_MEASURE_SCALAR", "UNIT"),
    ("TYPE_NUMERIC_VALUE", "MATHEMATICS"),
    ("TYPE_ORGANIZATION", "GROUP"),
    ("TYPE_COMMUNICATION_MSG", "LANGUAGE"),
    ("TYPE_ATTRIBUTE_PROPERTY", "QUALITY"),
    ("TYPE_RELATION_ROLE", "LINE"),
    ("TYPE_ALGORITHM_PROCEDURE", "PROGRAMMING"),
    ("TYPE_LEGAL_CONTRACT", "LEGAL"),
    ("TYPE_BIOLOGICAL_ORGANISM", "ORGANISM"),
    ("TYPE_SOFTWARE_SYSTEM", "COMPUTING"),
    ("TYPE_HARDWARE_DEVICE", "DEVICE"),
    ("TYPE_ASTRONOMICAL_BODY", "ASTRONOMY"),
    ("TYPE_GEOGRAPHICAL_LANDFORM", "LAND"),
    # Capabilities & Roles
    ("ROLE_AGENT_CAPABLE", "ABILITY"),
    ("ROLE_SENTIENT", "SENSE"),
    ("ROLE_MOVEABLE", "MOVE"),
    ("ROLE_COMMUNICATOR", "SPEAK"),
    ("ROLE_CONSUMABLE", "FOOD"),
    ("ROLE_CONTAINER", "BOX"),
    ("ROLE_INSTRUMENT_USABLE", "USE"),
    ("ROLE_VOLITIONAL_SOURCE", "DESIRE"),
    ("ROLE_COGNITIVE_SUBJECT", "MIND"),
    ("ROLE_AFFECTIVE_TARGET", "HAPPY"),
    ("ROLE_EPISTEMIC_AUTHORITY", "KNOWLEDGE"),
    ("ROLE_PATIENT_TARGET", "TAKE"),
    # Modalities
    ("MODALITY_LITERAL", "TRUE"),
    ("MODALITY_FIGURATIVE", "FORMAL"),
    ("MODALITY_HYPOTHETICAL", "LOGIC"),
    ("MODALITY_COUNTERFACTUAL", "LIE"),
    # WordNet Roots
    ("WN_ACT_ACTION", "ACT"),
    ("WN_ANIMAL_FAUNA", "ANIMAL"),
    ("WN_ARTIFACT_OBJECT", "DEVICE"),
    ("WN_ATTRIBUTE_PROP", "QUALITY"),
    ("WN_BODY_PART", "BODY"),
    ("WN_COGNITION_THOUGHT", "MIND"),
    ("WN_COMMUNICATION_INFO", "INFORMATION"),
    ("WN_EVENT_OCCURRENCE", "EVENT"),
    ("WN_FEELING_EMOTION", "HAPPY"),
    ("WN_FOOD_NUTRITION", "FOOD"),
    ("WN_GROUP_SOCIAL", "GROUP"),
    ("WN_LOCATION_PLACE", "PLACE"),
    ("WN_MOTIVE_REASON", "CAUSE"),
    ("WN_OBJECT_NATURAL", "ORGANISM"),
    ("WN_PERSON_HUMAN", "PERSON"),
    ("WN_PHENOMENON_NATURE", "GEOLOGY"),
    ("WN_PLANT_FLORA", "PLANT"),
    ("WN_POSSESSION_ASSET", "MONEY"),
    ("WN_PROCESS_SERIES", "ACTIVITY"),
    ("WN_QUANTITY_NUMBER", "AMOUNT"),
    ("WN_RELATION_LINK", "LINE"),
    # Physical & Cyber Affordances
    ("AFFORD_INCISED_CUTTING", "CUT"),
    ("AFFORD_PERCUSSIVE_IMPACT", "FORCE"),
    ("AFFORD_FLUID_CONTAINMENT", "WATER"),
    ("AFFORD_MECHANICAL_GRIP", "HAND"),
    ("AFFORD_PNEUMATIC_SUCTION", "WATER"),
    ("AFFORD_THERMAL_EXCHANGE", "HOT"),
    ("AFFORD_BALLISTIC_PROPULSION", "FORCE"),
    ("AFFORD_ADHESIVE_BONDING", "MATERIAL"),
    ("AFFORD_LEVERAGE_PRY", "FORCE"),
    ("AFFORD_TORQUE_ROTATION", "MOVE"),
    ("AFFORD_DRILL_PENETRATE", "CUT"),
    ("AFFORD_ABRASIVE_GRINDING", "SURFACE"),
    ("AFFORD_EXTRUSION_FORMING", "MATERIAL"),
    ("AFFORD_FASTENER_BOLT_LATCH", "FIT"),
    ("AFFORD_TENSION_CABLE_PULL", "LINE"),
    ("AFFORD_SPRING_SUSPENSION", "MOVE"),
    ("AFFORD_HYDRAULIC_ACTUATION", "WATER"),
    ("AFFORD_ROLLING_WHEEL_BEARING", "MOVE"),
    ("AFFORD_VALVE_FLOW_CONTROL", "STOP"),
    ("AFFORD_FILTER_SEPARATION", "CUT"),
    ("AFFORD_PUMP_FLUID_DISPLACEMENT", "WATER"),
    ("AFFORD_NOZZLE_ATOMIZATION", "WATER"),
    ("AFFORD_OPTICAL_MAGNIFICATION", "APPEARANCE"),
    ("AFFORD_OPTICAL_REFLECTION", "APPEARANCE"),
    ("AFFORD_ELECTRICAL_SWITCH_CONTACT", "POWER"),
    ("AFFORD_ELECTROMAGNETIC_SOLENOID", "POWER"),
    ("AFFORD_PIEZOELECTRIC_PRECISION", "POWER"),
    ("AFFORD_THERMAL_INSULATION_SHIELD", "HOT"),
    ("AFFORD_VIBRATION_DAMPING", "MOVE"),
    ("AFFORD_FLOATATION_BUOYANT_HULL", "NAUTICAL"),
    ("AFFORD_AERODYNAMIC_AIRFOIL_LIFT", "BIRD"),
    ("AFFORD_PARACHUTE_DRAG_DECEL", "STOP"),
    ("AFFORD_COMPUTE_EXECUTE", "COMPUTING"),
    ("AFFORD_PERSIST_STORAGE", "PROGRAMMING"),
    ("AFFORD_SOCKET_TRANSMIT", "INTERNET"),
    ("AFFORD_SOCKET_RECEIVE", "INTERNET"),
    ("AFFORD_ENCRYPT_CRYPTO", "PROGRAMMING"),
    ("AFFORD_DECRYPT_CRYPTO", "PROGRAMMING"),
    ("AFFORD_SIGN_CRYPTOGRAPHIC", "PROGRAMMING"),
    ("AFFORD_VERIFY_SIGNATURE", "TRUE"),
    ("AFFORD_QUERY_DATABASE", "COMPUTING"),
    ("AFFORD_MUTATE_DATABASE", "COMPUTING"),
    ("AFFORD_AUTHENTICATE_AUTH", "LEGAL"),
    ("AFFORD_AUTHORIZE_RBAC", "LEGAL"),
    ("AFFORD_SERIALIZE_BUFFER", "PROGRAMMING"),
    ("AFFORD_DESERIALIZE_BUFFER", "PROGRAMMING"),
    ("AFFORD_HTTP_REST_REQUEST", "INTERNET"),
    ("AFFORD_GRPC_RPC_INVOKE", "COMPUTING"),
    ("AFFORD_WEBSOCKET_DUPLEX", "INTERNET"),
    ("AFFORD_PUBLISH_EVENT_BUS", "INFORMATION"),
    ("AFFORD_SUBSCRIBE_EVENT_BUS", "INFORMATION"),
    ("AFFORD_CACHE_LOOKUP_KV", "COMPUTING"),
    ("AFFORD_CACHE_INVALIDATE", "COMPUTING"),
    ("AFFORD_SPAWN_CONTAINER", "BOX"),
    ("AFFORD_SCHEDULE_CRON_JOB", "TIME"),
    ("AFFORD_LOG_DIAGNOSTIC", "INFORMATION"),
    ("AFFORD_METRIC_GAUGE_EMIT", "UNIT"),
    ("AFFORD_DISTRIBUTED_LOCK", "STOP"),
    ("AFFORD_MAP_REDUCE_BATCH", "COMPUTING"),
    ("AFFORD_GPU_TENSOR_FORWARD", "COMPUTING"),
    ("AFFORD_VECTOR_INDEX_SEARCH", "COMPUTING"),
    ("AFFORD_FILE_COMPRESSION_ZIP", "PROGRAMMING"),
    ("AFFORD_FILE_DECOMPRESSION", "PROGRAMMING"),
    ("AFFORD_SCHEMA_MIGRATION", "CHANGE"),
    ("AFFORD_INGEST_NUTRIENT", "FOOD"),
    ("AFFORD_CHEMICAL_CATALYSIS", "CHEMISTRY"),
    ("AFFORD_OPTICAL_SENSE", "SENSE"),
    ("AFFORD_ACOUSTIC_SENSE", "SOUND"),
    ("AFFORD_TACTILE_SENSE", "SENSE"),
    ("AFFORD_THERMAL_SENSE", "HOT"),
    ("AFFORD_CHEMICAL_OLFACTION", "SENSE"),
    ("AFFORD_CHEMICAL_GUSTATION", "FOOD"),
    ("AFFORD_PROPRIOCEPTIVE_SENSE", "SENSE"),
    ("AFFORD_VESTIBULAR_EQUILIBRIUM", "CALM"),
    ("AFFORD_ELECTRORECEPTION_SENSE", "POWER"),
    ("AFFORD_MAGNETORECEPTION_SENSE", "POWER"),
    ("AFFORD_ECHOLOCATION_SONAR", "SOUND"),
    ("AFFORD_LIDAR_TIME_OF_FLIGHT", "PHYSICS"),
    ("AFFORD_RADAR_RF_REFLECTION", "PHYSICS"),
    ("AFFORD_METABOLIC_RESPIRATION", "LIFE"),
    ("AFFORD_PHOTOSYNTHESIS_LIGHT", "PLANT"),
    ("AFFORD_DNA_REPLICATION_COPY", "GENETICS"),
    ("AFFORD_RNA_TRANSCRIPTION", "GENETICS"),
    ("AFFORD_PROTEIN_TRANSLATION", "GENETICS"),
    ("AFFORD_IMMUNE_ANTIBODY_BIND", "MEDICINE"),
    ("AFFORD_CELLULAR_MITOSIS_SPLIT", "BIOLOGY"),
    ("AFFORD_CELLULAR_APOPTOSIS", "DEATH"),
    ("AFFORD_MEMBRANE_ION_CHANNEL", "BIOLOGY"),
    ("AFFORD_SYNAPTIC_NEUROTRANSMIT", "MIND"),
    ("AFFORD_HORMONE_ENDOCRINE_SEC", "BODY"),
    ("AFFORD_TOXIN_NEUTRALIZATION", "MEDICINE"),
    ("AFFORD_WOUND_HEALING_CLOT", "MEDICINE"),
    ("AFFORD_CIRCULATORY_PUMP_HEART", "BODY"),
    ("AFFORD_NEURAL_PLASTICITY_LTP", "MIND"),
    ("AFFORD_CIRCADIAN_RHYTHM_TICK", "TIME"),
    ("AFFORD_SYMBIOTIC_MICROBIOME", "BIOLOGY"),
    ("AFFORD_SPEECH_VOCALIZATION", "SPEAK"),
    ("AFFORD_DISPLAY_PIXEL_EMIT", "APPEARANCE"),
    ("AFFORD_HAPTIC_TACTILE_FEEDBACK", "SENSE"),
    ("AFFORD_FERMENTATION_ANAEROBIC", "CHEMISTRY"),
    ("AFFORD_PRECIPITATION_SOLID", "CHEMISTRY"),
    ("AFFORD_COMBUSTION_OXIDATION", "CHEMISTRY"),
    ("AFFORD_ELECTROLYSIS_SPLITTING", "CHEMISTRY"),
    ("AFFORD_POLYMERIZATION_CHAIN", "CHEMISTRY"),
    ("AFFORD_DISTILLATION_FRACTION", "CHEMISTRY"),
    ("AFFORD_CHROMATOGRAPHY_SEPARATE", "CHEMISTRY"),
    ("AFFORD_CRYSTALLIZATION_PURIFY", "CHEMISTRY"),
    ("AFFORD_LYOPHILIZATION_FREEZE_DRY", "CHEMISTRY"),
    ("AFFORD_CENTRIFUGATION_SPIN", "MOVE"),
    ("AFFORD_ULTRASONIC_CLEAN_CAV", "SOUND"),
    ("AFFORD_AUTOCLAVE_STERILIZATION", "MEDICINE"),
    ("AFFORD_UV_GERMICIDAL_IRRAD", "MEDICINE"),
    ("AFFORD_RADIATION_GAMMA_STERIL", "MEDICINE"),
    ("AFFORD_CRYOGENIC_FREEZING", "CHEMISTRY"),
    ("AFFORD_MAGNETIC_LEVITATION", "PHYSICS"),
    ("AFFORD_ION_THRUST_PROPULSION", "PHYSICS"),
    ("AFFORD_SOLAR_SAIL_PRESSURE", "ASTRONOMY"),
    ("AFFORD_RADIO_ANTENNA_EMISSION", "PHYSICS"),
    ("AFFORD_LASER_COHERENT_BEAM", "PHYSICS"),
    ("AFFORD_FIBER_OPTIC_INTERNAL_REF", "PHYSICS"),
    ("AFFORD_BATTERY_CHEMICAL_CHARGE", "POWER"),
    ("AFFORD_FUEL_CELL_CONVERSION", "POWER"),
    ("AFFORD_SUPERCAPACITOR_DISCHARGE", "POWER"),
    ("AFFORD_THERMOELECTRIC_SEEBECK", "POWER"),
    ("AFFORD_SOLAR_PHOTOVOLTAIC", "POWER"),
    ("AFFORD_WIND_TURBINE_HARVEST", "POWER"),
    ("AFFORD_HYDROELECTRIC_HARVEST", "POWER"),
    ("AFFORD_NUCLEAR_FISSION_HEAT", "POWER"),
    # Discrete Math, SI, Metric structures
    ("STRUCT_SET_UNORDERED", "SET"),
    ("STRUCT_SEQUENCE_ORDERED", "LINE"),
    ("STRUCT_GRAPH_NETWORK", "LINE"),
    ("STRUCT_TREE_HIERARCHY", "TREE"),
    ("STRUCT_DIRECTED_ACYCLIC_DAG", "LINE"),
    ("STRUCT_LATTICE_ALGEBRA", "MATHEMATICS"),
    ("STRUCT_MONOID_SEMIGROUP", "MATHEMATICS"),
    ("STRUCT_GROUP_ALGEBRA", "MATHEMATICS"),
    ("STRUCT_RING_FIELD", "MATHEMATICS"),
    ("STRUCT_VECTOR_SPACE", "MATHEMATICS"),
    ("STRUCT_MATRIX_TENSOR", "MATHEMATICS"),
    ("STRUCT_HILBERT_SPACE", "MATHEMATICS"),
    ("STRUCT_BANACH_SPACE", "MATHEMATICS"),
    ("STRUCT_TOPOLOGICAL_MANIFOLD", "GEOMETRY"),
    ("STRUCT_FIBER_BUNDLE", "GEOMETRY"),
    ("STRUCT_RIEMANNIAN_METRIC", "GEOMETRY"),
    ("STRUCT_CATEGORY_THEORY", "MATHEMATICS"),
    ("STRUCT_FUNCTOR_MAP", "MATHEMATICS"),
    ("STRUCT_NATURAL_TRANSFORMATION", "MATHEMATICS"),
    ("STRUCT_ADJUNCTION_MONAD", "MATHEMATICS"),
    ("STRUCT_QUOTIENT_STRUCTURE", "MATHEMATICS"),
    ("STRUCT_DIRECT_PRODUCT", "MATHEMATICS"),
    ("STRUCT_COPRODUCT_DISJOINT_SUM", "MATHEMATICS"),
    ("STRUCT_HOMOMORPHISM_MAP", "MATHEMATICS"),
    ("STRUCT_ISOMORPHISM_BIJECTION", "MATHEMATICS"),
    ("STRUCT_AUTOMORPHISM_SYMMETRY", "MATHEMATICS"),
    ("STRUCT_PROBABILITY_MEASURE", "MATHEMATICS"),
    ("STRUCT_SIGMA_ALGEBRA", "MATHEMATICS"),
    ("STRUCT_RANDOM_VARIABLE", "MATHEMATICS"),
    ("STRUCT_MARKOV_CHAIN", "MATHEMATICS"),
    ("STRUCT_MARTINGALE_PROCESS", "MATHEMATICS"),
    ("STRUCT_STOCHASTIC_DIFFUSION", "MATHEMATICS"),
    ("SI_DIM_LENGTH_L", "UNIT"),
    ("SI_DIM_MASS_M", "MASS"),
    ("SI_DIM_TIME_T", "TIME"),
    ("SI_DIM_ELECTRIC_CURRENT_I", "POWER"),
    ("SI_DIM_TEMPERATURE_THETA", "HOT"),
    ("SI_DIM_SUBSTANCE_AMOUNT_N", "AMOUNT"),
    ("SI_DIM_LUMINOUS_INTENSITY_J", "QUALITY"),
    ("METRIC_FREQUENCY_HERTZ", "UNIT"),
    ("METRIC_FORCE_NEWTON", "FORCE"),
    ("METRIC_PRESSURE_PASCAL", "UNIT"),
    ("METRIC_ENERGY_JOULE", "POWER"),
    ("METRIC_POWER_WATT", "POWER"),
    ("METRIC_ELECTRIC_CHARGE_COULOMB", "POWER"),
    ("METRIC_VOLTAGE_VOLT", "POWER"),
    ("METRIC_CAPACITANCE_FARAD", "POWER"),
    ("METRIC_RESISTANCE_OHM", "POWER"),
    ("METRIC_CONDUCTANCE_SIEMENS", "POWER"),
    ("METRIC_MAGNETIC_FLUX_WEBER", "POWER"),
    ("METRIC_MAGNETIC_FIELD_TESLA", "POWER"),
    ("METRIC_INDUCTANCE_HENRY", "POWER"),
    ("METRIC_LUMINOUS_FLUX_LUMEN", "UNIT"),
    ("METRIC_ILLUMINANCE_LUX", "UNIT"),
    ("METRIC_RADIOACTIVITY_BECQUEREL", "UNIT"),
    ("METRIC_RADIATION_DOSE_GRAY", "UNIT"),
    ("METRIC_DOSE_EQUIVALENT_SIEVERT", "UNIT"),
    ("METRIC_CATALYTIC_ACTIVITY_KATAL", "UNIT"),
    ("METRIC_CURRENCY_VALUE_FIAT", "MONEY"),
    ("METRIC_INFORMATION_ENTROPY_BIT", "INFORMATION"),
    ("METRIC_INFORMATION_NAT", "INFORMATION"),
    ("METRIC_COMPUTE_FLOP_COUNT", "COMPUTING"),
    ("METRIC_COMPUTE_MEMORY_BYTE", "COMPUTING"),
    ("METRIC_BANDWIDTH_BIT_PER_SEC", "COMPUTING"),
]

lines.append('# Semantic Bridge Alias Layer & Backward Compatibility')
lines.append('LEGACY_ONTOLOGY_ALIASES: Dict[str, str] = {')
for alias_k, target_hint in RAW_LEGACY_ONTOLOGY_ALIASES:
    resolved_name = resolve_target_alias(target_hint)
    lines.append(f'    "{alias_k}": "{resolved_name}",')
lines.append('}')
lines.append('')
lines.append('SLOT_ALIASES: Dict[str, str] = {')
lines.append('    "TOM_BELIEF_FIRST_ORDER": "TOM_FIRST_ORDER_BELIEF",')
lines.append('    "TOM_BELIEF_SECOND_ORDER": "TOM_SECOND_ORDER_BELIEF",')
lines.append('    "TOM_SHARED_ATTENTION": "TOM_JOINT_ATTENTION_FOCUS",')
lines.append('    "TOM_INTENTION": "PLAN_INTENDED_ACTION_STEP",')
lines.append('    "TOM_DESIRE": "DRIVE_CURIOSITY_EPISTEMIC",')
lines.append('    "ROLE_DECEPTIVE_PROJECTION": "INTENT_DECEPTIVE_PROJECTION",')
lines.append('    "ROLE_SARCASM_IRONY": "INTENT_IRONY_SARCASM",')
lines.append('    **LEGACY_ONTOLOGY_ALIASES,')
lines.append('}')
lines.append('')
lines.append('for alias_k, target_v in list(SLOT_ALIASES.items()):')
lines.append('    if target_v in SLOT_NAME_TO_INDEX:')
lines.append('        SLOT_NAME_TO_INDEX[alias_k] = SLOT_NAME_TO_INDEX[target_v]')
lines.append('')
lines.append('')
lines.append('def get_slot_by_name(name: str) -> Optional[SlotDefinition]:')
lines.append('    """Retrieves slot definition by exact name (supporting legacy aliases)."""')
lines.append('    idx = SLOT_NAME_TO_INDEX.get(name)')
lines.append('    if idx is not None:')
lines.append('        return CANONICAL_SLOTS[idx]')
lines.append('    return None')
lines.append('')
lines.append('')
lines.append('def get_slot_by_index(index: int) -> SlotDefinition:')
lines.append('    """Retrieves slot definition by index [0, 1023]."""')
lines.append('    if not (0 <= index < 1024):')
lines.append('        raise IndexError(f"Slot index {index} out of range [0, 1023]")')
lines.append('    return CANONICAL_SLOTS[index]')
lines.append('')
lines.append('')
lines.append('def get_slot_names() -> List[str]:')
lines.append('    """Returns an ordered list of all 1024 canonical slot names."""')
lines.append('    return [slot.name for slot in CANONICAL_SLOTS]')
lines.append('')
lines.append('')
lines.append('# Expose all 1024 slot definitions as module-level immutable integer constants')
lines.append('for _slot in CANONICAL_SLOTS:')
lines.append('    globals()[_slot.name] = _slot.index')
lines.append('')
lines.append('# Expose legacy aliases at module level')
lines.append('for alias_k, target_v in SLOT_ALIASES.items():')
lines.append('    if target_v in SLOT_NAME_TO_INDEX:')
lines.append('        globals()[alias_k] = SLOT_NAME_TO_INDEX[target_v]')
lines.append('')
lines.append('')
lines.append('def get_band_contract(band: Union[int, SlotBand]) -> BandContract:')
lines.append('    """Returns the polymorphic contract for the specified band."""')
lines.append('    if isinstance(band, int) and not isinstance(band, SlotBand):')
lines.append('        band = SlotBand(band)')
lines.append('    return BAND_CONTRACTS.get(band, BandContract.EPISTEMIC)')
lines.append('')
lines.append('')
lines.append('def get_slot_contract(slot: Union[int, str, SlotDefinition]) -> BandContract:')
lines.append('    """Returns the polymorphic contract for the specified slot."""')
lines.append('    if isinstance(slot, SlotDefinition):')
lines.append('        return slot.contract')
lines.append('    elif isinstance(slot, int):')
lines.append('        s_def = get_slot_by_index(slot)')
lines.append('        return s_def.contract if s_def else BandContract.EPISTEMIC')
lines.append('    elif isinstance(slot, str):')
lines.append('        s_def = get_slot_by_name(slot)')
lines.append('        return s_def.contract if s_def else BandContract.EPISTEMIC')
lines.append('    return BandContract.EPISTEMIC')
lines.append('')
lines.append('')
lines.append('def export_canonical_slots_layout(output_path: Union[str, Path] = "output/canonical_slots_layout.json") -> Path:')
lines.append('    """Exports the 1024 canonical slot definitions to a JSON file."""')
lines.append('    p = Path(output_path)')
lines.append('    p.parent.mkdir(parents=True, exist_ok=True)')
lines.append('    slots_layout = [')
lines.append('        {')
lines.append('            "index": s.index,')
lines.append('            "name": s.name,')
lines.append('            "band_id": int(s.band),')
lines.append('            "band_name": s.band.name,')
lines.append('            "category": s.category,')
lines.append('            "description": s.description,')
lines.append('        }')
lines.append('        for s in CANONICAL_SLOTS')
lines.append('    ]')
lines.append('    with open(p, "w", encoding="utf-8") as f:')
lines.append('        json.dump(slots_layout, f, indent=2)')
lines.append('    return p')
lines.append('')
lines.append('')
lines.append('__all__ = [')
lines.append('    "BandContract",')
lines.append('    "BAND_CONTRACTS",')
lines.append('    "get_band_contract",')
lines.append('    "get_slot_contract",')
lines.append('    "SlotBand",')
lines.append('    "SlotDefinition",')
lines.append('    "BAND_0_SLOTS",')
lines.append('    "BAND_1_SLOTS",')
lines.append('    "BAND_2_SLOTS",')
lines.append('    "BAND_3_SLOTS",')
lines.append('    "BAND_4_SLOTS",')
lines.append('    "BAND_5_SLOTS",')
lines.append('    "BAND_6_SLOTS",')
lines.append('    "BAND_7_SLOTS",')
lines.append('    "CANONICAL_SLOTS",')
lines.append('    "SLOT_NAME_TO_INDEX",')
lines.append('    "SLOT_INDEX_TO_NAME",')
lines.append('    "SLOT_ALIASES",')
lines.append('    "LEGACY_ONTOLOGY_ALIASES",')
lines.append('    "get_slot_by_name",')
lines.append('    "get_slot_by_index",')
lines.append('    "get_slot_names",')
lines.append('    "export_canonical_slots_layout",')
lines.append('] + [slot.name for slot in CANONICAL_SLOTS] + list(SLOT_ALIASES.keys())')
lines.append('')

target = Path("src/core/slots.py")
with open(target, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Successfully generated {target.resolve()} with 1024 slots!")
