"""Validation extraction corpus generator spanning FOLIO, ProofWriter, bAbI, CLUTRR, and Code ASTs."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.slots import CANONICAL_SLOTS, SLOT_NAME_TO_INDEX
from core.types import QuantaVector
from profiler.candidate_pool import build_candidate_pool, CandidateDimension


@dataclass
class GeneratedCorpus:
    canonical_matrix: np.ndarray  # (N, 1024) in {0, 1, 2, 3}
    candidate_matrix: np.ndarray  # (N, K) in {0, 1, 2, 3}
    labels: np.ndarray            # (N,) categorical label
    propositions: List[str]       # Human-readable proposition descriptions


class ValidationCorpusGenerator:
    """Generates synthetic and mapped multi-domain validation corpora for Information Bottleneck testing."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.candidates = build_candidate_pool()
        self.cand_name_to_idx = {c.name: c.id for c in self.candidates}

    def generate_corpus(self, num_samples: int = 5000) -> GeneratedCorpus:
        """Generates a balanced proposition corpus covering 5 distinct semantic regimes:
        1. First-Order Logic & FOLIO (Quantifiers, syllogisms, implications)
        2. ProofWriter (Deductive chains, rule premises, negation)
        3. bAbI (Spatial movement, agents, object transfers, temporal sequences)
        4. CLUTRR (Kinship, social relations, transitivity)
        5. Code AST (Functions, loops, conditionals, recursion, bindings)
        """
        samples_per_domain = num_samples // 5
        remainder = num_samples % 5

        canonical_rows: List[np.ndarray] = []
        candidate_rows: List[np.ndarray] = []
        labels: List[int] = []
        descriptions: List[str] = []

        domains = [
            ("FOLIO", 0, samples_per_domain + (1 if remainder > 0 else 0)),
            ("ProofWriter", 1, samples_per_domain + (1 if remainder > 1 else 0)),
            ("bAbI", 2, samples_per_domain + (1 if remainder > 2 else 0)),
            ("CLUTRR", 3, samples_per_domain + (1 if remainder > 3 else 0)),
            ("CodeAST", 4, samples_per_domain + (1 if remainder > 4 else 0)),
        ]

        for domain_name, domain_label, count in domains:
            for i in range(count):
                canon_vec = np.zeros(len(CANONICAL_SLOTS), dtype=np.uint8)
                cand_vec = np.zeros(len(self.candidates), dtype=np.uint8)

                desc = self._populate_domain_sample(
                    domain_name=domain_name,
                    canon_vec=canon_vec,
                    cand_vec=cand_vec,
                    sample_idx=i,
                )

                canonical_rows.append(canon_vec)
                candidate_rows.append(cand_vec)
                labels.append(domain_label)
                descriptions.append(desc)

        canonical_mat = np.stack(canonical_rows, axis=0)
        candidate_mat = np.stack(candidate_rows, axis=0)
        labels_arr = np.array(labels, dtype=np.int32)

        return GeneratedCorpus(
            canonical_matrix=canonical_mat,
            candidate_matrix=candidate_mat,
            labels=labels_arr,
            propositions=descriptions,
        )

    def _set_slot(self, canon_vec: np.ndarray, cand_vec: np.ndarray, slot_name: str, value: int):
        """Helper to activate both canonical slot and corresponding candidate slot if present."""
        if slot_name in SLOT_NAME_TO_INDEX:
            canon_vec[SLOT_NAME_TO_INDEX[slot_name]] = value & 0x03
        if slot_name in self.cand_name_to_idx:
            cand_vec[self.cand_name_to_idx[slot_name]] = value & 0x03

    def _populate_domain_sample(
        self,
        domain_name: str,
        canon_vec: np.ndarray,
        cand_vec: np.ndarray,
        sample_idx: int,
    ) -> str:
        """Populates slots across all 8 bands based on rich semantic templates."""
        # 1. Deterministic sample-specific semantic seed modifiers to guarantee uniqueness across >= 10,000 samples
        # Pick 2-3 background NSM / property slots based on sample index and RNG
        nsm_pool = [
            "NSM_I", "NSM_YOU", "NSM_SOMEONE", "NSM_SOMETHING", "NSM_PEOPLE", "NSM_BODY",
            "NSM_THIS", "NSM_SAME", "NSM_OTHER", "NSM_ONE", "NSM_TWO", "NSM_MUCH",
            "NSM_LITTLE", "NSM_SOME", "NSM_ALL", "NSM_MORE", "NSM_FEW", "NSM_PART",
            "NSM_GOOD", "NSM_BAD", "NSM_BIG", "NSM_SMALL", "NSM_VERY", "NSM_TRUE",
            "NSM_NOW", "NSM_BEFORE", "NSM_AFTER", "NSM_HERE", "NSM_ABOVE", "NSM_BELOW",
        ]
        chosen_nsm = self.rng.choice(nsm_pool, size=3, replace=False)
        for s in chosen_nsm:
            self._set_slot(canon_vec, cand_vec, s, int(self.rng.choice([1, 2])))

        # Pick a variable register for formal grounding
        var_reg = f"VAR_SLOT_X{sample_idx % 8}"
        self._set_slot(canon_vec, cand_vec, var_reg, 1)

        # 2. Domain-specific construction spanning the 8 bands
        if domain_name == "FOLIO":
            # First Order Logic & Sequent Proofs (Bands 0, 1, 2, 3, 6)
            quant = self.rng.choice(["FORALL", "EXISTS", "EXISTS_ONE", "MOST", "FEW", "NONE"])
            is_negated = bool(self.rng.random() < 0.3)
            proof_rule = self.rng.choice(["SEQ_MODUS_PONENS_STEP", "SEQ_RESOLUTION_STEP", "SEQ_STRICT_ENTAILMENT_TURN", "SOLVER_PROOF_VALIDATED"])
            e_type = self.rng.choice(["TYPE_HUMAN", "TYPE_ANIMATE", "TYPE_ABSTRACT_CONCEPT", "TYPE_ARTIFACT", "TYPE_ORGANIZATION", "TYPE_SUBSTANCE_MASS"])
            val_role = self.rng.choice(["VAL_X1_AGENT", "VAL_X2_PATIENT", "VAL_X3_DESTINATION", "VAL_X5_INSTRUMENT", "VAL_BENEFICIARY_SLOT"])
            epist_src = self.rng.choice(["EPIST_DEDUCTIVE_INFERENCE", "EPIST_AXIOMATIC_PREMISE", "EPIST_INDUCTIVE_GENERAL", "EPIST_ABDUCTIVE_BEST_EXPL"])

            self._set_slot(canon_vec, cand_vec, "TYPE_PROPOSITION", 1)
            self._set_slot(canon_vec, cand_vec, "GRAPH_ASSERTION_CLAIM", 1)
            self._set_slot(canon_vec, cand_vec, epist_src, 1)
            self._set_slot(canon_vec, cand_vec, "LJB_GANAI_IF_THEN", 1)
            self._set_slot(canon_vec, cand_vec, proof_rule, 1)
            self._set_slot(canon_vec, cand_vec, e_type, 1)
            self._set_slot(canon_vec, cand_vec, val_role, 1)

            if quant == "FORALL":
                self._set_slot(canon_vec, cand_vec, "QUANT_UNIVERSAL_FORALL", 1)
                self._set_slot(canon_vec, cand_vec, "LJB_RO_ALL_QUANT", 1)
            elif quant == "EXISTS":
                self._set_slot(canon_vec, cand_vec, "QUANT_EXISTENTIAL_EXISTS", 1)
                self._set_slot(canon_vec, cand_vec, "LJB_SUO_AT_LEAST_ONE", 1)
            elif quant == "EXISTS_ONE":
                self._set_slot(canon_vec, cand_vec, "QUANT_UNIQUENESS_EXISTS_ONE", 1)
            elif quant == "MOST":
                self._set_slot(canon_vec, cand_vec, "QUANT_MAJORITY_MOST", 1)
            elif quant == "FEW":
                self._set_slot(canon_vec, cand_vec, "QUANT_PAUCAL_FEW", 1)
            elif quant == "NONE":
                self._set_slot(canon_vec, cand_vec, "LJB_NO_NONE_QUANT", 1)

            if is_negated:
                self._set_slot(canon_vec, cand_vec, "LJB_NA_NEGATION", 2)
            else:
                self._set_slot(canon_vec, cand_vec, "NSM_TRUE", 1)

            return f"FOLIO #{sample_idx}: {quant} {e_type} ({val_role}) via {proof_rule}."

        elif domain_name == "ProofWriter":
            # Multi-step Deductive Chains & Non-monotonic s(CASP) Proofs (Bands 1, 2, 3, 6, 7)
            self._set_slot(canon_vec, cand_vec, "TYPE_STATE", 1)
            self._set_slot(canon_vec, cand_vec, "EPIST_AXIOMATIC_PREMISE", 1)
            self._set_slot(canon_vec, cand_vec, "SOLVER_CWA_CLOSED_WORLD", 1)
            self._set_slot(canon_vec, cand_vec, "LJB_CA_PRESENT_TENSE", 1)

            attr_slot = self.rng.choice(["NSM_GOOD", "NSM_BIG", "NSM_SMALL", "TYPE_ATTRIBUTE_PROPERTY", "WN_ATTRIBUTE_PROP", "PROP_HARDNESS_INDENTATION", "PROP_CONDUCTIVITY_THERMAL"])
            e_type = self.rng.choice(["TYPE_ANIMATE", "TYPE_NATURAL_OBJECT", "TYPE_SUBSTANCE_MASS", "TYPE_ARTIFACT"])
            solver_flag = self.rng.choice(["SOLVER_STABLE_MODEL_MEMBER", "SOLVER_DUAL_RULE_VERIFIED", "SOLVER_ABDUCIBLE_HYPOTHESIS", "SOLVER_COINDUCTION_LOOP_ACTIVE"])
            temporal_link = self.rng.choice(["TEMP_ALLEN_BEFORE", "TEMP_ALLEN_MEETS", "TEMP_ALLEN_DURING", "TEMP_SYNCHRONOUS_COINCIDE"])

            self._set_slot(canon_vec, cand_vec, attr_slot, 1)
            self._set_slot(canon_vec, cand_vec, e_type, 1)
            self._set_slot(canon_vec, cand_vec, solver_flag, 1)
            self._set_slot(canon_vec, cand_vec, temporal_link, 1)

            has_contradiction = bool(self.rng.random() < 0.15)
            if has_contradiction:
                self._set_slot(canon_vec, cand_vec, "SOLVER_CONTRADICTION_FLAG", 2)
                self._set_slot(canon_vec, cand_vec, "SOLVER_MUC_TARGETED", 1)
                self._set_slot(canon_vec, cand_vec, "SOLVER_RE_DENOISE_REQUIRED", 1)
            else:
                self._set_slot(canon_vec, cand_vec, "EPIST_PROB_CERTAIN", 1)

            return f"ProofWriter #{sample_idx}: Fact {e_type} has {attr_slot} ({solver_flag}, Contradiction={has_contradiction})."

        elif domain_name == "bAbI":
            # Physical Robotics, Kinematics, Spatial Mereotopology & Tool Affordances (Bands 0, 1, 3, 4, 7)
            action = self.rng.choice(["MOVE", "TOUCH", "SAY", "GIVE", "GRIP", "CUT", "HEAT", "PUMP"])
            rcc_spatial = self.rng.choice(["SPATIAL_RCC_DISCONNECTED", "SPATIAL_RCC_EXT_CONNECTED", "SPATIAL_RCC_PARTIAL_OVERLAP", "SPATIAL_RCC_TANGENTIAL_PART", "SPATIAL_RCC_NON_TANGENTIAL_P"])
            affordance = self.rng.choice(["AFFORD_MECHANICAL_GRIP", "AFFORD_INCISED_CUTTING", "AFFORD_FLUID_CONTAINMENT", "AFFORD_THERMAL_EXCHANGE", "AFFORD_PUMP_FLUID_DISPLACEMENT", "AFFORD_BALLISTIC_PROPULSION"])
            phys_prop = self.rng.choice(["PHYS_ACCELERATION_LINEAR", "PHYS_FORCE_IMPULSE_J", "STATE_SOLID_RIGID", "STATE_LIQUID_NEWTONIAN", "STATE_GAS_COMPRESSIBLE"])

            self._set_slot(canon_vec, cand_vec, "TYPE_EVENT", 1)
            self._set_slot(canon_vec, cand_vec, "LJB_PU_PAST_TENSE", 1)
            self._set_slot(canon_vec, cand_vec, "TYPE_HUMAN", 1)
            self._set_slot(canon_vec, cand_vec, "ROLE_AGENT_CAPABLE", 1)
            self._set_slot(canon_vec, cand_vec, "VAL_X1_AGENT", 1)
            self._set_slot(canon_vec, cand_vec, "EPIST_DIRECT_OBSERVATION", 1)
            self._set_slot(canon_vec, cand_vec, rcc_spatial, 1)
            self._set_slot(canon_vec, cand_vec, affordance, 1)
            self._set_slot(canon_vec, cand_vec, phys_prop, 1)

            if action == "MOVE":
                self._set_slot(canon_vec, cand_vec, "NSM_MOVE", 1)
                self._set_slot(canon_vec, cand_vec, "VAL_X3_DESTINATION", 1)
                self._set_slot(canon_vec, cand_vec, "VAL_X4_SOURCE", 1)
                self._set_slot(canon_vec, cand_vec, "TYPE_SPATIAL_REGION", 1)
            elif action == "TOUCH":
                self._set_slot(canon_vec, cand_vec, "NSM_TOUCH", 1)
                self._set_slot(canon_vec, cand_vec, "VAL_X2_PATIENT", 1)
                self._set_slot(canon_vec, cand_vec, "TYPE_ARTIFACT", 1)
            elif action == "SAY":
                self._set_slot(canon_vec, cand_vec, "NSM_SAY", 1)
                self._set_slot(canon_vec, cand_vec, "NSM_WORDS", 1)
                self._set_slot(canon_vec, cand_vec, "TYPE_COMMUNICATION_MSG", 1)
            elif action == "GIVE":
                self._set_slot(canon_vec, cand_vec, "NSM_DO", 1)
                self._set_slot(canon_vec, cand_vec, "NSM_HAVE", 1)
                self._set_slot(canon_vec, cand_vec, "VAL_X2_PATIENT", 1)
                self._set_slot(canon_vec, cand_vec, "VAL_X3_DESTINATION", 1)
            elif action in ("GRIP", "CUT", "HEAT", "PUMP"):
                self._set_slot(canon_vec, cand_vec, "NSM_DO", 1)
                self._set_slot(canon_vec, cand_vec, "VAL_X5_INSTRUMENT", 1)

            return f"bAbI #{sample_idx}: Action {action} with affordance {affordance} in {rcc_spatial}."

        elif domain_name == "CLUTRR":
            # Kinship, Multi-Agent Theory of Mind & Pearl Causal DAGs (Bands 0, 1, 3, 5, 7)
            self._set_slot(canon_vec, cand_vec, "TYPE_HUMAN", 1)
            self._set_slot(canon_vec, cand_vec, "TYPE_RELATION_ROLE", 1)
            self._set_slot(canon_vec, cand_vec, "WN_PERSON_HUMAN", 1)
            self._set_slot(canon_vec, cand_vec, "WN_GROUP_SOCIAL", 1)
            self._set_slot(canon_vec, cand_vec, "VAL_X1_AGENT", 1)
            self._set_slot(canon_vec, cand_vec, "VAL_X2_PATIENT", 1)
            self._set_slot(canon_vec, cand_vec, "EPIST_DEDUCTIVE_INFERENCE", 1)
            self._set_slot(canon_vec, cand_vec, "LJB_SOI_RECIPROCAL", 1)

            rel_type = self.rng.choice(["PARENT", "CHILD", "SIBLING", "SPOUSE", "COUSIN", "ANCESTOR"])
            tom_belief = self.rng.choice(["TOM_FIRST_ORDER_BELIEF", "TOM_SECOND_ORDER_BELIEF", "TOM_THIRD_ORDER_BELIEF", "TOM_SHARED_COMMON_GROUND", "TOM_EMPATHIC_STATE_MIRROR"])
            causal_level = self.rng.choice(["CAUSAL_L1_ASSOCIATIONAL", "CAUSAL_L2_INTERVENTIONAL_DO", "CAUSAL_L3_COUNTERFACTUAL", "CAUSAL_MECHANISM_LINK"])
            speech_intent = self.rng.choice(["INTENT_INFORMATIVE_ASSERT", "INTENT_DIRECTIVE_REQUEST", "INTENT_COMMISSIVE_PROMISE", "INTENT_EXPRESSIVE_EMOTION"])

            self._set_slot(canon_vec, cand_vec, tom_belief, 1)
            self._set_slot(canon_vec, cand_vec, causal_level, 1)
            self._set_slot(canon_vec, cand_vec, speech_intent, 1)

            if rel_type in ("PARENT", "CHILD", "ANCESTOR"):
                self._set_slot(canon_vec, cand_vec, "NSM_BORN", 1)
                self._set_slot(canon_vec, cand_vec, "EXT_DOM_MEREOLOGICAL_PART", 1)
            else:
                self._set_slot(canon_vec, cand_vec, "NSM_SAME", 1)

            return f"CLUTRR #{sample_idx}: Kinship {rel_type} with {tom_belief} and {causal_level}."

        elif domain_name == "CodeAST":
            # AST Graphs, Digital APIs, Math Structures & CTL Model Checking (Bands 1, 2, 3, 4, 7)
            ast_type = self.rng.choice(["FUNC", "LOOP", "BRANCH", "ASSIGN", "RETURN", "TRY_EXCEPT", "ASYNC_AWAIT", "MATCH_CASE"])
            math_struct = self.rng.choice(["STRUCT_GRAPH_NETWORK", "STRUCT_TREE_HIERARCHY", "STRUCT_MATRIX_TENSOR", "STRUCT_SET_UNORDERED", "STRUCT_SEQUENCE_ORDERED"])
            api_afford = self.rng.choice(["AFFORD_COMPUTE_EXECUTE", "AFFORD_PERSIST_STORAGE", "AFFORD_SOCKET_TRANSMIT", "AFFORD_ENCRYPT_CRYPTO", "AFFORD_QUERY_DATABASE", "AFFORD_AUTHENTICATE_AUTH", "AFFORD_SERIALIZE_BUFFER"])
            ctl_prop = self.rng.choice(["CTL_ALL_GLOBALLY_AG", "CTL_ALL_FINALLY_AF", "CTL_EXISTS_GLOBALLY_EG", "MODEL_CHECK_SAFETY_PROPERTY", "MODEL_CHECK_LIVENESS_PROPERTY"])

            self._set_slot(canon_vec, cand_vec, "TYPE_PROCESS", 1)
            self._set_slot(canon_vec, cand_vec, "GRAPH_IS_SUB_EXP", 1)
            self._set_slot(canon_vec, cand_vec, math_struct, 1)
            self._set_slot(canon_vec, cand_vec, api_afford, 1)
            self._set_slot(canon_vec, cand_vec, ctl_prop, 1)

            if ast_type == "FUNC":
                self._set_slot(canon_vec, cand_vec, "EXT_AST_FUNCTION_DEF", 1)
                self._set_slot(canon_vec, cand_vec, "GRAPH_ROOT_NODE", 1)
                self._set_slot(canon_vec, cand_vec, "GRAPH_INVOCATION_HEAD", 1)
                self._set_slot(canon_vec, cand_vec, "EXT_AST_SCOPE_ENTER", 1)
            elif ast_type == "LOOP":
                self._set_slot(canon_vec, cand_vec, "EXT_AST_CONTROL_LOOP", 1)
                self._set_slot(canon_vec, cand_vec, "GRAPH_CYCLIC_BACKLINK", 1)
                self._set_slot(canon_vec, cand_vec, "NSM_CONTINUOUS_RATE", 1)
            elif ast_type == "BRANCH":
                self._set_slot(canon_vec, cand_vec, "GRAPH_BRANCH_COND", 1)
                self._set_slot(canon_vec, cand_vec, "GRAPH_BRANCH_THEN", 1)
                self._set_slot(canon_vec, cand_vec, "GRAPH_BRANCH_ELSE", 1)
                self._set_slot(canon_vec, cand_vec, "NSM_IF", 1)
            elif ast_type == "ASSIGN":
                self._set_slot(canon_vec, cand_vec, "EXT_AST_VARIABLE_BINDING", 1)
                self._set_slot(canon_vec, cand_vec, "LJB_DU_IDENTITY", 1)
                self._set_slot(canon_vec, cand_vec, "TYPE_NUMERIC_VALUE", 1)
            elif ast_type == "RETURN":
                self._set_slot(canon_vec, cand_vec, "EXT_AST_RETURN", 1)
                self._set_slot(canon_vec, cand_vec, "GRAPH_RETURN_VALUE", 1)
                self._set_slot(canon_vec, cand_vec, "EXT_AST_SCOPE_EXIT", 1)
            elif ast_type in ("TRY_EXCEPT", "ASYNC_AWAIT", "MATCH_CASE"):
                self._set_slot(canon_vec, cand_vec, "LJB_ASYNC_CONCURRENT", 1)
                self._set_slot(canon_vec, cand_vec, "LJB_MUTEX_DEPENDENCY", 1)

            return f"CodeAST #{sample_idx}: Construct {ast_type} on {math_struct} ({api_afford})."

        return f"Unknown proposition #{sample_idx}"
