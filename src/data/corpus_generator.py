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
    canonical_matrix: np.ndarray  # (N, 256) in {0, 1, 2, 3}
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
                canon_vec = np.zeros(256, dtype=np.uint8)
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
        """Populates slots across all 4 bands based on the domain semantic template."""
        # Baseline noise/variation across bands to model natural entropy
        # Randomly activate a small background subset of general NSM/epistemic slots
        rand_nsm = self.rng.choice([6, 7, 8, 9, 10, 11, 12, 13, 14, 15], size=2, replace=False)
        for s in rand_nsm:
            canon_vec[s] = self.rng.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.05, 0.05])

        if domain_name == "FOLIO":
            # First Order Logic: quantifiers, implication, predicate structures
            quant = self.rng.choice(["ALL", "SOME", "ONE", "NONE"])
            is_negated = bool(self.rng.random() < 0.3)
            
            self._set_slot(canon_vec, cand_vec, "TYPE_PROPOSITION", 1)
            self._set_slot(canon_vec, cand_vec, "GRAPH_ASSERTION_CLAIM", 1)
            self._set_slot(canon_vec, cand_vec, "EPIST_DEDUCTIVE_INFERENCE", 1)
            self._set_slot(canon_vec, cand_vec, "LJB_GANAI_IF_THEN", 1)
            self._set_slot(canon_vec, cand_vec, "SOLVER_PROOF_VALIDATED", 1)

            if quant == "ALL":
                self._set_slot(canon_vec, cand_vec, "NSM_ALL", 1)
                self._set_slot(canon_vec, cand_vec, "LJB_RO_ALL_QUANT", 1)
            elif quant == "SOME":
                self._set_slot(canon_vec, cand_vec, "NSM_SOME", 1)
                self._set_slot(canon_vec, cand_vec, "LJB_SUO_AT_LEAST_ONE", 1)
            elif quant == "NONE":
                self._set_slot(canon_vec, cand_vec, "LJB_NO_NONE_QUANT", 1)

            if is_negated:
                self._set_slot(canon_vec, cand_vec, "LJB_NA_NEGATION", 2)
            else:
                self._set_slot(canon_vec, cand_vec, "NSM_TRUE", 1)

            # Randomize predicate entity types
            e_type = self.rng.choice(["TYPE_HUMAN", "TYPE_ANIMATE", "TYPE_ABSTRACT_CONCEPT", "TYPE_ARTIFACT"])
            self._set_slot(canon_vec, cand_vec, e_type, 1)
            self._set_slot(canon_vec, cand_vec, "VAL_X1_AGENT", 1)
            self._set_slot(canon_vec, cand_vec, "VAL_X2_PATIENT", 1)
            return f"FOLIO #{sample_idx}: {quant} {e_type} {'is not' if is_negated else 'is'} implied."

        elif domain_name == "ProofWriter":
            # Multi-step deductive facts and closed world reasoning
            self._set_slot(canon_vec, cand_vec, "TYPE_STATE", 1)
            self._set_slot(canon_vec, cand_vec, "EPIST_AXIOMATIC_PREMISE", 1)
            self._set_slot(canon_vec, cand_vec, "SOLVER_CWA_CLOSED_WORLD", 1)
            self._set_slot(canon_vec, cand_vec, "LJB_CA_PRESENT_TENSE", 1)

            has_contradiction = bool(self.rng.random() < 0.15)
            if has_contradiction:
                self._set_slot(canon_vec, cand_vec, "SOLVER_CONTRADICTION_FLAG", 2)
                self._set_slot(canon_vec, cand_vec, "SOLVER_MUC_TARGETED", 1)
                self._set_slot(canon_vec, cand_vec, "SOLVER_RE_DENOISE_REQUIRED", 1)
            else:
                self._set_slot(canon_vec, cand_vec, "EPIST_PROB_CERTAIN", 1)

            # Trait attribution
            attr_slot = self.rng.choice(["NSM_GOOD", "NSM_BIG", "NSM_SMALL", "TYPE_ATTRIBUTE_PROPERTY", "WN_ATTRIBUTE_PROP"])
            self._set_slot(canon_vec, cand_vec, attr_slot, 1)
            self._set_slot(canon_vec, cand_vec, "TYPE_ANIMATE", 1)
            self._set_slot(canon_vec, cand_vec, "VAL_EXPERIENCER", 1)
            return f"ProofWriter #{sample_idx}: Axiom fact with attribute {attr_slot} (Contradiction={has_contradiction})."

        elif domain_name == "bAbI":
            # Spatial navigation, agent actions, object pickup and drop
            action = self.rng.choice(["MOVE", "TOUCH", "SAY", "GIVE"])
            self._set_slot(canon_vec, cand_vec, "TYPE_EVENT", 1)
            self._set_slot(canon_vec, cand_vec, "LJB_PU_PAST_TENSE", 1)
            self._set_slot(canon_vec, cand_vec, "TYPE_HUMAN", 1)
            self._set_slot(canon_vec, cand_vec, "ROLE_AGENT_CAPABLE", 1)
            self._set_slot(canon_vec, cand_vec, "VAL_X1_AGENT", 1)
            self._set_slot(canon_vec, cand_vec, "MODALITY_LITERAL", 1)
            self._set_slot(canon_vec, cand_vec, "EPIST_DIRECT_OBSERVATION", 1)

            if action == "MOVE":
                self._set_slot(canon_vec, cand_vec, "NSM_MOVE", 1)
                self._set_slot(canon_vec, cand_vec, "VAL_X3_DESTINATION", 1)
                self._set_slot(canon_vec, cand_vec, "VAL_X4_SOURCE", 1)
                self._set_slot(canon_vec, cand_vec, "TYPE_SPATIAL_REGION", 1)
                self._set_slot(canon_vec, cand_vec, "WN_LOCATION_PLACE", 1)
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
            return f"bAbI #{sample_idx}: Agent performs {action} in physical world."

        elif domain_name == "CLUTRR":
            # Kinship & Social Relations
            self._set_slot(canon_vec, cand_vec, "TYPE_HUMAN", 1)
            self._set_slot(canon_vec, cand_vec, "TYPE_RELATION_ROLE", 1)
            self._set_slot(canon_vec, cand_vec, "WN_PERSON_HUMAN", 1)
            self._set_slot(canon_vec, cand_vec, "WN_GROUP_SOCIAL", 1)
            self._set_slot(canon_vec, cand_vec, "VAL_X1_AGENT", 1)
            self._set_slot(canon_vec, cand_vec, "VAL_X2_PATIENT", 1)
            self._set_slot(canon_vec, cand_vec, "EPIST_DEDUCTIVE_INFERENCE", 1)
            self._set_slot(canon_vec, cand_vec, "LJB_SOI_RECIPROCAL", 1)

            rel_type = self.rng.choice(["PARENT", "CHILD", "SIBLING", "SPOUSE"])
            if rel_type in ("PARENT", "CHILD"):
                self._set_slot(canon_vec, cand_vec, "NSM_BORN", 1)
                self._set_slot(canon_vec, cand_vec, "EXT_DOM_MEREOLOGICAL_PART", 1)
            else:
                self._set_slot(canon_vec, cand_vec, "NSM_SAME", 1)
            return f"CLUTRR #{sample_idx}: Kinship deduction of {rel_type} link."

        elif domain_name == "CodeAST":
            # Executable code, recursion, branches, variable bindings
            ast_type = self.rng.choice(["FUNC", "LOOP", "BRANCH", "ASSIGN", "RETURN"])
            self._set_slot(canon_vec, cand_vec, "TYPE_PROCESS", 1)
            self._set_slot(canon_vec, cand_vec, "GRAPH_IS_SUB_EXP", 1)

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

            # Mutex / concurrency tags occasionally
            if self.rng.random() < 0.2:
                self._set_slot(canon_vec, cand_vec, "LJB_ASYNC_CONCURRENT", 1)
                self._set_slot(canon_vec, cand_vec, "LJB_MUTEX_DEPENDENCY", 1)

            return f"CodeAST #{sample_idx}: AST construct {ast_type}."

        return f"Unknown proposition #{sample_idx}"
