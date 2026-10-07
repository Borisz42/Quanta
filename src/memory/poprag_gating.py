"""PoP-RAG Epistemic Gating for QUANTA SVM (Section 6).

Implements Polarity-of-Premise (PoP-RAG) epistemic gating over ASG edges and semantic
nodes using the calibrated Belnap 4-valued logic lattice:
    B_4 = {00_2, 01_2, 10_2, 11_2}
    - 00_2 (0): CONTRADICTION / BOTH -> Edge weight set to 0.0 (path completely pruned).
    - 01_2 (1): TRUE / FACT         -> Edge weight maintained at nominal value.
    - 11_2 (3): UNKNOWN / MAYBE     -> Edge weight scaled by calibrated confidence P.
    - 10_2 (2): FALSE / REFUTED     -> Inverted or suppressed based on query target polarity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from core.asg import QuantaNode
from core.binary_node import BelnapValue

logger = logging.getLogger("quanta.memory.poprag_gating")

# Default nominal base weights by relation category
DEFAULT_RELATION_WEIGHTS: Dict[str, float] = {
    # Pearl Causal links (high explanatory power)
    "CAUSAL_LEADS_TO": 1.2,
    "CAUSAL_MECHANISM_LINK": 1.2,
    "ENABLING_CONDITION": 1.1,
    "CAUSAL_PREVENTS": 1.1,
    # Allen Temporal intervals
    "TEMP_ALLEN_MEETS": 1.0,
    "TEMP_ALLEN_BEFORE": 1.0,
    "TEMP_ALLEN_OVERLAPS": 0.95,
    "TEMP_ALLEN_DURING": 0.95,
    "TEMP_ALLEN_STARTS": 0.95,
    "TEMP_ALLEN_FINISHES": 0.95,
    "TEMP_ALLEN_EQUALS": 1.0,
    # Thematic Valency slots
    "VAL_X1_AGENT": 1.0,
    "VAL_X2_PATIENT": 1.0,
    "VAL_LOCATION_SLOT": 1.0,
    "VAL_X5_INSTRUMENT": 1.0,
    "VAL_TIME_SLOT": 1.0,
    # Encyclopedic / structural links
    "LOCATED_IN": 0.9,
    "EDUCATED_AT": 0.9,
    "STUDIED_AT": 0.9,
    "BORN_IN": 0.9,
    "PART_OF": 0.9,
    "SHARES_BORDER": 0.9,
    "HEADQUARTERS": 0.9,
    "MEMBER_OF": 0.9,
    "COUNTRY": 0.9,
    "CAPITAL": 0.9,
    "CROSS_CHUNK_BRIDGE": 0.95,
    "CFG_NEXT": 1.0,
    "CALLS": 1.0,
    "INHERITS_FROM": 1.0,
    "IMPLEMENTS": 1.0,
    "IMPORTS": 1.0,
    "DATA_FLOW_DEF_USE": 1.0,
    "CO_OCCURS": 0.85,
}

DEFAULT_NOMINAL_WEIGHT: float = 1.0


class PoPRAGGating:
    """Polarity-of-Premise (PoP-RAG) epistemic gating module.
    
    Dynamically modulates graph transition probabilities based on:
    1. Belnap 4-valued lattice states (CONTRADICTION, TRUE, FALSE, UNKNOWN).
    2. Calibrated epistemic confidence values P in [0.0, 1.0].
    3. Query target polarity (positive assertion vs. negative refutation).
    """

    def __init__(
        self,
        relation_weights: Optional[Dict[str, float]] = None,
        default_nominal_weight: float = DEFAULT_NOMINAL_WEIGHT,
        contra_weight: float = 0.0,
        false_suppressed_weight: float = 0.0,
    ):
        """Initializes PoP-RAG epistemic gating.
        
        Args:
            relation_weights: Custom mapping from relation string to base weight.
            default_nominal_weight: Fallback nominal weight for unlisted relations.
            contra_weight: Weight assigned to contradictory edges (default 0.0 -> pruned).
            false_suppressed_weight: Weight assigned to false edges on positive queries (default 0.0).
        """
        self.relation_weights: Dict[str, float] = dict(DEFAULT_RELATION_WEIGHTS)
        if relation_weights:
            self.relation_weights.update(relation_weights)
        self.default_nominal_weight = float(default_nominal_weight)
        self.contra_weight = float(contra_weight)
        self.false_suppressed_weight = float(false_suppressed_weight)

    def compute_gate_factor(
        self,
        truth_state: Union[BelnapValue, int, str],
        confidence: float = 1.0,
        query_polarity: str = "positive",
    ) -> float:
        """Computes the epistemic modulation factor in [0.0, 1.0].
        
        Rules:
        - 00_2 (Contradiction): returns 0.0 (path completely pruned).
        - 01_2 (True): returns 1.0 (weight maintained at nominal value).
        - 11_2 (Unknown): returns calibrated confidence P in [0.0, 1.0] (attenuated).
        - 10_2 (False):
            - If query_polarity == "positive" -> returns 0.0 (suppressed).
            - If query_polarity == "negative" -> returns refutation confidence (inverted/active).
        
        Args:
            truth_state: Belnap lattice state (string, int, or BelnapValue).
            confidence: Calibrated confidence float in [0.0, 1.0].
            query_polarity: 'positive' (default) or 'negative' / 'refutation'.
            
        Returns:
            Multiplicative gate factor in [0.0, 1.0].
        """
        code = BelnapValue.from_str(truth_state)
        conf = float(np.clip(confidence, 0.0, 1.0))

        if code == BelnapValue.CONTRADICTION:
            # 00_2: Contradiction -> Prune path completely
            return self.contra_weight

        if code == BelnapValue.TRUE:
            # 01_2: True -> Maintained at nominal value
            return 1.0

        if code == BelnapValue.UNKNOWN:
            # 11_2: Unknown -> Scaled by calibrated confidence P
            return conf

        if code == BelnapValue.FALSE:
            # 10_2: False -> Suppressed on positive query, inverted on negative query
            is_negative_query = query_polarity.lower() in ("negative", "refutation", "counterfactual", "false")
            if is_negative_query:
                # Active: If claim was asserted False with high confidence, refutation strength is high
                refutation_strength = (1.0 - conf) if conf > 0.5 else max(0.5, 1.0 - conf)
                return float(np.clip(refutation_strength, 0.0, 1.0))
            return self.false_suppressed_weight

        return conf

    def get_base_weight(self, relation: str) -> float:
        """Gets nominal base weight for a relation type."""
        return self.relation_weights.get(relation, self.default_nominal_weight)

    def gate_edge_weight(
        self,
        base_weight: float,
        truth_state: Union[BelnapValue, int, str],
        confidence: float = 1.0,
        query_polarity: str = "positive",
    ) -> float:
        """Computes effective edge weight by multiplying base weight with epistemic gate."""
        factor = self.compute_gate_factor(truth_state, confidence, query_polarity)
        return float(base_weight * factor)

    def gate_edge(
        self,
        source_node: Optional[QuantaNode],
        target_node: Optional[QuantaNode],
        relation: str,
        base_weight: Optional[float] = None,
        edge_truth: Optional[Union[BelnapValue, int, str]] = None,
        edge_confidence: Optional[float] = None,
        query_polarity: str = "positive",
    ) -> float:
        """Gates a specific graph edge between source and target nodes.
        
        Incorporates both edge-level epistemic annotations and target/source node truth status.
        If either the edge, source node, or target node is in a contradictory state (00_2),
        the edge weight is strictly 0.0.
        
        Args:
            source_node: Source QuantaNode or None.
            target_node: Target QuantaNode or None.
            relation: Relation string (e.g. 'CAUSAL_LEADS_TO', 'VAL_X1_AGENT').
            base_weight: Optional explicit base weight (defaults to relation-specific weight).
            edge_truth: Optional explicit edge-level Belnap truth status.
            edge_confidence: Optional explicit edge-level confidence.
            query_polarity: 'positive' or 'negative'.
            
        Returns:
            Gated edge weight >= 0.0.
        """
        w0 = base_weight if base_weight is not None else self.get_base_weight(relation)

        # 1. Source node contradiction check: Can never spread out of a contradictory premise
        if source_node is not None:
            src_state = BelnapValue.from_str(getattr(source_node, "truth_status", "TRUE"))
            if src_state == BelnapValue.CONTRADICTION:
                return 0.0

        # 2. Edge-level epistemic gating
        edge_factor = 1.0
        if edge_truth is not None:
            e_conf = edge_confidence if edge_confidence is not None else 1.0
            edge_factor = self.compute_gate_factor(edge_truth, e_conf, query_polarity)
            if edge_factor <= 0.0:
                return 0.0

        # 3. Target node epistemic gating
        target_factor = 1.0
        if target_node is not None:
            tgt_state = BelnapValue.from_str(getattr(target_node, "truth_status", "TRUE"))
            tgt_conf = float(getattr(target_node, "confidence", 1.0))
            target_factor = self.compute_gate_factor(tgt_state, tgt_conf, query_polarity)
            if target_factor <= 0.0:
                return 0.0

        effective_factor = edge_factor * target_factor
        return float(w0 * effective_factor)

    def gate_weights_vectorized(
        self,
        base_weights: np.ndarray,
        belnap_states: np.ndarray,
        confidences: np.ndarray,
        query_polarity: str = "positive",
    ) -> np.ndarray:
        """Vectorized NumPy / SIMD calculation of gated weights for high-throughput batch evaluation.
        
        Args:
            base_weights: 1D array of float32/float64 base weights.
            belnap_states: 1D array of uint8 Belnap codes (0, 1, 2, 3).
            confidences: 1D array of float32/float64 confidences in [0.0, 1.0].
            query_polarity: 'positive' or 'negative'.
            
        Returns:
            1D array of modulated edge weights.
        """
        bw = np.asarray(base_weights, dtype=np.float32)
        states = np.asarray(belnap_states, dtype=np.uint8) & 0x03
        conf = np.clip(np.asarray(confidences, dtype=np.float32), 0.0, 1.0)

        multipliers = np.zeros_like(bw)

        # 01_2 (True) -> 1.0
        multipliers[states == BelnapValue.TRUE] = 1.0

        # 11_2 (Unknown) -> conf
        unknown_mask = (states == BelnapValue.UNKNOWN)
        multipliers[unknown_mask] = conf[unknown_mask]

        # 00_2 (Contradiction) -> self.contra_weight (0.0)
        multipliers[states == BelnapValue.CONTRADICTION] = self.contra_weight

        # 10_2 (False)
        is_negative = query_polarity.lower() in ("negative", "refutation", "counterfactual", "false")
        false_mask = (states == BelnapValue.FALSE)
        if is_negative:
            ref_weights = np.where(conf[false_mask] > 0.5, 1.0 - conf[false_mask], np.maximum(0.5, 1.0 - conf[false_mask]))
            multipliers[false_mask] = ref_weights
        else:
            multipliers[false_mask] = self.false_suppressed_weight

        return bw * multipliers


__all__ = [
    "DEFAULT_RELATION_WEIGHTS",
    "DEFAULT_NOMINAL_WEIGHT",
    "PoPRAGGating",
]
