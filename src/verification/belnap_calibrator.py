"""Calibrated Belnap Truth Lattice Mapper for QUANTA SVM (Section 5).

Maps Kev-4B relational posteriors and multi-source epistemic assertions directly
to the Belnap 4-valued logic lattice:
    B_4 = {00_2, 01_2, 10_2, 11_2}
    - 00_2 (0): CONTRADICTION / BOTH (True and False)
    - 01_2 (1): TRUE / Verified Fact
    - 10_2 (2): FALSE / Refuted
    - 11_2 (3): UNKNOWN / Maybe / Unverified

Includes:
1. BelnapLatticeMapper: Configurable probability thresholding (P >= 0.85 -> TRUE,
   P <= 0.15 -> FALSE, 0.15 < P < 0.85 -> UNKNOWN), uint8 confidence quantization
   (P in [0, 1] -> [0, 255]), and lattice algebraic meet/join operations.
2. TemperatureCalibrator: Post-hoc Platt/temperature scaling for logit/probability calibration
   with ECE and Brier score evaluation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.binary_node import BelnapValue

logger = logging.getLogger("quanta.verification.belnap_calibrator")

# Default thresholds specified in Section 5 master plan
DEFAULT_TRUE_THRESHOLD: float = 0.85
DEFAULT_FALSE_THRESHOLD: float = 0.15


def quantize_confidence(prob: float) -> int:
    """Quantizes confidence probability P in [0.0, 1.0] to uint8 in [0, 255].
    
    Exact mapping for the 128-byte C-compatible binary struct.
    """
    p = max(0.0, min(1.0, float(prob)))
    return int(round(p * 255.0))


def dequantize_confidence(conf_u8: int) -> float:
    """Dequantizes uint8 confidence in [0, 255] back to float in [0.0, 1.0]."""
    return float(max(0, min(255, int(conf_u8)))) / 255.0


class TemperatureCalibrator:
    """Post-hoc temperature scaling calibrator for neural model posteriors.
    
    Transforms probabilities or logits by temperature parameter T > 0:
        logit' = logit / T
        P' = sigma(logit')
    """

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature: float = float(temperature)

    def calibrate(self, prob: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Applies temperature scaling to probability values in [0.0, 1.0]."""
        if self.temperature == 1.0:
            return prob

        is_scalar = isinstance(prob, (int, float))
        p_arr = np.asarray(prob, dtype=np.float64)
        # Avoid log(0) and log(1)
        eps = 1e-7
        p_clipped = np.clip(p_arr, eps, 1.0 - eps)
        # Logit: log(p / (1 - p))
        logit = np.log(p_clipped / (1.0 - p_clipped))
        scaled_logit = logit / self.temperature
        calibrated = 1.0 / (1.0 + np.exp(-scaled_logit))

        if is_scalar:
            return float(calibrated.item())
        return calibrated

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Applies temperature scaling directly to raw logits."""
        scaled = np.asarray(logits, dtype=np.float64) / self.temperature
        # Numerically stable softmax
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Finds optimal temperature T minimizing negative log-likelihood (NLL)."""
        from scipy.optimize import minimize_scalar

        p_arr = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        y_arr = np.asarray(labels, dtype=np.float64)
        logits = np.log(p_arr / (1.0 - p_arr))

        def nll_objective(t: float) -> float:
            scaled = logits / max(1e-4, t)
            # Binary cross-entropy: - [y * log(sigma(z)) + (1-y) * log(1 - sigma(z))]
            loss = np.maximum(scaled, 0.0) - scaled * y_arr + np.log1p(np.exp(-np.abs(scaled)))
            return float(np.mean(loss))

        res = minimize_scalar(nll_objective, bounds=(0.05, 5.0), method="bounded")
        self.temperature = float(res.x)
        return self.temperature

    @staticmethod
    def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error (ECE) with equal-width bins."""
        p_arr = np.asarray(probs, dtype=np.float64)
        y_arr = np.asarray(labels, dtype=np.float64)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(p_arr)

        if n_total == 0:
            return 0.0

        for i in range(n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (p_arr >= bin_lower) & (p_arr <= bin_upper)
            else:
                mask = (p_arr >= bin_lower) & (p_arr < bin_upper)

            bin_size = int(np.sum(mask))
            if bin_size > 0:
                bin_acc = float(np.mean(y_arr[mask]))
                bin_conf = float(np.mean(p_arr[mask]))
                ece += (bin_size / n_total) * abs(bin_acc - bin_conf)

        return float(ece)

    @staticmethod
    def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
        """Computes Brier score (mean squared calibration error)."""
        p_arr = np.asarray(probs, dtype=np.float64)
        y_arr = np.asarray(labels, dtype=np.float64)
        if len(p_arr) == 0:
            return 0.0
        return float(np.mean((p_arr - y_arr) ** 2))


class BelnapLatticeMapper:
    """Calibrates and maps relational posteriors to the Belnap 4-valued logic lattice.
    
    Enforces the formal Section 5 threshold contract:
        - P >= 0.85  ==>  01_2 (BelnapValue.TRUE)
        - P <= 0.15  ==>  10_2 (BelnapValue.FALSE)
        - 0.15 < P < 0.85  ==>  11_2 (BelnapValue.UNKNOWN / MAYBE)
        - Conflicting / contradictory evidence ==> 00_2 (BelnapValue.CONTRADICTION)
    """

    def __init__(
        self,
        true_threshold: float = DEFAULT_TRUE_THRESHOLD,
        false_threshold: float = DEFAULT_FALSE_THRESHOLD,
        calibrator: Optional[TemperatureCalibrator] = None,
    ):
        if false_threshold >= true_threshold:
            raise ValueError(
                f"false_threshold ({false_threshold}) must be strictly less than "
                f"true_threshold ({true_threshold})"
            )
        self.true_threshold: float = float(true_threshold)
        self.false_threshold: float = float(false_threshold)
        self.calibrator: Optional[TemperatureCalibrator] = calibrator

    def map_probability(
        self,
        prob: float,
        has_conflict: bool = False,
    ) -> BelnapValue:
        """Maps scalar probability to Belnap lattice state."""
        if has_conflict:
            return BelnapValue.CONTRADICTION

        p = self.calibrator.calibrate(prob) if self.calibrator else float(prob)

        if p >= self.true_threshold:
            return BelnapValue.TRUE
        if p <= self.false_threshold:
            return BelnapValue.FALSE
        return BelnapValue.UNKNOWN

    def map_probabilities_to_belnap(
        self,
        probs: Union[np.ndarray, Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized mapping of probability array to Belnap uint8 codes and uint8 confidences.
        
        Returns:
            Tuple of:
            - belnap_codes: np.ndarray (uint8) of Belnap values (0, 1, 2, 3)
            - confidences: np.ndarray (uint8) quantized to [0, 255]
        """
        p_arr = np.asarray(probs, dtype=np.float32)
        if self.calibrator:
            p_arr = np.asarray(self.calibrator.calibrate(p_arr), dtype=np.float32)

        belnap = np.full(p_arr.shape, BelnapValue.UNKNOWN, dtype=np.uint8)
        belnap[p_arr >= self.true_threshold] = BelnapValue.TRUE
        belnap[p_arr <= self.false_threshold] = BelnapValue.FALSE

        conf_u8 = np.clip(np.round(p_arr * 255.0), 0, 255).astype(np.uint8)
        return belnap, conf_u8

    def map_prediction(
        self,
        prediction: Any,
    ) -> Tuple[BelnapValue, int]:
        """Maps a prediction object (e.g., KevPrediction or tuple) to (BelnapValue, conf_u8).
        
        Handles:
        - KevPrediction (with probability and optional label)
        - tuple: (label, probability) or (probability,)
        - dict: {"probability": p, ...}
        - float/int: raw scalar probability
        """
        prob: float = 0.5
        has_conflict: bool = False

        if hasattr(prediction, "probability"):
            prob = float(prediction.probability)
            if hasattr(prediction, "has_conflict"):
                has_conflict = bool(prediction.has_conflict)
        elif isinstance(prediction, dict):
            prob = float(prediction.get("probability", prediction.get("confidence", 0.5)))
            has_conflict = bool(prediction.get("has_conflict", False))
        elif isinstance(prediction, (tuple, list)):
            if len(prediction) >= 2 and isinstance(prediction[1], (int, float)):
                prob = float(prediction[1])
            elif len(prediction) >= 1 and isinstance(prediction[0], (int, float)):
                prob = float(prediction[0])
        elif isinstance(prediction, (int, float)):
            prob = float(prediction)

        lattice_val = self.map_probability(prob, has_conflict=has_conflict)
        conf_u8 = quantize_confidence(prob)
        return lattice_val, conf_u8

    @staticmethod
    def knowledge_meet(v1: Union[int, BelnapValue], v2: Union[int, BelnapValue]) -> BelnapValue:
        """Belnap knowledge meet (information lower bound, /\\_k).
        
        In the knowledge ordering:
            UNKNOWN <= TRUE <= CONTRADICTION
            UNKNOWN <= FALSE <= CONTRADICTION
            
        Meet rules:
            x /\\_k x = x
            UNKNOWN /\\_k x = UNKNOWN
            CONTRADICTION /\\_k x = x
            TRUE /\\_k FALSE = UNKNOWN
        """
        c1 = int(v1) & 0x03
        c2 = int(v2) & 0x03

        if c1 == c2:
            return BelnapValue(c1)
        if c1 == BelnapValue.UNKNOWN or c2 == BelnapValue.UNKNOWN:
            return BelnapValue.UNKNOWN
        if c1 == BelnapValue.CONTRADICTION:
            return BelnapValue(c2)
        if c2 == BelnapValue.CONTRADICTION:
            return BelnapValue(c1)
        # TRUE and FALSE meet at bottom of knowledge (UNKNOWN)
        return BelnapValue.UNKNOWN

    @staticmethod
    def knowledge_join(v1: Union[int, BelnapValue], v2: Union[int, BelnapValue]) -> BelnapValue:
        """Belnap knowledge join (information pooling / consensus, \\/_k).
        
        Pooling rules:
            x \\/_k x = x
            UNKNOWN \\/_k x = x
            CONTRADICTION \\/_k x = CONTRADICTION
            TRUE \\/_k FALSE = CONTRADICTION (both asserted simultaneously)
        """
        c1 = int(v1) & 0x03
        c2 = int(v2) & 0x03

        if c1 == c2:
            return BelnapValue(c1)
        if c1 == BelnapValue.CONTRADICTION or c2 == BelnapValue.CONTRADICTION:
            return BelnapValue.CONTRADICTION
        if c1 == BelnapValue.UNKNOWN:
            return BelnapValue(c2)
        if c2 == BelnapValue.UNKNOWN:
            return BelnapValue(c1)
        # One asserts TRUE and other asserts FALSE -> CONTRADICTION!
        return BelnapValue.CONTRADICTION

    def reconcile_sources(
        self,
        source_probs: Sequence[float],
    ) -> Tuple[BelnapValue, float]:
        """Reconciles multiple independent source probabilities for a single relation or fact.
        
        If sources produce mutually contradictory claims (e.g. source 1 says P >= 0.85,
        source 2 says P <= 0.15), returns BelnapValue.CONTRADICTION.
        
        Returns:
            Tuple of (BelnapValue, pooled_confidence_float)
        """
        if not source_probs:
            return BelnapValue.UNKNOWN, 0.5

        mapped_states = [self.map_probability(p) for p in source_probs]

        # Check for epistemic clash: both TRUE and FALSE present
        has_true = any(s == BelnapValue.TRUE for s in mapped_states)
        has_false = any(s == BelnapValue.FALSE for s in mapped_states)
        has_contra = any(s == BelnapValue.CONTRADICTION for s in mapped_states)

        if has_contra or (has_true and has_false):
            mean_conf = float(np.mean([abs(p - 0.5) * 2.0 for p in source_probs]))
            return BelnapValue.CONTRADICTION, mean_conf

        if has_true:
            true_probs = [p for p in source_probs if p >= self.true_threshold]
            return BelnapValue.TRUE, float(np.mean(true_probs))

        if has_false:
            false_probs = [p for p in source_probs if p <= self.false_threshold]
            return BelnapValue.FALSE, float(np.mean(false_probs))

        return BelnapValue.UNKNOWN, float(np.mean(source_probs))


__all__ = [
    "DEFAULT_TRUE_THRESHOLD",
    "DEFAULT_FALSE_THRESHOLD",
    "quantize_confidence",
    "dequantize_confidence",
    "TemperatureCalibrator",
    "BelnapLatticeMapper",
]
