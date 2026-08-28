"""Discrete Information Bottleneck profiler and entropy evaluation suite for QUANTA."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core.slots import get_slot_names


class QuantaInformationProfiler:
    """Evaluates channel capacity, individual slot entropy, and pairwise redundancy

    for 256-dimensional quaternary state spaces.
    """

    def __init__(self, data_matrix: np.ndarray, dimension_labels: Optional[List[str]] = None):
        """data_matrix: (N, 256) array with values in {0, 1, 2, 3}

        dimension_labels: List of 256 human-readable dimension names
        """
        self.X = data_matrix.astype(np.uint8)
        self.N, self.D = data_matrix.shape
        self.labels = dimension_labels or get_slot_names()
        assert self.D == 256, f"Matrix must contain exactly 256 dimensions, got {self.D}"
        assert len(self.labels) == 256, f"Labels must contain 256 names, got {len(self.labels)}"

    def compute_entropies(self) -> np.ndarray:
        """Calculates Shannon entropy H(D_i) for each dimension in bits.
        
        Theoretical Maximum: log2(4) = 2.0 bits.
        """
        entropies = np.zeros(self.D, dtype=np.float64)
        for i in range(self.D):
            col = self.X[:, i]
            counts = np.bincount(col, minlength=4)
            probs = counts / self.N
            probs = probs[probs > 0]
            entropies[i] = -np.sum(probs * np.log2(probs))
        return entropies

    def compute_pairwise_mutual_information(
        self,
        top_k_pairs: int = 10,
        subsample_dim_pairs: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """Finds the most redundant pairs of dimensions using Mutual Information I(D_i; D_j)."""
        redundant_pairs = []

        # Vectorized joint counts across 4x4 discrete space
        for i in range(self.D):
            col_i = self.X[:, i]
            for j in range(i + 1, self.D):
                col_j = self.X[:, j]

                # Joint probability distribution P(D_i, D_j)
                joint_counts = np.zeros((4, 4), dtype=np.float64)
                np.add.at(joint_counts, (col_i, col_j), 1.0)
                p_ij = joint_counts / self.N

                p_i = np.sum(p_ij, axis=1, keepdims=True)
                p_j = np.sum(p_ij, axis=0, keepdims=True)

                denom = p_i @ p_j
                mask = (p_ij > 0) & (denom > 0)
                mi = np.sum(p_ij[mask] * np.log2(p_ij[mask] / denom[mask]))

                redundant_pairs.append((self.labels[i], self.labels[j], float(max(0.0, mi))))

        # Sort by highest mutual information
        redundant_pairs.sort(key=lambda x: x[2], reverse=True)
        return redundant_pairs[:top_k_pairs]

    def compute_collision_rate(self, concept_ids: Optional[Sequence[str]] = None) -> float:
        """Measures whether two distinctly labeled concepts share the exact same vector.
        
        R_collision = |{ (a, b) | a != b and v(a) == v(b) }| / (N choose 2)
        """
        if self.N < 2:
            return 0.0

        # Find duplicate rows
        unique_rows, counts = np.unique(self.X, axis=0, return_counts=True)
        # Pairs of duplicates: sum(c * (c - 1) / 2)
        duplicate_pairs = int(np.sum(counts * (counts - 1) // 2))
        total_pairs = (self.N * (self.N - 1)) // 2
        return float(duplicate_pairs / total_pairs) if total_pairs > 0 else 0.0

    def compute_band_entropies(self) -> Dict[str, float]:
        """Calculates average entropy for each of the 4 bands."""
        entropies = self.compute_entropies()
        return {
            "Band 0 (NSM & Kinematics)": float(np.mean(entropies[0:64])),
            "Band 1 (Valencies & Topology)": float(np.mean(entropies[64:128])),
            "Band 2 (Ontology & Modality)": float(np.mean(entropies[128:192])),
            "Band 3 (Epistemics & Metarules)": float(np.mean(entropies[192:256])),
            "Total Mean Entropy": float(np.mean(entropies)),
        }

    def run_diagnostic_suite(
        self,
        dead_threshold: float = 0.1,
        redundancy_threshold: float = 0.5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Runs automated evaluation and outputs actionable dimension refactoring advice."""
        entropies = self.compute_entropies()
        redundancies = self.compute_pairwise_mutual_information(top_k_pairs=15)
        collision_rate = self.compute_collision_rate()
        band_stats = self.compute_band_entropies()

        dead_slots = [
            (idx, self.labels[idx], float(entropies[idx]))
            for idx in range(self.D)
            if entropies[idx] < dead_threshold
        ]

        high_redundancy_pairs = [
            (dim_a, dim_b, mi)
            for dim_a, dim_b, mi in redundancies
            if mi >= redundancy_threshold
        ]

        report = {
            "num_samples": self.N,
            "mean_entropy": float(np.mean(entropies)),
            "min_entropy": float(np.min(entropies)),
            "max_entropy": float(np.max(entropies)),
            "collision_rate": collision_rate,
            "band_entropies": band_stats,
            "dead_slots": dead_slots,
            "high_redundancy_pairs": high_redundancy_pairs,
        }

        if verbose:
            print("=== QUANTA INFORMATION PROFILER REPORT ===")
            print(f"Total Samples Evaluated: {self.N}")
            print(f"Average Dimension Entropy: {report['mean_entropy']:.3f} / 2.000 bits")
            print(f"Concept Collision Rate: {report['collision_rate']:.6f}")
            print("\n--- Band-wise Mean Entropy ---")
            for band_name, val in band_stats.items():
                print(f"  {band_name}: {val:.3f} bits")

            print(f"\n--- 1. Under-Utilized / Dead Dimensions (< {dead_threshold} bits) ---")
            if not dead_slots:
                print("  None! All dimensions meet minimum entropy requirements.")
            else:
                for idx, label, h in dead_slots:
                    print(f"  Slot {idx:03d} [{label}]: Entropy = {h:.4f} bits (Candidate for pruning/replacement)")

            print(f"\n--- 2. High Redundancy Pairs (Mutual Information > {redundancy_threshold} bits) ---")
            if not high_redundancy_pairs:
                print("  None! No pairs exceed the redundancy threshold.")
            else:
                for dim_a, dim_b, mi in high_redundancy_pairs:
                    print(f"  [{dim_a}] <---> [{dim_b}]: MI = {mi:.4f} bits (High co-linearity; consider collapsing)")

        return report
