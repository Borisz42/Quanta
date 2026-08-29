"""Discrete Information Bottleneck profiler and entropy evaluation suite for QUANTA,
integrating translator verification, symbolic soundness, and causal reasoning utility diagnostics.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from core.slots import get_slot_names
from profiler.verifier import VerificationSummary


class QuantaInformationProfiler:
    """Evaluates channel capacity, individual slot entropy, pairwise redundancy,
    translator grounding fidelity, and causal reasoning necessity for 1024-dimensional quaternary state spaces.
    """

    def __init__(
        self,
        data_matrix: np.ndarray,
        dimension_labels: Optional[List[str]] = None,
        verification_summary: Optional[VerificationSummary] = None,
    ):
        """data_matrix: (N, D) array with values in {0, 1, 2, 3}
        dimension_labels: List of human-readable dimension names
        verification_summary: Optional verification suite output
        """
        self.X = data_matrix.astype(np.uint8)
        self.N, self.D = data_matrix.shape
        if dimension_labels is not None:
            self.labels = dimension_labels
        else:
            all_names = get_slot_names()
            if self.D <= len(all_names):
                self.labels = all_names[:self.D]
            else:
                self.labels = all_names + [f"DIM_{i}" for i in range(len(all_names), self.D)]
        self.verification = verification_summary
        assert len(self.labels) == self.D, f"Labels must contain {self.D} names, got {len(self.labels)}"

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

    def compute_joint_entropy(self) -> float:
        """Calculates empirical joint Shannon entropy H(D_0, ..., D_{D-1}) in bits across observed vectors."""
        if self.N == 0:
            return 0.0
        _, counts = np.unique(self.X, axis=0, return_counts=True)
        probs = counts / self.N
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def compute_total_correlation(self) -> float:
        """Calculates Total Correlation (Watanabe multi-information):
        TC(D) = sum_{i=0}^{D-1} H(D_i) - H(D_0, ..., D_{D-1})
        in bits. Measures total multivariate redundancy across all dimensions.
        """
        marginal_entropies = self.compute_entropies()
        sum_marginal = float(np.sum(marginal_entropies))
        joint_entropy = self.compute_joint_entropy()
        tc = sum_marginal - joint_entropy
        return float(max(0.0, tc))

    def compute_pairwise_mutual_information(
        self,
        top_k_pairs: int = 10,
        subsample_dim_pairs: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """Finds the most redundant pairs of dimensions using Mutual Information I(D_i; D_j)."""
        if self.D < 2 or self.N == 0:
            return []

        D = self.D
        N = self.N
        X_int = self.X.astype(np.int32)

        # One-hot encoding of shape (N, D, 4)
        one_hot = np.zeros((N, D, 4), dtype=np.float32)
        for v in range(4):
            one_hot[:, :, v] = (X_int == v).astype(np.float32)

        flat_one_hot = one_hot.reshape(N, D * 4)
        # Fast BLAS GEMM joint count matrix: shape (D, 4, D, 4) -> transpose to (D, D, 4, 4)
        joint_matrix = np.dot(flat_one_hot.T, flat_one_hot).reshape(D, 4, D, 4).transpose(0, 2, 1, 3)

        p_xy = joint_matrix / float(N)
        p_x = p_xy.sum(axis=3, keepdims=True)  # shape (D, D, 4, 1)
        p_y = p_xy.sum(axis=2, keepdims=True)  # shape (D, D, 1, 4)
        denom = p_x * p_y
        mask = (p_xy > 0) & (denom > 0)

        log_ratio = np.zeros_like(p_xy)
        log_ratio[mask] = p_xy[mask] * np.log2(p_xy[mask] / denom[mask])
        mi_matrix = np.maximum(0.0, np.sum(log_ratio, axis=(2, 3)))  # shape (D, D)

        # Extract upper triangle (i < j)
        tri_i, tri_j = np.triu_indices(D, k=1)
        pair_mis = mi_matrix[tri_i, tri_j]

        # Top-K selection
        if len(pair_mis) > top_k_pairs:
            top_indices = np.argpartition(pair_mis, -top_k_pairs)[-top_k_pairs:]
            top_sorted = top_indices[np.argsort(-pair_mis[top_indices])]
        else:
            top_sorted = np.argsort(-pair_mis)

        return [
            (self.labels[tri_i[idx]], self.labels[tri_j[idx]], float(pair_mis[idx]))
            for idx in top_sorted
        ]

    def compute_collision_rate(self, concept_ids: Optional[Sequence[str]] = None) -> float:
        """Measures whether two distinctly labeled concepts share the exact same vector.
        
        R_collision = |{ (a, b) | a != b and v(a) == v(b) }| / (N choose 2)
        """
        if self.N < 2:
            return 0.0

        unique_rows, counts = np.unique(self.X, axis=0, return_counts=True)
        duplicate_pairs = int(np.sum(counts * (counts - 1) // 2))
        total_pairs = (self.N * (self.N - 1)) // 2
        return float(duplicate_pairs / total_pairs) if total_pairs > 0 else 0.0

    def compute_band_entropies(self) -> Dict[str, float]:
        """Calculates average entropy for each band."""
        entropies = self.compute_entropies()
        if self.D == 1024:
            return {
                "Band 0 (NSM & Kinematics)": float(np.mean(entropies[0:128])),
                "Band 1 (Valencies & Topology)": float(np.mean(entropies[128:256])),
                "Band 2 (Logic & Variables)": float(np.mean(entropies[256:384])),
                "Band 3 (Ontology & Structures)": float(np.mean(entropies[384:512])),
                "Band 4 (Affordances & Operations)": float(np.mean(entropies[512:640])),
                "Band 5 (ToM & Pragmatics)": float(np.mean(entropies[640:768])),
                "Band 6 (Proof & Deontics)": float(np.mean(entropies[768:896])),
                "Band 7 (Spatiotemporal & Causal)": float(np.mean(entropies[896:1024])),
                "Total Mean Entropy": float(np.mean(entropies)),
            }
        elif self.D == 256:
            return {
                "Band 0 (NSM & Kinematics)": float(np.mean(entropies[0:64])),
                "Band 1 (Valencies & Topology)": float(np.mean(entropies[64:128])),
                "Band 2 (Ontology & Modality)": float(np.mean(entropies[128:192])),
                "Band 3 (Epistemics & Metarules)": float(np.mean(entropies[192:256])),
                "Total Mean Entropy": float(np.mean(entropies)),
            }
        else:
            return {
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
        joint_entropy = self.compute_joint_entropy()
        total_correlation = self.compute_total_correlation()

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

        verification_data: Dict[str, Any] = {}
        if self.verification is not None:
            verification_data = self.verification.to_dict()

        report = {
            "num_samples": self.N,
            "mean_entropy": float(np.mean(entropies)),
            "min_entropy": float(np.min(entropies)),
            "max_entropy": float(np.max(entropies)),
            "joint_entropy": joint_entropy,
            "total_correlation": total_correlation,
            "collision_rate": collision_rate,
            "band_entropies": band_stats,
            "dead_slots": dead_slots,
            "high_redundancy_pairs": high_redundancy_pairs,
            "verification": verification_data,
        }

        if verbose:
            print("=== QUANTA INFORMATION PROFILER REPORT ===")
            print(f"Total Samples Evaluated: {self.N}")
            print(f"Average Dimension Entropy: {report['mean_entropy']:.3f} / 2.000 bits")
            print(f"Joint State Entropy: {report['joint_entropy']:.3f} bits")
            print(f"Total Correlation TC(D): {report['total_correlation']:.3f} bits")
            print(f"Concept Collision Rate: {report['collision_rate']:.6f}")
            if self.verification is not None:
                print(f"Translator Macro F1 Score: {self.verification.gold_alignment.macro_f1:.3f}")
                print(f"Cycle-Consistency Fidelity: {self.verification.cycle_consistency_score:.3f}")
                print(f"Symbolic Soundness Pass Rate: {self.verification.symbolic_soundness_rate*100:.1f}%")

            print("\n--- Band-wise Mean Entropy ---")
            for band_name, val in band_stats.items():
                print(f"  {band_name}: {val:.3f} bits")

            print(f"\n--- 1. Under-Utilized / Dead Dimensions (< {dead_threshold} bits) ---")
            if not dead_slots:
                print("  None! All dimensions meet minimum entropy requirements.")
            else:
                for idx, label, h in dead_slots:
                    diag = ""
                    if self.verification and label in self.verification.dead_slots_deficiency_type:
                        diag = f" -> [{self.verification.dead_slots_deficiency_type[label]}]"
                    print(f"  Slot {idx:03d} [{label}]: Entropy = {h:.4f} bits{diag}")

            print(f"\n--- 2. High Redundancy Pairs (Mutual Information > {redundancy_threshold} bits) ---")
            if not high_redundancy_pairs:
                print("  None! No pairs exceed the redundancy threshold.")
            else:
                for dim_a, dim_b, mi in high_redundancy_pairs:
                    print(f"  [{dim_a}] <---> [{dim_b}]: MI = {mi:.4f} bits (High co-linearity; consider collapsing)")

        return report

    def export_report(
        self,
        txt_path: Union[str, Path] = "output/information_profiler_report.txt",
        json_path: Optional[Union[str, Path]] = "output/information_profiler_report.json",
        report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Exports human-readable text and JSON report artifacts."""
        if report is None:
            report = self.run_diagnostic_suite(verbose=False)

        txt_file = Path(txt_path)
        txt_file.parent.mkdir(parents=True, exist_ok=True)

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("=== QUANTA INFORMATION PROFILER REPORT ===\n")
            f.write(f"Total Samples Evaluated: {report['num_samples']}\n")
            f.write(f"Average Dimension Entropy: {report['mean_entropy']:.3f} / 2.000 bits\n")
            f.write(f"Joint State Entropy: {report.get('joint_entropy', 0.0):.3f} bits\n")
            f.write(f"Total Correlation TC(D): {report.get('total_correlation', 0.0):.3f} bits\n")
            f.write(f"Concept Collision Rate: {report['collision_rate']:.6f}\n")

            ver = report.get("verification")
            if ver and "gold_alignment" in ver:
                f.write(f"Translator Macro F1 Score: {ver['gold_alignment']['macro_f1']:.3f}\n")
                f.write(f"Cycle-Consistency Fidelity: {ver['cycle_consistency_score']:.3f}\n")
                f.write(f"Symbolic Soundness Pass Rate: {ver['symbolic_soundness_rate']*100:.1f}%\n")
                f.write(f"Causal Reasoning Necessity Mean: {ver['causal_necessity_mean']:.3f}\n")

            f.write("\n--- Band-wise Mean Entropy ---\n")
            for band_name, val in report["band_entropies"].items():
                f.write(f"  {band_name}: {val:.3f} bits\n")

            f.write(f"\n--- Under-Utilized / Dead Dimensions ---\n")
            if not report["dead_slots"]:
                f.write("  None! All dimensions meet minimum entropy requirements.\n")
            else:
                diag_map = ver.get("dead_slots_diagnosis", {}) if ver else {}
                for idx, label, h in report["dead_slots"]:
                    diag_str = f" [{diag_map[label]}]" if label in diag_map else ""
                    f.write(f"  Slot {idx:03d} [{label}]: Entropy = {h:.4f} bits{diag_str}\n")

            f.write(f"\n--- High Redundancy Pairs ---\n")
            if not report["high_redundancy_pairs"]:
                f.write("  None! No pairs exceed the redundancy threshold.\n")
            else:
                for dim_a, dim_b, mi in report["high_redundancy_pairs"]:
                    f.write(f"  [{dim_a}] <---> [{dim_b}]: MI = {mi:.4f} bits\n")

        exported_paths = {"txt": txt_file}

        if json_path is not None:
            json_file = Path(json_path)
            json_file.parent.mkdir(parents=True, exist_ok=True)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            exported_paths["json"] = json_file

        return exported_paths
