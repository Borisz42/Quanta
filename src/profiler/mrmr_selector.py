"""Multi-Objective Minimal Redundancy Maximal Relevance (mRMR) dimension selector for discrete quaternary state spaces."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def discrete_entropy(x: np.ndarray, num_states: int = 4) -> float:
    """Computes Shannon entropy H(X) in bits for a 1D discrete array."""
    counts = np.bincount(x.astype(int), minlength=num_states)
    probs = counts / len(x)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def fast_discrete_mutual_info(x: np.ndarray, y: np.ndarray, num_x: int = 4, num_y: int = 4) -> float:
    """Computes Shannon mutual information I(X; Y) in bits using fast vectorized flat bin indexing."""
    n = len(x)
    flat_indices = x.astype(np.int32) * num_y + y.astype(np.int32)
    joint_counts = np.bincount(flat_indices, minlength=num_x * num_y).reshape((num_x, num_y)).astype(np.float64)
    
    p_xy = joint_counts / n
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    
    denom = p_x @ p_y
    mask = (p_xy > 0) & (denom > 0)
    
    mi = np.sum(p_xy[mask] * np.log2(p_xy[mask] / denom[mask]))
    return float(max(0.0, mi))


class MRMRSelector:
    """Selects an optimal subset of dimensions using Multi-Objective Information Utility Optimization."""

    def __init__(self, candidate_names: Sequence[str]):
        self.candidate_names = list(candidate_names)
        self.K = len(self.candidate_names)

    def select_dimensions_multi_objective(
        self,
        X: np.ndarray,
        f1_scores: Optional[np.ndarray] = None,
        causal_necessity: Optional[np.ndarray] = None,
        violation_rates: Optional[np.ndarray] = None,
        num_to_select: int = 256,
        w_f1: float = 0.35,
        w_entropy: float = 0.25,
        w_necessity: float = 0.20,
        w_redundancy: float = 0.15,
        w_violation: float = 0.05,
    ) -> List[Tuple[int, str, float]]:
        """Selects optimal dimensions using the Universal Neuro-Symbolic Information Utility function:
        U(D_i) = w_f1 * F1(D_i) + w_entropy * H(D_i) + w_nec * Nec(D_i) - w_red * Red(D_i) - w_viol * Viol(D_i)
        """
        N, K = X.shape
        assert K == self.K, f"Matrix shape mismatch: expected {self.K} features, got {K}"
        target_k = min(num_to_select, K)
        X_int = X.astype(np.int32)

        # 1. Compute individual entropies H(D_i)
        entropies = np.zeros(K, dtype=np.float64)
        for i in range(K):
            entropies[i] = discrete_entropy(X_int[:, i], num_states=4)
        # Normalize entropy to [0, 1] relative to max possible 2.0 bits
        norm_entropies = np.clip(entropies / 2.0, 0.0, 1.0)

        # 2. Defaults for auxiliary signals
        if f1_scores is None:
            # If no gold F1 provided, use normalized entropy as baseline fidelity
            f1_scores = norm_entropies
        elif len(f1_scores) < K:
            padded_f1 = np.zeros(K, dtype=np.float64)
            padded_f1[:len(f1_scores)] = f1_scores
            # For projected candidate dimensions beyond canonical, approximate F1 from their source slots
            padded_f1[len(f1_scores):] = norm_entropies[len(f1_scores):]
            f1_scores = padded_f1
        elif len(f1_scores) > K:
            f1_scores = f1_scores[:K]

        if causal_necessity is None:
            causal_necessity = norm_entropies
        elif len(causal_necessity) < K:
            padded_nec = np.zeros(K, dtype=np.float64)
            padded_nec[:len(causal_necessity)] = causal_necessity
            padded_nec[len(causal_necessity):] = norm_entropies[len(causal_necessity):]
            causal_necessity = padded_nec
        elif len(causal_necessity) > K:
            causal_necessity = causal_necessity[:K]

        if violation_rates is None:
            violation_rates = np.zeros(K, dtype=np.float64)
        elif len(violation_rates) < K:
            padded_v = np.zeros(K, dtype=np.float64)
            padded_v[:len(violation_rates)] = violation_rates
            violation_rates = padded_v
        elif len(violation_rates) > K:
            violation_rates = violation_rates[:K]

        # 3. Base static relevance score per candidate
        base_relevance = (
            w_f1 * f1_scores +
            w_entropy * norm_entropies +
            w_necessity * causal_necessity -
            w_violation * violation_rates
        )

        # Precompute one-hot matrix for ultra-fast vectorized joint count & MI calculation
        one_hot = np.zeros((N, K, 4), dtype=np.float32)
        for v in range(4):
            one_hot[:, :, v] = (X_int == v).astype(np.float32)
        flat_one_hot = one_hot.reshape(N, K * 4)

        def compute_all_mi_with(feat_idx: int) -> np.ndarray:
            # (4, N) @ (N, K * 4) -> (4, K * 4) -> (K, 4, 4)
            joint_counts = np.dot(one_hot[:, feat_idx, :].T, flat_one_hot).reshape(4, K, 4).transpose(1, 0, 2)
            p_xy = joint_counts / float(N)
            p_x = p_xy.sum(axis=2, keepdims=True)  # (K, 4, 1)
            p_y = p_xy.sum(axis=1, keepdims=True)  # (K, 1, 4)
            denom = p_x * p_y
            mask = (p_xy > 0) & (denom > 0)
            log_terms = np.zeros_like(p_xy)
            log_terms[mask] = p_xy[mask] * np.log2(p_xy[mask] / denom[mask])
            return np.maximum(0.0, np.sum(log_terms, axis=(1, 2)))

        selected_indices: List[int] = []
        selected_scores: List[float] = []
        remaining_mask = np.ones(K, dtype=bool)

        # Pick first feature: highest base relevance
        first_feat = int(np.argmax(base_relevance))
        selected_indices.append(first_feat)
        selected_scores.append(float(base_relevance[first_feat]))
        remaining_mask[first_feat] = False

        # Running sum of redundancies to selected features for fast vectorized O(1) update
        sum_redundancy = compute_all_mi_with(first_feat)

        while len(selected_indices) < target_k and np.any(remaining_mask):
            n_sel = len(selected_indices)
            mean_red = sum_redundancy / float(n_sel)
            norm_red = mean_red / 2.0
            scores = base_relevance - w_redundancy * norm_red
            scores[~remaining_mask] = -np.inf

            best_feat = int(np.argmax(scores))
            best_score = float(scores[best_feat])

            if scores[best_feat] == -np.inf:
                break

            selected_indices.append(best_feat)
            selected_scores.append(best_score)
            remaining_mask[best_feat] = False

            sum_redundancy += compute_all_mi_with(best_feat)

        return [
            (idx, self.candidate_names[idx], score)
            for idx, score in zip(selected_indices, selected_scores)
        ]

    def select_dimensions(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        num_to_select: int = 256,
        alpha_redundancy: float = 0.5,
    ) -> List[Tuple[int, str, float]]:
        """Backward-compatible selection method using Multi-Objective utility or relevance target."""
        if y is None or len(np.unique(y)) <= 1:
            return self.select_dimensions_multi_objective(
                X=X,
                num_to_select=num_to_select,
                w_redundancy=alpha_redundancy,
            )

        # If y provided, compute mutual info I(X; Y)
        N, K = X.shape
        num_y_classes = int(np.max(y)) + 1
        X_int = X.astype(np.int32)
        y_int = y.astype(np.int32)

        relevance = np.zeros(K, dtype=np.float64)
        for i in range(K):
            relevance[i] = fast_discrete_mutual_info(X_int[:, i], y_int, num_x=4, num_y=num_y_classes)

        norm_rel = np.clip(relevance / (np.max(relevance) + 1e-8), 0.0, 1.0)
        return self.select_dimensions_multi_objective(
            X=X,
            f1_scores=norm_rel,
            num_to_select=num_to_select,
            w_redundancy=alpha_redundancy,
        )


def export_optimal_dimensions(
    selected_dims: List[Tuple[int, str, float]],
    candidates: Sequence[Any],
    json_path: Union[str, Path] = "output/optimal_dimensions.json",
    csv_path: Optional[Union[str, Path]] = "output/optimal_dimensions.csv",
) -> Dict[str, Path]:
    """Exports multi-objective selected optimal dimensions to JSON and CSV formats."""
    cand_by_name = {c.name: c for c in candidates}
    cand_by_id = {c.id: c for c in candidates}

    json_file = Path(json_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    dims_data = []
    for rank, (cand_idx, name, score) in enumerate(selected_dims, start=1):
        cand_obj = cand_by_name.get(name) or cand_by_id.get(cand_idx)
        dims_data.append({
            "rank": rank,
            "id": getattr(cand_obj, "id", cand_idx),
            "name": name,
            "source": getattr(cand_obj, "source", "Unknown"),
            "category": getattr(cand_obj, "category", "General"),
            "description": getattr(cand_obj, "description", name),
            "mrmr_score": float(score),
            "utility_score": float(score),
        })

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(dims_data, f, indent=2)

    result_paths = {"json": json_file}

    if csv_path is not None:
        import csv
        csv_file = Path(csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Candidate_ID", "Name", "Source", "Category", "MRMR_Score", "Description"])
            for item in dims_data:
                writer.writerow([
                    item["rank"],
                    item["id"],
                    item["name"],
                    item["source"],
                    item["category"],
                    f"{item['mrmr_score']:.6f}",
                    item["description"],
                ])
        result_paths["csv"] = csv_file

    return result_paths
