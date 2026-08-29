"""Minimal Redundancy Maximal Relevance (mRMR) dimension selector for discrete quaternary state spaces."""

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
    """Selects an optimal subset of dimensions using mRMR optimization."""

    def __init__(self, candidate_names: Sequence[str]):
        self.candidate_names = list(candidate_names)
        self.K = len(self.candidate_names)

    def select_dimensions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_to_select: int = 256,
        alpha_redundancy: float = 1.0,
    ) -> List[Tuple[int, str, float]]:
        """Executes forward greedy mRMR selection.
        
        Args:
            X: (N, K) array of discrete candidate slot activations in {0, 1, 2, 3}.
            y: (N,) target class/topic semantic vector.
            num_to_select: Number of dimensions to select (e.g. 256).
            alpha_redundancy: Scaling weight for the redundancy penalty term.
            
        Returns:
            List of (selected_index, feature_name, mrmr_score).
        """
        N, K = X.shape
        assert K == self.K, f"Matrix shape mismatch: expected {self.K} features, got {K}"
        target_k = min(num_to_select, K)

        num_y_classes = int(np.max(y)) + 1
        X_int = X.astype(np.int32)
        y_int = y.astype(np.int32)

        # 1. Compute relevance I(f_i; Y) for each candidate feature
        relevance = np.zeros(K, dtype=np.float64)
        for i in range(K):
            relevance[i] = fast_discrete_mutual_info(X_int[:, i], y_int, num_x=4, num_y=num_y_classes)

        # Precompute pairwise mutual information cache on demand
        mi_cache: Dict[Tuple[int, int], float] = {}

        def get_mi(i: int, j: int) -> float:
            if i == j:
                return discrete_entropy(X_int[:, i], num_states=4)
            key = (min(i, j), max(i, j))
            if key not in mi_cache:
                mi_cache[key] = fast_discrete_mutual_info(X_int[:, i], X_int[:, j], num_x=4, num_y=4)
            return mi_cache[key]

        selected_indices: List[int] = []
        selected_scores: List[float] = []
        remaining_indices = set(range(K))

        # First feature: highest relevance
        first_feat = int(np.argmax(relevance))
        selected_indices.append(first_feat)
        selected_scores.append(float(relevance[first_feat]))
        remaining_indices.remove(first_feat)

        # Greedily select subsequent features
        # Keep running sum of redundancies to selected features for fast O(1) update
        # sum_redundancy[cand] = sum_{s in selected} I(cand; s)
        sum_redundancy = np.zeros(K, dtype=np.float64)
        for cand in remaining_indices:
            sum_redundancy[cand] = get_mi(cand, first_feat)

        while len(selected_indices) < target_k and remaining_indices:
            n_sel = len(selected_indices)
            best_feat = -1
            best_score = -float("inf")

            for cand in remaining_indices:
                mean_red = sum_redundancy[cand] / n_sel
                score = relevance[cand] - alpha_redundancy * mean_red

                if score > best_score:
                    best_score = score
                    best_feat = cand

            if best_feat == -1:
                break

            selected_indices.append(best_feat)
            selected_scores.append(float(best_score))
            remaining_indices.remove(best_feat)

            # Update running sum of redundancies with the newly added feature
            for cand in remaining_indices:
                sum_redundancy[cand] += get_mi(cand, best_feat)

        return [
            (idx, self.candidate_names[idx], score)
            for idx, score in zip(selected_indices, selected_scores)
        ]


def export_optimal_dimensions(
    selected_dims: List[Tuple[int, str, float]],
    candidates: Sequence[Any],
    json_path: Union[str, Path] = "output/optimal_256_dimensions.json",
    csv_path: Optional[Union[str, Path]] = "output/optimal_256_dimensions.csv",
) -> Dict[str, Path]:
    """Exports mRMR selected optimal dimensions to JSON and CSV formats."""
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

