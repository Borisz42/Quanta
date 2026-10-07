"""HippoRAG 2 Personalized PageRank (PPR) Spreading Activation for QUANTA SVM (Section 6).

Implements topological spreading activation over the Abstract Syntax Graph (ASG)
via Personalized PageRank (PPR), gated by PoP-RAG Belnap truth values:
    p^{(t+1)} = (1 - alpha) W p^{(t)} + alpha p^{(0)}
where:
    - W: Column-stochastic transition matrix gated by PoP-RAG Belnap truth values.
    - alpha in [0.15, 0.20]: Restart / teleport probability.
    - p^{(0)}: Query personalization vector over seed concept/entity/event nodes.
    - epsilon = 10^{-6}: L1 convergence tolerance.

Eliminates multi-hop semantic drift by routing exclusively over grounded ontological
edges (Allen intervals, Pearl causal mechanisms, and thematic valency frames) rather
than dense embedding space projections.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

from core.asg import QuantaGraph, QuantaNode
from memory.poprag_gating import PoPRAGGating

logger = logging.getLogger("quanta.memory.hipporag_ppr")


class HippoRAGRetriever:
    """HippoRAG 2 Spreading Activation Retriever using Personalized PageRank.
    
    Traverses the ASG topology with zero semantic drift by performing SIMD/NumPy
    vectorized power iteration over a column-stochastic transition matrix modulated
    by PoP-RAG epistemic truth gates.
    """

    DEFAULT_ALLOWED_RELATIONS: Set[str] = {
        # Pearl Causal
        "CAUSAL_LEADS_TO",
        "CAUSAL_MECHANISM_LINK",
        "ENABLING_CONDITION",
        "CAUSAL_PREVENTS",
        # Allen Temporal
        "TEMP_ALLEN_MEETS",
        "TEMP_ALLEN_BEFORE",
        "TEMP_ALLEN_OVERLAPS",
        "TEMP_ALLEN_DURING",
        "TEMP_ALLEN_STARTS",
        "TEMP_ALLEN_FINISHES",
        "TEMP_ALLEN_EQUALS",
        # Thematic Valencies
        "VAL_X1_AGENT",
        "VAL_X2_PATIENT",
        "VAL_LOCATION_SLOT",
        "VAL_X5_INSTRUMENT",
        "VAL_TIME_SLOT",
        # Structural / Encyclopedic
        "CALLS",
        "INHERITS_FROM",
        "IMPLEMENTS",
        "IMPORTS",
        "CFG_NEXT",
        "DATA_FLOW_DEF_USE",
        "CO_OCCURS",
        "LOCATED_IN",
        "CROSS_CHUNK_BRIDGE",
        "EDUCATED_AT",
        "STUDIED_AT",
        "COUNTRY",
        "CAPITAL",
        "BORN_IN",
        "SHARES_BORDER",
        "PART_OF",
        "MEMBER_OF",
        "HEADQUARTERS",
        "INSTANCE_OF",
        "SUBCLASS_OF",
    }

    def __init__(
        self,
        alpha: float = 0.15,
        convergence_tol: float = 1e-6,
        max_iter: int = 50,
        gating: Optional[PoPRAGGating] = None,
        bidirectional: bool = True,
        reverse_weight_factor: float = 0.8,
        allowed_relations: Optional[Set[str]] = None,
        localized_threshold_nodes: int = 10000,
        default_hops: int = 4,
    ):
        """Initializes the HippoRAG retriever.
        
        Args:
            alpha: Restart / personalization probability (typically 0.15 to 0.20).
            convergence_tol: L1 convergence threshold epsilon (default 1e-6).
            max_iter: Maximum number of power iteration steps.
            gating: PoPRAGGating instance for Belnap truth modulation.
            bidirectional: Whether to allow reverse traversal across directed relations.
            reverse_weight_factor: Decay factor applied to reverse edges (default 0.8).
            allowed_relations: Optional relation filter set.
            localized_threshold_nodes: Graph size threshold above which localized submatrix
                PPR is automatically engaged for sub-5ms latency on massive graphs.
            default_hops: Default maximum neighborhood radius for localized retrieval.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha: float = float(alpha)
        self.convergence_tol: float = float(convergence_tol)
        self.max_iter: int = int(max_iter)
        self.gating: PoPRAGGating = gating or PoPRAGGating()
        self.bidirectional: bool = bool(bidirectional)
        self.reverse_weight_factor: float = float(reverse_weight_factor)
        self.allowed_relations: Set[str] = allowed_relations or self.DEFAULT_ALLOWED_RELATIONS
        self.localized_threshold_nodes: int = localized_threshold_nodes
        self.default_hops: int = default_hops

    def build_transition_matrix(
        self,
        graph: Union[QuantaGraph, Any],
        gating: Optional[PoPRAGGating] = None,
        query_polarity: str = "positive",
        allowed_relations: Optional[Set[str]] = None,
    ) -> Tuple[sp.csr_matrix, Dict[str, int], List[str]]:
        """Constructs the column-stochastic transition matrix W from a QuantaGraph or node dictionary.
        
        Args:
            graph: QuantaGraph instance (or object with .nodes dict/property).
            gating: PoPRAGGating instance (defaults to self.gating).
            query_polarity: 'positive' or 'negative' query assertion.
            allowed_relations: Allowed edge types (defaults to self.allowed_relations).
            
        Returns:
            Tuple of:
            - W: scipy.sparse.csr_matrix (N x N, column-stochastic where out-degree > 0)
            - node_to_idx: Dict mapping node CID -> column/row integer index
            - idx_to_node: List mapping integer index -> node CID
        """
        gate = gating or self.gating
        rel_set = allowed_relations or self.allowed_relations

        if isinstance(graph, QuantaGraph):
            nodes_dict = graph.nodes
        elif hasattr(graph, "nodes"):
            nodes_dict = graph.nodes if isinstance(graph.nodes, dict) else {n.cid: n for n in graph.nodes}
        else:
            nodes_dict = {getattr(n, "cid", str(i)): n for i, n in enumerate(graph)}

        cids = list(nodes_dict.keys())
        N = len(cids)
        node_to_idx: Dict[str, int] = {cid: idx for idx, cid in enumerate(cids)}
        idx_to_node: List[str] = cids

        if N == 0:
            empty_mat = sp.csr_matrix((0, 0), dtype=np.float32)
            return empty_mat, node_to_idx, idx_to_node

        row_indices: List[int] = []
        col_indices: List[int] = []
        data_values: List[float] = []

        for u_cid, u_node in nodes_dict.items():
            u_idx = node_to_idx[u_cid]
            if not hasattr(u_node, "edges") or not u_node.edges:
                continue

            for rel, target_cids in u_node.edges.items():
                if rel_set and rel not in rel_set:
                    continue

                for v_cid in target_cids:
                    v_node = nodes_dict.get(v_cid)
                    if v_node is None or v_cid not in node_to_idx:
                        continue
                    v_idx = node_to_idx[v_cid]

                    # Gating: modulates base weight using Belnap truth status
                    w_forward = gate.gate_edge(
                        source_node=u_node,
                        target_node=v_node,
                        relation=rel,
                        query_polarity=query_polarity,
                    )

                    if w_forward > 0.0:
                        # Forward transition: out of u (col u_idx) into v (row v_idx)
                        row_indices.append(v_idx)
                        col_indices.append(u_idx)
                        data_values.append(w_forward)

                        # Bidirectional transition: out of v (col v_idx) into u (row u_idx)
                        if self.bidirectional and u_idx != v_idx:
                            w_rev = w_forward * self.reverse_weight_factor
                            if w_rev > 0.0:
                                row_indices.append(u_idx)
                                col_indices.append(v_idx)
                                data_values.append(w_rev)

        if not data_values:
            empty_mat = sp.csr_matrix((N, N), dtype=np.float32)
            return empty_mat, node_to_idx, idx_to_node

        # Raw adjacency matrix: shape (N, N), where col j is outgoing from node j
        adj = sp.csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(N, N),
            dtype=np.float32,
        )

        # Column-normalize to make W column-stochastic:
        # Sum of col j is sum over row i of adj[i, j]
        col_sums = np.array(adj.sum(axis=0)).flatten()
        non_zero = col_sums > 0.0
        inv_sums = np.zeros(N, dtype=np.float32)
        inv_sums[non_zero] = 1.0 / col_sums[non_zero]

        diag_inv = sp.diags(inv_sums, format="csr", dtype=np.float32)
        # W = adj * diag(inv_sums) normalizes each column
        W = adj.dot(diag_inv).tocsr()

        return W, node_to_idx, idx_to_node

    def power_iteration(
        self,
        W: sp.csr_matrix,
        p0: np.ndarray,
        alpha: float,
        convergence_tol: float,
        max_iter: int,
    ) -> Tuple[np.ndarray, int, float]:
        """Performs SIMD/NumPy vectorized power iteration for Personalized PageRank.
        
        Formula:
            p^{(t+1)} = (1 - alpha) W p^{(t)} + alpha p^{(0)}
        With dangling node probability mass redistributed to p^{(0)}.
        
        Args:
            W: Column-stochastic transition matrix (CSR format).
            p0: Initial personalization probability vector (L1 norm = 1.0).
            alpha: Teleport/restart probability in (0, 1).
            convergence_tol: L1 convergence threshold epsilon.
            max_iter: Maximum number of power iteration steps.
            
        Returns:
            Tuple of:
            - p_star: Converged steady-state probability distribution.
            - num_iters: Number of iterations taken to converge.
            - delta: Final L1 norm delta.
        """
        N = W.shape[0]
        if N == 0:
            return np.zeros(0, dtype=np.float32), 0, 0.0

        p = p0.copy()
        iters = 0
        delta = 1.0

        for iters in range(1, max_iter + 1):
            # v = W * p
            v = W.dot(p)

            # Dangling mass handling: columns with 0 outgoing edges lose probability mass
            v_sum = float(np.sum(v))
            dangling_mass = max(0.0, 1.0 - v_sum)

            # p^{(t+1)} = (1 - alpha) * (v + dangling_mass * p0) + alpha * p0
            #           = (1 - alpha) * v + ((1 - alpha) * dangling_mass + alpha) * p0
            restart_weight = (1.0 - alpha) * dangling_mass + alpha
            p_next = (1.0 - alpha) * v + restart_weight * p0

            # L1 norm delta
            delta = float(np.sum(np.abs(p_next - p)))
            p = p_next

            if delta < convergence_tol:
                break

        return p, iters, delta

    def compute_ppr(
        self,
        seed_cids: Union[Sequence[str], Dict[str, float]],
        graph: Optional[QuantaGraph] = None,
        W: Optional[sp.csr_matrix] = None,
        node_to_idx: Optional[Dict[str, int]] = None,
        idx_to_node: Optional[List[str]] = None,
        alpha: Optional[float] = None,
        convergence_tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        query_polarity: str = "positive",
        mode: str = "auto",
        max_hops: Optional[int] = None,
    ) -> Dict[str, float]:
        """Computes steady-state Personalized PageRank activation scores.
        
        Args:
            seed_cids: Seed node CIDs (list of CIDs or dict mapping CID -> seed weight).
            graph: QuantaGraph instance (used if W is not precomputed).
            W: Optional precomputed transition matrix.
            node_to_idx: Optional precomputed CID -> index mapping.
            idx_to_node: Optional precomputed index -> CID mapping.
            alpha: Optional override for restart probability.
            convergence_tol: Optional override for convergence tolerance.
            max_iter: Optional override for max iterations.
            query_polarity: 'positive' or 'negative'.
            mode: 'auto', 'full', or 'localized'.
            max_hops: Radius for localized submatrix extraction (default self.default_hops).
            
        Returns:
            Dict mapping node CID -> steady-state activation probability score.
        """
        eff_alpha = alpha if alpha is not None else self.alpha
        eff_tol = convergence_tol if convergence_tol is not None else self.convergence_tol
        eff_max_iter = max_iter if max_iter is not None else self.max_iter
        eff_hops = max_hops if max_hops is not None else self.default_hops

        # 1. Build or retrieve transition matrix
        if W is None or node_to_idx is None or idx_to_node is None:
            if graph is None:
                raise ValueError("Must provide either precomputed W or a QuantaGraph instance")
            W, node_to_idx, idx_to_node = self.build_transition_matrix(
                graph=graph,
                query_polarity=query_polarity,
            )

        N = W.shape[0]
        if N == 0:
            return {}

        # 2. Parse seed personalization vector p^{(0)}
        p0 = np.zeros(N, dtype=np.float32)
        valid_seed_indices: List[int] = []

        if isinstance(seed_cids, dict):
            for s_cid, s_wt in seed_cids.items():
                if s_cid in node_to_idx:
                    idx = node_to_idx[s_cid]
                    p0[idx] = max(0.0, float(s_wt))
                    valid_seed_indices.append(idx)
        else:
            for s_cid in seed_cids:
                if s_cid in node_to_idx:
                    idx = node_to_idx[s_cid]
                    p0[idx] = 1.0
                    valid_seed_indices.append(idx)

        total_seed_wt = float(np.sum(p0))
        if total_seed_wt > 0.0:
            p0 /= total_seed_wt
        else:
            # Fallback to uniform distribution if no valid seeds found
            p0.fill(1.0 / N)
            valid_seed_indices = list(range(min(N, 10)))

        # 3. Determine execution strategy: Localized submatrix for massive graphs
        use_localized = (
            (mode == "localized")
            or (mode == "auto" and N > self.localized_threshold_nodes and len(valid_seed_indices) < N // 2)
        )

        if use_localized and valid_seed_indices:
            # Extract k-hop reachable component from seeds along active non-zero edges
            visited: Set[int] = set(valid_seed_indices)
            frontier: List[int] = list(valid_seed_indices)

            for _ in range(eff_hops):
                if not frontier:
                    break
                next_frontier: List[int] = []
                for u in frontier:
                    start_ptr = W.indptr[u]
                    end_ptr = W.indptr[u + 1]
                    for v in W.indices[start_ptr:end_ptr]:
                        if v not in visited:
                            visited.add(v)
                            next_frontier.append(v)
                frontier = next_frontier

            sub_indices = np.array(sorted(visited), dtype=np.int32)
            M = len(sub_indices)

            # Submatrix slicing
            W_sub = W[sub_indices, :][:, sub_indices]
            # Normalize submatrix columns to ensure column-stochasticity on the active component
            sub_col_sums = np.array(W_sub.sum(axis=0)).flatten()
            sub_non_zero = sub_col_sums > 0.0
            sub_inv_sums = np.zeros(M, dtype=np.float32)
            sub_inv_sums[sub_non_zero] = 1.0 / sub_col_sums[sub_non_zero]
            W_sub = W_sub.dot(sp.diags(sub_inv_sums, format="csr", dtype=np.float32)).tocsr()

            sub_idx_map = {orig_idx: local_idx for local_idx, orig_idx in enumerate(sub_indices)}
            p0_sub = np.zeros(M, dtype=np.float32)
            for s_idx in valid_seed_indices:
                if s_idx in sub_idx_map:
                    p0_sub[sub_idx_map[s_idx]] = p0[s_idx]
            sub_p0_sum = float(np.sum(p0_sub))
            if sub_p0_sum > 0.0:
                p0_sub /= sub_p0_sum
            else:
                p0_sub.fill(1.0 / M)

            p_star_sub, _, _ = self.power_iteration(
                W=W_sub,
                p0=p0_sub,
                alpha=eff_alpha,
                convergence_tol=eff_tol,
                max_iter=eff_max_iter,
            )

            # Return scores for active nodes
            scores: Dict[str, float] = {}
            for local_idx, orig_idx in enumerate(sub_indices):
                score_val = float(p_star_sub[local_idx])
                if score_val > 0.0:
                    scores[idx_to_node[orig_idx]] = score_val
            return scores

        # 4. Standard full vectorized power iteration
        p_star, iters_taken, final_delta = self.power_iteration(
            W=W,
            p0=p0,
            alpha=eff_alpha,
            convergence_tol=eff_tol,
            max_iter=eff_max_iter,
        )

        scores: Dict[str, float] = {}
        for idx, score_val in enumerate(p_star):
            s_val = float(score_val)
            if s_val > 0.0:
                scores[idx_to_node[idx]] = s_val

        return scores

    def retrieve_top_nodes(
        self,
        seed_cids: Union[Sequence[str], Dict[str, float]],
        graph: QuantaGraph,
        top_k: int = 10,
        threshold: float = 0.001,
        query_polarity: str = "positive",
    ) -> List[Tuple[str, float]]:
        """Retrieves the top-K activated semantic nodes sorted by PPR activation score.
        
        Args:
            seed_cids: Seed node CIDs.
            graph: QuantaGraph instance.
            top_k: Maximum number of nodes to return.
            threshold: Minimum activation score cutoff.
            query_polarity: 'positive' or 'negative'.
            
        Returns:
            List of (node_cid, activation_score) tuples sorted in descending score order.
        """
        scores = self.compute_ppr(
            seed_cids=seed_cids,
            graph=graph,
            query_polarity=query_polarity,
        )

        filtered = [
            (cid, score) for cid, score in scores.items()
            if score >= threshold
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_k]

    def retrieve_subgraph(
        self,
        seed_cids: Union[Sequence[str], Dict[str, float]],
        graph: QuantaGraph,
        top_k: Optional[int] = None,
        threshold: float = 0.001,
        query_polarity: str = "positive",
        preserve_edges: bool = True,
    ) -> QuantaGraph:
        """Retrieves the activated sub-graph induced by top PPR-activated nodes.
        
        Args:
            seed_cids: Seed node CIDs.
            graph: QuantaGraph instance.
            top_k: Optional maximum node count cutoff.
            threshold: Minimum activation probability threshold.
            query_polarity: 'positive' or 'negative'.
            preserve_edges: Whether to copy interconnecting edges among admitted nodes.
            
        Returns:
            New QuantaGraph containing only admitted nodes with updated salience scores
            and interconnecting edges.
        """
        scores = self.compute_ppr(
            seed_cids=seed_cids,
            graph=graph,
            query_polarity=query_polarity,
        )

        admitted_cids = {
            cid for cid, score in scores.items()
            if score >= threshold
        }

        if top_k is not None and len(admitted_cids) > top_k:
            ranked = sorted(admitted_cids, key=lambda c: scores[c], reverse=True)
            admitted_cids = set(ranked[:top_k])

        subgraph = QuantaGraph()
        node_map: Dict[str, QuantaNode] = {}

        # 1. Clone admitted nodes and attach activation salience
        for cid in admitted_cids:
            orig_node = graph.get_node(cid)
            if orig_node is None:
                continue

            cloned = orig_node.clone()
            cloned.salience = float(scores.get(cid, orig_node.salience))
            setattr(cloned, "_ppr_activation", float(scores.get(cid, 0.0)))
            subgraph.add_node(cloned)
            node_map[cid] = cloned

        # Set root if available
        if graph.root_cid and graph.root_cid in node_map:
            subgraph.root_cid = graph.root_cid
        elif admitted_cids:
            # Root as highest activation node
            best_cid = max(admitted_cids, key=lambda c: scores.get(c, 0.0))
            subgraph.root_cid = best_cid

        # 2. Filter edges to only interconnect admitted nodes
        if preserve_edges:
            for cid, cloned_node in node_map.items():
                orig_node = graph.get_node(cid)
                if orig_node is None:
                    continue
                filtered_edges: Dict[str, List[str]] = {}
                for rel, targets in orig_node.edges.items():
                    kept_targets = [t for t in targets if t in admitted_cids]
                    if kept_targets:
                        filtered_edges[rel] = kept_targets
                cloned_node.edges = filtered_edges

        return subgraph


__all__ = [
    "HippoRAGRetriever",
]
