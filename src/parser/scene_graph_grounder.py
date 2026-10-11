"""Multimodal Scene Graph Grounder for QUANTA (§Session 5, exp-037f).

Connects continuous visual perception crops and diagram visual features directly
to QUANTA's discrete Band 1/2 AST/entity slots, Band 3/4 ConceptNet archetypes,
and Band 7 RCC-8 (Region Connection Calculus) spatial mereotopology predicates.

Configurable parameter:
- `scene.visual_archetype_threshold`: Similarity threshold theta_vis for visual archetype crop mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from config.multi_scale_config import MultiScaleConfig
from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, QuaternaryValue
from parser.lexical_grounder import WordNetLexicalGrounder, LEXNAME_TO_WN_ROOT_SLOT
from parser.mmap_grounder import MmapLexicalGrounder
from server.embedding_manager import EmbeddingManager

logger = logging.getLogger("quanta.parser.scene_graph")

# RCC-8 Predicates in Quanta Slots (Band 7)
RCC8_PREDICATES = {
    "EQ": "SPATIAL_RCC_CONGRUENT_EQ",          # Geometrically congruent regions (EQ)
    "NTPP": "SPATIAL_RCC_NON_TANG_PART",       # Non-tangential proper interior part (NTPP)
    "TPP": "SPATIAL_RCC_TANGENTIAL_PART",      # Tangential proper part (TPP)
    "NTPPi": "SPATIAL_RCC_NON_TANG_PART_INV",  # Non-tangential proper part inverse (NTPPi)
    "TPPi": "SPATIAL_RCC_TANG_PART_INV",       # Tangential proper part inverse (TPPi)
    "PO": "SPATIAL_RCC_PARTIAL_OVERLAP",       # Partially overlapping regions (PO)
    "EC": "SPATIAL_RCC_EXT_CONNECTED",         # Externally connected / boundary touching (EC)
    "DC": "SPATIAL_RCC_DISCONNECTED",          # Disconnected / disjoint regions (DC)
}


@dataclass
class VisualCrop:
    """Represents a visual perception region / bounding box crop."""
    crop_id: str
    label: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized in [0.0, 1.0]
    dense_vector: Optional[np.ndarray] = None
    confidence: float = 1.0
    description: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


def compute_rcc8_relation(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
    tolerance: float = 1e-4,
) -> str:
    """Computes Region Connection Calculus 8 (RCC-8) spatial predicate between two bounding boxes.

    Returns one of:
    - 'SPATIAL_RCC_CONGRUENT_EQ'      (EQ: Geometrically congruent / identical)
    - 'SPATIAL_RCC_NON_TANG_PART'     (NTPP: A is strictly interior inside B without touching boundary)
    - 'SPATIAL_RCC_TANGENTIAL_PART'    (TPP: A is proper part of B, touching at least one boundary)
    - 'SPATIAL_RCC_NON_TANG_PART_INV' (NTPPi: B is strictly interior inside A)
    - 'SPATIAL_RCC_TANG_PART_INV'     (TPPi: B is proper part of A, touching at least one boundary)
    - 'SPATIAL_RCC_PARTIAL_OVERLAP'   (PO: A and B partially overlap)
    - 'SPATIAL_RCC_EXT_CONNECTED'     (EC: Externally connected / boundary touching, interior disjoint)
    - 'SPATIAL_RCC_DISCONNECTED'      (DC: Disconnected / disjoint, no boundary contact)
    """
    x1a, y1a, x2a, y2a = bbox_a
    x1b, y1b, x2b, y2b = bbox_b

    # Ensure canonical coordinate ordering (x1 <= x2, y1 <= y2)
    if x1a > x2a:
        x1a, x2a = x2a, x1a
    if y1a > y2a:
        y1a, y2a = y2a, y1a
    if x1b > x2b:
        x1b, x2b = x2b, x1b
    if y1b > y2b:
        y1b, y2b = y2b, y1b

    # 1. Congruent Equal (EQ)
    if (
        abs(x1a - x1b) <= tolerance
        and abs(y1a - y1b) <= tolerance
        and abs(x2a - x2b) <= tolerance
        and abs(y2a - y2b) <= tolerance
    ):
        return RCC8_PREDICATES["EQ"]

    # Overlap intervals in X and Y
    ix1 = max(x1a, x1b)
    ix2 = min(x2a, x2b)
    iy1 = max(y1a, y1b)
    iy2 = min(y2a, y2b)

    dx = ix2 - ix1
    dy = iy2 - iy1

    # Disconnected in at least one dimension
    if dx < -tolerance or dy < -tolerance:
        return RCC8_PREDICATES["DC"]

    # Externally Connected: boundaries touch, zero-area intersection
    if (abs(dx) <= tolerance and dy >= -tolerance) or (abs(dy) <= tolerance and dx >= -tolerance):
        return RCC8_PREDICATES["EC"]

    # Both dimensions have positive overlap -> Interior regions overlap (dx > tolerance and dy > tolerance)
    # Check if A is a subset of B (A in B)
    a_in_b = (
        x1b <= x1a + tolerance
        and x2a <= x2b + tolerance
        and y1b <= y1a + tolerance
        and y2a <= y2b + tolerance
    )

    if a_in_b:
        # Check if A touches boundary of B
        touches = (
            abs(x1a - x1b) <= tolerance
            or abs(x2a - x2b) <= tolerance
            or abs(y1a - y1b) <= tolerance
            or abs(y2a - y2b) <= tolerance
        )
        if touches:
            return RCC8_PREDICATES["TPP"]
        return RCC8_PREDICATES["NTPP"]

    # Check if B is a subset of A (B in A)
    b_in_a = (
        x1a <= x1b + tolerance
        and x2b <= x2a + tolerance
        and y1a <= y1b + tolerance
        and y2b <= y2a + tolerance
    )

    if b_in_a:
        # Check if B touches boundary of A
        touches = (
            abs(x1b - x1a) <= tolerance
            or abs(x2b - x2a) <= tolerance
            or abs(y1b - y1a) <= tolerance
            or abs(y2b - y2a) <= tolerance
        )
        if touches:
            return RCC8_PREDICATES["TPPi"]
        return RCC8_PREDICATES["NTPPi"]

    # Partial Overlap (PO)
    return RCC8_PREDICATES["PO"]


def compute_directional_relations(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
    deadband: float = 0.05,
) -> List[str]:
    """Computes qualitative directional spatial relations from A to B based on centroids.

    In standard normalized image coordinates (origin at top-left, y increasing downwards):
    - cb_y < ca_y - deadband: B is Above A (NSM_ABOVE, DIR_3D_SUPERIOR_ABOVE_Z)
    - cb_y > ca_y + deadband: B is Below A (NSM_BELOW, DIR_3D_INFERIOR_BELOW_Z)
    - cb_x < ca_x - deadband: B is Left of A (DIR_3D_SINISTRAL_LEFT_X)
    - cb_x > ca_x + deadband: B is Right of A (DIR_3D_DEXTRAL_RIGHT_X)
    """
    ca_x = (bbox_a[0] + bbox_a[2]) / 2.0
    ca_y = (bbox_a[1] + bbox_a[3]) / 2.0
    cb_x = (bbox_b[0] + bbox_b[2]) / 2.0
    cb_y = (bbox_b[1] + bbox_b[3]) / 2.0

    rels = []
    # Vertical relations
    if cb_y < ca_y - deadband:
        rels.append("NSM_ABOVE")
        rels.append("DIR_3D_SUPERIOR_ABOVE_Z")
    elif cb_y > ca_y + deadband:
        rels.append("NSM_BELOW")
        rels.append("DIR_3D_INFERIOR_BELOW_Z")

    # Horizontal relations
    if cb_x < ca_x - deadband:
        rels.append("DIR_3D_SINISTRAL_LEFT_X")
    elif cb_x > ca_x + deadband:
        rels.append("DIR_3D_DEXTRAL_RIGHT_X")

    # Proximity relation
    dist = np.hypot(cb_x - ca_x, cb_y - ca_y)
    if dist < 0.25:
        rels.append("NSM_NEAR")
    elif dist > 0.65:
        rels.append("NSM_FAR")

    return rels


@dataclass
class GroundedVisualObject:
    """Represents a grounded visual entity in the Quanta neuro-symbolic representation."""
    crop: VisualCrop
    node: QuantaNode
    matched_archetype_slot: Optional[str]
    archetype_similarity: float
    wn_synset: Optional[str]
    wn_root_slot: Optional[str]


class SceneGraphGrounder:
    """Multimodal bridge mapping visual perception crops directly into ConceptNet archetypes

    and RCC-8 spatial predicates using calibrated parameters (§Session 5).
    """

    def __init__(
        self,
        config: Optional[MultiScaleConfig] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        mmap_grounder: Optional[MmapLexicalGrounder] = None,
        wordnet_grounder: Optional[WordNetLexicalGrounder] = None,
        visual_threshold: Optional[float] = None,
    ):
        self.config = config or MultiScaleConfig()
        self.embedding_manager = embedding_manager
        self.mmap_grounder = mmap_grounder or MmapLexicalGrounder()
        self.wordnet_grounder = wordnet_grounder or WordNetLexicalGrounder()
        self.visual_threshold = (
            float(visual_threshold)
            if visual_threshold is not None
            else float(self.config.get("scene.visual_archetype_threshold", 0.65))
        )

    def ground_crop(
        self,
        crop: VisualCrop,
        threshold: Optional[float] = None,
    ) -> GroundedVisualObject:
        """Grounds a visual crop into ConceptNet archetype slots and WordNet synsets.

        If dense_vector is not attached to crop, generates it using EmbeddingManager.
        Projects dense_vector against the ConceptNet archetype matrix and constructs
        a QuantaNode anchored to the crop.
        """
        th = threshold if threshold is not None else self.visual_threshold

        # 1. Ensure dense vector representation
        vec = crop.dense_vector
        if vec is None and self.embedding_manager is not None:
            text = crop.description if crop.description else f"visual crop: {crop.label}"
            vec = self.embedding_manager.embed_document(text)
            crop.dense_vector = vec

        matched_slot: Optional[str] = None
        sim_score: float = 0.0

        # 2. Project onto ConceptNet archetype matrix if available
        if vec is not None and self.mmap_grounder.has_archetypes():
            q_vec = np.asarray(vec, dtype=np.float32).flatten()
            if q_vec.shape[0] != self.mmap_grounder._archetype_dim:
                q_vec = q_vec[: self.mmap_grounder._archetype_dim]

            norm = float(np.linalg.norm(q_vec))
            if norm > 1e-12:
                q_vec = q_vec / norm

            # Dot-product against (256, D) archetype matrix
            scores = np.dot(self.mmap_grounder._archetype_mmap, q_vec)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score >= th:
                sim_score = best_score
                if best_idx < len(self.mmap_grounder._slot_names):
                    matched_slot = self.mmap_grounder._slot_names[best_idx]
                else:
                    matched_slot = f"CN_Q{best_idx+1:03d}"

        # 3. Ground against WordNet noun ontology
        wn_concept = None
        wn_synset_name = None
        wn_root_slot = None
        try:
            syn_id = self.wordnet_grounder.resolve_synset(crop.label, pos="n")
            if syn_id:
                wn_concept = self.wordnet_grounder.ground_synset(syn_id)
                if wn_concept:
                    wn_synset_name = wn_concept.synset_name
                    # Try lexname from synset
                    syn_key = syn_id[3:] if syn_id.startswith("wn:") else syn_id
                    try:
                        import nltk.corpus
                        s = nltk.corpus.wordnet.synset(syn_key)
                        lex = s.lexname()
                        if lex in LEXNAME_TO_WN_ROOT_SLOT:
                            wn_root_slot = LEXNAME_TO_WN_ROOT_SLOT[lex]
                    except Exception:
                        pass

                    # Fallback to active_slots
                    if not wn_root_slot:
                        for s_name in wn_concept.active_slots:
                            if s_name.startswith("WN_"):
                                wn_root_slot = s_name
                                break
                    # Categorical human ontological mapping
                    if not wn_root_slot and ("TYPE_HUMAN" in wn_concept.active_slots or crop.label in ("person", "human")):
                        wn_root_slot = "WN_PERSON_HUMAN"
        except Exception as e:
            logger.debug("WordNet grounding failed for %s: %s", crop.label, e)

        # 4. Construct content-addressed QuantaNode
        q_node = QuantaNode(
            anchor=f"vis:{crop.crop_id}",
            literal=crop.label,
            dense_vector=vec,
            confidence=crop.confidence,
        )
        # Spatial region marker
        q_node.set_slot("TYPE_SPATIAL_REGION", 1)
        q_node.set_slot("VAL_LOCATION_SLOT", 1)
        q_node.set_slot("GRAPH_LEAF", 1)

        # Ontological signatures from WordNet
        if wn_root_slot:
            q_node.set_slot(wn_root_slot, 1)

        # ConceptNet archetype slot
        if matched_slot:
            try:
                q_node.set_slot(matched_slot, 1)
            except Exception:
                pass

        # Belnap truth status
        q_node.truth_status = "TRUE"

        return GroundedVisualObject(
            crop=crop,
            node=q_node,
            matched_archetype_slot=matched_slot,
            archetype_similarity=sim_score,
            wn_synset=wn_synset_name,
            wn_root_slot=wn_root_slot,
        )

    def build_scene_graph(
        self,
        crops: List[VisualCrop],
        include_directional: bool = True,
    ) -> QuantaGraph:
        """Builds a unified neuro-symbolic QuantaGraph from detected visual crops.

        - Each crop becomes a QuantaNode with grounded ConceptNet and WordNet slots.
        - Pairwise spatial mereotopology edges are computed via RCC-8 and added between nodes.
        - Directional spatial relations (above, below, left, right) are added as secondary edges.
        - Root node 'scene:root' encapsulates the composite visual scene.
        """
        graph = QuantaGraph()
        scene_root = QuantaNode(
            anchor="scene:root",
            literal="visual_scene_root",
        )
        scene_root.set_slot("GRAPH_ROOT_NODE", 1)
        scene_root.set_slot("GRAPH_SCOPED_CONTEXT", 1)
        scene_root.set_slot("TYPE_SPATIAL_REGION", 1)
        graph.add_node(scene_root)
        graph.root_cid = scene_root.cid

        grounded_objects: List[GroundedVisualObject] = []
        for crop in crops:
            g_obj = self.ground_crop(crop)
            grounded_objects.append(g_obj)
            graph.add_node(g_obj.node)
            # Link root to each visual entity
            graph.add_edge(scene_root, "GRAPH_IS_SUB_EXP", g_obj.node)

        # Pairwise Spatial Relations (RCC-8 and Directional)
        num_crops = len(grounded_objects)
        for i in range(num_crops):
            g_a = grounded_objects[i]
            for j in range(num_crops):
                if i == j:
                    continue
                g_b = grounded_objects[j]
                rcc8_rel = compute_rcc8_relation(g_a.crop.bbox, g_b.crop.bbox)
                graph.add_edge(g_a.node, rcc8_rel, g_b.node)

                if include_directional:
                    dir_rels = compute_directional_relations(g_a.crop.bbox, g_b.crop.bbox)
                    for dir_rel in dir_rels:
                        graph.add_edge(g_a.node, dir_rel, g_b.node)

        return graph


__all__ = [
    "VisualCrop",
    "compute_rcc8_relation",
    "compute_directional_relations",
    "GroundedVisualObject",
    "SceneGraphGrounder",
    "RCC8_PREDICATES",
]
