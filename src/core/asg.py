"""Abstract Syntax Graph (ASG) and Content-Addressed Merkle nodes for QUANTA."""

from __future__ import annotations
from collections.abc import MutableMapping
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import blake3

from core.types import (
    QuantaVector,
    QuaternaryValue,
    EpistemicValue,
    StructuralValue,
    RoutingValue,
    RegisterValue,
    BandContract,
)
from core.slots import SLOT_NAME_TO_INDEX, get_slot_by_name, get_slot_contract


def _blake3_hash(data: bytes) -> str:
    """Computes a 256-bit BLAKE3 hash in hexadecimal format."""
    return blake3.blake3(data).hexdigest()


class QuantaNode:
    """Atomic Abstract Syntax Graph node with a 1024-dim quaternary vector,

    directed relation edges, and optional lexical/literal payloads.
    """

    __slots__ = (
        "vector",
        "edges",
        "anchor",
        "literal",
        "parent_cid",
        "_cid_cache",
    )

    def __init__(
        self,
        vector: Optional[Union[QuantaVector, Sequence[int], bytes, Dict[Union[int, str], int]]] = None,
        edges: Optional[Union[Dict[str, List[str]], Sequence[Tuple[str, Union[str, bytes]]]]] = None,
        anchor: Optional[str] = None,
        literal: Optional[Any] = None,
        parent_cid: Optional[str] = None,
        concept_label: Optional[str] = None,
        semantic_vector: Optional[Union[QuantaVector, Sequence[int], bytes, Dict[Union[int, str], int]]] = None,
    ):
        raw_vec = semantic_vector if vector is None else vector
        if raw_vec is None:
            self.vector = QuantaVector.zeros()
        elif isinstance(raw_vec, QuantaVector):
            self.vector = raw_vec.copy()
        elif isinstance(raw_vec, dict):
            self.vector = QuantaVector(raw_vec)
        else:
            self.vector = QuantaVector(raw_vec)

        # Edges: Map relation name -> list of child CIDs
        self.edges: Dict[str, List[str]] = {}
        if edges is not None:
            if isinstance(edges, dict):
                for rel, targets in edges.items():
                    self.edges[rel] = list(targets)
            else:
                for item in edges:
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        rel, target = item
                        target_str = target.hex() if isinstance(target, bytes) else str(target)
                        if rel not in self.edges:
                            self.edges[rel] = []
                        if target_str not in self.edges[rel]:
                            self.edges[rel].append(target_str)

        self.anchor: Optional[str] = anchor if concept_label is None else concept_label
        self.literal: Optional[Any] = literal
        self.parent_cid: Optional[str] = parent_cid
        self._cid_cache: Optional[str] = None

    def invalidate_cache(self):
        """Invalidates cached CID."""
        self._cid_cache = None

    def set_slot(self, slot: Union[int, str], value: Union[int, QuaternaryValue, StructuralValue, RegisterValue]):
        """Sets a slot by index or name."""
        self.vector[slot] = value
        self.invalidate_cache()

    def get_slot(self, slot: Union[int, str]) -> QuaternaryValue:
        """Gets slot value by index or name."""
        return self.vector[slot]

    def get_structural_slot(self, slot: Union[int, str]) -> StructuralValue:
        """Gets slot value as a strongly-typed StructuralValue (Band 1 routing)."""
        return self.vector.get_structural_slot(slot)

    def set_structural_slot(self, slot: Union[int, str], value: Union[int, StructuralValue]):
        """Sets structural routing slot (Band 1)."""
        self.vector[slot] = value
        self.invalidate_cache()

    def get_register_slot(self, slot: Union[int, str]) -> RegisterValue:
        """Gets slot value as a strongly-typed RegisterValue (Band 2 scoping)."""
        return self.vector.get_register_slot(slot)

    def set_register_slot(self, slot: Union[int, str], value: Union[int, RegisterValue]):
        """Sets register scoping slot (Band 2)."""
        self.vector[slot] = value
        self.invalidate_cache()

    def get_epistemic_slot(self, slot: Union[int, str]) -> EpistemicValue:
        """Gets slot value as a strongly-typed EpistemicValue (Bands 0, 3..7)."""
        return self.vector.get_epistemic_slot(slot)

    def set_epistemic_slot(self, slot: Union[int, str], value: Union[int, EpistemicValue]):
        """Sets epistemic slot (Bands 0, 3..7)."""
        self.vector[slot] = value
        self.invalidate_cache()

    def get_routing_type(self, relation_or_slot: Union[int, str]) -> StructuralValue:
        """Determines the routing type for a given relation slot."""
        return self.vector.get_structural_slot(relation_or_slot)

    def add_edge(self, relation: str, target_cid: str):
        """Adds a directed relation edge to a child CID."""
        if relation not in self.edges:
            self.edges[relation] = []
        if target_cid not in self.edges[relation]:
            self.edges[relation].append(target_cid)
        self.invalidate_cache()

    def compute_cid(self) -> str:
        """Computes the deterministic 256-bit BLAKE3 Content Identifier (CID).
        
        The hash incorporates:
        1. 256-byte packed quaternary semantic vector
        2. Lexical anchor string (UTF-8)
        3. Canonical JSON-serialized literal payload
        4. Deterministically sorted relation edges: (relation, sorted child CIDs)
        """
        hasher = blake3.blake3()
        # 1. Quaternary vector (256 bytes)
        hasher.update(self.vector.to_bytes())

        # 2. Lexical Anchor
        anchor_bytes = (self.anchor or "").encode("utf-8")
        hasher.update(len(anchor_bytes).to_bytes(4, "big"))
        hasher.update(anchor_bytes)

        # 3. Literal payload
        literal_repr = "" if self.literal is None else json.dumps(self.literal, sort_keys=True)
        literal_bytes = literal_repr.encode("utf-8")
        hasher.update(len(literal_bytes).to_bytes(4, "big"))
        hasher.update(literal_bytes)

        # 4. Sorted edges
        sorted_relations = sorted(self.edges.keys())
        hasher.update(len(sorted_relations).to_bytes(4, "big"))
        for rel in sorted_relations:
            rel_bytes = rel.encode("utf-8")
            hasher.update(len(rel_bytes).to_bytes(4, "big"))
            hasher.update(rel_bytes)

            sorted_targets = sorted(self.edges[rel])
            hasher.update(len(sorted_targets).to_bytes(4, "big"))
            for target_cid in sorted_targets:
                target_bytes = target_cid.encode("utf-8")
                hasher.update(len(target_bytes).to_bytes(4, "big"))
                hasher.update(target_bytes)

        cid = hasher.hexdigest()
        self._cid_cache = cid
        return cid

    @property
    def cid(self) -> str:
        if self._cid_cache is None:
            return self.compute_cid()
        return self._cid_cache

    @property
    def node_cid(self) -> str:
        """Hexadecimal 256-bit BLAKE3 Content Identifier."""
        return self.cid

    @property
    def node_cid_bytes(self) -> bytes:
        """32-byte binary representation of the BLAKE3 Content Identifier."""
        return bytes.fromhex(self.cid)

    @property
    def semantic_vector(self) -> QuantaVector:
        """Semantic vector representation of this node."""
        return self.vector

    @semantic_vector.setter
    def semantic_vector(self, value: Union[QuantaVector, Sequence[int], bytes, Dict[Union[int, str], int]]):
        if isinstance(value, QuantaVector):
            self.vector = value.copy()
        elif isinstance(value, dict):
            self.vector = QuantaVector(value)
        else:
            self.vector = QuantaVector(value)
        self.invalidate_cache()

    @property
    def concept_label(self) -> Optional[str]:
        """Lexical anchor or concept label."""
        return self.anchor

    @concept_label.setter
    def concept_label(self, value: Optional[str]):
        self.anchor = value
        self.invalidate_cache()

    @property
    def edge_table(self) -> List[Tuple[str, str]]:
        """Edge table as a list of (relation_type, child_cid) tuples."""
        table: List[Tuple[str, str]] = []
        for rel in sorted(self.edges.keys()):
            for child_cid in self.edges[rel]:
                table.append((rel, child_cid))
        return table

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node to dictionary format."""
        from core.slots import get_slot_by_index
        readable_slots = {get_slot_by_index(k).name: int(v) for k, v in self.vector.active_slots().items()}
        return {
            "cid": self.cid,
            "parent_cid": self.parent_cid,
            "anchor": self.anchor,
            "literal": self.literal,
            "edges": self.edges,
            "active_slots": readable_slots,
            "packed_bytes_hex": self.vector.to_bytes().hex(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuantaNode:
        """Deserializes node from dictionary."""
        if "packed_bytes_hex" in data:
            vec = QuantaVector.from_bytes(bytes.fromhex(data["packed_bytes_hex"]))
        elif "active_slots" in data:
            vec = QuantaVector(data["active_slots"])
        else:
            vec = QuantaVector.zeros()

        node = cls(
            vector=vec,
            edges=data.get("edges"),
            anchor=data.get("anchor"),
            literal=data.get("literal"),
            parent_cid=data.get("parent_cid"),
        )
        return node

    def __repr__(self) -> str:
        anchor_str = f", anchor='{self.anchor}'" if self.anchor else ""
        return f"QuantaNode(cid={self.cid[:8]}..., edges={len(self.edges)}{anchor_str})"


class QuantaGraph:
    """Abstract Syntax Graph representing interconnected QuantaNodes with Merkle folding."""

    def __init__(self, root_cid: Optional[str] = None):
        self._node_list: List[QuantaNode] = []
        self._cid_to_node: Dict[str, QuantaNode] = {}
        self.root_cid: Optional[str] = root_cid

    @property
    def nodes(self) -> Dict[str, QuantaNode]:
        """Returns a dynamic map of current CID -> QuantaNode."""
        d = {}
        for n in self._node_list:
            d[n.cid] = n
        return d

    def add_node(self, node: QuantaNode, set_as_root: bool = False) -> str:
        """Adds a node to the graph and returns its CID."""
        if node not in self._node_list:
            self._node_list.append(node)
        cid = node.compute_cid()
        self._cid_to_node[cid] = node
        if set_as_root or self.root_cid is None:
            self.root_cid = cid
        return cid

    def get_node(self, cid: str) -> Optional[QuantaNode]:
        """Retrieves node by CID (supporting current CIDs and historical alias CIDs)."""
        if cid in self._cid_to_node:
            node = self._cid_to_node[cid]
            if node in self._node_list:
                return node
        for n in self._node_list:
            if n.cid == cid:
                self._cid_to_node[cid] = n
                return n
        return None

    def _propagate_cid_updates(self, initial_replacements: Dict[str, str]):
        """Propagates CID updates bottom-up through the graph.
        
        Whenever a child node's CID changes (due to edge additions, rewiring, folding, or unfolding),
        all ancestor nodes referencing that child have their edge tables updated, and their
        CIDs are recursively recomputed up to the root.
        """
        replacements = dict(initial_replacements)
        if not replacements:
            return

        max_iterations = max(len(self._node_list) * 2, 20)
        iteration = 0
        changed = True

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            new_replacements: Dict[str, str] = {}

            for node in self._node_list:
                node_modified = False
                for rel, targets in list(node.edges.items()):
                    new_targets = []
                    for t in targets:
                        if t in replacements:
                            new_targets.append(replacements[t])
                            node_modified = True
                        else:
                            new_targets.append(t)
                    node.edges[rel] = new_targets

                if node_modified:
                    old_cid = node._cid_cache or node.cid
                    new_cid = node.compute_cid()
                    if old_cid != new_cid:
                        new_replacements[old_cid] = new_cid
                        if self.root_cid == old_cid:
                            self.root_cid = new_cid
                        changed = True

            replacements.update(new_replacements)

        # Update parent_cid pointers for all reachable children
        for node in self._node_list:
            for rel, targets in node.edges.items():
                for t in targets:
                    child = self.get_node(t)
                    if child is not None:
                        child.parent_cid = node.cid

        # Preserve alias mappings for historical CIDs while registering new CIDs
        for old_cid, new_cid in replacements.items():
            if old_cid in self._cid_to_node:
                self._cid_to_node[new_cid] = self._cid_to_node[old_cid]
        for n in self._node_list:
            self._cid_to_node[n.cid] = n

    def add_edge(self, source_cid_or_node: Union[str, QuantaNode], relation: str, target_cid_or_node: Union[str, QuantaNode]) -> str:
        """Adds a directed relation edge between two nodes in the graph."""
        if isinstance(source_cid_or_node, QuantaNode):
            src_node = source_cid_or_node
            if src_node not in self._node_list:
                self.add_node(src_node)
        else:
            src_node = self.get_node(source_cid_or_node)
            if src_node is None:
                raise KeyError(f"Source node CID '{source_cid_or_node}' not found in graph")

        if isinstance(target_cid_or_node, QuantaNode):
            dst_node = target_cid_or_node
            if dst_node not in self._node_list:
                self.add_node(dst_node)
        else:
            dst_node = self.get_node(target_cid_or_node)
            if dst_node is None:
                raise KeyError(f"Target node CID '{target_cid_or_node}' not found in graph")

        old_src_cid = src_node.cid
        src_node.add_edge(relation, dst_node.cid)
        dst_node.parent_cid = src_node.cid
        new_src_cid = src_node.compute_cid()

        if old_src_cid != new_src_cid:
            if self.root_cid == old_src_cid:
                self.root_cid = new_src_cid
            self._propagate_cid_updates({old_src_cid: new_src_cid})
        else:
            self._cid_to_node[new_src_cid] = src_node

        return src_node.cid

    @property
    def root(self) -> Optional[QuantaNode]:
        """Returns the root QuantaNode if set."""
        if self.root_cid is not None:
            return self.get_node(self.root_cid)
        return None

    def compute_merkle_root(self, algorithm: str = "blake3") -> str:
        """Folds all nodes in the graph into a top-level Merkle root CID hash.
        
        Args:
            algorithm: Cryptographic hashing algorithm ('blake3' or 'sha256').
        """
        if algorithm.lower() == "sha256":
            hasher = hashlib.sha256()
        else:
            hasher = blake3.blake3()
        current_nodes = self.nodes
        sorted_cids = sorted(current_nodes.keys())
        hasher.update(len(sorted_cids).to_bytes(4, "big"))
        for cid in sorted_cids:
            node = current_nodes[cid]
            hasher.update(cid.encode("utf-8"))
            hasher.update(node.vector.to_bytes())
        return hasher.hexdigest()

    def fold_subgraph(
        self,
        subtree_root_cid: str,
        storage: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """Folds a reachable subtree into a cryptographic Merkle pointer node.
        
        Args:
            subtree_root_cid: Root CID of the subtree to fold (or historical alias CID).
            storage: Optional dictionary or storage backend to preserve the folded sub-graph.
            
        Returns:
            Tuple of (pointer_node_cid, sub_merkle_cid).
        """
        root_node = self.get_node(subtree_root_cid)
        if root_node is None:
            raise KeyError(f"Subtree root CID '{subtree_root_cid}' not found in graph")
        actual_subtree_root_cid = root_node.cid

        # 1. Identify all reachable nodes in the subtree
        visited: Set[str] = set()
        def collect_reachable(curr: str):
            if curr in visited:
                return
            visited.add(curr)
            n = self.get_node(curr)
            if n:
                for targets in n.edges.values():
                    for t in targets:
                        collect_reachable(t)

        collect_reachable(actual_subtree_root_cid)

        # 2. Build isolated sub-graph
        sub_graph = QuantaGraph(root_cid=actual_subtree_root_cid)
        for cid in visited:
            node = self.get_node(cid)
            if node is None:
                continue
            new_node = QuantaNode(
                vector=node.vector.copy(),
                anchor=node.anchor,
                literal=node.literal,
                parent_cid=node.parent_cid,
            )
            new_node.edges = {k: list(v) for k, v in node.edges.items()}
            sub_graph.add_node(new_node)

        sub_merkle_cid = sub_graph.compute_merkle_root()

        if storage is not None:
            storage[sub_merkle_cid] = sub_graph.to_dict()

        # 3. Create pointer node in main graph preserving aggregate semantic proposition vector
        pointer_vec = sub_graph.to_proposition_vector()
        pointer_vec["GRAPH_MERKLE_FOLD_POINT"] = StructuralValue.ACTIVE_MERKLE
        pointer_vec["GRAPH_EXT_REFERENCE"] = StructuralValue.ACTIVE_MERKLE
        pointer_vec["TYPE_PROPOSITION"] = 1

        pointer_node = QuantaNode(
            vector=pointer_vec,
            anchor=f"merkle:{sub_merkle_cid}",
            literal=f"FoldedSubtree({sub_merkle_cid[:8]})",
        )
        pointer_cid = pointer_node.compute_cid()

        # 4. Remove all subtree nodes from self._node_list and _cid_to_node
        self._node_list = [n for n in self._node_list if n.cid not in visited]
        for cid in list(visited):
            self._cid_to_node.pop(cid, None)

        # 5. Add pointer node
        self._node_list.append(pointer_node)
        self._cid_to_node[pointer_cid] = pointer_node

        # 6. Rewire incoming edges and propagate CID updates bottom-up across ancestors
        if self.root_cid == actual_subtree_root_cid or self.root_cid == subtree_root_cid:
            self.root_cid = pointer_cid
        self._propagate_cid_updates({actual_subtree_root_cid: pointer_cid, subtree_root_cid: pointer_cid})

        return pointer_cid, sub_merkle_cid

    def unfold_subgraph(
        self,
        pointer_node_cid: str,
        storage: Dict[str, Any],
    ) -> str:
        """Unfolds a previously folded Merkle pointer node back into its full sub-graph structure.
        
        Args:
            pointer_node_cid: The CID of the fold pointer node.
            storage: Storage mapping sub_merkle_cid to serialized or QuantaGraph objects.
            
        Returns:
            The restored subtree root CID.
        """
        pointer_node = self.get_node(pointer_node_cid)
        if pointer_node is None:
            raise KeyError(f"Pointer node '{pointer_node_cid}' not found in graph")

        if not pointer_node.anchor or not pointer_node.anchor.startswith("merkle:"):
            raise ValueError(f"Node '{pointer_node_cid}' is not a valid Merkle fold pointer")

        sub_merkle_cid = pointer_node.anchor.split("merkle:")[1]
        if sub_merkle_cid not in storage:
            raise KeyError(f"Sub-graph with Merkle CID '{sub_merkle_cid}' not found in storage")

        stored_data = storage[sub_merkle_cid]
        if isinstance(stored_data, dict):
            sub_graph = QuantaGraph.from_dict(stored_data)
        elif isinstance(stored_data, QuantaGraph):
            sub_graph = stored_data
        else:
            raise TypeError(f"Unsupported storage payload type: {type(stored_data)}")

        # Verify tamper integrity across BLAKE3 (default) or SHA-256
        actual_merkle_blake3 = sub_graph.compute_merkle_root(algorithm="blake3")
        actual_merkle_sha256 = sub_graph.compute_merkle_root(algorithm="sha256")
        if sub_merkle_cid.lower() not in (actual_merkle_blake3.lower(), actual_merkle_sha256.lower()):
            raise ValueError(
                f"Merkle CID mismatch during unfold: expected {sub_merkle_cid}, but computed {actual_merkle_blake3} (storage payload tampered)"
            )

        restored_root_cid = sub_graph.root_cid
        if not restored_root_cid:
            raise ValueError("Unfolded sub-graph has no root CID")

        # 1. Remove pointer node from graph
        self._node_list = [n for n in self._node_list if n.cid != pointer_node_cid]
        self._cid_to_node.pop(pointer_node_cid, None)

        # 2. Add all nodes from sub_graph back into self
        for sub_node in sub_graph.nodes.values():
            new_node = QuantaNode(
                vector=sub_node.vector.copy(),
                anchor=sub_node.anchor,
                literal=sub_node.literal,
                parent_cid=sub_node.parent_cid,
            )
            new_node.edges = {k: list(v) for k, v in sub_node.edges.items()}
            if new_node not in self._node_list:
                self._node_list.append(new_node)
            self._cid_to_node[new_node.compute_cid()] = new_node

        # 3. Rewire incoming edges from main graph pointing to pointer_node_cid to restored_root_cid
        if self.root_cid == pointer_node_cid:
            self.root_cid = restored_root_cid
        self._propagate_cid_updates({pointer_node_cid: restored_root_cid})

        return restored_root_cid

    def validate_integrity(self) -> Tuple[bool, List[str]]:
        """Verifies that all child pointers exist and CIDs match node payloads."""
        errors: List[str] = []
        current_nodes = self.nodes
        for cid, node in current_nodes.items():
            expected_cid = node.compute_cid()
            if cid != expected_cid:
                errors.append(f"Node CID mismatch: index key is {cid}, but payload hashes to {expected_cid}")
            for rel, targets in node.edges.items():
                for target_cid in targets:
                    target_node = self.get_node(target_cid)
                    if target_node is None:
                        errors.append(f"Dangling edge in {cid}: relation '{rel}' points to missing CID {target_cid}")
                    elif target_node.compute_cid() != target_cid:
                        if rel not in ("GRAPH_RECURSIVE_REF", "GRAPH_CYCLIC_BACKLINK", "GRAPH_MERKLE_FOLD_POINT"):
                            errors.append(
                                f"Tampered target node or CID mismatch: edge '{rel}' in {cid} expects CID {target_cid}, but target payload hashes to {target_node.compute_cid()}"
                            )
        return len(errors) == 0, errors

    def topological_order(self) -> List[QuantaNode]:
        """Returns a topological sorting of the reachable graph."""
        visited: Set[str] = set()
        order: List[QuantaNode] = []
        current_nodes = self.nodes

        def dfs(curr_cid: str):
            if curr_cid in visited:
                return
            visited.add(curr_cid)
            node = current_nodes.get(curr_cid)
            if node is None:
                return
            for targets in node.edges.values():
                for t_cid in targets:
                    dfs(t_cid)
            order.append(node)

        if self.root_cid and self.root_cid in current_nodes:
            dfs(self.root_cid)
        for cid in sorted(current_nodes.keys()):
            if cid not in visited:
                dfs(cid)

        return order

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire graph to a dictionary."""
        current_nodes = self.nodes
        return {
            "root_cid": self.root_cid,
            "merkle_root": self.compute_merkle_root(),
            "nodes": {cid: node.to_dict() for cid, node in current_nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuantaGraph:
        """Deserializes graph from dictionary."""
        graph = cls(root_cid=data.get("root_cid"))
        for cid, node_dict in data.get("nodes", {}).items():
            node = QuantaNode.from_dict(node_dict)
            graph.add_node(node)
        return graph

    def to_json(self, indent: int = 2) -> str:
        """Serializes graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> QuantaGraph:
        """Deserializes graph from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_children(self, node_cid_or_node: Union[str, QuantaNode], relation: Optional[str] = None) -> List[QuantaNode]:
        """Retrieves child QuantaNode instances connected from the given node, optionally filtered by relation type."""
        if isinstance(node_cid_or_node, QuantaNode):
            src_cid = node_cid_or_node.cid
        else:
            src_cid = node_cid_or_node

        src_node = self.get_node(src_cid)
        if src_node is None:
            return []

        children: List[QuantaNode] = []
        if relation is not None:
            target_cids = src_node.edges.get(relation, [])
            for cid in target_cids:
                child = self.get_node(cid)
                if child is not None:
                    children.append(child)
        else:
            for rel, target_cids in src_node.edges.items():
                for cid in target_cids:
                    child = self.get_node(cid)
                    if child is not None and child not in children:
                        children.append(child)
        return children

    def to_proposition_vector(self) -> QuantaVector:
        """Aggregates all active slot values across the entire ASG graph into a single proposition vector via lattice join (⊔_k)."""
        vec = QuantaVector.zeros()
        for node in self.nodes.values():
            vec = vec.join(node.vector)
        return vec

    @property
    def tree_aggregate_vector(self) -> QuantaVector:
        """Whole-tree proposition vector aggregated via quaternary lattice join (⊔_k)."""
        return self.to_proposition_vector()

    def aggregate_vector(self) -> QuantaVector:
        """Returns the aggregated whole-tree proposition vector."""
        return self.to_proposition_vector()

    def __len__(self) -> int:
        return len(self._node_list)

    def __repr__(self) -> str:
        return f"QuantaGraph(nodes={len(self._node_list)}, root_cid={self.root_cid[:8] if self.root_cid else 'None'})"


def fold_subgraph(
    graph: QuantaGraph,
    subtree_root_cid: str,
    storage: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Folds a reachable subtree in graph into a Merkle pointer node.
    
    Args:
        graph: Target QuantaGraph to modify.
        subtree_root_cid: Root CID of the subtree to fold.
        storage: Optional dict or backend to store folded sub-graph.
        
    Returns:
        Tuple of (pointer_node_cid, sub_merkle_cid).
    """
    return graph.fold_subgraph(subtree_root_cid, storage=storage)


def _is_entity_node(node: QuantaNode) -> bool:
    """Identifies whether a QuantaNode represents an external entity (character, place, substance, etc.)."""
    if node.get_slot("TYPE_EVENT") != 0 or node.get_slot("WN_ACT_ACTION") != 0:
        return False
    if node.get_slot("GRAPH_VARIABLE_BIND") != 0:
        return True
    entity_slots = (
        "TYPE_HUMAN",
        "TYPE_ANIMATE",
        "TYPE_SPATIAL_REGION",
        "TYPE_ARTIFACT",
        "TYPE_NATURAL_OBJECT",
        "CN_Q072_SUBSTANCE",
        "TYPE_ORGANIZATION",
        "ROLE_AGENT_CAPABLE",
        "ROLE_SENTIENT",
        "ROLE_PATIENT_TARGET",
    )
    for slot in entity_slots:
        if node.get_slot(slot) != 0:
            return True
    if node.anchor and (node.anchor.startswith("cn:en:") and "(n)" in node.anchor):
        return True
    return False


def unfold_subgraph(
    folded_cid_or_node: Union[str, QuantaNode],
    storage: Union[Dict[str, Any], MutableMapping],
    target_graph: Optional[QuantaGraph] = None,
) -> Union[str, QuantaGraph]:
    """Unfolds a Merkle pointer node or Merkle CID with cryptographic verification.
    
    Verifies exact BLAKE3 / SHA-256 Merkle root integrity and rejects tampered storage payloads.
    
    Args:
        folded_cid_or_node: Pointer node instance, pointer node CID, or Merkle CID.
        storage: Storage mapping sub_merkle_cid to serialized or QuantaGraph objects.
        target_graph: Optional graph to unfold into in-place. If provided, returns the restored root CID.
                      If None, returns an isolated restored QuantaGraph.
                      
    Returns:
        Restored root CID string if target_graph is provided, or QuantaGraph if target_graph is None.
        
    Raises:
        KeyError: If Merkle CID is not found in storage.
        ValueError: If payload hash does not match expected Merkle CID (tamper detected).
    """
    if target_graph is not None:
        cid = folded_cid_or_node.cid if isinstance(folded_cid_or_node, QuantaNode) else str(folded_cid_or_node)
        return target_graph.unfold_subgraph(cid, storage)

    if isinstance(folded_cid_or_node, QuantaNode):
        if not folded_cid_or_node.anchor or not folded_cid_or_node.anchor.startswith("merkle:"):
            raise ValueError("Node is not a valid Merkle fold pointer")
        sub_merkle_cid = folded_cid_or_node.anchor.split("merkle:")[1]
    else:
        cid_str = str(folded_cid_or_node)
        if cid_str.startswith("merkle:"):
            sub_merkle_cid = cid_str.split("merkle:")[1]
        elif cid_str in storage:
            stored_val = storage[cid_str]
            if isinstance(stored_val, dict) and "merkle_root" in stored_val:
                sub_merkle_cid = cid_str
            elif isinstance(stored_val, dict) and "anchor" in stored_val and str(stored_val["anchor"]).startswith("merkle:"):
                sub_merkle_cid = str(stored_val["anchor"]).split("merkle:")[1]
            elif isinstance(stored_val, str) and stored_val.startswith("merkle:"):
                sub_merkle_cid = stored_val.split("merkle:")[1]
            elif isinstance(stored_val, str) and (len(stored_val) == 64 and all(c in "0123456789abcdefABCDEF" for c in stored_val)):
                sub_merkle_cid = stored_val
            else:
                sub_merkle_cid = cid_str
        elif f"chunk:{cid_str}" in storage:
            sub_merkle_cid = storage[f"chunk:{cid_str}"]
        elif f"chapter:{cid_str}" in storage:
            sub_merkle_cid = storage[f"chapter:{cid_str}"]
        elif f"book:{cid_str}" in storage:
            sub_merkle_cid = storage[f"book:{cid_str}"]
        else:
            sub_merkle_cid = cid_str

    if sub_merkle_cid not in storage:
        raise KeyError(f"Sub-graph with Merkle CID '{sub_merkle_cid}' not found in storage")

    stored_data = storage[sub_merkle_cid]
    if isinstance(stored_data, dict):
        sub_graph = QuantaGraph.from_dict(stored_data)
    elif isinstance(stored_data, QuantaGraph):
        sub_graph = stored_data
    elif isinstance(stored_data, str):
        sub_graph = QuantaGraph.from_json(stored_data)
    else:
        raise TypeError(f"Unsupported storage payload type: {type(stored_data)}")

    actual_merkle_blake3 = sub_graph.compute_merkle_root(algorithm="blake3")
    actual_merkle_sha256 = sub_graph.compute_merkle_root(algorithm="sha256")
    if sub_merkle_cid.lower() not in (actual_merkle_blake3.lower(), actual_merkle_sha256.lower()):
        raise ValueError(
            f"Merkle CID mismatch during unfold: expected {sub_merkle_cid}, but computed {actual_merkle_blake3} (storage payload tampered)"
        )
    return sub_graph


def fold_discourse_episode(
    graph: QuantaGraph,
    chunk_id: str,
    storage: Optional[Union[Dict[str, Any], MutableMapping]] = None,
    keep_entities: bool = True,
) -> QuantaNode:
    """Folds the internal event DAG of a discourse episode into a 32-byte Merkle fold node.
    
    In long-form documents, maintaining every individual event node in active memory
    is unnecessary. Once a paragraph/chunk is compiled and validated, its event DAG is
    sealed into a Sub-Graph Merkle CID (GRAPH_MERKLE_FOLD_POINT).
    
    External entity references are kept intact on the resulting fold node so that the
    working memory / active graph retains the entity-to-episode valencies.
    
    Args:
        graph: The QuantaGraph containing the episode (modified in-place).
        chunk_id: Identifier of the chunk (e.g. 'chunk_0001').
        storage: Optional dictionary or PageTableStorage to persist the folded sub-graph.
        keep_entities: If True, entity nodes remain in the active graph and are wired
                       to the fold node.
                       
    Returns:
        The fold QuantaNode (32-byte Merkle CID node) with aggregate quaternary vector.
    """
    if len(graph) == 0:
        raise ValueError("Cannot fold an empty QuantaGraph")

    # 1. Classify nodes: external entities vs internal episode nodes
    if keep_entities:
        entity_cids = {cid for cid, n in graph.nodes.items() if _is_entity_node(n)}
        internal_cids = {cid for cid in graph.nodes.keys() if cid not in entity_cids}
        if not internal_cids:
            # If no distinct internal nodes found, treat all nodes as internal
            internal_cids = set(graph.nodes.keys())
            entity_cids = set()
    else:
        internal_cids = set(graph.nodes.keys())
        entity_cids = set()

    # 2. Build isolated episode sub-graph
    episode_subgraph = QuantaGraph()
    for cid in internal_cids:
        node = graph.get_node(cid)
        if node is not None:
            cloned = QuantaNode(
                vector=node.vector.copy(),
                anchor=node.anchor,
                literal=node.literal,
                parent_cid=node.parent_cid,
            )
            cloned.edges = {k: list(v) for k, v in node.edges.items()}
            episode_subgraph.add_node(cloned)

    if graph.root_cid in internal_cids:
        episode_subgraph.root_cid = graph.root_cid
    else:
        for n in episode_subgraph.nodes.values():
            if n.get_slot("GRAPH_ROOT_NODE") == 1:
                episode_subgraph.root_cid = n.cid
                break
        if not episode_subgraph.root_cid and episode_subgraph.nodes:
            episode_subgraph.root_cid = next(iter(episode_subgraph.nodes.keys()))

    sub_merkle_cid = episode_subgraph.compute_merkle_root()

    # 3. Store sub-graph if storage provided
    if storage is not None:
        storage[sub_merkle_cid] = episode_subgraph.to_dict()
        storage[f"chunk:{chunk_id}"] = sub_merkle_cid

    # 4. Compute aggregate quaternary proposition vector via lattice join
    aggregate_vec = episode_subgraph.to_proposition_vector()
    aggregate_vec["GRAPH_MERKLE_FOLD_POINT"] = StructuralValue.ACTIVE_MERKLE
    aggregate_vec["GRAPH_EXT_REFERENCE"] = StructuralValue.ACTIVE_MERKLE
    aggregate_vec["TYPE_PROPOSITION"] = 1

    # 5. Extract and preserve external entity edges
    external_edges: Dict[str, List[str]] = {}
    for cid in internal_cids:
        node = graph.get_node(cid)
        if node is None:
            continue
        for rel, targets in node.edges.items():
            for t in targets:
                if t in entity_cids:
                    if rel not in external_edges:
                        external_edges[rel] = []
                    if t not in external_edges[rel]:
                        external_edges[rel].append(t)

    # 6. Create fold QuantaNode
    fold_node = QuantaNode(
        vector=aggregate_vec,
        anchor=f"merkle:{sub_merkle_cid}",
        literal={
            "chunk_id": chunk_id,
            "merkle_cid": sub_merkle_cid,
            "type": "discourse_episode_fold",
        },
    )
    for rel, targets in external_edges.items():
        for t in targets:
            fold_node.add_edge(rel, t)

    fold_cid = fold_node.compute_cid()

    # 7. Update target graph in-place
    graph._node_list = [n for n in graph._node_list if n.cid not in internal_cids]
    for cid in list(internal_cids):
        graph._cid_to_node.pop(cid, None)

    graph._node_list.append(fold_node)
    graph._cid_to_node[fold_cid] = fold_node

    if graph.root_cid in internal_cids or graph.root_cid is None:
        graph.root_cid = fold_cid

    replacements = {cid: fold_cid for cid in internal_cids}
    graph._propagate_cid_updates(replacements)

    return fold_node


def fold_chapter(
    chunk_nodes: Sequence[QuantaNode],
    chapter_id: str,
    chapter_title: Optional[str] = None,
    storage: Optional[Union[Dict[str, Any], MutableMapping]] = None,
) -> QuantaNode:
    """Folds constituent chunk fold nodes into a Chapter Merkle fold node.
    
    Args:
        chunk_nodes: Sequence of QuantaNode instances representing folded chunk episodes.
        chapter_id: Unique chapter identifier (e.g. 'chapter_01').
        chapter_title: Optional human-readable chapter heading/title.
        storage: Optional storage backend to persist the chapter sub-graph.
        
    Returns:
        The Chapter fold QuantaNode with Merkle CID and aggregate quaternary vector.
    """
    if not chunk_nodes:
        raise ValueError("Cannot fold an empty sequence of chunk nodes into a chapter")

    chapter_graph = QuantaGraph()
    for c_node in chunk_nodes:
        cloned = QuantaNode(
            vector=c_node.vector.copy(),
            anchor=c_node.anchor,
            literal=c_node.literal,
            parent_cid=c_node.parent_cid,
        )
        cloned.edges = {k: list(v) for k, v in c_node.edges.items()}
        chapter_graph.add_node(cloned)

    for i in range(len(chunk_nodes) - 1):
        chapter_graph.add_edge(chunk_nodes[i].cid, "TEMP_ALLEN_MEETS", chunk_nodes[i + 1].cid)

    chapter_merkle_cid = chapter_graph.compute_merkle_root()

    if storage is not None:
        storage[chapter_merkle_cid] = chapter_graph.to_dict()
        storage[f"chapter:{chapter_id}"] = chapter_merkle_cid

    chapter_vec = chapter_graph.to_proposition_vector()
    chapter_vec["GRAPH_MERKLE_FOLD_POINT"] = StructuralValue.ACTIVE_MERKLE
    chapter_vec["GRAPH_EXT_REFERENCE"] = StructuralValue.ACTIVE_MERKLE
    chapter_vec["TYPE_PROPOSITION"] = 1

    chapter_node = QuantaNode(
        vector=chapter_vec,
        anchor=f"merkle:{chapter_merkle_cid}",
        literal={
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "chunk_cids": [c.cid for c in chunk_nodes],
            "type": "chapter_fold",
        },
    )
    for c_node in chunk_nodes:
        chapter_node.add_edge("GRAPH_MERKLE_FOLD_POINT", c_node.cid)

    chapter_node.compute_cid()
    return chapter_node


def fold_book(
    chapter_nodes: Sequence[QuantaNode],
    book_id: str = "book_1",
    book_title: Optional[str] = None,
    storage: Optional[Union[Dict[str, Any], MutableMapping]] = None,
) -> QuantaNode:
    """Folds constituent chapter fold nodes into the top-level Book Merkle root.
    
    Args:
        chapter_nodes: Sequence of QuantaNode instances representing folded chapters.
        book_id: Unique book identifier (e.g. 'book_01').
        book_title: Optional human-readable book title.
        storage: Optional storage backend to persist the book graph.
        
    Returns:
        The Book Root fold QuantaNode with Book Root CID and aggregate quaternary vector.
    """
    if not chapter_nodes:
        raise ValueError("Cannot fold an empty sequence of chapter nodes into a book")

    book_graph = QuantaGraph()
    for ch_node in chapter_nodes:
        cloned = QuantaNode(
            vector=ch_node.vector.copy(),
            anchor=ch_node.anchor,
            literal=ch_node.literal,
            parent_cid=ch_node.parent_cid,
        )
        cloned.edges = {k: list(v) for k, v in ch_node.edges.items()}
        book_graph.add_node(cloned)

    for i in range(len(chapter_nodes) - 1):
        book_graph.add_edge(chapter_nodes[i].cid, "TEMP_ALLEN_MEETS", chapter_nodes[i + 1].cid)

    book_merkle_cid = book_graph.compute_merkle_root()

    if storage is not None:
        storage[book_merkle_cid] = book_graph.to_dict()
        storage[f"book:{book_id}"] = book_merkle_cid

    book_vec = book_graph.to_proposition_vector()
    book_vec["GRAPH_MERKLE_FOLD_POINT"] = StructuralValue.ACTIVE_MERKLE
    book_vec["GRAPH_EXT_REFERENCE"] = StructuralValue.ACTIVE_MERKLE
    book_vec["TYPE_PROPOSITION"] = 1

    book_node = QuantaNode(
        vector=book_vec,
        anchor=f"merkle:{book_merkle_cid}",
        literal={
            "book_id": book_id,
            "book_title": book_title,
            "chapter_cids": [ch.cid for ch in chapter_nodes],
            "type": "book_fold",
        },
    )
    for ch_node in chapter_nodes:
        book_node.add_edge("GRAPH_MERKLE_FOLD_POINT", ch_node.cid)

    book_node.compute_cid()
    return book_node


class HierarchicalMerkleBook:
    """High-level hierarchical Merkle coordinator for book-scale narrative memory."""

    def __init__(
        self,
        book_id: str = "book_1",
        title: Optional[str] = None,
        storage: Optional[Union[Dict[str, Any], MutableMapping]] = None,
    ):
        self.book_id = book_id
        self.title = title
        self.storage = storage if storage is not None else {}
        self.chapters: Dict[str, List[QuantaNode]] = {}
        self.chapter_nodes: Dict[str, QuantaNode] = {}
        self.book_node: Optional[QuantaNode] = None

    def add_chunk_episode(
        self,
        chunk_id: str,
        graph: QuantaGraph,
        chapter_id: str = "chapter_1",
        keep_entities: bool = True,
    ) -> QuantaNode:
        """Folds a discourse episode and records it in the given chapter."""
        chunk_node = fold_discourse_episode(
            graph=graph,
            chunk_id=chunk_id,
            storage=self.storage,
            keep_entities=keep_entities,
        )
        if chapter_id not in self.chapters:
            self.chapters[chapter_id] = []
        self.chapters[chapter_id].append(chunk_node)
        return chunk_node

    def add_chunk_node(self, chunk_node: QuantaNode, chapter_id: str = "chapter_1"):
        """Registers an already-folded chunk node into a chapter."""
        if chapter_id not in self.chapters:
            self.chapters[chapter_id] = []
        self.chapters[chapter_id].append(chunk_node)

    def fold_chapter(self, chapter_id: str, chapter_title: Optional[str] = None) -> QuantaNode:
        """Folds all registered chunk nodes for the given chapter into a Chapter Merkle fold node."""
        chunk_nodes = self.chapters.get(chapter_id, [])
        if not chunk_nodes:
            raise ValueError(f"No chunk episodes found for chapter '{chapter_id}'")
        ch_node = fold_chapter(
            chunk_nodes=chunk_nodes,
            chapter_id=chapter_id,
            chapter_title=chapter_title,
            storage=self.storage,
        )
        self.chapter_nodes[chapter_id] = ch_node
        return ch_node

    def fold_book(self) -> QuantaNode:
        """Folds all chapters in order into the Book Merkle root node."""
        ordered_chapters: List[QuantaNode] = []
        for ch_id in self.chapters.keys():
            if ch_id not in self.chapter_nodes:
                self.fold_chapter(ch_id)
            ordered_chapters.append(self.chapter_nodes[ch_id])

        if not ordered_chapters:
            raise ValueError("No chapters available to fold into book")

        self.book_node = fold_book(
            chapter_nodes=ordered_chapters,
            book_id=self.book_id,
            book_title=self.title,
            storage=self.storage,
        )
        return self.book_node

    @property
    def book_root_cid(self) -> Optional[str]:
        """The 32-byte (64-char hex) BLAKE3 Content Identifier of the Book Root Node."""
        return self.book_node.cid if self.book_node else None

    @property
    def book_merkle_cid(self) -> Optional[str]:
        """The Merkle CID of the folded book sub-graph."""
        if not self.book_node or not self.book_node.anchor:
            return None
        return self.book_node.anchor.split("merkle:")[1]

    def unfold_chunk(self, chunk_id_or_merkle_cid: str) -> QuantaGraph:
        """Dynamically unfolds a chunk episode sub-graph from storage with tamper verification."""
        cid = chunk_id_or_merkle_cid
        if f"chunk:{chunk_id_or_merkle_cid}" in self.storage:
            cid = self.storage[f"chunk:{chunk_id_or_merkle_cid}"]
        return unfold_subgraph(cid, self.storage)

    def unfold_chapter(self, chapter_id_or_merkle_cid: str) -> QuantaGraph:
        """Dynamically unfolds a chapter sub-graph from storage with tamper verification."""
        cid = chapter_id_or_merkle_cid
        if f"chapter:{chapter_id_or_merkle_cid}" in self.storage:
            cid = self.storage[f"chapter:{chapter_id_or_merkle_cid}"]
        return unfold_subgraph(cid, self.storage)

    def unfold_book(self) -> QuantaGraph:
        """Dynamically unfolds the book sub-graph from storage with tamper verification."""
        if not self.book_merkle_cid:
            raise ValueError("Book has not been folded yet")
        return unfold_subgraph(self.book_merkle_cid, self.storage)

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verifies cryptographic integrity across all stored chunks, chapters, and the book root."""
        errors: List[str] = []
        for ch_id, chunk_list in self.chapters.items():
            for c_node in chunk_list:
                if not c_node.anchor or not c_node.anchor.startswith("merkle:"):
                    errors.append(f"Chunk node {c_node.cid} missing merkle anchor")
                    continue
                merkle_cid = c_node.anchor.split("merkle:")[1]
                try:
                    self.unfold_chunk(merkle_cid)
                except Exception as e:
                    errors.append(f"Chunk {c_node.cid} integrity failed: {e}")

        for ch_id, ch_node in self.chapter_nodes.items():
            merkle_cid = ch_node.anchor.split("merkle:")[1]
            try:
                self.unfold_chapter(merkle_cid)
            except Exception as e:
                errors.append(f"Chapter '{ch_id}' integrity failed: {e}")

        if self.book_merkle_cid:
            try:
                self.unfold_book()
            except Exception as e:
                errors.append(f"Book integrity failed: {e}")

        return len(errors) == 0, errors


__all__ = [
    "QuantaNode",
    "QuantaGraph",
    "fold_subgraph",
    "unfold_subgraph",
    "fold_discourse_episode",
    "fold_chapter",
    "fold_book",
    "HierarchicalMerkleBook",
]


