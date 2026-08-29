"""Abstract Syntax Graph (ASG) and Content-Addressed Merkle nodes for QUANTA."""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import blake3

from core.types import QuantaVector, QuaternaryValue
from core.slots import SLOT_NAME_TO_INDEX, get_slot_by_name


def _blake3_hash(data: bytes) -> str:
    """Computes a 256-bit BLAKE3 hash in hexadecimal format."""
    return blake3.blake3(data).hexdigest()


class QuantaNode:
    """Atomic Abstract Syntax Graph node with a 256-dim quaternary vector,

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
        edges: Optional[Dict[str, List[str]]] = None,
        anchor: Optional[str] = None,
        literal: Optional[Any] = None,
        parent_cid: Optional[str] = None,
    ):
        if vector is None:
            self.vector = QuantaVector.zeros()
        elif isinstance(vector, QuantaVector):
            self.vector = vector.copy()
        elif isinstance(vector, dict):
            self.vector = QuantaVector(vector)
        else:
            self.vector = QuantaVector(vector)

        # Edges: Map relation name -> list of child CIDs
        self.edges: Dict[str, List[str]] = {}
        if edges is not None:
            for rel, targets in edges.items():
                self.edges[rel] = list(targets)

        self.anchor: Optional[str] = anchor
        self.literal: Optional[Any] = literal
        self.parent_cid: Optional[str] = parent_cid
        self._cid_cache: Optional[str] = None

    def invalidate_cache(self):
        """Invalidates cached CID."""
        self._cid_cache = None

    def set_slot(self, slot: Union[int, str], value: Union[int, QuaternaryValue]):
        """Sets a slot by index or name."""
        self.vector[slot] = value
        self.invalidate_cache()

    def get_slot(self, slot: Union[int, str]) -> QuaternaryValue:
        """Gets slot value by index or name."""
        return self.vector[slot]

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
        1. 64-byte packed quaternary semantic vector
        2. Lexical anchor string (UTF-8)
        3. Canonical JSON-serialized literal payload
        4. Deterministically sorted relation edges: (relation, sorted child CIDs)
        """
        hasher = blake3.blake3()
        # 1. Quaternary vector (64 bytes)
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

    def compute_merkle_root(self) -> str:
        """Folds all nodes in the graph into a top-level Merkle root CID hash."""
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
            subtree_root_cid: Root CID of the subtree to fold.
            storage: Optional dictionary or storage backend to preserve the folded sub-graph.
            
        Returns:
            Tuple of (pointer_node_cid, sub_merkle_cid).
        """
        current_nodes = self.nodes
        if subtree_root_cid not in current_nodes:
            raise KeyError(f"Subtree root CID '{subtree_root_cid}' not found in graph")

        # 1. Identify all reachable nodes in the subtree
        visited: Set[str] = set()
        def collect_reachable(curr: str):
            if curr in visited:
                return
            visited.add(curr)
            n = current_nodes.get(curr)
            if n:
                for targets in n.edges.values():
                    for t in targets:
                        collect_reachable(t)

        collect_reachable(subtree_root_cid)

        # 2. Build isolated sub-graph
        sub_graph = QuantaGraph(root_cid=subtree_root_cid)
        for cid in visited:
            node = current_nodes[cid]
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
        pointer_vec["GRAPH_MERKLE_FOLD_POINT"] = 1
        pointer_vec["GRAPH_EXT_REFERENCE"] = 1
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
        if self.root_cid == subtree_root_cid:
            self.root_cid = pointer_cid
        self._propagate_cid_updates({subtree_root_cid: pointer_cid})

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

        # Verify tamper integrity
        actual_merkle = sub_graph.compute_merkle_root()
        if actual_merkle != sub_merkle_cid:
            raise ValueError(
                f"Merkle CID mismatch during unfold: expected {sub_merkle_cid}, but computed {actual_merkle} (storage payload tampered)"
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
                    if target_cid not in current_nodes:
                        errors.append(f"Dangling edge in {cid}: relation '{rel}' points to missing CID {target_cid}")
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


def unfold_subgraph(
    folded_cid_or_node: Union[str, QuantaNode],
    storage: Dict[str, Any],
    target_graph: Optional[QuantaGraph] = None,
) -> Union[str, QuantaGraph]:
    """Unfolds a Merkle pointer node.
    
    Args:
        folded_cid_or_node: Pointer node instance, pointer node CID, or Merkle CID.
        storage: Storage mapping sub_merkle_cid to serialized or QuantaGraph objects.
        target_graph: Optional graph to unfold into in-place. If provided, returns the restored root CID.
                      If None, returns an isolated restored QuantaGraph.
                      
    Returns:
        Restored root CID string if target_graph is provided, or QuantaGraph if target_graph is None.
    """
    if target_graph is not None:
        cid = folded_cid_or_node.cid if isinstance(folded_cid_or_node, QuantaNode) else folded_cid_or_node
        return target_graph.unfold_subgraph(cid, storage)

    if isinstance(folded_cid_or_node, QuantaNode):
        if not folded_cid_or_node.anchor or not folded_cid_or_node.anchor.startswith("merkle:"):
            raise ValueError("Node is not a valid Merkle fold pointer")
        sub_merkle_cid = folded_cid_or_node.anchor.split("merkle:")[1]
    else:
        cid_str = str(folded_cid_or_node)
        if cid_str.startswith("merkle:"):
            sub_merkle_cid = cid_str.split("merkle:")[1]
        else:
            sub_merkle_cid = cid_str

    if sub_merkle_cid not in storage:
        raise KeyError(f"Sub-graph with Merkle CID '{sub_merkle_cid}' not found in storage")

    stored_data = storage[sub_merkle_cid]
    if isinstance(stored_data, dict):
        sub_graph = QuantaGraph.from_dict(stored_data)
    elif isinstance(stored_data, QuantaGraph):
        sub_graph = stored_data
    else:
        raise TypeError(f"Unsupported storage payload type: {type(stored_data)}")

    actual_merkle = sub_graph.compute_merkle_root()
    if actual_merkle != sub_merkle_cid:
        raise ValueError(
            f"Merkle CID mismatch during unfold: expected {sub_merkle_cid}, but computed {actual_merkle} (storage payload tampered)"
        )
    return sub_graph


