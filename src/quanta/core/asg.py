"""Abstract Syntax Graph (ASG) and Content-Addressed Merkle nodes for QUANTA."""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import blake3

from quanta.core.types import QuantaVector, QuaternaryValue
from quanta.core.slots import SLOT_NAME_TO_INDEX, get_slot_by_name


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
        from quanta.core.slots import get_slot_by_index
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
        """Retrieves node by CID."""
        if cid in self._cid_to_node:
            return self._cid_to_node[cid]
        for n in self._node_list:
            if n.cid == cid:
                return n
        return None

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

        # Update cache indices
        self._cid_to_node[new_src_cid] = src_node
        self._cid_to_node[old_src_cid] = src_node
        if self.root_cid == old_src_cid:
            self.root_cid = new_src_cid

        return new_src_cid

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

    def to_proposition_vector(self) -> QuantaVector:
        """Aggregates all active slot values across the entire ASG graph into a single proposition vector.
        
        Follows quaternary lattice algebra:
        - 0 (IRRELEVANT) + X -> X
        - 1 (TRUE) + 1 -> 1
        - 2 (FALSE) + 2 -> 2
        - 1 (TRUE) + 2 (FALSE) -> 3 (UNKNOWN / CONFLICT)
        - X + 3 (UNKNOWN) -> 3
        """
        vec = QuantaVector.zeros()
        for node in self.nodes.values():
            for slot_idx, val in node.vector.active_slots().items():
                curr = vec[slot_idx]
                if curr == QuaternaryValue.IRRELEVANT:
                    vec[slot_idx] = val
                elif curr == val:
                    continue
                elif (curr == QuaternaryValue.TRUE and val == QuaternaryValue.FALSE) or \
                     (curr == QuaternaryValue.FALSE and val == QuaternaryValue.TRUE):
                    vec[slot_idx] = QuaternaryValue.UNKNOWN
                elif curr == QuaternaryValue.UNKNOWN or val == QuaternaryValue.UNKNOWN:
                    vec[slot_idx] = QuaternaryValue.UNKNOWN
        return vec

    def __len__(self) -> int:
        return len(self._node_list)

    def __repr__(self) -> str:
        return f"QuantaGraph(nodes={len(self._node_list)}, root_cid={self.root_cid[:8] if self.root_cid else 'None'})"
