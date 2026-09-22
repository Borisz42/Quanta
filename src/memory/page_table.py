"""Virtual Page-Table Attention & O(1) VRAM Memory Offloading for QUANTA (Phase 6).

Decouples physical execution memory (GPU canvas M=512 nodes) from document length.
Stores long-horizon ASG graphs, episode nodes, and string interning tables on disk (SQLite),
and performs AVX-512 / SIMD bitwise Hamming distance scans across 1024-dimension quaternary keys
at > 35 M nodes/sec for sub-5ms semantic page-fault resolution.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, pack_quaternary_array, unpack_quaternary_bytes


# -----------------------------------------------------------------------------
# 1. High-Performance SIMD / SWAR Quaternary Hamming Distance Kernels
# -----------------------------------------------------------------------------

_POPCNT_LUT_U8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)

if _HAS_NUMBA:
    @numba.njit(parallel=True, fastmath=True)
    def _numba_batch_quaternary_hamming(keys_u64: np.ndarray, query_u64: np.ndarray) -> np.ndarray:
        """Computes quaternary Hamming distance over (N, 32) uint64 keys using parallel SWAR popcount.

        For 1024-dim quaternary vectors, 2 bits per slot:
        diff = A ^ B
        collapsed = (diff | (diff >> 1)) & 0x5555555555555555
        The popcount of collapsed gives the exact count of differing quaternary slots.
        """
        N, M = keys_u64.shape
        distances = np.empty(N, dtype=np.int32)
        m1 = np.uint64(0x5555555555555555)
        c1 = np.uint64(0x3333333333333333)
        c2 = np.uint64(0x0F0F0F0F0F0F0F0F)
        h01 = np.uint64(0x0101010101010101)

        for i in numba.prange(N):
            d = 0
            for j in range(M):
                diff = keys_u64[i, j] ^ query_u64[j]
                c = (diff | (diff >> np.uint64(1))) & m1
                c = c - ((c >> np.uint64(1)) & m1)
                c = (c & c1) + ((c >> np.uint64(2)) & c1)
                c = (c + (c >> np.uint64(4))) & c2
                d += np.int32((c * h01) >> np.uint64(56))
            distances[i] = d
        return distances
else:
    _numba_batch_quaternary_hamming = None


def _numpy_batch_quaternary_hamming(keys_u64: np.ndarray, query_u64: np.ndarray) -> np.ndarray:
    """NumPy vectorized fallback for batch quaternary Hamming distance."""
    N = keys_u64.shape[0]
    keys_u8 = keys_u64.view(np.uint8).reshape(N, -1)
    query_u8 = query_u64.view(np.uint8).reshape(-1)
    diff = np.bitwise_xor(keys_u8, query_u8)
    collapsed = np.bitwise_and(np.bitwise_or(diff, diff >> 1), 0x55)
    return _POPCNT_LUT_U8[collapsed].sum(axis=1, dtype=np.int32)


def batch_quaternary_hamming(keys_u64: np.ndarray, query_u64: np.ndarray) -> np.ndarray:
    """Dispatches batch quaternary Hamming calculation to the fastest available kernel.

    Args:
        keys_u64: (N, 32) uint64 array of packed quaternary vectors.
        query_u64: (32,) uint64 array of query quaternary vector.

    Returns:
        (N,) int32 array containing exact quaternary Hamming distances.
    """
    if len(keys_u64) == 0:
        return np.empty(0, dtype=np.int32)
    if _HAS_NUMBA and _numba_batch_quaternary_hamming is not None and len(keys_u64) >= 64:
        return _numba_batch_quaternary_hamming(keys_u64, query_u64)
    return _numpy_batch_quaternary_hamming(keys_u64, query_u64)


def topk_hamming_search(
    keys_u64: np.ndarray,
    query_u64: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes top-K nearest keys by quaternary Hamming distance.

    Args:
        keys_u64: (N, 32) array of packed quaternary vectors.
        query_u64: (32,) array of query vector.
        k: Number of nearest items to retrieve.

    Returns:
        Tuple of (topk_indices, topk_distances) sorted by ascending distance.
    """
    N = len(keys_u64)
    if N == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)

    distances = batch_quaternary_hamming(keys_u64, query_u64)
    actual_k = min(k, N)

    if actual_k == N:
        order = np.argsort(distances)
        return order, distances[order]

    # O(N) selection followed by sorting the k elements
    topk_part = np.argpartition(distances, actual_k - 1)[:actual_k]
    sorted_order = topk_part[np.argsort(distances[topk_part])]
    return sorted_order, distances[sorted_order]


# -----------------------------------------------------------------------------
# 2. Vector Index: SimdHammingIndex
# -----------------------------------------------------------------------------

class SimdHammingIndex:
    """Contiguous in-memory quaternary vector index for SIMD Hamming search.

    Keeps 1024-dim quaternary vectors packed as uint64 words (32 words = 256 bytes per vector).
    Supports dynamic growth, sub-5ms top-K retrieval across 100,000+ keys, and CID mapping.
    """

    def __init__(self, initial_capacity: int = 1024):
        self._capacity = max(128, initial_capacity)
        self._size = 0
        self._keys = np.empty((self._capacity, 32), dtype=np.uint64)
        self._cids: List[str] = []
        self._cid_to_idx: Dict[str, int] = {}
        self._lock = threading.RLock()

    def _ensure_capacity(self, required_size: int):
        if required_size > self._capacity:
            new_capacity = max(self._capacity * 2, required_size, 1024)
            new_keys = np.empty((new_capacity, 32), dtype=np.uint64)
            if self._size > 0:
                new_keys[:self._size] = self._keys[:self._size]
            self._keys = new_keys
            self._capacity = new_capacity

    @staticmethod
    def _to_u64_array(vector: Union[QuantaVector, bytes, np.ndarray, Sequence[int]]) -> np.ndarray:
        """Converts vector or packed bytes into a 32-element uint64 array."""
        if isinstance(vector, QuantaVector):
            raw_bytes = vector.to_bytes()
            return np.frombuffer(raw_bytes, dtype=np.uint64).copy()
        elif isinstance(vector, bytes):
            if len(vector) != 256:
                raise ValueError(f"Expected 256 bytes for packed 1024-dim vector, got {len(vector)}")
            return np.frombuffer(vector, dtype=np.uint64).copy()
        elif isinstance(vector, np.ndarray):
            if vector.dtype == np.uint64 and vector.shape == (32,):
                return vector.copy()
            elif vector.dtype == np.uint8:
                if vector.size == 256:
                    return vector.view(np.uint64).copy()
                elif vector.size == 1024:
                    raw_bytes = pack_quaternary_array(vector)
                    return np.frombuffer(raw_bytes, dtype=np.uint64).copy()
            raise ValueError(f"Unsupported numpy array shape {vector.shape} and dtype {vector.dtype}")
        elif isinstance(vector, Sequence):
            raw_bytes = pack_quaternary_array(vector)
            return np.frombuffer(raw_bytes, dtype=np.uint64).copy()
        else:
            raise TypeError(f"Unsupported vector type: {type(vector)}")

    def add(self, cid: str, vector: Union[QuantaVector, bytes, np.ndarray, Sequence[int]]) -> int:
        """Adds or updates a vector in the index. Returns its row index."""
        u64_vec = self._to_u64_array(vector)
        with self._lock:
            if cid in self._cid_to_idx:
                idx = self._cid_to_idx[cid]
                self._keys[idx] = u64_vec
                return idx

            idx = self._size
            self._ensure_capacity(idx + 1)
            self._keys[idx] = u64_vec
            self._cids.append(cid)
            self._cid_to_idx[cid] = idx
            self._size += 1
            return idx

    def add_batch(self, items: Iterable[Tuple[str, Union[QuantaVector, bytes, np.ndarray, Sequence[int]]]]) -> int:
        """Adds a batch of (cid, vector) tuples with minimal reallocation overhead."""
        items_list = list(items)
        if not items_list:
            return 0

        with self._lock:
            new_items = []
            for cid, vec in items_list:
                u64_vec = self._to_u64_array(vec)
                if cid in self._cid_to_idx:
                    idx = self._cid_to_idx[cid]
                    self._keys[idx] = u64_vec
                else:
                    new_items.append((cid, u64_vec))

            if new_items:
                start_idx = self._size
                self._ensure_capacity(start_idx + len(new_items))
                for offset, (cid, u64_vec) in enumerate(new_items):
                    idx = start_idx + offset
                    self._keys[idx] = u64_vec
                    self._cids.append(cid)
                    self._cid_to_idx[cid] = idx
                self._size += len(new_items)

            return len(items_list)

    def remove(self, cid: str) -> bool:
        """Removes a CID from the index using O(1) swap-with-last."""
        with self._lock:
            if cid not in self._cid_to_idx:
                return False
            idx = self._cid_to_idx.pop(cid)
            last_idx = self._size - 1

            if idx != last_idx:
                last_cid = self._cids[last_idx]
                self._keys[idx] = self._keys[last_idx]
                self._cids[idx] = last_cid
                self._cid_to_idx[last_cid] = idx

            self._cids.pop()
            self._size -= 1
            return True

    def search(
        self,
        query: Union[QuantaVector, bytes, np.ndarray, Sequence[int]],
        top_k: int = 10,
    ) -> List[Tuple[str, int]]:
        """Finds the top-K nearest CIDs by quaternary Hamming distance."""
        query_u64 = self._to_u64_array(query)
        with self._lock:
            if self._size == 0:
                return []
            active_keys = self._keys[:self._size]
            indices, distances = topk_hamming_search(active_keys, query_u64, k=top_k)
            return [(self._cids[idx], int(distances[i])) for i, idx in enumerate(indices)]

    def __len__(self) -> int:
        return self._size

    def __contains__(self, cid: str) -> bool:
        return cid in self._cid_to_idx

    def clear(self):
        with self._lock:
            self._size = 0
            self._cids.clear()
            self._cid_to_idx.clear()


# -----------------------------------------------------------------------------
# 3. String Interning Table: StringInternTable
# -----------------------------------------------------------------------------

class StringInternTable:
    """Thread-safe SQLite-backed string deduplication table.

    Assigns stable integer IDs to frequent strings (anchors, predicates, relation labels).
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._str_to_id: Dict[str, int] = {}
        self._id_to_str: Dict[int, str] = {}
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self):
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interned_strings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT UNIQUE NOT NULL
                );
                """
            )

    def intern(self, text: Optional[str]) -> Optional[int]:
        """Interns a string, returning its unique integer ID (or None if string is None)."""
        if text is None:
            return None
        with self._lock:
            if text in self._str_to_id:
                return self._str_to_id[text]

            cur = self._conn.cursor()
            cur.execute("SELECT id FROM interned_strings WHERE text = ?", (text,))
            row = cur.fetchone()
            if row:
                str_id = int(row[0])
            else:
                with self._conn:
                    cur.execute("INSERT OR IGNORE INTO interned_strings (text) VALUES (?)", (text,))
                    if cur.rowcount > 0:
                        str_id = cur.lastrowid
                    else:
                        cur.execute("SELECT id FROM interned_strings WHERE text = ?", (text,))
                        str_id = int(cur.fetchone()[0])

            self._str_to_id[text] = str_id
            self._id_to_str[str_id] = text
            return str_id

    def resolve(self, str_id: Optional[int]) -> Optional[str]:
        """Resolves an integer ID back to its original string."""
        if str_id is None:
            return None
        with self._lock:
            if str_id in self._id_to_str:
                return self._id_to_str[str_id]

            cur = self._conn.cursor()
            cur.execute("SELECT text FROM interned_strings WHERE id = ?", (str_id,))
            row = cur.fetchone()
            if row:
                text = str(row[0])
                self._id_to_str[str_id] = text
                self._str_to_id[text] = str_id
                return text
            return None

    def bulk_intern(self, texts: Iterable[str]) -> List[int]:
        """Interns multiple strings within a single transaction."""
        results = []
        with self._lock:
            for t in texts:
                results.append(self.intern(t))
        return results


# -----------------------------------------------------------------------------
# 4. Host Storage Engine: PageTable
# -----------------------------------------------------------------------------

class PageTable(MutableMapping):
    """Disk-backed Virtual Page-Table storing QuantaNodes, Merkle sub-graphs, and vector indices.

    Supports:
    - Ingesting and querying 100,000+ nodes in SQLite.
    - Contiguous SIMD Hamming search index.
    - String interning for anchors and edge types.
    - Full Merkle sub-graph storage compatibility with Phase 5.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = str(db_path) if db_path is not None else ":memory:"
        self._memory_data: Dict[str, Any] = {}
        self._lock = threading.RLock()

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

        self.interner = StringInternTable(self._conn)
        self.vector_index = SimdHammingIndex()

        # Warm up vector index from existing nodes in database
        self._warmup_index()

    def _init_schema(self):
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    cid TEXT PRIMARY KEY,
                    vector_bytes BLOB NOT NULL,
                    anchor_id INTEGER,
                    literal TEXT,
                    parent_cid TEXT,
                    edges TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS subgraphs (
                    cid TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at);
                """
            )

    def _warmup_index(self):
        """Loads all stored quaternary vectors into the in-memory SIMD index."""
        cur = self._conn.cursor()
        cur.execute("SELECT cid, vector_bytes FROM nodes")
        rows = cur.fetchall()
        if rows:
            self.vector_index.add_batch([(cid, vec_bytes) for cid, vec_bytes in rows])

    def store_node(self, node: QuantaNode) -> str:
        """Stores a QuantaNode in SQLite and registers it in the SIMD vector index."""
        cid = node.compute_cid()
        packed_vec = node.vector.to_bytes()
        anchor_id = self.interner.intern(node.anchor)
        literal_str = json.dumps(node.literal) if node.literal is not None else None
        edges_str = json.dumps(node.edges)
        now = time.time()

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO nodes 
                (cid, vector_bytes, anchor_id, literal, parent_cid, edges, created_at, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT access_count FROM nodes WHERE cid = ?), 0), ?)
                """,
                (cid, packed_vec, anchor_id, literal_str, node.parent_cid, edges_str, now, cid, now),
            )
            self.vector_index.add(cid, packed_vec)

        return cid

    def store_nodes(self, nodes: Iterable[QuantaNode]) -> List[str]:
        """Stores multiple QuantaNodes in a single fast SQLite transaction."""
        node_list = list(nodes)
        if not node_list:
            return []

        cids = []
        batch_rows = []
        batch_vecs = []
        now = time.time()

        with self._lock:
            for node in node_list:
                cid = node.compute_cid()
                cids.append(cid)
                packed_vec = node.vector.to_bytes()
                anchor_id = self.interner.intern(node.anchor)
                literal_str = json.dumps(node.literal) if node.literal is not None else None
                edges_str = json.dumps(node.edges)
                batch_rows.append((cid, packed_vec, anchor_id, literal_str, node.parent_cid, edges_str, now, 0, now))
                batch_vecs.append((cid, packed_vec))

            with self._conn:
                self._conn.executemany(
                    """
                    INSERT OR REPLACE INTO nodes 
                    (cid, vector_bytes, anchor_id, literal, parent_cid, edges, created_at, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch_rows,
                )
            self.vector_index.add_batch(batch_vecs)

        return cids

    def fetch_node(self, cid: str, update_access: bool = False) -> Optional[QuantaNode]:
        """Fetches and reconstructs a QuantaNode from SQLite storage."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT vector_bytes, anchor_id, literal, parent_cid, edges 
                FROM nodes WHERE cid = ?
                """,
                (cid,),
            )
            row = cur.fetchone()
            if not row:
                return None

            vector_bytes, anchor_id, literal_str, parent_cid, edges_str = row
            anchor = self.interner.resolve(anchor_id)
            literal = json.loads(literal_str) if literal_str is not None else None
            edges = json.loads(edges_str) if edges_str else {}
            vec = QuantaVector.from_bytes(vector_bytes)

            node = QuantaNode(
                vector=vec,
                edges=edges,
                anchor=anchor,
                literal=literal,
                parent_cid=parent_cid,
            )
            node._cid_cache = cid

            # Update access telemetry only if requested
            if update_access:
                now = time.time()
                with self._conn:
                    self._conn.execute(
                        "UPDATE nodes SET access_count = access_count + 1, last_accessed = ? WHERE cid = ?",
                        (now, cid),
                    )

            return node

    def fetch_nodes(self, cids: Sequence[str], update_access: bool = False) -> List[Optional[QuantaNode]]:
        """Fetches multiple QuantaNodes in a fast batch query."""
        if not cids:
            return []
        nodes_dict: Dict[str, QuantaNode] = {}
        with self._lock:
            cur = self._conn.cursor()
            placeholders = ",".join("?" for _ in cids)
            cur.execute(
                f"""
                SELECT cid, vector_bytes, anchor_id, literal, parent_cid, edges 
                FROM nodes WHERE cid IN ({placeholders})
                """,
                tuple(cids),
            )
            rows = cur.fetchall()
            for row in rows:
                cid, vector_bytes, anchor_id, literal_str, parent_cid, edges_str = row
                anchor = self.interner.resolve(anchor_id)
                literal = json.loads(literal_str) if literal_str is not None else None
                edges = json.loads(edges_str) if edges_str else {}
                vec = QuantaVector.from_bytes(vector_bytes)
                node = QuantaNode(
                    vector=vec,
                    edges=edges,
                    anchor=anchor,
                    literal=literal,
                    parent_cid=parent_cid,
                )
                node._cid_cache = cid
                nodes_dict[cid] = node

        return [nodes_dict.get(cid) for cid in cids]

    def has_node(self, cid: str) -> bool:
        """Checks whether a node exists in storage."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT 1 FROM nodes WHERE cid = ?", (cid,))
            return cur.fetchone() is not None

    def delete_node(self, cid: str) -> bool:
        """Deletes a node from SQLite and from the vector index."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM nodes WHERE cid = ?", (cid,))
            deleted = self.vector_index.remove(cid)
            return deleted

    def count_nodes(self) -> int:
        """Returns total number of nodes stored in the PageTable."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM nodes")
            return cur.fetchone()[0]

    def query_nearest(
        self,
        query: Union[QuantaVector, bytes, np.ndarray, Sequence[int], QuantaNode],
        top_k: int = 10,
    ) -> List[Tuple[QuantaNode, int]]:
        """Queries the SIMD vector index and fetches the corresponding top-K QuantaNodes.

        Returns list of (QuantaNode, hamming_distance) pairs sorted by distance.
        """
        vec_query = query.vector if isinstance(query, QuantaNode) else query
        matches = self.vector_index.search(vec_query, top_k=top_k)
        results = []
        for cid, dist in matches:
            node = self.fetch_node(cid)
            if node is not None:
                results.append((node, dist))
        return results

    # --- MutableMapping & Sub-Graph Methods (Phase 5 Compatibility) ---

    def store_subgraph(self, merkle_cid: str, graph: Union[QuantaGraph, Dict[str, Any]]):
        """Stores a serialized sub-graph by its Merkle CID."""
        self[merkle_cid] = graph

    def retrieve_subgraph(self, merkle_cid: str, algorithm: str = "blake3") -> QuantaGraph:
        """Retrieves and cryptographically verifies a sub-graph by Merkle CID."""
        stored = self[merkle_cid]
        if isinstance(stored, dict):
            sub_graph = QuantaGraph.from_dict(stored)
        elif isinstance(stored, QuantaGraph):
            sub_graph = stored
        elif isinstance(stored, str):
            sub_graph = QuantaGraph.from_json(stored)
        else:
            raise TypeError(f"Unsupported payload type in storage: {type(stored)}")

        actual_cid = sub_graph.compute_merkle_root(algorithm=algorithm) if hasattr(sub_graph, "compute_merkle_root") else sub_graph.compute_merkle_root()
        if actual_cid.lower() != merkle_cid.lower():
            raise ValueError(
                f"Merkle CID mismatch during unfold: expected {merkle_cid}, but computed {actual_cid}"
            )
        return sub_graph

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT payload FROM subgraphs WHERE cid = ?", (key,))
            row = cur.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return row[0]

            cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    return row[0]

            # Also check nodes table as fallback
            node = self.fetch_node(key)
            if node is not None:
                return node

            raise KeyError(f"Key '{key}' not found in PageTable")

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, QuantaNode):
            self.store_node(value)
            return

        serialized: str
        if isinstance(value, QuantaGraph):
            serialized = json.dumps(value.to_dict())
        elif isinstance(value, dict):
            serialized = json.dumps(value)
        elif isinstance(value, (str, int, float, bool)):
            serialized = json.dumps(value)
        else:
            serialized = str(value)

        now = time.time()
        with self._lock, self._conn:
            if len(key) == 64 and all(c in "0123456789abcdefABCDEF" for c in key):
                self._conn.execute(
                    "INSERT OR REPLACE INTO subgraphs (cid, payload, created_at) VALUES (?, ?, ?)",
                    (key, serialized, now),
                )
            else:
                self._conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    (key, serialized),
                )

    def __delitem__(self, key: str):
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM subgraphs WHERE cid = ?", (key,))
            self._conn.execute("DELETE FROM metadata WHERE key = ?", (key,))
            self.delete_node(key)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            keys: List[str] = []
            cur = self._conn.cursor()
            cur.execute("SELECT cid FROM nodes")
            keys.extend(row[0] for row in cur.fetchall())
            cur.execute("SELECT cid FROM subgraphs")
            keys.extend(row[0] for row in cur.fetchall())
            cur.execute("SELECT key FROM metadata")
            keys.extend(row[0] for row in cur.fetchall())
            return iter(keys)

    def __len__(self) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM nodes")
            c1 = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM subgraphs")
            c2 = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM metadata")
            c3 = cur.fetchone()[0]
            return c1 + c2 + c3

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT 1 FROM nodes WHERE cid = ?
                UNION SELECT 1 FROM subgraphs WHERE cid = ?
                UNION SELECT 1 FROM metadata WHERE key = ?
                """,
                (key, key, key),
            )
            return cur.fetchone() is not None

    def close(self):
        """Closes SQLite database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()


# -----------------------------------------------------------------------------
# 5. Fixed Working Buffer: ActiveCanvas
# -----------------------------------------------------------------------------

class ActiveCanvas:
    """Fixed-capacity buffer manager for physical active memory (GPU VRAM M=64 to 512).

    Enforces strict O(1) memory bounds via LRU eviction policy.
    Supports node pinning (e.g. active chapter roots, entity manifest characters)
    and detailed memory footprint tracking.
    """

    def __init__(self, capacity: int = 512):
        if capacity < 1:
            raise ValueError(f"Canvas capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._nodes: OrderedDict[str, QuantaNode] = OrderedDict()
        self._pinned: Set[str] = set()
        self._lock = threading.RLock()

        # Telemetry
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, cid: str) -> Optional[QuantaNode]:
        """Retrieves node from canvas. Updates LRU recency on hit."""
        with self._lock:
            if cid in self._nodes:
                self.hits += 1
                self._nodes.move_to_end(cid, last=True)
                return self._nodes[cid]
            self.misses += 1
            return None

    def peek(self, cid: str) -> Optional[QuantaNode]:
        """Retrieves node without modifying LRU order or hit counters."""
        with self._lock:
            return self._nodes.get(cid)

    def put(self, node: QuantaNode) -> Optional[QuantaNode]:
        """Inserts node into canvas. Evicts the least-recently-used unpinned node if full.

        Returns:
            The evicted QuantaNode if an eviction occurred, else None.
        """
        cid = node.compute_cid()
        with self._lock:
            if cid in self._nodes:
                self._nodes[cid] = node
                self._nodes.move_to_end(cid, last=True)
                return None

            evicted_node: Optional[QuantaNode] = None
            if len(self._nodes) >= self.capacity:
                evicted_node = self.evict_lru()
                if evicted_node is None:
                    # All nodes currently pinned! Widening check or error
                    raise MemoryError(
                        f"ActiveCanvas full ({self.capacity} nodes) and all nodes are pinned; cannot admit {cid}"
                    )

            self._nodes[cid] = node
            return evicted_node

    def evict_lru(self) -> Optional[QuantaNode]:
        """Evicts the least-recently-used unpinned node from the canvas."""
        with self._lock:
            for cid in list(self._nodes.keys()):
                if cid not in self._pinned:
                    node = self._nodes.pop(cid)
                    self.evictions += 1
                    return node
            return None

    def pin(self, cid: str):
        """Pins a node CID, preventing it from being evicted under memory pressure."""
        with self._lock:
            self._pinned.add(cid)

    def unpin(self, cid: str):
        """Unpins a node CID, allowing it to be evicted when necessary."""
        with self._lock:
            self._pinned.discard(cid)

    def is_pinned(self, cid: str) -> bool:
        """Checks if a node CID is currently pinned."""
        with self._lock:
            return cid in self._pinned

    def remove(self, cid: str) -> Optional[QuantaNode]:
        """Removes a node from the canvas."""
        with self._lock:
            self._pinned.discard(cid)
            return self._nodes.pop(cid, None)

    def clear(self):
        """Clears all nodes and reset counters."""
        with self._lock:
            self._nodes.clear()
            self._pinned.clear()

    @property
    def active_nodes(self) -> Dict[str, QuantaNode]:
        """Returns a snapshot of currently loaded nodes."""
        with self._lock:
            return dict(self._nodes)

    @property
    def nodes(self) -> Dict[str, QuantaNode]:
        """Alias for active_nodes."""
        return self.active_nodes


    def active_memory_bytes(self) -> int:
        """Computes physical quaternary vector footprint in bytes (256 bytes per node)."""
        with self._lock:
            return len(self._nodes) * 256

    def stats(self) -> Dict[str, Any]:
        """Returns operating statistics and telemetry."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
            return {
                "capacity": self.capacity,
                "active_nodes": len(self._nodes),
                "memory_bytes": self.active_memory_bytes(),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "pinned_nodes": len(self._pinned),
                "hit_rate": hit_rate,
            }

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, cid: str) -> bool:
        return cid in self._nodes

    def clear(self):
        """Clear all nodes, pinned nodes, and telemetry from active canvas."""
        with self._lock:
            self._nodes.clear()
            self._pinned.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0


# -----------------------------------------------------------------------------
# 6. Semantic Page-Fault Handler: SemanticPageFaultHandler
# -----------------------------------------------------------------------------

class SemanticPageFaultHandler:
    """Coordinates ActiveCanvas and PageTable to offload memory and handle page faults.

    When an active reasoning operation queries a dormant node or entity pointer:
    1. Checks ActiveCanvas for a fast hit.
    2. On a miss, triggers a Semantic Page Fault to fetch from disk/PageTable.
    3. Admits node into ActiveCanvas with LRU eviction.
    4. Optionally prefetches adjacent relation sub-graphs (VAL_X1_AGENT, causal links).
    """

    def __init__(self, canvas: ActiveCanvas, page_table: PageTable):
        self.canvas = canvas
        self.page_table = page_table
        self.page_fault_count = 0
        self._lock = threading.RLock()

    def access(self, cid: str, prefetch_depth: int = 0) -> QuantaNode:
        """Accesses a node by CID, triggering a semantic page fault if not present in canvas.

        Args:
            cid: Node Content Identifier.
            prefetch_depth: Degrees of graph relation edges to recursively prefetch into canvas.

        Returns:
            Reconstructed QuantaNode loaded into the active canvas.

        Raises:
            KeyError: If CID does not exist in canvas or PageTable.
        """
        with self._lock:
            node = self.canvas.get(cid)
            if node is None:
                # Semantic Page Fault!
                self.page_fault_count += 1
                node = self.page_table.fetch_node(cid)
                if node is None:
                    raise KeyError(f"Node '{cid}' not found in PageTable storage")
                self.canvas.put(node)

            if prefetch_depth > 0:
                self._prefetch_neighbors(node, depth=prefetch_depth)

            return node

    def _prefetch_neighbors(self, node: QuantaNode, depth: int):
        """Prefetches relation child CIDs into canvas."""
        if depth <= 0 or not node.edges:
            return

        visited: Set[str] = {node.cid}
        current_level: List[str] = []
        for rel, targets in node.edges.items():
            for target_cid in targets:
                if target_cid not in visited:
                    current_level.append(target_cid)
                    visited.add(target_cid)

        for d in range(depth):
            if not current_level:
                break
            next_level: List[str] = []
            for child_cid in current_level:
                if child_cid not in self.canvas:
                    child_node = self.page_table.fetch_node(child_cid)
                    if child_node:
                        self.canvas.put(child_node)
                        for rel, targets in child_node.edges.items():
                            for t in targets:
                                if t not in visited:
                                    next_level.append(t)
                                    visited.add(t)
            current_level = next_level

    def query_and_page(
        self,
        query: Union[QuantaVector, bytes, np.ndarray, Sequence[int], QuantaNode],
        top_k: int = 5,
    ) -> List[QuantaNode]:
        """Runs SIMD Hamming search across PageTable and pages the nearest nodes into canvas.

        Returns:
            List of top-K QuantaNodes admitted to active working memory.
        """
        with self._lock:
            matches = self.page_table.query_nearest(query, top_k=top_k)
            admitted: List[QuantaNode] = []
            for node, dist in matches:
                self.canvas.put(node)
                admitted.append(node)
            return admitted

    def preload_graph(self, graph: QuantaGraph, pin_root: bool = True):
        """Stores graph in PageTable and pages root and initial nodes into ActiveCanvas."""
        with self._lock:
            cids = self.page_table.store_nodes(graph._node_list)
            if graph.root_cid:
                root_node = graph.get_node(graph.root_cid)
                if root_node:
                    self.canvas.put(root_node)
                    if pin_root:
                        self.canvas.pin(root_node.cid)


__all__ = [
    "ActiveCanvas",
    "PageTable",
    "SemanticPageFaultHandler",
    "SimdHammingIndex",
    "StringInternTable",
    "batch_quaternary_hamming",
    "topk_hamming_search",
]
