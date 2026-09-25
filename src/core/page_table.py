"""Virtual Page-Table and Content-Addressed Sub-Graph Storage for QUANTA (Phase 5).

Provides persistent and in-memory storage for folded ASG sub-graphs, episode nodes,
and hierarchical Merkle roots with cryptographic integrity verification.
"""

from __future__ import annotations

from collections.abc import MutableMapping
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from core.asg import QuantaGraph, QuantaNode


class PageTableStorage(MutableMapping):
    """Content-addressed storage engine for QuantaGraphs and Merkle fold trees.

    Supports both in-memory dict semantics and disk-persisted SQLite tables.
    Provides automatic cryptographic hash verification on retrieval.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initializes storage.
        
        Args:
            db_path: Path to SQLite database file. If None or ':memory:', operates in-memory.
        """
        self.db_path = str(db_path) if db_path is not None else None
        self._memory_data: Dict[str, Any] = {}
        self._sqlite_conn: Optional[sqlite3.Connection] = None

        if self.db_path is not None and self.db_path != ":memory:":
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._init_sqlite_schema()
        elif self.db_path == ":memory:":
            self._sqlite_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._init_sqlite_schema()

    def _init_sqlite_schema(self):
        """Initializes SQLite tables for sub-graphs, nodes, and hierarchy records."""
        if not self._sqlite_conn:
            return
        with self._sqlite_conn:
            self._sqlite_conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS subgraphs (
                    cid TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    cid TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def __getitem__(self, key: str) -> Any:
        if self._sqlite_conn:
            cur = self._sqlite_conn.cursor()
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

            raise KeyError(f"Key '{key}' not found in storage")
        
        if key in self._memory_data:
            return self._memory_data[key]
        raise KeyError(f"Key '{key}' not found in storage")

    def __setitem__(self, key: str, value: Any):
        serialized: str
        if isinstance(value, QuantaGraph):
            payload_dict = value.to_dict()
            serialized = json.dumps(payload_dict)
            self._memory_data[key] = payload_dict
        elif isinstance(value, QuantaNode):
            payload_dict = value.to_dict()
            serialized = json.dumps(payload_dict)
            self._memory_data[key] = payload_dict
        elif isinstance(value, dict):
            serialized = json.dumps(value)
            self._memory_data[key] = value
        elif isinstance(value, (str, int, float, bool)):
            serialized = json.dumps(value)
            self._memory_data[key] = value
        else:
            serialized = str(value)
            self._memory_data[key] = value

        if self._sqlite_conn:
            now = time.time()
            with self._sqlite_conn:
                if len(key) == 64 and all(c in "0123456789abcdefABCDEF" for c in key):
                    self._sqlite_conn.execute(
                        "INSERT OR REPLACE INTO subgraphs (cid, payload, created_at) VALUES (?, ?, ?)",
                        (key, serialized, now),
                    )
                else:
                    self._sqlite_conn.execute(
                        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                        (key, serialized),
                    )

    def __delitem__(self, key: str):
        if self._sqlite_conn:
            with self._sqlite_conn:
                self._sqlite_conn.execute("DELETE FROM subgraphs WHERE cid = ?", (key,))
                self._sqlite_conn.execute("DELETE FROM metadata WHERE key = ?", (key,))
        self._memory_data.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        if self._sqlite_conn:
            keys: List[str] = []
            cur = self._sqlite_conn.cursor()
            cur.execute("SELECT cid FROM subgraphs")
            keys.extend(row[0] for row in cur.fetchall())
            cur.execute("SELECT key FROM metadata")
            keys.extend(row[0] for row in cur.fetchall())
            return iter(keys)
        return iter(self._memory_data)

    def __len__(self) -> int:
        if self._sqlite_conn:
            cur = self._sqlite_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM subgraphs")
            c1 = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM metadata")
            c2 = cur.fetchone()[0]
            return c1 + c2
        return len(self._memory_data)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if self._sqlite_conn:
            cur = self._sqlite_conn.cursor()
            cur.execute("SELECT 1 FROM subgraphs WHERE cid = ? UNION SELECT 1 FROM metadata WHERE key = ?", (key, key))
            return cur.fetchone() is not None
        return key in self._memory_data

    def store_subgraph(self, merkle_cid: str, graph: Union[QuantaGraph, Dict[str, Any]]):
        """Stores a serialized sub-graph by its Merkle CID."""
        self[merkle_cid] = graph

    def retrieve_subgraph(self, merkle_cid: str, algorithm: str = "blake3") -> QuantaGraph:
        """Retrieves and verifies the sub-graph against its Merkle CID.

        Args:
            merkle_cid: Expected Merkle CID.
            algorithm: Hash algorithm ('blake3' or 'sha256').

        Returns:
            Reconstructed QuantaGraph.

        Raises:
            KeyError: If CID is not found.
            ValueError: If payload has been tampered with.
        """
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
                f"Merkle CID mismatch during unfold: expected {merkle_cid}, but computed {actual_cid} (storage payload tampered)"
            )
        return sub_graph

    def close(self):
        """Closes SQLite database connection if open."""
        if self._sqlite_conn:
            self._sqlite_conn.close()
            self._sqlite_conn = None


def __getattr__(name: str):
    if name == "PageTable":
        from memory.page_table import PageTable
        return PageTable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PageTableStorage",
    "PageTable",
]

