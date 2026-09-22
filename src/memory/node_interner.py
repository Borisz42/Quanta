"""Canonical Node Interning & Global Hash-Consing Pool for QUANTA (Phase 7 / Section 1).

Provides thread-safe Flyweight interning for QuantaNodes across chunk parsing,
transduction, and translation cycles. Decouples ephemeral Band 2 variable registers
(VAR_SLOT_X0..X7) from immutable semantic identity, maximizing node reuse and
minimizing heap allocations.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import weakref

from core.asg import QuantaNode
from core.types import QuantaVector


class CanonicalNodeInterner:
    """Thread-safe, two-tier canonical hash-consing interner for QuantaNodes.

    - Tier 1: In-memory Flyweight cache using weakref.WeakValueDictionary.
      Nodes remain cached as long as active graphs, callers, or working memory
      retain a reference to them, preventing memory leaks while guaranteeing
      single-instance pointer identity.
    - Tier 2: Persistent SQLite-backed store (or PageTable integration) for
      cross-process, cross-chapter, or long-horizon recall.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        page_table: Optional[Any] = None,
    ):
        self._lock = threading.RLock()
        self._hits: int = 0
        self._misses: int = 0

        # Tier 1: Weak value dictionaries
        # canonical_cid -> QuantaNode
        self._weak_cid_cache: weakref.WeakValueDictionary[str, QuantaNode] = weakref.WeakValueDictionary()
        # (anchor, normalized_literal, core_vector_bytes) -> QuantaNode
        self._weak_semantic_cache: weakref.WeakValueDictionary[Tuple[str, str, bytes], QuantaNode] = weakref.WeakValueDictionary()
        # Fast lookup mapping: (anchor, normalized_literal) -> canonical_cid
        self._name_to_cid: Dict[Tuple[str, str], str] = {}

        # Tier 2: PageTable or standalone SQLite database
        self.page_table = page_table
        self._db_path = str(db_path) if db_path is not None else None
        self._sqlite_conn: Optional[sqlite3.Connection] = None

        if self.page_table is None and self._db_path is not None:
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._init_sqlite_schema()

    def _init_sqlite_schema(self):
        """Initializes standalone SQLite schema for Tier 2 canonical node store."""
        if self._sqlite_conn is None:
            return
        with self._lock, self._sqlite_conn:
            self._sqlite_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS canonical_interned_nodes (
                    canonical_cid TEXT PRIMARY KEY,
                    vector_bytes BLOB NOT NULL,
                    anchor TEXT,
                    literal TEXT,
                    edges TEXT,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
                """
            )
            self._sqlite_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_canonical_anchor_lit ON canonical_interned_nodes(anchor, literal)"
            )

    @staticmethod
    def _extract_core_vector_bytes(node: QuantaNode) -> bytes:
        """Extracts 256-byte vector with Band 2 execution registers masked to 0."""
        vec_bytes = bytearray(node.vector.to_bytes())
        # Band 2: slots 256-383 -> bytes 64..95
        if len(vec_bytes) >= 96:
            vec_bytes[64:96] = b"\x00" * 32
        return bytes(vec_bytes)

    @staticmethod
    def _normalize_literal(literal: Any) -> str:
        """Normalizes literal payload for canonical semantic key lookup."""
        if literal is None:
            return ""
        if isinstance(literal, str):
            return literal.strip().lower()
        return json.dumps(literal, sort_keys=True).lower()

    def intern_node(self, node: QuantaNode) -> QuantaNode:
        """Interns a QuantaNode in the canonical pool.

        If an equivalent node (matching canonical CID or semantic signature)
        already exists in Tier 1 or Tier 2, returns the pre-existing QuantaNode.
        Otherwise, registers the node in the pool and returns it.
        """
        canonical_cid = node.canonical_cid
        anchor = node.anchor or ""
        norm_lit = self._normalize_literal(node.literal)
        core_bytes = self._extract_core_vector_bytes(node)
        semantic_key = (anchor, norm_lit, core_bytes)
        name_key = (anchor, norm_lit)

        with self._lock:
            # 1. Tier 1 Cache Check (Fast path)
            existing = self._weak_cid_cache.get(canonical_cid)
            if existing is None:
                existing = self._weak_semantic_cache.get(semantic_key)

            if existing is not None:
                self._hits += 1
                # Update fast name mapping
                self._name_to_cid[name_key] = canonical_cid
                return existing

            # 2. Tier 2 Cache Check (PageTable or SQLite)
            tier2_node: Optional[QuantaNode] = None
            if self.page_table is not None:
                tier2_node = self.page_table.fetch_node(canonical_cid)
            elif self._sqlite_conn is not None:
                tier2_node = self._fetch_from_sqlite(canonical_cid)

            if tier2_node is not None:
                self._hits += 1
                self._weak_cid_cache[canonical_cid] = tier2_node
                self._weak_semantic_cache[semantic_key] = tier2_node
                self._name_to_cid[name_key] = canonical_cid
                return tier2_node

            # 3. Cache Miss: Register new node
            self._misses += 1
            self._weak_cid_cache[canonical_cid] = node
            self._weak_semantic_cache[semantic_key] = node
            self._name_to_cid[name_key] = canonical_cid

            # Persist to Tier 2 if configured
            if self.page_table is not None:
                self.page_table.store_node(node)
            elif self._sqlite_conn is not None:
                self._store_to_sqlite(node, canonical_cid)

            return node

    def lookup(
        self,
        anchor: Optional[str] = None,
        literal: Optional[str] = None,
        canonical_cid: Optional[str] = None,
    ) -> Optional[QuantaNode]:
        """Looks up an interned node by canonical CID or (anchor, literal)."""
        with self._lock:
            if canonical_cid is not None:
                # Tier 1
                node = self._weak_cid_cache.get(canonical_cid)
                if node is not None:
                    return node
                # Tier 2
                if self.page_table is not None:
                    node = self.page_table.fetch_node(canonical_cid)
                    if node is not None:
                        self._weak_cid_cache[canonical_cid] = node
                        return node
                elif self._sqlite_conn is not None:
                    node = self._fetch_from_sqlite(canonical_cid)
                    if node is not None:
                        self._weak_cid_cache[canonical_cid] = node
                        return node
                return None

            # Lookup by (anchor, literal)
            anchor_norm = anchor or ""
            norm_lit = self._normalize_literal(literal)
            name_key = (anchor_norm, norm_lit)

            cid = self._name_to_cid.get(name_key)
            if cid is not None:
                return self.lookup(canonical_cid=cid)

            # Fallback linear scan over Tier 1 weak semantic cache
            for (k_anchor, k_lit, _), candidate in list(self._weak_semantic_cache.items()):
                if k_anchor == anchor_norm and k_lit == norm_lit:
                    return candidate

            # Tier 2 fallback search
            if self._sqlite_conn is not None:
                cur = self._sqlite_conn.cursor()
                cur.execute(
                    "SELECT canonical_cid FROM canonical_interned_nodes WHERE anchor = ? AND literal = ? LIMIT 1",
                    (anchor_norm, norm_lit),
                )
                row = cur.fetchone()
                if row:
                    return self.lookup(canonical_cid=row[0])

            return None

    def lookup_by_literal(self, literal: str, anchor: Optional[str] = None) -> Optional[QuantaNode]:
        """Convenience method for name-based concept lookup."""
        return self.lookup(anchor=anchor, literal=literal)

    def _fetch_from_sqlite(self, canonical_cid: str) -> Optional[QuantaNode]:
        """Reconstructs QuantaNode from Tier 2 SQLite table."""
        if self._sqlite_conn is None:
            return None
        cur = self._sqlite_conn.cursor()
        cur.execute(
            "SELECT vector_bytes, anchor, literal, edges FROM canonical_interned_nodes WHERE canonical_cid = ?",
            (canonical_cid,),
        )
        row = cur.fetchone()
        if not row:
            return None
        vec_bytes, anchor, literal_str, edges_str = row
        vec = QuantaVector.from_bytes(vec_bytes)
        edges = json.loads(edges_str) if edges_str else {}
        literal = json.loads(literal_str) if literal_str else None
        node = QuantaNode(vector=vec, anchor=anchor, literal=literal, edges=edges)
        # Touch access count
        self._sqlite_conn.execute(
            "UPDATE canonical_interned_nodes SET access_count = access_count + 1 WHERE canonical_cid = ?",
            (canonical_cid,),
        )
        return node

    def _store_to_sqlite(self, node: QuantaNode, canonical_cid: str):
        """Stores QuantaNode into Tier 2 SQLite table."""
        if self._sqlite_conn is None:
            return
        edges_str = json.dumps(node.edges)
        lit_str = json.dumps(node.literal) if node.literal is not None else None
        vec_bytes = node.vector.to_bytes()
        now = time.time()
        with self._sqlite_conn:
            self._sqlite_conn.execute(
                """
                INSERT OR REPLACE INTO canonical_interned_nodes
                (canonical_cid, vector_bytes, anchor, literal, edges, created_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (canonical_cid, vec_bytes, node.anchor, lit_str, edges_str, now),
            )

    def stats(self) -> Dict[str, Any]:
        """Returns interning performance metrics and cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            reuse_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_queries": total,
                "reuse_rate": reuse_rate,
                "tier1_cached": len(self._weak_cid_cache),
                "total_interned": len(self._weak_cid_cache),
            }

    def reset_stats(self):
        """Resets hit/miss counters."""
        with self._lock:
            self._hits = 0
            self._misses = 0

    def clear(self):
        """Clears in-memory caches and resets stats."""
        with self._lock:
            self._weak_cid_cache.clear()
            self._weak_semantic_cache.clear()
            self._name_to_cid.clear()
            self._hits = 0
            self._misses = 0
            if self._sqlite_conn is not None:
                with self._sqlite_conn:
                    self._sqlite_conn.execute("DELETE FROM canonical_interned_nodes")


# ---------------------------------------------------------------------------
# Global Process-Wide Interner Singleton
# ---------------------------------------------------------------------------

_GLOBAL_INTERNER: Optional[CanonicalNodeInterner] = None
_GLOBAL_LOCK = threading.Lock()


def get_global_interner() -> CanonicalNodeInterner:
    """Gets or initializes the global CanonicalNodeInterner singleton."""
    global _GLOBAL_INTERNER
    if _GLOBAL_INTERNER is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_INTERNER is None:
                _GLOBAL_INTERNER = CanonicalNodeInterner()
                QuantaNode.set_default_interner(_GLOBAL_INTERNER)
    return _GLOBAL_INTERNER


def set_global_interner(interner: CanonicalNodeInterner):
    """Sets the global CanonicalNodeInterner singleton."""
    global _GLOBAL_INTERNER
    with _GLOBAL_LOCK:
        _GLOBAL_INTERNER = interner
        QuantaNode.set_default_interner(interner)


def reset_global_interner():
    """Resets the global CanonicalNodeInterner singleton."""
    global _GLOBAL_INTERNER
    with _GLOBAL_LOCK:
        if _GLOBAL_INTERNER is not None:
            _GLOBAL_INTERNER.clear()
        _GLOBAL_INTERNER = CanonicalNodeInterner()
        QuantaNode.set_default_interner(_GLOBAL_INTERNER)
