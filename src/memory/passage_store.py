"""Passage storage engine and dual-node bipartite provenance registry for QUANTA SVM.

Implements immutable raw source passage nodes (V_passage) with document identifiers,
character offsets, and high-performance text span retrieval.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

INGESTION_STATUS_COARSE_ONLY = "COARSE_ONLY"
INGESTION_STATUS_PENDING = "PENDING"
INGESTION_STATUS_FULL = "FULL"
VALID_INGESTION_STATUSES = {
    INGESTION_STATUS_COARSE_ONLY,
    INGESTION_STATUS_PENDING,
    INGESTION_STATUS_FULL,
}


@dataclass(slots=True)
class PassageRecord:
    """Immutable passage record representing a raw text node (V_passage).
    
    Attributes:
        passage_id: Unique passage identifier (e.g. 'P104').
        doc_id: Source document or corpus identifier (e.g. 'doc_physics_run_02').
        char_span: Absolute character range (start, end) within the source document.
        text: Raw verbatim text content of the passage.
        created_at: UNIX epoch timestamp of creation.
        granularity: Granularity level ('MICRO' or 'MACRO').
        parent_macro_id: Optional parent macro passage identifier if this is a micro passage.
        concept_codes: Grounded concept integer codes (uint16) for this passage.
    """

    passage_id: str
    doc_id: str
    char_span: Tuple[int, int]
    text: str
    created_at: float = field(default_factory=time.time)
    granularity: str = "MICRO"
    parent_macro_id: Optional[str] = None
    concept_codes: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if isinstance(self.char_span, list):
            object.__setattr__(self, "char_span", tuple(self.char_span))
        if len(self.char_span) != 2:
            raise ValueError(f"char_span must be a 2-tuple (start, end), got {self.char_span}")
        if self.char_span[0] > self.char_span[1]:
            raise ValueError(f"char_span start ({self.char_span[0]}) cannot exceed end ({self.char_span[1]})")
        if isinstance(self.concept_codes, (list, set)):
            object.__setattr__(self, "concept_codes", tuple(int(c) for c in self.concept_codes))
        elif not isinstance(self.concept_codes, tuple):
            object.__setattr__(self, "concept_codes", tuple(self.concept_codes))
        gran = str(self.granularity).upper()
        if gran not in ("MICRO", "MACRO"):
            gran = "MICRO"
        object.__setattr__(self, "granularity", gran)

    @property
    def span_start(self) -> int:
        return self.char_span[0]

    @property
    def span_end(self) -> int:
        return self.char_span[1]

    def __len__(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes passage record to dictionary."""
        return {
            "passage_id": self.passage_id,
            "doc_id": self.doc_id,
            "char_span": list(self.char_span),
            "text": self.text,
            "created_at": self.created_at,
            "granularity": self.granularity,
            "parent_macro_id": self.parent_macro_id,
            "concept_codes": list(self.concept_codes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PassageRecord:
        """Deserializes passage record from dictionary."""
        raw_span = data.get("char_span")
        if raw_span is None:
            raw_text = data.get("text", "")
            char_span = (0, len(raw_text))
        else:
            char_span = tuple(raw_span)

        raw_concepts = data.get("concept_codes", ())
        if isinstance(raw_concepts, str):
            try:
                raw_concepts = json.loads(raw_concepts)
            except Exception:
                raw_concepts = ()

        return cls(
            passage_id=str(data["passage_id"]),
            doc_id=str(data.get("doc_id", "default_doc")),
            char_span=char_span,
            text=str(data["text"]),
            created_at=float(data.get("created_at", time.time())),
            granularity=str(data.get("granularity", "MICRO")),
            parent_macro_id=data.get("parent_macro_id"),
            concept_codes=tuple(int(c) for c in raw_concepts) if raw_concepts else (),
        )


class PassageStore:
    """Thread-safe dual-node passage storage engine with in-memory cache and SQLite backing.
    
    Provides microsecond span lookups, batch insertion, offset-based spatial indexing,
    mutable passage ingestion states, and multi-scale hierarchy support.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initializes PassageStore.
        
        Args:
            db_path: Optional path to SQLite database. If ':memory:' or None, runs in-memory.
                     If a file path is provided, creates schema and persists to disk.
        """
        self._lock = threading.RLock()
        self._passages: Dict[str, PassageRecord] = {}
        # doc_id -> list of (span_start, span_end, passage_id) kept sorted by span_start
        self._doc_offset_index: Dict[str, List[Tuple[int, int, str]]] = {}
        # parent_macro_id -> list of micro passage_ids
        self._macro_to_micros: Dict[str, List[str]] = {}
        # passage_id -> (status, updated_at)
        self._passage_states: Dict[str, Tuple[str, float]] = {}

        self.db_path = str(db_path) if db_path is not None else None
        self._conn: Optional[sqlite3.Connection] = None

        if self.db_path is not None:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._init_sqlite_schema()
            self._load_from_sqlite()

    def _init_sqlite_schema(self) -> None:
        """Initializes SQLite tables and indices with migration support."""
        if self._conn is None:
            return
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS passages (
                    passage_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    span_start INTEGER NOT NULL,
                    span_end INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    granularity TEXT NOT NULL DEFAULT 'MICRO',
                    parent_macro_id TEXT,
                    concept_codes TEXT DEFAULT '[]'
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS passage_ingestion_state (
                    passage_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            # Migrate any missing columns before creating indices on them
            self._migrate_sqlite_schema()
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_passages_doc ON passages(doc_id, span_start)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_passages_parent ON passages(parent_macro_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_status ON passage_ingestion_state(status)"
            )

    def _migrate_sqlite_schema(self) -> None:
        """Migrates legacy SQLite databases by adding any missing columns."""
        if self._conn is None:
            return
        cur = self._conn.cursor()
        cur.execute("PRAGMA table_info(passages)")
        existing_cols = {row[1] for row in cur.fetchall()}

        if "granularity" not in existing_cols:
            self._conn.execute("ALTER TABLE passages ADD COLUMN granularity TEXT NOT NULL DEFAULT 'MICRO'")
        if "parent_macro_id" not in existing_cols:
            self._conn.execute("ALTER TABLE passages ADD COLUMN parent_macro_id TEXT")
        if "concept_codes" not in existing_cols:
            self._conn.execute("ALTER TABLE passages ADD COLUMN concept_codes TEXT DEFAULT '[]'")

    def _load_from_sqlite(self) -> None:
        """Loads existing passages and ingestion states from SQLite database into memory cache."""
        if self._conn is None:
            return
        with self._lock, self._conn:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT passage_id, doc_id, span_start, span_end, text, created_at, granularity, parent_macro_id, concept_codes FROM passages"
            )
            for row in cur.fetchall():
                pid, doc_id, s_start, s_end, text, created_at, gran, parent_m, concepts_raw = row
                try:
                    c_codes = tuple(json.loads(concepts_raw)) if concepts_raw else ()
                except Exception:
                    c_codes = ()
                rec = PassageRecord(
                    passage_id=pid,
                    doc_id=doc_id,
                    char_span=(int(s_start), int(s_end)),
                    text=text,
                    created_at=float(created_at),
                    granularity=str(gran or "MICRO"),
                    parent_macro_id=parent_m,
                    concept_codes=c_codes,
                )
                self._passages[pid] = rec
                self._index_passage(rec)

            cur.execute("SELECT passage_id, status, updated_at FROM passage_ingestion_state")
            for row in cur.fetchall():
                pid, status, updated_at = row
                self._passage_states[pid] = (status, float(updated_at))

    def _index_passage(self, rec: PassageRecord) -> None:
        """Internal helper to maintain sorted offset index and parent macro mapping."""
        doc_list = self._doc_offset_index.setdefault(rec.doc_id, [])
        entry = (rec.char_span[0], rec.char_span[1], rec.passage_id)
        idx_to_remove = None
        for i, item in enumerate(doc_list):
            if item[2] == rec.passage_id:
                idx_to_remove = i
                break
        if idx_to_remove is not None:
            doc_list.pop(idx_to_remove)

        bisect.insort(doc_list, entry)

        if rec.parent_macro_id:
            m_list = self._macro_to_micros.setdefault(rec.parent_macro_id, [])
            if rec.passage_id not in m_list:
                m_list.append(rec.passage_id)

    def add_passage(
        self,
        passage: Union[PassageRecord, Dict[str, Any]],
        persist: bool = True,
        initial_status: Optional[str] = None,
    ) -> PassageRecord:
        """Stores a single PassageRecord.
        
        Args:
            passage: PassageRecord or equivalent dict.
            persist: If True and SQLite is configured, writes to database immediately.
            initial_status: Optional initial ingestion status (COARSE_ONLY, PENDING, FULL).
            
        Returns:
            The stored PassageRecord.
        """
        if isinstance(passage, dict):
            rec = PassageRecord.from_dict(passage)
        else:
            rec = passage

        with self._lock:
            self._passages[rec.passage_id] = rec
            self._index_passage(rec)

            if initial_status is not None:
                self.set_ingestion_state(rec.passage_id, initial_status)

            if persist and self._conn is not None:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO passages (
                            passage_id, doc_id, span_start, span_end, text, created_at,
                            granularity, parent_macro_id, concept_codes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rec.passage_id,
                            rec.doc_id,
                            rec.char_span[0],
                            rec.char_span[1],
                            rec.text,
                            rec.created_at,
                            rec.granularity,
                            rec.parent_macro_id,
                            json.dumps(list(rec.concept_codes)),
                        ),
                    )
        return rec

    def add_passages(
        self,
        passages: Iterable[Union[PassageRecord, Dict[str, Any]]],
        persist: bool = True,
        initial_status: Optional[str] = None,
    ) -> List[PassageRecord]:
        """Batch-stores multiple passage records for high throughput.
        
        Args:
            passages: Sequence of PassageRecord or dicts.
            persist: If True and SQLite is configured, commits in a single transaction.
            initial_status: Optional initial ingestion status for all added passages.
            
        Returns:
            List of stored PassageRecord instances.
        """
        records: List[PassageRecord] = []
        rows: List[Tuple[str, str, int, int, str, float, str, Optional[str], str]] = []

        for p in passages:
            if isinstance(p, dict):
                rec = PassageRecord.from_dict(p)
            else:
                rec = p
            records.append(rec)
            rows.append((
                rec.passage_id,
                rec.doc_id,
                rec.char_span[0],
                rec.char_span[1],
                rec.text,
                rec.created_at,
                rec.granularity,
                rec.parent_macro_id,
                json.dumps(list(rec.concept_codes)),
            ))

        with self._lock:
            for rec in records:
                self._passages[rec.passage_id] = rec
                self._index_passage(rec)
                if initial_status is not None:
                    self.set_ingestion_state(rec.passage_id, initial_status)

            if persist and self._conn is not None and rows:
                with self._conn:
                    self._conn.executemany(
                        """
                        INSERT OR REPLACE INTO passages (
                            passage_id, doc_id, span_start, span_end, text, created_at,
                            granularity, parent_macro_id, concept_codes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
        return records

    def get_passage(self, passage_id: str) -> Optional[PassageRecord]:
        """Retrieves a PassageRecord by passage_id.
        
        Args:
            passage_id: Passage identifier string.
            
        Returns:
            PassageRecord or None if not found.
        """
        with self._lock:
            if passage_id in self._passages:
                return self._passages[passage_id]

            if self._conn is not None:
                cur = self._conn.cursor()
                cur.execute(
                    "SELECT passage_id, doc_id, span_start, span_end, text, created_at, granularity, parent_macro_id, concept_codes FROM passages WHERE passage_id = ?",
                    (passage_id,),
                )
                row = cur.fetchone()
                if row:
                    try:
                        c_codes = tuple(json.loads(row[8])) if row[8] else ()
                    except Exception:
                        c_codes = ()
                    rec = PassageRecord(
                        passage_id=row[0],
                        doc_id=row[1],
                        char_span=(int(row[2]), int(row[3])),
                        text=row[4],
                        created_at=float(row[5]),
                        granularity=str(row[6] or "MICRO"),
                        parent_macro_id=row[7],
                        concept_codes=c_codes,
                    )
                    self._passages[rec.passage_id] = rec
                    self._index_passage(rec)
                    return rec
            return None

    def set_ingestion_state(self, passage_id: str, status: str, updated_at: Optional[float] = None) -> None:
        """Sets the mutable ingestion status for a passage."""
        stat_norm = status.strip().upper()
        if stat_norm not in VALID_INGESTION_STATUSES:
            raise ValueError(f"Invalid ingestion status '{status}'. Must be one of {VALID_INGESTION_STATUSES}")
        ts = float(updated_at if updated_at is not None else time.time())
        with self._lock:
            self._passage_states[passage_id] = (stat_norm, ts)
            if self._conn is not None:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO passage_ingestion_state (passage_id, status, updated_at)
                        VALUES (?, ?, ?)
                        """,
                        (passage_id, stat_norm, ts),
                    )

    def get_ingestion_state(self, passage_id: str) -> Optional[str]:
        """Retrieves current ingestion status for a passage."""
        with self._lock:
            if passage_id in self._passage_states:
                return self._passage_states[passage_id][0]
            if self._conn is not None:
                cur = self._conn.cursor()
                cur.execute(
                    "SELECT status, updated_at FROM passage_ingestion_state WHERE passage_id = ?",
                    (passage_id,),
                )
                row = cur.fetchone()
                if row:
                    self._passage_states[passage_id] = (row[0], float(row[1]))
                    return row[0]
            return None

    def get_passages_by_status(self, status: str) -> List[PassageRecord]:
        """Returns all passages matching the given ingestion status."""
        stat_norm = status.strip().upper()
        with self._lock:
            results: List[PassageRecord] = []
            for pid, (st, _) in list(self._passage_states.items()):
                if st == stat_norm:
                    p = self.get_passage(pid)
                    if p is not None:
                        results.append(p)
            return results

    def get_all_passages(self) -> List[PassageRecord]:
        """Returns all registered passages across all granularities."""
        with self._lock:
            return list(self._passages.values())

    def get_macro_passages(self, doc_id: Optional[str] = None) -> List[PassageRecord]:
        """Returns all macro passages, optionally filtered by doc_id."""
        with self._lock:
            if doc_id is not None:
                passages = self.get_passages_for_doc(doc_id)
            else:
                passages = list(self._passages.values())
            return [p for p in passages if p.granularity == "MACRO"]

    def get_micro_passages(self, parent_macro_id: str) -> List[PassageRecord]:
        """Returns all micro passages linked to a parent macro passage."""
        with self._lock:
            pids = self._macro_to_micros.get(parent_macro_id, [])
            results: List[PassageRecord] = []
            for pid in pids:
                p = self.get_passage(pid)
                if p is not None:
                    results.append(p)
            return results

    def get_text_span(
        self,
        passage_id: str,
        span_start: Optional[int] = None,
        span_end: Optional[int] = None,
    ) -> Optional[str]:
        """Retrieves the verbatim text span for a passage.
        
        Handles both relative offsets within the passage text and absolute document-level offsets.
        
        Args:
            passage_id: Target passage identifier.
            span_start: Starting character offset (relative or absolute).
            span_end: Ending character offset (relative or absolute).
            
        Returns:
            The text span substring, or None if the passage is not found.
        """
        rec = self.get_passage(passage_id)
        if rec is None:
            return None

        text = rec.text
        text_len = len(text)

        if span_start is None and span_end is None:
            return text

        s_start = 0 if span_start is None else span_start
        s_end = text_len if span_end is None else span_end

        # Case 1: Absolute document offsets matching the passage char_span bounds
        doc_start, doc_end = rec.char_span
        if (s_start >= doc_start and s_end <= doc_end) and (doc_start > 0 or s_end > text_len):
            rel_start = max(0, s_start - doc_start)
            rel_end = min(text_len, s_end - doc_start)
            return text[rel_start:rel_end]

        # Case 2: Relative offsets within the passage text
        rel_start = max(0, min(s_start, text_len))
        rel_end = max(rel_start, min(s_end, text_len))
        return text[rel_start:rel_end]

    def get_passages_for_doc(self, doc_id: str) -> List[PassageRecord]:
        """Returns all passages belonging to a given doc_id, ordered by span start."""
        with self._lock:
            entries = self._doc_offset_index.get(doc_id, [])
            results: List[PassageRecord] = []
            for _, _, pid in entries:
                rec = self.get_passage(pid)
                if rec is not None:
                    results.append(rec)
            return results

    def find_passage_at_offset(self, doc_id: str, offset: int) -> Optional[PassageRecord]:
        """Finds the passage in doc_id that contains the given document character offset."""
        with self._lock:
            entries = self._doc_offset_index.get(doc_id, [])
            if not entries:
                return None
            for s_start, s_end, pid in entries:
                if s_start <= offset < s_end:
                    return self.get_passage(pid)
            return None

    def delete_passage(self, passage_id: str) -> bool:
        """Deletes a passage from in-memory cache and SQLite."""
        with self._lock:
            rec = self._passages.pop(passage_id, None)
            if rec is not None:
                doc_list = self._doc_offset_index.get(rec.doc_id, [])
                self._doc_offset_index[rec.doc_id] = [
                    item for item in doc_list if item[2] != passage_id
                ]
            self._passage_states.pop(passage_id, None)
            for m_id, micros in list(self._macro_to_micros.items()):
                if passage_id in micros:
                    micros.remove(passage_id)
            if passage_id in self._macro_to_micros:
                self._macro_to_micros.pop(passage_id, None)

            deleted_db = False
            if self._conn is not None:
                with self._conn:
                    self._conn.execute("DELETE FROM passage_ingestion_state WHERE passage_id = ?", (passage_id,))
                    cur = self._conn.execute("DELETE FROM passages WHERE passage_id = ?", (passage_id,))
                    deleted_db = cur.rowcount > 0

            return rec is not None or deleted_db

    def clear(self) -> None:
        """Clears all passages from memory and SQLite table."""
        with self._lock:
            self._passages.clear()
            self._doc_offset_index.clear()
            self._macro_to_micros.clear()
            self._passage_states.clear()
            if self._conn is not None:
                with self._conn:
                    self._conn.execute("DELETE FROM passage_ingestion_state")
                    self._conn.execute("DELETE FROM passages")

    def count(self) -> int:
        """Returns total number of stored passages."""
        with self._lock:
            return len(self._passages)

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, passage_id: str) -> bool:
        return self.get_passage(passage_id) is not None

    def close(self) -> None:
        """Closes SQLite database connection if open."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
