"""Working Memory Entity Manifest & Paging Engine for QUANTA.

Maintains global coreference across long-form documents (e.g., 100 chapters)
without token-window scaling penalties:
1. Active Entity Manifest: Keeps 5-15 highly salient entities in working memory.
2. Band 2 Register Binding: Maps top active entities to formal logic registers (VAR_SLOT_X0..X7).
3. Host-RAM SQLite Page Table: Persists dormant entities upon LRU eviction.
4. Fast Pre-Scan Surface Matcher: Scans chunk text in < 1 ms to resurrect dormant entities.
5. Neural Transducer Prompt Formatter: Renders compact ~100-token active entity blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

try:
    import blake3
    def _blake3_hash(data: bytes) -> str:
        return blake3.blake3(data).hexdigest()
except ImportError:
    import hashlib
    def _blake3_hash(data: bytes) -> str:
        return hashlib.blake3(data).hexdigest() if hasattr(hashlib, "blake3") else hashlib.sha256(data).hexdigest()

# Band 2 Formal Variable Registers in QUANTA (slots 280-287)
BAND_2_REGISTERS: List[str] = [f"VAR_SLOT_X{i}" for i in range(8)]

# Generic pronouns and determiners excluded from pre-scan surface resurrection triggers
PRE_SCAN_IGNORED_TOKENS: Set[str] = {
    "i", "me", "my", "myself", "we", "us", "our", "ours",
    "you", "your", "yours",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those",
    "a", "an", "the", "one", "someone", "something",
}


@dataclass
class EntityRecord:
    """Canonical tracking record for an entity in QUANTA working memory.
    
    Attributes:
        canonical_id: Unique identifier minted by the manifest (e.g. 'E1', 'E2').
        canonical_name: Primary human-readable descriptor (e.g. 'Dr. Eleanor Vance').
        category: High-level ontological category (e.g. 'PERSON', 'OBJECT', 'LOCATION').
        surface_aliases: List of alternative surface names and mention variations.
        register_binding: Optional Band 2 formal variable register ('VAR_SLOT_X0'..'X7').
        last_seen_chunk: Index of the most recent chunk containing this entity.
        salience_score: Dynamic cognitive salience, decayed across chunks and boosted by mentions.
        mention_count: Total cumulative mentions throughout the narrative.
        cid: Optional 256-bit BLAKE3 Merkle CID of the compiled entity node.
        properties: Open-ended attribute map for metadata (e.g. gender, role, status).
    """
    canonical_id: str
    canonical_name: str
    category: str = "OBJECT"
    surface_aliases: List[str] = field(default_factory=list)
    register_binding: Optional[str] = None
    last_seen_chunk: int = 0
    salience_score: float = 1.0
    mention_count: int = 1
    cid: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def add_alias(self, alias: str) -> bool:
        """Add a unique surface alias if not already present.
        
        Returns:
            True if alias was newly added, False if already present.
        """
        clean_alias = alias.strip()
        if not clean_alias:
            return False
        
        # Check case-insensitively against existing aliases and canonical name
        existing_lower = {a.lower() for a in self.surface_aliases}
        existing_lower.add(self.canonical_name.lower())
        
        if clean_alias.lower() in existing_lower:
            return False

        self.surface_aliases.append(clean_alias)
        return True

    def add_aliases(self, aliases: Iterable[str]) -> int:
        """Add multiple surface aliases, returning count of new aliases added."""
        added = 0
        for alias in aliases:
            if self.add_alias(alias):
                added += 1
        return added

    def update_salience(
        self,
        chunk_idx: int,
        boost: float = 1.0,
        decay_rate: float = 0.9,
    ) -> float:
        """Update salience score based on chunk recency and mention frequency.
        
        Applies exponential decay over the chunk distance since last seen,
        then adds the mention boost.
        """
        gap = max(0, chunk_idx - self.last_seen_chunk)
        decayed = self.salience_score * (decay_rate ** gap)
        self.salience_score = round(decayed + boost, 4)
        self.last_seen_chunk = chunk_idx
        self.mention_count += 1
        return self.salience_score

    def get_display_aliases(self) -> List[str]:
        """Return aliases distinct from the canonical name for concise rendering."""
        c_lower = self.canonical_name.strip().lower()
        distinct = []
        for a in self.surface_aliases:
            stripped = a.strip()
            if stripped.lower() != c_lower and stripped not in distinct:
                distinct.append(stripped)
        return distinct

    def compute_cid(self) -> str:
        """Compute deterministic 256-bit BLAKE3 Merkle CID for this entity."""
        if self.cid:
            return self.cid
        payload = f"entity:{self.canonical_name.strip().lower()}:{self.category.strip().upper()}".encode("utf-8")
        self.cid = _blake3_hash(payload)
        return self.cid

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity record to dictionary."""
        return {
            "canonical_id": self.canonical_id,
            "canonical_name": self.canonical_name,
            "category": self.category,
            "surface_aliases": list(self.surface_aliases),
            "register_binding": self.register_binding,
            "last_seen_chunk": self.last_seen_chunk,
            "salience_score": self.salience_score,
            "mention_count": self.mention_count,
            "cid": self.cid or self.compute_cid(),
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityRecord:
        """Construct EntityRecord from dictionary."""
        return cls(
            canonical_id=data["canonical_id"],
            canonical_name=data["canonical_name"],
            category=data.get("category", "OBJECT"),
            surface_aliases=list(data.get("surface_aliases", [])),
            register_binding=data.get("register_binding"),
            last_seen_chunk=data.get("last_seen_chunk", 0),
            salience_score=float(data.get("salience_score", 1.0)),
            mention_count=int(data.get("mention_count", 1)),
            cid=data.get("cid"),
            properties=dict(data.get("properties", {})),
        )


class EntityStorage:
    """Host-RAM / Disk SQLite Page Table for persistent entity storage.
    
    Provides sub-2ms lookups for dormant entities and indexed alias resolution.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initialize SQLite storage backend.
        
        Args:
            db_path: Path to SQLite database file or ':memory:' for transient RAM storage.
        """
        if db_path is None or db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            resolved = Path(db_path).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(resolved)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._create_schema()

    def _create_schema(self):
        """Set up entities and alias lookup tables."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    canonical_id TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    surface_aliases TEXT NOT NULL,
                    register_binding TEXT,
                    last_seen_chunk INTEGER NOT NULL,
                    salience_score REAL NOT NULL,
                    mention_count INTEGER NOT NULL,
                    cid TEXT,
                    properties TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    alias TEXT NOT NULL COLLATE NOCASE,
                    canonical_id TEXT NOT NULL,
                    PRIMARY KEY (alias, canonical_id),
                    FOREIGN KEY (canonical_id) REFERENCES entities(canonical_id) ON DELETE CASCADE
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias ON entity_aliases(alias COLLATE NOCASE);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_last_seen ON entities(last_seen_chunk);"
            )

    def save_entity(self, record: EntityRecord):
        """Insert or update an entity record and re-index all its aliases."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO entities (
                    canonical_id, canonical_name, category, surface_aliases,
                    register_binding, last_seen_chunk, salience_score,
                    mention_count, cid, properties
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.canonical_id,
                    record.canonical_name,
                    record.category,
                    json.dumps(record.surface_aliases),
                    record.register_binding,
                    record.last_seen_chunk,
                    record.salience_score,
                    record.mention_count,
                    record.cid or record.compute_cid(),
                    json.dumps(record.properties),
                ),
            )
            # Re-index aliases
            self._conn.execute(
                "DELETE FROM entity_aliases WHERE canonical_id = ?",
                (record.canonical_id,),
            )
            all_aliases = set(record.surface_aliases)
            all_aliases.add(record.canonical_name)
            for alias in all_aliases:
                clean = alias.strip()
                if clean:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO entity_aliases (alias, canonical_id)
                        VALUES (?, ?)
                        """,
                        (clean, record.canonical_id),
                    )

    def save_entities(self, records: Iterable[EntityRecord]):
        """Batch insert or update multiple entity records within a single atomic transaction."""
        with self._conn:
            for record in records:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO entities (
                        canonical_id, canonical_name, category, surface_aliases,
                        register_binding, last_seen_chunk, salience_score,
                        mention_count, cid, properties
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.canonical_id,
                        record.canonical_name,
                        record.category,
                        json.dumps(record.surface_aliases),
                        record.register_binding,
                        record.last_seen_chunk,
                        record.salience_score,
                        record.mention_count,
                        record.cid or record.compute_cid(),
                        json.dumps(record.properties),
                    ),
                )
                self._conn.execute(
                    "DELETE FROM entity_aliases WHERE canonical_id = ?",
                    (record.canonical_id,),
                )
                all_aliases = set(record.surface_aliases)
                all_aliases.add(record.canonical_name)
                for alias in all_aliases:
                    clean = alias.strip()
                    if clean:
                        self._conn.execute(
                            """
                            INSERT OR IGNORE INTO entity_aliases (alias, canonical_id)
                            VALUES (?, ?)
                            """,
                            (clean, record.canonical_id),
                        )

    def get_entity(self, canonical_id: str) -> Optional[EntityRecord]:
        """Retrieve entity by canonical ID."""
        cursor = self._conn.execute(
            "SELECT * FROM entities WHERE canonical_id = ?",
            (canonical_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def find_by_alias(self, alias: str) -> List[EntityRecord]:
        """Look up entities matching a given surface alias (case-insensitive)."""
        clean = alias.strip()
        if not clean:
            return []
        cursor = self._conn.execute(
            """
            SELECT e.* FROM entities e
            JOIN entity_aliases a ON e.canonical_id = a.canonical_id
            WHERE a.alias = ? COLLATE NOCASE
            """,
            (clean,),
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_all_aliases(self) -> Dict[str, List[str]]:
        """Return map from surface alias to list of canonical IDs."""
        cursor = self._conn.execute(
            "SELECT alias, canonical_id FROM entity_aliases"
        )
        mapping: Dict[str, List[str]] = {}
        for row in cursor.fetchall():
            alias = row["alias"]
            cid = row["canonical_id"]
            mapping.setdefault(alias, []).append(cid)
        return mapping

    def list_all_entities(self) -> List[EntityRecord]:
        """Return all persisted entities ordered by last_seen_chunk descending."""
        cursor = self._conn.execute(
            "SELECT * FROM entities ORDER BY last_seen_chunk DESC, salience_score DESC"
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def count(self) -> int:
        """Count total entities in storage."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM entities")
        return cursor.fetchone()[0]

    def delete_entity(self, canonical_id: str) -> bool:
        """Delete entity and its aliases from storage."""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM entities WHERE canonical_id = ?",
                (canonical_id,),
            )
            self._conn.execute(
                "DELETE FROM entity_aliases WHERE canonical_id = ?",
                (canonical_id,),
            )
            return cur.rowcount > 0

    def clear(self):
        """Clear all entities and aliases from storage."""
        with self._conn:
            self._conn.execute("DELETE FROM entity_aliases")
            self._conn.execute("DELETE FROM entities")

    def close(self):
        """Close SQLite database connection."""
        self._conn.close()

    def __enter__(self) -> EntityStorage:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> EntityRecord:
        return EntityRecord(
            canonical_id=row["canonical_id"],
            canonical_name=row["canonical_name"],
            category=row["category"],
            surface_aliases=json.loads(row["surface_aliases"]),
            register_binding=row["register_binding"],
            last_seen_chunk=row["last_seen_chunk"],
            salience_score=row["salience_score"],
            mention_count=row["mention_count"],
            cid=row["cid"],
            properties=json.loads(row["properties"]),
        )


class ActiveEntityManifest:
    """In-memory working set manager with LRU eviction and Band 2 register allocation.
    
    Maintains a bounded window (target size: 5-15 entities) of the most active
    and salient discourse entities, evicting dormant entities to EntityStorage.
    """

    def __init__(
        self,
        storage: EntityStorage,
        target_size: int = 10,
        min_size: int = 5,
        max_size: int = 15,
        available_registers: Optional[List[str]] = None,
    ):
        """Initialize the active entity manifest.
        
        Args:
            storage: Persistent storage backend.
            target_size: Preferred number of active working entities.
            min_size: Minimum threshold before LRU eviction can occur.
            max_size: Hard upper bound on in-memory active entities.
            available_registers: Pool of Band 2 registers (defaults to VAR_SLOT_X0..X7).
        """
        if min_size > max_size:
            raise ValueError(f"min_size ({min_size}) cannot exceed max_size ({max_size})")

        self.storage = storage
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
        self.available_registers = list(available_registers or BAND_2_REGISTERS)

        # In-memory working set: canonical_id -> EntityRecord
        self._active: Dict[str, EntityRecord] = {}

        # Register bindings: register_name -> canonical_id
        self._register_bindings: Dict[str, str] = {}

        # Counter for deterministic ID minting
        self._next_id_num = self._compute_next_id_num()

    def _compute_next_id_num(self) -> int:
        """Scan stored and active entities to initialize ID generator beyond highest existing E-id."""
        max_id = 0
        for entity in self.storage.list_all_entities():
            m = re.match(r"^E(\d+)$", entity.canonical_id)
            if m:
                max_id = max(max_id, int(m.group(1)))
        for cid in self._active.keys():
            m = re.match(r"^E(\d+)$", cid)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def mint_entity_id(self) -> str:
        """Mint the next deterministic canonical entity identifier (e.g. 'E1', 'E2')."""
        cid = f"E{self._next_id_num}"
        self._next_id_num += 1
        return cid

    def reset(self):
        """Reset active in-memory manifest state and register bindings."""
        self._active.clear()
        self._register_bindings.clear()
        self._next_id_num = self._compute_next_id_num()

    def _allocate_register(self, record: EntityRecord):
        """Assign an available Band 2 register slot to an active entity."""
        if record.register_binding and record.register_binding in self.available_registers:
            self._register_bindings[record.register_binding] = record.canonical_id
            return

        # Look for first unoccupied register or stale binding
        for reg in self.available_registers:
            bound_id = self._register_bindings.get(reg)
            if bound_id is None or bound_id not in self._active:
                self._register_bindings[reg] = record.canonical_id
                record.register_binding = reg
                return

        # If all registers are actively occupied, assign only if this entity has higher salience
        # than the lowest-salience currently bound entity
        lowest_reg = None
        lowest_salience = float("inf")
        for reg, bound_id in self._register_bindings.items():
            bound_rec = self._active.get(bound_id)
            score = bound_rec.salience_score if bound_rec else -1.0
            if score < lowest_salience:
                lowest_salience = score
                lowest_reg = reg

        if lowest_reg and record.salience_score > lowest_salience:
            prev_id = self._register_bindings[lowest_reg]
            if prev_id in self._active:
                self._active[prev_id].register_binding = None
            self._register_bindings[lowest_reg] = record.canonical_id
            record.register_binding = lowest_reg
        else:
            record.register_binding = None

    def _release_register(self, canonical_id: str):
        """Release any register held by an entity."""
        for reg, bound_id in list(self._register_bindings.items()):
            if bound_id == canonical_id:
                del self._register_bindings[reg]
                break

    def add_or_update(
        self,
        record: EntityRecord,
        chunk_idx: int = 0,
        boost_salience: bool = True,
    ) -> EntityRecord:
        """Add a new entity or update an existing entity in the active working set.
        
        Evicts the least-recently-used entity to storage if max_size is exceeded.
        """
        cid = record.canonical_id

        # Collision prevention: advance next_id_num if external E-id was supplied
        m = re.match(r"^E(\d+)$", cid)
        if m:
            num = int(m.group(1))
            if num >= self._next_id_num:
                self._next_id_num = num + 1

        if cid in self._active:
            existing = self._active[cid]
            if boost_salience:
                existing.update_salience(chunk_idx)
            existing.add_aliases(record.surface_aliases)
            existing.properties.update(record.properties)
            self._allocate_register(existing)
            self.storage.save_entity(existing)
            return existing

        # Check if we need to evict before adding a new entity
        if len(self._active) >= self.max_size:
            self.evict_lru(1)

        if boost_salience:
            record.update_salience(chunk_idx)

        self._active[cid] = record
        self._allocate_register(record)
        self.storage.save_entity(record)
        return record

    def evict(self, canonical_id: str) -> Optional[EntityRecord]:
        """Evict a specific entity from active working set to persistent storage."""
        if canonical_id not in self._active:
            return None

        record = self._active.pop(canonical_id)
        self._release_register(canonical_id)
        record.register_binding = None
        self.storage.save_entity(record)
        return record

    def evict_lru(self, count: int = 1) -> List[EntityRecord]:
        """Evict the `count` least recently used / lowest salience entities to storage.
        
        Sorts by (salience_score, last_seen_chunk) ascending.
        """
        if not self._active or count <= 0:
            return []

        # Sort active candidates by last_seen_chunk ascending, then salience_score ascending
        candidates = sorted(
            self._active.values(),
            key=lambda r: (r.last_seen_chunk, r.salience_score),
        )

        evicted = []
        for rec in candidates[:count]:
            ev = self.evict(rec.canonical_id)
            if ev:
                evicted.append(ev)

        return evicted

    def page_in(self, canonical_id: str, chunk_idx: int = 0) -> Optional[EntityRecord]:
        """Page an entity from persistent storage back into the active manifest.
        
        If active manifest is at max_size, evicts LRU entity first.
        """
        if canonical_id in self._active:
            rec = self._active[canonical_id]
            rec.update_salience(chunk_idx)
            self._allocate_register(rec)
            self.storage.save_entity(rec)
            return rec

        record = self.storage.get_entity(canonical_id)
        if not record:
            return None

        if len(self._active) >= self.max_size:
            self.evict_lru(1)

        record.update_salience(chunk_idx)
        self._active[canonical_id] = record
        self._allocate_register(record)
        self.storage.save_entity(record)
        return record

    def is_active(self, canonical_id: str) -> bool:
        """Check if an entity is currently in active working memory."""
        return canonical_id in self._active

    def get(self, canonical_id: str) -> Optional[EntityRecord]:
        """Get an entity from active working set or return None."""
        return self._active.get(canonical_id)

    def all_active(self, sort_by_salience: bool = True) -> List[EntityRecord]:
        """Return all active entities, optionally sorted by salience descending."""
        records = list(self._active.values())
        if sort_by_salience:
            records.sort(key=lambda r: (r.salience_score, r.last_seen_chunk), reverse=True)
        return records

    def values(self):
        """Return collection of active EntityRecords."""
        return self._active.values()

    def keys(self):
        """Return collection of active canonical IDs."""
        return self._active.keys()

    def items(self):
        """Return collection of (canonical_id, EntityRecord) pairs."""
        return self._active.items()

    def __iter__(self):
        return iter(self._active)

    def __len__(self) -> int:
        return len(self._active)

    def format_prompt_block(
        self,
        max_entities: Optional[int] = None,
        include_registers: bool = False,
        include_categories: bool = False,
    ) -> str:
        """Render active manifest into a compact ~100-token prompt block for the Transducer.
        
        Format:
        ACTIVE ENTITIES:
        - E1: Dr. Eleanor Vance (aliases: Eleanor, Vance)
        - E2: synthetic compound (aliases: polymer, specimen)
        """
        active = self.all_active(sort_by_salience=True)
        if max_entities is not None:
            active = active[:max_entities]

        if not active:
            return "ACTIVE ENTITIES:\n(None)"

        lines = ["ACTIVE ENTITIES:"]
        for ent in active:
            extra_tags = []
            if include_registers and ent.register_binding:
                extra_tags.append(f"[{ent.register_binding}]")
            if include_categories and ent.category:
                extra_tags.append(f"[{ent.category}]")
            tags_str = f" {' '.join(extra_tags)}" if extra_tags else ""

            aliases = ent.get_display_aliases()
            aliases_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
            lines.append(f"- {ent.canonical_id}:{tags_str} {ent.canonical_name}{aliases_str}")

        return "\n".join(lines)



def _normalize_diacritics(s: str) -> str:
    """Decompose and strip combining diacritical marks across all Unicode scripts."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


class EntityMatcher:
    """Sub-millisecond regex / Trie matcher for pre-scanning chunk surface text.
    
    Compiles known aliases into an optimized word-boundary pattern,
    ignoring common pronouns to avoid false-positive dormant resurrection.
    Supports language-agnostic prefix, Levenshtein, and character n-gram
    matching to handle agglutinative and inflected forms without language-specific rules.
    """

    def __init__(
        self,
        ignored_tokens: Optional[Set[str]] = None,
        enable_fuzzy: bool = True,
        fuzzy_threshold: float = 0.70,
        min_prefix_len: int = 3,
    ):
        self.ignored_tokens = set(ignored_tokens or PRE_SCAN_IGNORED_TOKENS)
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.min_prefix_len = min_prefix_len
        self._alias_to_ids: Dict[str, Set[str]] = {}
        self._prefix_to_aliases: Dict[str, List[Tuple[str, str, Set[str]]]] = {}
        self._compiled_regex: Optional[re.Pattern] = None
        self._dirty: bool = True

    def build_index(self, alias_to_ids_map: Dict[str, List[str]]):
        """Rebuild matcher index from an alias-to-canonical-IDs map."""
        self._alias_to_ids.clear()
        self._prefix_to_aliases.clear()
        for alias, ids in alias_to_ids_map.items():
            clean = alias.strip()
            if not clean:
                continue
            lower = clean.lower()
            if lower in self.ignored_tokens or len(lower) < 2:
                continue
            self._alias_to_ids.setdefault(lower, set()).update(ids)

            norm = _normalize_diacritics(clean)
            if len(norm) >= self.min_prefix_len:
                pfx = norm[:self.min_prefix_len]
                self._prefix_to_aliases.setdefault(pfx, []).append((norm, lower, self._alias_to_ids[lower]))

        self._dirty = True

    def _compile_pattern(self):
        """Compile regex with lookaround word boundaries, ordering longer aliases first."""
        if not self._alias_to_ids:
            self._compiled_regex = None
            self._dirty = False
            return

        # Sort aliases by length descending so longer compound names match before substrings
        sorted_aliases = sorted(self._alias_to_ids.keys(), key=len, reverse=True)
        escaped = [re.escape(a) for a in sorted_aliases]
        pattern_str = r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)"
        self._compiled_regex = re.compile(pattern_str, re.IGNORECASE)
        self._dirty = False

    @staticmethod
    def _ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
        """Compute character n-gram Dice similarity coefficient."""
        if s1 == s2:
            return 1.0
        if len(s1) < n or len(s2) < n:
            return 1.0 if s1 == s2 else 0.0
        ng1 = [s1[i : i + n] for i in range(len(s1) - n + 1)]
        ng2 = [s2[i : i + n] for i in range(len(s2) - n + 1)]
        from collections import Counter
        c1 = Counter(ng1)
        c2 = Counter(ng2)
        common = sum((c1 & c2).values())
        return (2.0 * common) / (len(ng1) + len(ng2))

    @staticmethod
    def _levenshtein_similarity(s1: str, s2: str) -> float:
        """Compute normalized Levenshtein edit distance similarity in [0.0, 1.0]."""
        if s1 == s2:
            return 1.0
        m, n = len(s1), len(s2)
        if m == 0 or n == 0:
            return 0.0
        if m < n:
            s1, s2 = s2, s1
            m, n = n, m
        previous_row = list(range(n + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        dist = previous_row[-1]
        return max(0.0, 1.0 - (dist / max(m, n)))

    def match_chunk(self, chunk_text: str, fuzzy: Optional[bool] = None) -> Dict[str, Set[str]]:
        """Scan chunk text for registered entity aliases.
        
        Performs sub-millisecond exact regex boundary matching first.
        If fuzzy is True (or default self.enable_fuzzy), performs language-agnostic
        prefix and character n-gram / Levenshtein matching to catch inflected,
        agglutinated, or compound surface forms without language-specific suffix tables.
        
        Returns:
            Dictionary mapping canonical_id -> set of matched surface alias strings.
        """
        if self._dirty:
            self._compile_pattern()

        if not chunk_text:
            return {}

        matches: Dict[str, Set[str]] = {}
        exact_matched_tokens: Set[str] = set()

        # 1. Exact regex boundary match (< 0.1 ms)
        if self._compiled_regex:
            for m in self._compiled_regex.finditer(chunk_text):
                matched_str = m.group(0)
                exact_matched_tokens.add(matched_str.lower())
                canonical_ids = self._alias_to_ids.get(matched_str.lower(), set())
                for cid in canonical_ids:
                    matches.setdefault(cid, set()).add(matched_str)

        # 2. Language-agnostic fuzzy/prefix matching for agglutinative/inflected forms
        do_fuzzy = self.enable_fuzzy if fuzzy is None else fuzzy
        if do_fuzzy and self._prefix_to_aliases:
            tokens = re.findall(r"[\w\u00C0-\u024F\u1E00-\u1EFF]+", chunk_text)
            for tok in tokens:
                tok_clean = tok.strip()
                tok_low = tok_clean.lower()
                if tok_low in exact_matched_tokens or tok_low in self.ignored_tokens or len(tok_low) < self.min_prefix_len:
                    continue

                norm_tok = _normalize_diacritics(tok_clean)
                if len(norm_tok) < self.min_prefix_len:
                    continue

                pfx = norm_tok[:self.min_prefix_len]
                candidates = self._prefix_to_aliases.get(pfx)
                if not candidates:
                    continue

                for norm_alias, alias_low, canonical_ids in candidates:
                    # A. Prefix match (agglutinative suffixes: e.g. kutya -> kutyát, kutyának, kutyával)
                    if norm_tok.startswith(norm_alias):
                        diff_len = len(norm_tok) - len(norm_alias)
                        if diff_len <= 6:
                            for cid in canonical_ids:
                                matches.setdefault(cid, set()).add(tok_clean)
                            continue

                    # B. Levenshtein edit distance & character n-gram overlap
                    if abs(len(norm_tok) - len(norm_alias)) <= 4:
                        lev_sim = self._levenshtein_similarity(norm_tok, norm_alias)
                        if lev_sim >= self.fuzzy_threshold:
                            for cid in canonical_ids:
                                matches.setdefault(cid, set()).add(tok_clean)
                            continue

                        ngram_sim = self._ngram_similarity(norm_tok, norm_alias, n=2)
                        if ngram_sim >= self.fuzzy_threshold:
                            for cid in canonical_ids:
                                matches.setdefault(cid, set()).add(tok_clean)
                            continue

        return matches


class EntityPagingEngine:
    """Unified Working Memory Entity Manifest and Paging Coordinator.
    
    Coordinates the active working set, SQLite page table, fast pre-scan matcher,
    and prompt block formatting for the Neural Discourse Transducer.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        target_size: int = 10,
        min_size: int = 5,
        max_size: int = 15,
    ):
        """Initialize the complete entity paging engine."""
        self.storage = EntityStorage(db_path=db_path)
        self.manifest = ActiveEntityManifest(
            storage=self.storage,
            target_size=target_size,
            min_size=min_size,
            max_size=max_size,
        )
        self.matcher = EntityMatcher()
        self.last_paged_in: List[EntityRecord] = []
        self._sync_matcher()

    def _sync_matcher(self):
        """Sync matcher with all registered aliases from storage."""
        alias_map = self.storage.get_all_aliases()
        self.matcher.build_index(alias_map)

    def register_new_entity(
        self,
        name: str,
        category: str = "OBJECT",
        aliases: Optional[List[str]] = None,
        chunk_idx: int = 0,
        properties: Optional[Dict[str, Any]] = None,
        cid: Optional[str] = None,
    ) -> EntityRecord:
        """Mint a new canonical entity ID and register it in active working memory."""
        canonical_id = cid or self.manifest.mint_entity_id()
        record = EntityRecord(
            canonical_id=canonical_id,
            canonical_name=name.strip(),
            category=category.strip().upper(),
            surface_aliases=list(aliases or []),
            last_seen_chunk=chunk_idx,
            salience_score=1.0,
            mention_count=1,
            properties=dict(properties or {}),
        )
        active_record = self.manifest.add_or_update(record, chunk_idx=chunk_idx)
        self._sync_matcher()
        return active_record

    def pre_scan_and_page(
        self,
        chunk_text: str,
        chunk_idx: int = 0,
    ) -> Tuple[List[EntityRecord], float]:
        """Pre-scan chunk text in < 1 ms, paging dormant entities from SQLite into active manifest.
        
        Returns:
            Tuple of (list_of_matched_entities, elapsed_milliseconds).
        """
        t0 = time.perf_counter()
        matches = self.matcher.match_chunk(chunk_text)

        paged_records: List[EntityRecord] = []
        for canonical_id in matches.keys():
            if self.manifest.is_active(canonical_id):
                rec = self.manifest.get(canonical_id)
                if rec:
                    rec.update_salience(chunk_idx)
                    self.storage.save_entity(rec)
                    paged_records.append(rec)
            else:
                # Dormant entity in SQLite: page back into active manifest
                rec = self.manifest.page_in(canonical_id, chunk_idx=chunk_idx)
                if rec:
                    paged_records.append(rec)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.last_paged_in = list(paged_records)
        return paged_records, elapsed_ms

    def update_from_extraction(
        self,
        extracted_entities: List[Union[Dict[str, Any], Any]],
        chunk_idx: int = 0,
    ) -> List[EntityRecord]:
        """Sync entity extractions from Neural Discourse Transducer output (Phase 3).
        
        Merges new aliases into existing entities and mints new records for new entities.
        Accepts either raw dictionaries or ExtractedEntity dataclass instances.
        """
        results: List[EntityRecord] = []
        aliases_modified = False

        for raw_ent in extracted_entities:
            if hasattr(raw_ent, "to_dict"):
                ent = raw_ent.to_dict()
            elif hasattr(raw_ent, "__dict__") and not isinstance(raw_ent, dict):
                ent = asdict(raw_ent) if is_dataclass(raw_ent) else raw_ent.__dict__
            elif isinstance(raw_ent, dict):
                ent = raw_ent
            else:
                continue

            cid = ent.get("canonical_id") or ent.get("id")
            name = ent.get("canonical_name") or ent.get("name", "")
            category = ent.get("category", "OBJECT")
            aliases = ent.get("surface_aliases", ent.get("aliases", []))
            props = ent.get("properties", {})

            if cid and (self.manifest.is_active(cid) or self.storage.get_entity(cid)):
                # Existing entity: update record
                rec = self.manifest.get(cid) or self.manifest.page_in(cid, chunk_idx)
                if rec:
                    rec.update_salience(chunk_idx)
                    if rec.add_aliases(aliases):
                        aliases_modified = True
                    rec.properties.update(props)
                    self.storage.save_entity(rec)
                    results.append(rec)
            else:
                # New entity
                new_rec = self.register_new_entity(
                    name=name,
                    category=category,
                    aliases=aliases,
                    chunk_idx=chunk_idx,
                    properties=props,
                    cid=cid if (cid and not self.storage.get_entity(cid)) else None,
                )
                aliases_modified = True
                results.append(new_rec)

        if aliases_modified:
            self._sync_matcher()

        return results

    def format_prompt_block(
        self,
        max_entities: Optional[int] = None,
        include_registers: bool = False,
        include_categories: bool = False,
    ) -> str:
        """Render active manifest into a compact ~100-token prompt block for the Transducer.
        
        Format:
        ACTIVE ENTITIES:
        - E1: Dr. Eleanor Vance (aliases: Eleanor, Vance)
        - E2: synthetic compound (aliases: polymer, specimen)
        """
        return self.manifest.format_prompt_block(
            max_entities=max_entities,
            include_registers=include_registers,
            include_categories=include_categories,
        )

    def estimate_manifest_tokens(self, prompt_str: Optional[str] = None) -> int:
        """Estimate token consumption of the formatted active manifest block (~1.33x words)."""
        prompt = prompt_str or self.format_prompt_block()
        words = len(prompt.split())
        return int(words * 1.33) + 1

    def reset(self):
        """Reset working memory storage, manifest, and matcher for a fresh document."""
        self.storage.clear()
        self.manifest.reset()
        self._sync_matcher()

    def close(self):
        """Clean up SQLite connection."""
        self.storage.close()

    def __enter__(self) -> EntityPagingEngine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
