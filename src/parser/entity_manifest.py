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

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

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
            "cid": self.cid,
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
            self.db_path = str(Path(db_path).resolve())

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
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
                    record.cid,
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
        """Scan stored entities to initialize ID generator beyond highest existing E-id."""
        max_id = 0
        for entity in self.storage.list_all_entities():
            m = re.match(r"^E(\d+)$", entity.canonical_id)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def mint_entity_id(self) -> str:
        """Mint the next deterministic canonical entity identifier (e.g. 'E1', 'E2')."""
        cid = f"E{self._next_id_num}"
        self._next_id_num += 1
        return cid

    def _allocate_register(self, record: EntityRecord):
        """Assign an available Band 2 register slot to an active entity."""
        if record.register_binding and record.register_binding in self.available_registers:
            self._register_bindings[record.register_binding] = record.canonical_id
            return

        # Look for first unoccupied register
        for reg in self.available_registers:
            if reg not in self._register_bindings:
                self._register_bindings[reg] = record.canonical_id
                record.register_binding = reg
                return

        # If all 8 registers are occupied, assign only if this entity has higher salience
        # than the lowest-salience currently bound entity
        lowest_reg = None
        lowest_salience = float("inf")
        for reg, bound_id in self._register_bindings.items():
            bound_rec = self._active.get(bound_id)
            if bound_rec and bound_rec.salience_score < lowest_salience:
                lowest_salience = bound_rec.salience_score
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

        # Sort active candidates by salience then last_seen_chunk ascending
        candidates = sorted(
            self._active.values(),
            key=lambda r: (r.salience_score, r.last_seen_chunk),
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

    def __len__(self) -> int:
        return len(self._active)


class EntityMatcher:
    """Sub-millisecond regex / Trie matcher for pre-scanning chunk surface text.
    
    Compiles known aliases into an optimized word-boundary pattern,
    ignoring common pronouns to avoid false-positive dormant resurrection.
    """

    def __init__(self, ignored_tokens: Optional[Set[str]] = None):
        self.ignored_tokens = set(ignored_tokens or PRE_SCAN_IGNORED_TOKENS)
        self._alias_to_ids: Dict[str, Set[str]] = {}
        self._compiled_regex: Optional[re.Pattern] = None
        self._dirty: bool = True

    def build_index(self, alias_to_ids_map: Dict[str, List[str]]):
        """Rebuild matcher index from an alias-to-canonical-IDs map."""
        self._alias_to_ids.clear()
        for alias, ids in alias_to_ids_map.items():
            clean = alias.strip()
            if not clean:
                continue
            lower = clean.lower()
            if lower in self.ignored_tokens or len(lower) < 2:
                continue
            self._alias_to_ids.setdefault(lower, set()).update(ids)

        self._dirty = True

    def _compile_pattern(self):
        """Compile regex with word boundaries, ordering longer aliases first."""
        if not self._alias_to_ids:
            self._compiled_regex = None
            self._dirty = False
            return

        # Sort aliases by length descending so longer compound names match before substrings
        sorted_aliases = sorted(self._alias_to_ids.keys(), key=len, reverse=True)
        escaped = [re.escape(a) for a in sorted_aliases]
        pattern_str = r"\b(?:" + "|".join(escaped) + r")\b"
        self._compiled_regex = re.compile(pattern_str, re.IGNORECASE)
        self._dirty = False

    def match_chunk(self, chunk_text: str) -> Dict[str, Set[str]]:
        """Scan chunk text for registered entity aliases.
        
        Returns:
            Dictionary mapping canonical_id -> set of matched surface alias strings.
        """
        if self._dirty:
            self._compile_pattern()

        if not self._compiled_regex or not chunk_text:
            return {}

        matches: Dict[str, Set[str]] = {}
        for m in self._compiled_regex.finditer(chunk_text):
            matched_str = m.group(0)
            canonical_ids = self._alias_to_ids.get(matched_str.lower(), set())
            for cid in canonical_ids:
                matches.setdefault(cid, set()).add(matched_str)

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
        return paged_records, elapsed_ms

    def update_from_extraction(
        self,
        extracted_entities: List[Dict[str, Any]],
        chunk_idx: int = 0,
    ) -> List[EntityRecord]:
        """Sync entity extractions from Neural Discourse Transducer output (Phase 3).
        
        Merges new aliases into existing entities and mints new records for new entities.
        """
        results: List[EntityRecord] = []
        aliases_modified = False

        for ent in extracted_entities:
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

    def format_prompt_block(self, max_entities: Optional[int] = None) -> str:
        """Render active manifest into a compact ~100-token prompt block for the Transducer.
        
        Format:
        ACTIVE ENTITIES:
        - E1: Dr. Eleanor Vance (aliases: Eleanor, Vance)
        - E2: synthetic compound (aliases: polymer, specimen)
        """
        active = self.manifest.all_active(sort_by_salience=True)
        if max_entities is not None:
            active = active[:max_entities]

        if not active:
            return "ACTIVE ENTITIES:\n(None)"

        lines = ["ACTIVE ENTITIES:"]
        for ent in active:
            aliases = ent.get_display_aliases()
            if aliases:
                aliases_str = f" (aliases: {', '.join(aliases)})"
            else:
                aliases_str = ""
            lines.append(f"- {ent.canonical_id}: {ent.canonical_name}{aliases_str}")

        return "\n".join(lines)

    def close(self):
        """Clean up SQLite connection."""
        self.storage.close()

    def __enter__(self) -> EntityPagingEngine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
