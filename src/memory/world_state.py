"""Dynamic World-State Tracking & Non-Monotonic Belief Revision for QUANTA (Phase 8 / Section 5).

Provides dynamic world model tracking over temporal validity intervals [t_start, t_end):
1. Fluent State Representation (EntityStateRecord):
   - Tracks entity_cid, property_slot (e.g. VAL_LOCATION_SLOT), value_cid,
     t_start, t_end, epistemic_status, initiating event_cid, and terminating_event_cid.
2. Non-Monotonic Transition Resolver (WorldStateManager):
   - When an event asserts a mutually exclusive property (e.g. moving a specimen),
     closes prior active state (t_end = event.time_start), wires TEMP_ALLEN_FINISHES edge,
     and asserts new state (t_start = event.time_start) without deleting historical records.
3. Temporal Point-in-Time State Queries (get_entity_state_at):
   - Query past states at specific timestamps ("Where was the specimen at 11:00 AM?")
     versus current state ("Where is it now?").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.slots import SLOT_NAME_TO_INDEX
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)


# Standard property slot aliases
PROPERTY_SLOT_ALIASES: Dict[str, str] = {
    "location": "VAL_LOCATION_SLOT",
    "loc": "VAL_LOCATION_SLOT",
    "at": "VAL_LOCATION_SLOT",
    "in": "VAL_LOCATION_SLOT",
    "val_location_slot": "VAL_LOCATION_SLOT",
    "state": "PHYSICAL_STATE",
    "phase": "PHYSICAL_STATE",
    "phase_state": "PHYSICAL_STATE",
    "physical_state": "PHYSICAL_STATE",
    "possession": "POSSESSION",
    "held_by": "POSSESSION",
    "possessed_by": "POSSESSION",
    "owner": "POSSESSION",
}

# Mutually exclusive property slots where a new assertion closes prior open intervals
MUTUALLY_EXCLUSIVE_PROPERTIES: Set[str] = {
    "VAL_LOCATION_SLOT",
    "PHYSICAL_STATE",
    "POSSESSION",
}

# Verbs representing movement, relocation, or transfer of entities
MOTION_VERBS: Set[str] = {
    "move", "moved", "moving",
    "relocate", "relocated", "relocating",
    "transfer", "transferred", "transferring",
    "transport", "transported", "transporting",
    "shift", "shifted", "shifting",
    "carry", "carried", "carrying",
    "bring", "brought", "bringing",
    "enter", "entered", "entering",
    "exit", "exited", "exiting",
    "walk", "walked", "walking",
    "run", "ran", "running",
    "fly", "flew", "flying",
    "travel", "traveled", "travelling",
    "head", "headed", "heading",
    "arrive", "arrived", "arriving",
    "depart", "departed", "departing",
    "isolate", "isolated", "isolating",
    "store", "stored", "storing",
    "place", "placed", "placing",
    "put", "putting",
    "deposit", "deposited", "depositing",
}

# Verbs representing static state assertion
STATE_VERBS: Set[str] = {
    "be", "is", "are", "was", "were", "been", "being",
    "remain", "remained", "remaining",
    "stay", "stayed", "staying",
    "reside", "resided", "residing",
    "contain", "contained", "containing",
    "rest", "rested", "resting",
}


def parse_timestamp(value: Any) -> Optional[float]:
    """Robustly parses a temporal descriptor into a numeric float timestamp (e.g. hours or seconds).

    Supports:
    - Numeric floats and ints (e.g. 10.0, 14, 0).
    - 12-hour format strings (e.g. '10:00 AM', '02:00 PM', '2 PM', '11:30 AM').
    - 24-hour format strings (e.g. '10:00', '14:00', '14:30').
    - Natural language milestones (e.g. 'dawn' -> 6.0, 'noon' -> 12.0, 'midnight' -> 0.0, 'dusk' -> 18.0).
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    raw = value.strip().lower()
    if not raw:
        return None

    # Check natural milestones
    if "dawn" in raw or "morning" in raw and not any(ch.isdigit() for ch in raw):
        return 6.0
    if "noon" in raw or "midday" in raw:
        return 12.0
    if "midnight" in raw:
        return 0.0
    if "dusk" in raw or "evening" in raw and not any(ch.isdigit() for ch in raw):
        return 18.0
    if "night" in raw and not any(ch.isdigit() for ch in raw):
        return 22.0

    # Match 12-hour time: e.g. "10:00 AM", "02:00 PM", "2:30 pm", "10 am", "2pm"
    match_12h = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", raw)
    if match_12h:
        hours = int(match_12h.group(1))
        minutes = int(match_12h.group(2) or 0)
        meridiem = match_12h.group(3)
        if meridiem == "pm" and hours < 12:
            hours += 12
        elif meridiem == "am" and hours == 12:
            hours = 0
        return hours + minutes / 60.0

    # Match 24-hour time: e.g. "14:00", "09:30"
    match_24h = re.search(r"\b(\d{1,2}):(\d{2})\b", raw)
    if match_24h:
        hours = int(match_24h.group(1))
        minutes = int(match_24h.group(2))
        return hours + minutes / 60.0

    # Pure numeric string
    try:
        return float(raw)
    except ValueError:
        pass

    # Extract first isolated number as hours/time
    num_match = re.search(r"\b\d+(?:\.\d+)?\b", raw)
    if num_match:
        try:
            return float(num_match.group(0))
        except ValueError:
            pass

    return None


@dataclass
class EntityStateRecord:
    """Dynamic fluent property state of an entity over a temporal interval [t_start, t_end).

    Attributes:
        entity_cid: Canonical CID or identifier of the subject entity.
        property_slot: Fluent slot name (e.g. "VAL_LOCATION_SLOT", "PHYSICAL_STATE", "POSSESSION").
        value_cid: Canonical CID or identifier of the value entity / concept node.
        t_start: Interval start boundary (float).
        t_end: Interval end boundary (float or None if open-ended/current).
        epistemic_status: Epistemic certainty (e.g. "FACT", "OBSERVATION", "HYPOTHESIS", "BELIEF").
        event_cid: CID of the event that asserted or initiated this state.
        terminating_event_cid: CID of the event that closed/invalidated this state.
        metadata: Extensible metadata dictionary (entity name, value label, raw text, etc.).
    """

    entity_cid: str
    property_slot: str
    value_cid: str
    t_start: float
    t_end: Optional[float] = None
    epistemic_status: str = "FACT"
    event_cid: Optional[str] = None
    terminating_event_cid: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active_at(self, timestamp: float) -> bool:
        """Check whether this state record is active at a given point in time."""
        if timestamp < self.t_start:
            return False
        if self.t_end is not None and timestamp >= self.t_end:
            return False
        return True

    @property
    def is_current(self) -> bool:
        """Check whether this state record is currently open-ended and active."""
        return self.t_end is None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state record to a plain dictionary."""
        return {
            "entity_cid": self.entity_cid,
            "property_slot": self.property_slot,
            "value_cid": self.value_cid,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "epistemic_status": self.epistemic_status,
            "event_cid": self.event_cid,
            "terminating_event_cid": self.terminating_event_cid,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityStateRecord:
        """Construct an EntityStateRecord from a dictionary."""
        return cls(
            entity_cid=str(data["entity_cid"]),
            property_slot=str(data["property_slot"]),
            value_cid=str(data["value_cid"]),
            t_start=float(data["t_start"]),
            t_end=float(data["t_end"]) if data.get("t_end") is not None else None,
            epistemic_status=str(data.get("epistemic_status", "FACT")),
            event_cid=data.get("event_cid"),
            terminating_event_cid=data.get("terminating_event_cid"),
            metadata=dict(data.get("metadata", {})),
        )


class WorldStateManager:
    """Thread-safe dynamic world model tracking fluents, transitions, and belief revisions.

    Maintains temporal interval states for entities across discourse chunks, solves
    non-monotonic transitions by closing past intervals (t_end) and asserting new ones,
    and enables zero-cost temporal point-in-time state queries.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        mutually_exclusive_properties: Optional[Set[str]] = None,
    ):
        """Initialize WorldStateManager.

        Args:
            db_path: Optional SQLite database file path for durable persistence.
            mutually_exclusive_properties: Optional set of fluent slot names treated as
                mutually exclusive (defaults to MUTUALLY_EXCLUSIVE_PROPERTIES).
        """
        self._lock = threading.RLock()
        self.mutually_exclusive_properties: Set[str] = set(
            mutually_exclusive_properties or MUTUALLY_EXCLUSIVE_PROPERTIES
        )

        # In-memory indices
        self._records: List[EntityStateRecord] = []
        # (entity_cid, property_slot) -> active record
        self._active_records: Dict[Tuple[str, str], EntityStateRecord] = {}
        # entity_cid -> list of all historical records
        self._by_entity: Dict[str, List[EntityStateRecord]] = {}

        # Entity node lookup: CID -> QuantaNode
        self._node_cache: Dict[str, QuantaNode] = {}
        # Canonical / surface name -> CID
        self._name_to_cid: Dict[str, str] = {}
        # CID -> canonical name
        self._cid_to_name: Dict[str, str] = {}

        # Pending graph edges: list of (src_cid, relation, tgt_cid)
        self._pending_graph_edges: List[Tuple[str, str, str]] = []

        # Monotonic virtual clock fallback for timeless narratives
        self._virtual_clock: float = 0.0

        # Tier-2 SQLite persistence
        self._db_path = str(db_path) if db_path is not None else None
        self._conn: Optional[sqlite3.Connection] = None
        if self._db_path is not None:
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._init_sqlite_schema()

    def _init_sqlite_schema(self):
        """Initialize SQLite persistence tables."""
        if self._conn is None:
            return
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_state_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_cid TEXT NOT NULL,
                    property_slot TEXT NOT NULL,
                    value_cid TEXT NOT NULL,
                    t_start REAL NOT NULL,
                    t_end REAL,
                    epistemic_status TEXT NOT NULL,
                    event_cid TEXT,
                    terminating_event_cid TEXT,
                    metadata_json TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_esr_entity_prop ON entity_state_records(entity_cid, property_slot)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_esr_time ON entity_state_records(t_start, t_end)"
            )

    def _save_record_to_sqlite(self, rec: EntityStateRecord):
        """Persist an EntityStateRecord into SQLite."""
        if self._conn is None:
            return
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO entity_state_records (
                    entity_cid, property_slot, value_cid, t_start, t_end,
                    epistemic_status, event_cid, terminating_event_cid, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.entity_cid,
                    rec.property_slot,
                    rec.value_cid,
                    rec.t_start,
                    rec.t_end,
                    rec.epistemic_status,
                    rec.event_cid,
                    rec.terminating_event_cid,
                    json.dumps(rec.metadata),
                ),
            )

    def _update_record_t_end_in_sqlite(self, rec: EntityStateRecord):
        """Update t_end and terminating_event_cid in SQLite."""
        if self._conn is None:
            return
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE entity_state_records
                SET t_end = ?, terminating_event_cid = ?
                WHERE entity_cid = ? AND property_slot = ? AND value_cid = ? AND t_start = ?
                """,
                (
                    rec.t_end,
                    rec.terminating_event_cid,
                    rec.entity_cid,
                    rec.property_slot,
                    rec.value_cid,
                    rec.t_start,
                ),
            )

    def normalize_property_slot(self, prop: str) -> str:
        """Normalize property or slot name to canonical form."""
        clean = prop.strip().lower()
        return PROPERTY_SLOT_ALIASES.get(clean, prop)

    def register_node(
        self,
        node: QuantaNode,
        aliases: Optional[Sequence[str]] = None,
    ):
        """Register a QuantaNode and its lexical aliases in internal lookup tables."""
        with self._lock:
            cid = node.cid
            self._node_cache[cid] = node
            if hasattr(node, "canonical_cid"):
                self._node_cache[node.canonical_cid] = node

            lit = (node.literal or "").strip()
            anc = (node.anchor or "").strip()

            if lit:
                self._name_to_cid[lit.lower()] = cid
                self._cid_to_name[cid] = lit
                # Also index clean version
                clean_lit = re.sub(r"^(?:the|a|an|dr\.|mr\.|ms\.)\s+", "", lit.lower()).strip()
                if clean_lit:
                    self._name_to_cid[clean_lit] = cid
            if anc:
                self._name_to_cid[anc.lower()] = cid
                if cid not in self._cid_to_name:
                    self._cid_to_name[cid] = anc

            if aliases:
                for a in aliases:
                    clean_a = str(a).strip().lower()
                    if clean_a:
                        self._name_to_cid[clean_a] = cid

    def resolve_entity_cid(self, entity_cid_or_name: str) -> str:
        """Resolve entity name, alias, or CID string into canonical CID."""
        with self._lock:
            if entity_cid_or_name in self._node_cache:
                return entity_cid_or_name
            clean = entity_cid_or_name.strip().lower()
            if clean in self._name_to_cid:
                return self._name_to_cid[clean]
            stripped = re.sub(r"^(?:the|a|an|dr\.|mr\.|ms\.)\s+", "", clean).strip()
            if stripped in self._name_to_cid:
                return self._name_to_cid[stripped]
            # Match substring in name_to_cid keys
            for name, cid in self._name_to_cid.items():
                if clean == name or clean in name or name in clean:
                    return cid
            return entity_cid_or_name

    def assert_state(
        self,
        entity_cid: str,
        property_slot: str,
        value_cid: str,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        epistemic_status: str = "FACT",
        event_cid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        graph: Optional[QuantaGraph] = None,
    ) -> EntityStateRecord:
        """Directly asserts a fluent state record, performing non-monotonic revision if needed."""
        with self._lock:
            slot = self.normalize_property_slot(property_slot)
            ent_cid = self.resolve_entity_cid(entity_cid)
            val_cid = self.resolve_entity_cid(value_cid)

            if t_start is None:
                self._virtual_clock += 1.0
                t_start = self._virtual_clock
            else:
                self._virtual_clock = max(self._virtual_clock, float(t_start))

            # Non-Monotonic Transition Resolution:
            # If property is mutually exclusive, close prior open state
            prior_record = self._active_records.get((ent_cid, slot))
            if prior_record is not None and slot in self.mutually_exclusive_properties:
                if prior_record.value_cid != val_cid or prior_record.t_end is None:
                    prior_record.t_end = t_start
                    prior_record.terminating_event_cid = event_cid
                    self._update_record_t_end_in_sqlite(prior_record)

                    # Wire TEMP_ALLEN_FINISHES edge into graph if both events exist
                    if prior_record.event_cid and event_cid:
                        self._pending_graph_edges.append(
                            (prior_record.event_cid, "TEMP_ALLEN_FINISHES", event_cid)
                        )
                        if graph is not None:
                            try:
                                if prior_record.event_cid in graph.nodes and event_cid in graph.nodes:
                                    graph.add_edge(prior_record.event_cid, "TEMP_ALLEN_FINISHES", event_cid)
                            except Exception:
                                pass

            # Open new state record
            new_record = EntityStateRecord(
                entity_cid=ent_cid,
                property_slot=slot,
                value_cid=val_cid,
                t_start=t_start,
                t_end=t_end,
                epistemic_status=epistemic_status,
                event_cid=event_cid,
                metadata=metadata or {},
            )

            self._records.append(new_record)
            self._active_records[(ent_cid, slot)] = new_record
            if ent_cid not in self._by_entity:
                self._by_entity[ent_cid] = []
            self._by_entity[ent_cid].append(new_record)

            self._save_record_to_sqlite(new_record)
            return new_record

    def update_from_event(
        self,
        event: Union[QuantaNode, ExtractedEvent, Dict[str, Any]],
        graph: Optional[QuantaGraph] = None,
        current_time: Optional[float] = None,
    ) -> List[EntityStateRecord]:
        """Inspects an event and executes non-monotonic belief revision if state transitions occur.

        Handles:
        1. Relocation/Motion events:
           - Moving/transferring entity into a location closes its prior location interval
             at event.time_start and asserts the new location starting at event.time_start.
           - Wires TEMP_ALLEN_FINISHES edge between the prior event and the current event.
        2. Static state assertion events:
           - Sets initial or updated entity location or state.
        """
        with self._lock:
            # Normalize event representation
            event_cid: Optional[str] = None
            pred = ""
            agent_id: Optional[str] = None
            patient_id: Optional[str] = None
            theme_id: Optional[str] = None
            location_id: Optional[str] = None
            t_start: Optional[float] = None
            t_end: Optional[float] = None
            raw_text = ""
            epistemic_status = "FACT"

            if isinstance(event, QuantaNode):
                event_cid = event.cid
                pred = str(event.literal or event.anchor or "").lower()
                t_start = getattr(event, "time_start", None)
                t_end = getattr(event, "time_end", None)
                raw_text = str(event.literal or "")
                # Extract valencies from edges
                if "VAL_X1_AGENT" in event.edges and event.edges["VAL_X1_AGENT"]:
                    agent_id = event.edges["VAL_X1_AGENT"][0]
                if "VAL_X2_PATIENT" in event.edges and event.edges["VAL_X2_PATIENT"]:
                    patient_id = event.edges["VAL_X2_PATIENT"][0]
                if "VAL_LOCATION_SLOT" in event.edges and event.edges["VAL_LOCATION_SLOT"]:
                    location_id = event.edges["VAL_LOCATION_SLOT"][0]
            elif isinstance(event, ExtractedEvent):
                event_cid = event.id
                pred = (event.predicate or "").lower()
                agent_id = event.agent_id
                patient_id = event.patient_id
                theme_id = event.theme_id
                location_id = event.location_id
                t_start = event.time_start
                t_end = event.time_end
                raw_text = event.raw_text or ""
                if t_start is None and event.temporal_anchor:
                    t_start = parse_timestamp(event.temporal_anchor)
                if t_start is None and event.time_interval:
                    t_start = parse_timestamp(event.time_interval.start)
                if t_end is None and event.time_interval:
                    t_end = parse_timestamp(event.time_interval.end)
            elif isinstance(event, dict):
                event_cid = event.get("id") or event.get("cid")
                pred = str(event.get("predicate") or "").lower()
                agent_id = event.get("agent_id") or event.get("agent")
                patient_id = event.get("patient_id") or event.get("patient")
                theme_id = event.get("theme_id") or event.get("theme")
                location_id = event.get("location_id") or event.get("location")
                t_start = parse_timestamp(event.get("time_start") or event.get("temporal_anchor"))
                t_end = parse_timestamp(event.get("time_end"))
                raw_text = event.get("raw_text") or ""

            # Check timestamp overrides
            if t_start is None and current_time is not None:
                t_start = current_time
            if t_start is None and raw_text:
                t_start = parse_timestamp(raw_text)

            # Determine verb stem / action
            pred_words = set(re.findall(r"\b[a-z]+\b", pred.lower()))
            if raw_text:
                pred_words.update(re.findall(r"\b[a-z]+\b", raw_text.lower()))

            is_motion = bool(pred_words.intersection(MOTION_VERBS))
            is_state = bool(pred_words.intersection(STATE_VERBS))

            created_records: List[EntityStateRecord] = []

            # 1. Location state transition
            if location_id is not None:
                # Target entity: prefer patient or theme (transitive movement of object)
                # Fallback to agent (intransitive self-movement, e.g. "Eleanor walked into Chamber B")
                target_entity = patient_id or theme_id or agent_id
                if target_entity is not None:
                    rec = self.assert_state(
                        entity_cid=target_entity,
                        property_slot="VAL_LOCATION_SLOT",
                        value_cid=location_id,
                        t_start=t_start,
                        t_end=t_end,
                        epistemic_status=epistemic_status,
                        event_cid=event_cid,
                        metadata={
                            "predicate": pred,
                            "raw_text": raw_text,
                            "motion": is_motion,
                            "state": is_state,
                        },
                        graph=graph,
                    )
                    created_records.append(rec)

            return created_records

    def get_entity_state_record_at(
        self,
        entity_cid_or_name: Optional[str] = None,
        property_name: str = "VAL_LOCATION_SLOT",
        timestamp: Optional[float] = None,
        entity_name_or_cid: Optional[str] = None,
    ) -> Optional[EntityStateRecord]:
        """Finds the EntityStateRecord active at a given timestamp (or current if None)."""
        target = entity_cid_or_name if entity_cid_or_name is not None else entity_name_or_cid
        if target is None:
            return None

        with self._lock:
            ent_cid = self.resolve_entity_cid(target)
            slot = self.normalize_property_slot(property_name)

            records = self._by_entity.get(ent_cid, [])
            # Filter by matching property slot
            matching = [r for r in records if r.property_slot == slot]
            if not matching:
                return None

            if timestamp is None:
                # Query current active record
                for r in reversed(matching):
                    if r.is_current:
                        return r
                # If no open record, return the latest one
                return matching[-1]

            # Query historical point-in-time
            t = float(timestamp)
            for r in reversed(matching):
                if r.is_active_at(t):
                    return r

            return None

    def get_entity_state_at(
        self,
        entity_cid_or_name: Optional[str] = None,
        property_name: str = "VAL_LOCATION_SLOT",
        timestamp: Optional[float] = None,
        graph: Optional[QuantaGraph] = None,
        entity_name_or_cid: Optional[str] = None,
    ) -> Optional[QuantaNode]:
        """Returns the QuantaNode representing the property value valid at a specific timestamp.

        Args:
            entity_cid_or_name: Entity CID or surface name (e.g. 'specimen').
            property_name: Property name or slot (e.g. 'VAL_LOCATION_SLOT' or 'location').
            timestamp: Specific point-in-time timestamp (or None for current state).
            graph: Optional QuantaGraph to resolve node from if not in internal cache.
            entity_name_or_cid: Alias parameter for entity_cid_or_name.

        Returns:
            QuantaNode representing the value (e.g. Containment Cell 4 node), or None.
        """
        target = entity_cid_or_name if entity_cid_or_name is not None else entity_name_or_cid
        if target is None:
            return None

        with self._lock:
            rec = self.get_entity_state_record_at(
                entity_cid_or_name=target,
                property_name=property_name,
                timestamp=timestamp,
            )
            if rec is None:
                return None

            val_cid = rec.value_cid
            if val_cid in self._node_cache:
                return self._node_cache[val_cid]
            if graph is not None:
                node = graph.get_node(val_cid)
                if node is not None:
                    self._node_cache[val_cid] = node
                    return node

            # Check CanonicalNodeInterner if available
            from core.asg import QuantaNode
            interner = QuantaNode.get_default_interner()
            if interner is not None and hasattr(interner, "lookup"):
                node = interner.lookup(canonical_cid=val_cid)
                if node is not None:
                    self._node_cache[val_cid] = node
                    return node

            # Return synthetic stub node if node object is missing but CID is known
            val_name = self._cid_to_name.get(val_cid, val_cid)
            stub = QuantaNode(anchor=f"entity:{val_name}", literal=val_name)
            return stub

    def get_state_history(
        self,
        entity_cid_or_name: Optional[str] = None,
        property_name: Optional[str] = None,
        entity_name_or_cid: Optional[str] = None,
    ) -> List[EntityStateRecord]:
        """Returns the complete chronological history of state records for an entity."""
        target = entity_cid_or_name if entity_cid_or_name is not None else entity_name_or_cid
        if target is None:
            return []

        with self._lock:
            ent_cid = self.resolve_entity_cid(target)
            records = list(self._by_entity.get(ent_cid, []))
            if property_name is not None:
                slot = self.normalize_property_slot(property_name)
                records = [r for r in records if r.property_slot == slot]
            records.sort(key=lambda r: r.t_start)
            return records

    def explain_state_timeline(
        self,
        entity_cid_or_name: Optional[str] = None,
        property_name: str = "VAL_LOCATION_SLOT",
        entity_name_or_cid: Optional[str] = None,
    ) -> List[str]:
        """Returns human-readable explanation strings describing state transitions over time."""
        target = entity_cid_or_name if entity_cid_or_name is not None else entity_name_or_cid
        if target is None:
            return []

        with self._lock:
            records = self.get_state_history(target, property_name)
            if not records:
                return []

            ent_name = self._cid_to_name.get(
                self.resolve_entity_cid(target), target
            )
            lines: List[str] = []
            for r in records:
                val_name = self._cid_to_name.get(r.value_cid, r.value_cid)
                if r.t_end is None:
                    lines.append(
                        f"From {r.t_start:g} onwards: {ent_name} is in {val_name} (current)"
                    )
                else:
                    lines.append(
                        f"From {r.t_start:g} to {r.t_end:g}: {ent_name} was in {val_name}"
                    )
            return lines

    def sync_graph(self, graph: QuantaGraph):
        """Synchronizes world state with a QuantaGraph.

        - Indexes all nodes in the graph into internal lookup cache.
        - Resolves entity and event local IDs (E1, Ev1) to compiled node CIDs.
        - Applies any pending cross-chunk TEMP_ALLEN_FINISHES relation edges.
        """
        with self._lock:
            for node in graph.nodes.values():
                self.register_node(node)

            # Map entity_nodes if attached by compiler
            entity_nodes = getattr(graph, "entity_nodes", {})
            for eid, node in entity_nodes.items():
                self.register_node(node)
                self._name_to_cid[eid] = node.cid
                self._name_to_cid[eid.lower()] = node.cid
                if node.literal:
                    self._name_to_cid[str(node.literal).lower()] = node.cid
                for rec in self._records:
                    if rec.entity_cid in (eid, eid.lower()):
                        rec.entity_cid = node.cid
                    if rec.value_cid in (eid, eid.lower()):
                        rec.value_cid = node.cid

            # Map event_nodes if attached by compiler
            event_nodes = getattr(graph, "event_nodes", {})
            for evid, node in event_nodes.items():
                self.register_node(node)
                self._name_to_cid[evid] = node.cid
                self._name_to_cid[evid.lower()] = node.cid
                for rec in self._records:
                    if rec.event_cid in (evid, evid.lower()):
                        rec.event_cid = node.cid
                    if rec.terminating_event_cid in (evid, evid.lower()):
                        rec.terminating_event_cid = node.cid


            # Rebuild _by_entity and _active_records cleanly from updated _records
            self._by_entity = {}
            self._active_records = {}
            for rec in self._records:
                if rec.entity_cid not in self._by_entity:
                    self._by_entity[rec.entity_cid] = []
                self._by_entity[rec.entity_cid].append(rec)
                if rec.is_current:
                    self._active_records[(rec.entity_cid, rec.property_slot)] = rec

            # Apply pending edges
            remaining: List[Tuple[str, str, str]] = []
            for src_id, rel, tgt_id in self._pending_graph_edges:
                src_cid = self.resolve_entity_cid(src_id)
                tgt_cid = self.resolve_entity_cid(tgt_id)
                if src_cid in graph.nodes and tgt_cid in graph.nodes:
                    try:
                        graph.add_edge(src_cid, rel, tgt_cid)
                    except Exception:
                        remaining.append((src_id, rel, tgt_id))
                else:
                    remaining.append((src_id, rel, tgt_id))
            self._pending_graph_edges = remaining

    def close(self):
        """Close SQLite database connection if open."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None



    def reset(self):
        """Reset internal world state, active records, and node caches."""
        with self._lock:
            self._records.clear()
            self._active_records.clear()
            self._by_entity.clear()
            self._node_cache.clear()
            self._name_to_cid.clear()
            self._cid_to_name.clear()
            self._pending_graph_edges.clear()
            self._virtual_clock = 0.0


__all__ = [
    "EntityStateRecord",
    "WorldStateManager",
    "PROPERTY_SLOT_ALIASES",
    "MUTUALLY_EXCLUSIVE_PROPERTIES",
    "MOTION_VERBS",
    "STATE_VERBS",
    "parse_timestamp",
]
