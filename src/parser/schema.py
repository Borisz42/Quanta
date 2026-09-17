"""Typed intermediate schemas for QUANTA Neural Discourse Transduction (Phase 3).

Defines the canonical, strongly-typed JSON intermediate representation produced
by the Neural Discourse Transducer (Qwen-4B / LM Studio / GGUF) and consumed by
the Symbolic ASG Compiler (Phase 4).

Guarantees:
1. Pure semantic representation: zero punctuation or word-token nodes.
2. Explicit foreign keys for entities (e.g., agent_id: 'E1', patient_id: 'E2').
3. Strict typed models for entities, events, spatio-temporal/causal relations,
   and epistemic propositions.
4. Cryptographic readiness and foreign-key integrity validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Set, Union


# Ontological category constants for ExtractedEntity
ENTITY_CATEGORIES: Set[str] = {
    "PERSON",
    "OBJECT",
    "SUBSTANCE",
    "LOCATION",
    "ORGANIZATION",
    "ANIMAL",
    "ARTIFACT",
    "NATURAL_OBJECT",
    "INSTRUMENT",
}

# Standard grammatical tense constants for ExtractedEvent
EVENT_TENSES: Set[str] = {
    "PAST",
    "PRESENT",
    "FUTURE",
    "PAST_PERFECT",
    "PRESENT_PERFECT",
    "FUTURE_PERFECT",
}

# Standard epistemic statuses for ExtractedProposition
EPISTEMIC_STATUSES: Set[str] = {
    "FACT",
    "HYPOTHESIS",
    "OBSERVATION",
    "DOUBTED",
    "PROHIBITED",
    "BELIEF",
    "KNOWLEDGE",
    "UNVERIFIED",
}


@dataclass
class ExtractedEntity:
    """Canonical physical or agentive entity extracted from discourse.

    Attributes:
        id: Unique canonical identifier within chunk/session (e.g. 'E1', 'E2').
        canonical_name: Primary descriptor (e.g. 'Dr. Eleanor Vance').
        category: Ontological category (e.g. 'PERSON', 'SUBSTANCE', 'LOCATION').
        surface_aliases: Surface text aliases and pronouns (e.g. ['Eleanor', 'Vance', 'she']).
        properties: Additional metadata (e.g. role, state, attributes).
    """
    id: str
    canonical_name: str
    category: str = "OBJECT"
    surface_aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to a plain serializable dictionary."""
        return {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "category": self.category,
            "surface_aliases": list(self.surface_aliases),
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedEntity:
        """Construct an ExtractedEntity from a dictionary with key normalization."""
        cid = data.get("id") or data.get("canonical_id") or ""
        name = data.get("canonical_name") or data.get("name") or ""
        cat = data.get("category") or "OBJECT"
        aliases = data.get("surface_aliases") or data.get("aliases") or []
        props = data.get("properties") or {}
        return cls(
            id=str(cid),
            canonical_name=str(name),
            category=str(cat).upper(),
            surface_aliases=[str(a) for a in aliases if a],
            properties=dict(props),
        )


@dataclass
class ExtractedEvent:
    """Discrete physical, cognitive, communicative, or relational event predicate.

    Attributes:
        id: Unique event identifier (e.g. 'Ev1', 'Ev2').
        predicate: Verb lemma or root action descriptor (e.g. 'isolate', 'verify').
        agent_id: Foreign key to ExtractedEntity (Band 1 VAL_X1_AGENT).
        patient_id: Foreign key to ExtractedEntity (Band 1 VAL_X2_PATIENT).
        theme_id: Optional theme / topic entity foreign key.
        location_id: Foreign key to ExtractedEntity (Band 1 VAL_LOCATION_SLOT).
        instrument_id: Foreign key to ExtractedEntity (Band 1 VAL_X5_INSTRUMENT).
        temporal_anchor: Surface temporal descriptor (e.g. 'at dawn', 'three hours later').
        tense: Grammatical tense (e.g. 'PAST', 'PRESENT', 'FUTURE').
        aspect: Grammatical aspect (e.g. 'SIMPLE', 'PERFECT', 'PROGRESSIVE').
        polarity: Boolean polarity (True for affirmative, False for negated).
        modality: Optional modal flavor (e.g. 'CERTAIN', 'POSSIBLE', 'OBLIGATION').
        raw_text: Source sentence or clause string for provenance tracking.
        arguments: Extensible argument map for adverbs, manner, or custom slots.
    """
    id: str
    predicate: str
    agent_id: Optional[str] = None
    patient_id: Optional[str] = None
    theme_id: Optional[str] = None
    location_id: Optional[str] = None
    instrument_id: Optional[str] = None
    temporal_anchor: Optional[str] = None
    time_start: Optional[Union[int, float]] = None
    time_end: Optional[Union[int, float]] = None
    tense: str = "PAST"
    aspect: str = "SIMPLE"
    polarity: bool = True
    modality: Optional[str] = None
    raw_text: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to a plain serializable dictionary."""
        return {
            "id": self.id,
            "predicate": self.predicate,
            "agent_id": self.agent_id,
            "patient_id": self.patient_id,
            "theme_id": self.theme_id,
            "location_id": self.location_id,
            "instrument_id": self.instrument_id,
            "temporal_anchor": self.temporal_anchor,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "tense": self.tense,
            "aspect": self.aspect,
            "polarity": self.polarity,
            "modality": self.modality,
            "raw_text": self.raw_text,
            "arguments": dict(self.arguments),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedEvent:
        """Construct an ExtractedEvent from a dictionary with key normalization."""
        ev_id = data.get("id") or data.get("event_id") or ""
        pred = data.get("predicate") or data.get("verb") or data.get("action") or ""
        return cls(
            id=str(ev_id),
            predicate=str(pred).lower(),
            agent_id=data.get("agent_id") or data.get("agent"),
            patient_id=data.get("patient_id") or data.get("patient"),
            theme_id=data.get("theme_id") or data.get("theme"),
            location_id=data.get("location_id") or data.get("location"),
            instrument_id=data.get("instrument_id") or data.get("instrument"),
            temporal_anchor=data.get("temporal_anchor") or data.get("time"),
            time_start=data.get("time_start"),
            time_end=data.get("time_end"),
            tense=str(data.get("tense", "PAST")).upper(),
            aspect=str(data.get("aspect", "SIMPLE")).upper(),
            polarity=bool(data.get("polarity", True)),
            modality=data.get("modality"),
            raw_text=data.get("raw_text") or data.get("text"),
            arguments=dict(data.get("arguments") or {}),
        )


@dataclass
class ExtractedRelation:
    """Inter-event spatio-temporal, causal, or dependency relation.

    Maps to Band 7 Allen Intervals (e.g. TEMP_ALLEN_MEETS, TEMP_ALLEN_BEFORE)
    or Pearl Causal DAG links (e.g. CAUSAL_MECHANISM_LINK, CAUSAL_PREVENTIVE_BLOCK).

    Attributes:
        relation_type: Formal relation name or Allen/Causal slot name.
        source_id: Originating event or entity ID.
        target_id: Destination event or entity ID.
        mechanism: Optional explanation or linking phrase (e.g. 'prompted by structural retention').
        confidence: Confidence score in [0.0, 1.0].
    """
    relation_type: str
    source_id: str
    target_id: str
    mechanism: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary."""
        return {
            "relation_type": self.relation_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "mechanism": self.mechanism,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedRelation:
        """Construct ExtractedRelation from dictionary."""
        rel = data.get("relation_type") or data.get("type") or "TEMP_ALLEN_AFTER"
        src = data.get("source_id") or data.get("source") or ""
        tgt = data.get("target_id") or data.get("target") or ""
        mech = data.get("mechanism") or data.get("description")
        conf = float(data.get("confidence", 1.0))
        return cls(
            relation_type=str(rel),
            source_id=str(src),
            target_id=str(tgt),
            mechanism=mech,
            confidence=conf,
        )


@dataclass
class ExtractedProposition:
    """Epistemic claim, observation, hypothesis, or higher-order proposition.

    Attributes:
        id: Unique proposition identifier (e.g. 'P1', 'P2').
        claim_text: Full claim text (e.g. 'specimen exhibited anomalous lattice expansion').
        predicate: Root predicate of the proposition.
        subject_id: Entity ID or Event ID that forms the subject of the proposition.
        epistemic_status: Status in {FACT, HYPOTHESIS, OBSERVATION, DOUBTED, PROHIBITED}.
        source_agent_id: Foreign key to ExtractedEntity holding or asserting the belief.
        event_id: Optional event ID that established or tested this proposition.
        properties: Additional properties.
    """
    id: str
    claim_text: str
    predicate: Optional[str] = None
    subject_id: Optional[str] = None
    epistemic_status: str = "FACT"
    source_agent_id: Optional[str] = None
    event_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert proposition to dictionary."""
        return {
            "id": self.id,
            "claim_text": self.claim_text,
            "predicate": self.predicate,
            "subject_id": self.subject_id,
            "epistemic_status": self.epistemic_status,
            "source_agent_id": self.source_agent_id,
            "event_id": self.event_id,
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedProposition:
        """Construct ExtractedProposition from dictionary."""
        pid = data.get("id") or data.get("prop_id") or ""
        claim = data.get("claim_text") or data.get("claim") or data.get("text") or ""
        return cls(
            id=str(pid),
            claim_text=str(claim),
            predicate=data.get("predicate"),
            subject_id=data.get("subject_id") or data.get("subject"),
            epistemic_status=str(data.get("epistemic_status", "FACT")).upper(),
            source_agent_id=data.get("source_agent_id") or data.get("source_agent"),
            event_id=data.get("event_id"),
            properties=dict(data.get("properties") or {}),
        )


@dataclass
class DiscourseExtractionResult:
    """Normalized output container from Neural Discourse Transduction.

    Contains all entities, events, spatio-temporal/causal relations, and
    epistemic propositions extracted from a single discourse chunk.
    """
    chunk_id: Optional[str] = None
    entities: List[ExtractedEntity] = field(default_factory=list)
    events: List[ExtractedEvent] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    propositions: List[ExtractedProposition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_entity(self, entity_id: str) -> Optional[ExtractedEntity]:
        """Find entity by ID."""
        for ent in self.entities:
            if ent.id == entity_id:
                return ent
        return None

    def get_event(self, event_id: str) -> Optional[ExtractedEvent]:
        """Find event by ID."""
        for ev in self.events:
            if ev.id == event_id:
                return ev
        return None

    def get_proposition(self, prop_id: str) -> Optional[ExtractedProposition]:
        """Find proposition by ID."""
        for p in self.propositions:
            if p.id == prop_id:
                return p
        return None

    def validate_foreign_keys(self) -> List[str]:
        """Validate all foreign-key cross-references across entities, events, and relations.

        Returns:
            List of error messages; empty if 100% valid.
        """
        errors: List[str] = []
        entity_ids = {e.id for e in self.entities}
        event_ids = {ev.id for ev in self.events}
        prop_ids = {p.id for p in self.propositions}
        all_ids = entity_ids | event_ids | prop_ids

        # Validate event entity references
        for ev in self.events:
            if ev.agent_id and ev.agent_id not in entity_ids:
                errors.append(f"Event '{ev.id}' agent_id '{ev.agent_id}' does not exist in entities.")
            if ev.patient_id and ev.patient_id not in entity_ids:
                errors.append(f"Event '{ev.id}' patient_id '{ev.patient_id}' does not exist in entities.")
            if ev.theme_id and ev.theme_id not in entity_ids:
                errors.append(f"Event '{ev.id}' theme_id '{ev.theme_id}' does not exist in entities.")
            if ev.location_id and ev.location_id not in entity_ids:
                errors.append(f"Event '{ev.id}' location_id '{ev.location_id}' does not exist in entities.")
            if ev.instrument_id and ev.instrument_id not in entity_ids:
                errors.append(f"Event '{ev.id}' instrument_id '{ev.instrument_id}' does not exist in entities.")

        # Validate relations
        for rel in self.relations:
            if rel.source_id and rel.source_id not in all_ids:
                errors.append(f"Relation '{rel.relation_type}' source_id '{rel.source_id}' does not exist.")
            if rel.target_id and rel.target_id not in all_ids:
                errors.append(f"Relation '{rel.relation_type}' target_id '{rel.target_id}' does not exist.")

        # Validate propositions
        for p in self.propositions:
            if p.source_agent_id and p.source_agent_id not in entity_ids:
                errors.append(f"Proposition '{p.id}' source_agent_id '{p.source_agent_id}' does not exist in entities.")
            if p.event_id and p.event_id not in event_ids:
                errors.append(f"Proposition '{p.id}' event_id '{p.event_id}' does not exist in events.")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire extraction result to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "entities": [e.to_dict() for e in self.entities],
            "events": [ev.to_dict() for ev in self.events],
            "relations": [r.to_dict() for r in self.relations],
            "propositions": [p.to_dict() for p in self.propositions],
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize extraction result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscourseExtractionResult:
        """Construct DiscourseExtractionResult from dictionary."""
        entities = [ExtractedEntity.from_dict(e) for e in data.get("entities", [])]
        events = [ExtractedEvent.from_dict(ev) for ev in data.get("events", [])]
        relations = [ExtractedRelation.from_dict(r) for r in data.get("relations", [])]
        propositions = [ExtractedProposition.from_dict(p) for p in data.get("propositions", [])]
        return cls(
            chunk_id=data.get("chunk_id"),
            entities=entities,
            events=events,
            relations=relations,
            propositions=propositions,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> DiscourseExtractionResult:
        """Construct DiscourseExtractionResult from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """Return canonical JSON Schema for OpenAI-compatible structured output."""
        return {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "canonical_name": {"type": "string"},
                            "category": {
                                "type": "string",
                                "enum": [
                                    "PERSON",
                                    "OBJECT",
                                    "SUBSTANCE",
                                    "LOCATION",
                                    "ORGANIZATION",
                                    "ANIMAL",
                                    "ARTIFACT",
                                    "NATURAL_OBJECT",
                                    "INSTRUMENT",
                                ],
                            },
                            "surface_aliases": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "properties": {"type": "object"},
                        },
                        "required": ["id", "canonical_name", "category", "surface_aliases"],
                    },
                },
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "predicate": {"type": "string"},
                            "agent_id": {"type": ["string", "null"]},
                            "patient_id": {"type": ["string", "null"]},
                            "theme_id": {"type": ["string", "null"]},
                            "location_id": {"type": ["string", "null"]},
                            "instrument_id": {"type": ["string", "null"]},
                            "temporal_anchor": {"type": ["string", "null"]},
                            "tense": {"type": "string"},
                            "aspect": {"type": "string"},
                            "polarity": {"type": "boolean"},
                            "modality": {"type": ["string", "null"]},
                            "raw_text": {"type": ["string", "null"]},
                        },
                        "required": ["id", "predicate", "tense", "polarity"],
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "relation_type": {"type": "string"},
                            "source_id": {"type": "string"},
                            "target_id": {"type": "string"},
                            "mechanism": {"type": ["string", "null"]},
                            "confidence": {"type": "number"},
                        },
                        "required": ["relation_type", "source_id", "target_id"],
                    },
                },
                "propositions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "claim_text": {"type": "string"},
                            "predicate": {"type": ["string", "null"]},
                            "subject_id": {"type": ["string", "null"]},
                            "epistemic_status": {"type": "string"},
                            "source_agent_id": {"type": ["string", "null"]},
                            "event_id": {"type": ["string", "null"]},
                        },
                        "required": ["id", "claim_text", "epistemic_status"],
                    },
                },
            },
            "required": ["entities", "events", "relations", "propositions"],
        }
