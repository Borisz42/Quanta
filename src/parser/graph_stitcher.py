"""Multi-Chunk Directed Acyclic Graph (DAG) Stitcher for QUANTA (Phase 3).

Aggregates multiple single-chunk extraction results (DiscourseExtractionResult)
into a unified, coherent narrative Directed Acyclic Graph (DAG):
1. Global Entity Resolution & Canonicalization:
   - Maps chunk-local entity IDs to globally unique canonical IDs (E1, E2, ...).
   - Matches entities across surface aliases, canonical names, and ontological categories.
   - Merges newly discovered surface aliases and properties into canonical entity records.
   - Seamlessly integrates with ActiveEntityManifest / EntityPagingEngine.
2. Event & Relation Re-Mapping:
   - Re-indexes event IDs across chunks (Ev1, Ev2, ...).
   - Updates all event foreign keys (agent_id, patient_id, theme_id, location_id, instrument_id)
     to point to the resolved global entity IDs.
   - Re-maps relation endpoints (source_id, target_id) for intra-chunk relations.
3. Cross-Chunk Temporal Interval Synthesis:
   - For sequential narrative chunks without explicit temporal gaps, synthesizes inter-chunk
     Allen temporal relations (TEMP_ALLEN_MEETS or TEMP_ALLEN_BEFORE) linking the terminal
     event of chunk N to the initial event of chunk N+1.
   - Preserves DAG acyclicity and temporal monotonicity.
4. Proposition Re-Mapping:
   - Re-indexes proposition IDs (P1, P2, ...).
   - Updates source_agent_id, event_id, and subject_id to resolved global IDs.
5. Compilation Bridge:
   - stitch(chunks) -> DiscourseExtractionResult
   - stitch_to_graph(chunks, compiler) -> QuantaGraph
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph
from parser.asg_compiler import ASGCompiler
from parser.entity_manifest import (
    ActiveEntityManifest,
    EntityMatcher,
    EntityPagingEngine,
    EntityRecord,
    EntityStorage,
)
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)


# Titles and honorifics stripped during entity name normalization
HONORIFICS: Set[str] = {
    "dr", "dr.", "doctor",
    "mr", "mr.", "mister",
    "ms", "ms.", "miss",
    "mrs", "mrs.",
    "prof", "prof.", "professor",
    "sir", "madam", "lady", "lord",
    "director", "supervisor", "officer", "agent",
}

# Articles and possessives stripped during token matching
ARTICLES_AND_POSSESSIVES: Set[str] = {
    "a", "an", "the",
    "his", "her", "its", "their", "our", "my", "your",
    "this", "that", "these", "those",
}

# Pronouns excluded from triggering false-positive entity unifications
GENERIC_PRONOUNS: Set[str] = {
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "someone", "something", "anyone", "anything", "one",
}

# Regex patterns indicating explicit temporal gaps between episodes
TEMPORAL_GAP_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:hours?|days?|weeks?|months?|years?|minutes?)\s+later\b", re.IGNORECASE),
    re.compile(r"\b(?:the\s+)?next\s+(?:day|morning|afternoon|evening|night|week|month|year)\b", re.IGNORECASE),
    re.compile(r"\bafter\s+(?:a\s+while|some\s+time|several|hours?|days?)\b", re.IGNORECASE),
    re.compile(r"\bthe\s+following\s+(?:day|morning|afternoon|evening|night)\b", re.IGNORECASE),
    re.compile(r"\blater\s+(?:that|the)\s+(?:day|afternoon|evening|night)\b", re.IGNORECASE),
    re.compile(r"\bsubsequently\b", re.IGNORECASE),
    re.compile(r"\bmeanwhile\b", re.IGNORECASE),
]


def _clean_str(text: Optional[str]) -> str:
    """Normalize whitespace and strip string."""
    if not text:
        return ""
    return " ".join(text.strip().split())


def _normalize_name(name: str) -> str:
    """Normalize entity name for matching (lowercase, stripped honorifics & articles)."""
    clean = _clean_str(name).lower()
    words = clean.split()
    while words and words[0].strip(".,") in HONORIFICS:
        words.pop(0)
    while words and words[0] in ARTICLES_AND_POSSESSIVES:
        words.pop(0)
    return " ".join(words)


def _extract_name_tokens(name: str) -> Set[str]:
    """Extract informative lowercase tokens from an entity name or alias."""
    clean = _clean_str(name).lower()
    # Remove punctuation
    tokens = re.findall(r"\b[a-z0-9_-]+\b", clean)
    informative = {
        t for t in tokens
        if t not in HONORIFICS and t not in ARTICLES_AND_POSSESSIVES and t not in GENERIC_PRONOUNS and len(t) > 1
    }
    return informative


def _categories_compatible(cat1: str, cat2: str) -> bool:
    """Check whether two ontological categories are compatible for entity merging."""
    c1 = (cat1 or "OBJECT").upper()
    c2 = (cat2 or "OBJECT").upper()

    if c1 == c2:
        return True

    # Generic OBJECT is compatible with any physical category
    if c1 == "OBJECT" or c2 == "OBJECT":
        return True

    # Agentive / Animate groupings
    animate_set = {"PERSON", "ORGANIZATION", "ANIMAL"}
    if c1 in animate_set and c2 in animate_set:
        return True

    # Inanimate physical object groupings
    inanimate_set = {"SUBSTANCE", "ARTIFACT", "NATURAL_OBJECT", "INSTRUMENT"}
    if c1 in inanimate_set and c2 in inanimate_set:
        return True

    # Cross-domain clashes (e.g. PERSON vs LOCATION, PERSON vs SUBSTANCE)
    return False


def _detect_temporal_gap(event: ExtractedEvent, chunk_text: Optional[str] = None) -> bool:
    """Check whether an event or chunk indicates an explicit temporal gap."""
    sources: List[str] = []
    if event.temporal_anchor:
        sources.append(event.temporal_anchor)
    if event.raw_text:
        sources.append(event.raw_text)
    if chunk_text:
        sources.append(chunk_text)

    for src in sources:
        for pat in TEMPORAL_GAP_PATTERNS:
            if pat.search(src):
                return True
    return False


class GraphStitcher:
    """Aggregates multi-chunk discourse extraction results into a unified narrative DAG.

    Performs global identity consolidation, cross-chunk foreign-key re-mapping,
    and inter-chunk Allen temporal interval synthesis.
    """

    def __init__(
        self,
        manifest: Optional[ActiveEntityManifest] = None,
        paging_engine: Optional[EntityPagingEngine] = None,
        storage: Optional[EntityStorage] = None,
        default_temporal_relation: str = "TEMP_ALLEN_MEETS",
        auto_detect_temporal_gap: bool = True,
        entity_similarity_threshold: float = 0.7,
        compiler: Optional[ASGCompiler] = None,
        alias_clusters: Optional[List[Iterable[str]]] = None,
        coreference_map: Optional[Dict[str, str]] = None,
    ):
        """Initialize GraphStitcher.

        Args:
            manifest: Optional working memory ActiveEntityManifest instance.
            paging_engine: Optional EntityPagingEngine instance.
            storage: Optional persistent EntityStorage instance.
            default_temporal_relation: Relation synthesized between adjacent chunks
                ('TEMP_ALLEN_MEETS' or 'TEMP_ALLEN_BEFORE').
            auto_detect_temporal_gap: If True, uses 'TEMP_ALLEN_BEFORE' when a temporal
                gap phrase (e.g. 'three hours later') is detected on the chunk boundary.
            entity_similarity_threshold: Minimum matching confidence score to merge entities.
            compiler: Optional ASGCompiler for stitch_to_graph.
            alias_clusters: Optional explicit list of alias/name sets known to corefer.
            coreference_map: Optional explicit mapping from surface name to canonical name.
        """
        self.storage = storage if storage is not None else EntityStorage(":memory:")
        self.paging_engine = paging_engine if paging_engine is not None else EntityPagingEngine(db_path=":memory:")
        self.manifest = manifest if manifest is not None else self.paging_engine.manifest
        self.default_temporal_relation = default_temporal_relation
        self.auto_detect_temporal_gap = auto_detect_temporal_gap
        self.entity_similarity_threshold = entity_similarity_threshold
        self.compiler = compiler

        # Explicit coreference overrides
        self.alias_clusters: List[Set[str]] = [
            {_clean_str(a).lower() for a in cluster if _clean_str(a)}
            for cluster in (alias_clusters or [])
        ]
        self.coreference_map: Dict[str, str] = {
            _clean_str(k).lower(): _clean_str(v)
            for k, v in (coreference_map or {}).items()
            if _clean_str(k)
        }

        # Internal stitching state
        self._global_entities: Dict[str, ExtractedEntity] = {}
        self._global_events: Dict[str, ExtractedEvent] = {}
        self._global_relations: List[ExtractedRelation] = []
        self._global_propositions: Dict[str, ExtractedProposition] = {}

        # Symbol tables: (chunk_index, local_id) -> global_id
        self._entity_id_map: Dict[Tuple[int, str], str] = {}
        self._event_id_map: Dict[Tuple[int, str], str] = {}
        self._prop_id_map: Dict[Tuple[int, str], str] = {}

        # Sequencing counters
        self._chunk_count: int = 0
        self._chunk_ids: List[str] = []
        self._next_entity_num: int = 1
        self._next_event_num: int = 1
        self._next_prop_num: int = 1
        self._last_terminal_event_id: Optional[str] = None

    def reset(self):
        """Reset internal stitching state for a fresh document run."""
        self._global_entities.clear()
        self._global_events.clear()
        self._global_relations.clear()
        self._global_propositions.clear()
        self._entity_id_map.clear()
        self._event_id_map.clear()
        self._prop_id_map.clear()
        self._chunk_count = 0
        self._chunk_ids.clear()
        self._next_entity_num = 1
        self._next_event_num = 1
        self._next_prop_num = 1
        self._last_terminal_event_id = None
        self.manifest.reset()

    # ---------------------------------------------------------------------------
    # Global Entity Resolution & Matching
    # ---------------------------------------------------------------------------

    def _match_entity(self, incoming: ExtractedEntity, existing: ExtractedEntity) -> float:
        """Compute matching confidence score in [0.0, 1.0] between two entities."""
        # 1. Ontological category compatibility check
        if not _categories_compatible(incoming.category, existing.category):
            return 0.0

        inc_name_norm = _normalize_name(incoming.canonical_name)
        exist_name_norm = _normalize_name(existing.canonical_name)

        # 2. Check explicit coreference map or alias clusters
        if self.coreference_map:
            for alias in [incoming.canonical_name] + incoming.surface_aliases:
                target = self.coreference_map.get(_clean_str(alias).lower())
                if target:
                    target_norm = _normalize_name(target)
                    if target_norm in (exist_name_norm, _normalize_name(existing.canonical_name)):
                        return 1.0

        if self.alias_clusters:
            all_inc = {_clean_str(a).lower() for a in [incoming.canonical_name] + incoming.surface_aliases if _clean_str(a)}
            all_exist = {_clean_str(a).lower() for a in [existing.canonical_name] + existing.surface_aliases if _clean_str(a)}
            for cluster in self.alias_clusters:
                if (all_inc & cluster) and (all_exist & cluster):
                    return 1.0

        # 3. Exact normalized canonical name match
        if inc_name_norm and inc_name_norm == exist_name_norm:
            return 1.0

        # 4. Surface alias cross-checks
        inc_aliases_norm = {_normalize_name(a) for a in incoming.surface_aliases if _clean_str(a)}
        exist_aliases_norm = {_normalize_name(a) for a in existing.surface_aliases if _clean_str(a)}

        # Filter out empty or generic pronoun names
        inc_informative = {a for a in inc_aliases_norm | {inc_name_norm} if a and a not in GENERIC_PRONOUNS and len(a) > 1}
        exist_informative = {a for a in exist_aliases_norm | {exist_name_norm} if a and a not in GENERIC_PRONOUNS and len(a) > 1}

        # Exact alias overlap
        shared = inc_informative & exist_informative
        if shared:
            return 1.0

        # 5. Token overlap / Substring matching
        inc_tokens = _extract_name_tokens(incoming.canonical_name)
        for a in incoming.surface_aliases:
            inc_tokens |= _extract_name_tokens(a)

        exist_tokens = _extract_name_tokens(existing.canonical_name)
        for a in existing.surface_aliases:
            exist_tokens |= _extract_name_tokens(a)

        if inc_tokens and exist_tokens:
            intersection = inc_tokens & exist_tokens
            if intersection:
                # If one set of tokens is a complete subset of the other
                if inc_tokens.issubset(exist_tokens) or exist_tokens.issubset(inc_tokens):
                    return 0.95

                # Jaccard similarity across tokens
                jaccard = len(intersection) / len(inc_tokens | exist_tokens)
                if jaccard >= 0.5:
                    return 0.90
                if jaccard >= 0.33 and (incoming.category == existing.category == "PERSON"):
                    return 0.80

        # 6. ActiveEntityManifest matching fallback
        if self.manifest.is_active(existing.id):
            active_rec = self.manifest.get(existing.id)
            if active_rec:
                for a in incoming.surface_aliases + [incoming.canonical_name]:
                    clean_a = _clean_str(a).lower()
                    if clean_a and clean_a not in GENERIC_PRONOUNS:
                        if clean_a in {al.lower() for al in active_rec.surface_aliases} or clean_a == active_rec.canonical_name.lower():
                            return 0.95

        return 0.0

    def _find_matching_global_entity(self, incoming: ExtractedEntity) -> Optional[ExtractedEntity]:
        """Find the best matching global entity exceeding the similarity threshold."""
        best_match: Optional[ExtractedEntity] = None
        best_score: float = 0.0

        for existing in self._global_entities.values():
            score = self._match_entity(incoming, existing)
            if score > best_score:
                best_score = score
                best_match = existing

        if best_score >= self.entity_similarity_threshold:
            return best_match
        return None

    def _unify_entity(self, target: ExtractedEntity, incoming: ExtractedEntity):
        """Merge incoming entity information into existing canonical global entity."""
        # 1. Deduplicated surface aliases aggregation
        existing_aliases_lower = {a.strip().lower() for a in target.surface_aliases}
        existing_aliases_lower.add(target.canonical_name.strip().lower())

        for alias in [incoming.canonical_name] + incoming.surface_aliases:
            clean = _clean_str(alias)
            if clean and clean.lower() not in existing_aliases_lower:
                target.surface_aliases.append(clean)
                existing_aliases_lower.add(clean.lower())

        # 2. Canonical name refinement: prefer the more complete/descriptive name
        inc_tokens = _extract_name_tokens(incoming.canonical_name)
        tgt_tokens = _extract_name_tokens(target.canonical_name)
        if len(inc_tokens) > len(tgt_tokens) and len(incoming.canonical_name) > len(target.canonical_name):
            # Incoming is more complete (e.g. 'Dr. Eleanor Vance' over 'Eleanor')
            # Add previous canonical name to aliases
            if target.canonical_name not in target.surface_aliases:
                target.surface_aliases.append(target.canonical_name)
            target.canonical_name = incoming.canonical_name

        # 3. Category refinement: upgrade generic OBJECT if incoming is specific
        if target.category == "OBJECT" and incoming.category != "OBJECT":
            target.category = incoming.category

        # 4. Properties merge
        target.properties.update(incoming.properties)

        # 5. Update ActiveEntityManifest / EntityPagingEngine
        rec = self.manifest.get(target.id)
        if rec:
            rec.canonical_name = target.canonical_name
            rec.category = target.category
            rec.add_aliases(target.surface_aliases)
            rec.properties.update(target.properties)
            self.storage.save_entity(rec)

    def _mint_global_entity_id(self) -> str:
        """Mint next unique global entity ID (E1, E2, ...)."""
        while True:
            cid = f"E{self._next_entity_num}"
            self._next_entity_num += 1
            if cid not in self._global_entities:
                return cid

    def _mint_global_event_id(self) -> str:
        """Mint next unique global event ID (Ev1, Ev2, ...)."""
        while True:
            ev_id = f"Ev{self._next_event_num}"
            self._next_event_num += 1
            if ev_id not in self._global_events:
                return ev_id

    def _mint_global_prop_id(self) -> str:
        """Mint next unique global proposition ID (P1, P2, ...)."""
        while True:
            pid = f"P{self._next_prop_num}"
            self._next_prop_num += 1
            if pid not in self._global_propositions:
                return pid

    # ---------------------------------------------------------------------------
    # Core Chunk Ingestion & Stitching
    # ---------------------------------------------------------------------------

    def add_chunk(self, chunk: DiscourseExtractionResult):
        """Ingest and stitch a single discourse extraction chunk into the running DAG."""
        chunk_idx = self._chunk_count
        self._chunk_count += 1
        chunk_id = chunk.chunk_id or f"chunk_{chunk_idx:04d}"
        self._chunk_ids.append(chunk_id)

        # 1. Global Entity Resolution
        for ent in chunk.entities:
            matched = self._find_matching_global_entity(ent)
            if matched is not None:
                self._unify_entity(matched, ent)
                self._entity_id_map[(chunk_idx, ent.id)] = matched.id
            else:
                new_id = self._mint_global_entity_id()
                new_ent = ExtractedEntity(
                    id=new_id,
                    canonical_name=ent.canonical_name,
                    category=ent.category,
                    surface_aliases=list(ent.surface_aliases),
                    properties=dict(ent.properties),
                )
                self._global_entities[new_id] = new_ent
                self._entity_id_map[(chunk_idx, ent.id)] = new_id

                # Register in ActiveEntityManifest
                self.manifest.add_or_update(
                    EntityRecord(
                        canonical_id=new_id,
                        canonical_name=new_ent.canonical_name,
                        category=new_ent.category,
                        surface_aliases=list(new_ent.surface_aliases),
                        last_seen_chunk=chunk_idx,
                        properties=dict(new_ent.properties),
                    ),
                    chunk_idx=chunk_idx,
                )

        # 2. Event Re-Mapping
        current_chunk_remapped_event_ids: List[str] = []
        for ev in chunk.events:
            global_ev_id = self._mint_global_event_id()
            self._event_id_map[(chunk_idx, ev.id)] = global_ev_id
            current_chunk_remapped_event_ids.append(global_ev_id)

            # Remap foreign keys to global entity IDs
            remap_agent = self._entity_id_map.get((chunk_idx, ev.agent_id)) if ev.agent_id else None
            remap_patient = self._entity_id_map.get((chunk_idx, ev.patient_id)) if ev.patient_id else None
            remap_theme = self._entity_id_map.get((chunk_idx, ev.theme_id)) if ev.theme_id else None
            remap_loc = self._entity_id_map.get((chunk_idx, ev.location_id)) if ev.location_id else None
            remap_inst = self._entity_id_map.get((chunk_idx, ev.instrument_id)) if ev.instrument_id else None

            remapped_event = ExtractedEvent(
                id=global_ev_id,
                predicate=ev.predicate,
                agent_id=remap_agent or (ev.agent_id if ev.agent_id in self._global_entities else None),
                patient_id=remap_patient or (ev.patient_id if ev.patient_id in self._global_entities else None),
                theme_id=remap_theme or (ev.theme_id if ev.theme_id in self._global_entities else None),
                location_id=remap_loc or (ev.location_id if ev.location_id in self._global_entities else None),
                instrument_id=remap_inst or (ev.instrument_id if ev.instrument_id in self._global_entities else None),
                temporal_anchor=ev.temporal_anchor,
                tense=ev.tense,
                aspect=ev.aspect,
                polarity=ev.polarity,
                modality=ev.modality,
                raw_text=ev.raw_text,
                arguments=dict(ev.arguments),
            )
            self._global_events[global_ev_id] = remapped_event

        # 3. Cross-Chunk Temporal Interval Synthesis
        if current_chunk_remapped_event_ids:
            initial_event_id = current_chunk_remapped_event_ids[0]
            if self._last_terminal_event_id is not None:
                # Check for explicit temporal gap
                has_gap = False
                if self.auto_detect_temporal_gap and chunk.events:
                    has_gap = _detect_temporal_gap(chunk.events[0])

                rel_type = "TEMP_ALLEN_BEFORE" if has_gap else self.default_temporal_relation

                # Synthesize inter-chunk Allen relation
                inter_chunk_relation = ExtractedRelation(
                    relation_type=rel_type,
                    source_id=self._last_terminal_event_id,
                    target_id=initial_event_id,
                    mechanism="cross_chunk_temporal_sequence",
                    confidence=1.0,
                )
                self._global_relations.append(inter_chunk_relation)

            # Update terminal event tracker
            self._last_terminal_event_id = current_chunk_remapped_event_ids[-1]

        # 4. Intra-Chunk Relation Re-Mapping
        for rel in chunk.relations:
            # Remap source_id
            src_id = rel.source_id
            if (chunk_idx, src_id) in self._event_id_map:
                new_src = self._event_id_map[(chunk_idx, src_id)]
            elif (chunk_idx, src_id) in self._entity_id_map:
                new_src = self._entity_id_map[(chunk_idx, src_id)]
            elif src_id in self._global_events or src_id in self._global_entities:
                new_src = src_id
            else:
                new_src = src_id

            # Remap target_id
            tgt_id = rel.target_id
            if (chunk_idx, tgt_id) in self._event_id_map:
                new_tgt = self._event_id_map[(chunk_idx, tgt_id)]
            elif (chunk_idx, tgt_id) in self._entity_id_map:
                new_tgt = self._entity_id_map[(chunk_idx, tgt_id)]
            elif tgt_id in self._global_events or tgt_id in self._global_entities:
                new_tgt = tgt_id
            else:
                new_tgt = tgt_id

            remapped_rel = ExtractedRelation(
                relation_type=rel.relation_type,
                source_id=new_src,
                target_id=new_tgt,
                mechanism=rel.mechanism,
                confidence=rel.confidence,
            )
            self._global_relations.append(remapped_rel)

        # 5. Proposition Re-Mapping
        for prop in chunk.propositions:
            global_pid = self._mint_global_prop_id()
            self._prop_id_map[(chunk_idx, prop.id)] = global_pid

            # Remap source agent
            remap_src_agent = self._entity_id_map.get((chunk_idx, prop.source_agent_id)) if prop.source_agent_id else None
            # Remap event_id
            remap_ev = self._event_id_map.get((chunk_idx, prop.event_id)) if prop.event_id else None
            # Remap subject_id (could be entity or event)
            remap_sub = None
            if prop.subject_id:
                if (chunk_idx, prop.subject_id) in self._entity_id_map:
                    remap_sub = self._entity_id_map[(chunk_idx, prop.subject_id)]
                elif (chunk_idx, prop.subject_id) in self._event_id_map:
                    remap_sub = self._event_id_map[(chunk_idx, prop.subject_id)]
                else:
                    remap_sub = prop.subject_id

            remapped_prop = ExtractedProposition(
                id=global_pid,
                claim_text=prop.claim_text,
                predicate=prop.predicate,
                subject_id=remap_sub,
                epistemic_status=prop.epistemic_status,
                source_agent_id=remap_src_agent or prop.source_agent_id,
                event_id=remap_ev or prop.event_id,
                properties=dict(prop.properties),
            )
            self._global_propositions[global_pid] = remapped_prop

    def get_result(self) -> DiscourseExtractionResult:
        """Return the consolidated, strongly-typed DiscourseExtractionResult.

        Guarantees foreign-key integrity across all entities, events, relations,
        and propositions.
        """
        stitched_id = (
            f"stitched_{len(self._chunk_ids)}_chunks"
            if self._chunk_ids
            else "stitched_empty"
        )

        result = DiscourseExtractionResult(
            chunk_id=stitched_id,
            entities=list(self._global_entities.values()),
            events=list(self._global_events.values()),
            relations=list(self._global_relations),
            propositions=list(self._global_propositions.values()),
            metadata={
                "stitched_chunks_count": self._chunk_count,
                "source_chunk_ids": list(self._chunk_ids),
                "total_entities": len(self._global_entities),
                "total_events": len(self._global_events),
                "total_relations": len(self._global_relations),
                "total_propositions": len(self._global_propositions),
            },
        )

        # Validate foreign keys
        fk_errors = result.validate_foreign_keys()
        if fk_errors:
            raise ValueError(
                f"GraphStitcher produced invalid foreign key cross-references: {fk_errors}"
            )

        return result

    def stitch(self, chunks: List[DiscourseExtractionResult]) -> DiscourseExtractionResult:
        """Stitch a sequence of discourse chunks into a unified extraction result."""
        self.reset()
        for chunk in chunks:
            self.add_chunk(chunk)
        return self.get_result()

    def stitch_to_graph(
        self,
        chunks: List[DiscourseExtractionResult],
        compiler: Optional[ASGCompiler] = None,
        validate: bool = True,
    ) -> QuantaGraph:
        """Stitch extraction chunks and compile directly into a validated QuantaGraph."""
        stitched_res = self.stitch(chunks)
        asg_comp = compiler or self.compiler or ASGCompiler()
        return asg_comp.compile(stitched_res, validate=validate)


# ---------------------------------------------------------------------------
# Module-Level Convenience Functions
# ---------------------------------------------------------------------------

def stitch(
    chunks: List[DiscourseExtractionResult],
    **kwargs,
) -> DiscourseExtractionResult:
    """Convenience function to stitch extraction results into a unified result."""
    stitcher = GraphStitcher(**kwargs)
    return stitcher.stitch(chunks)


def stitch_to_graph(
    chunks: List[DiscourseExtractionResult],
    compiler: Optional[ASGCompiler] = None,
    validate: bool = True,
    **kwargs,
) -> QuantaGraph:
    """Convenience function to stitch extraction results and compile to QuantaGraph."""
    stitcher = GraphStitcher(**kwargs)
    return stitcher.stitch_to_graph(chunks, compiler=compiler, validate=validate)


__all__ = [
    "GraphStitcher",
    "stitch",
    "stitch_to_graph",
    "HONORIFICS",
    "ARTICLES_AND_POSSESSIVES",
    "GENERIC_PRONOUNS",
    "TEMPORAL_GAP_PATTERNS",
]
