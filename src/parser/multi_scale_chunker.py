"""Hierarchical Chunker and Multi-Scale Segmentation Engine for QUANTA (§2.1).

Segments documents into two synchronized tiers of discourse representation:
1. MacroBlock: Coarse structural/thematic units (~800-1500 tokens), tracking concept codes
   and surface entities for high-speed pre-filtering and coarse semantic anchoring.
2. DiscourseChunk (Micro): Fine-grained episodic units (150-350 words) suitable for neural
   transduction and precise realization bypass.

Each micro chunk explicitly references its parent MacroBlock via `parent_macro_id`.
Concept profiling is performed via zero-copy MmapLexicalGrounder.resolve_concept_code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from config.multi_scale_config import MultiScaleConfig
from parser.chunker import DiscourseChunk, DiscourseChunker
from parser.mmap_grounder import MmapLexicalGrounder

logger = logging.getLogger("quanta.parser.multi_scale_chunker")

# Stopwords and generic terms to avoid cluttering concept codes
GENERIC_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "can", "could", "should", "would", "may", "might", "must",
    "this", "that", "these", "those", "it", "its", "they", "them", "their", "we", "us",
    "our", "you", "your", "he", "him", "his", "she", "her", "which", "who", "whom", "what",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "just", "don", "shouldn", "now", "also", "then", "into",
}

# Regex to extract candidate surface terms (capitalized entities and content words)
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b")
WORD_PATTERN = re.compile(r"\b[a-zA-Z]{3,}\b")


@dataclass
class MacroBlock:
    """A coarse structural block encompassing one or more micro DiscourseChunks.

    Attributes:
        macro_id: Unique deterministic identifier (e.g., 'macro_0000').
        char_span: Document-level character start and end offsets (global_start, global_end).
        text: Verbatim concatenated text of the block.
        micro_chunks: Ordered list of child DiscourseChunk objects.
        concept_codes: List of unique ConceptNet uint16 codes grounded from the block.
        surface_entities: List of surface entity strings extracted from the block.
        token_count_estimate: Estimated token count of the block text.
    """
    macro_id: str
    char_span: Tuple[int, int]
    text: str
    micro_chunks: List[DiscourseChunk] = field(default_factory=list)
    concept_codes: List[int] = field(default_factory=list)
    surface_entities: List[str] = field(default_factory=list)
    token_count_estimate: int = 0

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize MacroBlock to JSON-compatible dictionary."""
        return {
            "macro_id": self.macro_id,
            "char_span": list(self.char_span),
            "text": self.text,
            "micro_chunks": [c.to_dict() for c in self.micro_chunks],
            "concept_codes": list(self.concept_codes),
            "surface_entities": list(self.surface_entities),
            "token_count_estimate": self.token_count_estimate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MacroBlock:
        """Construct MacroBlock from serialized dictionary."""
        chunks = [
            DiscourseChunk.from_dict(c)
            for c in data.get("micro_chunks", [])
        ]
        span = tuple(data.get("char_span", [0, 0]))
        return cls(
            macro_id=str(data["macro_id"]),
            char_span=(int(span[0]), int(span[1])),
            text=str(data["text"]),
            micro_chunks=chunks,
            concept_codes=[int(c) for c in data.get("concept_codes", [])],
            surface_entities=[str(e) for e in data.get("surface_entities", [])],
            token_count_estimate=int(data.get("token_count_estimate", 0)),
        )


class HierarchicalChunker:
    """Two-tier hierarchical document segmenter.

    Partitions documents into coarse MacroBlocks and fine DiscourseChunks while
    recording bi-directional parent/child linkages and profiling ConceptNet concept codes.
    """

    DEFAULT_MACRO_TARGET_TOKENS = 1000
    DEFAULT_MICRO_TARGET_WORDS = 250

    def __init__(
        self,
        macro_target_tokens: Optional[int] = None,
        micro_target_words: Optional[int] = None,
        config: Optional[MultiScaleConfig] = None,
        grounder: Optional[MmapLexicalGrounder] = None,
    ):
        """Initialize the HierarchicalChunker.

        Args:
            macro_target_tokens: Target token count for coarse MacroBlocks.
            micro_target_words: Target word count for micro DiscourseChunks.
            config: Optional MultiScaleConfig instance for parameter lookup.
            grounder: Optional MmapLexicalGrounder instance for concept code resolution.
        """
        cfg = config or MultiScaleConfig()
        self.macro_target_tokens = (
            macro_target_tokens
            if macro_target_tokens is not None
            else (cfg.chunker_macro_target_tokens or self.DEFAULT_MACRO_TARGET_TOKENS)
        )
        self.micro_target_words = (
            micro_target_words
            if micro_target_words is not None
            else (cfg.chunker_micro_target_words or self.DEFAULT_MICRO_TARGET_WORDS)
        )

        # Micro chunker uses min_words ~ half of target, max_words ~ 1.4x target
        min_micro = max(80, int(self.micro_target_words * 0.6))
        max_micro = max(min_micro + 50, int(self.micro_target_words * 1.4))
        self.micro_chunker = DiscourseChunker(
            min_words=min_micro,
            max_words=max_micro,
        )

        self.grounder = grounder or MmapLexicalGrounder.get_default()

        # Telemetry counters for profiling cost
        self.total_profiling_time_ms: float = 0.0
        self.total_concept_queries: int = 0
        self.total_concept_hits: int = 0

    def chunk_document(
        self,
        text: str,
    ) -> Tuple[List[MacroBlock], List[DiscourseChunk]]:
        """Segment input text into synchronized MacroBlocks and micro DiscourseChunks.

        Returns:
            Tuple of (macro_blocks, micro_chunks).
        """
        if not text or not text.strip():
            return [], []

        # 1. Segment text into micro chunks via DiscourseChunker
        micro_chunks = self.micro_chunker.chunk_document(text)
        if not micro_chunks:
            return [], []

        # 2. Group micro chunks into MacroBlocks based on macro_target_tokens
        macro_blocks: List[MacroBlock] = []
        current_macro_micros: List[DiscourseChunk] = []
        current_macro_tokens: int = 0
        macro_idx = 0

        def finalize_macro_block(micros: List[DiscourseChunk], idx: int) -> MacroBlock:
            mid = f"macro_{idx:04d}"
            start_char = micros[0].global_offset
            end_char = micros[-1].global_end_offset
            block_text = text[start_char:end_char]
            token_estimate = sum(m.token_count_estimate for m in micros)

            # Link micro chunks to this macro block
            for m in micros:
                m.parent_macro_id = mid

            # Extract surface entities and concept codes
            entities, concept_codes = self._profile_block_concepts(block_text)

            return MacroBlock(
                macro_id=mid,
                char_span=(start_char, end_char),
                text=block_text,
                micro_chunks=list(micros),
                concept_codes=concept_codes,
                surface_entities=entities,
                token_count_estimate=token_estimate,
            )

        for micro in micro_chunks:
            est_tokens = micro.token_count_estimate or int(micro.word_count * 1.33)
            # If current block already has chunks and adding this micro exceeds target,
            # cut current block (unless current is empty)
            if current_macro_micros and (current_macro_tokens + est_tokens > self.macro_target_tokens):
                mb = finalize_macro_block(current_macro_micros, macro_idx)
                macro_blocks.append(mb)
                macro_idx += 1
                current_macro_micros = [micro]
                current_macro_tokens = est_tokens
            else:
                current_macro_micros.append(micro)
                current_macro_tokens += est_tokens

        if current_macro_micros:
            mb = finalize_macro_block(current_macro_micros, macro_idx)
            macro_blocks.append(mb)

        return macro_blocks, micro_chunks

    def _profile_block_concepts(self, block_text: str) -> Tuple[List[str], List[int]]:
        """Extract salient surface entities and ground concept codes.

        Measures and tracks execution time and hit rate.
        """
        t0 = time.perf_counter()

        # Extract capitalized entities
        entity_matches = ENTITY_PATTERN.findall(block_text)
        entities: List[str] = []
        seen_entities: Set[str] = set()
        for ent in entity_matches:
            ent_clean = ent.strip()
            ent_lower = ent_clean.lower()
            if ent_lower not in GENERIC_STOPWORDS and ent_lower not in seen_entities and len(ent_clean) > 2:
                seen_entities.add(ent_lower)
                entities.append(ent_clean)

        # Extract candidate content words for concept codes
        words = WORD_PATTERN.findall(block_text)
        candidate_terms: Set[str] = set()
        for ent in entities:
            candidate_terms.add(ent.lower())
        for w in words:
            w_lower = w.lower()
            if w_lower not in GENERIC_STOPWORDS:
                candidate_terms.add(w_lower)

        # Query concept codes via zero-copy MmapLexicalGrounder
        concept_codes: List[int] = []
        seen_codes: Set[int] = set()

        if self.grounder and self.grounder.is_available():
            for term in candidate_terms:
                self.total_concept_queries += 1
                code = self.grounder.resolve_concept_code(term)
                if code is not None:
                    self.total_concept_hits += 1
                    if code not in seen_codes:
                        seen_codes.add(code)
                        concept_codes.append(code)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.total_profiling_time_ms += dt_ms

        return entities, sorted(concept_codes)

    def profiling_stats(self) -> Dict[str, Any]:
        """Return cumulative telemetry for concept profiling."""
        total_q = self.total_concept_queries
        hit_rate = (self.total_concept_hits / total_q) if total_q > 0 else 0.0
        avg_ms = (self.total_profiling_time_ms / total_q) if total_q > 0 else 0.0
        return {
            "total_profiling_time_ms": self.total_profiling_time_ms,
            "total_concept_queries": self.total_concept_queries,
            "total_concept_hits": self.total_concept_hits,
            "hit_rate": hit_rate,
            "avg_ms_per_query": avg_ms,
        }
