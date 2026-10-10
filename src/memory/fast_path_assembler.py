"""Fast-Path Context Assembler & Mode Bush for QUANTA Dynamic Ingestion (§Phase 3, exp-029*).

Settles the main dynamic ingestion design question:
Should the first answer wait for full graph construction, or answer from
filtered raw text and build the graph asynchronously?

Variants:
- M-A: raw-only (filtered raw spans only, 0 synchronous GPU transduction calls)
- M-B: hot-transduce(N) (graph briefing from top-N filtered chunks + raw spans)
- M-C: full (full graph transduction & PPR applied to kept set)
- M-D: coverage-adaptive (M-A when coverage >= threshold, else M-B)
- passthrough: default B0 behavior (full ingestion of document)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from config.multi_scale_config import MultiScaleConfig
from core.asg import QuantaGraph, QuantaNode
from memory.context_assembler import (
    DualStreamContext,
    DualStreamContextAssembler,
    ProjectedPassage,
)
from memory.passage_store import PassageRecord, PassageStore
from parser.chunker import DiscourseChunk
from parser.multi_scale_chunker import MacroBlock
from parser.task_boundary_extractor import ExtractedTaskIntent

logger = logging.getLogger("quanta.memory.fast_path")


class FastPathMode(str, Enum):
    """Enumeration of Fast-Path operational modes."""
    RAW_ONLY = "raw_only"               # M-A
    HOT_TRANSDUCE = "hot_transduce"     # M-B
    FULL = "full"                       # M-C
    COVERAGE_ADAPTIVE = "coverage_adaptive" # M-D
    PASSTHROUGH = "passthrough"         # Control / B0


@dataclass
class FastPathContext:
    """Structured container holding fast-path assembled context."""
    context_text: str
    stream1_briefing: str
    stream2_passages: str
    mode: str
    transduced_count: int
    total_kept_count: int
    coverage_score: Optional[float] = None
    estimated_tokens: int = 0
    passages: List[ProjectedPassage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FastPathResult:
    """Comprehensive container for the result of answer_long_context."""
    answer: str
    context: str
    isolated_query: str
    context_document: str
    mode: str
    coverage_score: Optional[float]
    units_scored: int
    units_kept: int
    units_transduced: int
    deferred_units: int
    subgraph: Optional[QuantaGraph]
    stage_timings: Dict[str, float]
    gpu_calls: int
    tokens_estimated: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "context": self.context,
            "isolated_query": self.isolated_query,
            "context_document": self.context_document[:200] + ("..." if len(self.context_document) > 200 else ""),
            "mode": self.mode,
            "coverage_score": self.coverage_score,
            "units_scored": self.units_scored,
            "units_kept": self.units_kept,
            "units_transduced": self.units_transduced,
            "deferred_units": self.deferred_units,
            "stage_timings": self.stage_timings,
            "gpu_calls": self.gpu_calls,
            "tokens_estimated": self.tokens_estimated,
            "metadata": self.metadata,
        }


class FastPathCoverageEvaluator:
    """Evaluates whether filtered units provide sufficient coverage for query intent.

    Used by M-D (coverage-adaptive) to decide whether to bypass transduction
    (raw-only) or trigger hot-transduction.
    """

    DEFAULT_STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "is", "was", "are", "were", "what", "which", "who", "whom", "where", "when",
        "why", "how", "did", "does", "do", "done", "has", "have", "had", "by", "from",
        "this", "that", "these", "those", "it", "its", "their", "them", "they", "he",
        "she", "his", "her", "be", "been", "being", "about", "into", "through",
    }

    def __init__(self, stopword_set: Optional[Set[str]] = None):
        self.stopwords = stopword_set or self.DEFAULT_STOPWORDS

    def check_coverage(
        self,
        intent: Union[ExtractedTaskIntent, str],
        units: Sequence[Any],
        target_entities: Optional[Sequence[str]] = None,
    ) -> float:
        """Calculates a coverage score in [0.0, 1.0].

        Args:
            intent: ExtractedTaskIntent or raw query string.
            units: Sequence of ScoredUnit, DiscourseChunk, MacroBlock, or text strings.
            target_entities: Optional list of explicit target entities from query.

        Returns:
            Float coverage score in [0.0, 1.0].
        """
        if isinstance(intent, ExtractedTaskIntent):
            query_text = intent.query_text
            entities = list(intent.target_entities or [])
        else:
            query_text = str(intent)
            entities = list(target_entities or [])

        if not query_text.strip() and not entities:
            return 1.0
        if not units:
            return 0.0

        # Collect concatenated text of all kept units
        texts = []
        for u in units:
            if hasattr(u, "unit") and hasattr(u.unit, "text"):
                texts.append(u.unit.text)
            elif hasattr(u, "text"):
                texts.append(u.text)
            else:
                texts.append(str(u))
        combined_text_lower = " ".join(texts).lower()

        # 1. Entity Coverage
        entity_cov: Optional[float] = None
        if entities:
            matched_entities = sum(1 for e in entities if e.strip() and e.lower().strip() in combined_text_lower)
            entity_cov = float(matched_entities / len(entities))

        # 2. Key Term Coverage
        q_clean = re.sub(r"[^\w\s]", " ", query_text.lower())
        q_terms = [t for t in q_clean.split() if t not in self.stopwords and len(t) >= 2]
        if q_terms:
            matched_terms = sum(1 for t in q_terms if t in combined_text_lower)
            term_cov = float(matched_terms / len(q_terms))
        else:
            term_cov = 1.0

        # Weighted combination
        if entity_cov is not None:
            combined = 0.6 * entity_cov + 0.4 * term_cov
        else:
            combined = term_cov

        return float(max(0.0, min(1.0, round(combined, 4))))


class FastPathAssembler:
    """Orchestrates fast-path context construction across variants M-A..M-D (§Phase 3).

    Reuses DualStreamContextAssembler. When the graph is empty (e.g. M-A raw-only),
    stream 1 (logical briefing) is omitted.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        hot_transduce_n: Optional[int] = None,
        coverage_threshold: Optional[float] = None,
        config: Optional[MultiScaleConfig] = None,
        dual_stream_assembler: Optional[DualStreamContextAssembler] = None,
        coverage_evaluator: Optional[FastPathCoverageEvaluator] = None,
    ):
        self.config = config
        raw_mode = mode or (config.fast_path_mode if config else "passthrough")
        self.mode = self.normalize_mode(raw_mode)

        if hot_transduce_n is not None:
            self.hot_transduce_n = int(hot_transduce_n)
        elif config and config.fast_path_hot_transduce_n is not None:
            self.hot_transduce_n = int(config.fast_path_hot_transduce_n)
        else:
            self.hot_transduce_n = 2

        if coverage_threshold is not None:
            self.coverage_threshold = float(coverage_threshold)
        elif config and config.fast_path_coverage_threshold is not None:
            self.coverage_threshold = float(config.fast_path_coverage_threshold)
        else:
            self.coverage_threshold = 0.75

        self.dual_stream_assembler = dual_stream_assembler or DualStreamContextAssembler()
        self.coverage_evaluator = coverage_evaluator or FastPathCoverageEvaluator()

    @classmethod
    def normalize_mode(cls, mode_str: Optional[str]) -> str:
        """Normalizes mode alias into canonical FastPathMode string."""
        if not mode_str:
            return FastPathMode.PASSTHROUGH.value
        m = str(mode_str).lower().strip().replace("-", "_")
        if m in ("m_a", "m_a_raw_only", "raw_only", "raw"):
            return FastPathMode.RAW_ONLY.value
        if m in ("m_b", "m_b_hot_transduce", "hot_transduce", "hot"):
            return FastPathMode.HOT_TRANSDUCE.value
        if m in ("m_c", "m_c_full", "full"):
            return FastPathMode.FULL.value
        if m in ("m_d", "m_d_coverage_adaptive", "coverage_adaptive", "adaptive"):
            return FastPathMode.COVERAGE_ADAPTIVE.value
        if m in ("passthrough", "b0", "control"):
            return FastPathMode.PASSTHROUGH.value
        return m

    def resolve_effective_mode(
        self,
        intent: Union[ExtractedTaskIntent, str],
        kept_units: Sequence[Any],
        target_entities: Optional[Sequence[str]] = None,
    ) -> Tuple[str, float]:
        """Resolves operational mode and calculates coverage score.

        In M-D (coverage-adaptive), dynamically chooses RAW_ONLY if coverage
        meets coverage_threshold, else HOT_TRANSDUCE.
        """
        cov_score = self.coverage_evaluator.check_coverage(
            intent=intent,
            units=kept_units,
            target_entities=target_entities,
        )

        if self.mode == FastPathMode.COVERAGE_ADAPTIVE.value:
            if cov_score >= self.coverage_threshold:
                effective = FastPathMode.RAW_ONLY.value
            else:
                effective = FastPathMode.HOT_TRANSDUCE.value
            return effective, cov_score

        return self.mode, cov_score

    def _units_to_projected_passages(
        self,
        units: Sequence[Any],
        passage_store: Optional[PassageStore] = None,
    ) -> List[ProjectedPassage]:
        """Converts heterogeneous discourse units into ProjectedPassage records."""
        projected: List[ProjectedPassage] = []

        for idx, u in enumerate(units):
            if isinstance(u, ProjectedPassage):
                projected.append(u)
                continue

            # ScoredUnit from relevance_filter
            if hasattr(u, "unit") and hasattr(u, "score"):
                inner = u.unit
                text = inner.text if hasattr(inner, "text") else str(inner)
                pid = getattr(u, "unit_id", f"P_{idx+1}")
                doc_id = getattr(inner, "doc_id", "doc_01")
                span = getattr(u, "char_span", (0, len(text)))
                score = float(u.score)
            elif isinstance(u, (DiscourseChunk, MacroBlock)):
                text = u.text
                pid = getattr(u, "chunk_id", getattr(u, "macro_id", f"P_{idx+1}"))
                doc_id = getattr(u, "doc_id", "doc_01")
                span = getattr(u, "char_span", (0, len(text)))
                score = 1.0
            elif isinstance(u, PassageRecord):
                text = u.text
                pid = u.passage_id
                doc_id = u.doc_id
                span = u.char_span
                score = 1.0
            else:
                text = str(u)
                pid = f"P_{idx+1}"
                doc_id = "doc_01"
                span = (0, len(text))
                score = 1.0

            # Register in passage store if available and not present
            if passage_store is not None and pid not in passage_store:
                try:
                    passage_store.add_passage(
                        PassageRecord(
                            passage_id=pid,
                            doc_id=doc_id,
                            char_span=span,
                            text=text,
                        )
                    )
                except Exception:
                    pass

            p_rec = ProjectedPassage(
                passage_id=pid,
                doc_id=doc_id,
                text=text,
                score=score,
                char_span=span,
            )
            projected.append(p_rec)

        return projected

    def assemble(
        self,
        subgraph: Optional[QuantaGraph] = None,
        kept_units: Sequence[Any] = (),
        passage_store: Optional[PassageStore] = None,
        max_tokens: int = 2500,
        query_text: str = "",
        effective_mode: Optional[str] = None,
        coverage_score: Optional[float] = None,
        activations: Optional[Dict[str, float]] = None,
        transduced_count: Optional[int] = None,
    ) -> FastPathContext:
        """Assembles prompt context according to effective fast-path mode.

        Args:
            subgraph: Optional retrieved QuantaGraph ASG.
            kept_units: Kept discourse units from relevance filter.
            passage_store: Optional PassageStore instance.
            max_tokens: Maximum token budget for context.
            query_text: Natural user query.
            effective_mode: Resolved mode (overrides self.mode if provided).
            coverage_score: Precomputed coverage score.
            activations: Optional node activation scores for PPR.
            transduced_count: Explicit count of synchronously transduced units.

        Returns:
            FastPathContext container with assembled context and stream breakdowns.
        """
        mode = self.normalize_mode(effective_mode or self.mode)
        passages = self._units_to_projected_passages(kept_units, passage_store=passage_store)

        has_graph = subgraph is not None and len(subgraph.nodes) > 0
        total_kept = len(kept_units)

        # ---------------------------------------------------------------------
        # Variant M-A: raw-only
        # Filtered raw spans only. Stream 1 is omitted.
        # ---------------------------------------------------------------------
        if mode == FastPathMode.RAW_ONLY.value:
            stream1_briefing = ""
            stream2_passages = self.dual_stream_assembler.assemble_passage_stream(
                passages=passages,
                max_tokens=max_tokens,
            )
            full_context = stream2_passages
            actual_transduced = 0

        # ---------------------------------------------------------------------
        # Variant M-B: hot-transduce(N)
        # Graph briefing from top-N filtered chunks + raw spans
        # ---------------------------------------------------------------------
        elif mode == FastPathMode.HOT_TRANSDUCE.value:
            actual_transduced = transduced_count if transduced_count is not None else min(self.hot_transduce_n, total_kept)

            if has_graph:
                logical_budget = max(10, int(max_tokens * self.dual_stream_assembler.logical_ratio))
                stream1_briefing = self.dual_stream_assembler.assemble_logical_briefing(
                    subgraph=subgraph,
                    max_tokens=logical_budget,
                )
            else:
                stream1_briefing = ""

            briefing_tokens = int(len(stream1_briefing.split()) * 1.33) if stream1_briefing else 0
            passage_budget = max(10, max_tokens - briefing_tokens)

            stream2_passages = self.dual_stream_assembler.assemble_passage_stream(
                passages=passages,
                max_tokens=passage_budget,
            )

            if stream1_briefing and stream2_passages:
                full_context = f"{stream1_briefing}\n\n{stream2_passages}"
            else:
                full_context = stream1_briefing or stream2_passages

        # ---------------------------------------------------------------------
        # Variant M-C: full & Control / passthrough (B0)
        # Full B0 dual-stream behavior applied over kept set (or full context)
        # ---------------------------------------------------------------------
        else:
            actual_transduced = transduced_count if transduced_count is not None else total_kept

            if has_graph:
                dual_ctx = self.dual_stream_assembler.assemble_dual_stream_context(
                    subgraph=subgraph,
                    passage_store=passage_store,
                    max_tokens=max_tokens,
                    query_text=query_text,
                    activations=activations,
                )
                stream1_briefing = dual_ctx.logical_briefing
                stream2_passages = dual_ctx.passage_stream
                full_context = dual_ctx.full_context
                if dual_ctx.passages:
                    passages = dual_ctx.passages
            else:
                # Fallback if no graph nodes produced: emit raw passage stream
                stream1_briefing = ""
                stream2_passages = self.dual_stream_assembler.assemble_passage_stream(
                    passages=passages,
                    max_tokens=max_tokens,
                )
                full_context = stream2_passages

        est_tokens = int(len(full_context.split()) * 1.33) if full_context else 0

        return FastPathContext(
            context_text=full_context,
            stream1_briefing=stream1_briefing,
            stream2_passages=stream2_passages,
            mode=mode,
            transduced_count=actual_transduced,
            total_kept_count=total_kept,
            coverage_score=coverage_score,
            estimated_tokens=est_tokens,
            passages=passages,
            metadata={
                "hot_transduce_n": self.hot_transduce_n,
                "coverage_threshold": self.coverage_threshold,
                "has_graph": has_graph,
            },
        )


__all__ = [
    "FastPathAssembler",
    "FastPathContext",
    "FastPathCoverageEvaluator",
    "FastPathMode",
    "FastPathResult",
]
