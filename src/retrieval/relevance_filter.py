"""Relevance Filter Engine and Variant Bush for QUANTA Dynamic Ingestion (§2.2).

Provides high-speed scoring and filtering of discourse units (micro chunks or macro blocks)
to discard irrelevant context without dropping gold evidence.

Variants supported:
- F-A: micro lexical BM25 against query (0 GPU calls)
- F-B: micro concept-code overlap via MmapLexicalGrounder (0 GPU calls)
- F-C: micro Kev-4B single-token prefill logprob (n_micro GPU calls)
- F-D: macro (full text) Kev-4B prefill logprob (n_macro GPU calls)
- F-E: macro (head only) Kev-4B prefill logprob (n_macro GPU calls)
- F-F: cascade (lexical/concept prefilter + Kev reranking)
- passthrough: default B0 behavior (keeps 100% of units)

Supports keep policies (threshold, top-k, token budget) and calibration (temperature/Platt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from config.multi_scale_config import MultiScaleConfig
from models.kev_engine import KevDecisionEngine, MockKevEngine, RelevanceScoreResult
from parser.chunker import DiscourseChunk
from parser.mmap_grounder import MmapLexicalGrounder
from parser.multi_scale_chunker import MacroBlock
from parser.task_boundary_extractor import ExtractedTaskIntent


@dataclass
class ScoredUnit:
    """A discourse unit annotated with relevance score, calibration, and token size."""
    unit: Union[DiscourseChunk, MacroBlock]
    unit_id: str
    score: float  # Calibrated probability in [0.0, 1.0]
    raw_score: float  # Raw logit / BM25 / overlap score
    raw_logprob: Optional[float] = None
    tokens: int = 0
    char_span: Tuple[int, int] = (0, 0)
    granularity: str = "micro"  # "micro" or "macro"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "score": self.score,
            "raw_score": self.raw_score,
            "raw_logprob": self.raw_logprob,
            "tokens": self.tokens,
            "char_span": list(self.char_span),
            "granularity": self.granularity,
            "metadata": self.metadata,
        }


class RelevanceFilter:
    """Unified relevance filter orchestrator with pluggable scorers and keep policies."""

    def __init__(
        self,
        strategy: Optional[str] = None,
        unit: Optional[str] = None,
        keep_threshold: Optional[float] = None,
        keep_top_k: Optional[int] = None,
        keep_budget_tokens: Optional[int] = None,
        calibration: Optional[Dict[str, Any]] = None,
        config: Optional[MultiScaleConfig] = None,
        kev_engine: Optional[Union[KevDecisionEngine, MockKevEngine]] = None,
        grounder: Optional[MmapLexicalGrounder] = None,
    ):
        cfg = config or MultiScaleConfig()
        self.strategy = (strategy or cfg.filter_strategy or "passthrough").upper()
        self.unit_type = (unit or cfg.filter_unit or "micro").lower()
        self.keep_threshold = (
            keep_threshold if keep_threshold is not None else cfg.filter_keep_threshold
        )
        self.keep_top_k = (
            keep_top_k if keep_top_k is not None else cfg.filter_keep_top_k
        )
        self.keep_budget_tokens = (
            keep_budget_tokens if keep_budget_tokens is not None else cfg.filter_keep_budget_tokens
        )
        self.calibration = calibration or cfg.get("filter.calibration") or {}

        self.kev_engine = kev_engine or MockKevEngine()
        self.grounder = grounder or MmapLexicalGrounder.get_default()

        # Telemetry
        self.last_scoring_time_ms: float = 0.0
        self.last_gpu_calls: int = 0

    # -----------------------------------------------------------------------
    # Public Scoring & Filtering Interface
    # -----------------------------------------------------------------------

    def score(
        self,
        intent: Union[str, ExtractedTaskIntent],
        units: Sequence[Union[DiscourseChunk, MacroBlock]],
    ) -> List[ScoredUnit]:
        """Score each discourse unit according to the configured strategy."""
        t0 = time.perf_counter()
        query_text = (
            intent.query_text if isinstance(intent, ExtractedTaskIntent) else str(intent)
        )

        if not units:
            self.last_scoring_time_ms = 0.0
            self.last_gpu_calls = 0
            return []

        strat = self.strategy
        if strat in ("PASSTHROUGH", "PASS-THROUGH", "NONE"):
            scored = self._score_passthrough(units)
            self.last_gpu_calls = 0
        elif strat in ("F-A", "LEXICAL", "BM25"):
            scored = self._score_bm25(query_text, units)
            self.last_gpu_calls = 0
        elif strat in ("F-B", "CONCEPT", "CONCEPT_OVERLAP"):
            scored = self._score_concept_overlap(query_text, units)
            self.last_gpu_calls = 0
        elif strat in ("F-C", "KEV_MICRO"):
            scored = self._score_kev_micro(query_text, units)
            self.last_gpu_calls = len(units)
        elif strat in ("F-D", "KEV_MACRO", "MACRO_FULL"):
            scored = self._score_kev_macro(query_text, units, head_only=False)
            self.last_gpu_calls = len(units)
        elif strat in ("F-E", "KEV_HEAD", "MACRO_HEAD"):
            scored = self._score_kev_macro(query_text, units, head_only=True)
            self.last_gpu_calls = len(units)
        elif strat in ("F-F", "CASCADE"):
            scored, gpu_calls = self._score_cascade(query_text, units)
            self.last_gpu_calls = gpu_calls
        else:
            logger.warning("Unknown filter strategy '%s', falling back to passthrough", strat)
            scored = self._score_passthrough(units)
            self.last_gpu_calls = 0

        self.last_scoring_time_ms = (time.perf_counter() - t0) * 1000.0
        return scored

    def apply_keep_policy(
        self,
        scored_units: Sequence[ScoredUnit],
    ) -> Tuple[List[ScoredUnit], List[ScoredUnit]]:
        """Apply configured threshold, top-k, and token budget keep policies.

        Returns:
            Tuple of (kept_units, discarded_units).
        """
        if self.strategy in ("PASSTHROUGH", "PASS-THROUGH", "NONE"):
            return list(scored_units), []

        # If no keep constraints are configured, keep all
        if (
            self.keep_threshold is None
            and self.keep_top_k is None
            and self.keep_budget_tokens is None
        ):
            return list(scored_units), []

        # Sort units by score descending
        sorted_units = sorted(scored_units, key=lambda u: u.score, reverse=True)

        # 1. Filter by threshold if set
        if self.keep_threshold is not None:
            threshold_passed = [u for u in sorted_units if u.score >= self.keep_threshold]
            threshold_rejected = [u for u in sorted_units if u.score < self.keep_threshold]
        else:
            threshold_passed = list(sorted_units)
            threshold_rejected = []

        # Always keep at least 1 unit if units exist, to avoid zero-context failure
        if not threshold_passed and sorted_units:
            threshold_passed = [sorted_units[0]]
            threshold_rejected = sorted_units[1:]

        # 2. Filter by top_k if set
        if self.keep_top_k is not None:
            top_k = max(1, int(self.keep_top_k))
            top_k_kept = threshold_passed[:top_k]
            top_k_rejected = threshold_passed[top_k:] + threshold_rejected
        else:
            top_k_kept = threshold_passed
            top_k_rejected = threshold_rejected

        # 3. Filter by token budget if set
        if self.keep_budget_tokens is not None:
            budget = max(1, int(self.keep_budget_tokens))
            budget_kept: List[ScoredUnit] = []
            budget_rejected: List[ScoredUnit] = list(top_k_rejected)
            current_tokens = 0
            for u in top_k_kept:
                u_tok = max(1, u.tokens)
                if not budget_kept or (current_tokens + u_tok <= budget):
                    budget_kept.append(u)
                    current_tokens += u_tok
                else:
                    budget_rejected.append(u)
            kept_final = budget_kept
            discarded_final = budget_rejected
        else:
            kept_final = top_k_kept
            discarded_final = top_k_rejected

        # Re-sort kept units into original document order
        unit_id_order = {u.unit_id: idx for idx, u in enumerate(scored_units)}
        kept_final_sorted = sorted(kept_final, key=lambda u: unit_id_order.get(u.unit_id, 0))

        return kept_final_sorted, discarded_final

    def filter(
        self,
        intent: Union[str, ExtractedTaskIntent],
        units: Sequence[Union[DiscourseChunk, MacroBlock]],
    ) -> List[ScoredUnit]:
        """Convenience end-to-end method: score and apply keep policies."""
        scored = self.score(intent, units)
        kept, _ = self.apply_keep_policy(scored)
        return kept

    # -----------------------------------------------------------------------
    # Variant Implementations
    # -----------------------------------------------------------------------

    def _score_passthrough(
        self,
        units: Sequence[Union[DiscourseChunk, MacroBlock]],
    ) -> List[ScoredUnit]:
        """Pass-through: score every unit at 1.0 (B0 behavior)."""
        res: List[ScoredUnit] = []
        for u in units:
            u_id, u_toks, span, gran = self._extract_unit_meta(u)
            res.append(
                ScoredUnit(
                    unit=u,
                    unit_id=u_id,
                    score=1.0,
                    raw_score=1.0,
                    raw_logprob=0.0,
                    tokens=u_toks,
                    char_span=span,
                    granularity=gran,
                    metadata={"strategy": "passthrough"},
                )
            )
        return res

    def _score_bm25(
        self,
        query: str,
        units: Sequence[Union[DiscourseChunk, MacroBlock]],
    ) -> List[ScoredUnit]:
        """Variant F-A: Okapi BM25 scoring over tokenized vocabulary."""
        q_terms = [t.lower() for t in re.findall(r"\b[A-Za-z0-9_]{2,}\b", query)]
        if not q_terms:
            return self._score_passthrough(units)

        # Tokenize corpus units
        doc_tokens_list: List[List[str]] = []
        doc_len_list: List[int] = []
        for u in units:
            u_text = u.text if hasattr(u, "text") else ""
            tokens = [t.lower() for t in re.findall(r"\b[A-Za-z0-9_]{2,}\b", u_text)]
            doc_tokens_list.append(tokens)
            doc_len_list.append(len(tokens))

        N = len(units)
        avgdl = (sum(doc_len_list) / N) if N > 0 else 1.0
        k1 = 1.5
        b = 0.75

        # Document frequencies for query terms
        df: Dict[str, int] = {}
        for q in q_terms:
            df[q] = sum(1 for doc in doc_tokens_list if q in doc)

        # IDF calculation with standard floor
        idf: Dict[str, float] = {}
        for q, count in df.items():
            idf[q] = math.log((N - count + 0.5) / (count + 0.5) + 1.0)

        raw_scores: List[float] = []
        for doc, doc_len in zip(doc_tokens_list, doc_len_list):
            doc_tf: Dict[str, int] = {}
            for t in doc:
                doc_tf[t] = doc_tf.get(t, 0) + 1

            s = 0.0
            denom_len = k1 * (1.0 - b + b * (doc_len / max(1.0, avgdl)))
            for q in q_terms:
                freq = doc_tf.get(q, 0)
                if freq > 0:
                    s += idf.get(q, 0.0) * ((freq * (k1 + 1.0)) / (freq + denom_len))
            raw_scores.append(s)

        # Calibrate / normalize scores into [0.0, 1.0]
        max_s = max(raw_scores) if raw_scores else 1.0
        min_s = min(raw_scores) if raw_scores else 0.0
        denom = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0

        res: List[ScoredUnit] = []
        for u, r_score in zip(units, raw_scores):
            u_id, u_toks, span, gran = self._extract_unit_meta(u)
            norm_score = (r_score - min_s) / denom
            cal_score = self._apply_calibration(norm_score, raw_score=r_score)
            res.append(
                ScoredUnit(
                    unit=u,
                    unit_id=u_id,
                    score=cal_score,
                    raw_score=r_score,
                    raw_logprob=math.log(max(1e-5, cal_score)),
                    tokens=u_toks,
                    char_span=span,
                    granularity=gran,
                    metadata={"strategy": "F-A_BM25"},
                )
            )
        return res

    def _score_concept_overlap(
        self,
        query: str,
        units: Sequence[Union[DiscourseChunk, MacroBlock]],
    ) -> List[ScoredUnit]:
        """Variant F-B: Concept-code set overlap via MmapLexicalGrounder."""
        # Ground query terms into concept codes
        q_words = re.findall(r"\b[A-Za-z]{3,}\b", query)
        q_codes: Set[int] = set()
        if self.grounder and self.grounder.is_available():
            for w in q_words:
                c = self.grounder.resolve_concept_code(w)
                if c is not None:
                    q_codes.add(c)

        res: List[ScoredUnit] = []
        for u in units:
            u_id, u_toks, span, gran = self._extract_unit_meta(u)
            # Retrieve or compute unit concept codes
            if isinstance(u, MacroBlock) and u.concept_codes:
                u_codes = set(u.concept_codes)
            else:
                u_words = re.findall(r"\b[A-Za-z]{3,}\b", u.text)
                u_codes = set()
                if self.grounder and self.grounder.is_available():
                    for w in u_words:
                        c = self.grounder.resolve_concept_code(w)
                        if c is not None:
                            u_codes.add(c)

            if q_codes and u_codes:
                overlap = len(q_codes & u_codes)
                raw_overlap = overlap / len(q_codes)
            else:
                # Fallback to lexical token overlap if no concept codes resolved
                q_toks = {w.lower() for w in q_words}
                u_toks_set = {w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", u.text)}
                overlap = len(q_toks & u_toks_set)
                raw_overlap = (overlap / max(1, len(q_toks))) * 0.8

            cal_score = self._apply_calibration(raw_overlap, raw_score=raw_overlap)
            res.append(
                ScoredUnit(
                    unit=u,
                    unit_id=u_id,
                    score=cal_score,
                    raw_score=raw_overlap,
                    raw_logprob=math.log(max(1e-5, cal_score)),
                    tokens=u_toks,
                    char_span=span,
                    granularity=gran,
                    metadata={"strategy": "F-B_CONCEPT", "matched_codes": len(q_codes & u_codes) if q_codes else 0},
                )
            )
        return res

    def _score_kev_micro(
        self,
        query: str,
        units: Sequence[Union[DiscourseChunk, MacroBlock]],
    ) -> List[ScoredUnit]:
        """Variant F-C: Micro chunk Kev-4B single-token letter logprobs."""
        texts = [u.text for u in units]
        # Query Kev-4B batch across parallel decode slots
        kev_results = self.kev_engine.score_relevance_batch(
            intent=query,
            texts=texts,
        )

        res: List[ScoredUnit] = []
        for u, k_res in zip(units, kev_results):
            u_id, u_toks, span, gran = self._extract_unit_meta(u)
            raw_prob = k_res.get("RELEVANT", 0.08)
            raw_lp = k_res.raw_logprobs.get("A", math.log(max(1e-5, raw_prob)))
            cal_score = self._apply_calibration(raw_prob, raw_score=raw_lp)
            res.append(
                ScoredUnit(
                    unit=u,
                    unit_id=u_id,
                    score=cal_score,
                    raw_score=raw_lp,
                    raw_logprob=raw_lp,
                    tokens=u_toks,
                    char_span=span,
                    granularity=gran,
                    metadata={"strategy": "F-C_KEV_MICRO", "top_label": k_res.top_label},
                )
            )
        return res

    def _score_kev_macro(
        self,
        query: str,
        units: Sequence[Union[DiscourseChunk, MacroBlock]],
        head_only: bool = False,
    ) -> List[ScoredUnit]:
        """Variant F-D (full text) or F-E (head only) over MacroBlocks."""
        texts: List[str] = []
        for u in units:
            full_t = u.text
            if head_only:
                # F-E: extract first 2 sentences or first 60 words
                sents = re.split(r"(?<=[.!?])\s+", full_t)
                head_text = " ".join(sents[:2])
                if len(head_text.split()) > 80:
                    head_text = " ".join(head_text.split()[:80])
                texts.append(head_text)
            else:
                texts.append(full_t)

        kev_results = self.kev_engine.score_relevance_batch(
            intent=query,
            texts=texts,
        )

        strat_name = "F-E_KEV_HEAD" if head_only else "F-D_KEV_MACRO"
        res: List[ScoredUnit] = []
        for u, k_res in zip(units, kev_results):
            u_id, u_toks, span, gran = self._extract_unit_meta(u)
            raw_prob = k_res.get("RELEVANT", 0.08)
            raw_lp = k_res.raw_logprobs.get("A", math.log(max(1e-5, raw_prob)))
            cal_score = self._apply_calibration(raw_prob, raw_score=raw_lp)
            res.append(
                ScoredUnit(
                    unit=u,
                    unit_id=u_id,
                    score=cal_score,
                    raw_score=raw_lp,
                    raw_logprob=raw_lp,
                    tokens=u_toks,
                    char_span=span,
                    granularity=gran,
                    metadata={"strategy": strat_name, "top_label": k_res.top_label, "head_only": head_only},
                )
            )
        return res

    def _score_cascade(
        self,
        query: str,
        units: Sequence[Union[DiscourseChunk, MacroBlock]],
    ) -> Tuple[List[ScoredUnit], int]:
        """Variant F-F: Stage 1 BM25 pre-filter followed by Stage 2 Kev-4B re-ranking."""
        stage1 = self._score_bm25(query, units)
        # Stage 1 prefilter: keep top 50% or top 8 units
        sorted_s1 = sorted(stage1, key=lambda u: u.score, reverse=True)
        keep_n = max(2, min(len(units), max(len(units) // 2, 6)))
        prefiltered_candidates = sorted_s1[:keep_n]
        unselected = sorted_s1[keep_n:]

        # Stage 2: Kev scoring on candidate units
        cand_units = [c.unit for c in prefiltered_candidates]
        stage2_scored = self._score_kev_micro(query, cand_units)

        # Merge results: candidate units receive Kev score; rejected receive their low BM25 score scaled down
        res: List[ScoredUnit] = list(stage2_scored)
        for u_s1 in unselected:
            res.append(
                ScoredUnit(
                    unit=u_s1.unit,
                    unit_id=u_s1.unit_id,
                    score=u_s1.score * 0.1,  # penalized for missing stage 1
                    raw_score=u_s1.raw_score,
                    raw_logprob=u_s1.raw_logprob,
                    tokens=u_s1.tokens,
                    char_span=u_s1.char_span,
                    granularity=u_s1.granularity,
                    metadata={"strategy": "F-F_CASCADE_STAGE1_REJECT"},
                )
            )

        # Restore original order
        id_order = {u.chunk_id if hasattr(u, "chunk_id") else u.macro_id: idx for idx, u in enumerate(units)}
        res_sorted = sorted(res, key=lambda s: id_order.get(s.unit_id, 0))
        return res_sorted, len(cand_units)

    # -----------------------------------------------------------------------
    # Helper & Calibration Methods
    # -----------------------------------------------------------------------

    def _apply_calibration(self, prob: float, raw_score: float) -> float:
        """Apply Platt or Temperature scaling if configured in profile."""
        if not self.calibration or not isinstance(self.calibration, dict):
            return max(0.0, min(1.0, prob))

        method = str(self.calibration.get("method", "")).lower()
        if method == "platt":
            a = float(self.calibration.get("a", 1.0))
            b = float(self.calibration.get("b", 0.0))
            # Logit transform
            p_clamped = max(1e-4, min(1.0 - 1e-4, prob))
            logit = math.log(p_clamped / (1.0 - p_clamped))
            calibrated_logit = a * logit + b
            return 1.0 / (1.0 + math.exp(-calibrated_logit))
        elif method == "temperature":
            temp = float(self.calibration.get("temperature", 1.0))
            p_clamped = max(1e-4, min(1.0 - 1e-4, prob))
            logit = math.log(p_clamped / (1.0 - p_clamped))
            calibrated_logit = logit / max(0.01, temp)
            return 1.0 / (1.0 + math.exp(-calibrated_logit))

        return max(0.0, min(1.0, prob))

    @staticmethod
    def _extract_unit_meta(
        unit: Union[DiscourseChunk, MacroBlock],
    ) -> Tuple[str, int, Tuple[int, int], str]:
        """Extract ID, token estimate, span, and granularity from unit."""
        if isinstance(unit, MacroBlock):
            return (
                unit.macro_id,
                unit.token_count_estimate or int(unit.word_count * 1.33),
                unit.char_span,
                "macro",
            )
        elif isinstance(unit, DiscourseChunk):
            return (
                unit.chunk_id,
                unit.token_count_estimate or int(unit.word_count * 1.33),
                (unit.global_offset, unit.global_end_offset),
                "micro",
            )
        else:
            u_text = getattr(unit, "text", "")
            return (
                getattr(unit, "chunk_id", getattr(unit, "macro_id", "unit_0000")),
                len(u_text.split()),
                (0, len(u_text)),
                "micro",
            )
