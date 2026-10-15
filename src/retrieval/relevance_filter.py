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
- F-G: dense hybrid (BM25 + EmbeddingGemma 2 dense cosine similarities via RRF)
- passthrough: default B0 behavior (keeps 100% of units)

Supports keep policies (threshold, top-k, token budget) and calibration (temperature/Platt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from config.multi_scale_config import MultiScaleConfig
from models.kev_engine import KevDecisionEngine, MockKevEngine, RelevanceScoreResult
from parser.chunker import DiscourseChunk
from parser.mmap_grounder import MmapLexicalGrounder
from parser.multi_scale_chunker import MacroBlock
from parser.task_boundary_extractor import (
    ExtractedTaskIntent,
    QUESTION_STOPWORDS,
    clean_core_query,
    extract_target_entities,
)

logger = logging.getLogger("quanta.retrieval.relevance_filter")


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
        embedding_manager: Optional[Any] = None,
        rrf_k: Optional[int] = None,
        weight_lexical: Optional[float] = None,
        weight_dense: Optional[float] = None,
        dense_fast_path_threshold: Optional[float] = None,
        matryoshka_dim: Optional[int] = None,
    ):
        cfg = config or MultiScaleConfig()
        self.config = cfg
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

        # ANVC Hybrid Dense Bi-Encoder Parameters (§Session 2)
        self.embedding_manager = embedding_manager
        self.rrf_k = int(rrf_k if rrf_k is not None else cfg.filter_rrf_k)
        self.weight_lexical = float(weight_lexical if weight_lexical is not None else cfg.filter_weight_lexical)
        self.weight_dense = float(weight_dense if weight_dense is not None else cfg.filter_weight_dense)
        self.dense_fast_path_threshold = (
            dense_fast_path_threshold
            if dense_fast_path_threshold is not None
            else cfg.filter_dense_fast_path_threshold
        )
        self.matryoshka_dim = int(matryoshka_dim if matryoshka_dim is not None else cfg.embedding_matryoshka_dim)
        self.last_fast_path_triggered: bool = False

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
        if isinstance(intent, ExtractedTaskIntent):
            query_text = intent.query_text
            target_entities = list(intent.target_entities)
        else:
            query_text = str(intent)
            target_entities = extract_target_entities(query_text)

        if not units:
            self.last_scoring_time_ms = 0.0
            self.last_gpu_calls = 0
            return []

        strat = self.strategy
        if strat in ("PASSTHROUGH", "PASS-THROUGH", "NONE"):
            scored = self._score_passthrough(units)
            self.last_gpu_calls = 0
        elif strat in ("F-A", "LEXICAL", "BM25"):
            scored = self._score_bm25(query_text, units, target_entities=target_entities)
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
            scored, gpu_calls = self._score_cascade(query_text, units, target_entities=target_entities)
            self.last_gpu_calls = gpu_calls
        elif strat in ("F-G", "DENSE_HYBRID", "HYBRID", "RRF"):
            scored, gpu_calls = self._score_dense_hybrid(query_text, units, target_entities=target_entities)
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

        # Sort units by score descending, strictly prioritizing exact identifier hits
        sorted_units = sorted(
            scored_units,
            key=lambda u: (bool(u.metadata.get("has_exact_identifier", False)), u.score),
            reverse=True,
        )

        # 1. Filter by threshold if set (exact identifier matches always pass)
        if self.keep_threshold is not None:
            threshold_passed = [
                u for u in sorted_units
                if u.score >= self.keep_threshold or bool(u.metadata.get("has_exact_identifier", False))
            ]
            threshold_rejected = [
                u for u in sorted_units
                if u.score < self.keep_threshold and not bool(u.metadata.get("has_exact_identifier", False))
            ]
        else:
            threshold_passed = list(sorted_units)
            threshold_rejected = []

        # Always keep at least 1 unit if units exist, to avoid zero-context failure
        if not threshold_passed and sorted_units:
            threshold_passed = [sorted_units[0]]
            threshold_rejected = sorted_units[1:]

        # 2. Filter by top_k if set (exact identifier matches are never dropped by top_k)
        if self.keep_top_k is not None:
            top_k = max(1, int(self.keep_top_k))
            exact_id_units = [u for u in threshold_passed if u.metadata.get("has_exact_identifier", False)]
            non_exact_units = [u for u in threshold_passed if not u.metadata.get("has_exact_identifier", False)]
            slots_remaining = max(0, top_k - len(exact_id_units))
            top_k_kept = exact_id_units + non_exact_units[:slots_remaining]
            top_k_rejected = non_exact_units[slots_remaining:] + threshold_rejected
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
                is_exact_id = bool(u.metadata.get("has_exact_identifier", False))
                # Elastic expansion up to 3500 tokens for units matching exact technical identifiers
                effective_budget = max(budget, 3500) if is_exact_id else budget
                if not budget_kept or (current_tokens + u_tok <= effective_budget):
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
        target_entities: Optional[Sequence[str]] = None,
    ) -> List[ScoredUnit]:
        """Variant F-A+: Okapi BM25 scoring with stopword filtering & entity/identifier boosting."""
        clean_q = clean_core_query(query) if query else ""
        raw_tokens = [t.lower() for t in re.findall(r"\b[A-Za-z0-9_]{2,}\b", clean_q or query)]
        if not raw_tokens:
            return self._score_passthrough(units)

        # Exclude QUESTION_STOPWORDS unless no content terms remain
        content_tokens = [t for t in raw_tokens if t not in QUESTION_STOPWORDS]
        q_terms = content_tokens if content_tokens else raw_tokens

        # Collect target entities and technical identifiers from query & target_entities
        all_entities: List[str] = list(target_entities or [])
        # Extract technical alphanumeric variable IDs directly from query (e.g. ACC-9042, SRV-ALPHA, Vault 81)
        for code in re.findall(r"\b[A-Za-z0-9]*\d+[A-Za-z0-9_-]*\b|\b[A-Z0-9]{2,}(?:[-_][A-Z0-9]+)+\b|\bVault\s+\d+\b", f"{clean_q} {query}"):
            if code not in all_entities:
                all_entities.append(code)

        tech_entities: List[str] = []
        multi_word_entities: List[str] = []
        single_word_entities: List[str] = []

        for ent in all_entities:
            ent_clean = ent.strip()
            if not ent_clean or ent_clean.lower() in QUESTION_STOPWORDS:
                continue
            if re.search(r"\b[A-Za-z0-9]*\d+[A-Za-z0-9_-]*\b|\b[A-Z0-9]{2,}(?:[-_][A-Z0-9]+)+|\bVault\s+\d+", ent_clean):
                tech_entities.append(ent_clean)
            elif " " in ent_clean:
                multi_word_entities.append(ent_clean)
            elif len(ent_clean) > 2:
                single_word_entities.append(ent_clean)

        # Tokenize corpus units
        doc_tokens_list: List[List[str]] = []
        doc_len_list: List[int] = []
        doc_texts: List[str] = []
        for u in units:
            u_text = u.text if hasattr(u, "text") else ""
            doc_texts.append(u_text)
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
        unit_exact_id_flags: List[bool] = []
        unit_matched_entities: List[List[str]] = []

        for doc, doc_len, u_text in zip(doc_tokens_list, doc_len_list, doc_texts):
            doc_tf: Dict[str, int] = {}
            for t in doc:
                doc_tf[t] = doc_tf.get(t, 0) + 1

            s = 0.0
            denom_len = k1 * (1.0 - b + b * (doc_len / max(1.0, avgdl)))
            for q in q_terms:
                freq = doc_tf.get(q, 0)
                if freq > 0:
                    s += idf.get(q, 0.0) * ((freq * (k1 + 1.0)) / (freq + denom_len))

            # Compute exact-match bonuses for entities
            entity_boost = 0.0
            has_exact_id = False
            matched: List[str] = []

            # 1. Technical alphanumeric identifiers (+8.0 boost)
            for te in tech_entities:
                if re.search(r"\b" + re.escape(te) + r"\b", u_text, re.IGNORECASE):
                    entity_boost += 8.0
                    has_exact_id = True
                    matched.append(te)

            # 2. Multi-word proper names (+6.0 boost)
            for mwe in multi_word_entities:
                if re.search(r"\b" + re.escape(mwe) + r"\b", u_text, re.IGNORECASE):
                    entity_boost += 6.0
                    matched.append(mwe)

            # 3. Standalone proper nouns / salient query entities (+4.0 boost)
            for swe in single_word_entities:
                if re.search(r"\b" + re.escape(swe) + r"\b", u_text, re.IGNORECASE):
                    entity_boost += 4.0
                    matched.append(swe)

            # Multiplicative boost for units with entity matches
            if entity_boost > 0.0:
                mult_factor = 2.0 if has_exact_id else 1.5
                boosted_s = (s + entity_boost) * mult_factor
            else:
                boosted_s = s

            raw_scores.append(boosted_s)
            unit_exact_id_flags.append(has_exact_id)
            unit_matched_entities.append(matched)

        # Calibrate / normalize scores into [0.0, 1.0]
        max_s = max(raw_scores) if raw_scores else 1.0
        min_s = min(raw_scores) if raw_scores else 0.0
        denom = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0

        res: List[ScoredUnit] = []
        for u, r_score, exact_id, matched in zip(units, raw_scores, unit_exact_id_flags, unit_matched_entities):
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
                    metadata={
                        "strategy": "F-A_BM25",
                        "has_exact_identifier": exact_id,
                        "matched_entities": matched,
                    },
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
        target_entities: Optional[Sequence[str]] = None,
    ) -> Tuple[List[ScoredUnit], int]:
        """Variant F-F: Stage 1 BM25 pre-filter followed by Stage 2 Kev-4B re-ranking."""
        stage1 = self._score_bm25(query, units, target_entities=target_entities)
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

    def _score_dense_hybrid(
        self,
        query: str,
        units: Sequence[Union[DiscourseChunk, MacroBlock]],
        target_entities: Optional[Sequence[str]] = None,
    ) -> Tuple[List[ScoredUnit], int]:
        """Variant F-G: Dense Hybrid combining BM25 lexical with EmbeddingGemma 2 bi-encoder via RRF.

        Formulation:
            RRF(d) = (w_lex / (k_rrf + rank_BM25(d))) + (w_dense / (k_rrf + rank_dense(d)))
        Fast-Path Dynamic Coupling:
            If max_d s_dense(d) >= theta_fast, bypass BM25 and immediately route top unit to fast path.
        """
        if self.embedding_manager is None:
            try:
                from server.embedding_manager import EmbeddingManager
                self.embedding_manager = EmbeddingManager.get_default(config=self.config)
            except Exception as e:
                logger.warning("Could not instantiate EmbeddingManager (%s); using mock fallback", e)
                from server.embedding_manager import MockEmbeddingManager
                self.embedding_manager = MockEmbeddingManager(matryoshka_dim=self.matryoshka_dim)

        mgr = self.embedding_manager
        dim = self.matryoshka_dim

        # 1. Embed query
        q_vec = mgr.embed_query(query, matryoshka_dim=dim)
        if hasattr(q_vec, "ndim") and q_vec.ndim > 1:
            q_vec = q_vec.squeeze()
        q_vec = np.asarray(q_vec, dtype=np.float32)

        # 2. Check / compute dense embeddings for all candidate units
        unembedded_indices = []
        for i, u in enumerate(units):
            emb = getattr(u, "dense_embedding", None)
            if emb is None:
                unembedded_indices.append(i)
            else:
                emb_arr = np.asarray(emb, dtype=np.float32)
                if emb_arr.shape[-1] != dim:
                    from server.embedding_manager import truncate_and_normalize
                    emb_arr = truncate_and_normalize(emb_arr, dim)
                    u.dense_embedding = emb_arr

        gpu_calls = 1  # 1 call for query
        if unembedded_indices:
            unembedded_texts = [units[i].text for i in unembedded_indices]
            batch_vecs = mgr.embed_documents(unembedded_texts, matryoshka_dim=dim)
            for i, vec in zip(unembedded_indices, batch_vecs):
                units[i].dense_embedding = vec
            gpu_calls += len(unembedded_indices)

        # Stack unit vectors into matrix U: shape (N, D)
        unit_vecs = np.stack([np.asarray(u.dense_embedding, dtype=np.float32) for u in units], axis=0)
        # Cosine similarity via dot-product (both q_vec and unit_vecs are L2-normalized)
        dense_scores = np.dot(unit_vecs, q_vec).astype(float)
        dense_scores = np.clip(dense_scores, -1.0, 1.0)

        # 3. Fast-Path Dynamic Coupling Check
        max_dense = float(np.max(dense_scores)) if len(dense_scores) > 0 else 0.0
        theta_fast = self.dense_fast_path_threshold
        fast_path_triggered = (theta_fast is not None and max_dense >= theta_fast)
        self.last_fast_path_triggered = fast_path_triggered

        # 4. Standard Reciprocal Rank Fusion (RRF)
        bm25_scored = self._score_bm25(query, units, target_entities=target_entities)
        bm25_raw = [s.raw_score for s in bm25_scored]
        any_exact_ids = any(s.metadata.get("has_exact_identifier", False) for s in bm25_scored)

        if fast_path_triggered and not any_exact_ids:
            # Fast-path bypass: directly rank by dense cosine similarity without BM25 only when no specific entity/code requested
            min_dense = float(np.min(dense_scores))
            denom = (max_dense - min_dense) if (max_dense - min_dense) > 1e-6 else 1.0

            res: List[ScoredUnit] = []
            for idx, u in enumerate(units):
                u_id, u_toks, span, gran = self._extract_unit_meta(u)
                d_score = float(dense_scores[idx])
                norm_score = max(0.0, min(1.0, (d_score - min_dense) / denom))
                cal_score = self._apply_calibration(norm_score, raw_score=d_score)
                dense_rank = int(np.sum(dense_scores > d_score)) + 1
                res.append(
                    ScoredUnit(
                        unit=u,
                        unit_id=u_id,
                        score=cal_score,
                        raw_score=d_score,
                        raw_logprob=math.log(max(1e-5, cal_score)),
                        tokens=u_toks,
                        char_span=span,
                        granularity=gran,
                        metadata={
                            "strategy": "F-G_DENSE_HYBRID",
                            "dense_score": d_score,
                            "dense_rank": dense_rank,
                            "fast_path_bypassed": True,
                            "dense_fast_path_triggered": True,
                            "has_exact_identifier": False,
                            "matched_entities": [],
                        },
                    )
                )
            return res, gpu_calls

        # 1-based ranks
        bm25_order = sorted(range(len(units)), key=lambda i: bm25_raw[i], reverse=True)
        bm25_ranks = {orig_idx: rank + 1 for rank, orig_idx in enumerate(bm25_order)}

        dense_order = sorted(range(len(units)), key=lambda i: dense_scores[i], reverse=True)
        dense_ranks = {orig_idx: rank + 1 for rank, orig_idx in enumerate(dense_order)}

        k_rrf = self.rrf_k
        w_lex = self.weight_lexical
        w_dense = self.weight_dense

        rrf_scores: List[float] = []
        for i in range(len(units)):
            r_lex = bm25_ranks[i]
            r_dns = dense_ranks[i]
            val = (w_lex / (k_rrf + r_lex)) + (w_dense / (k_rrf + r_dns))
            has_id = bool(bm25_scored[i].metadata.get("has_exact_identifier", False))
            if has_id:
                val += 1.0  # Dominant boost ensuring exact identifier/entity matches outrank distractor noise
            rrf_scores.append(val)

        min_rrf = min(rrf_scores) if rrf_scores else 0.0
        max_rrf = max(rrf_scores) if rrf_scores else 1.0
        denom = (max_rrf - min_rrf) if (max_rrf - min_rrf) > 1e-6 else 1.0

        res: List[ScoredUnit] = []
        for i, u in enumerate(units):
            u_id, u_toks, span, gran = self._extract_unit_meta(u)
            r_val = rrf_scores[i]
            norm_score = max(0.0, min(1.0, (r_val - min_rrf) / denom))
            cal_score = self._apply_calibration(norm_score, raw_score=r_val)
            res.append(
                ScoredUnit(
                    unit=u,
                    unit_id=u_id,
                    score=cal_score,
                    raw_score=r_val,
                    raw_logprob=math.log(max(1e-5, cal_score)),
                    tokens=u_toks,
                    char_span=span,
                    granularity=gran,
                    metadata={
                        "strategy": "F-G_DENSE_HYBRID",
                        "rrf_score": r_val,
                        "bm25_score": bm25_raw[i],
                        "dense_score": float(dense_scores[i]),
                        "bm25_rank": bm25_ranks[i],
                        "dense_rank": dense_ranks[i],
                        "fast_path_triggered": False,
                        "has_exact_identifier": bool(bm25_scored[i].metadata.get("has_exact_identifier", False)),
                        "matched_entities": bm25_scored[i].metadata.get("matched_entities", []),
                        "w_lex": w_lex,
                        "w_dense": w_dense,
                        "k_rrf": k_rrf,
                    },
                )
            )

        return res, gpu_calls

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
