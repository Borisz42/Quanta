"""High-performance character and byte-level span aligner for QUANTA SVM.

Maps extracted surface entity/event tokens back to exact character offsets
and byte offsets in the source text chunk. Resolves token-boundary shifts,
whitespace variants, morphology affixes, and punctuation trimming.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass(slots=True)
class SpanAlignment:
    """Represents a grounded character and byte span in source text.
    
    Attributes:
        char_span: 0-indexed half-open character range (start, end) in source string.
        byte_span: 0-indexed half-open byte range (start, end) in UTF-8 encoded source bytes.
        matched_text: The exact substring in source_text[char_span[0]:char_span[1]].
        confidence: Alignment confidence score in [0.0, 1.0].
        is_exact: True if the surface token matched identically without fuzzy matching.
    """

    char_span: Tuple[int, int]
    byte_span: Tuple[int, int]
    matched_text: str
    confidence: float = 1.0
    is_exact: bool = True

    @property
    def start(self) -> int:
        return self.char_span[0]

    @property
    def end(self) -> int:
        return self.char_span[1]

    @property
    def byte_start(self) -> int:
        return self.byte_span[0]

    @property
    def byte_end(self) -> int:
        return self.byte_span[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "char_span": list(self.char_span),
            "byte_span": list(self.byte_span),
            "matched_text": self.matched_text,
            "confidence": self.confidence,
            "is_exact": self.is_exact,
        }


class SpanAligner:
    """Robust, fast span aligner resolving surface tokens to source text offsets."""

    PUNCT_TRIM = "\"'`.,;:()[]{}<>«»„“\t\n\r"

    def __init__(self, fuzzy_threshold: float = 0.70):
        self.fuzzy_threshold = fuzzy_threshold

    @staticmethod
    def char_to_byte_span(source_text: str, char_span: Tuple[int, int]) -> Tuple[int, int]:
        """Convert character offsets [start, end) into UTF-8 byte offsets [b_start, b_end)."""
        c_start, c_end = char_span
        c_start = max(0, min(c_start, len(source_text)))
        c_end = max(c_start, min(c_end, len(source_text)))

        prefix_bytes = len(source_text[:c_start].encode("utf-8"))
        span_bytes = len(source_text[c_start:c_end].encode("utf-8"))
        return (prefix_bytes, prefix_bytes + span_bytes)

    @staticmethod
    def byte_to_char_span(source_text: str, byte_span: Tuple[int, int]) -> Tuple[int, int]:
        """Convert UTF-8 byte offsets into character offsets."""
        b_start, b_end = byte_span
        raw_bytes = source_text.encode("utf-8")
        b_start = max(0, min(b_start, len(raw_bytes)))
        b_end = max(b_start, min(b_end, len(raw_bytes)))

        prefix_chars = len(raw_bytes[:b_start].decode("utf-8", errors="ignore"))
        span_chars = len(raw_bytes[b_start:b_end].decode("utf-8", errors="ignore"))
        return (prefix_chars, prefix_chars + span_chars)

    def align(
        self,
        source_text: str,
        surface_token: str,
        start_hint: Optional[int] = None,
    ) -> Optional[SpanAlignment]:
        """Align a surface token back to source text.
        
        Args:
            source_text: The complete raw passage string.
            surface_token: The extracted surface word or phrase.
            start_hint: Optional starting character index to prioritize closest occurrence.
            
        Returns:
            SpanAlignment or None if no acceptable match is found.
        """
        if not source_text or not surface_token:
            return None

        # 1. Punctuation / quote trimming
        clean_token = surface_token.strip(self.PUNCT_TRIM)
        start_pos = 0 if start_hint is None else max(0, min(start_hint, len(source_text)))

        # If token was wrapped in quotes/punctuation, prioritize the clean stripped phrase
        if clean_token and clean_token != surface_token:
            idx = source_text.find(clean_token, start_pos)
            if idx == -1 and start_pos > 0:
                idx = source_text.find(clean_token)
            if idx != -1:
                c_span = (idx, idx + len(clean_token))
                b_span = self.char_to_byte_span(source_text, c_span)
                return SpanAlignment(
                    char_span=c_span,
                    byte_span=b_span,
                    matched_text=source_text[c_span[0]:c_span[1]],
                    confidence=0.98,
                    is_exact=True,
                )

        # 2. Exact match (case-sensitive)
        idx = source_text.find(surface_token, start_pos)
        if idx == -1 and start_pos > 0:
            idx = source_text.find(surface_token)

        if idx != -1:
            c_span = (idx, idx + len(surface_token))
            b_span = self.char_to_byte_span(source_text, c_span)
            return SpanAlignment(
                char_span=c_span,
                byte_span=b_span,
                matched_text=source_text[c_span[0]:c_span[1]],
                confidence=1.0,
                is_exact=True,
            )

        target_token = clean_token if clean_token else surface_token

        # 3. Case-insensitive exact match
        lower_source = source_text.lower()
        lower_target = target_token.lower()
        idx_lower = lower_source.find(lower_target, start_pos)
        if idx_lower == -1 and start_pos > 0:
            idx_lower = lower_source.find(lower_target)

        if idx_lower != -1:
            c_span = (idx_lower, idx_lower + len(target_token))
            b_span = self.char_to_byte_span(source_text, c_span)
            return SpanAlignment(
                char_span=c_span,
                byte_span=b_span,
                matched_text=source_text[c_span[0]:c_span[1]],
                confidence=0.95,
                is_exact=False,
            )

        # 4. Whitespace normalized match (resolving tabs, newlines, multiple spaces)
        words = target_token.split()
        if len(words) > 1:
            escaped_words = [re.escape(w) for w in words]
            pattern_str = r"\s+".join(escaped_words)
            m = re.search(pattern_str, source_text[start_pos:], flags=re.IGNORECASE)
            if m:
                c_span = (start_pos + m.start(), start_pos + m.end())
            else:
                m2 = re.search(pattern_str, source_text, flags=re.IGNORECASE)
                if m2:
                    c_span = (m2.start(), m2.end())
                else:
                    c_span = None

            if c_span is not None:
                b_span = self.char_to_byte_span(source_text, c_span)
                return SpanAlignment(
                    char_span=c_span,
                    byte_span=b_span,
                    matched_text=source_text[c_span[0]:c_span[1]],
                    confidence=0.92,
                    is_exact=False,
                )

        # 5. Word boundary / stem / affix matching
        # E.g. agglutinative suffixes or plurals (target='cylinder', source='cylinders')
        word_pattern = rf"\b{re.escape(target_token)}[a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ-]*\b"
        m_word = re.search(word_pattern, source_text[start_pos:], flags=re.IGNORECASE)
        if m_word:
            c_span = (start_pos + m_word.start(), start_pos + m_word.end())
        else:
            m_word2 = re.search(word_pattern, source_text, flags=re.IGNORECASE)
            c_span = (m_word2.start(), m_word2.end()) if m_word2 else None

        if c_span is not None:
            b_span = self.char_to_byte_span(source_text, c_span)
            return SpanAlignment(
                char_span=c_span,
                byte_span=b_span,
                matched_text=source_text[c_span[0]:c_span[1]],
                confidence=0.88,
                is_exact=False,
            )

        # 6. Fuzzy window search using SequenceMatcher for slight typos/variations
        best_span: Optional[Tuple[int, int]] = None
        best_ratio = 0.0

        target_len = len(target_token)
        min_win = max(1, int(target_len * 0.8))
        max_win = int(target_len * 1.3) + 2

        # Tokenize source text into candidate token n-grams
        words_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", source_text)]
        for i in range(len(words_with_spans)):
            for j in range(i + 1, min(i + 6, len(words_with_spans) + 1)):
                win_text = source_text[words_with_spans[i][1]:words_with_spans[j - 1][2]]
                win_len = len(win_text)
                if min_win <= win_len <= max_win:
                    ratio = difflib.SequenceMatcher(None, lower_target, win_text.lower()).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_span = (words_with_spans[i][1], words_with_spans[j - 1][2])

        if best_span is not None and best_ratio >= self.fuzzy_threshold:
            b_span = self.char_to_byte_span(source_text, best_span)
            return SpanAlignment(
                char_span=best_span,
                byte_span=b_span,
                matched_text=source_text[best_span[0]:best_span[1]],
                confidence=round(best_ratio, 3),
                is_exact=False,
            )

        return None

    def align_all(
        self,
        source_text: str,
        surface_tokens: Sequence[str],
        sequential: bool = True,
    ) -> List[Optional[SpanAlignment]]:
        """Align multiple surface tokens, optionally advancing search start position monotonically."""
        results: List[Optional[SpanAlignment]] = []
        last_offset = 0

        for token in surface_tokens:
            hint = last_offset if sequential else None
            alignment = self.align(source_text, token, start_hint=hint)
            results.append(alignment)
            if sequential and alignment is not None:
                last_offset = alignment.end

        return results
