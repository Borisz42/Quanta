"""Discourse Chunker and Segmentation Engine for QUANTA.

Partitions long-form documents, book chapters, narratives, and dialogue into
thematic discourse episodes (150-400 words) while strictly preserving sentence
boundaries and recording exact global source offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass(frozen=True)
class SentenceSpan:
    """Character span of an individual sentence within a chunk and source document.
    
    Attributes:
        sentence_idx: 0-indexed position of this sentence within the chunk.
        text: Verbatim string of the sentence.
        start_char: Character start offset relative to the chunk text.
        end_char: Character end offset relative to the chunk text.
        global_start_char: Character start offset relative to the full document.
        global_end_char: Character end offset relative to the full document.
    """
    sentence_idx: int
    text: str
    start_char: int
    end_char: int
    global_start_char: int
    global_end_char: int


@dataclass
class DiscourseChunk:
    """A coherent episode of discourse bounded by sentence/paragraph/chapter boundaries.
    
    Attributes:
        chunk_id: Deterministic identifier (e.g. 'chunk_0000').
        text: Verbatim text segment of the chunk.
        sentence_spans: Spans of all sentences composing this chunk.
        paragraph_index: Index of the primary (or first) paragraph in this chunk.
        paragraph_indices: Indices of all paragraphs spanned by this chunk.
        chapter_id: Identifier of the chapter/section (e.g. 'chapter_1', 'prologue').
        chapter_title: Human-readable title of the chapter/section if detected.
        global_offset: Start character offset in the source document.
        global_end_offset: End character offset in the source document.
        word_count: Number of whitespace-delimited words in the chunk text.
        token_count_estimate: Estimated token count (~1.33x words).
    """
    chunk_id: str
    text: str
    sentence_spans: List[SentenceSpan] = field(default_factory=list)
    paragraph_index: int = 0
    paragraph_indices: List[int] = field(default_factory=list)
    chapter_id: Optional[str] = None
    chapter_title: Optional[str] = None
    global_offset: int = 0
    global_end_offset: int = 0
    word_count: int = 0
    token_count_estimate: int = 0


@dataclass
class _SentenceUnit:
    """Internal sentence representation before chunk grouping."""
    text: str
    global_start: int
    global_end: int
    paragraph_idx: int
    chapter_id: Optional[str]
    chapter_title: Optional[str]
    is_scene_break: bool = False

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class DiscourseChunker:
    """Streaming Discourse Chunker and Boundary Segmentation Engine.
    
    Partitions long-form English documents into coherent episodes (150-400 words)
    suitable for attention-optimal neural transducer ingestion ($O(1)$ physical canvas).
    """

    # Chapter / structural section delimiters
    CHAPTER_PATTERNS = [
        # Markdown headers: # Chapter 1, ## Part II, # Prologue
        re.compile(r"^(#{1,3})\s+(chapter\s+[\dIVXLCDM]+(?::\s*.*)?)$", re.IGNORECASE),
        re.compile(r"^(#{1,3})\s+(book\s+[\dIVXLCDM]+(?::\s*.*)?)$", re.IGNORECASE),
        re.compile(r"^(#{1,3})\s+(act\s+[\dIVXLCDM]+(?::\s*.*)?)$", re.IGNORECASE),
        re.compile(r"^(#{1,3})\s+(part\s+[\dIVXLCDM]+(?::\s*.*)?)$", re.IGNORECASE),
        re.compile(r"^(#{1,3})\s+(prologue|epilogue|interlude|introduction|conclusion|preface)$", re.IGNORECASE),
        # Plain text headings: CHAPTER 1, Chapter I: The Awakening, etc.
        re.compile(r"^(chapter\s+[\dIVXLCDM]+(?::\s*.*)?)$", re.IGNORECASE),
        re.compile(r"^(book\s+[\dIVXLCDM]+(?::\s*.*)?)$", re.IGNORECASE),
        re.compile(r"^(act\s+[\dIVXLCDM]+(?::\s*.*)?)$", re.IGNORECASE),
        re.compile(r"^(part\s+[\dIVXLCDM]+(?::\s*.*)?)$", re.IGNORECASE),
        re.compile(r"^(prologue|epilogue|interlude|introduction|conclusion|preface)$", re.IGNORECASE),
    ]

    # Scene break delimiters: ***, ---, ___, * * *, etc.
    SCENE_BREAK_PATTERN = re.compile(r"^\s*(\*{3,}|-{3,}|_{3,}|(\*\s+){2,}\*|(-\s+){2,}-)\s*$")

    # Common English abbreviations that do not terminate sentences
    ABBREVIATIONS = {
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "i.e", "e.g",
        "ph.d", "m.d", "b.a", "m.a", "st", "ave", "rd", "blvd", "dept", "vol",
        "no", "gen", "gov", "rep", "sen", "lt", "col", "capt", "u.s", "u.k",
    }

    def __init__(
        self,
        min_words: int = 100,
        max_words: int = 400,
        spacy_model: Optional[str] = "en_core_web_sm",
        use_spacy: bool = True,
    ):
        """Initialize the DiscourseChunker.
        
        Args:
            min_words: Minimum target word count for a chunk (default: 100).
            max_words: Maximum target word count for a chunk (default: 400).
            spacy_model: Name of spaCy model to load for sentence boundary detection.
            use_spacy: If True and spaCy is installed, use spaCy for sentence segmentation;
                       otherwise use the built-in rule-based sentence detector.
        """
        if min_words > max_words:
            raise ValueError(f"min_words ({min_words}) cannot exceed max_words ({max_words})")

        self.min_words = min_words
        self.max_words = max_words
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self._nlp = None

        if self.use_spacy:
            try:
                # Fast sentencizer pipeline or full model
                if spacy_model:
                    self._nlp = spacy.load(spacy_model)
                else:
                    self._nlp = spacy.blank("en")
                    self._nlp.add_pipe("sentencizer")
            except Exception:
                # Fallback to blank English sentencizer or rule-based
                try:
                    self._nlp = spacy.blank("en")
                    self._nlp.add_pipe("sentencizer")
                except Exception:
                    self._nlp = None
                    self.use_spacy = False

    def is_chapter_heading(self, line: str) -> Optional[Tuple[str, str]]:
        """Check if a stripped line matches a chapter or structural section boundary.
        
        Returns:
            Tuple of (chapter_id, chapter_title) if matched, else None.
        """
        stripped = line.strip()
        if not stripped:
            return None

        for pattern in self.CHAPTER_PATTERNS:
            m = pattern.match(stripped)
            if m:
                # Format a canonical chapter_id from the heading
                heading_text = m.group(2) if len(m.groups()) >= 2 and m.group(2) else m.group(1)
                clean_title = heading_text.strip()
                # Create slug-like ID
                slug = re.sub(r"[^\w\s-]", "", clean_title.lower())
                chapter_id = re.sub(r"[\s-]+", "_", slug).strip("_")
                return chapter_id, clean_title

        return None

    def is_scene_break(self, line: str) -> bool:
        """Check if a line represents a scene break delimiter (e.g. '***' or '---')."""
        return bool(self.SCENE_BREAK_PATTERN.match(line.strip()))

    def split_into_sentences(self, text: str, base_offset: int = 0) -> List[Tuple[str, int, int]]:
        """Split a paragraph or block of text into sentences with document-relative offsets.
        
        Args:
            text: Text to segment.
            base_offset: Starting character offset of `text` in the global document.
            
        Returns:
            List of (sentence_text, global_start_char, global_end_char).
        """
        if not text or not text.strip():
            return []

        if self.use_spacy and self._nlp is not None:
            return self._split_sentences_spacy(text, base_offset)
        else:
            return self._split_sentences_rule_based(text, base_offset)

    def _split_sentences_spacy(self, text: str, base_offset: int) -> List[Tuple[str, int, int]]:
        """Sentence segmentation via spaCy."""
        doc = self._nlp(text)
        results = []
        for sent in doc.sents:
            sent_str = sent.text
            if not sent_str.strip():
                continue
            # Find exact character span relative to text
            start = sent.start_char
            end = sent.end_char
            # Strip trailing/leading spaces while maintaining precise offsets
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                clean_sent = text[start:end]
                results.append((clean_sent, base_offset + start, base_offset + end))
        return results

    def _split_sentences_rule_based(self, text: str, base_offset: int) -> List[Tuple[str, int, int]]:
        """Rule-based sentence segmentation with abbreviation and quote preservation."""
        results = []
        # Match terminal punctuation followed by whitespace, quote + whitespace, or end of string
        pattern = re.compile(r'([.?!]+[\'")\]}]*)(?=\s+[A-Z"“\'\—]|\s*$)')
        start = 0
        text_len = len(text)

        while start < text_len:
            # Skip leading whitespace
            while start < text_len and text[start].isspace():
                start += 1
            if start >= text_len:
                break

            m = pattern.search(text, start)
            if not m:
                # Remainder is the last sentence
                end = text_len
                while end > start and text[end - 1].isspace():
                    end -= 1
                if start < end:
                    results.append((text[start:end], base_offset + start, base_offset + end))
                break

            punct_end = m.end(1)
            candidate_end = punct_end
            candidate_text = text[start:candidate_end]

            # Check if this period is an abbreviation
            period_idx = candidate_text.rfind(".")
            if period_idx != -1:
                # Extract preceding token
                token_match = re.search(r'([A-Za-z]+)\.$', candidate_text[:period_idx + 1])
                if token_match:
                    token = token_match.group(1).lower()
                    if token in self.ABBREVIATIONS:
                        # Skip this match and continue search from punct_end
                        m = pattern.search(text, punct_end)
                        if m:
                            candidate_end = m.end(1)
                        else:
                            candidate_end = text_len

            # Clean trailing whitespace
            end = candidate_end
            while end > start and text[end - 1].isspace():
                end -= 1

            if start < end:
                results.append((text[start:end], base_offset + start, base_offset + end))
            start = candidate_end

        return results

    def _parse_source_units(self, text: str) -> List[_SentenceUnit]:
        """Scan full text, identify chapter titles/scene breaks/paragraphs, and emit sentence units."""
        units: List[_SentenceUnit] = []
        
        # Split text into lines while keeping track of character offsets
        line_offsets: List[Tuple[str, int]] = []
        curr_offset = 0
        for line in text.splitlines(keepends=True):
            line_offsets.append((line, curr_offset))
            curr_offset += len(line)

        current_chapter_id: Optional[str] = None
        current_chapter_title: Optional[str] = None
        paragraph_idx = 0
        current_para_lines: List[Tuple[str, int]] = []

        def flush_paragraph():
            nonlocal paragraph_idx, current_para_lines
            if not current_para_lines:
                return

            para_text = "".join(l[0] for l in current_para_lines)
            para_start_offset = current_para_lines[0][1]

            # Segment paragraph into sentences
            sents = self.split_into_sentences(para_text, base_offset=para_start_offset)
            for sent_text, s_start, s_end in sents:
                units.append(_SentenceUnit(
                    text=sent_text,
                    global_start=s_start,
                    global_end=s_end,
                    paragraph_idx=paragraph_idx,
                    chapter_id=current_chapter_id,
                    chapter_title=current_chapter_title,
                    is_scene_break=False,
                ))

            paragraph_idx += 1
            current_para_lines = []

        for line, line_offset in line_offsets:
            stripped = line.strip()

            # 1. Check for Chapter / Section delimiter
            ch_info = self.is_chapter_heading(stripped)
            if ch_info:
                flush_paragraph()
                current_chapter_id, current_chapter_title = ch_info
                continue

            # 2. Check for Scene break (e.g. '***')
            if self.is_scene_break(stripped):
                flush_paragraph()
                units.append(_SentenceUnit(
                    text=stripped,
                    global_start=line_offset,
                    global_end=line_offset + len(stripped),
                    paragraph_idx=paragraph_idx,
                    chapter_id=current_chapter_id,
                    chapter_title=current_chapter_title,
                    is_scene_break=True,
                ))
                paragraph_idx += 1
                continue

            # 3. Check for empty line (paragraph separator)
            if not stripped:
                flush_paragraph()
                continue

            # 4. Check for dialogue speaker turn indentation or quote start
            if current_para_lines and (line.startswith("    ") or line.startswith("\t")):
                flush_paragraph()

            current_para_lines.append((line, line_offset))

        # Flush final paragraph
        flush_paragraph()
        return units

    def _build_chunk(
        self,
        chunk_idx: int,
        sentences: List[_SentenceUnit],
        source_text: str,
    ) -> DiscourseChunk:
        """Construct a DiscourseChunk from an accumulated sequence of sentence units."""
        first_unit = sentences[0]
        last_unit = sentences[-1]

        global_start = first_unit.global_start
        global_end = last_unit.global_end
        verbatim_text = source_text[global_start:global_end]

        chunk_id = f"chunk_{chunk_idx:04d}"
        para_indices = sorted(list(set(u.paragraph_idx for u in sentences)))
        primary_para = para_indices[0] if para_indices else 0

        # Build SentenceSpans relative to chunk text
        sentence_spans: List[SentenceSpan] = []
        for idx, u in enumerate(sentences):
            # Chunk-relative coordinates
            rel_start = u.global_start - global_start
            rel_end = u.global_end - global_start
            sentence_spans.append(SentenceSpan(
                sentence_idx=idx,
                text=u.text,
                start_char=rel_start,
                end_char=rel_end,
                global_start_char=u.global_start,
                global_end_char=u.global_end,
            ))

        words = len(verbatim_text.split())
        token_estimate = int(words * 1.33) + 1

        return DiscourseChunk(
            chunk_id=chunk_id,
            text=verbatim_text,
            sentence_spans=sentence_spans,
            paragraph_index=primary_para,
            paragraph_indices=para_indices,
            chapter_id=first_unit.chapter_id,
            chapter_title=first_unit.chapter_title,
            global_offset=global_start,
            global_end_offset=global_end,
            word_count=words,
            token_count_estimate=token_estimate,
        )

    def stream_chunks(
        self,
        text_or_lines: Union[str, Iterable[str]],
    ) -> Iterator[DiscourseChunk]:
        """Stream discourse chunks incrementally from text or lines.
        
        Ensures chunks never split mid-sentence and conform to [min_words, max_words] limits.
        Flushes immediately when encountering chapter boundaries or scene breaks.
        """
        if isinstance(text_or_lines, str):
            full_text = text_or_lines
        else:
            full_text = "".join(text_or_lines)

        units = self._parse_source_units(full_text)
        if not units:
            return

        chunk_counter = 0
        current_sentences: List[_SentenceUnit] = []
        current_word_count = 0

        for unit in units:
            # Handle Scene Breaks
            if unit.is_scene_break:
                if current_sentences:
                    yield self._build_chunk(chunk_counter, current_sentences, full_text)
                    chunk_counter += 1
                    current_sentences = []
                    current_word_count = 0
                continue

            # Handle Chapter Changes
            if current_sentences and unit.chapter_id != current_sentences[0].chapter_id:
                yield self._build_chunk(chunk_counter, current_sentences, full_text)
                chunk_counter += 1
                current_sentences = []
                current_word_count = 0

            unit_words = unit.word_count

            # Single sentence exceeds max_words
            if unit_words >= self.max_words:
                # If we already have pending sentences, flush them first
                if current_sentences:
                    yield self._build_chunk(chunk_counter, current_sentences, full_text)
                    chunk_counter += 1
                    current_sentences = []
                    current_word_count = 0
                # Yield the oversized sentence intact in its own chunk
                yield self._build_chunk(chunk_counter, [unit], full_text)
                chunk_counter += 1
                continue

            # Check if adding this sentence exceeds max_words
            if current_word_count + unit_words > self.max_words:
                if current_word_count >= self.min_words:
                    # Satisfied minimum threshold: flush current buffer
                    yield self._build_chunk(chunk_counter, current_sentences, full_text)
                    chunk_counter += 1
                    current_sentences = [unit]
                    current_word_count = unit_words
                else:
                    # Current buffer is below min_words; appending exceeds max_words.
                    # Choose whether to flush now or group based on proximity to [min_words, max_words]
                    if current_word_count > 0:
                        yield self._build_chunk(chunk_counter, current_sentences, full_text)
                        chunk_counter += 1
                        current_sentences = [unit]
                        current_word_count = unit_words
                    else:
                        current_sentences.append(unit)
                        current_word_count += unit_words
            else:
                current_sentences.append(unit)
                current_word_count += unit_words

        # Flush any remaining sentences
        if current_sentences:
            yield self._build_chunk(chunk_counter, current_sentences, full_text)

    def chunk_document(self, text: str) -> List[DiscourseChunk]:
        """Convenience method to chunk an entire document string into a list of DiscourseChunks."""
        return list(self.stream_chunks(text))
