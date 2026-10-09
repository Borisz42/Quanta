"""Task Boundary & Query Extractor (§Phase 1, exp-027*).

Reliably isolates the core question or task directive q from bulky context documents
and conversational prompts. Provides modular strategies:
- QE-A: Legacy proxy heuristics (control / pass-through baseline)
- QE-B: Head-directive & head-question detection + imperative/question density scoring
- QE-C: Structured chat-message role awareness across multi-turn messages
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("quanta.parser.task_boundary")

INTERROGATIVE_WORDS = {
    "what", "who", "where", "why", "when", "how", "which", "whose", "whom",
    "compare", "is", "are", "can", "could", "would", "should", "does", "do", "did",
    # Hungarian interrogatives
    "mi", "mit", "milyen", "melyik", "hová", "hol", "mikor", "miért", "hogyan", "ki", "kit",
}

IMPERATIVE_VERBS = {
    "find", "answer", "extract", "determine", "identify", "calculate", "tell",
    "summarize", "explain", "track", "search", "locate", "provide", "return",
    "verify", "resolve", "state", "describe", "list", "name", "show", "give",
    "analyze", "evaluate", "compute", "rank", "select", "choose", "predict",
}

HEAD_DIRECTIVE_PATTERNS = [
    r"^(?:You are\b|Your task is\b|Instructions?\b|Please answer\b|Based on the following\b|Given the following\b|Answer the following\b|Read the following\b)",
    r"^(?:Kérjük\b|A feladatod\b|Utasítások?\b|Válaszolj az alábbi\b)",
]

DOCUMENT_START_MARKERS = [
    r"^(?:Paragraph|Narrative Block|Section|Passage|Document|Doc|Chapter|Source|Forrás|Szöveg)\s*(?:\[\d+\]|\d+:?|\b)",
    r"^===+\s*(?:Document|Context|Passages?)\s*===+",
]

QUESTION_STOPWORDS = {
    "what", "who", "where", "why", "when", "how", "which", "whose", "whom",
    "is", "are", "was", "were", "the", "a", "an", "in", "on", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "over",
    "under", "again", "further", "then", "once", "here", "there", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "answer", "instructions", "question",
    "query", "prompt", "final", "direct", "directly", "strictly", "provided",
    "conclude", "format", "based", "exact",
}


@dataclass
class ExtractedTaskIntent:
    """Represents the decomposed task query, context document, and boundary provenance."""
    query_text: str
    context_text: Optional[str] = None
    boundary_location: str = "FALLBACK"  # "HEAD", "TAIL", "SYSTEM", "EXPLICIT", "CHAT_ROLES", "FALLBACK"
    char_span: Optional[Tuple[int, int]] = None
    target_entities: List[str] = field(default_factory=list)
    strategy_used: str = "QE-A"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "context_text": self.context_text,
            "boundary_location": self.boundary_location,
            "char_span": list(self.char_span) if self.char_span else None,
            "target_entities": self.target_entities,
            "strategy_used": self.strategy_used,
        }


def extract_target_entities(text: str) -> List[str]:
    """Extracts salient entity seeds, technical identifiers, and proper nouns from a query string."""
    if not text:
        return []

    entities: List[str] = []
    seen = set()

    def _add(e: str):
        cleaned = e.strip(" \t\r\n.,;:!?'\"()[]{}")
        if not cleaned:
            return
        lower = cleaned.lower()
        if lower in seen or lower in QUESTION_STOPWORDS or len(cleaned) < 2:
            return
        seen.add(lower)
        entities.append(cleaned)

    # 1. Quoted strings (e.g. 'Answer: <final answer>', "Vault 81")
    for qm in re.findall(r"['\"]([^'\"]{2,60})['\"]", text):
        if not any(stop in qm.lower() for stop in ("answer", "final answer")):
            _add(qm)

    # 2. Alphanumeric codes, models, and technical identifiers (e.g. PHANTOM-9092, DELTA-X99, Vault 81)
    for code in re.findall(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b|\bVault\s+\d+\b", text):
        _add(code)

    # 3. Capitalized multi-word proper nouns (e.g. John von Neumann, Library of Alexandria)
    for pn in re.findall(r"\b[A-Z][a-z]+(?:\s+(?:von|van|de|da|of|the)\s+[A-Z][a-z]+|\s+[A-Z][a-z]+)+\b", text):
        _add(pn)

    # 4. Standalone capitalized proper nouns (avoiding sentence starter if it's a stopword)
    words = text.split()
    for i, w in enumerate(words):
        w_clean = w.strip(" \t\r\n.,;:!?'\"()[]{}")
        if w_clean and w_clean[0].isupper() and len(w_clean) > 2:
            if i == 0 and w_clean.lower() in QUESTION_STOPWORDS:
                continue
            _add(w_clean)

    return entities


# -----------------------------------------------------------------------------
# Standalone Legacy Functions (Preserved for 100% Backward Compatibility)
# -----------------------------------------------------------------------------

def decompose_query_context(text: Optional[str]) -> Tuple[Optional[str], str]:
    """Decomposes a single bulky user query into (context_document, core_question).

    Handles:
    - Explicit section delimiters:
      'Context:\n...\n\nQuestion: ...'
      'Documentation:\n...\n\nQuery: ...'
      'Source:\n...\n\nQuestion: ...'
      'Eredeti forrásdokumentumok:\n...\n\nKérdés: ...'
    - Text blocks where context precedes a terminal question sentence or paragraph.
    """
    if not text or not text.strip():
        return None, ""

    clean = text.strip()

    # 1. Explicit delimiter patterns (pre-filtered by keyword to eliminate regex backtracking)
    clean_lower = clean.lower()
    has_explicit_candidate = any(
        kw in clean_lower
        for kw in ("context", "documentation", "background", "source", "passage", "forrás", "szöveg", "question", "query", "prompt", "kérdés")
    )
    if has_explicit_candidate:
        explicit_patterns = [
            r"(?:===+\s*|#{1,6}\s*|\*{1,2})?(?:(?:Document\s+)?Context|Documentation|Background|Source|Passages?|Forrás(?:dokumentumok)?|Szöveg|Eredeti forrásdokumentumok)(?:\*{1,2}|===+)?\s*:\s*\n*([\s\S]*?)\n\s*(?:Question|Query|Prompt|Kérdés)\s*:\s*\n*([\s\S]*)",
            r"([\s\S]*?)\n\s*(?:Question|Query|Prompt|Kérdés)\s*:\s*\n*([\s\S]*)",
        ]
        for pat in explicit_patterns:
            m = re.search(pat, clean, re.IGNORECASE)
            if m:
                doc = m.group(1).strip()
                q = m.group(2).strip()
                if len(doc.split()) >= 15:
                    return doc, q

    # 2. Paragraph split heuristic if text has context (> 20 words) and ends with interrogative sentence
    paras = [p.strip() for p in clean.split("\n\n") if p.strip()]
    if len(paras) >= 2 and len(clean.split()) >= 20:
        last_para = paras[-1]
        if "?" in last_para or any(last_para.lower().startswith(w) for w in ("what", "who", "where", "why", "when", "how", "compare", "mi", "mit", "milyen", "melyik", "hová", "hol")):
            doc = "\n\n".join(paras[:-1]).strip()
            return doc, last_para

    # 3. Sentence split heuristic if ending sentence is an interrogative
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if len(sents) >= 3 and len(clean.split()) >= 20:
        last_sent = sents[-1]
        if last_sent.endswith("?") or any(last_sent.lower().startswith(w) for w in ("what", "who", "where", "why", "when", "how", "compare", "mi", "mit", "milyen", "melyik", "hová", "hol")):
            doc = " ".join(sents[:-1]).strip()
            return doc, last_sent

    return None, clean


def extract_context_from_system(messages: Sequence[Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extracts background context from a system message if present (e.g. 'Context:\n...' or 'Document context:\n...')."""
    for m in messages:
        # Support both ChatMessage objects with .role/.content and dicts
        role = getattr(m, "role", None) if hasattr(m, "role") else (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) if hasattr(m, "content") else (m.get("content") if isinstance(m, dict) else None)

        if role == "system" and content:
            text = str(content).strip()
            pat = (
                r"(?:===+\s*|#{1,6}\s*|\*{1,2})?"
                r"(?:(?:Document\s+)?Context|Documentation|Source|Passages?|Forrás(?:dokumentumok)?)"
                r"(?:\*{1,2}|===+)?\s*:\s*\n*([\s\S]*)"
            )
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                doc = match.group(1).strip()
                instruction = text[:match.start()].strip()
                if len(doc.split()) >= 15:
                    return doc, instruction
    return None, None


# -----------------------------------------------------------------------------
# Modular Task Boundary Extractor Class
# -----------------------------------------------------------------------------

class TaskBoundaryExtractor:
    """Modular Task Boundary & Query Extractor supporting QE-A, QE-B, and QE-C strategies."""

    def __init__(
        self,
        strategy: Optional[str] = None,
        config: Optional[Any] = None,
        head_window_words: int = 60,
        tail_window_words: int = 60,
    ):
        """Initializes the extractor with configured strategy.

        Args:
            strategy: Explicit strategy string ("QE-A", "QE-B", "QE-C", or "passthrough").
            config: Optional MultiScaleConfig instance.
            head_window_words: Token/word span size for head directive evaluation.
            tail_window_words: Token/word span size for tail query evaluation.
        """
        resolved_strat = strategy
        if resolved_strat is None and config is not None:
            resolved_strat = getattr(config, "query_extractor_strategy", None)
            if resolved_strat is None and hasattr(config, "get"):
                resolved_strat = config.get("query_extractor.strategy", "passthrough")

        if resolved_strat is None:
            resolved_strat = "passthrough"

        s_norm = str(resolved_strat).strip().upper().replace("_", "-")
        if s_norm in ("PASSTHROUGH", "CONTROL", "LEGACY", "QE-A"):
            self.strategy = "QE-A"
        elif s_norm in ("ENHANCED", "QE-B"):
            self.strategy = "QE-B"
        elif s_norm in ("CHAT", "STRUCTURED", "QE-C"):
            self.strategy = "QE-C"
        else:
            logger.warning("Unrecognized extractor strategy '%s'; defaulting to QE-A", strategy)
            self.strategy = "QE-A"

        self.head_window_words = head_window_words
        self.tail_window_words = tail_window_words

    def decompose(self, text: Optional[str]) -> Tuple[Optional[str], str]:
        """Drop-in replacement for decompose_query_context returning (context_text, query_text)."""
        intent = self.extract(text=text)
        return intent.context_text, intent.query_text

    def extract(
        self,
        text: Optional[str] = None,
        messages: Optional[Sequence[Any]] = None,
    ) -> ExtractedTaskIntent:
        """Decomposes prompt text or structured messages into an ExtractedTaskIntent."""
        # 1. Check structured messages if strategy is QE-C or messages provided
        if messages and (self.strategy == "QE-C" or text is None):
            intent = self._extract_from_messages(messages)
            if intent is not None:
                return intent

        # If text is not provided but messages exist, extract user content
        raw_text = text
        if raw_text is None and messages:
            for m in reversed(messages):
                role = getattr(m, "role", None) if hasattr(m, "role") else (m.get("role") if isinstance(m, dict) else None)
                content = getattr(m, "content", None) if hasattr(m, "content") else (m.get("content") if isinstance(m, dict) else None)
                if role == "user" and content:
                    raw_text = str(content)
                    break

        if not raw_text or not raw_text.strip():
            return ExtractedTaskIntent(
                query_text="",
                context_text=None,
                boundary_location="FALLBACK",
                char_span=None,
                target_entities=[],
                strategy_used=self.strategy,
            )

        clean = raw_text.strip()

        # Strategy-specific execution
        if self.strategy == "QE-A":
            return self._extract_qe_a(clean)
        elif self.strategy in ("QE-B", "QE-C"):
            return self._extract_qe_b(clean)
        else:
            return self._extract_qe_a(clean)

    # -------------------------------------------------------------------------
    # Internal Extractors
    # -------------------------------------------------------------------------

    def _extract_qe_a(self, clean: str) -> ExtractedTaskIntent:
        """QE-A: Legacy heuristic control (only explicit delimiters and tail interrogatives)."""
        doc, q = decompose_query_context(clean)

        loc = "FALLBACK"
        if doc is not None:
            # Check if explicit or tail
            if clean.startswith(doc):
                loc = "TAIL"
            else:
                loc = "EXPLICIT"

        char_span = self._find_char_span(clean, q)
        entities = extract_target_entities(q)

        return ExtractedTaskIntent(
            query_text=q,
            context_text=doc,
            boundary_location=loc,
            char_span=char_span,
            target_entities=entities,
            strategy_used="QE-A",
        )

    def _extract_qe_b(self, clean: str) -> ExtractedTaskIntent:
        """QE-B: Enhanced extractor with head/tail windows and imperative density scoring."""
        # 1. Explicit Section Delimiters (both Context -> Question and Question -> Context)
        explicit_res = self._check_explicit_delimiters(clean)
        if explicit_res:
            return explicit_res

        # 2. Check for Head-Query / Head-Directive structure
        head_res = self._check_head_boundary(clean)
        if head_res:
            return head_res

        # 3. Check for Tail-Query structure
        tail_res = self._check_tail_boundary(clean)
        if tail_res:
            return tail_res

        # 4. Window density scoring fallback over head vs tail
        density_res = self._check_density_scoring(clean)
        if density_res:
            return density_res

        # 5. Default Fallback
        return ExtractedTaskIntent(
            query_text=clean,
            context_text=None,
            boundary_location="FALLBACK",
            char_span=(0, len(clean)),
            target_entities=extract_target_entities(clean),
            strategy_used="QE-B",
        )

    def _check_explicit_delimiters(self, clean: str) -> Optional[ExtractedTaskIntent]:
        """Checks for explicit delimiter boundaries, supporting both orders."""
        clean_lower = clean.lower()
        if not any(
            kw in clean_lower
            for kw in ("context", "documentation", "background", "source", "passage", "forrás", "szöveg", "question", "query", "prompt", "kérdés", "task", "utasítás")
        ):
            return None

        # Standard: Context ... Question ...
        pat_cq = (
            r"(?:===+\s*|#{1,6}\s*|\*{1,2})?"
            r"(?:(?:Document\s+)?Context|Documentation|Background|Source|Passages?|Forrás(?:dokumentumok)?|Szöveg|Eredeti forrásdokumentumok)"
            r"(?:\*{1,2}|===+)?\s*:\s*\n*([\s\S]*?)\n\s*(?:Question|Query|Prompt|Kérdés)\s*:\s*\n*([\s\S]*)"
        )
        m = re.search(pat_cq, clean, re.IGNORECASE)
        if m:
            doc = m.group(1).strip()
            q = m.group(2).strip()
            if len(doc.split()) >= 15:
                return ExtractedTaskIntent(
                    query_text=q,
                    context_text=doc,
                    boundary_location="EXPLICIT",
                    char_span=self._find_char_span(clean, q),
                    target_entities=extract_target_entities(q),
                    strategy_used="QE-B",
                )

        # Inverted: Question ... Context ...
        pat_qc = (
            r"(?:===+\s*|#{1,6}\s*|\*{1,2})?"
            r"(?:Question|Query|Prompt|Kérdés|Task|Utasítás)"
            r"(?:\*{1,2}|===+)?\s*:\s*\n*([\s\S]*?)\n\s*"
            r"(?:===+\s*|#{1,6}\s*|\*{1,2})?"
            r"(?:(?:Document\s+)?Context|Documentation|Background|Source|Passages?|Forrás(?:dokumentumok)?|Szöveg)"
            r"(?:\*{1,2}|===+)?\s*:\s*\n*([\s\S]*)"
        )
        m2 = re.search(pat_qc, clean, re.IGNORECASE)
        if m2:
            q = m2.group(1).strip()
            doc = m2.group(2).strip()
            if len(doc.split()) >= 15:
                return ExtractedTaskIntent(
                    query_text=q,
                    context_text=doc,
                    boundary_location="EXPLICIT",
                    char_span=self._find_char_span(clean, q),
                    target_entities=extract_target_entities(q),
                    strategy_used="QE-B",
                )

        # Generic section split: Context: <doc>
        pat_ctx_only = (
            r"(?:===+\s*|#{1,6}\s*|\*{1,2})?"
            r"(?:(?:Document\s+)?Context|Passages?|Documentation)\s*:\s*\n*([\s\S]*)"
        )
        m3 = re.match(pat_ctx_only, clean, re.IGNORECASE)
        if m3 and len(clean.split()) >= 25:
            doc = m3.group(1).strip()
            return ExtractedTaskIntent(
                query_text="",
                context_text=doc,
                boundary_location="EXPLICIT",
                char_span=None,
                target_entities=[],
                strategy_used="QE-B",
            )

        return None

    def _check_head_boundary(self, clean: str) -> Optional[ExtractedTaskIntent]:
        """Detects if query or instructions reside in the head of the prompt preceding document context."""
        paras = [p.strip() for p in clean.split("\n\n") if p.strip()]
        if len(paras) < 2 or len(clean.split()) < 20:
            return None

        # Check paragraph 1 or paragraphs [0..k] for head-directive or interrogative
        candidate_q_paras = []
        doc_start_idx = -1

        for i, para in enumerate(paras):
            is_doc_start = any(re.search(marker, para, re.IGNORECASE) for marker in DOCUMENT_START_MARKERS)
            if is_doc_start:
                doc_start_idx = i
                break

            # If not doc start, check if it's directive or question
            is_directive = any(re.search(pat, para, re.IGNORECASE) for pat in HEAD_DIRECTIVE_PATTERNS)
            is_interrogative = "?" in para or self._starts_with_interrogative(para)

            if is_directive or is_interrogative:
                candidate_q_paras.append(para)
            else:
                # If we encounter a paragraph that is neither, stop collecting head query
                break

        if doc_start_idx > 0 and candidate_q_paras:
            # Clear marker-based boundary found
            q_text = "\n\n".join(candidate_q_paras).strip()
            doc_text = "\n\n".join(paras[doc_start_idx:]).strip()
            if len(doc_text.split()) >= 15:
                return ExtractedTaskIntent(
                    query_text=q_text,
                    context_text=doc_text,
                    boundary_location="HEAD",
                    char_span=self._find_char_span(clean, q_text),
                    target_entities=extract_target_entities(q_text),
                    strategy_used="QE-B",
                )

        # Check first paragraph: if first para is question or directive and remaining text is long
        first_para = paras[0]
        first_words = len(first_para.split())
        rem_text = "\n\n".join(paras[1:]).strip()
        rem_words = len(rem_text.split())

        if first_words <= self.head_window_words and rem_words >= 20:
            has_q = "?" in first_para or self._starts_with_interrogative(first_para)
            has_directive = any(re.search(pat, first_para, re.IGNORECASE) for pat in HEAD_DIRECTIVE_PATTERNS)
            if has_q or has_directive:
                # Check if second para is also part of query instructions (e.g. Instructions: Track...)
                if len(paras) >= 3 and len(paras[1].split()) <= 30 and any(re.search(pat, paras[1], re.IGNORECASE) for pat in HEAD_DIRECTIVE_PATTERNS):
                    q_text = f"{first_para}\n\n{paras[1]}"
                    doc_text = "\n\n".join(paras[2:]).strip()
                else:
                    q_text = first_para
                    doc_text = rem_text

                return ExtractedTaskIntent(
                    query_text=q_text,
                    context_text=doc_text,
                    boundary_location="HEAD",
                    char_span=self._find_char_span(clean, q_text),
                    target_entities=extract_target_entities(q_text),
                    strategy_used="QE-B",
                )

        return None

    def _check_tail_boundary(self, clean: str) -> Optional[ExtractedTaskIntent]:
        """Detects if query resides in the tail of the prompt preceded by document context."""
        paras = [p.strip() for p in clean.split("\n\n") if p.strip()]
        if len(paras) >= 2 and len(clean.split()) >= 20:
            last_para = paras[-1]
            if len(last_para.split()) <= self.tail_window_words:
                if "?" in last_para or self._starts_with_interrogative(last_para):
                    doc_text = "\n\n".join(paras[:-1]).strip()
                    return ExtractedTaskIntent(
                        query_text=last_para,
                        context_text=doc_text,
                        boundary_location="TAIL",
                        char_span=self._find_char_span(clean, last_para),
                        target_entities=extract_target_entities(last_para),
                        strategy_used="QE-B",
                    )

        # Sentence-level tail fallback
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
        if len(sents) >= 3 and len(clean.split()) >= 20:
            last_sent = sents[-1]
            if len(last_sent.split()) <= self.tail_window_words:
                if last_sent.endswith("?") or self._starts_with_interrogative(last_sent):
                    doc_text = " ".join(sents[:-1]).strip()
                    return ExtractedTaskIntent(
                        query_text=last_sent,
                        context_text=doc_text,
                        boundary_location="TAIL",
                        char_span=self._find_char_span(clean, last_sent),
                        target_entities=extract_target_entities(last_sent),
                        strategy_used="QE-B",
                    )

        return None

    def _check_density_scoring(self, clean: str) -> Optional[ExtractedTaskIntent]:
        """Calculates imperative and interrogative density in head and tail windows."""
        words = clean.split()
        if len(words) < 25:
            return None

        head_span = " ".join(words[:self.head_window_words])
        tail_span = " ".join(words[-self.tail_window_words:])

        head_score = self._compute_intent_score(head_span)
        tail_score = self._compute_intent_score(tail_span)

        # Significant intent threshold: score >= 2.0
        if head_score >= 2.0 and head_score > tail_score:
            # Find closest sentence or paragraph break in head
            paras = clean.split("\n\n")
            if len(paras) >= 2:
                q_text = paras[0].strip()
                doc_text = "\n\n".join(paras[1:]).strip()
                return ExtractedTaskIntent(
                    query_text=q_text,
                    context_text=doc_text,
                    boundary_location="HEAD",
                    char_span=self._find_char_span(clean, q_text),
                    target_entities=extract_target_entities(q_text),
                    strategy_used="QE-B",
                )

        if tail_score >= 2.0 and tail_score > head_score:
            paras = clean.split("\n\n")
            if len(paras) >= 2:
                q_text = paras[-1].strip()
                doc_text = "\n\n".join(paras[:-1]).strip()
                return ExtractedTaskIntent(
                    query_text=q_text,
                    context_text=doc_text,
                    boundary_location="TAIL",
                    char_span=self._find_char_span(clean, q_text),
                    target_entities=extract_target_entities(q_text),
                    strategy_used="QE-B",
                )

        return None

    def _extract_from_messages(self, messages: Sequence[Any]) -> Optional[ExtractedTaskIntent]:
        """Structured chat-message boundary extractor for QE-C."""
        # 1. System context check
        sys_doc, sys_inst = extract_context_from_system(messages)

        # 2. Extract user and assistant turns
        user_turns: List[Tuple[int, str]] = []
        for idx, m in enumerate(messages):
            role = getattr(m, "role", None) if hasattr(m, "role") else (m.get("role") if isinstance(m, dict) else None)
            content = getattr(m, "content", None) if hasattr(m, "content") else (m.get("content") if isinstance(m, dict) else None)
            if role == "user" and content:
                user_turns.append((idx, str(content).strip()))

        if not user_turns:
            return None

        last_user_idx, last_user_content = user_turns[-1]

        # Case A: System message provided bulky context, user provided question
        if sys_doc:
            q_text = last_user_content
            if sys_inst:
                q_text = f"{sys_inst}\n\n{q_text}".strip()
            return ExtractedTaskIntent(
                query_text=q_text,
                context_text=sys_doc,
                boundary_location="SYSTEM",
                char_span=None,
                target_entities=extract_target_entities(q_text),
                strategy_used="QE-C",
            )

        # Case B: Multi-turn prompt where earlier user provided bulky document and last user provided question
        if len(user_turns) >= 2:
            prev_user_idx, prev_user_content = user_turns[-2]
            prev_words = len(prev_user_content.split())
            last_words = len(last_user_content.split())
            if prev_words >= 30 and last_words <= self.tail_window_words:
                return ExtractedTaskIntent(
                    query_text=last_user_content,
                    context_text=prev_user_content,
                    boundary_location="CHAT_ROLES",
                    char_span=None,
                    target_entities=extract_target_entities(last_user_content),
                    strategy_used="QE-C",
                )

        # Case C: Single bulky user message containing both context and question -> evaluate with QE-B
        intent = self._extract_qe_b(last_user_content)
        intent.strategy_used = "QE-C"
        return intent

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _starts_with_interrogative(text: str) -> bool:
        """Returns True if text begins with an interrogative question word."""
        clean = text.strip().lower()
        first_token = re.split(r"\s+", clean)[0].strip(".,;:!?\"'")
        return first_token in INTERROGATIVE_WORDS

    @staticmethod
    def _compute_intent_score(span_text: str) -> float:
        """Computes a heuristic score reflecting question/instructional density in a text span."""
        score = 0.0
        # Question marks
        score += span_text.count("?") * 2.0

        # Token analysis
        tokens = [t.strip(".,;:!?\"'()[]").lower() for t in span_text.split() if t.strip()]
        for tok in tokens:
            if tok in INTERROGATIVE_WORDS:
                score += 1.0
            if tok in IMPERATIVE_VERBS:
                score += 1.5

        return score

    @staticmethod
    def _find_char_span(full_text: str, query_text: str) -> Optional[Tuple[int, int]]:
        """Locates character start and end index of query_text within full_text."""
        if not full_text or not query_text:
            return None
        idx = full_text.find(query_text)
        if idx != -1:
            return (idx, idx + len(query_text))
        return None
