"""Canonical Record DSL Lexer, Parser, AST, and Serializer for QUANTA SVM.

Replaces nested S-expressions with human-readable, auditable flat Record DSL blocks:
- passage P... { ... }
- entity E... "..." { ... }
- event EV... "..." { ... }
- relation R... { ... }

Provides lossless round-trip compilation between Record DSL, AST schemas,
DiscourseExtractionResult, and QuantaGraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from memory.passage_store import PassageRecord, PassageStore
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
    ExtractedTimeInterval,
)


# ---------------------------------------------------------------------------
# AST Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PassageBlock:
    """AST node for a passage record block (V_passage)."""
    id: str
    doc_id: str = "default_doc"
    char_span: Tuple[int, int] = (0, 0)
    text: str = ""
    created_at: float = field(default_factory=time.time)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_passage_record(self) -> PassageRecord:
        """Converts to a PassageRecord instance."""
        return PassageRecord(
            passage_id=self.id,
            doc_id=self.doc_id,
            char_span=self.char_span,
            text=self.text,
            created_at=self.created_at,
        )

    @classmethod
    def from_passage_record(cls, rec: PassageRecord) -> PassageBlock:
        """Constructs from a PassageRecord instance."""
        return cls(
            id=rec.passage_id,
            doc_id=rec.doc_id,
            char_span=rec.char_span,
            text=rec.text,
            created_at=rec.created_at,
        )


@dataclass
class EntityBlock:
    """AST node for an entity definition block."""
    id: str
    canonical_name: str
    category: str = "OBJECT"
    passage_id: Optional[str] = None
    char_span: Optional[Tuple[int, int]] = None
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    salience: float = 1.0
    confidence: float = 1.0

    def to_extracted_entity(self) -> ExtractedEntity:
        """Converts to an ExtractedEntity instance."""
        props = dict(self.properties)
        if self.passage_id:
            props["passage_id"] = self.passage_id
        if self.char_span:
            props["char_span"] = list(self.char_span)
        props["salience"] = self.salience
        props["confidence"] = self.confidence
        return ExtractedEntity(
            id=self.id,
            canonical_name=self.canonical_name,
            category=self.category,
            surface_aliases=list(self.aliases),
            properties=props,
        )

    @classmethod
    def from_extracted_entity(cls, ent: ExtractedEntity) -> EntityBlock:
        """Constructs from an ExtractedEntity instance."""
        props = dict(ent.properties)
        passage_id = props.pop("passage_id", None)
        char_span = props.pop("char_span", None)
        if char_span is not None and isinstance(char_span, (list, tuple)) and len(char_span) == 2:
            span_tuple = (int(char_span[0]), int(char_span[1]))
        else:
            span_tuple = None
        salience = float(props.pop("salience", 1.0))
        confidence = float(props.pop("confidence", 1.0))

        return cls(
            id=ent.id,
            canonical_name=ent.canonical_name,
            category=ent.category,
            passage_id=passage_id,
            char_span=span_tuple,
            aliases=list(ent.surface_aliases),
            properties=props,
            salience=salience,
            confidence=confidence,
        )


@dataclass
class EventBlock:
    """AST node for an event definition block."""
    id: str
    predicate: str
    agent_id: Optional[str] = None
    patient_id: Optional[str] = None
    theme_id: Optional[str] = None
    location_id: Optional[str] = None
    instrument_id: Optional[str] = None
    passage_id: Optional[str] = None
    char_span: Optional[Tuple[int, int]] = None
    time_interval: Optional[Tuple[Any, Any]] = None
    tense: str = "PAST"
    aspect: str = "SIMPLE"
    val: str = "TRUE"
    modality: Optional[str] = None
    confidence: float = 1.0
    intent: Optional[str] = None
    epistemic: Optional[str] = None
    raw_text: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    salience: float = 1.0

    def to_extracted_event(self) -> ExtractedEvent:
        """Converts to an ExtractedEvent instance."""
        t_interval = None
        if self.time_interval:
            t_interval = ExtractedTimeInterval(
                start=self.time_interval[0],
                end=self.time_interval[1],
            )
        args = dict(self.arguments)
        if self.passage_id:
            args["passage_id"] = self.passage_id
        if self.char_span:
            args["char_span"] = list(self.char_span)
        if self.intent:
            args["intent"] = self.intent
        if self.epistemic:
            args["epistemic"] = self.epistemic
        args["confidence"] = self.confidence
        args["salience"] = self.salience

        return ExtractedEvent(
            id=self.id,
            predicate=self.predicate,
            agent_id=self.agent_id,
            patient_id=self.patient_id,
            theme_id=self.theme_id,
            location_id=self.location_id,
            instrument_id=self.instrument_id,
            time_interval=t_interval,
            tense=self.tense,
            aspect=self.aspect,
            val=self.val,
            modality=self.modality,
            raw_text=self.raw_text,
            arguments=args,
        )

    @classmethod
    def from_extracted_event(cls, ev: ExtractedEvent) -> EventBlock:
        """Constructs from an ExtractedEvent instance."""
        args = dict(ev.arguments) if ev.arguments else {}
        passage_id = args.pop("passage_id", None)
        char_span = args.pop("char_span", None)
        if char_span is not None and isinstance(char_span, (list, tuple)) and len(char_span) == 2:
            span_tuple = (int(char_span[0]), int(char_span[1]))
        else:
            span_tuple = None

        t_interval = None
        if ev.time_interval:
            t_interval = (ev.time_interval.start, ev.time_interval.end)

        intent = args.pop("intent", None)
        epistemic = args.pop("epistemic", None)
        confidence = float(args.pop("confidence", 1.0))
        salience = float(args.pop("salience", 1.0))

        return cls(
            id=ev.id,
            predicate=ev.predicate,
            agent_id=ev.agent_id,
            patient_id=ev.patient_id,
            theme_id=ev.theme_id,
            location_id=ev.location_id,
            instrument_id=ev.instrument_id,
            passage_id=passage_id,
            char_span=span_tuple,
            time_interval=t_interval,
            tense=ev.tense or "PAST",
            aspect=ev.aspect or "SIMPLE",
            val=ev.val or "TRUE",
            modality=ev.modality,
            confidence=confidence,
            intent=intent,
            epistemic=epistemic,
            raw_text=ev.raw_text,
            arguments=args,
            salience=salience,
        )


@dataclass
class RelationBlock:
    """AST node for a relational edge block."""
    id: str
    rel_type: str
    source_id: str
    target_id: str
    confidence: float = 1.0
    truth_status: str = "TRUE"
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_extracted_relation(self) -> ExtractedRelation:
        """Converts to an ExtractedRelation instance."""
        props = dict(self.properties)
        props["truth_status"] = self.truth_status
        return ExtractedRelation(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            relation_type=self.rel_type,
            confidence=self.confidence,
            properties=props,
        )

    @classmethod
    def from_extracted_relation(cls, rel: ExtractedRelation) -> RelationBlock:
        """Constructs from an ExtractedRelation instance."""
        props = dict(rel.properties) if rel.properties else {}
        truth = props.pop("truth_status", "TRUE")
        return cls(
            id=rel.id or f"R_{rel.source_id}_{rel.target_id}",
            rel_type=rel.relation_type,
            source_id=rel.source_id,
            target_id=rel.target_id,
            confidence=float(rel.confidence) if rel.confidence is not None else 1.0,
            truth_status=str(truth),
            properties=props,
        )


@dataclass
class RecordDSLDocument:
    """Top-level AST representing a complete Canonical Record DSL document."""
    passages: List[PassageBlock] = field(default_factory=list)
    entities: List[EntityBlock] = field(default_factory=list)
    events: List[EventBlock] = field(default_factory=list)
    relations: List[RelationBlock] = field(default_factory=list)

    def to_dsl(self) -> str:
        """Serializes document to canonical Record DSL text."""
        return RecordDSLSerializer().serialize(self)

    def to_extraction_result(self) -> DiscourseExtractionResult:
        """Converts document to a DiscourseExtractionResult instance."""
        extracted_entities = [e.to_extracted_entity() for e in self.entities]
        extracted_events = [ev.to_extracted_event() for ev in self.events]
        extracted_relations = [r.to_extracted_relation() for r in self.relations]

        # Extract propositions from events with epistemic status
        propositions: List[ExtractedProposition] = []
        for ev in self.events:
            if ev.epistemic or ev.val != "TRUE":
                status = ev.epistemic or ("OBSERVATION" if ev.val == "TRUE" else "HYPOTHESIS")
                propositions.append(
                    ExtractedProposition(
                        id=f"Prop_{ev.id}",
                        event_id=ev.id,
                        epistemic_status=status.upper(),
                        modality=ev.modality or "CERTAIN",
                        confidence=ev.confidence,
                        source="direct_observation" if not ev.epistemic else ev.epistemic.lower(),
                    )
                )

        return DiscourseExtractionResult(
            entities=extracted_entities,
            events=extracted_events,
            relations=extracted_relations,
            propositions=propositions,
        )

    @classmethod
    def from_extraction_result(
        cls,
        result: DiscourseExtractionResult,
        passages: Optional[Sequence[PassageRecord]] = None,
    ) -> RecordDSLDocument:
        """Constructs a RecordDSLDocument from a DiscourseExtractionResult."""
        doc = cls()
        if passages:
            doc.passages = [PassageBlock.from_passage_record(p) for p in passages]

        doc.entities = [EntityBlock.from_extracted_entity(e) for e in result.entities]
        doc.events = [EventBlock.from_extracted_event(ev) for ev in result.events]
        doc.relations = [RelationBlock.from_extracted_relation(r) for r in result.relations]

        # Annotate event epistemic values from propositions
        prop_map = {p.event_id: p for p in result.propositions if p.event_id}
        for ev_block in doc.events:
            if ev_block.id in prop_map:
                prop = prop_map[ev_block.id]
                if not ev_block.epistemic and prop.epistemic_status:
                    ev_block.epistemic = prop.epistemic_status
                if prop.confidence and ev_block.confidence == 1.0:
                    ev_block.confidence = prop.confidence

        return doc


# ---------------------------------------------------------------------------
# Lexer & Token Definitions
# ---------------------------------------------------------------------------

class TokenType(Enum):
    IDENT = auto()       # e.g. passage, entity, P101, E1, TRUE
    STRING = auto()      # "..."
    NUMBER = auto()      # 123, 123.45
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    COLON = auto()       # :
    COMMA = auto()       # ,
    ARROW = auto()       # ->
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    col: int


class RecordDSLSyntaxError(Exception):
    """Raised when Record DSL lexing or parsing fails."""
    def __init__(self, message: str, line: int = 0, col: int = 0):
        super().__init__(f"Line {line}, Col {col}: {message}")
        self.line = line
        self.col = col


class RecordDSLLexer:
    """Recursive-descent lexer for Canonical Record DSL."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.length = len(text)

    def _peek(self) -> str:
        if self.pos < self.length:
            return self.text[self.pos]
        return ""

    def _advance(self) -> str:
        ch = self._peek()
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.pos < self.length:
            ch = self._peek()

            # Skip whitespace
            if ch.isspace():
                self._advance()
                continue

            # Skip comments (# or //)
            if ch == "#" or (ch == "/" and self.pos + 1 < self.length and self.text[self.pos + 1] == "/"):
                while self.pos < self.length and self._peek() != "\n":
                    self._advance()
                continue

            line = self.line
            col = self.col

            # Punctuation
            if ch == "{":
                self._advance()
                tokens.append(Token(TokenType.LBRACE, "{", line, col))
            elif ch == "}":
                self._advance()
                tokens.append(Token(TokenType.RBRACE, "}", line, col))
            elif ch == "[":
                self._advance()
                tokens.append(Token(TokenType.LBRACKET, "[", line, col))
            elif ch == "]":
                self._advance()
                tokens.append(Token(TokenType.RBRACKET, "]", line, col))
            elif ch == ":":
                self._advance()
                tokens.append(Token(TokenType.COLON, ":", line, col))
            elif ch == ",":
                self._advance()
                tokens.append(Token(TokenType.COMMA, ",", line, col))
            elif ch == "-" and self.pos + 1 < self.length and self.text[self.pos + 1] == ">":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.ARROW, "->", line, col))

            # String literal
            elif ch == '"' or ch == "'":
                tokens.append(self._read_string(ch, line, col))

            # Numbers (integer or float, optional leading sign if followed by digit)
            elif ch.isdigit() or (ch in ("-", "+") and self.pos + 1 < self.length and self.text[self.pos + 1].isdigit()):
                tokens.append(self._read_number(line, col))

            # Identifiers
            elif ch.isalpha() or ch == "_":
                tokens.append(self._read_ident(line, col))

            else:
                raise RecordDSLSyntaxError(f"Unexpected character: {ch!r}", line, col)

        tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return tokens

    def _read_string(self, quote_char: str, line: int, col: int) -> Token:
        self._advance()  # opening quote
        chars = []
        while self.pos < self.length:
            ch = self._peek()
            if ch == quote_char:
                self._advance()  # closing quote
                return Token(TokenType.STRING, "".join(chars), line, col)
            if ch == "\\":
                self._advance()
                esc = self._peek()
                if esc == "n":
                    chars.append("\n")
                elif esc == "t":
                    chars.append("\t")
                elif esc == "r":
                    chars.append("\r")
                elif esc == "\\":
                    chars.append("\\")
                elif esc == quote_char:
                    chars.append(quote_char)
                else:
                    chars.append(esc)
                self._advance()
            else:
                chars.append(ch)
                self._advance()

        raise RecordDSLSyntaxError(f"Unterminated string literal starting at {quote_char}", line, col)

    def _read_number(self, line: int, col: int) -> Token:
        start_pos = self.pos
        if self._peek() in ("-", "+"):
            self._advance()

        has_dot = False
        while self.pos < self.length:
            ch = self._peek()
            if ch.isdigit():
                self._advance()
            elif ch == "." and not has_dot and self.pos + 1 < self.length and self.text[self.pos + 1].isdigit():
                has_dot = True
                self._advance()
            elif ch in ("e", "E"):
                self._advance()
                if self._peek() in ("+", "-"):
                    self._advance()
            else:
                break

        num_str = self.text[start_pos : self.pos]
        val = float(num_str) if has_dot or "e" in num_str.lower() else int(num_str)
        return Token(TokenType.NUMBER, val, line, col)

    def _read_ident(self, line: int, col: int) -> Token:
        start_pos = self.pos
        while self.pos < self.length:
            ch = self._peek()
            # Allow alphanumeric, underscore, dot, and hyphen in identifiers
            if ch.isalnum() or ch in ("_", "-", "."):
                self._advance()
            else:
                break
        ident_str = self.text[start_pos : self.pos]
        return Token(TokenType.IDENT, ident_str, line, col)


# ---------------------------------------------------------------------------
# Recursive-Descent Parser
# ---------------------------------------------------------------------------

class RecordDSLParser:
    """Recursive-descent parser for Canonical Record DSL."""

    def __init__(self, tokens: Optional[List[Token]] = None):
        self.tokens: List[Token] = tokens or []
        self.cursor = 0

    @classmethod
    def parse(cls, dsl_text: str) -> RecordDSLDocument:
        """Parses a Record DSL string into a RecordDSLDocument AST."""
        lexer = RecordDSLLexer(dsl_text)
        tokens = lexer.tokenize()
        parser = cls(tokens)
        return parser.parse_document()

    def _current(self) -> Token:
        if self.cursor < len(self.tokens):
            return self.tokens[self.cursor]
        return self.tokens[-1]

    def _peek(self) -> Token:
        return self._current()

    def _peek_next(self) -> Optional[Token]:
        if self.cursor + 1 < len(self.tokens):
            return self.tokens[self.cursor + 1]
        return None

    def _advance(self) -> Token:
        tok = self._current()
        if self.cursor < len(self.tokens) - 1:
            self.cursor += 1
        return tok

    def _match(self, expected_type: TokenType) -> bool:
        return self._current().type == expected_type

    def _consume(self, expected_type: TokenType, err_msg: str) -> Token:
        tok = self._current()
        if tok.type != expected_type:
            raise RecordDSLSyntaxError(
                f"{err_msg}. Got {tok.type.name} ({tok.value!r})",
                tok.line,
                tok.col,
            )
        return self._advance()

    def parse_document(self) -> RecordDSLDocument:
        """Parses the entire token stream into a RecordDSLDocument."""
        doc = RecordDSLDocument()

        while not self._match(TokenType.EOF):
            tok = self._current()
            if tok.type != TokenType.IDENT:
                raise RecordDSLSyntaxError(
                    f"Expected block keyword ('passage', 'entity', 'event', 'relation'), got {tok.value!r}",
                    tok.line,
                    tok.col,
                )

            keyword = tok.value.lower()
            if keyword == "passage":
                doc.passages.append(self._parse_passage_block())
            elif keyword == "entity":
                doc.entities.append(self._parse_entity_block())
            elif keyword == "event":
                doc.events.append(self._parse_event_block())
            elif keyword == "relation":
                doc.relations.append(self._parse_relation_block())
            else:
                raise RecordDSLSyntaxError(
                    f"Unknown block declaration {tok.value!r}. Must be one of passage, entity, event, relation.",
                    tok.line,
                    tok.col,
                )

        return doc

    def _parse_passage_block(self) -> PassageBlock:
        self._advance()  # consume 'passage'
        passage_id = ""
        if self._match(TokenType.IDENT) or self._match(TokenType.STRING):
            passage_id = str(self._advance().value)

        body = self._parse_block_body()
        p_id = passage_id or str(body.get("id") or f"P{len(body)}")
        doc_id = str(body.get("doc_id", "default_doc"))

        raw_span = body.get("span") or body.get("char_span")
        if raw_span and isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
            char_span = (int(raw_span[0]), int(raw_span[1]))
        else:
            s_start = int(body.get("span_start", 0))
            s_end = int(body.get("span_end", len(str(body.get("text", "")))))
            char_span = (s_start, s_end)

        text = str(body.get("text", ""))
        created_at = float(body.get("created_at", time.time()))

        return PassageBlock(
            id=p_id,
            doc_id=doc_id,
            char_span=char_span,
            text=text,
            created_at=created_at,
            properties=body,
        )

    def _parse_entity_block(self) -> EntityBlock:
        self._advance()  # consume 'entity'
        ent_id = ""
        canonical_name = ""

        if self._match(TokenType.IDENT) or self._match(TokenType.STRING):
            ent_id = str(self._advance().value)

        # Optional canonical name in quotes: entity E1 "Albert Einstein" { ... }
        if self._match(TokenType.STRING):
            canonical_name = str(self._advance().value)

        body = self._parse_block_body()
        e_id = ent_id or str(body.get("id") or "E1")
        name = canonical_name or str(body.get("canonical_name") or body.get("name") or e_id)
        category = str(body.get("category", "OBJECT")).upper()

        passage_id = body.get("passage") or body.get("passage_id")
        passage_id = str(passage_id) if passage_id is not None else None

        raw_span = body.get("span") or body.get("char_span")
        if raw_span and isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
            char_span = (int(raw_span[0]), int(raw_span[1]))
        elif "span_start" in body and "span_end" in body:
            char_span = (int(body["span_start"]), int(body["span_end"]))
        else:
            char_span = None

        aliases = body.get("aliases") or body.get("surface_aliases") or []
        if isinstance(aliases, (list, tuple)):
            aliases = [str(a) for a in aliases]
        else:
            aliases = [str(aliases)]

        props = body.get("properties", {})
        if not isinstance(props, dict):
            props = {}

        salience = float(body.get("salience", 1.0))
        confidence = float(body.get("confidence", 1.0))

        return EntityBlock(
            id=e_id,
            canonical_name=name,
            category=category,
            passage_id=passage_id,
            char_span=char_span,
            aliases=aliases,
            properties=props,
            salience=salience,
            confidence=confidence,
        )

    def _parse_event_block(self) -> EventBlock:
        self._advance()  # consume 'event'
        ev_id = ""
        predicate = ""

        if self._match(TokenType.IDENT) or self._match(TokenType.STRING):
            ev_id = str(self._advance().value)

        # Optional predicate in quotes: event EV1 "discover" { ... }
        if self._match(TokenType.STRING):
            predicate = str(self._advance().value)

        body = self._parse_block_body()
        e_id = ev_id or str(body.get("id") or "EV1")
        pred = predicate or str(body.get("predicate") or body.get("action") or "event")

        agent_id = body.get("agent") or body.get("agent_id")
        agent_id = str(agent_id) if agent_id is not None else None

        patient_id = body.get("patient") or body.get("patient_id")
        patient_id = str(patient_id) if patient_id is not None else None

        theme_id = body.get("theme") or body.get("theme_id")
        theme_id = str(theme_id) if theme_id is not None else None

        location_id = body.get("location") or body.get("location_id")
        location_id = str(location_id) if location_id is not None else None

        instrument_id = body.get("instrument") or body.get("instrument_id")
        instrument_id = str(instrument_id) if instrument_id is not None else None

        passage_id = body.get("passage") or body.get("passage_id")
        passage_id = str(passage_id) if passage_id is not None else None

        raw_span = body.get("span") or body.get("char_span")
        if raw_span and isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
            char_span = (int(raw_span[0]), int(raw_span[1]))
        elif "span_start" in body and "span_end" in body:
            char_span = (int(body["span_start"]), int(body["span_end"]))
        else:
            char_span = None

        raw_time = body.get("time") or body.get("time_interval")
        if raw_time and isinstance(raw_time, (list, tuple)) and len(raw_time) == 2:
            time_interval = (raw_time[0], raw_time[1])
        elif isinstance(raw_time, dict):
            time_interval = (raw_time.get("start"), raw_time.get("end"))
        else:
            time_interval = None

        tense = str(body.get("tense", "PAST")).upper()
        aspect = str(body.get("aspect", "SIMPLE")).upper()
        val = str(body.get("val", body.get("truth_status", body.get("belnap", "TRUE")))).upper()
        modality = body.get("modality")
        confidence = float(body.get("confidence", 1.0))
        intent = body.get("intent") or body.get("speech_act")
        epistemic = body.get("epistemic") or body.get("evidence_source")
        raw_text = body.get("raw_text")

        args = body.get("arguments", {})
        if not isinstance(args, dict):
            args = {}

        salience = float(body.get("salience", 1.0))

        return EventBlock(
            id=e_id,
            predicate=pred,
            agent_id=agent_id,
            patient_id=patient_id,
            theme_id=theme_id,
            location_id=location_id,
            instrument_id=instrument_id,
            passage_id=passage_id,
            char_span=char_span,
            time_interval=time_interval,
            tense=tense,
            aspect=aspect,
            val=val,
            modality=str(modality) if modality else None,
            confidence=confidence,
            intent=str(intent).upper() if intent else None,
            epistemic=str(epistemic).upper() if epistemic else None,
            raw_text=str(raw_text) if raw_text else None,
            arguments=args,
            salience=salience,
        )

    def _parse_relation_block(self) -> RelationBlock:
        self._advance()  # consume 'relation'
        rel_id = ""
        rel_type = ""

        if self._match(TokenType.IDENT) or self._match(TokenType.STRING):
            rel_id = str(self._advance().value)

        # Optional relation type: relation R1 "CAUSAL_MECHANISM_LINK" { ... }
        if self._match(TokenType.STRING) or self._match(TokenType.IDENT):
            if not self._match(TokenType.LBRACE):
                rel_type = str(self._advance().value)

        body = self._parse_block_body()
        r_id = rel_id or str(body.get("id") or "R1")
        r_type = rel_type or str(body.get("type") or body.get("relation_type") or "RELATED")

        src = str(body.get("source") or body.get("source_id") or body.get("from") or "")
        tgt = str(body.get("target") or body.get("target_id") or body.get("to") or "")
        conf = float(body.get("confidence", 1.0))
        truth = str(body.get("truth", body.get("truth_status", "TRUE"))).upper()

        props = body.get("properties", {})
        if not isinstance(props, dict):
            props = {}

        return RelationBlock(
            id=r_id,
            rel_type=r_type,
            source_id=src,
            target_id=tgt,
            confidence=conf,
            truth_status=truth,
            properties=props,
        )

    def _parse_block_body(self) -> Dict[str, Any]:
        """Parses { key: value, ... } block body."""
        self._consume(TokenType.LBRACE, "Expected '{' to start block body")
        pairs: Dict[str, Any] = {}

        while not self._match(TokenType.RBRACE) and not self._match(TokenType.EOF):
            tok = self._current()
            if tok.type not in (TokenType.IDENT, TokenType.STRING):
                raise RecordDSLSyntaxError(
                    f"Expected attribute key, got {tok.type.name} ({tok.value!r})",
                    tok.line,
                    tok.col,
                )

            key = str(self._advance().value).lower()
            self._consume(TokenType.COLON, f"Expected ':' after key '{key}'")
            val = self._parse_value()
            pairs[key] = val

            # Optional comma separator
            if self._match(TokenType.COMMA):
                self._advance()

        self._consume(TokenType.RBRACE, "Expected '}' to terminate block body")
        return pairs

    def _parse_value(self) -> Any:
        tok = self._current()

        if tok.type == TokenType.STRING:
            self._advance()
            return tok.value

        if tok.type == TokenType.NUMBER:
            self._advance()
            return tok.value

        if tok.type == TokenType.IDENT:
            self._advance()
            ident = str(tok.value)
            if ident.lower() == "true":
                return True
            if ident.lower() == "false":
                return False
            if ident.lower() in ("null", "none", "nil"):
                return None
            return ident

        if tok.type == TokenType.LBRACKET:
            return self._parse_list()

        if tok.type == TokenType.LBRACE:
            return self._parse_map()

        raise RecordDSLSyntaxError(f"Unexpected token for value: {tok.value!r}", tok.line, tok.col)

    def _parse_list(self) -> List[Any]:
        self._consume(TokenType.LBRACKET, "Expected '['")
        items = []
        while not self._match(TokenType.RBRACKET) and not self._match(TokenType.EOF):
            items.append(self._parse_value())
            if self._match(TokenType.COMMA):
                self._advance()
        self._consume(TokenType.RBRACKET, "Expected ']'")
        return items

    def _parse_map(self) -> Dict[str, Any]:
        self._consume(TokenType.LBRACE, "Expected '{'")
        mapping = {}
        while not self._match(TokenType.RBRACE) and not self._match(TokenType.EOF):
            k_tok = self._advance()
            key = str(k_tok.value)
            self._consume(TokenType.COLON, f"Expected ':' after map key {key!r}")
            mapping[key] = self._parse_value()
            if self._match(TokenType.COMMA):
                self._advance()
        self._consume(TokenType.RBRACE, "Expected '}'")
        return mapping


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

class RecordDSLSerializer:
    """Serializes AST, extraction results, or QuantaGraph into canonical Record DSL text."""

    def serialize(self, doc: RecordDSLDocument) -> str:
        """Serializes a RecordDSLDocument into pristine Record DSL."""
        sections = []

        # 1. Passages
        for p in doc.passages:
            escaped_text = json.dumps(p.text)
            lines = [
                f"passage {p.id} {{",
                f'    doc_id: "{p.doc_id}"',
                f"    span: [{p.char_span[0]}, {p.char_span[1]}]",
                f"    text: {escaped_text}",
            ]
            if p.created_at:
                lines.append(f"    created_at: {p.created_at}")
            lines.append("}")
            sections.append("\n".join(lines))

        # 2. Entities
        for e in doc.entities:
            escaped_name = json.dumps(e.canonical_name)
            lines = [
                f"entity {e.id} {escaped_name} {{",
                f'    category: "{e.category}"',
            ]
            if e.passage_id:
                lines.append(f"    passage: {e.passage_id}")
            if e.char_span:
                lines.append(f"    span: [{e.char_span[0]}, {e.char_span[1]}]")
            if e.aliases:
                aliases_json = json.dumps(e.aliases)
                lines.append(f"    aliases: {aliases_json}")
            if e.salience != 1.0:
                lines.append(f"    salience: {e.salience}")
            if e.confidence != 1.0:
                lines.append(f"    confidence: {e.confidence}")
            if e.properties:
                props_json = json.dumps(e.properties)
                lines.append(f"    properties: {props_json}")
            lines.append("}")
            sections.append("\n".join(lines))

        # 3. Events
        for ev in doc.events:
            escaped_pred = json.dumps(ev.predicate)
            lines = [
                f"event {ev.id} {escaped_pred} {{",
            ]
            if ev.agent_id:
                lines.append(f"    agent: {ev.agent_id}")
            if ev.patient_id:
                lines.append(f"    patient: {ev.patient_id}")
            if ev.theme_id:
                lines.append(f"    theme: {ev.theme_id}")
            if ev.location_id:
                lines.append(f"    location: {ev.location_id}")
            if ev.instrument_id:
                lines.append(f"    instrument: {ev.instrument_id}")
            if ev.passage_id:
                lines.append(f"    passage: {ev.passage_id}")
            if ev.char_span:
                lines.append(f"    span: [{ev.char_span[0]}, {ev.char_span[1]}]")
            if ev.time_interval:
                t0, t1 = ev.time_interval
                lines.append(f"    time: [{json.dumps(t0)}, {json.dumps(t1)}]")
            if ev.tense:
                lines.append(f'    tense: "{ev.tense}"')
            if ev.aspect:
                lines.append(f'    aspect: "{ev.aspect}"')
            if ev.val:
                lines.append(f'    val: "{ev.val}"')
            if ev.modality:
                lines.append(f'    modality: "{ev.modality}"')
            if ev.confidence != 1.0:
                lines.append(f"    confidence: {ev.confidence}")
            if ev.intent:
                lines.append(f'    intent: "{ev.intent}"')
            if ev.epistemic:
                lines.append(f'    epistemic: "{ev.epistemic}"')
            if ev.raw_text:
                lines.append(f"    raw_text: {json.dumps(ev.raw_text)}")
            if ev.arguments:
                lines.append(f"    arguments: {json.dumps(ev.arguments)}")
            lines.append("}")
            sections.append("\n".join(lines))

        # 4. Relations
        for r in doc.relations:
            lines = [
                f"relation {r.id} {{",
                f'    type: "{r.rel_type}"',
                f"    source: {r.source_id}",
                f"    target: {r.target_id}",
            ]
            if r.confidence != 1.0:
                lines.append(f"    confidence: {r.confidence}")
            if r.truth_status != "TRUE":
                lines.append(f'    truth: "{r.truth_status}"')
            if r.properties:
                lines.append(f"    properties: {json.dumps(r.properties)}")
            lines.append("}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) + ("\n" if sections else "")

    def serialize_extraction(
        self,
        extraction: DiscourseExtractionResult,
        passages: Optional[Sequence[PassageRecord]] = None,
    ) -> str:
        """Serializes DiscourseExtractionResult to Record DSL."""
        doc = RecordDSLDocument.from_extraction_result(extraction, passages)
        return self.serialize(doc)

    def serialize_graph(
        self,
        graph: QuantaGraph,
        passage_store: Optional[PassageStore] = None,
    ) -> str:
        """Serializes QuantaGraph into Record DSL text."""
        doc = RecordDSLDocument()

        # 1. Harvest passages
        seen_passages: Set[str] = set()
        for node in graph._node_list:
            if node.passage_id and node.passage_id not in seen_passages:
                seen_passages.add(node.passage_id)
                if passage_store and node.passage_id in passage_store:
                    rec = passage_store.get_passage(node.passage_id)
                    if rec:
                        doc.passages.append(PassageBlock.from_passage_record(rec))
                        continue
                doc.passages.append(
                    PassageBlock(
                        id=node.passage_id,
                        doc_id="doc_graph",
                        char_span=(node.span_start or 0, node.span_end or 0),
                        text="",
                    )
                )

        # 2. Partition nodes into entities and events
        cid_to_name: Dict[str, str] = {}
        entity_count = 0
        event_count = 0

        valency_edges = {
            "VAL_X1_AGENT",
            "VAL_X2_PATIENT",
            "VAL_LOCATION_SLOT",
            "VAL_X5_INSTRUMENT",
            "VAL_CLAUSAL_COMPLEMENT",
        }

        for node in graph._node_list:
            # Check if event: has valency edges, clausal edges, or tense/aspect slots
            is_event = False
            for rel in node.edges.keys():
                if rel in valency_edges:
                    is_event = True
                    break
            if not is_event and hasattr(node, "node_type") and getattr(node, "node_type") == "event":
                is_event = True

            if is_event:
                event_count += 1
                ev_id = f"EV{event_count}"
                cid_to_name[node.cid] = ev_id
            else:
                entity_count += 1
                ent_id = f"E{entity_count}"
                cid_to_name[node.cid] = ent_id

        # 3. Build EntityBlocks & EventBlocks
        for node in graph._node_list:
            node_id = cid_to_name[node.cid]
            span = (node.span_start, node.span_end) if node.span_start is not None and node.span_end is not None else None

            if node_id.startswith("E") and not node_id.startswith("EV"):
                doc.entities.append(
                    EntityBlock(
                        id=node_id,
                        canonical_name=node.anchor or node_id,
                        passage_id=node.passage_id,
                        char_span=span,
                        salience=node.salience,
                        confidence=node.confidence,
                    )
                )
            else:
                agent_id = None
                patient_id = None
                location_id = None
                instrument_id = None

                for rel, targets in node.edges.items():
                    for tgt in targets:
                        tgt_name = cid_to_name.get(tgt)
                        if not tgt_name:
                            continue
                        if rel == "VAL_X1_AGENT":
                            agent_id = tgt_name
                        elif rel == "VAL_X2_PATIENT":
                            patient_id = tgt_name
                        elif rel == "VAL_LOCATION_SLOT":
                            location_id = tgt_name
                        elif rel == "VAL_X5_INSTRUMENT":
                            instrument_id = tgt_name

                t_interval = None
                if node.time_start is not None or node.time_end is not None:
                    t_interval = (node.time_start, node.time_end)

                doc.events.append(
                    EventBlock(
                        id=node_id,
                        predicate=node.anchor or "event",
                        agent_id=agent_id,
                        patient_id=patient_id,
                        location_id=location_id,
                        instrument_id=instrument_id,
                        passage_id=node.passage_id,
                        char_span=span,
                        time_interval=t_interval,
                        val=node.truth_status,
                        confidence=node.confidence,
                        salience=node.salience,
                    )
                )

        # 4. Build RelationBlocks (non-valency edges)
        rel_idx = 0
        for node in graph._node_list:
            src_name = cid_to_name.get(node.cid)
            if not src_name:
                continue
            for rel_type, targets in node.edges.items():
                if rel_type in valency_edges or rel_type == "GRAPH_IS_SUB_EXP":
                    continue
                for tgt in targets:
                    tgt_name = cid_to_name.get(tgt)
                    if tgt_name:
                        rel_idx += 1
                        doc.relations.append(
                            RelationBlock(
                                id=f"R{rel_idx}",
                                rel_type=rel_type,
                                source_id=src_name,
                                target_id=tgt_name,
                                confidence=node.confidence,
                                truth_status=node.truth_status,
                            )
                        )

        return self.serialize(doc)


# Top-level helper functions
def parse_record_dsl(dsl_text: str) -> RecordDSLDocument:
    """Convenience function to parse a Record DSL string."""
    return RecordDSLParser.parse(dsl_text)


def serialize_to_record_dsl(
    obj: Union[RecordDSLDocument, DiscourseExtractionResult, QuantaGraph],
    passage_store: Optional[PassageStore] = None,
) -> str:
    """Convenience function to serialize AST, extraction, or QuantaGraph to Record DSL text."""
    serializer = RecordDSLSerializer()
    if isinstance(obj, RecordDSLDocument):
        return serializer.serialize(obj)
    if isinstance(obj, DiscourseExtractionResult):
        passages = list(passage_store.values()) if passage_store else None
        return serializer.serialize_extraction(obj, passages=passages)
    if isinstance(obj, QuantaGraph):
        return serializer.serialize_graph(obj, passage_store=passage_store)
    raise TypeError(f"Cannot serialize object of type {type(obj)} to Record DSL")


__all__ = [
    "PassageBlock",
    "EntityBlock",
    "EventBlock",
    "RelationBlock",
    "RecordDSLDocument",
    "RecordDSLSyntaxError",
    "RecordDSLLexer",
    "RecordDSLParser",
    "RecordDSLSerializer",
    "parse_record_dsl",
    "serialize_to_record_dsl",
]
