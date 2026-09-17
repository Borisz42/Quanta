"""High-performance S-Expression Lexer, Parser, AST Converter, and Compiler Bridge for QUANTA.

Implements recursive-descent parsing for the formal GBNF-constrained S-expression
representation of Quanta Abstract Syntax Graphs (ASG).

Key components:
1. SExprLexer: Character-level tokenizer with exact line/column tracking and comment stripping.
2. SExprParser: Recursive-descent parser building SExprAtom and SExprList trees with syntax errors.
3. SExprASTConverter: Bidirectional lossless mapper between S-expressions and DiscourseExtractionResult.
4. parse_to_asg: Direct compilation bridge from S-expression string to 1024-D QuantaGraph.
"""

from __future__ import annotations

from enum import Enum, auto
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from core.asg import QuantaGraph
from parser.asg_compiler import ASGCompiler
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)


class SExprTokenType(Enum):
    """Token types for S-expression lexical analysis."""
    LPAREN = auto()
    RPAREN = auto()
    KEYWORD = auto()
    SYMBOL = auto()
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    EOF = auto()


class SExprSyntaxError(Exception):
    """Raised when an S-expression fails lexical or grammatical analysis."""

    def __init__(self, message: str, line: int = 1, column: int = 1):
        formatted = f"SExprSyntaxError at line {line}, col {column}: {message}"
        super().__init__(formatted)
        self.raw_message = message
        self.line = line
        self.column = column


class SExprToken:
    """A lexical token with precise source coordinates."""

    __slots__ = ("type", "value", "line", "column")

    def __init__(self, token_type: SExprTokenType, value: Any, line: int, column: int):
        self.type = token_type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self) -> str:
        return f"SExprToken({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"


class SExprLexer:
    """Character-level lexer for S-expressions with line and column tracking."""

    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.column = 1

    def _peek(self) -> Optional[str]:
        if self.pos < self.length:
            return self.source[self.pos]
        return None

    def _advance(self) -> Optional[str]:
        if self.pos < self.length:
            ch = self.source[self.pos]
            self.pos += 1
            if ch == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            return ch
        return None

    def tokenize(self) -> List[SExprToken]:
        """Tokenize entire input into a list of SExprToken objects terminating in EOF."""
        tokens: List[SExprToken] = []

        while self.pos < self.length:
            ch = self._peek()

            # Skip whitespace
            if ch in (" ", "\t", "\r", "\n"):
                self._advance()
                continue

            # Skip comments: ;; or ; until newline
            if ch == ";":
                while self.pos < self.length and self._peek() != "\n":
                    self._advance()
                continue

            start_line = self.line
            start_col = self.column

            if ch == "(":
                self._advance()
                tokens.append(SExprToken(SExprTokenType.LPAREN, "(", start_line, start_col))
            elif ch == ")":
                self._advance()
                tokens.append(SExprToken(SExprTokenType.RPAREN, ")", start_line, start_col))
            elif ch == '"':
                tokens.append(self._lex_string(start_line, start_col))
            elif ch == ":":
                tokens.append(self._lex_keyword(start_line, start_col))
            else:
                # Number, Boolean, or Symbol
                tokens.append(self._lex_atom(start_line, start_col))

        tokens.append(SExprToken(SExprTokenType.EOF, None, self.line, self.column))
        return tokens

    def _lex_string(self, start_line: int, start_col: int) -> SExprToken:
        """Scan a double-quoted string with escape sequences."""
        self._advance()  # consume opening "
        chars: List[str] = []

        while True:
            if self.pos >= self.length:
                raise SExprSyntaxError(
                    "Unterminated string literal (missing closing quote)",
                    line=start_line,
                    column=start_col,
                )

            ch = self._advance()
            if ch == '"':
                break
            elif ch == "\\":
                if self.pos >= self.length:
                    raise SExprSyntaxError(
                        "Unexpected end of input inside escape sequence",
                        line=self.line,
                        column=self.column,
                    )
                esc = self._advance()
                if esc == '"':
                    chars.append('"')
                elif esc == "\\":
                    chars.append("\\")
                elif esc == "/":
                    chars.append("/")
                elif esc == "b":
                    chars.append("\b")
                elif esc == "f":
                    chars.append("\f")
                elif esc == "n":
                    chars.append("\n")
                elif esc == "r":
                    chars.append("\r")
                elif esc == "t":
                    chars.append("\t")
                elif esc == "u":
                    hex_chars = ""
                    for _ in range(4):
                        if self.pos >= self.length:
                            raise SExprSyntaxError(
                                "Incomplete Unicode escape sequence",
                                line=self.line,
                                column=self.column,
                            )
                        hex_chars += self._advance()
                    try:
                        chars.append(chr(int(hex_chars, 16)))
                    except ValueError:
                        raise SExprSyntaxError(
                            f"Invalid Unicode escape '\\u{hex_chars}'",
                            line=self.line,
                            column=self.column,
                        )
                else:
                    chars.append(esc)
            else:
                chars.append(ch)

        return SExprToken(SExprTokenType.STRING, "".join(chars), start_line, start_col)

    def _lex_keyword(self, start_line: int, start_col: int) -> SExprToken:
        """Scan a keyword token (e.g., :id, :pred, :raw-text)."""
        self._advance()  # consume leading :
        chars: List[str] = []

        while self.pos < self.length:
            ch = self._peek()
            if ch in (" ", "\t", "\r", "\n", "(", ")", ";", '"'):
                break
            chars.append(self._advance())

        if not chars:
            raise SExprSyntaxError("Empty keyword starting with ':'", line=start_line, column=start_col)

        kw = ":" + "".join(chars)
        return SExprToken(SExprTokenType.KEYWORD, kw, start_line, start_col)

    def _lex_atom(self, start_line: int, start_col: int) -> SExprToken:
        """Scan a number, boolean, or symbol."""
        chars: List[str] = []

        while self.pos < self.length:
            ch = self._peek()
            if ch in (" ", "\t", "\r", "\n", "(", ")", ";", '"'):
                break
            chars.append(self._advance())

        raw = "".join(chars)
        if not raw:
            raise SExprSyntaxError("Unexpected character", line=start_line, column=start_col)

        # 1. Check boolean
        if raw in ("TRUE", "true"):
            return SExprToken(SExprTokenType.BOOLEAN, True, start_line, start_col)
        if raw in ("FALSE", "false"):
            return SExprToken(SExprTokenType.BOOLEAN, False, start_line, start_col)

        # 2. Check number
        num_val = self._try_parse_number(raw)
        if num_val is not None:
            return SExprToken(SExprTokenType.NUMBER, num_val, start_line, start_col)

        # 3. Default to Symbol
        return SExprToken(SExprTokenType.SYMBOL, raw, start_line, start_col)

    @staticmethod
    def _try_parse_number(val: str) -> Optional[Union[int, float]]:
        if re.match(r"^-?[0-9]+$", val):
            try:
                return int(val)
            except ValueError:
                return None
        if re.match(r"^-?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?$", val):
            try:
                return float(val)
            except ValueError:
                return None
        return None


class SExprAtom:
    """An atomic S-expression leaf node (symbol, keyword, string, number, or boolean)."""

    __slots__ = ("token", "type", "value", "line", "column")

    def __init__(self, token: SExprToken):
        self.token = token
        self.type = token.type
        self.value = token.value
        self.line = token.line
        self.column = token.column

    def is_symbol(self) -> bool:
        return self.type == SExprTokenType.SYMBOL

    def is_keyword(self) -> bool:
        return self.type == SExprTokenType.KEYWORD

    def is_string(self) -> bool:
        return self.type == SExprTokenType.STRING

    def is_number(self) -> bool:
        return self.type == SExprTokenType.NUMBER

    def is_boolean(self) -> bool:
        return self.type == SExprTokenType.BOOLEAN

    def as_str(self) -> str:
        return str(self.value)

    def as_bool(self) -> bool:
        if isinstance(self.value, bool):
            return self.value
        val_str = str(self.value).strip().lower()
        return val_str in ("true", "1", "t")

    def __repr__(self) -> str:
        return f"SExprAtom({self.value!r})"


class SExprList:
    """A composite S-expression node containing an ordered sequence of child nodes."""

    __slots__ = ("elements", "line", "column")

    def __init__(self, elements: List[Union[SExprAtom, SExprList]], line: int, column: int):
        self.elements = elements
        self.line = line
        self.column = column

    def head_symbol(self) -> Optional[str]:
        """Return the string symbol of the first element if it is an atom."""
        if self.elements and isinstance(self.elements[0], SExprAtom):
            return self.elements[0].as_str()
        return None

    def get_keyword(
        self,
        keywords: Union[str, List[str]],
        default: Any = None,
    ) -> Any:
        """Find the value following any of the specified keyword names.

        Matches keywords case-insensitively, with or without leading colon,
        and supporting hyphens and underscores interchangeably.
        """
        if isinstance(keywords, str):
            candidate_keys = {keywords}
        else:
            candidate_keys = set(keywords)

        norm_candidates = {self._norm_key(k) for k in candidate_keys}

        i = 0
        while i < len(self.elements) - 1:
            elem = self.elements[i]
            if isinstance(elem, SExprAtom) and elem.is_keyword():
                norm = self._norm_key(elem.as_str())
                if norm in norm_candidates:
                    return self.elements[i + 1]
            i += 1

        return default

    @staticmethod
    def _norm_key(k: str) -> str:
        if k.startswith(":"):
            k = k[1:]
        return k.lower().replace("-", "_")

    def to_plist_dict(self) -> Dict[str, Any]:
        """Convert a list of alternating :key value items into a dictionary."""
        result: Dict[str, Any] = {}
        i = 0
        if self.elements and isinstance(self.elements[0], SExprAtom) and not self.elements[0].is_keyword():
            i = 1
        while i < len(self.elements) - 1:
            k_elem = self.elements[i]
            v_elem = self.elements[i + 1]
            if isinstance(k_elem, SExprAtom) and k_elem.is_keyword():
                k = k_elem.as_str()
                if k.startswith(":"):
                    k = k[1:]
                if isinstance(v_elem, SExprAtom):
                    result[k] = v_elem.value
                elif isinstance(v_elem, SExprList):
                    result[k] = v_elem.to_plist_dict()
                i += 2
            else:
                i += 1
        return result

    def __repr__(self) -> str:
        return f"SExprList({self.elements!r})"


class SExprParser:
    """Recursive-descent parser producing SExprAtom and SExprList AST trees."""

    def __init__(self, tokens: List[SExprToken]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> SExprToken:
        return self.tokens[self.pos]

    def _advance(self) -> SExprToken:
        tok = self.tokens[self.pos]
        if tok.type != SExprTokenType.EOF:
            self.pos += 1
        return tok

    def parse(self) -> SExprList:
        """Parse the input token stream into a single root SExprList.

        Raises:
            SExprSyntaxError: If input is empty, unclosed, or contains trailing tokens.
        """
        tok = self._peek()
        if tok.type == SExprTokenType.EOF:
            raise SExprSyntaxError("Empty S-expression input", line=tok.line, column=tok.column)

        node = self._parse_node()

        # Check for unexpected trailing tokens after root
        remaining = self._peek()
        if remaining.type != SExprTokenType.EOF:
            raise SExprSyntaxError(
                f"Unexpected token {remaining.value!r} after root S-expression",
                line=remaining.line,
                column=remaining.column,
            )

        if not isinstance(node, SExprList):
            raise SExprSyntaxError(
                f"Root S-expression must be a parenthesized list, got {type(node).__name__}",
                line=node.line,
                column=node.column,
            )

        return node

    def _parse_node(self) -> Union[SExprAtom, SExprList]:
        tok = self._peek()

        if tok.type == SExprTokenType.LPAREN:
            return self._parse_list()
        elif tok.type == SExprTokenType.RPAREN:
            raise SExprSyntaxError(
                "Unexpected closing parenthesis ')' without matching opening parenthesis",
                line=tok.line,
                column=tok.column,
            )
        elif tok.type == SExprTokenType.EOF:
            raise SExprSyntaxError("Unexpected end of input", line=tok.line, column=tok.column)
        else:
            self._advance()
            return SExprAtom(tok)

    def _parse_list(self) -> SExprList:
        open_tok = self._advance()  # consume '('
        open_line = open_tok.line
        open_col = open_tok.column
        elements: List[Union[SExprAtom, SExprList]] = []

        while True:
            tok = self._peek()
            if tok.type == SExprTokenType.EOF:
                raise SExprSyntaxError(
                    f"Unclosed parenthesis (opened at line {open_line}, col {open_col})",
                    line=open_line,
                    column=open_col,
                )
            if tok.type == SExprTokenType.RPAREN:
                self._advance()  # consume ')'
                break
            elements.append(self._parse_node())

        return SExprList(elements, open_line, open_col)


class SExprASTConverter:
    """Bidirectional converter between SExpr AST and DiscourseExtractionResult."""

    @classmethod
    def from_ast(cls, root: SExprList) -> DiscourseExtractionResult:
        """Convert an AST SExprList (graph ...) into a typed DiscourseExtractionResult."""
        head = root.head_symbol()
        if head != "graph":
            raise SExprSyntaxError(
                f"Root S-expression must begin with 'graph', got {head!r}",
                line=root.line,
                column=root.column,
            )

        chunk_id_node = root.get_keyword([":chunk-id", ":chunk_id", ":id"])
        chunk_id: Optional[str] = None
        if isinstance(chunk_id_node, SExprAtom):
            chunk_id = chunk_id_node.as_str()

        metadata_node = root.get_keyword([":metadata", ":meta"])
        metadata: Dict[str, Any] = {}
        if isinstance(metadata_node, SExprList):
            metadata = metadata_node.to_plist_dict()

        entities: List[ExtractedEntity] = []
        events: List[ExtractedEvent] = []
        relations: List[ExtractedRelation] = []
        propositions: List[ExtractedProposition] = []

        # Iterate over child clauses
        # Skip the leading 'graph' symbol and any immediate root keyword arguments
        i = 1
        while i < len(root.elements):
            elem = root.elements[i]

            # If it is a root keyword argument pair, skip both
            if isinstance(elem, SExprAtom) and elem.is_keyword():
                i += 2
                continue

            if isinstance(elem, SExprList):
                clause_head = elem.head_symbol()
                if clause_head == "entity":
                    entities.append(cls._convert_entity(elem))
                elif clause_head == "event":
                    events.append(cls._convert_event(elem))
                elif clause_head == "relation":
                    relations.append(cls._convert_relation(elem))
                elif clause_head in ("proposition", "prop"):
                    propositions.append(cls._convert_proposition(elem))
                else:
                    # Ignore unrecognized clause heads gracefully
                    pass
            i += 1

        return DiscourseExtractionResult(
            chunk_id=chunk_id,
            entities=entities,
            events=events,
            relations=relations,
            propositions=propositions,
            metadata=metadata,
        )

    @classmethod
    def _convert_entity(cls, node: SExprList) -> ExtractedEntity:
        # :id
        id_node = node.get_keyword([":id"])
        ent_id = id_node.as_str() if isinstance(id_node, SExprAtom) else ""

        # :label / :canonical_name
        label_node = node.get_keyword([":label", ":name", ":canonical_name", ":canonical-name"])
        canonical_name = label_node.as_str() if isinstance(label_node, SExprAtom) else ""

        # :type / :category
        type_node = node.get_keyword([":type", ":category"])
        category = type_node.as_str().upper() if isinstance(type_node, SExprAtom) else "OBJECT"

        # :surface / :aliases
        surface_node = node.get_keyword([":surface", ":aliases", ":alias"])
        surface_aliases: List[str] = []
        if isinstance(surface_node, SExprAtom):
            surface_aliases = [surface_node.as_str()]
        elif isinstance(surface_node, SExprList):
            surface_aliases = [
                child.as_str()
                for child in surface_node.elements
                if isinstance(child, SExprAtom)
            ]

        # :props
        props_node = node.get_keyword([":props", ":properties"])
        properties: Dict[str, Any] = {}
        if isinstance(props_node, SExprList):
            properties = props_node.to_plist_dict()

        return ExtractedEntity(
            id=ent_id,
            canonical_name=canonical_name,
            category=category,
            surface_aliases=surface_aliases,
            properties=properties,
        )

    @classmethod
    def _convert_event(cls, node: SExprList) -> ExtractedEvent:
        # :id
        id_node = node.get_keyword([":id", ":event_id", ":event-id"])
        ev_id = id_node.as_str() if isinstance(id_node, SExprAtom) else ""

        # :pred / :predicate
        pred_node = node.get_keyword([":pred", ":predicate", ":action", ":verb"])
        predicate = pred_node.as_str().lower() if isinstance(pred_node, SExprAtom) else ""

        def extract_fk(keys: List[str]) -> Optional[str]:
            v = node.get_keyword(keys)
            if isinstance(v, SExprAtom):
                val_str = v.as_str()
                if val_str.lower() in ("nil", "none", "null", ""):
                    return None
                return val_str
            return None

        agent_id = extract_fk([":agent", ":agent_id", ":agent-id"])
        patient_id = extract_fk([":patient", ":patient_id", ":patient-id"])
        theme_id = extract_fk([":theme", ":theme_id", ":theme-id"])
        location_id = extract_fk([":location", ":location_id", ":location-id"])
        instrument_id = extract_fk([":instrument", ":instrument_id", ":instrument-id"])

        time_node = node.get_keyword([":time", ":temporal_anchor", ":temporal-anchor"])
        temporal_anchor: Optional[str] = None
        time_start: Optional[Union[int, float]] = None
        time_end: Optional[Union[int, float]] = None

        if isinstance(time_node, SExprList):
            # Form: (interval :start 2018 :end 2020)
            plist = time_node.to_plist_dict()
            s_val = plist.get("start")
            e_val = plist.get("end")
            if s_val is not None and str(s_val).lower() not in ("nil", "none", "null"):
                try:
                    f_val = float(s_val)
                    time_start = int(f_val) if f_val.is_integer() else f_val
                except (ValueError, TypeError):
                    time_start = None
            if e_val is not None and str(e_val).lower() not in ("nil", "none", "null"):
                try:
                    f_val = float(e_val)
                    time_end = int(f_val) if f_val.is_integer() else f_val
                except (ValueError, TypeError):
                    time_end = None
            temporal_anchor = f"interval:{time_start}-{time_end}"
        elif isinstance(time_node, SExprAtom):
            temporal_anchor = time_node.as_str()
            if temporal_anchor and temporal_anchor.lower() in ("nil", "none", "null"):
                temporal_anchor = None
            elif temporal_anchor:
                # Could be a single number like "2018"
                try:
                    f_val = float(temporal_anchor)
                    time_start = int(f_val) if f_val.is_integer() else f_val
                except (ValueError, TypeError):
                    pass

        tense_node = node.get_keyword([":tense"])
        tense = tense_node.as_str().upper() if isinstance(tense_node, SExprAtom) else "PAST"

        aspect_node = node.get_keyword([":aspect"])
        aspect = aspect_node.as_str().upper() if isinstance(aspect_node, SExprAtom) else "SIMPLE"

        pol_node = node.get_keyword([":polarity", ":val"])
        if isinstance(pol_node, SExprAtom):
            polarity = pol_node.as_bool()
        else:
            polarity = True

        mod_node = node.get_keyword([":modality"])
        modality = mod_node.as_str() if isinstance(mod_node, SExprAtom) else None
        if modality and modality.lower() in ("nil", "none", "null"):
            modality = None

        raw_node = node.get_keyword([":raw-text", ":raw_text", ":raw", ":text"])
        raw_text = raw_node.as_str() if isinstance(raw_node, SExprAtom) else None
        if raw_text and raw_text.lower() in ("nil", "none", "null"):
            raw_text = None

        args_node = node.get_keyword([":args", ":arguments"])
        arguments: Dict[str, Any] = {}
        if isinstance(args_node, SExprList):
            arguments = args_node.to_plist_dict()

        return ExtractedEvent(
            id=ev_id,
            predicate=predicate,
            agent_id=agent_id,
            patient_id=patient_id,
            theme_id=theme_id,
            location_id=location_id,
            instrument_id=instrument_id,
            temporal_anchor=temporal_anchor,
            time_start=time_start,
            time_end=time_end,
            tense=tense,
            aspect=aspect,
            polarity=polarity,
            modality=modality,
            raw_text=raw_text,
            arguments=arguments,
        )

    @classmethod
    def _convert_relation(cls, node: SExprList) -> ExtractedRelation:
        type_node = node.get_keyword([":type", ":relation_type", ":relation-type"])
        relation_type = type_node.as_str() if isinstance(type_node, SExprAtom) else "TEMP_ALLEN_AFTER"

        src_node = node.get_keyword([":source", ":source_id", ":source-id"])
        source_id = src_node.as_str() if isinstance(src_node, SExprAtom) else ""

        tgt_node = node.get_keyword([":target", ":target_id", ":target-id"])
        target_id = tgt_node.as_str() if isinstance(tgt_node, SExprAtom) else ""

        mech_node = node.get_keyword([":mechanism", ":description"])
        mechanism = mech_node.as_str() if isinstance(mech_node, SExprAtom) else None
        if mechanism and mechanism.lower() in ("nil", "none", "null"):
            mechanism = None

        conf_node = node.get_keyword([":confidence"])
        confidence = 1.0
        if isinstance(conf_node, SExprAtom):
            try:
                confidence = float(conf_node.value)
            except (ValueError, TypeError):
                confidence = 1.0

        return ExtractedRelation(
            relation_type=relation_type,
            source_id=source_id,
            target_id=target_id,
            mechanism=mechanism,
            confidence=confidence,
        )

    @classmethod
    def _convert_proposition(cls, node: SExprList) -> ExtractedProposition:
        id_node = node.get_keyword([":id", ":prop_id", ":prop-id"])
        prop_id = id_node.as_str() if isinstance(id_node, SExprAtom) else ""

        claim_node = node.get_keyword([":claim", ":claim_text", ":claim-text", ":text"])
        claim_text = claim_node.as_str() if isinstance(claim_node, SExprAtom) else ""

        status_node = node.get_keyword([":status", ":epistemic_status", ":epistemic-status"])
        epistemic_status = status_node.as_str().upper() if isinstance(status_node, SExprAtom) else "FACT"

        def extract_opt_str(keys: List[str]) -> Optional[str]:
            v = node.get_keyword(keys)
            if isinstance(v, SExprAtom):
                val = v.as_str()
                if val.lower() in ("nil", "none", "null", ""):
                    return None
                return val
            return None

        source_agent_id = extract_opt_str([":source", ":source_agent_id", ":source-agent-id", ":source_agent"])
        subject_id = extract_opt_str([":subject", ":subject_id", ":subject-id"])
        event_id = extract_opt_str([":event", ":event_id", ":event-id"])
        predicate = extract_opt_str([":pred", ":predicate"])

        props_node = node.get_keyword([":props", ":properties"])
        properties: Dict[str, Any] = {}
        if isinstance(props_node, SExprList):
            properties = props_node.to_plist_dict()

        return ExtractedProposition(
            id=prop_id,
            claim_text=claim_text,
            predicate=predicate,
            subject_id=subject_id,
            epistemic_status=epistemic_status,
            source_agent_id=source_agent_id,
            event_id=event_id,
            properties=properties,
        )

    @classmethod
    def to_sexpr(cls, result: DiscourseExtractionResult, pretty: bool = True) -> str:
        """Serialize a DiscourseExtractionResult into a canonical S-expression string."""
        lines: List[str] = []
        indent = "  " if pretty else ""
        nl = "\n" if pretty else " "

        root_parts = ["(graph"]
        if result.chunk_id:
            root_parts.append(f':chunk-id {json.dumps(result.chunk_id)}')
        if result.metadata:
            meta_str = cls._format_plist(result.metadata)
            root_parts.append(f":metadata {meta_str}")

        lines.append(" ".join(root_parts))

        # 1. Entities
        for ent in result.entities:
            parts = [
                "(entity",
                f":id {ent.id}",
                f":type {ent.category}",
                f":label {json.dumps(ent.canonical_name)}",
            ]
            if len(ent.surface_aliases) == 1:
                parts.append(f":surface {json.dumps(ent.surface_aliases[0])}")
            elif len(ent.surface_aliases) > 1:
                aliases_str = " ".join(json.dumps(a) for a in ent.surface_aliases)
                parts.append(f":surface ({aliases_str})")
            else:
                parts.append(":surface ()")

            if ent.properties:
                parts.append(f":props {cls._format_plist(ent.properties)}")
            parts.append(")")
            lines.append(f"{indent}{' '.join(parts)}")

        # 2. Events
        for ev in result.events:
            parts = [
                "(event",
                f":id {ev.id}",
                f":pred {ev.predicate}",
            ]
            if ev.agent_id:
                parts.append(f":agent {ev.agent_id}")
            if ev.patient_id:
                parts.append(f":patient {ev.patient_id}")
            if ev.theme_id:
                parts.append(f":theme {ev.theme_id}")
            if ev.location_id:
                parts.append(f":location {ev.location_id}")
            if ev.instrument_id:
                parts.append(f":instrument {ev.instrument_id}")
            if ev.time_start is not None or ev.time_end is not None:
                s_str = str(ev.time_start) if ev.time_start is not None else "nil"
                e_str = str(ev.time_end) if ev.time_end is not None else "nil"
                parts.append(f":time (interval :start {s_str} :end {e_str})")
            elif ev.temporal_anchor:
                parts.append(f":time {json.dumps(ev.temporal_anchor)}")
            parts.append(f":tense {ev.tense}")
            if ev.aspect and ev.aspect != "SIMPLE":
                parts.append(f":aspect {ev.aspect}")
            parts.append(f":polarity {'TRUE' if ev.polarity else 'FALSE'}")
            if ev.modality:
                parts.append(f":modality {json.dumps(ev.modality)}")
            if ev.raw_text:
                parts.append(f":raw-text {json.dumps(ev.raw_text)}")
            if ev.arguments:
                parts.append(f":args {cls._format_plist(ev.arguments)}")
            parts.append(")")
            lines.append(f"{indent}{' '.join(parts)}")

        # 3. Relations
        for rel in result.relations:
            parts = [
                "(relation",
                f":type {rel.relation_type}",
                f":source {rel.source_id}",
                f":target {rel.target_id}",
            ]
            if rel.mechanism:
                parts.append(f":mechanism {json.dumps(rel.mechanism)}")
            if rel.confidence != 1.0:
                parts.append(f":confidence {rel.confidence:.4f}".rstrip("0").rstrip("."))
            parts.append(")")
            lines.append(f"{indent}{' '.join(parts)}")

        # 4. Propositions
        for p in result.propositions:
            parts = [
                "(proposition",
                f":id {p.id}",
                f":claim {json.dumps(p.claim_text)}",
            ]
            if p.subject_id:
                parts.append(f":subject {p.subject_id}")
            parts.append(f":status {p.epistemic_status}")
            if p.source_agent_id:
                parts.append(f":source {p.source_agent_id}")
            if p.event_id:
                parts.append(f":event {p.event_id}")
            if p.predicate:
                parts.append(f":pred {p.predicate}")
            if p.properties:
                parts.append(f":props {cls._format_plist(p.properties)}")
            parts.append(")")
            lines.append(f"{indent}{' '.join(parts)}")

        lines.append(")")
        return nl.join(lines)

    @classmethod
    def _format_plist(cls, d: Dict[str, Any]) -> str:
        """Format a python dict into a Lisp plist S-expression string."""
        pairs: List[str] = []
        for k, v in d.items():
            kw = f":{k}" if not k.startswith(":") else k
            if isinstance(v, str):
                pairs.append(f"{kw} {json.dumps(v)}")
            elif isinstance(v, bool):
                pairs.append(f"{kw} {'TRUE' if v else 'FALSE'}")
            elif isinstance(v, (int, float)):
                pairs.append(f"{kw} {v}")
            elif isinstance(v, dict):
                pairs.append(f"{kw} {cls._format_plist(v)}")
            else:
                pairs.append(f"{kw} {json.dumps(str(v))}")
        return f"({' '.join(pairs)})"


def parse_sexpr(sexpr_str: str) -> DiscourseExtractionResult:
    """Parse an S-expression string into a typed DiscourseExtractionResult."""
    lexer = SExprLexer(sexpr_str)
    tokens = lexer.tokenize()
    parser = SExprParser(tokens)
    ast = parser.parse()
    return SExprASTConverter.from_ast(ast)


def to_sexpr(result: DiscourseExtractionResult, pretty: bool = True) -> str:
    """Serialize a DiscourseExtractionResult into a canonical S-expression string."""
    return SExprASTConverter.to_sexpr(result, pretty=pretty)


def parse_to_asg(
    sexpr_str: str,
    compiler: Optional[ASGCompiler] = None,
    validate: bool = True,
) -> QuantaGraph:
    """Parse an S-expression string and compile it directly into a QuantaGraph.

    Args:
        sexpr_str: The S-expression string to parse.
        compiler: Optional ASGCompiler instance. Defaults to a new ASGCompiler().
        validate: Whether to execute Clingo ASP validation on the compiled graph.

    Returns:
        Compiled, content-addressed 1024-D QuantaGraph.

    Raises:
        SExprSyntaxError: If the S-expression has syntax or structural errors.
        ASGCompilationError: If foreign keys or Clingo validation constraints fail.
    """
    extraction_result = parse_sexpr(sexpr_str)
    if compiler is None:
        compiler = ASGCompiler()
    return compiler.compile(extraction_result, validate=validate)
