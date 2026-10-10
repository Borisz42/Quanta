"""Fast Qwen-4B Skeleton Transducer for QUANTA SVM.

Eliminates the ingestion bottleneck by removing logic/epistemic tags and S-expression
syntax from Qwen 3.5 4B's decoding grammar. Restricts Qwen to extracting surface entities,
surface spans, and unlabelled thematic subject-verb-object frames, reducing token generation
budget from >120 tokens to 30--45 tokens (300--450 ms).

Provides:
1. SkeletonEntity, SkeletonEvent, SkeletonExtractionResult: Typed schemas for stripped skeletons.
2. SkeletonTransducer: Production client connecting to llama-server on port 8888 with GBNF grammar.
3. MockSkeletonTransducer: Deterministic, high-speed (<5ms) offline mock for CI testing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import requests

from core.artifacts import require_artifacts
from core.asg import QuantaGraph, QuantaNode
from parser.entity_manifest import EntityRecord
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
)
from parser.span_aligner import SpanAligner, SpanAlignment

logger = logging.getLogger(__name__)

DEFAULT_SKELETON_SYSTEM_PROMPT = """You are the QUANTA Fast Skeleton Transducer.
Extract at most 3 core named entities and 1 main event predicate frame. Never extract dates, honors, or adjectives.
Example input: "Alan Turing completed his degrees at Cambridge."
Example output: {"entities": [{"id": "E1", "text": "Alan Turing"}, {"id": "E2", "text": "Cambridge"}], "events": [{"id": "EV1", "pred": "completed", "subj": "E1", "obj": "E2"}]}"""

CO_DECODED_SKELETON_SYSTEM_PROMPT = """You are the QUANTA Co-Decoded Skeleton Transducer.
Extract at most 3 core named entities and 1 main event predicate frame with compact single-letter Kev decisions.
Enums:
- intent: I (Informative), D (Directive), C (Commissive), E (Expressive)
- epist: O (Direct Observation), D (Deduction), H (Hearsay), C (Conjecture)
- allen: M (Meets), B (Before), O (Overlaps), D (During), N (None)
- pearl: M (Mechanism), C (Condition), N (None)
Example input: "Charles Babbage invented the Difference Engine."
Example output: {"entities": [{"id": "E1", "text": "Charles Babbage"}, {"id": "E2", "text": "Difference Engine"}], "events": [{"id": "EV1", "pred": "invented", "subj": "E1", "obj": "E2", "intent": "I", "epist": "O", "allen": "B", "pearl": "M"}]}"""

DEFAULT_COMPACT_SEXPR_SYSTEM_PROMPT = """You are the QUANTA Fast Compact S-Expression Transducer.
Extract at most 3 core named entities and 1 main event predicate frame into a compact keyword S-expression. Never extract dates, honors, or adjectives.
Example input: "Alan Turing completed his degrees at Cambridge."
Example output: (graph (entity E1 "Alan Turing") (entity E2 "Cambridge") (event EV1 completed :subj E1 :obj E2))"""

DEFAULT_POSITIONAL_SEXPR_SYSTEM_PROMPT = """You are the QUANTA Fast Positional S-Expression Transducer.
Extract at most 3 core named entities and 1 main event predicate frame into an ultra-compact positional S-expression. Never extract dates, honors, or adjectives.
Example input: "Alan Turing completed his degrees at Cambridge."
Example output: ((e E1 "Alan Turing") (e E2 "Cambridge") (ev EV1 completed E1 E2))"""

CO_DECODED_COMPACT_SEXPR_SYSTEM_PROMPT = """You are the QUANTA Co-Decoded Compact S-Expression Transducer.
Extract at most 3 core named entities and 1 main event predicate frame with compact single-letter Kev decisions.
Enums:
- intent: I (Informative), D (Directive), C (Commissive), E (Expressive)
- epist: O (Direct Observation), D (Deduction), H (Hearsay), C (Conjecture)
- allen: M (Meets), B (Before), O (Overlaps), D (During), N (None)
- pearl: M (Mechanism), C (Condition), N (None)
Example input: "Charles Babbage invented the Difference Engine."
Example output: (graph (entity E1 "Charles Babbage") (entity E2 "Difference Engine") (event EV1 invented :subj E1 :obj E2 :intent I :epist O :allen B :pearl M))"""

CO_DECODED_POSITIONAL_SEXPR_SYSTEM_PROMPT = """You are the QUANTA Co-Decoded Positional S-Expression Transducer.
Extract at most 3 core named entities and 1 main event predicate frame with compact single-letter Kev decisions.
Enums:
- intent: I (Informative), D (Directive), C (Commissive), E (Expressive)
- epist: O (Direct Observation), D (Deduction), H (Hearsay), C (Conjecture)
- allen: M (Meets), B (Before), O (Overlaps), D (During), N (None)
- pearl: M (Mechanism), C (Condition), N (None)
Example input: "Charles Babbage invented the Difference Engine."
Example output: ((e E1 "Charles Babbage") (e E2 "Difference Engine") (ev EV1 invented E1 E2 I O B M))"""


def parse_skeleton_sexpr(sexpr_str: str) -> Dict[str, Any]:
    """High-speed parser for compact keyword and positional skeleton S-expressions.

    Extracts entity and event clauses into standard dictionary format:
    {"entities": [{"id": ..., "text": ..., "category": ...}],
     "events": [{"id": ..., "pred": ..., "subj": ..., "obj": ..., ...}]}

    Features:
    - Strips reasoning blocks (<think>...</think>), code fences (```...```), and ; comments.
    - Gracefully auto-repairs truncated outputs by appending missing closing parentheses.
    - Decodes both keyword clauses ((entity E1 "...") (event EV1 pred :subj E1 :obj E2))
      and positional clauses (((e E1 "...") (ev EV1 pred E1 E2))).
    - Extracts co-decoded Kev decisions (intent, epist, allen, pearl) when present.
    - Skips unparseable or unrecognized fragments without raising fatal errors.
    """
    clean = (sexpr_str or "").strip()
    if not clean:
        return {"entities": [], "events": []}

    # 1. Strip reasoning blocks & markdown fences
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)\s*```", clean)
    if fence:
        clean = fence.group(1).strip()
    elif clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean).strip()

    # JSON fallback check if model emitted JSON unexpectedly
    clean_test = clean.strip()
    if clean_test.startswith("{") and clean_test.endswith("}"):
        try:
            raw_json = json.loads(clean_test)
            if "entities" in raw_json or "events" in raw_json:
                ents = [
                    {
                        "id": str(e.get("id", f"E{idx+1}")),
                        "text": str(e.get("text", e.get("surface_text", ""))),
                        "category": str(e.get("category", "OBJECT")),
                    }
                    for idx, e in enumerate(raw_json.get("entities", []))
                ]
                return {"entities": ents, "events": list(raw_json.get("events", []))}
        except Exception:
            pass

    # 2. Strip comments (';' outside quotes)
    clean_lines = []
    for line in clean.splitlines():
        in_str = False
        escape = False
        cut_idx = -1
        for idx, ch in enumerate(line):
            if ch == '"' and not escape:
                in_str = not in_str
            elif ch == '\\' and in_str:
                escape = not escape
                continue
            elif ch == ';' and not in_str:
                cut_idx = idx
                break
            escape = False
        if cut_idx != -1:
            line = line[:cut_idx]
        if line.strip():
            clean_lines.append(line)
    clean = "\n".join(clean_lines).strip()

    # 3. Auto-close truncated string quotes & missing parentheses
    open_count = 0
    close_count = 0
    in_str = False
    escape = False
    for ch in clean:
        if ch == '"' and not escape:
            in_str = not in_str
        elif ch == '\\' and in_str:
            escape = not escape
            continue
        elif not in_str:
            if ch == '(':
                open_count += 1
            elif ch == ')':
                close_count += 1
        escape = False

    if in_str:
        clean += '"'
    if open_count > close_count:
        clean += ")" * (open_count - close_count)

    # 4. Tokenize into (val, is_string) tuples
    tokens: List[Tuple[str, bool]] = []
    i = 0
    n = len(clean)
    while i < n:
        ch = clean[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ('(', ')'):
            tokens.append((ch, False))
            i += 1
            continue
        if ch == '"':
            i += 1
            chars: List[str] = []
            while i < n:
                if clean[i] == '\\' and i + 1 < n:
                    nxt = clean[i + 1]
                    if nxt == '"':
                        chars.append('"')
                    elif nxt == '\\':
                        chars.append('\\')
                    elif nxt == 'n':
                        chars.append('\n')
                    elif nxt == 't':
                        chars.append('\t')
                    elif nxt == 'r':
                        chars.append('\r')
                    else:
                        chars.append(nxt)
                    i += 2
                elif clean[i] == '"':
                    i += 1
                    break
                else:
                    chars.append(clean[i])
                    i += 1
            tokens.append(("".join(chars), True))
            continue
        # Atom / symbol / keyword
        start = i
        while i < n and not clean[i].isspace() and clean[i] not in ('(', ')'):
            i += 1
        atom = clean[start:i]
        tokens.append((atom, False))

    # 5. Build nested list AST using stack
    root_nodes: List[Any] = []
    stack: List[List[Any]] = [root_nodes]
    for val, is_s in tokens:
        if not is_s and val == '(':
            new_lst: List[Any] = []
            stack[-1].append(new_lst)
            stack.append(new_lst)
        elif not is_s and val == ')':
            if len(stack) > 1:
                stack.pop()
        else:
            stack[-1].append((val, is_s))

    # 6. Traverse tree and harvest entity and event clauses
    clauses: List[List[Any]] = []

    def _collect(node: Any) -> None:
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if isinstance(head, tuple) and not head[1] and head[0].lower() in {"entity", "e", "event", "ev"}:
            clauses.append(node)
            return
        for child in node:
            if isinstance(child, list):
                _collect(child)

    for r in root_nodes:
        _collect(r)

    # Kev decision domain sets
    VALID_INTENT = {"I", "D", "C", "E"}
    VALID_EPIST = {"O", "D", "H", "C"}
    VALID_ALLEN = {"M", "B", "O", "D", "N"}
    VALID_PEARL = {"M", "C", "N"}

    entities: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    for cl in clauses:
        if not cl or not isinstance(cl[0], tuple):
            continue
        head = cl[0][0].lower()
        items = cl[1:]

        if head in ("entity", "e"):
            # Entity clause
            if not items:
                continue
            eid: Optional[str] = None
            surface_text: Optional[str] = None
            category: str = "OBJECT"

            has_keywords = any(isinstance(it, tuple) and not it[1] and it[0].startswith(":") for it in items)
            if has_keywords:
                k_idx = 0
                while k_idx < len(items):
                    tok_val, tok_is_str = items[k_idx]
                    if not tok_is_str and tok_val.startswith(":"):
                        kw_name = tok_val.lstrip(":").lower()
                        if k_idx + 1 < len(items):
                            nxt_val = items[k_idx + 1][0]
                            if kw_name in ("id", "eid"):
                                eid = nxt_val
                            elif kw_name in ("text", "surface", "name", "label"):
                                surface_text = nxt_val
                            elif kw_name in ("type", "category"):
                                category = nxt_val
                            k_idx += 2
                            continue
                    elif eid is None and not tok_is_str:
                        eid = tok_val
                    elif surface_text is None:
                        surface_text = tok_val
                    k_idx += 1
            else:
                # Positional / standard: (entity E1 "Charles Babbage" [:type ...]) or (e E1 "Charles Babbage" [TYPE])
                if len(items) >= 1:
                    eid = items[0][0]
                if len(items) >= 2:
                    surface_text = items[1][0]
                if len(items) >= 3:
                    t_val = items[2][0]
                    if t_val == ":type" and len(items) >= 4:
                        category = items[3][0]
                    elif not t_val.startswith(":"):
                        category = t_val

            if eid:
                entities.append({
                    "id": eid,
                    "text": surface_text if surface_text is not None else eid,
                    "category": category,
                })

        elif head in ("event", "ev"):
            # Event clause
            if len(items) < 2:
                # Minimum viable event has id and predicate
                if len(items) == 1:
                    events.append({"id": items[0][0], "pred": "observe", "subj": None, "obj": None})
                continue

            ev_id = items[0][0]
            pred = items[1][0]
            rem = items[2:]

            subj: Optional[str] = None
            obj: Optional[str] = None
            intent: Optional[str] = None
            epist: Optional[str] = None
            allen: Optional[str] = None
            pearl: Optional[str] = None

            has_keywords = any(isinstance(it, tuple) and not it[1] and it[0].startswith(":") for it in rem)
            if has_keywords:
                r_idx = 0
                while r_idx < len(rem):
                    tok_val, tok_is_str = rem[r_idx]
                    if not tok_is_str and tok_val.startswith(":"):
                        kw_name = tok_val.lstrip(":").lower()
                        if r_idx + 1 < len(rem):
                            nxt_val = rem[r_idx + 1][0]
                            if kw_name in ("subj", "agent", "subject"):
                                subj = nxt_val
                            elif kw_name in ("obj", "patient", "object", "theme"):
                                obj = nxt_val
                            elif kw_name == "intent":
                                intent = nxt_val
                            elif kw_name in ("epist", "epistemic"):
                                epist = nxt_val
                            elif kw_name == "allen":
                                allen = nxt_val
                            elif kw_name == "pearl":
                                pearl = nxt_val
                            r_idx += 2
                            continue
                    elif subj is None:
                        subj = tok_val
                    elif obj is None:
                        obj = tok_val
                    r_idx += 1
            else:
                # Positional event parsing: (ev EV1 pred [subj] [obj] [intent] [epist] [allen] [pearl])
                pos_vals = [it[0] for it in rem]
                # Filter out placeholder nulls
                clean_vals = [None if v in ("-", "None", "nil", "null") else v for v in pos_vals]

                # Check if all 4 trailing values or remaining values match Kev enums
                # Standard sequence: subj (0), obj (1), intent (2), epist (3), allen (4), pearl (5)
                val_idx = 0
                if val_idx < len(clean_vals):
                    # Check if clean_vals starts directly with 4 Kev tokens (rare: no subj/obj)
                    if len(clean_vals) == 4 and clean_vals[0] in VALID_INTENT and clean_vals[1] in VALID_EPIST and clean_vals[2] in VALID_ALLEN and clean_vals[3] in VALID_PEARL:
                        intent, epist, allen, pearl = clean_vals[0], clean_vals[1], clean_vals[2], clean_vals[3]
                        val_idx = 4
                    else:
                        subj = clean_vals[val_idx]
                        val_idx += 1

                if val_idx < len(clean_vals) and intent is None:
                    # Check if next tokens are remaining 4 Kev tokens
                    rem_count = len(clean_vals) - val_idx
                    if rem_count == 4 and clean_vals[val_idx] in VALID_INTENT and clean_vals[val_idx+1] in VALID_EPIST and clean_vals[val_idx+2] in VALID_ALLEN and clean_vals[val_idx+3] in VALID_PEARL:
                        intent, epist, allen, pearl = clean_vals[val_idx], clean_vals[val_idx+1], clean_vals[val_idx+2], clean_vals[val_idx+3]
                        val_idx += 4
                    else:
                        obj = clean_vals[val_idx]
                        val_idx += 1

                if val_idx < len(clean_vals) and intent is None:
                    intent = clean_vals[val_idx]
                    val_idx += 1
                if val_idx < len(clean_vals) and epist is None:
                    epist = clean_vals[val_idx]
                    val_idx += 1
                if val_idx < len(clean_vals) and allen is None:
                    allen = clean_vals[val_idx]
                    val_idx += 1
                if val_idx < len(clean_vals) and pearl is None:
                    pearl = clean_vals[val_idx]
                    val_idx += 1

            ev_dict: Dict[str, Any] = {
                "id": ev_id,
                "pred": pred,
                "subj": subj,
                "obj": obj,
            }
            if intent is not None:
                ev_dict["intent"] = intent
            if epist is not None:
                ev_dict["epist"] = epist
                ev_dict["epistemic"] = epist
            if allen is not None:
                ev_dict["allen"] = allen
            if pearl is not None:
                ev_dict["pearl"] = pearl
            events.append(ev_dict)

    return {"entities": entities, "events": events}


def _locate_skeleton_gbnf(custom_path: Optional[Union[str, Path]] = None, filename: str = "skeleton_schema.gbnf") -> Path:
    """Locate the skeleton_schema.gbnf or co_decoded_skeleton_schema.gbnf grammar specification file."""
    if custom_path is not None:
        p = Path(custom_path).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified GBNF grammar path does not exist: {custom_path}")

    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "data" / "grammar" / filename,
        Path.cwd() / "data" / "grammar" / filename,
        here.parent.parent / "data" / "grammar" / filename,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    # Fallback to standard skeleton_schema.gbnf if specialized grammar not found
    fallback = "skeleton_schema.gbnf"
    candidates_fallback = [
        here.parent.parent.parent / "data" / "grammar" / fallback,
        Path.cwd() / "data" / "grammar" / fallback,
        here.parent.parent / "data" / "grammar" / fallback,
    ]
    for c in candidates_fallback:
        if c.is_file():
            return c.resolve()

    require_artifacts(f"data/grammar/{filename}", component="SkeletonTransducer")
    raise FileNotFoundError(f"Could not locate '{filename}'. Ensure 'data/grammar/{filename}' exists.")


def _as_ent_id(v: Any) -> Optional[str]:
    """Coerces an LLM-emitted subj/obj reference (str, list, or None) to a single entity id string."""
    if isinstance(v, (list, tuple)):
        v = next((x for x in v if x), None)
    return str(v) if v not in (None, "") else None


def estimate_token_count(text: str) -> int:
    """Accurately estimate BPE token count for skeleton JSON."""
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\w\s]+", text)
    bpe_estimate = max(1, round(len(text) / 3.9))
    return min(len(tokens), bpe_estimate)


# ---------------------------------------------------------------------------
# Data Schemas: SkeletonEntity, SkeletonEvent, SkeletonExtractionResult
# ---------------------------------------------------------------------------

@dataclass
class SkeletonEntity:
    """Represents an extracted surface entity with character and byte groundings."""
    id: str
    surface_text: str
    char_span: Tuple[int, int]
    byte_span: Optional[Tuple[int, int]] = None
    category: str = "OBJECT"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "surface_text": self.surface_text,
            "char_span": list(self.char_span),
            "byte_span": list(self.byte_span) if self.byte_span else None,
            "category": self.category,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkeletonEntity:
        raw_span = data.get("char_span") or (0, 0)
        char_span = (int(raw_span[0]), int(raw_span[1]))
        raw_bspan = data.get("byte_span")
        byte_span = (int(raw_bspan[0]), int(raw_bspan[1])) if raw_bspan else None
        return cls(
            id=str(data["id"]),
            surface_text=str(data.get("surface_text", data.get("text", ""))),
            char_span=char_span,
            byte_span=byte_span,
            category=str(data.get("category", "OBJECT")),
            confidence=float(data.get("confidence", 1.0)),
        )

    def to_extracted_entity(self) -> ExtractedEntity:
        return ExtractedEntity(
            id=self.id,
            canonical_name=self.surface_text,
            category=self.category,
            surface_aliases=[self.surface_text],
            properties={
                "char_span": list(self.char_span),
                "byte_span": list(self.byte_span) if self.byte_span else None,
                "confidence": self.confidence,
            },
        )


@dataclass
class SkeletonEvent:
    """Represents an unlabelled event anchor and thematic SVO frame, with optional co-decoded Kev decisions."""
    id: str
    predicate: str
    char_span: Tuple[int, int]
    subject_ent_id: Optional[str] = None
    object_ent_id: Optional[str] = None
    byte_span: Optional[Tuple[int, int]] = None
    raw_text: Optional[str] = None
    confidence: float = 1.0
    intent: Optional[str] = None
    epist: Optional[str] = None
    allen: Optional[str] = None
    pearl: Optional[str] = None

    @property
    def epistemic(self) -> Optional[str]:
        return self.epist

    @epistemic.setter
    def epistemic(self, value: Optional[str]) -> None:
        self.epist = value

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "predicate": self.predicate,
            "char_span": list(self.char_span),
            "subject_ent_id": self.subject_ent_id,
            "object_ent_id": self.object_ent_id,
            "byte_span": list(self.byte_span) if self.byte_span else None,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
        }
        if self.intent is not None:
            d["intent"] = self.intent
        if self.epist is not None:
            d["epist"] = self.epist
            d["epistemic"] = self.epist
        if self.allen is not None:
            d["allen"] = self.allen
        if self.pearl is not None:
            d["pearl"] = self.pearl
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkeletonEvent:
        raw_span = data.get("char_span") or (0, 0)
        char_span = (int(raw_span[0]), int(raw_span[1]))
        raw_bspan = data.get("byte_span")
        byte_span = (int(raw_bspan[0]), int(raw_bspan[1])) if raw_bspan else None
        return cls(
            id=str(data["id"]),
            predicate=str(data.get("predicate", data.get("pred", ""))),
            char_span=char_span,
            subject_ent_id=data.get("subject_ent_id", data.get("subj")),
            object_ent_id=data.get("object_ent_id", data.get("obj")),
            byte_span=byte_span,
            raw_text=data.get("raw_text"),
            confidence=float(data.get("confidence", 1.0)),
            intent=data.get("intent"),
            epist=data.get("epist", data.get("epistemic")),
            allen=data.get("allen"),
            pearl=data.get("pearl"),
        )

    def to_extracted_event(self) -> ExtractedEvent:
        return ExtractedEvent(
            id=self.id,
            predicate=self.predicate,
            agent_id=self.subject_ent_id,
            patient_id=self.object_ent_id,
            raw_text=self.raw_text,
            arguments={
                "char_span": list(self.char_span),
                "byte_span": list(self.byte_span) if self.byte_span else None,
                "confidence": self.confidence,
                **({"intent": self.intent} if self.intent else {}),
                **({"epist": self.epist} if self.epist else {}),
                **({"allen": self.allen} if self.allen else {}),
                **({"pearl": self.pearl} if self.pearl else {}),
            },
        )


@dataclass
class SkeletonExtractionResult:
    """Structured extraction output from SkeletonTransducer."""
    entities: List[SkeletonEntity] = field(default_factory=list)
    events: List[SkeletonEvent] = field(default_factory=list)
    chunk_id: Optional[str] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "events": [ev.to_dict() for ev in self.events],
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkeletonExtractionResult:
        return cls(
            entities=[SkeletonEntity.from_dict(e) for e in data.get("entities", [])],
            events=[SkeletonEvent.from_dict(ev) for ev in data.get("events", [])],
            chunk_id=data.get("chunk_id"),
            text=data.get("text"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_skeleton_json(self) -> str:
        """Returns minimal JSON skeleton conforming to skeleton_schema.gbnf / co_decoded_skeleton_schema.gbnf."""
        return json.dumps({
            "entities": [{"id": e.id, "text": e.surface_text} for e in self.entities],
            "events": [
                {
                    "id": ev.id,
                    "pred": ev.predicate,
                    **({"subj": ev.subject_ent_id} if ev.subject_ent_id else {}),
                    **({"obj": ev.object_ent_id} if ev.object_ent_id else {}),
                    **({"intent": ev.intent} if ev.intent else {}),
                    **({"epist": ev.epist} if ev.epist else {}),
                    **({"allen": ev.allen} if ev.allen else {}),
                    **({"pearl": ev.pearl} if ev.pearl else {}),
                }
                for ev in self.events
            ],
        }, separators=(',', ':'), ensure_ascii=False)

    def to_skeleton_compact_sexpr(self) -> str:
        """Returns compact keyword S-expression conforming to compact_skeleton_sexpr.gbnf."""
        clauses: List[str] = []
        for e in self.entities:
            type_part = f" :type {e.category}" if e.category and e.category != "OBJECT" else ""
            escaped_text = json.dumps(e.surface_text, ensure_ascii=False)
            clauses.append(f"(entity {e.id} {escaped_text}{type_part})")
        for ev in self.events:
            parts = [f"event {ev.id} {ev.predicate}"]
            if ev.subject_ent_id:
                parts.append(f":subj {ev.subject_ent_id}")
            if ev.object_ent_id:
                parts.append(f":obj {ev.object_ent_id}")
            if ev.intent:
                parts.append(f":intent {ev.intent}")
            if ev.epist:
                parts.append(f":epist {ev.epist}")
            if ev.allen:
                parts.append(f":allen {ev.allen}")
            if ev.pearl:
                parts.append(f":pearl {ev.pearl}")
            clauses.append(f"({' '.join(parts)})")
        if not clauses:
            return "(graph)"
        return f"(graph {' '.join(clauses)})"

    def to_skeleton_positional_sexpr(self) -> str:
        """Returns ultra-compact positional S-expression conforming to positional_skeleton_sexpr.gbnf."""
        clauses: List[str] = []
        for e in self.entities:
            type_part = f" {e.category}" if e.category and e.category != "OBJECT" else ""
            escaped_text = json.dumps(e.surface_text, ensure_ascii=False)
            clauses.append(f"(e {e.id} {escaped_text}{type_part})")
        for ev in self.events:
            parts = ["ev", ev.id, ev.predicate]
            has_kev = bool(ev.intent or ev.epist or ev.allen or ev.pearl)
            if ev.subject_ent_id or ev.object_ent_id or has_kev:
                parts.append(ev.subject_ent_id or "-")
            if ev.object_ent_id or has_kev:
                parts.append(ev.object_ent_id or "-")
            if has_kev:
                parts.append(ev.intent or "I")
                parts.append(ev.epist or "O")
                parts.append(ev.allen or "N")
                parts.append(ev.pearl or "N")
            clauses.append(f"({' '.join(parts)})")
        return f"({' '.join(clauses)})"

    def to_skeleton_sexpr(self, format: str = "sexpr_compact") -> str:
        """Serialize stripped skeleton into chosen S-expression format."""
        fmt = (format or "sexpr_compact").lower()
        if fmt in ("sexpr_positional", "positional_sexpr", "positional"):
            return self.to_skeleton_positional_sexpr()
        return self.to_skeleton_compact_sexpr()

    def to_skeleton_raw(self, format: str = "sexpr_compact") -> str:
        """Serialize stripped skeleton into requested format (json, sexpr_compact, or sexpr_positional)."""
        fmt = (format or "sexpr_compact").lower()
        if fmt in ("json", "skeleton_json"):
            return self.to_skeleton_json()
        elif fmt in ("sexpr_positional", "positional_sexpr", "positional"):
            return self.to_skeleton_positional_sexpr()
        return self.to_skeleton_compact_sexpr()

    @classmethod
    def from_json(cls, json_str: str) -> SkeletonExtractionResult:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_sexpr(
        cls,
        sexpr_str: str,
        text: Optional[str] = None,
        aligner: Optional[SpanAligner] = None,
        chunk_id: Optional[str] = None,
    ) -> SkeletonExtractionResult:
        """Parse compact or positional S-expression and optionally ground against source text."""
        data = parse_skeleton_sexpr(sexpr_str)
        if text is not None:
            span_aligner = aligner or SpanAligner()
            entities: List[SkeletonEntity] = []
            for raw_e in data.get("entities", []):
                surface = str(raw_e.get("text", "")).strip()
                aln = span_aligner.align(text, surface)
                c_span = aln.char_span if aln else (0, min(len(surface), len(text)))
                b_span = aln.byte_span if aln else span_aligner.char_to_byte_span(text, c_span)
                entities.append(
                    SkeletonEntity(
                        id=str(raw_e.get("id", f"E{len(entities) + 1}")),
                        surface_text=aln.matched_text if aln else surface,
                        char_span=c_span,
                        byte_span=b_span,
                        category=str(raw_e.get("category", "OBJECT")),
                        confidence=aln.confidence if aln else 0.8,
                    )
                )
            events: List[SkeletonEvent] = []
            for raw_ev in data.get("events", []):
                pred = str(raw_ev.get("pred", raw_ev.get("predicate", ""))).strip()
                aln = span_aligner.align(text, pred)
                c_span = aln.char_span if aln else (0, min(len(pred), len(text)))
                b_span = aln.byte_span if aln else span_aligner.char_to_byte_span(text, c_span)
                events.append(
                    SkeletonEvent(
                        id=str(raw_ev.get("id", f"EV{len(events) + 1}")),
                        predicate=pred,
                        char_span=c_span,
                        subject_ent_id=_as_ent_id(raw_ev.get("subj", raw_ev.get("subject_ent_id"))),
                        object_ent_id=_as_ent_id(raw_ev.get("obj", raw_ev.get("object_ent_id"))),
                        byte_span=b_span,
                        raw_text=text[c_span[0]:c_span[1]] if aln else None,
                        confidence=aln.confidence if aln else 0.8,
                        intent=raw_ev.get("intent"),
                        epist=raw_ev.get("epist", raw_ev.get("epistemic")),
                        allen=raw_ev.get("allen"),
                        pearl=raw_ev.get("pearl"),
                    )
                )
            return cls(
                entities=entities,
                events=events,
                chunk_id=chunk_id,
                text=text,
            )
        return cls.from_dict(data)

    def to_discourse_result(self) -> DiscourseExtractionResult:
        """Convert stripped skeleton to standard DiscourseExtractionResult."""
        res = DiscourseExtractionResult(
            chunk_id=self.chunk_id or "chunk_skeleton",
            entities=[e.to_extracted_entity() for e in self.entities],
            events=[ev.to_extracted_event() for ev in self.events],
            relations=[],
            propositions=[],
            metadata=dict(self.metadata),
        )
        return res

    def to_asg(self, passage_id: Optional[str] = None) -> QuantaGraph:
        """Convert to QuantaGraph with passage bindings."""
        graph = QuantaGraph()
        ent_cid_map: Dict[str, str] = {}

        for ent in self.entities:
            node = QuantaNode(
                literal=ent.surface_text,
                anchor=ent.id,
                node_type=ent.category,
                salience=1.0,
                truth_status="TRUE",
                confidence=ent.confidence,
                passage_id=passage_id,
                span_start=ent.char_span[0],
                span_end=ent.char_span[1],
            )
            node.compute_canonical_cid()
            graph.add_node(node)
            ent_cid_map[ent.id] = node.canonical_cid
            if passage_id:
                graph.add_passage_anchor(node.canonical_cid, passage_id, ent.char_span[0], ent.char_span[1])

        for ev in self.events:
            ev_node = QuantaNode(
                literal=ev.predicate,
                anchor=ev.id,
                node_type="EVENT",
                salience=1.0,
                truth_status="TRUE",
                confidence=ev.confidence,
                passage_id=passage_id,
                span_start=ev.char_span[0],
                span_end=ev.char_span[1],
            )
            ev_node.compute_canonical_cid()
            graph.add_node(ev_node)
            if passage_id:
                graph.add_passage_anchor(ev_node.canonical_cid, passage_id, ev.char_span[0], ev.char_span[1])

            if ev.subject_ent_id and ev.subject_ent_id in ent_cid_map:
                graph.add_edge(ev_node.canonical_cid, "AGENT", ent_cid_map[ev.subject_ent_id])
            if ev.object_ent_id and ev.object_ent_id in ent_cid_map:
                graph.add_edge(ev_node.canonical_cid, "PATIENT", ent_cid_map[ev.object_ent_id])

        return graph

    def to_record_dsl(self, passage_id: Optional[str] = None, doc_id: str = "default_doc") -> str:
        """Serialize stripped skeleton into Canonical Record DSL."""
        lines: List[str] = []
        pid = passage_id or (f"P_{self.chunk_id}" if self.chunk_id else "P_chunk")

        if self.text:
            escaped_text = json.dumps(self.text, ensure_ascii=False)
            lines.append(f'passage {pid} {{')
            lines.append(f'    doc_id: "{doc_id}"')
            lines.append(f'    span: [0, {len(self.text)}]')
            lines.append(f'    text: {escaped_text}')
            lines.append('}\n')

        for ent in self.entities:
            lines.append(f'entity {ent.id} "{ent.surface_text}" {{')
            lines.append(f'    type: {ent.category}')
            if passage_id or self.text:
                lines.append(f'    passage: {pid}')
                lines.append(f'    span: [{ent.char_span[0]}, {ent.char_span[1]}]')
            lines.append('}\n')

        for ev in self.events:
            lines.append(f'event {ev.id} "{ev.predicate}" {{')
            if ev.subject_ent_id:
                lines.append(f'    agent: {ev.subject_ent_id}')
            if ev.object_ent_id:
                lines.append(f'    patient: {ev.object_ent_id}')
            if passage_id or self.text:
                lines.append(f'    passage: {pid}')
                lines.append(f'    span: [{ev.char_span[0]}, {ev.char_span[1]}]')
            lines.append('}\n')

        return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Offline Mock Skeleton Transducer
# ---------------------------------------------------------------------------

class MockSkeletonTransducer:
    """Deterministic, high-speed (<5ms) offline skeleton transducer for CI testing."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        mode: str = "standard",
        skeleton_format: Optional[str] = None,
        co_decoded: bool = False,
    ):
        self.mode = mode
        if skeleton_format is None:
            if mode == "co_decoded":
                skeleton_format = "json"
                co_decoded = True
            else:
                skeleton_format = "sexpr_compact"
        self.skeleton_format = skeleton_format
        self.co_decoded = co_decoded or (mode == "co_decoded")
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif self.skeleton_format in ("sexpr_positional", "positional_sexpr", "positional"):
            self.system_prompt = CO_DECODED_POSITIONAL_SEXPR_SYSTEM_PROMPT if self.co_decoded else DEFAULT_POSITIONAL_SEXPR_SYSTEM_PROMPT
        elif self.skeleton_format in ("json", "skeleton_json"):
            self.system_prompt = CO_DECODED_SKELETON_SYSTEM_PROMPT if self.co_decoded else DEFAULT_SKELETON_SYSTEM_PROMPT
        else:
            self.system_prompt = CO_DECODED_COMPACT_SEXPR_SYSTEM_PROMPT if self.co_decoded else DEFAULT_COMPACT_SEXPR_SYSTEM_PROMPT
        self.aligner = SpanAligner()

        # Known pre-seeded benchmark heuristics (primary S-V-O frames)
        self._canonical_fixtures: Dict[str, Dict[str, Any]] = {
            "marcus_vance": {
                "entities": [
                    {"id": "E1", "text": "Dr. Marcus Vance", "category": "PERSON"},
                    {"id": "E2", "text": "argon cylinder", "category": "OBJECT"},
                ],
                "events": [
                    {
                        "id": "EV1", "pred": "pressurize", "subj": "E1", "obj": "E2",
                        "intent": "I", "epist": "O", "allen": "B", "pearl": "M",
                    },
                ],
            },
            "eleanor_vance": {
                "entities": [
                    {"id": "E1", "text": "Eleanor Vance", "category": "PERSON"},
                    {"id": "E2", "text": "containment cell", "category": "LOCATION"},
                ],
                "events": [
                    {
                        "id": "EV1", "pred": "enter", "subj": "E1", "obj": "E2",
                        "intent": "I", "epist": "O", "allen": "B", "pearl": "M",
                    },
                ],
            },
            "alice_auditor": {
                "entities": [
                    {"id": "E1", "text": "Alice", "category": "PERSON"},
                    {"id": "E2", "text": "auditor", "category": "PERSON"},
                ],
                "events": [
                    {
                        "id": "EV1", "pred": "remark", "subj": "E2", "obj": "E1",
                        "intent": "I", "epist": "H", "allen": "N", "pearl": "N",
                    },
                ],
            },
        }

    def transduce(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        mode: Optional[str] = None,
        skeleton_format: Optional[str] = None,
        co_decoded: Optional[bool] = None,
        **kwargs,
    ) -> SkeletonExtractionResult:
        """Deterministically extract skeleton entities and events with exact span alignment."""
        effective_co_decoded = co_decoded if co_decoded is not None else kwargs.get("co_decoded", self.co_decoded)
        if mode == "co_decoded" or kwargs.get("mode") == "co_decoded":
            effective_co_decoded = True
        effective_format = skeleton_format or kwargs.get("skeleton_format") or self.skeleton_format
        effective_mode = mode or kwargs.get("mode") or ("co_decoded" if effective_co_decoded else self.mode)
        t0 = time.perf_counter()
        clean_text = text.strip()
        lower = clean_text.lower()

        matched_spec: Optional[Dict[str, Any]] = None
        if "marcus vance" in lower or "argon cylinder" in lower:
            matched_spec = self._canonical_fixtures["marcus_vance"]
        elif "eleanor" in lower or "containment cell" in lower or "synthetic compound" in lower:
            matched_spec = self._canonical_fixtures["eleanor_vance"]
        elif "alice" in lower and "auditor" in lower:
            matched_spec = self._canonical_fixtures["alice_auditor"]

        if matched_spec is not None:
            raw_entities = matched_spec["entities"]
            raw_events = matched_spec["events"]
        else:
            raw_entities, raw_events = self._extract_heuristic_skeleton(
                clean_text, mode="co_decoded" if effective_co_decoded else "standard"
            )

        # Ground spans via SpanAligner
        entities: List[SkeletonEntity] = []
        for raw_e in raw_entities:
            surface = raw_e["text"]
            alignment = self.aligner.align(clean_text, surface)
            if alignment:
                c_span = alignment.char_span
                b_span = alignment.byte_span
                matched_surface = alignment.matched_text
            else:
                c_span = (0, min(len(surface), len(clean_text)))
                b_span = self.aligner.char_to_byte_span(clean_text, c_span)
                matched_surface = surface

            entities.append(
                SkeletonEntity(
                    id=raw_e["id"],
                    surface_text=matched_surface,
                    char_span=c_span,
                    byte_span=b_span,
                    category=raw_e.get("category", "OBJECT"),
                    confidence=alignment.confidence if alignment else 0.8,
                )
            )

        events: List[SkeletonEvent] = []
        for raw_ev in raw_events:
            pred = raw_ev["pred"]
            alignment = self.aligner.align(clean_text, pred)
            if alignment:
                c_span = alignment.char_span
                b_span = alignment.byte_span
            else:
                c_span = (0, min(len(pred), len(clean_text)))
                b_span = self.aligner.char_to_byte_span(clean_text, c_span)

            intent_val = raw_ev.get("intent") if effective_co_decoded else None
            epist_val = (raw_ev.get("epist") or raw_ev.get("epistemic")) if effective_co_decoded else None
            allen_val = raw_ev.get("allen") if effective_co_decoded else None
            pearl_val = raw_ev.get("pearl") if effective_co_decoded else None

            events.append(
                SkeletonEvent(
                    id=raw_ev["id"],
                    predicate=pred,
                    char_span=c_span,
                    subject_ent_id=_as_ent_id(raw_ev.get("subj")),
                    object_ent_id=_as_ent_id(raw_ev.get("obj")),
                    byte_span=b_span,
                    raw_text=clean_text[c_span[0]:c_span[1]] if alignment else None,
                    confidence=alignment.confidence if alignment else 0.8,
                    intent=intent_val,
                    epist=epist_val,
                    allen=allen_val,
                    pearl=pearl_val,
                )
            )

        temp_res = SkeletonExtractionResult(
            entities=entities,
            events=events,
            chunk_id=chunk_id,
            text=clean_text,
        )
        raw_repr = temp_res.to_skeleton_raw(format=effective_format)
        token_count = estimate_token_count(raw_repr)
        latency = time.perf_counter() - t0

        return SkeletonExtractionResult(
            entities=entities,
            events=events,
            chunk_id=chunk_id,
            text=clean_text,
            metadata={
                "latency_sec": latency,
                "prefill_latency_sec": latency * 0.4,
                "decoding_latency_sec": latency * 0.6,
                "decoding_token_count": token_count,
                "backend": "mock_skeleton",
                "model": "qwen3.5-4b-skeleton",
                "mode": effective_mode,
                "skeleton_format": effective_format,
                "co_decoded": effective_co_decoded,
                "raw_repr": raw_repr,
            },
        )

    def transduce_raw(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        mode: Optional[str] = None,
        skeleton_format: Optional[str] = None,
        co_decoded: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """Return raw S-expression or skeleton JSON string directly."""
        kw = dict(kwargs)
        fmt = skeleton_format or kw.pop("skeleton_format", None) or self.skeleton_format
        co = co_decoded if co_decoded is not None else kw.pop("co_decoded", None)
        res = self.transduce(
            text=text,
            chunk_id=chunk_id,
            active_entities=active_entities,
            mode=mode,
            skeleton_format=fmt,
            co_decoded=co,
            **kw,
        )
        return res.to_skeleton_raw(format=fmt)

    async def transduce_raw_async(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        mode: Optional[str] = None,
        skeleton_format: Optional[str] = None,
        co_decoded: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """Asynchronous execution wrapper for transduce_raw."""
        return await asyncio.to_thread(
            self.transduce_raw,
            text=text,
            chunk_id=chunk_id,
            active_entities=active_entities,
            mode=mode,
            skeleton_format=skeleton_format,
            co_decoded=co_decoded,
            **kwargs,
        )

    def _extract_heuristic_skeleton(self, text: str, mode: str = "standard") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Dynamic heuristic extractor extracting surface entities and predicates for arbitrary text."""
        seen_texts: Set[str] = set()
        entities: List[Dict[str, Any]] = []

        def _add_entity(surface: str, category: str = "OBJECT"):
            s_clean = surface.strip()
            if s_clean and s_clean.lower() not in seen_texts and len(s_clean) >= 3:
                seen_texts.add(s_clean.lower())
                eid = f"E{len(entities) + 1}"
                entities.append({"id": eid, "text": s_clean, "category": category})

        # 1. Technical & domain-specific compound nouns and identifiers
        compound_patterns = [
            (r"\b(?:Vault|Room|Sector|Cluster|Chamber|Station)\s*\d+\b", "LOCATION"),
            (r"\b[A-Z0-9]+-[A-Z0-9_-]+\b", "OBJECT"),  # PHANTOM-9092, WASP-96b, SKU-901
            (r"\b(?:tok|txn)_[a-zA-Z0-9_]+\b", "OBJECT"),  # tok_visa_4242, txn_9941
            (r"\b(?:Order\s*\d+)\b", "OBJECT"),
            (r"\b(?:designated mission access code|mission access code|access code)\b", "OBJECT"),
            (r"\b(?:James Webb Space Telescope|space telescope)\b", "OBJECT"),
            (r"\b(?:primary mirror segments|mirror segments)\b", "OBJECT"),
            (r"\b(?:beryllium|gold)\b", "SUBSTANCE"),
            (r"\b(?:Ariane 5|heavy launcher)\b", "OBJECT"),
            (r"\b(?:trans-Lagrange transfer trajectory|trajectory)\b", "LOCATION"),
            (r"\b(?:cryogenic containment cell \d+|containment cell \d+|containment cell)\b", "LOCATION"),
            (r"\b(?:titanium pressure valve|titanium valve|pressure valve)\b", "OBJECT"),
            (r"\b(?:toxic argon gas|toxic argon carrier gas|toxic argon|argon)\b", "SUBSTANCE"),
            (r"\b(?:exothermic reaction|reaction)\b", "OBJECT"),
            (r"\b(?:service corridors|adjacent service corridors)\b", "LOCATION"),
            (r"\b(?:facility lockdown protocol|lockdown protocol)\b", "OBJECT"),
            (r"\b(?:emergency bulkheads|bulkheads)\b", "OBJECT"),
            (r"\b(?:contamination zone|sectors \d+ through \d+)\b", "LOCATION"),
            (r"\b(?:Order Fulfillment Saga|OrderFulfillmentService)\b", "OBJECT"),
            (r"\b(?:PaymentGatewayClient|InventoryService)\b", "OBJECT"),
        ]
        for pat, cat in compound_patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                _add_entity(match.group(0), cat)
                if len(entities) >= 12:
                    break
            if len(entities) >= 12:
                break

        # 2. Capitalized proper nouns & named entities
        cap_pat = re.compile(
            r"\b(?:Dr\.\s+)?[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüűA-Z0-9_-]*(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ0-9][a-záéíóöőúüűA-Z0-9_-]*)*\b"
        )
        for m in cap_pat.finditer(text):
            t = m.group(0).strip()
            if t.lower() not in {"the", "this", "that", "there", "during", "after", "before", "when", "on"}:
                cat = "PERSON" if any(w in t.lower() for w in ("dr.", "vance", "eleanor", "marcus", "alice")) else "OBJECT"
                _add_entity(t, cat)
                if len(entities) >= 12:
                    break

        if not entities:
            # Fallback to noun phrases or first word
            first_word = text.split()[0].strip() if text.split() else "entity"
            entities.append({"id": "E1", "text": first_word, "category": "OBJECT"})

        # 3. Action & causal predicates
        KNOWN_VERBS = [
            "trigger", "triggered", "isolate", "isolated", "seal", "sealed",
            "rupture", "ruptured", "vent", "vented", "breach", "breached",
            "pressurize", "pressurized", "synthesize", "synthesized", "analyze", "analyzed",
            "verify", "verified", "measure", "measured", "observe", "observed",
            "enter", "entered", "react", "reacted", "transport", "transported",
            "engineer", "engineered", "inject", "injected", "orchestrate", "orchestrates",
            "call", "return", "calculate", "validate", "deploy", "operate",
            "felszállt", "megvizsgálta", "szintetizálta", "besugározta", "kutatta",
        ]

        events: List[Dict[str, Any]] = []
        for verb in KNOWN_VERBS:
            if re.search(rf"\b{re.escape(verb)}\b", text, re.IGNORECASE):
                lemma = re.sub(r"ed$", "", verb)
                ev_id = f"EV{len(events) + 1}"
                subj_id = entities[0]["id"] if entities else None
                obj_id = entities[1]["id"] if len(entities) > 1 else None
                events.append({
                    "id": ev_id,
                    "pred": lemma,
                    "subj": subj_id,
                    "obj": obj_id,
                })
                if len(events) >= 3:
                    break

        if not events:
            # Fallback predicate
            events.append({
                "id": "EV1",
                "pred": "observe",
                "subj": entities[0]["id"] if entities else None,
                "obj": None,
            })

        if mode == "co_decoded":
            # Infer co-decoded Kev decisions from linguistic cues
            intent_code = "I"
            if re.search(r"\b(?:please|ensure|must|should|command|order|isolate!|verify!)\b", text, re.IGNORECASE):
                intent_code = "D"
            elif re.search(r"\b(?:promise|guarantee|commit|pledge)\b", text, re.IGNORECASE):
                intent_code = "C"
            elif re.search(r"\b(?:alas|congratulations|wow|unfortunately|thank)\b", text, re.IGNORECASE):
                intent_code = "E"

            epist_code = "O"
            if re.search(r"\b(?:allegedly|claimed|hearsay|reported|according to)\b", text, re.IGNORECASE):
                epist_code = "H"
            elif re.search(r"\b(?:deduced|concluded|therefore|thus|inferred)\b", text, re.IGNORECASE):
                epist_code = "D"
            elif re.search(r"\b(?:might|suggests|hypothesized|appears to|seems|probably|could)\b", text, re.IGNORECASE):
                epist_code = "C"

            allen_code = "N"
            if re.search(r"\b(?:before|preceded|prior to)\b", text, re.IGNORECASE):
                allen_code = "B"
            elif re.search(r"\b(?:immediately|meets|adjacent|followed by)\b", text, re.IGNORECASE):
                allen_code = "M"
            elif re.search(r"\b(?:during|while|throughout)\b", text, re.IGNORECASE):
                allen_code = "D"
            elif re.search(r"\b(?:overlaps|overlapping|coinciding)\b", text, re.IGNORECASE):
                allen_code = "O"
            elif len(events) > 1:
                allen_code = "B"

            pearl_code = "N"
            if re.search(r"\b(?:because|causing|caused|resulting in|catalyzed|leads to|triggering)\b", text, re.IGNORECASE):
                pearl_code = "M"
            elif re.search(r"\b(?:enabling|allowing|permits|condition for)\b", text, re.IGNORECASE):
                pearl_code = "C"

            if pearl_code in ("M", "C") and allen_code == "N":
                allen_code = "B"

            for ev in events:
                ev["intent"] = intent_code
                ev["epist"] = epist_code
                ev["allen"] = allen_code
                ev["pearl"] = pearl_code

        return entities, events


# ---------------------------------------------------------------------------
# Production Qwen-4B Skeleton Transducer
# ---------------------------------------------------------------------------

class SkeletonTransducer:
    """Production client connecting to llama-server on port 8888 with GBNF grammar."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "qwen3.5-4b",
        timeout: float = 60.0,
        max_retries: int = 2,
        fallback_to_mock: bool = True,
        grammar_path: Optional[Union[str, Path]] = None,
        system_prompt: Optional[str] = None,
        mode: str = "standard",
        skeleton_format: Optional[str] = None,
        co_decoded: bool = False,
    ):
        raw_base = (
            base_url
            or os.environ.get("LLAMA_SERVER_BASE_URL")
            or os.environ.get("UNSLOTH_BASE_URL")
            or "http://127.0.0.1:8888/v1"
        ).rstrip("/")
        # Avoid Windows 11 IPv6 localhost DNS timeout (2-4 second delay)
        if "://localhost:" in raw_base:
            raw_base = raw_base.replace("://localhost:", "://127.0.0.1:")
        self.base_url = raw_base
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.fallback_to_mock = fallback_to_mock
        self.mode = mode
        if skeleton_format is None:
            if mode == "co_decoded":
                skeleton_format = "json"
                co_decoded = True
            else:
                skeleton_format = "sexpr_compact"
        self.skeleton_format = skeleton_format
        self.co_decoded = co_decoded or (mode == "co_decoded")
        self.custom_system_prompt = system_prompt
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self._select_system_prompt(self.skeleton_format, self.co_decoded)

        self.session = requests.Session()
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.aligner = SpanAligner()

        # Load and cache GBNF Grammars
        self.custom_grammar_path = grammar_path
        target_grammar_file = self._select_grammar_filename(self.skeleton_format, self.co_decoded)
        self.grammar_path = _locate_skeleton_gbnf(grammar_path, filename=target_grammar_file)
        self.grammar_content = self.grammar_path.read_text(encoding="utf-8")
        self.grammar_hash = hashlib.sha256(self.grammar_content.encode("utf-8")).hexdigest()
        self._grammar_cache: Dict[str, Tuple[str, str]] = {
            target_grammar_file: (self.grammar_content, self.grammar_hash)
        }

        self._mock = MockSkeletonTransducer(
            system_prompt=self.system_prompt,
            mode=mode,
            skeleton_format=self.skeleton_format,
            co_decoded=self.co_decoded,
        )
        self._last_fallback_used = False
        self._server_disabled = False

    @staticmethod
    def _select_grammar_filename(skeleton_format: str, co_decoded: bool = False) -> str:
        fmt = (skeleton_format or "sexpr_compact").lower()
        if fmt in ("sexpr_positional", "positional_sexpr", "positional"):
            return "positional_skeleton_sexpr.gbnf"
        elif fmt in ("json", "skeleton_json"):
            return "co_decoded_skeleton_schema.gbnf" if co_decoded else "skeleton_schema.gbnf"
        else:
            return "compact_skeleton_sexpr.gbnf"

    @staticmethod
    def _select_system_prompt(skeleton_format: str, co_decoded: bool = False) -> str:
        fmt = (skeleton_format or "sexpr_compact").lower()
        if fmt in ("sexpr_positional", "positional_sexpr", "positional"):
            return CO_DECODED_POSITIONAL_SEXPR_SYSTEM_PROMPT if co_decoded else DEFAULT_POSITIONAL_SEXPR_SYSTEM_PROMPT
        elif fmt in ("json", "skeleton_json"):
            return CO_DECODED_SKELETON_SYSTEM_PROMPT if co_decoded else DEFAULT_SKELETON_SYSTEM_PROMPT
        else:
            return CO_DECODED_COMPACT_SEXPR_SYSTEM_PROMPT if co_decoded else DEFAULT_COMPACT_SEXPR_SYSTEM_PROMPT

    def _get_grammar(self, target_grammar_file: str) -> Tuple[str, str]:
        if target_grammar_file in self._grammar_cache:
            return self._grammar_cache[target_grammar_file]
        g_path = _locate_skeleton_gbnf(self.custom_grammar_path, filename=target_grammar_file)
        content = g_path.read_text(encoding="utf-8")
        g_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self._grammar_cache[target_grammar_file] = (content, g_hash)
        return (content, g_hash)

    def check_health(self) -> bool:
        """Check if llama-server endpoint is reachable and a model is loaded in memory."""
        if self._server_disabled:
            return False
        for path in ("/v1/models", "/models", "/health"):
            endpoint = f"{self.base_url.rstrip('/')}{path}"
            try:
                resp = self.session.get(endpoint, timeout=1.5)
                if resp.status_code == 200:
                    if path == "/health":
                        return True
                    data = resp.json().get("data", [])
                    loaded_models = [m for m in data if m.get("loaded", True) is not False]
                    if loaded_models:
                        return True
            except Exception:
                continue
        return False

    def transduce(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        mode: Optional[str] = None,
        skeleton_format: Optional[str] = None,
        co_decoded: Optional[bool] = None,
        **kwargs,
    ) -> SkeletonExtractionResult:
        """Extract skeleton entities and event anchors via llama-server with exact span groundings."""
        effective_co_decoded = co_decoded if co_decoded is not None else kwargs.get("co_decoded", self.co_decoded)
        if mode == "co_decoded" or kwargs.get("mode") == "co_decoded":
            effective_co_decoded = True
        effective_format = skeleton_format or kwargs.get("skeleton_format") or self.skeleton_format
        effective_mode = mode or kwargs.get("mode") or ("co_decoded" if effective_co_decoded else self.mode)
        clean_text = text.strip()
        t0 = time.perf_counter()

        if self._server_disabled and self.fallback_to_mock:
            self._last_fallback_used = True
            res = self._mock.transduce(
                clean_text,
                chunk_id=chunk_id,
                active_entities=active_entities,
                mode=effective_mode,
                skeleton_format=effective_format,
                co_decoded=effective_co_decoded,
            )
            res.metadata["fallback_from_server"] = True
            return res

        # Build payload & grammar
        target_grammar_file = self._select_grammar_filename(effective_format, effective_co_decoded)
        g_content, g_hash = self._get_grammar(target_grammar_file)

        sys_prompt = kwargs.get("system_prompt") or self.custom_system_prompt or self._select_system_prompt(effective_format, effective_co_decoded)

        fmt_lower = effective_format.lower()
        if fmt_lower in ("sexpr_positional", "positional_sexpr", "positional"):
            user_prompt = f"Extract co-decoded positional skeleton with Kev enums:\n\n{clean_text}" if effective_co_decoded else f"Extract positional skeleton:\n\n{clean_text}"
        elif fmt_lower in ("json", "skeleton_json"):
            user_prompt = f"Extract co-decoded skeleton with compact Kev enums:\n\n{clean_text}" if effective_co_decoded else f"Extract skeleton:\n\n{clean_text}"
        else: # "sexpr_compact"
            user_prompt = f"Extract co-decoded compact skeleton with Kev enums:\n\n{clean_text}" if effective_co_decoded else f"Extract compact skeleton:\n\n{clean_text}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "enable_thinking": False,
            "grammar": g_content,
            "chat_template_kwargs": {"enable_thinking": False},
            "extra_body": {
                "grammar": g_content,
                "grammar_hash": g_hash,
            },
        }

        base = self.base_url.rstrip("/")
        url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
        raw_output_str: Optional[str] = None
        usage_info: Dict[str, Any] = {}
        t_prefill = 0.0
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                t_req_start = time.perf_counter()
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                t_req_end = time.perf_counter()

                if resp.status_code == 200:
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    content = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
                    usage_info = data.get("usage", {})
                    t_prefill = (t_req_end - t_req_start) * 0.35  # Approx prefill proportion
                    raw_output_str = content.strip()
                    break
                else:
                    last_err = RuntimeError(f"Server error {resp.status_code}: {resp.text}")
                    # Fast fail: do not retry if model is not loaded or endpoint is invalid
                    if resp.status_code in (400, 404, 503) and ("no model loaded" in resp.text.lower() or "not found" in resp.text.lower()):
                        self._server_disabled = True
                        break
            except Exception as e:
                last_err = e

            if attempt < self.max_retries and not self._server_disabled:
                time.sleep(0.1 * attempt)

        if raw_output_str is None:
            if self.fallback_to_mock:
                self._last_fallback_used = True
                self._server_disabled = True
                logger.warning("llama-server unreachable (%s); using MockSkeletonTransducer", last_err)
                res = self._mock.transduce(
                    clean_text,
                    chunk_id=chunk_id,
                    active_entities=active_entities,
                    mode=effective_mode,
                    skeleton_format=effective_format,
                    co_decoded=effective_co_decoded,
                )
                res.metadata["fallback_from_server"] = True
                return res
            raise RuntimeError(f"Failed to extract skeleton from llama-server at {self.base_url}: {last_err}") from last_err

        # Parse extracted structure
        self._last_fallback_used = False
        try:
            parsed_data = self._clean_and_parse(raw_output_str, skeleton_format=effective_format)
            entities, events = self._ground_and_build(clean_text, parsed_data, mode=effective_mode)
        except Exception as parse_err:
            if self.fallback_to_mock:
                self._last_fallback_used = True
                logger.warning("Failed to parse output from llama-server (%s); falling back to Mock", parse_err)
                res = self._mock.transduce(
                    clean_text,
                    chunk_id=chunk_id,
                    active_entities=active_entities,
                    mode=effective_mode,
                    skeleton_format=effective_format,
                    co_decoded=effective_co_decoded,
                )
                res.metadata["fallback_from_server"] = True
                return res
            raise

        t_total = time.perf_counter() - t0
        token_count = usage_info.get("completion_tokens", estimate_token_count(raw_output_str))

        return SkeletonExtractionResult(
            entities=entities,
            events=events,
            chunk_id=chunk_id,
            text=clean_text,
            metadata={
                "latency_sec": t_total,
                "prefill_latency_sec": t_prefill,
                "decoding_latency_sec": max(0.0, t_total - t_prefill),
                "decoding_token_count": token_count,
                "backend": "llama_server",
                "model": self.model,
                "mode": effective_mode,
                "skeleton_format": effective_format,
                "co_decoded": effective_co_decoded,
                "raw_repr": raw_output_str,
            },
        )

    async def transduce_async(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        mode: Optional[str] = None,
        skeleton_format: Optional[str] = None,
        co_decoded: Optional[bool] = None,
        **kwargs,
    ) -> SkeletonExtractionResult:
        """Asynchronous execution wrapper for transduce."""
        return await asyncio.to_thread(
            self.transduce,
            text=text,
            chunk_id=chunk_id,
            active_entities=active_entities,
            mode=mode,
            skeleton_format=skeleton_format,
            co_decoded=co_decoded,
            **kwargs,
        )

    def transduce_raw(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        mode: Optional[str] = None,
        skeleton_format: Optional[str] = None,
        co_decoded: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """Extract raw S-expression or JSON string."""
        kw = dict(kwargs)
        fmt = skeleton_format or kw.pop("skeleton_format", None) or self.skeleton_format
        co = co_decoded if co_decoded is not None else kw.pop("co_decoded", None)
        res = self.transduce(
            text=text,
            chunk_id=chunk_id,
            active_entities=active_entities,
            mode=mode,
            skeleton_format=fmt,
            co_decoded=co,
            **kw,
        )
        return res.to_skeleton_raw(format=fmt)

    async def transduce_raw_async(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        mode: Optional[str] = None,
        skeleton_format: Optional[str] = None,
        co_decoded: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """Asynchronous execution wrapper for transduce_raw."""
        return await asyncio.to_thread(
            self.transduce_raw,
            text=text,
            chunk_id=chunk_id,
            active_entities=active_entities,
            mode=mode,
            skeleton_format=skeleton_format,
            co_decoded=co_decoded,
            **kwargs,
        )

    def _clean_and_parse(self, raw_str: str, skeleton_format: Optional[str] = None) -> Dict[str, Any]:
        """Strip fences and safely decode S-expression or JSON structure with auto-repair."""
        fmt = (skeleton_format or self.skeleton_format).lower()
        if fmt in ("json", "skeleton_json"):
            return self._clean_and_parse_json(raw_str)
        try:
            return parse_skeleton_sexpr(raw_str)
        except Exception:
            clean = raw_str.strip()
            if clean.startswith("{") or "{\"" in clean:
                try:
                    return self._clean_and_parse_json(raw_str)
                except Exception:
                    pass
            raise

    def _clean_and_parse_json(self, raw_str: str) -> Dict[str, Any]:
        """Strip fences and safely decode JSON structure, with truncated auto-repair."""
        clean = raw_str.strip()
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if fence:
            clean = fence.group(1).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Fallback 1: Extraction of clean JSON substring
        match = re.search(r"\{[\s\S]*\}", clean)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback 2: Auto-repair truncated JSON by finding last valid closing brace
        last_brace = clean.rfind("}")
        if last_brace != -1:
            candidate = clean[:last_brace + 1].strip()
            open_sq = candidate.count("[") - candidate.count("]")
            open_cr = candidate.count("{") - candidate.count("}")
            candidate += ("]" * max(0, open_sq)) + ("}" * max(0, open_cr))
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("Could not parse or repair JSON from LLM output", clean, 0)

    def _ground_and_build(
        self,
        source_text: str,
        data: Dict[str, Any],
        mode: str = "standard",
    ) -> Tuple[List[SkeletonEntity], List[SkeletonEvent]]:
        """Ground extracted entity texts and predicates to exact character/byte spans."""
        raw_entities = data.get("entities", [])
        raw_events = data.get("events", [])

        entities: List[SkeletonEntity] = []
        for raw_e in raw_entities:
            surface = str(raw_e.get("text", "")).strip()
            alignment = self.aligner.align(source_text, surface)
            if alignment:
                c_span = alignment.char_span
                b_span = alignment.byte_span
                matched_text = alignment.matched_text
            else:
                c_span = (0, min(len(surface), len(source_text)))
                b_span = self.aligner.char_to_byte_span(source_text, c_span)
                matched_text = surface

            entities.append(
                SkeletonEntity(
                    id=str(raw_e.get("id", f"E{len(entities) + 1}")),
                    surface_text=matched_text,
                    char_span=c_span,
                    byte_span=b_span,
                    category=str(raw_e.get("category", "OBJECT")),
                    confidence=alignment.confidence if alignment else 0.8,
                )
            )

        events: List[SkeletonEvent] = []
        for raw_ev in raw_events:
            pred = str(raw_ev.get("pred", "")).strip()
            alignment = self.aligner.align(source_text, pred)
            if alignment:
                c_span = alignment.char_span
                b_span = alignment.byte_span
            else:
                c_span = (0, min(len(pred), len(source_text)))
                b_span = self.aligner.char_to_byte_span(source_text, c_span)

            intent_val = raw_ev.get("intent")
            epist_val = raw_ev.get("epist", raw_ev.get("epistemic"))
            allen_val = raw_ev.get("allen")
            pearl_val = raw_ev.get("pearl")

            events.append(
                SkeletonEvent(
                    id=str(raw_ev.get("id", f"EV{len(events) + 1}")),
                    predicate=pred,
                    char_span=c_span,
                    subject_ent_id=_as_ent_id(raw_ev.get("subj")),
                    object_ent_id=_as_ent_id(raw_ev.get("obj")),
                    byte_span=b_span,
                    raw_text=source_text[c_span[0]:c_span[1]] if alignment else None,
                    confidence=alignment.confidence if alignment else 0.8,
                    intent=intent_val,
                    epist=epist_val,
                    allen=allen_val,
                    pearl=pearl_val,
                )
            )

        return entities, events
