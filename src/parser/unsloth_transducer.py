"""Unsloth Neural Discourse Transducer for QUANTA (Phase 2).

Connects to a local Small Language Model (SLM) served via Unsloth Desktop/Studio,
vLLM, or an OpenAI-compatible endpoint at http://localhost:8888/v1.
Enforces constrained decoding via GBNF (GGML BNF) grammar injection
(data/grammar/quanta_asg.gbnf) to emit typed, compact S-expressions.

Provides:
1. UnslothTransducer: Production client with GBNF grammar injection, health checks,
   retries, active entity coreference manifest injection, and mock fallback.
2. MockUnslothTransducer: Deterministic, high-speed (<5ms) offline mock supporting
   pre-seeded fixtures and dynamic heuristic S-expression synthesis.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union

import requests

from core.artifacts import require_artifacts, MissingArtifactError
from parser.entity_manifest import EntityRecord
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)
from parser.sexpr_parser import parse_sexpr, to_sexpr
from parser.transducer import (
    BaseDiscourseTransducer,
    CANONICAL_ELEANOR_VANCE_FIXTURE,
    CANONICAL_ELEANOR_VANCE_TEXT,
    CANONICAL_STRESS_1_FIXTURE,
    CANONICAL_STRESS_1_TEXT,
    CANONICAL_STRESS_2_FIXTURE,
    CANONICAL_STRESS_2_TEXT,
    CANONICAL_STRESS_3_FIXTURE,
    CANONICAL_STRESS_3_TEXT,
    CANONICAL_STRESS_4_FIXTURE,
    CANONICAL_STRESS_4_TEXT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported SLM Model Architectures & Presets (Phase 3.4)
# ---------------------------------------------------------------------------

MODEL_QWEN_4B = "qwen3.5-4b"          # Default: Qwen 3.5 4B Dense (Hybrid Linear Attention)
MODEL_QWEN_2B = "qwen3.5-2b"          # Ultra-low VRAM profile (<4GB physical VRAM)
MODEL_GEMMA_4 = "gemma-4-mtp"         # High-throughput profile (Gemma 4 with Multi-Token Prediction)
DEFAULT_MODEL = MODEL_QWEN_4B

MODEL_PRESETS: Dict[str, str] = {
    "qwen3.5-4b": MODEL_QWEN_4B,
    "qwen-4b": MODEL_QWEN_4B,
    "qwen_4b": MODEL_QWEN_4B,
    "qwen": MODEL_QWEN_4B,
    "default": MODEL_QWEN_4B,
    "qwen3.5-2b": MODEL_QWEN_2B,
    "qwen-2b": MODEL_QWEN_2B,
    "qwen_2b": MODEL_QWEN_2B,
    "ultra-low-vram": MODEL_QWEN_2B,
    "low-vram": MODEL_QWEN_2B,
    "gemma-4": MODEL_GEMMA_4,
    "gemma-4-mtp": MODEL_GEMMA_4,
    "gemma_4": MODEL_GEMMA_4,
    "gemma": MODEL_GEMMA_4,
    "high-throughput": MODEL_GEMMA_4,
    "mtp": MODEL_GEMMA_4,
}


def resolve_model_name(model_name_or_alias: Optional[str]) -> str:
    """Resolve user-friendly model aliases and presets to canonical model names."""
    if not model_name_or_alias or not str(model_name_or_alias).strip():
        return DEFAULT_MODEL
    cleaned = str(model_name_or_alias).strip().lower()
    return MODEL_PRESETS.get(cleaned, str(model_name_or_alias).strip())


# ---------------------------------------------------------------------------
# Mentalese S-Expression System Prompt with 1-Shot In-Context Demonstration
# ---------------------------------------------------------------------------

DEFAULT_UNSLOTH_SYSTEM_PROMPT = """You are the QUANTA Neural Discourse Transducer, a high-precision neuro-symbolic semantic extractor.
Your task is to extract an Entity-Event Directed Acyclic Graph (DAG) from discourse in any language into a strict S-expression conforming to the formal GBNF grammar.

RULES & SCHEMA CONSTRAINTS:
1. OUTPUT FORMAT:
   - Output MUST be a single parenthesized S-expression starting with `(graph ...)`
   - Do NOT emit Markdown formatting, backticks, commentary, or extraneous text.
2. PHYSICAL/AGENTIVE ENTITIES:
   - Permitted categories: PERSON, OBJECT, SUBSTANCE, LOCATION, ORGANIZATION, ANIMAL, ARTIFACT, NATURAL_OBJECT, INSTRUMENT.
   - Format: `(entity :id <id> :type <category> :label "<canonical_name>" :surface <surface_expr> [:props (...)])`
   - If an "ACTIVE ENTITIES:" manifest is provided, you MUST reuse existing IDs (e.g., E1, E2) whenever the text refers to them, their aliases, or pronouns referring to them.
   - Mint new IDs (E1, E2, ... or continuing beyond the highest existing ID) ONLY for genuinely novel entities.
3. EVENT PREDICATES:
   - Format: `(event :id <id> :pred <predicate> [:agent <id>] [:patient <id>] [:theme <id>] [:location <id>] [:instrument <id>] [:time "<time>"] [:tense <tense>] [:aspect <aspect>] [:polarity TRUE|FALSE] [:raw-text "<text>"])`
   - Link arguments directly to entity IDs. When an event takes another event as a clausal complement (e.g., pretend, know, believe, suspect, prove, want, obligate), set :theme <event_id> or :patient <event_id>.
4. SPATIO-TEMPORAL & CAUSAL RELATIONS:
   - Format: `(relation :type <rel_type> :source <id> :target <id> [:mechanism "<mech>"] [:confidence <num>])`
   - Permitted types: TEMP_ALLEN_MEETS, TEMP_ALLEN_BEFORE, TEMP_ALLEN_DURING, CAUSAL_MECHANISM_LINK, CAUSAL_PREVENTIVE_BLOCK.
5. EPISTEMIC PROPOSITIONS:
   - Format: `(proposition :id <id> :claim "<claim_text>" [:subject <id>] [:status <epistemic_status>] [:source <id>] [:event <id>])`
   - Permitted statuses: FACT, HYPOTHESIS, OBSERVATION, DOUBTED, PROHIBITED, BELIEF, KNOWLEDGE, UNVERIFIED.
6. MULTILINGUAL & CROSS-LINGUAL DISCOURSE:
   - Accept input in any human language (e.g., Hungarian, German, Turkish, Mandarin, English).
   - Universal English Pivot: Always map entity :label and event :pred to canonical English pivot words (e.g., :label "dog" for "kutya" or "狗", :pred bark for "ugat" or "吠").
   - Surface Retention: In the :surface field, always preserve the exact inflected surface word or phrase as it appeared in the source text (e.g., :surface "A kutya", :surface "a postást", :surface "die Probe").

DEMONSTRATION 1 (SIMPLE SVO & TIME):
[USER INPUT]
ACTIVE ENTITIES:
- E1: Dr. Marcus Vance (aliases: Marcus)
- E2: argon cylinder (aliases: cylinder)

CHUNK TEXT:
Dr. Marcus Vance pressurized the argon cylinder inside the test chamber at noon. He verified that the valve held.

[ASSISTANT RESPONSE]
(graph :chunk-id "demo_chunk"
  (entity :id E1 :type PERSON :label "Dr. Marcus Vance" :surface ("Marcus" "He"))
  (entity :id E2 :type OBJECT :label "argon cylinder" :surface "cylinder")
  (entity :id E3 :type LOCATION :label "test chamber" :surface "chamber")
  (event :id Ev1 :pred pressurize :agent E1 :patient E2 :location E3 :time "at noon" :tense PAST :polarity TRUE :raw-text "Dr. Marcus Vance pressurized the argon cylinder inside the test chamber at noon.")
  (event :id Ev2 :pred verify :agent E1 :time "immediately" :tense PAST :polarity TRUE :raw-text "He verified that the valve held.")
  (relation :type TEMP_ALLEN_MEETS :source Ev1 :target Ev2 :mechanism "sequential verification")
  (proposition :id P1 :claim "the valve held" :status FACT :source E1 :event Ev2)
)

DEMONSTRATION 2 (COMPLEX CLAUSAL & COUNTERFACTUAL DISCOURSE):
[USER INPUT]
CHUNK TEXT:
Had Alice not falsely pretended to know that Bob believed her investment was secure, the auditor wouldn't have sarcastically remarked that her due diligence was a stroke of genius.

[ASSISTANT RESPONSE]
(graph :chunk-id "complex_demo"
  (entity :id E1 :type PERSON :label "Alice" :surface ("Alice" "her"))
  (entity :id E2 :type PERSON :label "Bob" :surface "Bob")
  (entity :id E3 :type PERSON :label "auditor" :surface "auditor")
  (entity :id E4 :type OBJECT :label "investment" :surface "her investment")
  (entity :id E5 :type OBJECT :label "due diligence" :surface "her due diligence")
  (event :id Ev1 :pred pretend :agent E1 :theme Ev2 :polarity FALSE :modality "COUNTERFACTUAL" :raw-text "Had Alice not falsely pretended to know that Bob believed her investment was secure")
  (event :id Ev2 :pred know :agent E1 :theme Ev3 :tense PAST :polarity TRUE :raw-text "to know that Bob believed her investment was secure")
  (event :id Ev3 :pred believe :agent E2 :patient E4 :tense PAST :polarity TRUE :raw-text "Bob believed her investment was secure")
  (event :id Ev4 :pred remark :agent E3 :patient E1 :polarity FALSE :modality "COUNTERFACTUAL" :raw-text "the auditor wouldn't have sarcastically remarked that her due diligence was a stroke of genius")
  (relation :type CAUSAL_MECHANISM_LINK :source Ev1 :target Ev4 :mechanism "counterfactual condition")
  (proposition :id P1 :claim "her investment was secure" :status BELIEF :source E2 :event Ev3)
  (proposition :id P2 :claim "her due diligence was a stroke of genius" :status OBSERVATION :source E3 :event Ev4)
)

DEMONSTRATION 3 (AGGLUTINATIVE DISCOURSE / HUNGARIAN):
[USER INPUT]
CHUNK TEXT:
A kutya megugatta a postást a kertben.

[ASSISTANT RESPONSE]
(graph :chunk-id "demo_hu"
  (entity :id E1 :type ANIMAL :label "dog" :surface "A kutya")
  (entity :id E2 :type PERSON :label "postman" :surface "a postást")
  (entity :id E3 :type LOCATION :label "garden" :surface "a kertben")
  (event :id Ev1 :pred bark :agent E1 :patient E2 :location E3 :time "in the past" :tense PAST :polarity TRUE :raw-text "A kutya megugatta a postást a kertben.")
)

DEMONSTRATION 4 (ISOLATING DISCOURSE / MANDARIN):
[USER INPUT]
CHUNK TEXT:
科学家在实验室里合成了新型聚合物。

[ASSISTANT RESPONSE]
(graph :chunk-id "demo_zh"
  (entity :id E1 :type PERSON :label "scientist" :surface "科学家")
  (entity :id E2 :type LOCATION :label "laboratory" :surface "实验室")
  (entity :id E3 :type SUBSTANCE :label "polymer" :surface "新型聚合物")
  (event :id Ev1 :pred synthesize :agent E1 :patient E3 :location E2 :time "in the past" :tense PAST :polarity TRUE :raw-text "科学家在实验室里合成了新型聚合物。")
)

DEMONSTRATION 5 (FUSIONAL & COMPOUND DISCOURSE / GERMAN):
[USER INPUT]
CHUNK TEXT:
Der Forscher untersuchte die Probe im Laboratorium.

[ASSISTANT RESPONSE]
(graph :chunk-id "demo_de"
  (entity :id E1 :type PERSON :label "researcher" :surface "Der Forscher")
  (entity :id E2 :type OBJECT :label "sample" :surface "die Probe")
  (entity :id E3 :type LOCATION :label "laboratory" :surface "im Laboratorium")
  (event :id Ev1 :pred examine :agent E1 :patient E2 :location E3 :time "in the past" :tense PAST :polarity TRUE :raw-text "Der Forscher untersuchte die Probe im Laboratorium.")
)
"""


def _locate_gbnf_grammar(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Locate the quanta_asg.gbnf grammar specification file."""
    if custom_path is not None:
        p = Path(custom_path).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified GBNF grammar path does not exist: {custom_path}")

    # Search candidates
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "data" / "grammar" / "quanta_asg.gbnf",
        Path.cwd() / "data" / "grammar" / "quanta_asg.gbnf",
        here.parent.parent / "data" / "grammar" / "quanta_asg.gbnf",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    require_artifacts("data/grammar/quanta_asg.gbnf", component="UnslothTransducer")
    raise FileNotFoundError(
        "Could not locate 'quanta_asg.gbnf'. Ensure 'data/grammar/quanta_asg.gbnf' exists."
    )


def _format_entity_manifest(
    active_entities: Optional[List[EntityRecord]] = None,
    active_manifest_prompt: Optional[str] = None,
) -> Optional[str]:
    """Format active entity records into an 'ACTIVE ENTITIES:' prompt block."""
    if active_manifest_prompt and active_manifest_prompt.strip():
        return active_manifest_prompt.strip()

    if not active_entities:
        return None

    lines = ["ACTIVE ENTITIES:"]
    for ent in active_entities:
        if hasattr(ent, "get_display_aliases"):
            aliases = ent.get_display_aliases()
        else:
            aliases = getattr(ent, "surface_aliases", [])
        if aliases:
            lines.append(f"- {ent.canonical_id}: {ent.canonical_name} (aliases: {', '.join(aliases)})")
        else:
            lines.append(f"- {ent.canonical_id}: {ent.canonical_name}")

    return "\n".join(lines)


def _clean_sexpr_output(text: str) -> str:
    """Strip reasoning tags and code fences from raw model output."""
    clean = text.strip()
    # Strip <think>...</think> if model output reasoning tokens
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    # Strip ```lisp / ```scheme / ```sexpr / ``` fences
    fence_match = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)\s*```", clean)
    if fence_match:
        clean = fence_match.group(1).strip()
    elif clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean).strip()
    return clean


# ---------------------------------------------------------------------------
# Deterministic Offline Mock Transducer
# ---------------------------------------------------------------------------

class MockUnslothTransducer(BaseDiscourseTransducer):
    """Deterministic, high-performance offline S-expression transducer.

    Guarantees sub-5ms execution time for CI and offline environments without
    requiring a running GPU server or network access. Supports standard benchmarks
    (e.g., Eleanor Vance) and dynamically adapts entity IDs from active entity manifests.
    """

    def __init__(
        self,
        fixtures: Optional[Dict[str, Union[str, DiscourseExtractionResult]]] = None,
        system_prompt: str = DEFAULT_UNSLOTH_SYSTEM_PROMPT,
    ):
        super().__init__(system_prompt=system_prompt)
        self.fixtures: Dict[str, DiscourseExtractionResult] = {}

        # Register canonical gold-standard fixtures
        self.register_fixture("eleanor_vance", CANONICAL_ELEANOR_VANCE_FIXTURE)
        self.register_fixture(CANONICAL_ELEANOR_VANCE_TEXT, CANONICAL_ELEANOR_VANCE_FIXTURE)
        self.register_fixture("stress_1", CANONICAL_STRESS_1_FIXTURE)
        self.register_fixture(CANONICAL_STRESS_1_TEXT, CANONICAL_STRESS_1_FIXTURE)
        self.register_fixture("stress_2", CANONICAL_STRESS_2_FIXTURE)
        self.register_fixture(CANONICAL_STRESS_2_TEXT, CANONICAL_STRESS_2_FIXTURE)
        self.register_fixture("stress_3", CANONICAL_STRESS_3_FIXTURE)
        self.register_fixture(CANONICAL_STRESS_3_TEXT, CANONICAL_STRESS_3_FIXTURE)
        self.register_fixture("stress_4", CANONICAL_STRESS_4_FIXTURE)
        self.register_fixture(CANONICAL_STRESS_4_TEXT, CANONICAL_STRESS_4_FIXTURE)

        self.repair_fixtures: Dict[str, DiscourseExtractionResult] = {}
        self.repair_callback: Optional[Callable[..., Optional[Union[str, DiscourseExtractionResult]]]] = None
        self.realization_fixtures: Dict[str, str] = {}

        try:
            p = _locate_gbnf_grammar()
            self.grammar_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            self.grammar_hash = "mock_grammar_hash"

        if fixtures:
            for k, v in fixtures.items():
                self.register_fixture(k, v)

    def warm_grammar(self, url: Optional[str] = None) -> bool:
        """Mock implementation of grammar pre-warming."""
        return True

    def register_fixture(self, key: str, fixture: Union[str, DiscourseExtractionResult]):
        """Register a fixture keyed by string identifier or exact text."""
        if isinstance(fixture, str):
            parsed = parse_sexpr(fixture)
            self.fixtures[key] = parsed
        else:
            self.fixtures[key] = fixture

    def register_realization_fixture(self, key: str, text: str):
        """Register a reverse realization fixture keyed by S-expression, chunk-id, or identifier."""
        clean_key = " ".join(key.split()).strip()
        self.realization_fixtures[key] = text
        self.realization_fixtures[clean_key] = text

    def realize_text(
        self,
        sexpr_or_graph: Any,
        target_lang: str = "hungarian",
        **kwargs,
    ) -> str:
        """Deterministically realize an S-expression or QuantaGraph into fluent target-language text."""
        from parser.sexpr_parser import serialize_to_sexpr
        if isinstance(sexpr_or_graph, str):
            sexpr = sexpr_or_graph
        else:
            try:
                sexpr = serialize_to_sexpr(sexpr_or_graph, pretty=False)
            except Exception:
                sexpr = str(sexpr_or_graph)

        norm_sexpr = " ".join(sexpr.split()).strip()

        # 1. Exact match in realization fixtures
        if sexpr in self.realization_fixtures:
            return self.realization_fixtures[sexpr]
        if norm_sexpr in self.realization_fixtures:
            return self.realization_fixtures[norm_sexpr]

        # 2. Key/substring match in realization fixtures
        for k, v in self.realization_fixtures.items():
            if k in sexpr or k in norm_sexpr:
                return v

        # 3. Dynamic language heuristic fallback
        lang = target_lang.lower().strip()
        if lang in ("hu", "hungarian"):
            if "dog" in sexpr and "bark" in sexpr:
                return "A kutya megugatta a postást a kertben."
            if "polymer" in sexpr and ("synthesize" in sexpr or "synthes" in sexpr):
                return "Dr. Kovács János szintetizálta az új polimert a laboratóriumban."
            return "A kísérlet sikeres volt a laboratóriumban."
        elif lang in ("de", "german"):
            if "researcher" in sexpr or "sample" in sexpr or "examine" in sexpr:
                return "Der Forscher untersuchte die Probe im Laboratorium."
            return "Das Experiment war im Labor erfolgreich."
        elif lang in ("tr", "turkish"):
            if "dog" in sexpr or "bark" in sexpr or "postman" in sexpr:
                return "Köpek bahçede postacıya havladı."
            return "Deney laboratuvarda başarılı oldu."
        elif lang in ("zh", "mandarin", "chinese"):
            if "scientist" in sexpr or "polymer" in sexpr or "synthesize" in sexpr:
                return "科学家在实验室里合成了新型聚合物。"
            return "实验在实验室中成功完成。"
        elif lang in ("en", "english"):
            return "The experiment was successful in the laboratory."

        return f"[{target_lang.capitalize()} realization]: {norm_sexpr[:100]}"

    def register_repair_fixture(self, key: str, fixture: Union[str, DiscourseExtractionResult]):
        """Register a fixture to be returned when a repair request is received."""
        if isinstance(fixture, str):
            parsed = parse_sexpr(fixture)
            self.repair_fixtures[key] = parsed
        else:
            self.repair_fixtures[key] = fixture

    def transduce_raw(
        self,
        text: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        chunk_text: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Return raw S-expression string conforming to GBNF grammar."""
        content = text if text is not None else chunk_text
        if content is None:
            raise ValueError("Must provide either 'text' or 'chunk_text'")
        res = self.transduce(
            text=content,
            active_entities=active_entities,
            chunk_id=chunk_id,
            active_manifest_prompt=active_manifest_prompt,
            **kwargs,
        )
        return to_sexpr(res, pretty=True)

    def transduce(
        self,
        text: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        chunk_text: Optional[str] = None,
        **kwargs,
    ) -> DiscourseExtractionResult:
        """Transduce discourse text into typed DiscourseExtractionResult."""
        content = text if text is not None else chunk_text
        if content is None:
            raise ValueError("Must provide either 'text' or 'chunk_text'")
        t0 = time.perf_counter()
        norm_text = " ".join(content.split()).strip()

        matched: Optional[DiscourseExtractionResult] = None
        repair_req = kwargs.get("repair_request")

        # 0. Check repair callback or repair fixtures if repair_request is present
        if repair_req:
            if self.repair_callback:
                cb_res = self.repair_callback(content, **kwargs)
                if cb_res is not None:
                    if isinstance(cb_res, str):
                        matched = parse_sexpr(cb_res)
                    else:
                        matched = cb_res
            if matched is None:
                if content in self.repair_fixtures:
                    matched = self.repair_fixtures[content]
                elif norm_text in self.repair_fixtures:
                    matched = self.repair_fixtures[norm_text]
                else:
                    for k, fix in self.repair_fixtures.items():
                        if k.lower() in content.lower():
                            matched = fix
                            break

        # 1. Exact match in fixtures
        if matched is None:
            if content in self.fixtures:
                matched = self.fixtures[content]
            elif norm_text in self.fixtures:
                matched = self.fixtures[norm_text]

        # 2. Substring heuristics for canonical benchmarks
        if matched is None:
            lower = content.lower()
            if "eleanor" in lower or "containment cell" in lower or "synthetic compound" in lower:
                matched = CANONICAL_ELEANOR_VANCE_FIXTURE
            elif "alice" in lower and "auditor" in lower:
                matched = CANONICAL_STRESS_1_FIXTURE
            elif "drone" in lower and "airspace" in lower:
                matched = CANONICAL_STRESS_2_FIXTURE
            elif "investigator" in lower and "alibi" in lower:
                matched = CANONICAL_STRESS_3_FIXTURE
            elif "decree" in lower and "commissioner" in lower:
                matched = CANONICAL_STRESS_4_FIXTURE
            else:
                for key, fix in self.fixtures.items():
                    if key.lower() in lower:
                        matched = fix
                        break

        # Clone and customize result if matched
        if matched is not None:
            result = self._clone_and_adapt(matched, content, chunk_id, active_entities)
        else:
            # 3. Dynamic synthesis for novel text
            result = self._synthesize_dynamic(content, chunk_id, active_entities)
        latency = time.perf_counter() - t0
        result.metadata["latency_sec"] = latency
        result.metadata["backend"] = "mock_unsloth"
        result.metadata["model"] = "mock-unsloth-slm"
        return result

    async def transduce_async(
        self,
        text: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        chunk_text: Optional[str] = None,
        **kwargs,
    ) -> DiscourseExtractionResult:
        """Asynchronous execution wrapper for transduce."""
        return await asyncio.to_thread(
            self.transduce,
            text=text,
            active_entities=active_entities,
            chunk_id=chunk_id,
            active_manifest_prompt=active_manifest_prompt,
            chunk_text=chunk_text,
            **kwargs,
        )

    async def transduce_raw_async(
        self,
        text: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        chunk_text: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Asynchronous execution wrapper for transduce_raw."""
        return await asyncio.to_thread(
            self.transduce_raw,
            text=text,
            active_entities=active_entities,
            chunk_id=chunk_id,
            active_manifest_prompt=active_manifest_prompt,
            chunk_text=chunk_text,
            **kwargs,
        )

    def _clone_and_adapt(
        self,
        source: DiscourseExtractionResult,
        text: str,
        chunk_id: Optional[str],
        active_entities: Optional[List[EntityRecord]],
    ) -> DiscourseExtractionResult:
        """Clone source fixture and remap entity IDs if active entities match."""
        data = source.to_dict()
        if chunk_id:
            data["chunk_id"] = chunk_id

        res = DiscourseExtractionResult.from_dict(data)

        # If active_entities provided, reconcile matching entity IDs
        if active_entities:
            id_map: Dict[str, str] = {}
            for target_ent in res.entities:
                for active in active_entities:
                    matched = False
                    if active.canonical_name.lower() in target_ent.canonical_name.lower():
                        matched = True
                    for alias in active.surface_aliases:
                        if alias.lower() in [a.lower() for a in target_ent.surface_aliases]:
                            matched = True
                            break
                    if matched:
                        id_map[target_ent.id] = active.canonical_id
                        target_ent.id = active.canonical_id
                        break

            # Remap foreign keys in events
            if id_map:
                for ev in res.events:
                    if ev.agent_id in id_map:
                        ev.agent_id = id_map[ev.agent_id]
                    if ev.patient_id in id_map:
                        ev.patient_id = id_map[ev.patient_id]
                    if ev.theme_id in id_map:
                        ev.theme_id = id_map[ev.theme_id]
                    if ev.location_id in id_map:
                        ev.location_id = id_map[ev.location_id]
                    if ev.instrument_id in id_map:
                        ev.instrument_id = id_map[ev.instrument_id]

        return res

    def _synthesize_dynamic(
        self,
        text: str,
        chunk_id: Optional[str],
        active_entities: Optional[List[EntityRecord]],
    ) -> DiscourseExtractionResult:
        """Dynamically generate a valid S-expression extraction for unknown text."""
        words = text.split()
        first_cap = next((w.strip(".,;:\"'!?") for w in words if w and w[0].isupper() and len(w) > 1), "Agent")

        # Check if active entities match text
        matched_active: Optional[EntityRecord] = None
        if active_entities:
            text_lower = text.lower()
            for ent in active_entities:
                if ent.canonical_name.lower() in text_lower:
                    matched_active = ent
                    break
                for alias in ent.surface_aliases:
                    if alias.lower() in text_lower:
                        matched_active = ent
                        break
                if matched_active:
                    break

        if matched_active is not None:
            ent_id = matched_active.canonical_id
            ent_name = matched_active.canonical_name
            ent_category = matched_active.category
            aliases = list(matched_active.surface_aliases)
        else:
            ent_id = "E1"
            ent_name = first_cap
            ent_category = "PERSON"
            aliases = [first_cap]

        entities = [
            ExtractedEntity(
                id=ent_id,
                canonical_name=ent_name,
                category=ent_category,
                surface_aliases=aliases,
            )
        ]

        # Extract event predicate from verb-like token or default to observe
        events = [
            ExtractedEvent(
                id="Ev1",
                predicate="observe",
                agent_id=ent_id,
                temporal_anchor="present",
                tense="PAST",
                polarity=True,
                raw_text=text[:120].strip(),
            )
        ]

        propositions = [
            ExtractedProposition(
                id="P1",
                claim_text=text[:80].strip(),
                epistemic_status="FACT",
                source_agent_id=ent_id,
                event_id="Ev1",
            )
        ]

        return DiscourseExtractionResult(
            chunk_id=chunk_id or "dynamic_chunk",
            entities=entities,
            events=events,
            relations=[],
            propositions=propositions,
            metadata={"transducer": "MockUnslothTransducer_dynamic"},
        )


# Alias for backward compatibility
MockSExprTransducer = MockUnslothTransducer


# ---------------------------------------------------------------------------
# Production Unsloth Transducer Client
# ---------------------------------------------------------------------------

class UnslothTransducer(BaseDiscourseTransducer):
    """Production client for local Small Language Models (Qwen 3.5 / Gemma 4) via Unsloth.

    Connects to http://localhost:8888/v1 (OpenAI-compatible chat completions)
    with fallback to http://localhost:1234/v1, native GBNF grammar injection
    (extra_body={"grammar": ...}), active entity manifest injection, retry backoff,
    health checking, and automatic mock fallback.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        grammar_path: Optional[Union[str, Path]] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        api_key: str = "unsloth",
        session: Optional[requests.Session] = None,
        system_prompt: Optional[str] = None,
        fallback_to_mock: bool = True,
        mock_transducer: Optional[MockUnslothTransducer] = None,
        fallback_base_url: Optional[str] = None,
    ):
        super().__init__(system_prompt=system_prompt or DEFAULT_UNSLOTH_SYSTEM_PROMPT)
        self.base_url = (
            base_url
            or os.environ.get("UNSLOTH_BASE_URL")
            or "http://localhost:8888/v1"
        ).rstrip("/")
        fallback_candidate = (
            fallback_base_url
            if fallback_base_url is not None
            else os.environ.get("UNSLOTH_FALLBACK_URL", "http://localhost:1234/v1")
        )
        self.fallback_base_url = fallback_candidate.rstrip("/") if fallback_candidate else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.api_key = api_key
        self.session = session or requests.Session()
        self.fallback_to_mock = fallback_to_mock
        self._mock = mock_transducer or MockUnslothTransducer(system_prompt=self.system_prompt)
        self._last_fallback_used = False

        # 1. Load GBNF Grammar & Pre-Warm State Machine
        self.grammar_path = _locate_gbnf_grammar(grammar_path)
        self.grammar_content = self.grammar_path.read_text(encoding="utf-8")
        self.grammar_hash = hashlib.sha256(self.grammar_content.encode("utf-8")).hexdigest()
        self._compiled_grammar_rules = self._precompile_grammar_rules(self.grammar_content)
        self._grammar_warmed = False

        # 2. Model resolution (lazy default to prevent eager network calls during init)
        self.model = resolve_model_name(model)

    def set_model(self, model: str) -> str:
        """Set active model profile at runtime, resolving aliases and presets."""
        self.model = resolve_model_name(model)
        return self.model

    @staticmethod
    def _precompile_grammar_rules(content: str) -> Dict[str, str]:
        """Pre-parse GBNF grammar productions into cached rule table."""
        rules: Dict[str, str] = {}
        for line in content.splitlines():
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            if "::=" in line_str:
                parts = line_str.split("::=", 1)
                rules[parts[0].strip()] = parts[1].strip()
        return rules

    def warm_grammar(self, base_url: Optional[str] = None) -> bool:
        """Pre-warm grammar state machine on server or verify local pre-compilation."""
        if not self._compiled_grammar_rules:
            return False
        self._grammar_warmed = True
        target = base_url or self.base_url
        if self.check_health(target):
            try:
                headers = self._get_headers()
                test_payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "grammar": self.grammar_content,
                    "grammar_hash": self.grammar_hash,
                }
                resp = self.session.post(
                    f"{target.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=test_payload,
                    timeout=2.0,
                )
                return resp.status_code in (200, 400)
            except Exception:
                pass
        return True

    def _get_headers(self) -> Dict[str, str]:
        """Compose request headers, omitting Authorization for keyless local endpoints."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "unsloth":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _detect_model(self, base_url: Optional[str] = None) -> str:
        """Query /models to detect active model or default to Qwen 3.5 4B."""
        target = (base_url or self.base_url).rstrip("/")
        url = f"{target}/models"
        headers: Dict[str, str] = {}
        if self.api_key and self.api_key != "unsloth":
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = self.session.get(url, headers=headers, timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", []) if "id" in m]
                for m in models:
                    if "qwen3.5-4b" in m.lower():
                        return m
                for m in models:
                    if "qwen" in m.lower() or "gemma" in m.lower():
                        return m
                if models:
                    return models[0]
        except Exception:
            pass
        return DEFAULT_MODEL

    def check_health(self, url: Optional[str] = None) -> bool:
        """Check if Unsloth endpoint is reachable and responsive."""
        target = (url or self.base_url).rstrip("/")
        endpoint = f"{target}/models"
        headers: Dict[str, str] = {}
        if self.api_key and self.api_key != "unsloth":
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = self.session.get(endpoint, headers=headers, timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        """Alias for check_health(). Checks primary and secondary endpoints."""
        if self.check_health(self.base_url):
            return True
        if self.fallback_base_url and self.check_health(self.fallback_base_url):
            return True
        return False

    def _build_system_prompt(
        self,
        active_entities: Optional[List[EntityRecord]] = None,
        active_manifest_prompt: Optional[str] = None,
    ) -> str:
        """Compose system prompt optionally embedding active entity manifest."""
        manifest_text = _format_entity_manifest(active_entities, active_manifest_prompt)
        if manifest_text:
            return f"{self.system_prompt}\n\nCURRENT CONTEXT:\n{manifest_text}"
        return self.system_prompt

    def _build_user_prompt(
        self,
        chunk_text: str,
        active_entities: Optional[List[EntityRecord]] = None,
        active_manifest_prompt: Optional[str] = None,
        repair_request: Optional[str] = None,
    ) -> str:
        """Compose user prompt containing discourse text, active entity manifest, and optional repair request."""
        parts: List[str] = []
        manifest_text = _format_entity_manifest(active_entities, active_manifest_prompt)
        if manifest_text:
            parts.append(manifest_text)
        parts.append(f"CHUNK TEXT:\n{chunk_text.strip()}")
        if repair_request and repair_request.strip():
            parts.append(repair_request.strip())
        return "\n\n".join(parts)

    def _build_payload(
        self,
        text: str,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Construct OpenAI-compatible request payload with GBNF grammar injection."""
        repair_req = kwargs.get("repair_request")
        system_content = self._build_system_prompt(active_entities, active_manifest_prompt)
        user_content = self._build_user_prompt(
            text, active_entities, active_manifest_prompt, repair_request=repair_req
        )
        target_model = resolve_model_name(kwargs.get("model") or self.model)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        extra_body = {
            "grammar": self.grammar_content,
            "grammar_hash": self.grammar_hash,
            "guided_grammar": self.grammar_content,
            **kwargs.get("extra_body", {}),
        }

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "grammar": self.grammar_content,
            "grammar_hash": self.grammar_hash,
            "extra_body": extra_body,
        }

        for opt_key in ("top_p", "seed", "stop", "presence_penalty", "frequency_penalty"):
            if opt_key in kwargs:
                payload[opt_key] = kwargs[opt_key]

        return payload

    def transduce_raw(
        self,
        text: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        chunk_text: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Send inference request to Unsloth server and return raw S-expression string."""
        content = text if text is not None else chunk_text
        if content is None:
            raise ValueError("Must provide either 'text' or 'chunk_text'")

        headers = self._get_headers()
        payload = self._build_payload(
            text=content,
            active_entities=active_entities,
            chunk_id=chunk_id,
            active_manifest_prompt=active_manifest_prompt,
            **kwargs,
        )

        candidate_urls: List[str] = [self.base_url]
        if self.fallback_base_url and self.fallback_base_url != self.base_url:
            candidate_urls.append(self.fallback_base_url)

        last_err: Optional[Exception] = None
        raw_sexpr: Optional[str] = None

        for base_url in candidate_urls:
            url = f"{base_url}/chat/completions"
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = self.session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content_str = data["choices"][0]["message"]["content"]
                        raw_sexpr = _clean_sexpr_output(content_str)
                        break
                    elif resp.status_code in {500, 502, 503, 504}:
                        last_err = RuntimeError(f"Server error {resp.status_code}: {resp.text}")
                    else:
                        resp.raise_for_status()
                except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                    last_err = e

                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (2 ** (attempt - 1)))

            if raw_sexpr is not None:
                break

        if raw_sexpr is None:
            if self.fallback_to_mock:
                self._last_fallback_used = True
                logger.warning(
                    "Unsloth servers at %s unreachable (%s); falling back to MockUnslothTransducer",
                    candidate_urls,
                    last_err,
                )
                return self._mock.transduce_raw(
                    text=content,
                    active_entities=active_entities,
                    chunk_id=chunk_id,
                    active_manifest_prompt=active_manifest_prompt,
                    chunk_text=chunk_text,
                    **kwargs,
                )
            raise RuntimeError(
                f"Failed to extract S-expression from Unsloth server at {self.base_url}: {last_err}"
            ) from last_err

        self._last_fallback_used = False
        return raw_sexpr

    def transduce(
        self,
        text: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        chunk_text: Optional[str] = None,
        **kwargs,
    ) -> DiscourseExtractionResult:
        """Transduce discourse text into typed DiscourseExtractionResult."""
        content = text if text is not None else chunk_text
        if content is None:
            raise ValueError("Must provide either 'text' or 'chunk_text'")

        t0 = time.perf_counter()
        try:
            self._last_fallback_used = False
            raw_sexpr = self.transduce_raw(
                text=content,
                active_entities=active_entities,
                chunk_id=chunk_id,
                active_manifest_prompt=active_manifest_prompt,
                chunk_text=chunk_text,
                **kwargs,
            )
            result = parse_sexpr(raw_sexpr)
            if chunk_id and not result.chunk_id:
                result.chunk_id = chunk_id
            latency = time.perf_counter() - t0
            result.metadata["latency_sec"] = latency
            if self._last_fallback_used:
                result.metadata["backend"] = "mock_unsloth"
                result.metadata["fallback_from_unsloth"] = True
                result.metadata["model"] = "mock-unsloth-slm"
            else:
                result.metadata["backend"] = "unsloth"
                result.metadata["model"] = resolve_model_name(kwargs.get("model") or self.model)
            return result
        except Exception as e:
            if self.fallback_to_mock:
                self._last_fallback_used = True
                logger.warning(
                    "Unsloth transduction failed (%s); falling back to MockUnslothTransducer",
                    e,
                )
                res = self._mock.transduce(
                    text=content,
                    active_entities=active_entities,
                    chunk_id=chunk_id,
                    active_manifest_prompt=active_manifest_prompt,
                    chunk_text=chunk_text,
                    **kwargs,
                )
                res.metadata["fallback_from_unsloth"] = True
                return res
            raise

    async def transduce_async(
        self,
        text: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        chunk_text: Optional[str] = None,
        **kwargs,
    ) -> DiscourseExtractionResult:
        """Asynchronous execution wrapper for transduce."""
        return await asyncio.to_thread(
            self.transduce,
            text=text,
            active_entities=active_entities,
            chunk_id=chunk_id,
            active_manifest_prompt=active_manifest_prompt,
            chunk_text=chunk_text,
            **kwargs,
        )

    async def transduce_raw_async(
        self,
        text: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        chunk_text: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Asynchronous execution wrapper for transduce_raw."""
        return await asyncio.to_thread(
            self.transduce_raw,
            text=text,
            active_entities=active_entities,
            chunk_id=chunk_id,
            active_manifest_prompt=active_manifest_prompt,
            chunk_text=chunk_text,
            **kwargs,
        )

    def realize_text(
        self,
        sexpr_or_graph: Any,
        target_lang: str = "hungarian",
        **kwargs,
    ) -> str:
        """Realize an S-expression or QuantaGraph into fluent target-language text via SLM."""
        from parser.sexpr_parser import serialize_to_sexpr
        if isinstance(sexpr_or_graph, str):
            sexpr = sexpr_or_graph
        else:
            try:
                sexpr = serialize_to_sexpr(sexpr_or_graph, pretty=True)
            except Exception:
                sexpr = str(sexpr_or_graph)

        lang_title = target_lang.strip().capitalize()
        system_content = (
            "You are the QUANTA Neural Realizer. Your task is to realize formal Mentalese "
            f"S-expressions into fluent, natural {lang_title} text. "
            "Output ONLY the realized sentence without explanations, quotes, or markdown."
        )
        user_content = f"Realize this semantic graph as fluent {lang_title} text:\n\n{sexpr}"
        target_model = resolve_model_name(kwargs.get("model") or self.model)
        headers = self._get_headers()
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 512),
        }

        candidate_urls: List[str] = [self.base_url]
        if self.fallback_base_url and self.fallback_base_url != self.base_url:
            candidate_urls.append(self.fallback_base_url)

        last_err: Optional[Exception] = None
        realized: Optional[str] = None

        for base_url in candidate_urls:
            url = f"{base_url}/chat/completions"
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = self.session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        out_str = data["choices"][0]["message"]["content"]
                        realized = _clean_sexpr_output(out_str)
                        break
                    elif resp.status_code in {500, 502, 503, 504}:
                        last_err = RuntimeError(f"Server error {resp.status_code}: {resp.text}")
                    else:
                        resp.raise_for_status()
                except Exception as e:
                    last_err = e

                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (2 ** (attempt - 1)))

            if realized is not None:
                break

        if realized is None:
            if self.fallback_to_mock:
                self._last_fallback_used = True
                return self._mock.realize_text(sexpr_or_graph, target_lang=target_lang, **kwargs)
            raise RuntimeError(f"Failed to realize text with Unsloth SLM: {last_err}") from last_err

        return realized

    async def realize_text_async(
        self,
        sexpr_or_graph: Any,
        target_lang: str = "hungarian",
        **kwargs,
    ) -> str:
        """Asynchronous execution wrapper for realize_text."""
        return await asyncio.to_thread(
            self.realize_text,
            sexpr_or_graph=sexpr_or_graph,
            target_lang=target_lang,
            **kwargs,
        )
