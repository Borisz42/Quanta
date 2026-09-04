"""Neural Discourse Transducer for QUANTA (Phase 3).

Translates open-domain English discourse chunks into normalized intermediate JSON
(Entities with mentions, Events with E-IDs, Allen Intervals, Causal Links,
and Epistemic Propositions) for downstream consumption by the Symbolic ASG Compiler.

Supported backends:
1. LMStudioTransducer: Connects to local OpenAI-compatible REST endpoint
   (e.g., Qwen-4B via LM Studio on http://localhost:1234/v1).
2. LocalGGUFTransducer: Direct in-process execution using llama-cpp-python.
3. MockTransducer: Deterministic offline mock with pre-recorded fixtures for CI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union

import requests

from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 3.5: Optimized System Prompt with 1-Shot In-Context Demonstration
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are the QUANTA Neural Discourse Transducer, a high-precision neuro-symbolic semantic extractor.
Your task is to extract an Entity-Event Directed Acyclic Graph (DAG) from English discourse chunks into strict, canonical JSON.

RULES & SCHEMA CONSTRAINTS:
1. PHYSICAL/AGENTIVE ENTITIES ONLY:
   - Extract real-world referents into "entities": people, substances, physical objects, locations, institutions.
   - Permitted categories: PERSON, OBJECT, SUBSTANCE, LOCATION, ORGANIZATION, ANIMAL, ARTIFACT, NATURAL_OBJECT.
   - DO NOT create entities for abstract propositions, hypotheses, thoughts, or events (e.g., "the discovery", "expansion", "audit" are NOT entities).
2. ACTIVE ENTITY MANIFEST & COREFERENCE:
   - If an "ACTIVE ENTITIES:" manifest is provided, you MUST reuse existing IDs (e.g. E1, E2) whenever the text refers to them, their aliases, or pronouns referring to them.
   - Consolidate synonyms: append newly observed surface forms or pronouns into "surface_aliases" for that entity.
   - Mint new IDs (E1, E2, ... or continuing from the highest ID) ONLY for genuinely novel entities not present in the manifest.
3. EVENT PREDICATES:
   - Extract discrete physical, cognitive, communicative, or relational actions into "events".
   - "predicate": Root verb lemma in lowercase (e.g., "isolate", "note", "doubt", "verify", "retain", "prohibit").
   - Explicit Foreign Keys: Link arguments directly to entity IDs:
     * "agent_id": Entity performing the action (Band 1 VAL_X1_AGENT).
     * "patient_id": Entity affected by the action (Band 1 VAL_X2_PATIENT).
     * "location_id": Entity serving as location/container (Band 1 VAL_LOCATION_SLOT).
     * "instrument_id": Entity used as instrument (Band 1 VAL_X5_INSTRUMENT).
     * Set to null if an argument is absent.
   - "temporal_anchor": Surface temporal phrase (e.g. "at dawn", "three hours later").
   - "tense": "PAST", "PRESENT", "FUTURE", "PAST_PERFECT", etc.
   - "polarity": true if affirmative, false if negated.
4. SPATIO-TEMPORAL & CAUSAL RELATIONS:
   - Link events together via formal Band 7 relations in "relations":
     * Allen Temporal Intervals: "TEMP_ALLEN_MEETS" (immediately followed), "TEMP_ALLEN_BEFORE" (precedes with gap), "TEMP_ALLEN_DURING" (concurrent).
     * Pearl Causal Links: "CAUSAL_MECHANISM_LINK" (direct cause/prompting), "CAUSAL_PREVENTIVE_BLOCK" (inhibition/prevention).
5. EPISTEMIC PROPOSITIONS:
   - Extract abstract claims, hypotheses, beliefs, and conditions into "propositions".
   - "epistemic_status": "FACT", "HYPOTHESIS", "OBSERVATION", "DOUBTED", "PROHIBITED".
   - "source_agent_id": Entity ID asserting or holding the belief (or null).

OUTPUT FORMAT: Strict JSON matching the schema with keys "entities", "events", "relations", "propositions". No commentary or extra text.

ONE-SHOT DEMONSTRATION:
[USER INPUT]
ACTIVE ENTITIES:
- E1: Dr. Marcus Vance (aliases: Marcus)
- E2: argon cylinder (aliases: cylinder)

CHUNK TEXT:
Dr. Marcus Vance pressurized the argon cylinder inside the test chamber at noon. He verified that the valve held.

[ASSISTANT RESPONSE]
{
  "entities": [
    {"id": "E1", "canonical_name": "Dr. Marcus Vance", "category": "PERSON", "surface_aliases": ["Marcus", "He"], "properties": {}},
    {"id": "E2", "canonical_name": "argon cylinder", "category": "OBJECT", "surface_aliases": ["cylinder"], "properties": {}},
    {"id": "E3", "canonical_name": "test chamber", "category": "LOCATION", "surface_aliases": ["chamber"], "properties": {}}
  ],
  "events": [
    {
      "id": "Ev1",
      "predicate": "pressurize",
      "agent_id": "E1",
      "patient_id": "E2",
      "location_id": "E3",
      "temporal_anchor": "at noon",
      "tense": "PAST",
      "polarity": true,
      "raw_text": "Dr. Marcus Vance pressurized the argon cylinder inside the test chamber at noon."
    },
    {
      "id": "Ev2",
      "predicate": "verify",
      "agent_id": "E1",
      "patient_id": null,
      "temporal_anchor": null,
      "tense": "PAST",
      "polarity": true,
      "raw_text": "He verified that the valve held."
    }
  ],
  "relations": [
    {
      "relation_type": "TEMP_ALLEN_MEETS",
      "source_id": "Ev1",
      "target_id": "Ev2",
      "mechanism": "sequential verification"
    }
  ],
  "propositions": [
    {
      "id": "P1",
      "claim_text": "the valve held",
      "predicate": "hold",
      "epistemic_status": "FACT",
      "source_agent_id": "E1",
      "event_id": "Ev2"
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Canonical Eleanor Vance Fixture (Phase 3.6 / Phase 4 Gold Standard)
# ---------------------------------------------------------------------------

CANONICAL_ELEANOR_VANCE_TEXT = (
    "Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn. "
    "She immediately noted that this specimen exhibited anomalous crystalline lattice expansion, which "
    "strongly suggested an unobserved phase transition. Although her supervisor initially doubted the validity "
    "of the discovery, Eleanor verified the hypothesis three hours later by replicating the transformation "
    "within the same vessel. The resulting polymer retained its structural integrity throughout the afternoon, "
    "prompting the laboratory director to prohibit all competing tests until her synthesis protocol could be formally audited."
)

CANONICAL_ELEANOR_VANCE_FIXTURE = DiscourseExtractionResult(
    chunk_id="chunk_eleanor_vance",
    entities=[
        ExtractedEntity(
            id="E1",
            canonical_name="Dr. Eleanor Vance",
            category="PERSON",
            surface_aliases=["Eleanor Vance", "Eleanor", "Vance", "she", "her"],
            properties={"title": "Dr.", "role": "researcher"},
        ),
        ExtractedEntity(
            id="E2",
            canonical_name="synthetic compound",
            category="SUBSTANCE",
            surface_aliases=["volatile synthetic compound", "specimen", "polymer"],
            properties={"state": "volatile"},
        ),
        ExtractedEntity(
            id="E3",
            canonical_name="cryogenic containment cell",
            category="LOCATION",
            surface_aliases=["containment cell", "vessel", "cell", "same vessel"],
            properties={"type": "containment"},
        ),
        ExtractedEntity(
            id="E4",
            canonical_name="supervisor",
            category="PERSON",
            surface_aliases=["her supervisor"],
            properties={"role": "supervisor"},
        ),
        ExtractedEntity(
            id="E5",
            canonical_name="laboratory director",
            category="PERSON",
            surface_aliases=["director"],
            properties={"role": "director"},
        ),
    ],
    events=[
        ExtractedEvent(
            id="Ev1",
            predicate="isolate",
            agent_id="E1",
            patient_id="E2",
            location_id="E3",
            temporal_anchor="at dawn",
            tense="PAST",
            polarity=True,
            raw_text="Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn.",
        ),
        ExtractedEvent(
            id="Ev2",
            predicate="note",
            agent_id="E1",
            patient_id=None,
            location_id=None,
            temporal_anchor="immediately",
            tense="PAST",
            polarity=True,
            raw_text="She immediately noted that this specimen exhibited anomalous crystalline lattice expansion, which strongly suggested an unobserved phase transition.",
        ),
        ExtractedEvent(
            id="Ev3",
            predicate="doubt",
            agent_id="E4",
            patient_id=None,
            temporal_anchor="initially",
            tense="PAST",
            polarity=True,
            raw_text="Although her supervisor initially doubted the validity of the discovery",
        ),
        ExtractedEvent(
            id="Ev4",
            predicate="verify",
            agent_id="E1",
            patient_id=None,
            location_id="E3",
            temporal_anchor="three hours later",
            tense="PAST",
            polarity=True,
            raw_text="Eleanor verified the hypothesis three hours later by replicating the transformation within the same vessel.",
        ),
        ExtractedEvent(
            id="Ev5",
            predicate="retain",
            agent_id="E2",
            patient_id=None,
            temporal_anchor="throughout the afternoon",
            tense="PAST",
            polarity=True,
            raw_text="The resulting polymer retained its structural integrity throughout the afternoon",
        ),
        ExtractedEvent(
            id="Ev6",
            predicate="prohibit",
            agent_id="E5",
            patient_id=None,
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="prompting the laboratory director to prohibit all competing tests until her synthesis protocol could be formally audited.",
        ),
    ],
    relations=[
        ExtractedRelation(
            relation_type="TEMP_ALLEN_MEETS",
            source_id="Ev1",
            target_id="Ev2",
            mechanism="immediately after isolation",
        ),
        ExtractedRelation(
            relation_type="TEMP_ALLEN_BEFORE",
            source_id="Ev1",
            target_id="Ev4",
            mechanism="three hours later",
        ),
        ExtractedRelation(
            relation_type="TEMP_ALLEN_BEFORE",
            source_id="Ev3",
            target_id="Ev4",
            mechanism="initially doubted before verification",
        ),
        ExtractedRelation(
            relation_type="TEMP_ALLEN_DURING",
            source_id="Ev5",
            target_id="Ev5",
            mechanism="throughout the afternoon",
        ),
        ExtractedRelation(
            relation_type="CAUSAL_MECHANISM_LINK",
            source_id="Ev5",
            target_id="Ev6",
            mechanism="retention prompted prohibition",
        ),
    ],
    propositions=[
        ExtractedProposition(
            id="P1",
            claim_text="specimen exhibited anomalous crystalline lattice expansion",
            epistemic_status="OBSERVATION",
            subject_id="E2",
            event_id="Ev2",
        ),
        ExtractedProposition(
            id="P2",
            claim_text="unobserved phase transition occurred",
            epistemic_status="HYPOTHESIS",
            subject_id="E2",
            event_id="Ev2",
        ),
        ExtractedProposition(
            id="P3",
            claim_text="validity of the discovery",
            epistemic_status="DOUBTED",
            source_agent_id="E4",
            event_id="Ev3",
        ),
        ExtractedProposition(
            id="P4",
            claim_text="hypothesis verified via replication",
            epistemic_status="FACT",
            source_agent_id="E1",
            event_id="Ev4",
        ),
        ExtractedProposition(
            id="P5",
            claim_text="synthesis protocol audited",
            epistemic_status="PROHIBITED",
            source_agent_id="E5",
            event_id="Ev6",
        ),
    ],
    metadata={"provenance": "canonical_fixture", "source": "Eleanor Vance Benchmark"},
)


# ---------------------------------------------------------------------------
# Base Transducer Abstract Class
# ---------------------------------------------------------------------------

class BaseDiscourseTransducer(ABC):
    """Abstract base class for all Neural Discourse Transducers."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def _build_user_prompt(
        self,
        chunk_text: str,
        active_manifest_prompt: Optional[str] = None,
    ) -> str:
        """Compose user prompt including active entity manifest and discourse text."""
        parts: List[str] = []
        if active_manifest_prompt and active_manifest_prompt.strip():
            parts.append(active_manifest_prompt.strip())
            parts.append("")
        parts.append(f"CHUNK TEXT:\n{chunk_text.strip()}")
        return "\n".join(parts)

    @abstractmethod
    def transduce(
        self,
        chunk_text: str,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        **kwargs,
    ) -> DiscourseExtractionResult:
        """Transduce a discourse chunk into normalized intermediate JSON schema."""
        pass


# ---------------------------------------------------------------------------
# 3.4: Deterministic Mock Transducer (Fast Offline Testing)
# ---------------------------------------------------------------------------

class MockTransducer(BaseDiscourseTransducer):
    """Deterministic mock transducer for offline unit and regression testing.

    Returns canonical pre-recorded fixtures when recognized patterns match,
    or generates safe structured representations for arbitrary chunks.
    """

    def __init__(
        self,
        fixtures: Optional[Dict[str, DiscourseExtractionResult]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        super().__init__(system_prompt=system_prompt)
        self.fixtures: Dict[str, DiscourseExtractionResult] = {}
        # Pre-seed canonical Eleanor Vance fixture
        self.register_fixture("eleanor_vance", CANONICAL_ELEANOR_VANCE_FIXTURE)
        self.register_fixture(CANONICAL_ELEANOR_VANCE_TEXT, CANONICAL_ELEANOR_VANCE_FIXTURE)

        if fixtures:
            for k, v in fixtures.items():
                self.register_fixture(k, v)

    def register_fixture(self, key: str, fixture: DiscourseExtractionResult):
        """Register a fixture keyed by string identifier or exact text snippet."""
        self.fixtures[key] = fixture

    def transduce(
        self,
        chunk_text: str,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        **kwargs,
    ) -> DiscourseExtractionResult:
        """Return fixture if matched, or generate synthetic extraction result."""
        # 1. Exact match in fixtures
        if chunk_text in self.fixtures:
            result = self._clone_result(self.fixtures[chunk_text], chunk_id)
            return result

        # 2. Key/substring match
        for key, fix in self.fixtures.items():
            if key.lower() in chunk_text.lower() or ("eleanor" in key.lower() and "eleanor" in chunk_text.lower()):
                result = self._clone_result(fix, chunk_id)
                return result

        # 3. Fallback heuristic extraction for arbitrary chunks
        return self._heuristic_fallback(chunk_text, chunk_id, active_manifest_prompt)

    def _clone_result(
        self,
        source: DiscourseExtractionResult,
        chunk_id: Optional[str],
    ) -> DiscourseExtractionResult:
        """Deep clone a fixture result and assign target chunk_id."""
        data = source.to_dict()
        if chunk_id:
            data["chunk_id"] = chunk_id
        return DiscourseExtractionResult.from_dict(data)

    def _heuristic_fallback(
        self,
        chunk_text: str,
        chunk_id: Optional[str],
        active_manifest_prompt: Optional[str],
    ) -> DiscourseExtractionResult:
        """Generate a valid, deterministic extraction result for arbitrary text."""
        words = chunk_text.split()
        first_cap = next((w.strip(".,;:\"'") for w in words if w and w[0].isupper()), "Agent")
        
        entities = [
            ExtractedEntity(
                id="E1",
                canonical_name=first_cap,
                category="PERSON",
                surface_aliases=[first_cap],
            )
        ]
        events = [
            ExtractedEvent(
                id="Ev1",
                predicate="observe",
                agent_id="E1",
                temporal_anchor="present",
                tense="PAST",
                polarity=True,
                raw_text=chunk_text[:120],
            )
        ]
        relations: List[ExtractedRelation] = []
        propositions = [
            ExtractedProposition(
                id="P1",
                claim_text=chunk_text[:80],
                epistemic_status="FACT",
                source_agent_id="E1",
                event_id="Ev1",
            )
        ]

        return DiscourseExtractionResult(
            chunk_id=chunk_id or "heuristic_chunk",
            entities=entities,
            events=events,
            relations=relations,
            propositions=propositions,
            metadata={"transducer": "MockTransducer_heuristic"},
        )


# ---------------------------------------------------------------------------
# 3.2: LMStudioTransducer (Local OpenAI-Compatible REST Client)
# ---------------------------------------------------------------------------

class LMStudioTransducer(BaseDiscourseTransducer):
    """Client for local quantized models running in LM Studio via REST API.

    Connects to http://localhost:1234/v1/chat/completions (OpenAI compatible).
    Enforces temperature 0.0, retry logic, connection health checks, and JSON mode.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen2.5-4b-instruct",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        super().__init__(system_prompt=system_prompt)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.api_key = api_key or "lm-studio"
        self.session = session or requests.Session()

    def check_health(self) -> bool:
        """Check if LM Studio endpoint is reachable and responsive."""
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = self.session.get(url, headers=headers, timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        """Alias for check_health()."""
        return self.check_health()

    def transduce(
        self,
        chunk_text: str,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        **kwargs,
    ) -> DiscourseExtractionResult:
        """Call LM Studio endpoint to extract discourse graph as JSON."""
        user_prompt = self._build_user_prompt(chunk_text, active_manifest_prompt)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        last_err: Optional[Exception] = None
        raw_text_resp = ""
        t0 = time.perf_counter()

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
                    raw_text_resp = data["choices"][0]["message"]["content"]
                    break
                elif resp.status_code in {500, 502, 503, 504}:
                    last_err = RuntimeError(f"Server error {resp.status_code}: {resp.text}")
                else:
                    resp.raise_for_status()
            except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                last_err = e

            if attempt < self.max_retries:
                time.sleep(self.retry_backoff * (2 ** (attempt - 1)))

        if not raw_text_resp and last_err:
            raise RuntimeError(
                f"Failed to extract discourse from LM Studio after {self.max_retries} attempts: {last_err}"
            )

        latency_sec = time.perf_counter() - t0
        extracted = self._parse_json_response(raw_text_resp, chunk_id)
        extracted.metadata["latency_sec"] = latency_sec
        extracted.metadata["backend"] = "lm_studio"
        extracted.metadata["model"] = self.model
        return extracted

    def _parse_json_response(
        self,
        text: str,
        chunk_id: Optional[str],
    ) -> DiscourseExtractionResult:
        """Parse raw model output, stripping markdown formatting if present."""
        clean = text.strip()
        # Strip ```json ... ``` code fences if model generated them
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean).strip()

        parsed = json.loads(clean)
        if chunk_id:
            parsed["chunk_id"] = chunk_id
        result = DiscourseExtractionResult.from_dict(parsed)
        
        # Log validation warnings if foreign keys dangle
        errs = result.validate_foreign_keys()
        if errs:
            logger.warning("LM Studio extraction produced foreign key warnings: %s", errs)
            result.metadata["foreign_key_warnings"] = errs

        return result


# ---------------------------------------------------------------------------
# 3.3: LocalGGUFTransducer (Direct In-Process via llama-cpp-python)
# ---------------------------------------------------------------------------

class LocalGGUFTransducer(BaseDiscourseTransducer):
    """Direct in-process GGUF model runner using llama-cpp-python."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        super().__init__(system_prompt=system_prompt)
        self.model_path = Path(model_path) if model_path else None
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm: Optional[Any] = None

    def load_model(self):
        """Lazy load the GGUF model into memory/VRAM."""
        if self._llm is not None:
            return

        if not self.model_path:
            raise ValueError("No model_path specified for LocalGGUFTransducer.")
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model file not found at: {self.model_path}")

        try:
            import llama_cpp  # type: ignore
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for LocalGGUFTransducer. "
                "Install it via 'pip install llama-cpp-python'."
            ) from e

        logger.info("Loading GGUF model from %s (gpu_layers=%d)", self.model_path, self.n_gpu_layers)
        self._llm = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    def is_available(self) -> bool:
        """Check if model file exists and is accessible."""
        return bool(self.model_path and self.model_path.exists())

    def transduce(
        self,
        chunk_text: str,
        chunk_id: Optional[str] = None,
        active_manifest_prompt: Optional[str] = None,
        **kwargs,
    ) -> DiscourseExtractionResult:
        """Run GGUF inference in-process and parse JSON schema."""
        self.load_model()
        user_prompt = self._build_user_prompt(chunk_text, active_manifest_prompt)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.perf_counter()
        resp = self._llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        latency_sec = time.perf_counter() - t0

        raw_content = resp["choices"][0]["message"]["content"]
        clean = raw_content.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean).strip()

        parsed = json.loads(clean)
        if chunk_id:
            parsed["chunk_id"] = chunk_id

        result = DiscourseExtractionResult.from_dict(parsed)
        result.metadata["latency_sec"] = latency_sec
        result.metadata["backend"] = "local_gguf"
        result.metadata["model_path"] = str(self.model_path)
        return result


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

def create_transducer(backend: str = "mock", **kwargs) -> BaseDiscourseTransducer:
    """Create a discourse transducer instance for the specified backend.

    Args:
        backend: Backend name in {'mock', 'lmstudio', 'gguf', 'auto'}.
        **kwargs: Backend-specific configuration parameters.

    Returns:
        Configured BaseDiscourseTransducer instance.
    """
    normalized = backend.lower().strip()
    if normalized == "mock":
        return MockTransducer(**kwargs)
    elif normalized in {"lmstudio", "lm_studio", "rest"}:
        return LMStudioTransducer(**kwargs)
    elif normalized in {"gguf", "llama_cpp", "local"}:
        return LocalGGUFTransducer(**kwargs)
    elif normalized == "auto":
        # Check if LM Studio is reachable; otherwise use MockTransducer
        candidate = LMStudioTransducer(**kwargs)
        if candidate.check_health():
            return candidate
        return MockTransducer()
    else:
        raise ValueError(f"Unknown transducer backend: {backend}. Expected 'mock', 'lmstudio', 'gguf', or 'auto'.")
