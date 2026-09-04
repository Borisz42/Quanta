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


CANONICAL_STRESS_1_TEXT = (
    "Had Alice not falsely pretended to know that Bob believed her investment was secure, "
    "the auditor wouldn't have sarcastically remarked that her due diligence was a stroke of genius."
)

CANONICAL_STRESS_1_FIXTURE = DiscourseExtractionResult(
    chunk_id="chunk_stress_1",
    entities=[
        ExtractedEntity(id="E1", canonical_name="Alice", category="PERSON", surface_aliases=["her"]),
        ExtractedEntity(id="E2", canonical_name="Bob", category="PERSON", surface_aliases=["Bob"]),
        ExtractedEntity(id="E3", canonical_name="investment", category="OBJECT", surface_aliases=["her investment"]),
        ExtractedEntity(id="E4", canonical_name="auditor", category="PERSON", surface_aliases=["the auditor"]),
        ExtractedEntity(id="E5", canonical_name="due diligence", category="ARTIFACT", surface_aliases=["her due diligence", "stroke of genius"]),
    ],
    events=[
        ExtractedEvent(
            id="Ev1",
            predicate="pretend",
            agent_id="E1",
            patient_id=None,
            temporal_anchor=None,
            tense="PAST",
            polarity=False,
            raw_text="Had Alice not falsely pretended to know",
        ),
        ExtractedEvent(
            id="Ev2",
            predicate="know",
            agent_id="E1",
            patient_id=None,
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="to know that Bob believed her investment was secure",
        ),
        ExtractedEvent(
            id="Ev3",
            predicate="believe",
            agent_id="E2",
            patient_id=None,
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="Bob believed her investment was secure",
        ),
        ExtractedEvent(
            id="Ev4",
            predicate="secure",
            agent_id=None,
            patient_id="E3",
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="her investment was secure",
        ),
        ExtractedEvent(
            id="Ev5",
            predicate="remark",
            agent_id="E4",
            patient_id="E5",
            temporal_anchor=None,
            tense="PAST",
            polarity=False,
            raw_text="the auditor wouldn't have sarcastically remarked that her due diligence was a stroke of genius",
        ),
    ],
    relations=[
        ExtractedRelation(relation_type="CAUSAL_MECHANISM_LINK", source_id="Ev1", target_id="Ev5", mechanism="counterfactual condition"),
        ExtractedRelation(relation_type="TEMP_ALLEN_BEFORE", source_id="Ev1", target_id="Ev5", mechanism="prior pretence"),
    ],
    propositions=[
        ExtractedProposition(id="P1", claim_text="Alice falsely pretended", epistemic_status="HYPOTHESIS", subject_id="E1", event_id="Ev1"),
        ExtractedProposition(id="P2", claim_text="Bob believed investment secure", epistemic_status="FACT", subject_id="E2", event_id="Ev3"),
        ExtractedProposition(id="P3", claim_text="auditor sarcastically remarked", epistemic_status="DOUBTED", subject_id="E4", event_id="Ev5"),
    ],
    metadata={"provenance": "canonical_fixture", "source": "Stress Test 1"},
)

CANONICAL_STRESS_2_TEXT = (
    "While the drone was accelerating into the restricted airspace before dusk, "
    "the operator plausibly suspected, but could not deduce with certainty, "
    "that the left wingtip was tangentially touching the perimeter wire."
)

CANONICAL_STRESS_2_FIXTURE = DiscourseExtractionResult(
    chunk_id="chunk_stress_2",
    entities=[
        ExtractedEntity(id="E1", canonical_name="drone", category="ARTIFACT", surface_aliases=["the drone"]),
        ExtractedEntity(id="E2", canonical_name="restricted airspace", category="LOCATION", surface_aliases=["the restricted airspace"]),
        ExtractedEntity(id="E3", canonical_name="operator", category="PERSON", surface_aliases=["the operator"]),
        ExtractedEntity(id="E4", canonical_name="left wingtip", category="ARTIFACT", surface_aliases=["the left wingtip"]),
        ExtractedEntity(id="E5", canonical_name="perimeter wire", category="ARTIFACT", surface_aliases=["the perimeter wire"]),
    ],
    events=[
        ExtractedEvent(
            id="Ev1",
            predicate="accelerate",
            agent_id="E1",
            location_id="E2",
            temporal_anchor="before dusk",
            tense="PAST",
            aspect="PROGRESSIVE",
            polarity=True,
            raw_text="While the drone was accelerating into the restricted airspace before dusk",
        ),
        ExtractedEvent(
            id="Ev2",
            predicate="suspect",
            agent_id="E3",
            patient_id=None,
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="the operator plausibly suspected",
        ),
        ExtractedEvent(
            id="Ev3",
            predicate="deduce",
            agent_id="E3",
            patient_id=None,
            temporal_anchor=None,
            tense="PAST",
            polarity=False,
            raw_text="could not deduce with certainty",
        ),
        ExtractedEvent(
            id="Ev4",
            predicate="touch",
            agent_id="E4",
            patient_id="E5",
            temporal_anchor=None,
            tense="PAST",
            aspect="PROGRESSIVE",
            polarity=True,
            raw_text="the left wingtip was tangentially touching the perimeter wire",
        ),
    ],
    relations=[
        ExtractedRelation(relation_type="TEMP_ALLEN_DURING", source_id="Ev1", target_id="Ev2", mechanism="during acceleration"),
        ExtractedRelation(relation_type="TEMP_ALLEN_DURING", source_id="Ev4", target_id="Ev2", mechanism="concurrent touch"),
    ],
    propositions=[
        ExtractedProposition(id="P1", claim_text="drone accelerating into restricted airspace", epistemic_status="OBSERVATION", subject_id="E1", event_id="Ev1"),
        ExtractedProposition(id="P2", claim_text="operator suspected touch", epistemic_status="HYPOTHESIS", subject_id="E3", event_id="Ev2"),
        ExtractedProposition(id="P3", claim_text="deduction not certain", epistemic_status="DOUBTED", subject_id="E3", event_id="Ev3"),
    ],
    metadata={"provenance": "canonical_fixture", "source": "Stress Test 2"},
)

CANONICAL_STRESS_3_TEXT = (
    "Every investigator who doubted that any suspect had necessarily committed every crime "
    "secretly wanted someone to prove the absolute impossibility of an accomplice's alibi."
)

CANONICAL_STRESS_3_FIXTURE = DiscourseExtractionResult(
    chunk_id="chunk_stress_3",
    entities=[
        ExtractedEntity(id="E1", canonical_name="investigator", category="PERSON", surface_aliases=["Every investigator"]),
        ExtractedEntity(id="E2", canonical_name="suspect", category="PERSON", surface_aliases=["any suspect"]),
        ExtractedEntity(id="E3", canonical_name="crime", category="OBJECT", surface_aliases=["every crime"]),
        ExtractedEntity(id="E4", canonical_name="accomplice", category="PERSON", surface_aliases=["someone", "an accomplice"]),
        ExtractedEntity(id="E5", canonical_name="alibi", category="ARTIFACT", surface_aliases=["an accomplice's alibi"]),
    ],
    events=[
        ExtractedEvent(
            id="Ev1",
            predicate="doubt",
            agent_id="E1",
            patient_id=None,
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="Every investigator who doubted",
        ),
        ExtractedEvent(
            id="Ev2",
            predicate="commit",
            agent_id="E2",
            patient_id="E3",
            temporal_anchor=None,
            tense="PAST",
            aspect="PERFECT",
            polarity=True,
            raw_text="any suspect had necessarily committed every crime",
        ),
        ExtractedEvent(
            id="Ev3",
            predicate="want",
            agent_id="E1",
            patient_id=None,
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="secretly wanted someone to prove",
        ),
        ExtractedEvent(
            id="Ev4",
            predicate="prove",
            agent_id="E4",
            patient_id="E5",
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="to prove the absolute impossibility of an accomplice's alibi",
        ),
    ],
    relations=[
        ExtractedRelation(relation_type="TEMP_ALLEN_BEFORE", source_id="Ev1", target_id="Ev3", mechanism="prior doubt"),
    ],
    propositions=[
        ExtractedProposition(id="P1", claim_text="investigator doubted", epistemic_status="DOUBTED", subject_id="E1", event_id="Ev1"),
        ExtractedProposition(id="P2", claim_text="wanted proof of impossibility", epistemic_status="FACT", subject_id="E1", event_id="Ev3"),
    ],
    metadata={"provenance": "canonical_fixture", "source": "Stress Test 3"},
)

CANONICAL_STRESS_4_TEXT = (
    "By declaring this very decree to be legally void, the council obligated the commissioner "
    "to prevent its future enforcement unless the clause could recursively validate its own origin."
)

CANONICAL_STRESS_4_FIXTURE = DiscourseExtractionResult(
    chunk_id="chunk_stress_4",
    entities=[
        ExtractedEntity(id="E1", canonical_name="decree", category="ARTIFACT", surface_aliases=["this very decree"]),
        ExtractedEntity(id="E2", canonical_name="council", category="ORGANIZATION", surface_aliases=["the council"]),
        ExtractedEntity(id="E3", canonical_name="commissioner", category="PERSON", surface_aliases=["the commissioner"]),
        ExtractedEntity(id="E4", canonical_name="clause", category="ARTIFACT", surface_aliases=["the clause"]),
    ],
    events=[
        ExtractedEvent(
            id="Ev1",
            predicate="declare",
            agent_id="E2",
            patient_id="E1",
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="declaring this very decree to be legally void",
        ),
        ExtractedEvent(
            id="Ev2",
            predicate="obligate",
            agent_id="E2",
            patient_id="E3",
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="the council obligated the commissioner",
        ),
        ExtractedEvent(
            id="Ev3",
            predicate="prevent",
            agent_id="E3",
            patient_id="E1",
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="to prevent its future enforcement",
        ),
        ExtractedEvent(
            id="Ev4",
            predicate="validate",
            agent_id="E4",
            patient_id="E4",
            temporal_anchor=None,
            tense="PAST",
            polarity=True,
            raw_text="the clause could recursively validate its own origin",
        ),
    ],
    relations=[
        ExtractedRelation(relation_type="CAUSAL_MECHANISM_LINK", source_id="Ev1", target_id="Ev2", mechanism="declaration caused obligation"),
        ExtractedRelation(relation_type="CAUSAL_PREVENTIVE_BLOCK", source_id="Ev3", target_id="Ev1", mechanism="prevention blocks enforcement"),
    ],
    propositions=[
        ExtractedProposition(id="P1", claim_text="decree legally void", epistemic_status="FACT", subject_id="E2", event_id="Ev1"),
        ExtractedProposition(id="P2", claim_text="commissioner obligated to prevent", epistemic_status="PROHIBITED", subject_id="E3", event_id="Ev2"),
        ExtractedProposition(id="P3", claim_text="clause validates origin", epistemic_status="HYPOTHESIS", subject_id="E4", event_id="Ev4"),
    ],
    metadata={"provenance": "canonical_fixture", "source": "Stress Test 4"},
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
        # Pre-seed stress test fixtures 1-4
        self.register_fixture("stress_1", CANONICAL_STRESS_1_FIXTURE)
        self.register_fixture(CANONICAL_STRESS_1_TEXT, CANONICAL_STRESS_1_FIXTURE)
        self.register_fixture("stress_2", CANONICAL_STRESS_2_FIXTURE)
        self.register_fixture(CANONICAL_STRESS_2_TEXT, CANONICAL_STRESS_2_FIXTURE)
        self.register_fixture("stress_3", CANONICAL_STRESS_3_FIXTURE)
        self.register_fixture(CANONICAL_STRESS_3_TEXT, CANONICAL_STRESS_3_FIXTURE)
        self.register_fixture("stress_4", CANONICAL_STRESS_4_FIXTURE)
        self.register_fixture(CANONICAL_STRESS_4_TEXT, CANONICAL_STRESS_4_FIXTURE)

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
        norm_text = " ".join(chunk_text.split()).strip()

        # 1. Exact match in fixtures
        if chunk_text in self.fixtures:
            return self._clone_result(self.fixtures[chunk_text], chunk_id)
        if norm_text in self.fixtures:
            return self._clone_result(self.fixtures[norm_text], chunk_id)

        # 2. Key/substring match
        lower = chunk_text.lower()
        if "eleanor" in lower or "containment cell" in lower or "synthetic compound" in lower:
            return self._clone_result(CANONICAL_ELEANOR_VANCE_FIXTURE, chunk_id)
        if "alice" in lower and "auditor" in lower:
            return self._clone_result(CANONICAL_STRESS_1_FIXTURE, chunk_id)
        if "drone" in lower and "airspace" in lower:
            return self._clone_result(CANONICAL_STRESS_2_FIXTURE, chunk_id)
        if "investigator" in lower and "alibi" in lower:
            return self._clone_result(CANONICAL_STRESS_3_FIXTURE, chunk_id)
        if "decree" in lower and "commissioner" in lower:
            return self._clone_result(CANONICAL_STRESS_4_FIXTURE, chunk_id)

        for key, fix in self.fixtures.items():
            if key.lower() in lower:
                return self._clone_result(fix, chunk_id)

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
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        super().__init__(system_prompt=system_prompt)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.api_key = api_key or "lm-studio"
        self.session = session or requests.Session()
        self.model = model or self._detect_model()

    def _detect_model(self) -> str:
        """Query /models to find the currently active or preferred model."""
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = self.session.get(url, headers=headers, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", []) if "id" in m]
                for m in models:
                    if "qwen3.5-4b" in m.lower():
                        return m
                for m in models:
                    if "qwen" in m.lower():
                        return m
                if models:
                    return models[0]
        except Exception:
            pass
        return "qwen3.5-4b-mtp"

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
        target_model = kwargs.get("model") or self.model or self._detect_model()
        # Use assistant prefill to prevent runaway reasoning loops and guarantee prompt JSON start
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "{\n"},
        ]
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 2048),
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
                    content = data["choices"][0]["message"]["content"]
                    raw_text_resp = content
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
        extracted.metadata["model"] = target_model
        return extracted

    def _parse_json_response(
        self,
        text: str,
        chunk_id: Optional[str],
    ) -> DiscourseExtractionResult:
        """Parse raw model output, stripping reasoning tags and markdown formatting if present."""
        clean = text.strip()
        # Strip <think>...</think> reasoning blocks if model generated them
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
        # Strip ```json ... ``` code fences if model generated them
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if fence_match:
            clean = fence_match.group(1).strip()
        elif clean.startswith("```"):
            clean = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean).strip()
        # If model output continued from assistant prefill '{\n', prepend the missing '{'
        if not clean.startswith("{") and any(k in clean for k in ('"entities"', '"events"', '"chunk_id"', '"propositions"')):
            clean = "{\n" + clean

        # Auto-repair omitted 'canonical_name' key in entity entries if model emits compact objects
        clean = re.sub(
            r'("id":\s*"[^"]+",)\s*"([^":\n]+)",\s*("(?:category|surface_aliases|properties)":)',
            r'\1 "canonical_name": "\2", \3',
            clean,
        )

        # Isolate the outermost JSON object using balanced bracket counting
        start_idx = clean.find("{")
        if start_idx != -1:
            depth = 0
            in_str = False
            esc = False
            matched = False
            for i in range(start_idx, len(clean)):
                c = clean[i]
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            clean = clean[start_idx : i + 1]
                            matched = True
                            break
            if not matched:
                clean = clean[start_idx:]

        # Strip trailing commas that violate standard JSON
        clean = re.sub(r",\s*([\]}])", r"\1", clean)
        clean = re.sub(r",\s*([\]}])", r"\1", clean)

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Fallback attempt with more aggressive comma cleaning
            clean_fixed = re.sub(r",\s*([\]}])", r"\1", clean)
            parsed = json.loads(clean_fixed)
        if chunk_id:
            parsed["chunk_id"] = chunk_id

        # Auto-sanitize categories and foreign keys
        entities = parsed.get("entities", [])
        events = parsed.get("events", [])
        known_entity_ids = {e.get("id") for e in entities if isinstance(e, dict) and "id" in e}

        valid_categories = {
            "PERSON", "OBJECT", "SUBSTANCE", "LOCATION",
            "ORGANIZATION", "ANIMAL", "ARTIFACT", "NATURAL_OBJECT",
        }
        for e in entities:
            if isinstance(e, dict):
                cat = str(e.get("category", "OBJECT")).upper()
                if cat not in valid_categories:
                    e["category"] = "OBJECT"

        for ev in events:
            if isinstance(ev, dict):
                for k in ("agent_id", "patient_id", "theme_id", "location_id", "instrument_id"):
                    val = ev.get(k)
                    if val and val not in known_entity_ids:
                        ev[k] = None

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
