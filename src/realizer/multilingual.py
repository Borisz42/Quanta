"""Generalized Multilingual Realizer Architecture for QUANTA (Phase 8B).

Provides a unified, typologically principled framework to unroll discrete Mentalese ASGs
(Sigma^1024) into diverse natural languages across the world's major morphological families:
1. Isolating / Analytic (e.g., Mandarin Chinese, Vietnamese, Analytic English)
2. Agglutinative (e.g., Turkish, Hungarian, Finnish, Swahili, Japanese)
3. Fusional / Inflectional (e.g., German, Latin, Spanish, Russian, Sanskrit)
"""

from __future__ import annotations
import abc
from dataclasses import dataclass, field
import enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from core.asg import QuantaGraph, QuantaNode
from core.slots import get_slot_by_name
from core.types import QuantaVector, QuaternaryValue
from realizer.english_nlg import ConceptVectorDecoder


class MorphologicalType(enum.Enum):
    """Major morphosyntactic typological classifications."""
    ISOLATING = "isolating"        # Minimal bound morphemes, rigid word order, free particles
    AGGLUTINATIVE = "agglutinative"# Monomorphemic concatenative affixes, regular suffix chains
    FUSIONAL = "fusional"          # Portmanteau inflections (fused person/number/tense/case)
    POLYSYNTHETIC = "polysynthetic"# Holophrastic sentence-words with noun incorporation


class WordOrder(enum.Enum):
    """Canonical constituent word orders."""
    SVO = "SVO"
    SOV = "SOV"
    VSO = "VSO"
    VOS = "VOS"
    OVS = "OVS"
    OSV = "OSV"


@dataclass(frozen=True)
class LanguageConfig:
    """Configuration profile for a specific natural language adapter."""
    lang_code: str
    language_name: str
    morph_type: MorphologicalType
    canonical_word_order: WordOrder = WordOrder.SVO
    has_grammatical_gender: bool = False
    has_morphological_cases: bool = False
    supports_vowel_harmony: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class LanguageAdapter(abc.ABC):
    """Abstract Base Class for language-specific realization adapters."""

    def __init__(self, config: LanguageConfig):
        self.config = config
        self.vector_decoder = ConceptVectorDecoder.get_instance()

    @abc.abstractmethod
    def realize_noun_phrase(
        self,
        node: QuantaNode,
        graph: QuantaGraph,
        case: Optional[str] = None,
    ) -> str:
        """Realizes an atomic or compound entity node into a fully inflected noun phrase."""
        pass

    @abc.abstractmethod
    def inflect_verb(
        self,
        verb_node: QuantaNode,
        graph: QuantaGraph,
        is_negated: bool = False,
        is_uncertain: bool = False,
    ) -> str:
        """Inflects or modifies a predicate action verb according to tense, aspect, and polarity."""
        pass

    @abc.abstractmethod
    def format_clause(
        self,
        subject: Optional[str],
        verb: str,
        patient: Optional[str],
        modifiers: List[str],
    ) -> str:
        """Arranges constituent noun phrases and verb into a fluent, typologically ordered clause."""
        pass

    def extract_lemma(self, node: QuantaNode) -> str:
        """Extracts lemma from anchor, literal, or 2-tier ConceptVectorDecoder."""
        if node.anchor:
            anchor = node.anchor
            if anchor.startswith("cn:"):
                # e.g., cn:en:golden retriever (n) -> golden retriever
                cleaned = anchor[3:]
                if ":" in cleaned:
                    cleaned = cleaned.split(":", 1)[1]
                if " (" in cleaned:
                    cleaned = cleaned.split(" (")[0]
                return cleaned.replace("_", " ")
            elif anchor.startswith("wn:"):
                # e.g., wn:dog.n.01 -> dog
                cleaned = anchor[3:].split(".")[0]
                return cleaned.replace("_", " ")
            elif anchor.startswith("func:") or anchor.startswith("var:"):
                return anchor.split(":", 1)[1]

        if node.literal and not node.literal.startswith("AST:"):
            return node.literal

        # 2-Tier vector decoding fallback (Bands 3 & 4: 384..639)
        vec_256 = node.vector._data[384:640]
        if vec_256.any():
            decoded, _ = self.vector_decoder.decode_vector(vec_256)
            if decoded:
                return decoded

        return "thing"

    def realize_graph(self, graph: QuantaGraph) -> str:
        """Universal realization entry point unrolling an ASG into the target language."""
        if not graph.root:
            return ""

        root = graph.root
        is_negated = (root.get_slot("LJB_NA_NEGATION") == QuaternaryValue.FALSE)
        is_uncertain = (root.get_slot("GRAPH_QUERY_TARGET") == QuaternaryValue.UNKNOWN or
                        root.get_slot("NSM_MAYBE") == QuaternaryValue.UNKNOWN)

        # 1. Subject (VAL_X1_AGENT)
        subject_str = None
        agent_cids = root.edges.get("VAL_X1_AGENT", [])
        if agent_cids:
            agent_node = graph.get_node(agent_cids[0])
            if agent_node:
                subject_str = self.realize_noun_phrase(agent_node, graph, case="nominative")

        # 2. Patient / Object (VAL_X2_PATIENT)
        patient_str = None
        patient_cids = root.edges.get("VAL_X2_PATIENT", [])
        if patient_cids:
            patient_node = graph.get_node(patient_cids[0])
            if patient_node:
                patient_str = self.realize_noun_phrase(patient_node, graph, case="accusative")

        # 3. Location / Modifier Phrases
        modifiers: List[str] = []
        loc_cids = root.edges.get("VAL_LOCATION_SLOT", [])
        for l_cid in loc_cids:
            loc_node = graph.get_node(l_cid)
            if loc_node:
                mod_str = self.realize_noun_phrase(loc_node, graph, case="locative")
                modifiers.append(mod_str)

        # 4. Inflect Verb
        verb_str = self.inflect_verb(root, graph, is_negated=is_negated, is_uncertain=is_uncertain)

        # 5. Assemble clause
        return self.format_clause(subject_str, verb_str, patient_str, modifiers)


class IsolatingLanguageAdapter(LanguageAdapter):
    """Reference adapter for Isolating / Analytic morphology (e.g. SVO, free particles)."""

    def realize_noun_phrase(
        self,
        node: QuantaNode,
        graph: QuantaGraph,
        case: Optional[str] = None,
    ) -> str:
        lemma = self.extract_lemma(node)
        determiner = ""
        if node.get_slot("NSM_ONE") == 1:
            determiner = "one "
        elif node.get_slot("NSM_THIS") == 1:
            determiner = "the "
        elif node.get_slot("NSM_ALL") == 1:
            determiner = "every "

        if case == "locative":
            return f"at {determiner}{lemma}".strip()
        return f"{determiner}{lemma}".strip()

    def inflect_verb(
        self,
        verb_node: QuantaNode,
        graph: QuantaGraph,
        is_negated: bool = False,
        is_uncertain: bool = False,
    ) -> str:
        lemma = self.extract_lemma(verb_node)
        particles: List[str] = []

        if is_uncertain:
            particles.append("maybe")

        if is_negated:
            particles.append("not")

        if verb_node.get_slot("LJB_PU_PAST_TENSE") == 1:
            particles.append("did")

        if particles:
            return f"{' '.join(particles)} {lemma}".strip()
        return lemma

    def format_clause(
        self,
        subject: Optional[str],
        verb: str,
        patient: Optional[str],
        modifiers: List[str],
    ) -> str:
        parts: List[str] = []
        if subject:
            parts.append(subject)
        parts.append(verb)
        if patient:
            parts.append(patient)
        if modifiers:
            parts.extend(modifiers)

        raw = " ".join(parts).strip()
        if raw and not raw.endswith((".", "?", "!")):
            raw += "."
        return raw.capitalize() if raw else ""


class AgglutinativeLanguageAdapter(LanguageAdapter):
    """Reference adapter for Agglutinative morphology (concatenative suffix chains, flexible word order)."""

    CASE_SUFFIXES: Dict[str, str] = {
        "nominative": "",
        "accusative": "-t",
        "locative": "-ban",
        "dative": "-nak",
        "instrumental": "-val",
        "allative": "-hoz",
        "ablative": "-tol",
    }

    def realize_noun_phrase(
        self,
        node: QuantaNode,
        graph: QuantaGraph,
        case: Optional[str] = None,
    ) -> str:
        lemma = self.extract_lemma(node)
        suffix = self.CASE_SUFFIXES.get(case or "nominative", "")

        prefix = ""
        if node.get_slot("NSM_ONE") == 1:
            prefix = "egy "
        elif node.get_slot("NSM_THIS") == 1:
            prefix = "a "
        elif node.get_slot("NSM_ALL") == 1:
            prefix = "minden "

        return f"{prefix}{lemma}{suffix}".strip()

    def inflect_verb(
        self,
        verb_node: QuantaNode,
        graph: QuantaGraph,
        is_negated: bool = False,
        is_uncertain: bool = False,
    ) -> str:
        lemma = self.extract_lemma(verb_node)
        tense_suffix = ""

        if verb_node.get_slot("LJB_PU_PAST_TENSE") == 1:
            tense_suffix = "-ott"
        elif verb_node.get_slot("LJB_BA_FUTURE_TENSE") == 1:
            tense_suffix = "-ni fog"

        verb_form = f"{lemma}{tense_suffix}".strip()

        if is_uncertain:
            verb_form = f"talán {verb_form}"
        if is_negated:
            verb_form = f"nem {verb_form}"

        return verb_form.strip()

    def format_clause(
        self,
        subject: Optional[str],
        verb: str,
        patient: Optional[str],
        modifiers: List[str],
    ) -> str:
        # Default SOV / S-Mod-O-V order typical of many agglutinative languages
        parts: List[str] = []
        if subject:
            parts.append(subject)
        if patient:
            parts.append(patient)
        if modifiers:
            parts.extend(modifiers)
        parts.append(verb)

        raw = " ".join(parts).strip()
        if raw and not raw.endswith((".", "?", "!")):
            raw += "."
        return raw.capitalize() if raw else ""


class FusionalLanguageAdapter(LanguageAdapter):
    """Reference adapter for Fusional / Inflectional morphology (portmanteau inflections)."""

    def realize_noun_phrase(
        self,
        node: QuantaNode,
        graph: QuantaGraph,
        case: Optional[str] = None,
    ) -> str:
        lemma = self.extract_lemma(node)
        case_norm = (case or "nominative").lower()

        det = ""
        if node.get_slot("NSM_THIS") == 1:
            if case_norm == "accusative":
                det = "den "
            elif case_norm == "locative":
                det = "in dem "
            else:
                det = "der "
        elif node.get_slot("NSM_ONE") == 1:
            if case_norm == "accusative":
                det = "einen "
            elif case_norm == "locative":
                det = "in einem "
            else:
                det = "ein "
        elif case_norm == "locative":
            det = "in "

        return f"{det}{lemma}".strip()

    def inflect_verb(
        self,
        verb_node: QuantaNode,
        graph: QuantaGraph,
        is_negated: bool = False,
        is_uncertain: bool = False,
    ) -> str:
        lemma = self.extract_lemma(verb_node)
        is_past = (verb_node.get_slot("LJB_PU_PAST_TENSE") == 1)

        # Fusional 3SG portmanteau: past vs present
        if is_past:
            inflected = f"{lemma}te" if not lemma.endswith("e") else f"{lemma}t"
        else:
            inflected = f"{lemma}t"

        if is_negated:
            inflected = f"nicht {inflected}"
        if is_uncertain:
            inflected = f"vielleicht {inflected}"

        return inflected

    def format_clause(
        self,
        subject: Optional[str],
        verb: str,
        patient: Optional[str],
        modifiers: List[str],
    ) -> str:
        # Standard V2 / SVO clause structure
        parts: List[str] = []
        if subject:
            parts.append(subject)
        parts.append(verb)
        if patient:
            parts.append(patient)
        if modifiers:
            parts.extend(modifiers)

        raw = " ".join(parts).strip()
        if raw and not raw.endswith((".", "?", "!")):
            raw += "."
        return raw.capitalize() if raw else ""


class MultilingualRealizerRegistry:
    """Registry and factory for discovering and dispatching typologically diverse language adapters."""

    _registry: Dict[str, LanguageAdapter] = {}
    _initialized: bool = False

    @classmethod
    def initialize_defaults(cls):
        if cls._initialized:
            return

        # 1. Analytic / Isolating Archetype
        cls.register_adapter(
            "isolating_ref",
            IsolatingLanguageAdapter(
                LanguageConfig(
                    lang_code="isolating_ref",
                    language_name="Isolating Reference Language",
                    morph_type=MorphologicalType.ISOLATING,
                    canonical_word_order=WordOrder.SVO,
                )
            ),
        )

        # 2. Agglutinative Archetype
        cls.register_adapter(
            "agglutinative_ref",
            AgglutinativeLanguageAdapter(
                LanguageConfig(
                    lang_code="agglutinative_ref",
                    language_name="Agglutinative Reference Language",
                    morph_type=MorphologicalType.AGGLUTINATIVE,
                    canonical_word_order=WordOrder.SOV,
                    has_morphological_cases=True,
                )
            ),
        )

        # 3. Fusional Archetype
        cls.register_adapter(
            "fusional_ref",
            FusionalLanguageAdapter(
                LanguageConfig(
                    lang_code="fusional_ref",
                    language_name="Fusional Reference Language",
                    morph_type=MorphologicalType.FUSIONAL,
                    canonical_word_order=WordOrder.SVO,
                    has_morphological_cases=True,
                    has_grammatical_gender=True,
                )
            ),
        )

        cls._initialized = True

    @classmethod
    def register_adapter(cls, lang_code: str, adapter: LanguageAdapter):
        """Registers a custom language adapter instance."""
        cls._registry[lang_code.lower()] = adapter

    @classmethod
    def get_adapter(cls, lang_code: str) -> Optional[LanguageAdapter]:
        """Retrieves a registered language adapter by language code."""
        cls.initialize_defaults()
        return cls._registry.get(lang_code.lower())

    @classmethod
    def list_supported_languages(cls) -> List[str]:
        """Returns all registered language codes."""
        cls.initialize_defaults()
        return sorted(list(cls._registry.keys()))

    @classmethod
    def realize_graph(cls, graph: QuantaGraph, lang_code: str = "isolating_ref") -> str:
        """Convenience method to realize an ASG into the specified language."""
        adapter = cls.get_adapter(lang_code)
        if not adapter:
            raise ValueError(f"Unsupported language code '{lang_code}'. Available: {cls.list_supported_languages()}")
        return adapter.realize_graph(graph)


__all__ = [
    "MorphologicalType",
    "WordOrder",
    "LanguageConfig",
    "LanguageAdapter",
    "IsolatingLanguageAdapter",
    "AgglutinativeLanguageAdapter",
    "FusionalLanguageAdapter",
    "MultilingualRealizerRegistry",
]
