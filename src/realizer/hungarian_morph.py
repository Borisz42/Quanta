"""Hungarian Morphological Realizer for QUANTA.

Demonstrates linguistic generalizability to agglutinative morphology with a full
vowel-harmony engine and case suffix inflection system.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, QuaternaryValue


class HungarianRealizer:
    """Agglutinative Hungarian NLG realizer with vowel harmony and case marking."""

    BACK_VOWELS = set("aáoóuúAÁOÓUÚ")
    FRONT_UNROUNDED = set("eéiíEÉIÍ")
    FRONT_ROUNDED = set("öőüűÖŐÜŰ")
    FRONT_VOWELS = FRONT_UNROUNDED | FRONT_ROUNDED
    ALL_VOWELS = BACK_VOWELS | FRONT_VOWELS

    # Common lemma translations for anchor/literal grounding
    DICTIONARY = {
        "dog": "kutya",
        "golden retriever": "golden retriever",
        "golden_retriever": "golden retriever",
        "cat": "macska",
        "mailman": "postás",
        "postman": "postás",
        "person": "ember",
        "human": "ember",
        "rock": "kő",
        "stone": "kő",
        "garden": "kert",
        "house": "ház",
        "car": "autó",
        "tree": "fa",
        "water": "víz",
        "book": "könyv",
        "city": "város",
        "stick": "bot",
        "bite": "harap",
        "chase": "kerget",
        "run": "fut",
        "walk": "sétál",
        "see": "lát",
        "hear": "hall",
        "think": "gondol",
        "know": "tud",
        "say": "mond",
        "tell": "mesél",
        "give": "ad",
        "take": "vesz",
        "go": "megy",
        "come": "jön",
        "live": "él",
        "die": "meghal",
        "want": "akar",
        "feel": "érez",
        "move": "mozog",
        "enter": "belép",
        "touch": "érint",
        "big": "nagy",
        "small": "kis",
        "good": "jó",
        "bad": "rossz",
        "brown": "barna",
        "quick": "gyors",
    }

    def realize_graph(self, graph: QuantaGraph) -> str:
        """Realizes a QuantaGraph ASG into an agglutinative Hungarian sentence."""
        root = graph.root
        if root is None:
            return ""

        # 1. Resolve Root Verb
        verb_base = self._extract_hungarian_verb(root)
        is_past = root.get_slot("LJB_PU_PAST_TENSE") == 1 or \
                  root.get_slot("LJB_ZI_SHORT_PAST") == 1 or \
                  root.get_slot("LJB_ZA_MEDIUM_PAST") == 1 or \
                  root.get_slot("LJB_ZU_LONG_PAST") == 1
        is_future = root.get_slot("LJB_BA_FUTURE_TENSE") == 1
        is_negated = root.get_slot("LJB_NA_NEGATION") == 2

        # 2. Conjugate Verb
        conjugated_verb = self._conjugate_verb(verb_base, is_past=is_past, is_future=is_future)

        # 3. Resolve Subject (Nominative: no suffix)
        subject_str = self._resolve_entity_with_case(graph, root, "VAL_X1_AGENT", case="nom")

        # 4. Resolve Patient (Accusative: -t / -ot / -et / -öt)
        patient_str = self._resolve_entity_with_case(graph, root, "VAL_X2_PATIENT", case="acc")
        if not patient_str and "VAL_EXPERIENCER" in root.edges:
            patient_str = self._resolve_entity_with_case(graph, root, "VAL_EXPERIENCER", case="acc")

        # 5. Resolve Prepositional / Valency Cases
        # Destination -> Illative (-ba/-be) or Sublative (-ra/-re)
        dest_str = self._resolve_entity_with_case(graph, root, "VAL_X3_DESTINATION", case="ill")
        # Source -> Elative (-ból/-ből)
        source_str = self._resolve_entity_with_case(graph, root, "VAL_X4_SOURCE", case="ela")
        # Instrument -> Instrumental (-val/-vel)
        inst_str = self._resolve_entity_with_case(graph, root, "VAL_X5_INSTRUMENT", case="ins")
        # Location -> Inessive (-ban/-ben)
        loc_str = self._resolve_entity_with_case(graph, root, "VAL_LOCATION_SLOT", case="ine")

        # Predicate adjective for copula sentences
        pred_adj = ""
        if root.get_slot("NSM_BIG") == 1:
            pred_adj = "nagy"
        elif root.get_slot("NSM_SMALL") == 1:
            pred_adj = "kis"
        elif root.get_slot("NSM_GOOD") == 1:
            pred_adj = "jó"
        elif root.get_slot("NSM_BAD") == 1:
            pred_adj = "rossz"

        # Assemble Hungarian sentence: Subject + Modifiers + Object + (Nem) + Verb / Predicate Adj
        tokens: List[str] = []
        if subject_str:
            tokens.append(subject_str)
        if loc_str:
            tokens.append(loc_str)
        if source_str:
            tokens.append(source_str)
        if dest_str:
            tokens.append(dest_str)
        if inst_str:
            tokens.append(inst_str)
        if patient_str:
            tokens.append(patient_str)
        if is_negated:
            tokens.append("nem")
        if verb_base == "van":
            if is_past:
                tokens.append("volt")
            elif not pred_adj:
                tokens.append("van")
        else:
            tokens.append(conjugated_verb)
        if pred_adj:
            tokens.append(pred_adj)

        raw = " ".join(tokens).strip()
        if not raw:
            return ""
        return raw[0].upper() + raw[1:] + "."

    def get_vowel_harmony(self, word: str) -> str:
        """Determines vowel harmony of a word: 'back', 'front_unrounded', or 'front_rounded'."""
        vowels = [ch for ch in word if ch in self.ALL_VOWELS]
        if not vowels:
            return "back"  # Default fallback

        # Last vowel usually governs suffix harmony
        last_vowel = vowels[-1]
        if last_vowel in self.BACK_VOWELS:
            return "back"
        if last_vowel in self.FRONT_ROUNDED:
            return "front_rounded"
        
        # If last is front unrounded (e, é, i, í), check preceding vowels if any
        if len(vowels) >= 2 and any(v in self.BACK_VOWELS for v in vowels[:-1]):
            # Words like 'kávé', 'tányér', 'férfi'
            return "back"

        return "front_unrounded"

    def apply_case_suffix(self, noun: str, case: str) -> str:
        """Applies Hungarian agglutinative case suffix with morphophonological vowel harmony."""
        noun = noun.strip()
        if not noun:
            return ""

        harmony = self.get_vowel_harmony(noun)
        ends_with_vowel = noun[-1] in self.ALL_VOWELS

        # Stem vowel lengthening (a -> á, e -> é before suffixes)
        stem = noun
        if ends_with_vowel:
            if noun.endswith("a"):
                stem = noun[:-1] + "á"
            elif noun.endswith("e"):
                stem = noun[:-1] + "é"

        # 1. Nominative (nom)
        if case == "nom":
            return noun

        # 2. Accusative (acc: -t / -ot / -et / -öt / -at)
        elif case == "acc":
            if ends_with_vowel:
                return stem + "t"
            if harmony == "back":
                return stem + "ot"
            elif harmony == "front_rounded":
                return stem + "öt"
            else:
                return stem + "et"

        # 3. Inessive (ine: -ban / -ben)
        elif case == "ine":
            return stem + ("ban" if harmony == "back" else "ben")

        # 4. Illative (ill: -ba / -be)
        elif case == "ill":
            return stem + ("ba" if harmony == "back" else "be")

        # 5. Elative (ela: -ból / -ből)
        elif case == "ela":
            return stem + ("ból" if harmony == "back" else "ből")

        # 6. Instrumental (ins: -val / -vel with assimilation)
        elif case == "ins":
            if ends_with_vowel:
                return stem + ("val" if harmony == "back" else "vel")
            last_consonant = noun[-1]
            suffix_vowel = "a" if harmony == "back" else "e"
            return stem + last_consonant + suffix_vowel + "l"

        # 7. Sublative (sub: -ra / -re)
        elif case == "sub":
            return stem + ("ra" if harmony == "back" else "re")

        # 8. Allative (all: -hoz / -hez / -höz)
        elif case == "all":
            if harmony == "back":
                return stem + "hoz"
            elif harmony == "front_rounded":
                return stem + "höz"
            else:
                return stem + "hez"

        return noun

    def _conjugate_verb(self, verb_base: str, is_past: bool, is_future: bool) -> str:
        """Conjugates a Hungarian verb stem in 3rd person singular."""
        harmony = self.get_vowel_harmony(verb_base)

        if is_future:
            return f"fog {verb_base}ni"

        if is_past:
            if harmony == "back":
                return verb_base + "ott"
            elif harmony == "front_rounded":
                return verb_base + "ött"
            else:
                return verb_base + "ett"

        # Present tense 3sg
        return verb_base

    def _extract_hungarian_verb(self, node: QuantaNode) -> str:
        """Translates verb anchor or prime to Hungarian verb stem."""
        if node.anchor and node.anchor.startswith("wn:"):
            parts = node.anchor[3:].split(".")
            lemma = parts[0].replace("_", " ").lower()
            if lemma in self.DICTIONARY:
                return self.DICTIONARY[lemma]

        if node.literal and isinstance(node.literal, str):
            lit = node.literal.strip().lower()
            if lit in self.DICTIONARY:
                return self.DICTIONARY[lit]

        if node.get_slot("NSM_MOVE") != 0:
            return "fut"
        if node.get_slot("NSM_SAY") != 0:
            return "mond"
        if node.get_slot("NSM_SEE") != 0:
            return "lát"
        if node.get_slot("NSM_HEAR") != 0:
            return "hall"
        if node.get_slot("NSM_THINK") != 0:
            return "gondol"
        if node.get_slot("NSM_DO") != 0:
            return "tesz"

        return "van"

    def _resolve_entity_with_case(
        self,
        graph: QuantaGraph,
        node: QuantaNode,
        relation: str,
        case: str,
    ) -> str:
        """Resolves target node of relation edge into an inflected Hungarian noun phrase."""
        if relation not in node.edges or not node.edges[relation]:
            return ""

        target_cid = node.edges[relation][0]
        target_node = graph.get_node(target_cid)
        if target_node is None:
            return ""

        # Extract Hungarian noun base
        noun_base = ""
        if target_node.anchor and target_node.anchor.startswith("wn:"):
            parts = target_node.anchor[3:].split(".")
            lemma = parts[0].replace("_", " ").lower()
            noun_base = self.DICTIONARY.get(lemma, lemma)

        if not noun_base and target_node.literal:
            lit = str(target_node.literal).strip().lower()
            noun_base = self.DICTIONARY.get(lit, lit)

        if not noun_base:
            if target_node.get_slot("TYPE_HUMAN") == 1:
                noun_base = "ember"
            elif target_node.get_slot("TYPE_ANIMATE") == 1:
                noun_base = "állat"
            else:
                noun_base = "dolog"

        # Apply case suffix to head noun
        inflected_noun = self.apply_case_suffix(noun_base, case)

        # Prepend determiner / article: 'a' / 'az' or quantifier
        article = "a"
        if inflected_noun and inflected_noun[0].lower() in "aáeéoóiíuúöőüű":
            article = "az"

        if target_node.get_slot("LJB_RO_ALL_QUANT") == 1 or target_node.get_slot("NSM_ALL") == 1:
            article = "minden"
        elif target_node.get_slot("NSM_TWO") == 1:
            article = "két"
        elif target_node.get_slot("NSM_SOME") == 1:
            article = "egy"

        # Adjective
        adj = ""
        if target_node.get_slot("NSM_BIG") == 1:
            adj = "nagy"
        elif target_node.get_slot("NSM_SMALL") == 1:
            adj = "kis"

        tokens = [article]
        if adj:
            tokens.append(adj)
        tokens.append(inflected_noun)

        return " ".join(tokens)
