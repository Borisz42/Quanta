"""Hungarian Forward Parser for QUANTA.

Translates Hungarian natural language sentences into Content-Addressed Abstract Syntax Graphs (ASGs)
and 256-dimensional quaternary semantic vectors using morphological case analysis.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, QuaternaryValue
from parser.lexical_grounder import WordNetLexicalGrounder


class HungarianForwardParser:
    """Morphological forward parser mapping Hungarian sentences to Quanta ASG topologies."""

    HU_TO_EN_DICTIONARY = {
        "kutya": "dog",
        "golden retriever": "golden retriever",
        "macska": "cat",
        "postás": "mailman",
        "ember": "person",
        "kő": "rock",
        "kert": "garden",
        "ház": "house",
        "autó": "car",
        "fa": "tree",
        "víz": "water",
        "könyv": "book",
        "város": "city",
        "bot": "stick",
        "labda": "ball",
        "harap": "bite",
        "kerget": "chase",
        "fut": "run",
        "sétál": "walk",
        "lát": "see",
        "hall": "hear",
        "gondol": "think",
        "tud": "know",
        "mond": "say",
        "mesél": "tell",
        "ad": "give",
        "vesz": "take",
        "megy": "go",
        "jön": "come",
        "él": "live",
        "meghal": "die",
        "akar": "want",
        "érez": "feel",
        "mozog": "move",
        "belép": "enter",
        "érint": "touch",
        "van": "be",
    }

    ADJECTIVES = {
        "nagy": "big",
        "kis": "small",
        "kicsi": "small",
        "jó": "good",
        "rossz": "bad",
        "barna": "brown",
        "fekete": "black",
        "fehér": "white",
        "gyors": "fast",
        "lassú": "slow",
    }

    ARTICLES = {"a", "az", "egy", "minden", "két", "kettő"}

    def __init__(self, offline_cache_path: Optional[str] = None):
        self.grounder = WordNetLexicalGrounder(offline_cache_path=offline_cache_path)

    def parse_sentence(self, text: str, domain_context: Optional[str] = None) -> QuantaGraph:
        """Parses a Hungarian sentence into a QuantaGraph ASG."""
        clean_text = text.strip().rstrip(".?!")
        words = clean_text.split()
        graph = QuantaGraph()

        # 1. Identify Verb, Tense & Negation
        has_negation = False
        verb_stem = None
        is_past = False

        for i, word in enumerate(words):
            w_lower = word.lower()
            if w_lower == "nem":
                has_negation = True
                continue

            stem, past_detected = self._stem_verb(w_lower)
            if stem in self.HU_TO_EN_DICTIONARY and self.HU_TO_EN_DICTIONARY[stem] in (
                "bite", "chase", "run", "walk", "see", "hear", "think", "know",
                "say", "tell", "give", "take", "go", "come", "live", "die",
                "want", "feel", "move", "enter", "touch", "be"
            ):
                verb_stem = stem
                is_past = past_detected
                break

        if not verb_stem:
            verb_stem = "van"

        en_verb = self.HU_TO_EN_DICTIONARY.get(verb_stem, "be")
        polarity = 2 if has_negation else 1

        # Construct Root Node
        root_node = QuantaNode(literal=text.strip())
        root_node.set_slot("MODALITY_LITERAL", 1)
        root_node.set_slot("GRAPH_ROOT_NODE", 1)
        root_node.anchor = f"wn:{en_verb}.v.01"

        if is_past:
            root_node.set_slot("LJB_PU_PAST_TENSE", 1)
        else:
            root_node.set_slot("LJB_CA_PRESENT_TENSE", 1)

        if has_negation:
            root_node.set_slot("LJB_NA_NEGATION", 2)
        else:
            root_node.set_slot("NSM_TRUE", 1)

        # Apply NSM prime mapping to Root
        self._apply_verb_primes(root_node, en_verb, polarity)
        root_cid = graph.add_node(root_node, set_as_root=True)

        # 2. Extract Noun Phrases & Case Roles
        # Segment into token groups
        noun_phrases = self._extract_noun_phrases(words, verb_stem)

        for np_tokens, case in noun_phrases:
            entity_node = self._create_entity_from_tokens(np_tokens, case)
            if not entity_node:
                continue

            entity_cid = graph.add_node(entity_node)

            if case == "nom":
                entity_node.set_slot("VAL_X1_AGENT", 1)
                entity_node.set_slot("ROLE_AGENT_CAPABLE", 1)
                root_node.set_slot("VAL_X1_AGENT", 1)
                graph.add_edge(root_node, "VAL_X1_AGENT", entity_node)
            elif case == "acc":
                entity_node.set_slot("VAL_X2_PATIENT", 1)
                root_node.set_slot("VAL_X2_PATIENT", 1)
                graph.add_edge(root_node, "VAL_X2_PATIENT", entity_node)
            elif case == "ine":
                entity_node.set_slot("VAL_LOCATION_SLOT", 1)
                entity_node.set_slot("TYPE_SPATIAL_REGION", 1)
                entity_node.set_slot("NSM_INSIDE", 1)
                root_node.set_slot("VAL_LOCATION_SLOT", 1)
                graph.add_edge(root_node, "VAL_LOCATION_SLOT", entity_node)
            elif case in ("ill", "sub", "all"):
                entity_node.set_slot("VAL_X3_DESTINATION", 1)
                entity_node.set_slot("TYPE_SPATIAL_REGION", 1)
                root_node.set_slot("VAL_X3_DESTINATION", 1)
                graph.add_edge(root_node, "VAL_X3_DESTINATION", entity_node)
            elif case in ("ela", "abl"):
                entity_node.set_slot("VAL_X4_SOURCE", 1)
                entity_node.set_slot("TYPE_SPATIAL_REGION", 1)
                root_node.set_slot("VAL_X4_SOURCE", 1)
                graph.add_edge(root_node, "VAL_X4_SOURCE", entity_node)
            elif case == "ins":
                entity_node.set_slot("VAL_X5_INSTRUMENT", 1)
                entity_node.set_slot("ROLE_INSTRUMENT_USABLE", 1)
                root_node.set_slot("VAL_X5_INSTRUMENT", 1)
                graph.add_edge(root_node, "VAL_X5_INSTRUMENT", entity_node)

        return graph

    def _stem_verb(self, word: str) -> Tuple[str, bool]:
        """Stems a Hungarian verb and detects past tense."""
        w = word.lower()
        if w in self.HU_TO_EN_DICTIONARY:
            return w, False

        # Past tense suffixes: -ott, -ett, -ött, -t
        past_suffixes = ["ott", "ett", "ött", "t"]
        for suff in past_suffixes:
            if w.endswith(suff) and len(w) > len(suff) + 2:
                stem = w[:-len(suff)]
                if stem in self.HU_TO_EN_DICTIONARY:
                    return stem, True
                # Check doubled consonant reduction (e.g. kergetett -> kerget)
                if stem.endswith("t") and stem[:-1] in self.HU_TO_EN_DICTIONARY:
                    return stem[:-1], True

        return w, False

    def _extract_noun_phrases(self, words: List[str], verb_stem: str) -> List[Tuple[List[str], str]]:
        """Segments tokens into noun phrases and determines their case."""
        nps = []
        current_np: List[str] = []

        for word in words:
            w_lower = word.lower()
            if w_lower == "nem" or w_lower.startswith(verb_stem):
                if current_np:
                    case = self._detect_case(current_np[-1])
                    nps.append((list(current_np), case))
                    current_np = []
                continue

            current_np.append(word)

            # If token has a case suffix or is a known noun
            case = self._detect_case(w_lower)
            if case != "nom":
                nps.append((list(current_np), case))
                current_np = []

        if current_np:
            case = self._detect_case(current_np[-1])
            nps.append((list(current_np), case))

        return nps

    def _detect_case(self, word: str) -> str:
        """Identifies grammatical case suffix from word ending."""
        w = word.lower()
        if w.endswith(("ban", "ben")):
            return "ine"
        if w.endswith(("ba", "be")):
            return "ill"
        if w.endswith(("ból", "ből")):
            return "ela"
        if w.endswith(("ra", "re")):
            return "sub"
        if w.endswith(("hoz", "hez", "höz")):
            return "all"
        if w.endswith(("val", "vel")) or (len(w) > 3 and w.endswith("al") and w[-3] == w[-4]) or (len(w) > 3 and w.endswith("el") and w[-3] == w[-4]):
            return "ins"
        if w.endswith("t") or w.endswith(("ot", "et", "öt", "at", "át", "ét")):
            # Check it's not a verb
            stem = self._stem_noun(w)
            if stem in self.HU_TO_EN_DICTIONARY:
                return "acc"

        return "nom"

    def _stem_noun(self, word: str) -> str:
        """Strips Hungarian case suffixes to recover the noun lemma."""
        w = word.lower()
        if w in self.HU_TO_EN_DICTIONARY:
            return w

        # Common suffixes
        suffixes = [
            ("ban", "ine"), ("ben", "ine"),
            ("ba", "ill"), ("be", "ill"),
            ("ból", "ela"), ("ből", "ela"),
            ("ra", "sub"), ("re", "sub"),
            ("hoz", "all"), ("hez", "all"), ("höz", "all"),
            ("val", "ins"), ("vel", "ins"),
            ("ot", "acc"), ("et", "acc"), ("öt", "acc"), ("at", "acc"),
            ("t", "acc"),
        ]

        for suff, _ in suffixes:
            if w.endswith(suff) and len(w) > len(suff) + 1:
                stem = w[:-len(suff)]
                # Handle stem vowel shortening (á -> a, é -> e)
                if stem.endswith("á"):
                    shortened = stem[:-1] + "a"
                    if shortened in self.HU_TO_EN_DICTIONARY:
                        return shortened
                elif stem.endswith("é"):
                    shortened = stem[:-1] + "e"
                    if shortened in self.HU_TO_EN_DICTIONARY:
                        return shortened
                elif stem in self.HU_TO_EN_DICTIONARY:
                    return stem
                # Consonant degemination for instrumental (e.g. bottal -> bot)
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    degem = stem[:-1]
                    if degem in self.HU_TO_EN_DICTIONARY:
                        return degem

        return w

    def _create_entity_from_tokens(self, tokens: List[str], case: str) -> Optional[QuantaNode]:
        """Creates a grounded QuantaNode from noun phrase tokens."""
        if not tokens:
            return None

        # Separate article, adjectives, head noun
        head_word = tokens[-1]
        stem = self._stem_noun(head_word)
        en_lemma = self.HU_TO_EN_DICTIONARY.get(stem, stem)

        try:
            concept = self.grounder.ground_synset(en_lemma)
            node = QuantaNode(vector=concept.vector.copy(), anchor=concept.synset_name, literal=en_lemma)
        except Exception:
            node = QuantaNode(literal=en_lemma)
            node.set_slot("TYPE_INANIMATE_PHYSICAL", 1)
            node.anchor = f"wn:{en_lemma}.n.01"

        # Apply adjectives
        for token in tokens[:-1]:
            t_lower = token.lower()
            if t_lower in self.ADJECTIVES:
                en_adj = self.ADJECTIVES[t_lower]
                if en_adj == "big":
                    node.set_slot("NSM_BIG", 1)
                elif en_adj == "small":
                    node.set_slot("NSM_SMALL", 1)
                elif en_adj == "good":
                    node.set_slot("NSM_GOOD", 1)
                elif en_adj == "bad":
                    node.set_slot("NSM_BAD", 1)

            # Quantifiers
            if t_lower in ("minden", "mindegyik"):
                node.set_slot("NSM_ALL", 1)
                node.set_slot("LJB_RO_ALL_QUANT", 1)
            elif t_lower in ("két", "kettő"):
                node.set_slot("NSM_TWO", 1)
            elif t_lower == "egy":
                node.set_slot("NSM_ONE", 1)

        return node

    def _apply_verb_primes(self, node: QuantaNode, en_verb: str, polarity: int):
        """Applies NSM primes to the root verb."""
        motion = {"move", "run", "walk", "chase", "go", "come", "enter"}
        speech = {"say", "tell"}
        cognition = {"think", "know"}
        perception = {"see", "hear"}
        contact = {"bite", "touch"}

        if en_verb in motion:
            node.set_slot("NSM_MOVE", polarity)
            node.set_slot("TYPE_EVENT", 1)
        elif en_verb in speech:
            node.set_slot("NSM_SAY", polarity)
            node.set_slot("NSM_WORDS", polarity)
            node.set_slot("TYPE_COMMUNICATION_MSG", 1)
        elif en_verb in cognition:
            node.set_slot("NSM_THINK", polarity)
            if en_verb == "know":
                node.set_slot("NSM_KNOW", polarity)
            node.set_slot("TYPE_STATE", 1)
        elif en_verb in perception:
            if en_verb == "see":
                node.set_slot("NSM_SEE", polarity)
            else:
                node.set_slot("NSM_HEAR", polarity)
            node.set_slot("TYPE_EVENT", 1)
            node.set_slot("VAL_EXPERIENCER", 1)
        elif en_verb in contact:
            node.set_slot("NSM_DO", polarity)
            node.set_slot("NSM_TOUCH", polarity)
            node.set_slot("TYPE_EVENT", 1)
        else:
            node.set_slot("NSM_DO", polarity)
            node.set_slot("TYPE_EVENT", 1)
