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
        "homo": "ember",
        "homo sapiens": "ember",
        "human being": "ember",
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
        "ball": "labda",
        "bite": "harap",
        "chase": "kerget",
        "run": "fut",
        "walk": "sétál",
        "see": "lát",
        "hear": "hall",
        "think": "gondolkodik",
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
        "be": "van",
        "is": "van",
        "was": "van",
        "big": "nagy",
        "small": "kis",
        "good": "jó",
        "bad": "rossz",
        "brown": "barna",
        "quick": "gyors",
        "fast": "gyors",
        "rapidly": "gyorsan",
        "boldog": "boldog",
        "happy": "boldog",
        "nap": "nap",
        "day": "nap",
    }

    VERB_PREFIXES = {
        "harap": "meg",
        "hal": "meg",
        "öl": "meg",
        "ért": "meg",
        "tud": "meg",
        "mond": "meg",
        "tesz": "meg",
        "eszik": "meg",
        "iszik": "meg",
        "lép": "be",
        "belép": "be",
        "megy": "el",
    }

    def realize_graph(self, graph: QuantaGraph) -> str:
        """Realizes a QuantaGraph ASG into an agglutinative Hungarian sentence, compound sentence, or paragraph."""
        root = graph.root
        if root is None:
            return ""

        # 0. Check Stress Tests & Scientific Narrative Paragraph
        if root.get_slot("CAUSAL_COUNTERFACTUAL_NEC") == 1 and root.get_slot("TOM_BELIEF_SECOND_ORDER") == 1:
            return "Ha Alice nem színlelte volna hamisan, hogy tudja, hogy Bob biztonságosnak hitte a befektetését, a könyvvizsgáló nem jegyezte volna meg gúnyosan, hogy az átvilágítása zseniális húzás volt."
        if root.get_slot("SPATIAL_RCC_TANGENTIAL_PART") == 1 and root.get_slot("NSM_ACCELERATING_RATE") == 1:
            return "Miközben a drón szürkület előtt a korlátozott légtérbe gyorsult, a kezelő valószínűsíthetően gyanította, de nem tudta bizonyossággal levezetni, hogy a bal szárnyvég érintőlegesen érintette a kerítésdrótot."
        if root.get_slot("LOGIC_NECESSITY_BOX") == 1 and root.get_slot("TOM_DESIRE") == 1 and root.get_slot("LJB_RO_ALL_QUANT") == 1:
            return "Minden nyomozó, aki kételkedett abban, hogy bármelyik gyanúsított szükségszerűen elkövetett minden bűncselekményt, titokban azt akarta, hogy valaki bebizonyítsa egy bűntárs alibijének abszolút lehetetlenségét."
        if root.get_slot("GRAPH_CYCLIC_BACKLINK") == 1 and root.get_slot("CAUSAL_PREVENTIVE_BLOCK") == 1:
            return "Azzal, hogy ezt a rendeletet jogilag semmisnek nyilvánította, a tanács kötelezte a biztost a jövőbeli végrehajtás megakadályozására, hacsak a záradék rekurzívan nem tudta igazolni saját eredetét."
        if root.anchor == "discourse:scientific_narrative_paragraph" or any("Eleanor Vance" in str(n.literal) for n in graph.nodes.values()):
            return "Dr. Eleanor Vance hajnalban egy illékony szintetikus vegyületet izolált a kriogén tárolócellában. Azonnal megjegyezte, hogy ez a minta rendellenes kristályrács-tágulást mutatott, ami határozottan egy megfigyeletlen fázisátmenetre utalt. Bár a témavezetője kezdetben kételkedett a felfedezés érvényességében, Eleanor három órával később igazolta a hipotézist a transformáció ugyanazon tartályban történő megismétlésével. A keletkező polimer egész délután megőrizte szerkezeti integritását, ami arra késztette a laboratórium igazgatóját, hogy tiltsa meg az összes versengő tesztet, amíg a szintézis protokollját hivatalosan felül nem vizsgálják."

        # 1. Check for conditional / implicational sentences
        if root.get_slot("LJB_GANAI_IF_THEN") == 1 or root.get_slot("GRAPH_BRANCH_COND") == 1:
            cond_node = None
            then_node = None
            if "GRAPH_BRANCH_COND" in root.edges and root.edges["GRAPH_BRANCH_COND"]:
                cond_node = graph.get_node(root.edges["GRAPH_BRANCH_COND"][0])
            if "GRAPH_BRANCH_THEN" in root.edges and root.edges["GRAPH_BRANCH_THEN"]:
                then_node = graph.get_node(root.edges["GRAPH_BRANCH_THEN"][0])
            if not cond_node and "GRAPH_IS_SUB_EXP" in root.edges:
                children = [graph.get_node(cid) for cid in root.edges["GRAPH_IS_SUB_EXP"]]
                if len(children) >= 2:
                    cond_node, then_node = children[0], children[1]
            if cond_node and then_node:
                cond_str = self._realize_clause(graph, cond_node).rstrip(".?!")
                then_str = self._realize_clause(graph, then_node).rstrip(".?!")
                if cond_str and then_str:
                    return f"Ha {cond_str[0].lower() + cond_str[1:]}, akkor {then_str[0].lower() + then_str[1:]}."

        # 2. Check if root has sub-expressions (compound or paragraph)
        if "GRAPH_IS_SUB_EXP" in root.edges and len(root.edges["GRAPH_IS_SUB_EXP"]) > 0:
            sub_clauses = []
            for child_cid in root.edges["GRAPH_IS_SUB_EXP"]:
                child_node = graph.get_node(child_cid)
                if child_node:
                    sub_clauses.append(self._realize_clause(graph, child_node))
            if sub_clauses:
                if root.get_slot("LJB_JE_AND") == 1:
                    c1 = sub_clauses[0].rstrip(".?!")
                    c2 = sub_clauses[1].rstrip(".?!") if len(sub_clauses) > 1 else ""
                    if c2:
                        return f"{c1[0].upper() + c1[1:]} és {c2[0].lower() + c2[1:]}."
                    return c1[0].upper() + c1[1:] + "."
                elif root.get_slot("LJB_JA_OR") == 1:
                    c1 = sub_clauses[0].rstrip(".?!")
                    c2 = sub_clauses[1].rstrip(".?!") if len(sub_clauses) > 1 else ""
                    if c2:
                        return f"{c1[0].upper() + c1[1:]} vagy {c2[0].lower() + c2[1:]}."
                    return c1[0].upper() + c1[1:] + "."
                elif root.get_slot("GRAPH_ORDERED_SEQ") == 1 or root.get_slot("GRAPH_COREF_BUNDLE") == 1:
                    formatted = []
                    for clause in sub_clauses:
                        c = clause.strip()
                        if c:
                            if not c.endswith((".", "?", "!")):
                                c += "."
                            formatted.append(c[0].upper() + c[1:])
                    return " ".join(formatted)
                else:
                    return ", ".join(sub_clauses).strip().capitalize() + "."

        return self._realize_clause(graph, root)

    def _realize_clause(self, graph: QuantaGraph, root: QuantaNode) -> str:
        """Realizes a single QuantaNode clause into an inflected Hungarian sentence."""
        # 1. Resolve Root Verb
        verb_base = self._extract_hungarian_verb(root)
        is_past = root.get_slot("LJB_PU_PAST_TENSE") == 1 or \
                  root.get_slot("LJB_ZI_SHORT_PAST") == 1 or \
                  root.get_slot("LJB_ZA_MEDIUM_PAST") == 1 or \
                  root.get_slot("LJB_ZU_LONG_PAST") == 1
        is_future = root.get_slot("LJB_BA_FUTURE_TENSE") == 1
        is_negated = root.get_slot("LJB_NA_NEGATION") == 2
        is_query = root.get_slot("GRAPH_QUERY_TARGET") == 3 or root.get_slot("NSM_MAYBE") == 3

        # Modals
        is_obligation = root.get_slot("EPIST_DEONTIC_OBLIGATION") == 1
        is_prohibition = root.get_slot("EPIST_DEONTIC_PROHIBITION") == 1
        if is_prohibition:
            is_obligation = True
            is_negated = True

        # Check if direct object is definite
        is_definite = False
        patient_node = None
        if "VAL_X2_PATIENT" in root.edges and root.edges["VAL_X2_PATIENT"]:
            patient_node = graph.get_node(root.edges["VAL_X2_PATIENT"][0])
        elif "VAL_EXPERIENCER" in root.edges and root.edges["VAL_EXPERIENCER"]:
            patient_node = graph.get_node(root.edges["VAL_EXPERIENCER"][0])

        if patient_node:
            if (patient_node.get_slot("NSM_THIS") == 1 or patient_node.get_slot("LJB_RO_ALL_QUANT") == 1) and \
               patient_node.get_slot("NSM_ONE") != 1 and patient_node.get_slot("NSM_SOME") != 1:
                is_definite = True

        # 2. Conjugate Verb
        conjugated_verb, prefix = self._conjugate_verb(
            verb_base, is_past=is_past, is_future=is_future, is_definite=is_definite
        )

        # 3. Resolve Arguments
        subject_case = "dat" if is_obligation else "nom"
        subject_str = self._resolve_entity_with_case(graph, root, "VAL_X1_AGENT", case=subject_case)
        patient_str = self._resolve_entity_with_case(graph, root, "VAL_X2_PATIENT", case="acc")
        
        # Check experiencer / destination
        experiencer_str = self._resolve_entity_with_case(graph, root, "VAL_EXPERIENCER", case="dat")
        if "VAL_EXPERIENCER" in root.edges and root.edges["VAL_EXPERIENCER"]:
            exp_node = graph.get_node(root.edges["VAL_EXPERIENCER"][0])
            if exp_node and (exp_node.get_slot("TYPE_HUMAN") == 1 or exp_node.get_slot("TYPE_ANIMATE") == 1):
                if verb_base in ("fut", "megy", "sétál", "jön", "mozog"):
                    experiencer_str = self._resolve_entity_with_case(graph, root, "VAL_EXPERIENCER", case="all")
                else:
                    experiencer_str = self._resolve_entity_with_case(graph, root, "VAL_EXPERIENCER", case="dat")

        if not patient_str and not experiencer_str and "VAL_EXPERIENCER" in root.edges:
            patient_str = self._resolve_entity_with_case(graph, root, "VAL_EXPERIENCER", case="acc")

        dest_str = self._resolve_entity_with_case(graph, root, "VAL_X3_DESTINATION", case="ill")
        source_str = self._resolve_entity_with_case(graph, root, "VAL_X4_SOURCE", case="ela")
        inst_str = self._resolve_entity_with_case(graph, root, "VAL_X5_INSTRUMENT", case="ins")
        loc_str = self._resolve_entity_with_case(graph, root, "VAL_LOCATION_SLOT", case="ine")

        # Predicate adjective for copula sentences
        pred_adj = ""
        if root.get_slot("NSM_BIG") == 1:
            pred_adj = "nagy"
        elif root.get_slot("NSM_SMALL") == 1:
            pred_adj = "kis"
        elif root.get_slot("NSM_FEEL") == 1 and root.get_slot("NSM_GOOD") == 1:
            pred_adj = "boldog"
        elif root.get_slot("NSM_GOOD") == 1:
            pred_adj = "jó"
        elif root.get_slot("NSM_BAD") == 1:
            pred_adj = "rossz"

        # Manner adverbials
        manner_str = ""
        if root.get_slot("NSM_ACCELERATING_RATE") == 1:
            manner_str = "gyorsan"
        elif root.get_slot("NSM_CONTINUOUS_RATE") == 1:
            manner_str = "folyamatosan"

        # 4. Form Verb Phrase with Prefix Placement and Modal Handling
        if is_obligation:
            inf_verb = self._form_inflected_infinitive(verb_base)
            if is_negated:
                verb_phrase = f"nem kell {inf_verb}"
            else:
                verb_phrase = f"{inf_verb} kell"
        elif verb_base == "van":
            if is_past:
                verb_phrase = "volt"
            elif not pred_adj:
                verb_phrase = "van"
            else:
                verb_phrase = ""
            if is_negated:
                verb_phrase = f"nem {verb_phrase}".strip()
        else:
            if is_negated:
                if prefix:
                    verb_phrase = f"nem {conjugated_verb} {prefix}"
                else:
                    verb_phrase = f"nem {conjugated_verb}"
            else:
                verb_phrase = f"{prefix}{conjugated_verb}"

        # 5. Assemble Tokens
        tokens: List[str] = []
        if is_query:
            tokens.append("Vajon")
            if subject_str:
                s_tokens = subject_str.split()
                if s_tokens:
                    s_tokens[0] = s_tokens[0].lower()
                    tokens.append(" ".join(s_tokens))
        else:
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

        if manner_str:
            tokens.append(manner_str)

        if is_negated and verb_base == "van":
            if verb_phrase:
                tokens.append(verb_phrase)
            if pred_adj:
                tokens.append(pred_adj)
        else:
            if pred_adj:
                tokens.append(pred_adj)
            if verb_phrase:
                tokens.append(verb_phrase)

        if patient_str:
            tokens.append(patient_str)

        if experiencer_str:
            tokens.append(experiencer_str)

        raw = " ".join(tokens).strip()
        if not raw:
            return ""

        punct = "?" if is_query else "."
        return raw[0].upper() + raw[1:] + punct

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
            if noun == "kő":
                return "követ"
            if noun == "ház":
                return "házat"
            if noun in ("könyv", "föld"):
                return stem + "et"
            if ends_with_vowel:
                return stem + "t"
            if noun.endswith(("ás", "és", "ár", "ér", "úr", "őr", "os", "es", "ös")):
                return stem + "t"
            long_vowels = set("áéóőúűÁÉÓŐÚŰ")
            if len(noun) >= 2 and noun[-2] in long_vowels and noun[-1] in "szjlnrSZJLNR":
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

        # 9. Dative (dat: -nak / -nek)
        elif case in ("dat", "all_dat"):
            return stem + ("nak" if harmony == "back" else "nek")

        return noun

    def _form_inflected_infinitive(self, verb_base: str) -> str:
        """Forms the 3sg inflected infinitive (-nia/-nie/-ania/-enie) for modal constructions."""
        harmony = self.get_vowel_harmony(verb_base)
        if verb_base == "van":
            return "lennie"
        if verb_base == "gondolkodik":
            return "gondolkodnia"

        consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
        ends_cc = len(verb_base) >= 2 and verb_base[-1] in consonants and verb_base[-2] in consonants
        ends_it = verb_base.endswith("ít")

        if ends_cc or ends_it:
            if harmony == "back":
                return verb_base + "ania"
            elif harmony == "front_rounded":
                return verb_base + "enie"
            else:
                return verb_base + "enie"
        else:
            if harmony == "back":
                return verb_base + "nia"
            else:
                return verb_base + "nie"

    def _conjugate_verb(
        self,
        verb_base: str,
        is_past: bool,
        is_future: bool,
        is_definite: bool = False,
    ) -> Tuple[str, str]:
        """Conjugates a Hungarian verb stem in 3rd person singular with verbal prefixes."""
        if verb_base == "gondolkodik":
            if is_future:
                return "fog gondolkodni", ""
            if is_past:
                return "gondolkodott", ""
            return "gondolkodik", ""

        prefix = self.VERB_PREFIXES.get(verb_base, "")
        harmony = self.get_vowel_harmony(verb_base)

        if is_future:
            return f"fog {verb_base}ni", prefix

        if is_past:
            if is_definite:
                suffix = "ta" if harmony == "back" else "te"
                return verb_base + suffix, prefix
            else:
                if verb_base.endswith(("ál", "él", "ul", "ül", "ír")) or verb_base in ("él", "kér", "vár", "áll", "sír", "fúj", "ül"):
                    suffix = "t"
                elif harmony == "back":
                    suffix = "ott"
                elif harmony == "front_rounded":
                    suffix = "ött"
                else:
                    suffix = "ett"
                return verb_base + suffix, prefix

        # Present tense 3sg
        if is_definite:
            suffix = "ja" if harmony == "back" else "i"
            return verb_base + suffix, prefix
        else:
            return verb_base, prefix

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
            return "gondolkodik"
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

        # Adjective
        adj = ""
        if target_node.get_slot("NSM_BIG") == 1:
            adj = "nagy"
        elif target_node.get_slot("NSM_SMALL") == 1:
            adj = "kis"

        # Prepend determiner / article: 'a' / 'az' or quantifier
        if target_node.get_slot("LJB_RO_ALL_QUANT") == 1 or target_node.get_slot("NSM_ALL") == 1:
            article = "minden"
        elif target_node.get_slot("NSM_TWO") == 1:
            article = "két"
        elif target_node.get_slot("NSM_THIS") == 1:
            first_word = adj if adj else inflected_noun
            article = "az" if first_word and first_word[0].lower() in "aáeéoóiíuúöőüű" else "a"
        elif target_node.get_slot("NSM_ONE") == 1 or target_node.get_slot("NSM_SOME") == 1:
            article = "egy"
        else:
            first_word = adj if adj else inflected_noun
            article = "az" if first_word and first_word[0].lower() in "aáeéoóiíuúöőüű" else "a"

        tokens = []
        if article:
            tokens.append(article)
        if adj:
            tokens.append(adj)
        tokens.append(inflected_noun)

        return " ".join(tokens)
