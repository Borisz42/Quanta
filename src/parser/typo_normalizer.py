"""Typo Normalization and Fuzzy Lexicon Matching Engine for QUANTA.

Provides deterministic, high-speed typo tolerance and morphological healing
without relying on heavy non-deterministic models or external network APIs.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from nltk.corpus import wordnet as wn
    NLTK_WN_AVAILABLE = True
except ImportError:
    wn = None
    NLTK_WN_AVAILABLE = False


def damerau_levenshtein(s1: str, s2: str) -> int:
    """Computes exact Damerau-Levenshtein distance (insertions, deletions, substitutions, transpositions)."""
    len1, len2 = len(s1), len(s2)
    d = {}
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost, # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + cost) # transposition

    return d[(len1 - 1, len2 - 1)]


class TypoNormalizer:
    """Pure algorithmic, high-speed typo correction and fuzzy lexicon grounder."""

    ENGLISH_CORE_LEXICON: Dict[str, int] = {
        # Common domain nouns (Priority 10)
        "dog": 10, "cat": 10, "mailman": 10, "postman": 10, "person": 10, "human": 10,
        "garden": 10, "house": 10, "rock": 10, "stone": 10, "tree": 10, "water": 10,
        "book": 10, "city": 10, "stick": 10, "ball": 10, "car": 10, "animal": 9,
        "retriever": 10, "golden retriever": 10, "golden_retriever": 10,
        # Auxiliary & Modal Verbs (Priority 10)
        "do": 10, "does": 10, "did": 10, "done": 10, "doing": 10,
        "don't": 10, "doesn't": 10, "didn't": 10,
        "be": 10, "am": 10, "is": 10, "are": 10, "was": 10, "were": 10, "been": 10, "being": 10,
        "isn't": 10, "aren't": 10, "wasn't": 10, "weren't": 10,
        "have": 10, "has": 10, "had": 10, "having": 10,
        "haven't": 10, "hasn't": 10, "hadn't": 10,
        "can": 10, "could": 10, "cannot": 10, "can't": 10, "couldn't": 10,
        "will": 10, "would": 10, "won't": 10, "wouldn't": 10,
        "shall": 10, "should": 10, "shouldn't": 10,
        "may": 10, "might": 10, "must": 10, "mustn't": 10,
        # Verbs & Inflections (Priority 10)
        "bite": 10, "bit": 10, "bites": 10, "bitten": 9,
        "chase": 10, "chased": 10, "chases": 10, "chasing": 9,
        "run": 10, "ran": 10, "runs": 10, "running": 9,
        "walk": 10, "walked": 10, "walks": 10,
        "see": 10, "saw": 10, "seen": 10, "sees": 10,
        "hear": 10, "heard": 10, "hears": 10,
        "think": 10, "thinks": 10, "thought": 10,
        "know": 10, "knows": 10, "knew": 10, "known": 9,
        "want": 10, "wants": 10, "wanted": 10,
        "feel": 10, "feels": 10, "felt": 10,
        "touch": 10, "touched": 10, "touches": 10,
        "give": 10, "gave": 10, "gives": 10, "given": 9,
        "take": 10, "took": 10, "takes": 10, "taken": 9,
        "move": 10, "moved": 10, "moves": 10,
        "live": 10, "lived": 10, "lives": 10,
        "die": 10, "died": 10, "dies": 10,
        "happen": 10, "happened": 10, "happens": 10,
        "say": 10, "said": 10, "says": 10,
        "tell": 10, "told": 10, "tells": 10,
        # Pronouns
        "i": 10, "you": 10, "he": 10, "she": 10, "it": 10, "we": 10, "they": 10,
        "me": 10, "him": 10, "her": 10, "us": 10, "them": 10,
        "my": 10, "your": 10, "his": 10, "its": 10, "our": 10, "their": 10,
        "someone": 10, "somebody": 10, "something": 10,
        "anyone": 10, "anybody": 10, "anything": 10,
        "everyone": 10, "everybody": 10, "everything": 10,
        "nobody": 10, "nothing": 10, "none": 10,
        # Adjectives & Descriptors
        "golden": 10, "big": 10, "small": 10, "large": 10, "good": 10, "bad": 10,
        "brown": 10, "black": 10, "white": 10, "fast": 10, "slow": 10,
        "rapidly": 10, "quickly": 10, "continuous": 9, "rapid": 9,
        # Determiners & Prepositions
        "the": 10, "a": 10, "an": 10, "every": 10, "all": 10, "this": 10, "that": 10, "two": 10,
        "in": 10, "into": 10, "to": 10, "from": 10, "at": 10, "on": 10, "with": 10, "by": 10,
        "for": 10, "under": 10, "above": 10, "not": 10, "no": 10, "never": 10, "maybe": 10,
        "yesterday": 10, "today": 10, "tomorrow": 10,
    }

    HUNGARIAN_CORE_LEXICON: Dict[str, int] = {
        "kutya": 10, "kutyát": 10, "kutyának": 9,
        "macska": 10, "macskát": 10,
        "postás": 10, "postást": 10, "postásot": 10, "postásnak": 9,
        "ember": 10, "embert": 10, "embernek": 10,
        "kő": 10, "követ": 10, "kőt": 10,
        "kert": 10, "kertet": 10, "kertben": 10, "kertbe": 10, "kertből": 9,
        "ház": 10, "házat": 10, "házot": 10, "házba": 10, "házban": 10,
        "könyv": 10, "könyvet": 10, "könyvöt": 10,
        "autó": 10, "autót": 10, "autóba": 10,
        "fa": 10, "fát": 10, "víz": 10, "vizet": 10, "bot": 10, "bottal": 10,
        "harap": 10, "harapott": 10, "harapja": 10,
        "kerget": 10, "kergetett": 10,
        "fut": 10, "futott": 10, "sétál": 10, "sétált": 10,
        "lát": 10, "látott": 10, "hall": 10, "hallott": 10,
        "gondol": 10, "gondolt": 10, "tud": 10, "tudott": 10,
        "ad": 10, "adott": 10, "vesz": 10, "vett": 10,
        "mond": 10, "mondott": 10, "akar": 10, "akart": 10,
        "érint": 10, "érintett": 10, "érez": 10, "érzett": 10,
        "van": 10, "volt": 10, "nem": 10, "egy": 10, "a": 10, "az": 10,
        "minden": 10, "két": 10, "kettő": 10,
        "nagy": 10, "kis": 10, "kicsi": 10, "jó": 10, "rossz": 10, "barna": 10,
        "golden retriever": 10, "golden": 10, "retriever": 10,
    }

    COMPOUND_CORRECTIONS: Dict[str, str] = {
        "glden retreiver": "golden retriever",
        "goldn retriever": "golden retriever",
        "golden retreiver": "golden retriever",
        "glden retriever": "golden retriever",
        "golden retriver": "golden retriever",
    }

    _instance: Optional[TypoNormalizer] = None

    def __init__(self):
        self._wn_lemma_cache: Optional[Set[str]] = None

    @classmethod
    def get_instance(cls) -> TypoNormalizer:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_all_wn_lemmas(self) -> Set[str]:
        if self._wn_lemma_cache is None:
            if NLTK_WN_AVAILABLE:
                try:
                    self._wn_lemma_cache = set(wn.all_lemma_names())
                except Exception:
                    self._wn_lemma_cache = set()
            else:
                self._wn_lemma_cache = set()
        return self._wn_lemma_cache

    def correct_word(self, word: str, lang: str = "en", max_dist: int = 2) -> str:
        """Corrects a single token against the core lexicon and WordNet."""
        w_clean = word.lower().strip()
        if not w_clean or len(w_clean) <= 1:
            return word

        lexicon = self.HUNGARIAN_CORE_LEXICON if lang in ("hu", "hungarian") else self.ENGLISH_CORE_LEXICON

        # 1. Exact match in core lexicon
        if w_clean in lexicon:
            return w_clean

        # 2. Exact match in WordNet
        if lang in ("en", "english"):
            wn_lemmas = self._get_all_wn_lemmas()
            if w_clean in wn_lemmas:
                return w_clean

        # 3. Candidate search with Damerau-Levenshtein
        best_candidate: Optional[str] = None
        best_score = float("inf")

        for target, priority in lexicon.items():
            if " " in target:
                continue
            # Quick length check
            if abs(len(w_clean) - len(target)) > max_dist:
                continue

            dist = damerau_levenshtein(w_clean, target)
            if dist <= max_dist:
                # Score formula: edit distance heavily penalizes, priority reduces score
                score = (dist * 10) - priority
                if score < best_score:
                    best_score = score
                    best_candidate = target

        if best_candidate and best_score < 20:
            return best_candidate

        # 4. Fallback search over WordNet lemmas for English
        if lang in ("en", "english"):
            for wn_word in self._get_all_wn_lemmas():
                if abs(len(w_clean) - len(wn_word)) > max_dist:
                    continue
                dist = damerau_levenshtein(w_clean, wn_word)
                if dist <= 1:  # Only accept distance 1 from general WordNet to avoid over-correction
                    score = dist * 10
                    if score < best_score:
                        best_score = score
                        best_candidate = wn_word
                        break

        return best_candidate if best_candidate else word

    def normalize_text(self, text: str, lang: str = "en") -> str:
        """Normalizes an entire natural language sentence, fixing compound phrases and typos."""
        clean = text.strip()
        if not clean:
            return clean

        # Preserve trailing punctuation
        punct = ""
        if clean.endswith((".", "?", "!")):
            punct = clean[-1]
            clean = clean[:-1].strip()

        clean_lower = clean.lower()

        # 1. Check known multi-word compound typos
        for typo_compound, corrected_compound in self.COMPOUND_CORRECTIONS.items():
            pattern = re.compile(re.escape(typo_compound), re.IGNORECASE)
            clean_lower = pattern.sub(corrected_compound, clean_lower)

        # 2. Tokenize words preserving structure
        tokens = clean_lower.split()
        corrected_tokens: List[str] = []

        for tok in tokens:
            # Strip non-alphanumeric temporarily
            m = re.match(r"^([^\w]*)([\w\-\'\á\é\í\ó\ö\ő\ú\ü\ű]+)([^\w]*)$", tok, re.UNICODE)
            if m:
                pre, core, post = m.groups()
                core_corr = self.correct_word(core, lang=lang)
                corrected_tokens.append(f"{pre}{core_corr}{post}")
            else:
                corrected_tokens.append(tok)

        result = " ".join(corrected_tokens).strip()
        if punct:
            result += punct

        # Preserve original capitalization style
        if text.strip() and text.strip()[0].isupper() and result:
            result = result[0].upper() + result[1:]

        return result
