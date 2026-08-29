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
        # Common domain nouns - Animals (Priority 10)
        "dog": 10, "puppy": 10, "cat": 10, "kitten": 10, "mailman": 10, "postman": 10,
        "person": 10, "human": 10, "man": 10, "woman": 10, "child": 10, "boy": 10, "girl": 10, "baby": 10,
        "animal": 10, "mammal": 10, "bird": 10, "fish": 10, "reptile": 10, "insect": 10,
        "lion": 10, "tiger": 10, "bear": 10, "squirrel": 10, "eagle": 10, "mouse": 10, "rat": 10,
        "elephant": 10, "horse": 10, "cow": 10, "pig": 10, "sheep": 10, "snake": 10, "deer": 10,
        "rabbit": 10, "monkey": 10, "wolf": 10, "fox": 10, "retriever": 10, "golden retriever": 10, "golden_retriever": 10,
        # Human professions & Kinship (Priority 10)
        "student": 10, "teacher": 10, "professor": 10, "doctor": 10, "nurse": 10,
        "police": 10, "officer": 10, "judge": 10, "lawyer": 10, "scientist": 10, "worker": 10,
        "driver": 10, "cook": 10, "author": 10, "father": 10, "mother": 10, "son": 10,
        "daughter": 10, "brother": 10, "sister": 10, "grandfather": 10, "grandmother": 10,
        "grandson": 10, "granddaughter": 10, "uncle": 10, "aunt": 10, "nephew": 10, "niece": 10,
        "cousin": 10, "husband": 10, "wife": 10, "parent": 10, "sibling": 10, "friend": 10,
        "enemy": 10, "partner": 10, "neighbor": 10,
        # Places & Locations (Priority 10)
        "garden": 10, "house": 10, "home": 10, "building": 10, "room": 10, "kitchen": 10,
        "bedroom": 10, "bathroom": 10, "hallway": 10, "office": 10, "school": 10,
        "university": 10, "hospital": 10, "store": 10, "shop": 10, "market": 10, "city": 10,
        "town": 10, "village": 10, "street": 10, "road": 10, "forest": 10, "mountain": 10,
        "river": 10, "lake": 10, "ocean": 10, "sea": 10, "park": 10, "field": 10, "library": 10,
        # Tools, Artifacts & Objects (Priority 10)
        "rock": 10, "stone": 10, "stick": 10, "ball": 10, "car": 10, "automobile": 10,
        "vehicle": 10, "bicycle": 10, "truck": 10, "train": 10, "plane": 10, "boat": 10, "ship": 10,
        "book": 10, "paper": 10, "letter": 10, "message": 10, "computer": 10, "phone": 10,
        "table": 10, "chair": 10, "bed": 10, "desk": 10, "door": 10, "window": 10,
        "box": 10, "container": 10, "bag": 10, "cup": 10, "bottle": 10, "glass": 10,
        "key": 10, "clock": 10, "lamp": 10, "hammer": 10, "knife": 10, "fork": 10, "spoon": 10,
        "tool": 10, "weapon": 10, "sword": 10, "gun": 10,
        # Food, Substances, Materials & Nature (Priority 10)
        "water": 10, "milk": 10, "juice": 10, "tea": 10, "coffee": 10, "apple": 10,
        "fruit": 10, "banana": 10, "orange": 10, "bread": 10, "meat": 10, "cheese": 10,
        "food": 10, "meal": 10, "sugar": 10, "salt": 10, "gold": 10, "silver": 10, "iron": 10,
        "wood": 10, "air": 10, "fire": 10, "earth": 10, "soil": 10, "ice": 10, "snow": 10,
        "rain": 10, "wind": 10, "sun": 10, "moon": 10, "star": 10, "tree": 10, "plant": 10,
        "flower": 10, "grass": 10,
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
        "put": 10, "place": 10, "set": 10, "lay": 10, "drop": 10, "dropped": 10,
        "hold": 10, "held": 10, "grasp": 10, "grip": 10, "carry": 10, "carried": 10,
        "buy": 10, "bought": 10, "sell": 10, "sold": 10, "pay": 10, "paid": 10,
        "meet": 10, "met": 10, "visit": 10, "visited": 10, "greet": 10,
        "search": 10, "seek": 10, "sought": 10, "hunt": 10, "find": 10, "found": 10,
        "help": 10, "helped": 10, "assist": 10, "support": 10,
        "kill": 10, "killed": 10, "destroy": 10, "destroyed": 10, "damage": 10, "break": 10, "broke": 10,
        "repair": 10, "fix": 10, "fixed": 10,
        "create": 10, "created": 10, "build": 10, "built": 10, "construct": 10, "write": 10, "wrote": 10,
        "understand": 10, "understood": 10, "believe": 10, "suppose": 10, "imagine": 10,
        "remember": 10, "forget": 10, "forgot": 10, "learn": 10, "study": 10,
        "need": 10, "needed": 10, "wish": 10, "hope": 10, "like": 10, "liked": 10, "love": 10, "loved": 10,
        "hate": 10, "hated": 10, "dislike": 10,
        "step": 10, "swim": 10, "swam": 10, "climb": 10, "fall": 10, "fell": 10,
        "enter": 10, "entered": 10, "exit": 10, "cross": 10, "return": 10,
        "ask": 10, "asked": 10, "answer": 10, "answered": 10, "reply": 10,
        # Pronouns
        "i": 10, "you": 10, "he": 10, "she": 10, "it": 10, "we": 10, "they": 10,
        "me": 10, "him": 10, "her": 10, "us": 10, "them": 10,
        "my": 10, "your": 10, "his": 10, "its": 10, "our": 10, "their": 10,
        "someone": 10, "somebody": 10, "something": 10,
        "anyone": 10, "anybody": 10, "anything": 10,
        "everyone": 10, "everybody": 10, "everything": 10,
        "nobody": 10, "nothing": 10, "none": 10,
        # Adjectives & Descriptors
        "golden": 10, "big": 10, "small": 10, "tiny": 10, "large": 10, "huge": 10,
        "good": 10, "bad": 10, "great": 10, "terrible": 10,
        "brown": 10, "black": 10, "white": 10, "red": 10, "blue": 10, "green": 10, "yellow": 10, "gray": 10, "grey": 10,
        "dark": 10, "light": 10, "bright": 10,
        "hot": 10, "cold": 10, "warm": 10, "cool": 10,
        "fast": 10, "slow": 10, "rapidly": 10, "quickly": 10, "continuous": 9, "rapid": 9,
        "hard": 10, "soft": 10, "rough": 10, "smooth": 10,
        "clean": 10, "dirty": 10, "young": 10, "old": 10, "new": 10, "ancient": 10,
        "smart": 10, "clever": 10, "intelligent": 10, "quiet": 10, "loud": 10, "silent": 10, "noisy": 10,
        "round": 10, "square": 10, "sharp": 10, "blunt": 10, "strong": 10, "weak": 10,
        "happy": 10, "sad": 10, "angry": 10, "calm": 10, "safe": 10, "dangerous": 10,
        "true": 10, "false": 10, "valid": 10, "invalid": 10,
        # Determiners, Conjunctions & Prepositions
        "the": 10, "a": 10, "an": 10, "and": 10, "or": 10, "nor": 10, "but": 10, "yet": 10, "so": 10,
        "if": 10, "then": 10, "because": 10, "since": 10, "as": 10, "while": 10, "of": 10,
        "every": 10, "all": 10, "some": 10, "any": 10, "each": 10, "both": 10, "many": 10, "few": 10,
        "this": 10, "that": 10, "these": 10, "those": 10,
        "one": 10, "two": 10, "three": 10, "four": 10, "five": 10,
        "in": 10, "into": 10, "to": 10, "from": 10, "at": 10, "on": 10, "with": 10, "by": 10,
        "for": 10, "under": 10, "above": 10, "below": 10, "between": 10, "near": 10, "inside": 10, "outside": 10,
        "through": 10, "over": 10, "out": 10, "off": 10, "up": 10, "down": 10, "about": 10,
        "around": 10, "behind": 10, "against": 10, "without": 10, "within": 10,
        "not": 10, "no": 10, "never": 10, "maybe": 10, "perhaps": 10,
        "yesterday": 10, "today": 10, "tomorrow": 10, "before": 10, "after": 10, "during": 10,
        "who": 10, "whom": 10, "whose": 10, "what": 10, "which": 10, "where": 10, "when": 10, "how": 10, "why": 10,
    }

    HUNGARIAN_CORE_LEXICON: Dict[str, int] = {
        "kutya": 10, "kutyát": 10, "kutyának": 10,
        "macska": 10, "macskát": 10,
        "postás": 10, "postást": 10, "postásot": 10, "postásnak": 10,
        "ember": 10, "embert": 10, "embernek": 10,
        "kő": 10, "követ": 10, "kőt": 10,
        "kert": 10, "kertet": 10, "kertben": 10, "kertbe": 10, "kertből": 10,
        "ház": 10, "házat": 10, "házot": 10, "házba": 10, "házban": 10,
        "könyv": 10, "könyvet": 10, "könyvöt": 10,
        "autó": 10, "autót": 10, "autóba": 10,
        "fa": 10, "fát": 10, "víz": 10, "vizet": 10, "bot": 10, "bottal": 10, "labda": 10, "labdát": 10,
        "harap": 10, "harapott": 10, "harapja": 10, "harapta": 10, "megharapta": 10, "megharapott": 10,
        "kerget": 10, "kergetett": 10, "kergette": 10, "kergeti": 10,
        "fut": 10, "futott": 10, "elfutott": 10, "befutott": 10, "sétál": 10, "sétált": 10, "elsétált": 10, "elsétálott": 10,
        "lát": 10, "látott": 10, "látta": 10, "látja": 10, "meglátott": 10, "meglátta": 10, "meglátja": 10,
        "hall": 10, "hallott": 10,
        "gondol": 10, "gondolt": 10, "gondolkodik": 10, "gondolkozik": 10, "tud": 10, "tudott": 10,
        "ad": 10, "adott": 10, "adta": 10, "vesz": 10, "vett": 10,
        "mond": 10, "mondott": 10, "akar": 10, "akart": 10, "akarja": 10, "akarta": 10,
        "érint": 10, "érintett": 10, "érez": 10, "érzett": 10, "él": 10, "élt": 10,
        "van": 10, "volt": 10, "nem": 10, "egy": 10, "a": 10, "az": 10, "vajon": 10,
        "minden": 10, "két": 10, "kettő": 10,
        "nagy": 10, "kis": 10, "kicsi": 10, "jó": 10, "rossz": 10, "barna": 10, "gyors": 10, "gyorsan": 10,
        "golden retriever": 10, "golden": 10, "retriever": 10,
    }

    COMPOUND_CORRECTIONS: Dict[str, str] = {
        "glden retreiver": "golden retriever",
        "goldn retriever": "golden retriever",
        "golden retreiver": "golden retriever",
        "glden retriever": "golden retriever",
        "golden retriver": "golden retriever",
        "bald egle": "bald eagle",
        "blad eagle": "bald eagle",
        "polr bear": "polar bear",
        "polar baer": "polar bear",
        "livng room": "living room",
        "dinng room": "dining room",
        "post offce": "post office",
        "high schol": "high school",
    }

    _instance: Optional[TypoNormalizer] = None

    def __init__(self):
        self._wn_lemma_cache: Optional[Set[str]] = None
        self._wn_bucket_cache: Optional[Dict[Tuple[str, int], List[str]]] = None

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

    def _get_wn_bucket_candidates(self, first_char: str, target_len: int) -> List[str]:
        if self._wn_bucket_cache is None:
            self._wn_bucket_cache = {}
            lemmas = self._get_all_wn_lemmas()
            for lem in lemmas:
                if lem and len(lem) <= 25 and "_" not in lem:
                    k = (lem[0].lower(), len(lem))
                    self._wn_bucket_cache.setdefault(k, []).append(lem)
        
        candidates = []
        for l in (target_len - 1, target_len, target_len + 1):
            if l > 1:
                candidates.extend(self._wn_bucket_cache.get((first_char, l), []))
        return candidates

    def correct_word(self, word: str, lang: str = "en", max_dist: int = 2) -> str:
        """Corrects a single token against the core lexicon and WordNet."""
        w_clean = word.lower().strip()
        if not w_clean or len(w_clean) <= 1:
            return word

        # Fast exact match if word is numeric or variable
        if w_clean.isdigit() or "_" in w_clean:
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

        # 3. Candidate search with Damerau-Levenshtein against core lexicon
        best_candidate: Optional[str] = None
        best_score = float("inf")

        for target, priority in lexicon.items():
            if " " in target:
                continue
            if abs(len(w_clean) - len(target)) > max_dist:
                continue

            dist = damerau_levenshtein(w_clean, target)
            if dist <= max_dist:
                score = (dist * 10) - priority
                if score < best_score:
                    best_score = score
                    best_candidate = target

        if best_candidate and best_score < 20:
            return best_candidate

        # 4. Fallback search over indexed WordNet lemma buckets (O(1) bucket lookup, max distance 1)
        if lang in ("en", "english") and len(w_clean) >= 3:
            first_char = w_clean[0]
            bucket = self._get_wn_bucket_candidates(first_char, len(w_clean))
            for wn_word in bucket:
                dist = damerau_levenshtein(w_clean, wn_word)
                if dist <= 1:
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
