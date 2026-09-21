"""WordNet lexical grounder for mapping lemmas and synsets to Band 2 Ontological Signatures."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


def find_quanta_data_file(filename: str) -> Optional[Path]:
    """Robust multi-path locator for Quanta offline data files (SQLite DBs, codebooks, JSONs).

    Searches:
    1. Direct Path('data') / filename (current working directory)
    2. Local repo root: (Path(__file__).resolve().parents[2] / 'data' / filename)
    3. Environment variable QUANTA_DATA_DIR / filename
    4. Git common directory parent / 'data' / filename (for git worktrees)
    5. Fallback well-known paths (e.g., C:/Users/PC/Documents/GitHub/Quanta/data/)

    If found outside local data/, attempts to create an NTFS hardlink in local data/ for zero-copy access.
    """
    repo_root = Path(__file__).resolve().parents[2]
    local_data_dir = repo_root / "data"
    local_target = local_data_dir / filename
    if local_target.exists():
        return local_target

    cwd_target = Path("data") / filename
    if cwd_target.exists():
        return cwd_target.resolve()

    # Search candidates
    candidates: List[Path] = []

    env_data = os.environ.get("QUANTA_DATA_DIR")
    if env_data:
        candidates.append(Path(env_data) / filename)

    # Git common dir (worktree parent)
    try:
        git_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if git_dir:
            common_root = Path(git_dir).resolve().parent
            candidates.append(common_root / "data" / filename)
    except Exception:
        pass

    # Well-known fallback paths on current system
    candidates.append(Path(r"C:\Users\PC\Documents\GitHub\Quanta\data") / filename)
    candidates.append(repo_root.parent / "Quanta" / "data" / filename)

    for cand in candidates:
        if cand.exists():
            # Try to create local hardlink for zero-copy high performance
            try:
                local_data_dir.mkdir(parents=True, exist_ok=True)
                if not local_target.exists():
                    os.link(str(cand), str(local_target))
                    return local_target
            except Exception:
                pass
            return cand

    return None

try:
    from nltk.corpus import wordnet as wn
    NLTK_WN_AVAILABLE = True
except ImportError:
    wn = None
    NLTK_WN_AVAILABLE = False

from core.slots import get_slot_by_name
from core.types import QuantaVector, QuaternaryValue
from core.asg import QuantaNode
from core.valency import validate_valency, TypeConstraintRegistry, ValencyConstraint


LEXNAME_TO_WN_ROOT_SLOT: Dict[str, str] = {
    "noun.act": "WN_ACT_ACTION",
    "noun.animal": "WN_ANIMAL_FAUNA",
    "noun.artifact": "WN_ARTIFACT_OBJECT",
    "noun.attribute": "WN_ATTRIBUTE_PROP",
    "noun.body": "WN_BODY_PART",
    "noun.cognition": "WN_COGNITION_THOUGHT",
    "noun.communication": "WN_COMMUNICATION_INFO",
    "noun.event": "WN_EVENT_OCCURRENCE",
    "noun.feeling": "WN_FEELING_EMOTION",
    "noun.food": "WN_FOOD_NUTRITION",
    "noun.group": "WN_GROUP_SOCIAL",
    "noun.location": "WN_LOCATION_PLACE",
    "noun.motive": "WN_MOTIVE_REASON",
    "noun.object": "WN_OBJECT_NATURAL",
    "noun.person": "WN_PERSON_HUMAN",
    "noun.phenomenon": "WN_PHENOMENON_NATURE",
    "noun.plant": "WN_PLANT_FLORA",
    "noun.possession": "WN_POSSESSION_ASSET",
    "noun.process": "WN_PROCESS_SERIES",
    "noun.quantity": "WN_QUANTITY_NUMBER",
    "noun.relation": "WN_RELATION_LINK",
}


@dataclass
class GroundedLexicalConcept:
    lemma: str
    synset_name: str
    definition: str
    pos: str
    hypernym_path: List[str]
    active_slots: Dict[str, int]
    vector: QuantaVector


class ConceptNetLexicalGrounder:
    """Resolves lexical entities and verbs into canonical QUANTA quaternary vectors using ConceptNet 256-D dimensions."""

    _default_instance: Optional[ConceptNetLexicalGrounder] = None

    def __init__(self, offline_cache_path: Optional[Union[str, Path]] = None):
        self._cache: Dict[str, GroundedLexicalConcept] = {}
        if offline_cache_path is None:
            default_db = find_quanta_data_file("conceptnet_offline.db")
            if default_db and default_db.exists():
                offline_cache_path = default_db
        self.offline_cache_path = Path(offline_cache_path) if offline_cache_path else None
        self._conn: Optional[sqlite3.Connection] = None
        if self.offline_cache_path and self.offline_cache_path.exists():
            self._conn = sqlite3.connect(str(self.offline_cache_path), check_same_thread=False)
        else:
            logger.warning("ConceptNet offline database not found. Lexical concept grounding will be impaired.")

    @classmethod
    def get_default(cls) -> ConceptNetLexicalGrounder:
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def resolve_concept(
        self,
        word: str,
        pos: Optional[str] = "n",
        lang: str = "en",
    ) -> Optional[GroundedLexicalConcept]:
        """Resolves a lemma/word into a GroundedLexicalConcept with 256-D ConceptNet quaternary vector."""
        w_clean = word.strip().lower()
        if not w_clean:
            return None

        # Clean anchor prefix if present
        if w_clean.startswith("cn:"):
            w_clean = w_clean[3:]
            if f":{lang}:" in f":{w_clean}":
                parts = w_clean.split(":")
                if len(parts) >= 2:
                    w_clean = parts[1]
            if " (" in w_clean and w_clean.endswith(")"):
                w_clean = w_clean.split(" (")[0]
        elif w_clean.startswith("wn:"):
            w_clean = w_clean[3:].split(".")[0]

        # 1. Punctuation Delimiters
        if pos in ("PUNCT", "punct") or w_clean in ('.', ',', ';', ':', '!', '?', '-', '--', '—', '"', "'", '`', '...', '(', ')', '[', ']', '{', '}', '/', '\\'):
            return GroundedLexicalConcept(
                lemma=w_clean,
                synset_name=f"punct:{w_clean}",
                definition=f"Punctuation delimiter '{w_clean}'",
                pos="punct",
                hypernym_path=[],
                active_slots={},
                vector=QuantaVector({}),
            )

        # 2. English Contractions & Abbreviations
        if w_clean == "'s":
            if pos in ("PART", "POS", "case", "pos") or (pos_norm == "n" and not w_clean.isalpha()):
                return GroundedLexicalConcept(
                    lemma="'s",
                    synset_name="gram:case:possessive",
                    definition="Possessive case marker 's",
                    pos="case",
                    hypernym_path=[],
                    active_slots={"NSM_HAVE": 1},
                    vector=QuantaVector({"NSM_HAVE": 1}),
                )
            be_concept = self.resolve_concept("be", pos="v", lang=lang)
            if be_concept:
                return GroundedLexicalConcept(
                    lemma="'s",
                    synset_name="cn:en:be (v)",
                    definition="Contraction of 'is' ('s)",
                    pos="v",
                    hypernym_path=be_concept.hypernym_path,
                    active_slots=dict(be_concept.active_slots, LJB_CA_PRESENT_TENSE=1),
                    vector=be_concept.vector.copy(),
                )
        elif w_clean in ("'re", "'m"):
            be_concept = self.resolve_concept("be", pos="v", lang=lang)
            if be_concept:
                return GroundedLexicalConcept(
                    lemma=w_clean,
                    synset_name="cn:en:be (v)",
                    definition=f"Contraction of 'be' ({w_clean})",
                    pos="v",
                    hypernym_path=be_concept.hypernym_path,
                    active_slots=dict(be_concept.active_slots, LJB_CA_PRESENT_TENSE=1),
                    vector=be_concept.vector.copy(),
                )
        elif w_clean == "'ve":
            have_concept = self.resolve_concept("have", pos="v", lang=lang)
            if have_concept:
                return GroundedLexicalConcept(
                    lemma="'ve",
                    synset_name="cn:en:have (v)",
                    definition="Contraction of 'have' ('ve)",
                    pos="v",
                    hypernym_path=have_concept.hypernym_path,
                    active_slots=dict(have_concept.active_slots, LJB_PU_PAST_TENSE=1),
                    vector=have_concept.vector.copy(),
                )
        elif w_clean == "'ll":
            will_concept = self.resolve_concept("will", pos="v", lang=lang)
            if will_concept:
                return GroundedLexicalConcept(
                    lemma="'ll",
                    synset_name="cn:en:will (v)",
                    definition="Contraction of 'will' ('ll)",
                    pos="v",
                    hypernym_path=will_concept.hypernym_path,
                    active_slots=dict(will_concept.active_slots, LJB_BA_FUTURE_TENSE=1),
                    vector=will_concept.vector.copy(),
                )
        elif w_clean == "'d":
            if pos in ("MD", "modal", "aux") or pos_norm == "v":
                would_concept = self.resolve_concept("would", pos="v", lang=lang) or self.resolve_concept("would", pos="n", lang=lang)
                if would_concept:
                    return GroundedLexicalConcept(
                        lemma="'d",
                        synset_name="cn:en:would (n)",
                        definition="Contraction of 'would' ('d)",
                        pos="v",
                        hypernym_path=would_concept.hypernym_path,
                        active_slots=dict(would_concept.active_slots, MODALITY_COUNTERFACTUAL=1),
                        vector=would_concept.vector.copy(),
                    )
            had_concept = self.resolve_concept("have", pos="v", lang=lang)
            if had_concept:
                return GroundedLexicalConcept(
                    lemma="'d",
                    synset_name="cn:en:have (v)",
                    definition="Contraction of 'had' ('d)",
                    pos="v",
                    hypernym_path=had_concept.hypernym_path,
                    active_slots=dict(had_concept.active_slots, LJB_PU_PAST_TENSE=1),
                    vector=had_concept.vector.copy(),
                )
        elif w_clean == "n't":
            return GroundedLexicalConcept(
                lemma="n't",
                synset_name="cn:en:not (n)",
                definition="Logical negation (n't)",
                pos="n",
                hypernym_path=[],
                active_slots={"LJB_NA_NEGATION": 2},
                vector=QuantaVector({"LJB_NA_NEGATION": 2}),
            )
        elif w_clean in ("dr.", "dr", "prof.", "prof", "mr.", "mr", "mrs.", "mrs", "ms.", "ms"):
            abbr_clean = w_clean.rstrip(".")
            return GroundedLexicalConcept(
                lemma=w_clean,
                synset_name=f"gram:title:{abbr_clean}",
                definition=f"Honorific title abbreviation '{w_clean}'",
                pos="n",
                hypernym_path=[],
                active_slots={"TYPE_HUMAN": 1, "ROLE_AGENT_CAPABLE": 1},
                vector=QuantaVector({"TYPE_HUMAN": 1, "ROLE_AGENT_CAPABLE": 1}),
            )

        lemma_under = w_clean.replace(" ", "_")
        lemma_space = w_clean.replace("_", " ")
        pos_norm = pos.lower()[0] if pos else "n"

        cache_key = f"cn:{lang}:{lemma_under} ({pos_norm})"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._conn:
            return None

        cur = self._conn.cursor()
        pos_keys = [
            f"cn:{lang}:{lemma_under} ({pos_norm})",
            f"cn:{lang}:{lemma_space} ({pos_norm})",
            f"cn:{lemma_under} ({pos_norm})",
            f"cn:{lemma_space} ({pos_norm})",
        ]
        
        placeholders_pos = ",".join("?" * len(pos_keys))
        cur.execute(
            f"SELECT key, lemma, pos, active_slots, packed_bytes_hex FROM concepts WHERE key IN ({placeholders_pos}) LIMIT 1",
            pos_keys
        )
        row = cur.fetchone()
        
        if not row:
            cur.execute(
                "SELECT key, lemma, pos, active_slots, packed_bytes_hex FROM concepts WHERE lemma IN (?, ?) AND pos=? LIMIT 1",
                (lemma_space, lemma_under, pos_norm)
            )
            row = cur.fetchone()

        if not row:
            unqualified_keys = [
                f"cn:{lang}:{lemma_under}",
                f"cn:{lang}:{lemma_space}",
                f"cn:{lemma_under}",
                f"cn:{lemma_space}",
                lemma_under,
                lemma_space,
            ]
            placeholders_unq = ",".join("?" * len(unqualified_keys))
            cur.execute(
                f"SELECT key, lemma, pos, active_slots, packed_bytes_hex FROM concepts WHERE key IN ({placeholders_unq}) LIMIT 1",
                unqualified_keys
            )
            row = cur.fetchone()

        if not row:
            cur.execute(
                "SELECT key, lemma, pos, active_slots, packed_bytes_hex FROM concepts WHERE lemma IN (?, ?) LIMIT 1",
                (lemma_space, lemma_under)
            )
            row = cur.fetchone()

        if not row:
            try:
                from parser.typo_normalizer import TypoNormalizer
                norm = TypoNormalizer.get_instance()
                corr = norm.normalize_text(w_clean, lang=lang).lower()
                corr_space = corr.replace("_", " ")
                corr_under = corr.replace(" ", "_")
                if corr_space != lemma_space:
                    cur.execute(
                        "SELECT key, lemma, pos, active_slots, packed_bytes_hex FROM concepts WHERE lemma IN (?, ?) LIMIT 1",
                        (corr_space, corr_under)
                    )
                    row = cur.fetchone()
            except Exception:
                pass

        if row:
            key, r_lemma, r_pos, active_slots_raw, packed_hex = row
            active_slots = json.loads(active_slots_raw) if isinstance(active_slots_raw, str) else active_slots_raw
            if packed_hex:
                vec = QuantaVector(bytes.fromhex(packed_hex))
            else:
                vec = QuantaVector(active_slots)

            is_human = (active_slots.get("TYPE_HUMAN") == 1 and active_slots.get("TYPE_ANIMATE") != 1) or active_slots.get("WN_PERSON_HUMAN") == 1 or (NLTK_WN_AVAILABLE and wn and len(wn.synsets(r_lemma, pos=wn.NOUN)) > 0 and wn.synsets(r_lemma, pos=wn.NOUN)[0].lexname() == "noun.person")
            if is_human:
                from core.slots import get_slot_by_name
                slot_art = get_slot_by_name("TYPE_ARTIFACT")
                slot_inan = get_slot_by_name("TYPE_INANIMATE_PHYSICAL")
                vec["TYPE_HUMAN"] = 1
                vec["TYPE_ANIMATE"] = 1
                vec["ROLE_AGENT_CAPABLE"] = 1
                vec["ROLE_SENTIENT"] = 1
                vec["TYPE_NATURAL_OBJECT"] = 1
                vec["TYPE_ARTIFACT"] = 0
                vec["TYPE_INANIMATE_PHYSICAL"] = 0
                if slot_art:
                    vec[slot_art.name] = 0
                    active_slots.pop(slot_art.name, None)
                if slot_inan:
                    vec[slot_inan.name] = 0
                    active_slots.pop(slot_inan.name, None)
                active_slots["TYPE_HUMAN"] = 1
                active_slots["TYPE_ANIMATE"] = 1
                active_slots["ROLE_AGENT_CAPABLE"] = 1
                active_slots["ROLE_SENTIENT"] = 1
                active_slots["TYPE_NATURAL_OBJECT"] = 1
                active_slots.pop("TYPE_ARTIFACT", None)
                active_slots.pop("TYPE_INANIMATE_PHYSICAL", None)

            if not active_slots or len(active_slots) == 0:
                # Fallback: attempt WordNet enrichment when ConceptNet 5.7 has empty slots
                try:
                    wn_grounder = WordNetLexicalGrounder.get_default()
                    if wn_grounder:
                        wn_syn = wn_grounder.resolve_synset(r_lemma, pos=r_pos)
                        if wn_syn:
                            wn_root_slot_idx = wn_grounder.get_wordnet_root_category(wn_syn)
                            if wn_root_slot_idx is not None:
                                from core.slots import SLOT_INDEX_TO_NAME
                                wn_slot_name = SLOT_INDEX_TO_NAME.get(wn_root_slot_idx)
                                if wn_slot_name:
                                    active_slots[wn_slot_name] = 1
                                    vec[wn_slot_name] = 1
                except Exception:
                    pass

            if r_pos == "v" or pos_norm == "v":
                if "WN_ACT_ACTION" not in active_slots:
                    active_slots["WN_ACT_ACTION"] = 1
                    vec["WN_ACT_ACTION"] = 1
                if "CN_Q115_EVENT" not in active_slots:
                    active_slots["CN_Q115_EVENT"] = 1
                    vec["CN_Q115_EVENT"] = 1
                if "CN_Q092_ACT" not in active_slots:
                    active_slots["CN_Q092_ACT"] = 1
                    vec["CN_Q092_ACT"] = 1

            canonical_anchor = f"cn:{lang}:{r_lemma} ({r_pos})"
            concept = GroundedLexicalConcept(
                lemma=r_lemma,
                synset_name=canonical_anchor,
                definition=f"ConceptNet grounded entity {r_lemma} ({r_pos})",
                pos=r_pos,
                hypernym_path=[],
                active_slots=active_slots,
                vector=vec,
            )
            self._cache[cache_key] = concept
            self._cache[key] = concept
            return concept

    def infer_from_grammar(
        self,
        lemma: str,
        pos: str = "n",
        tag: Optional[str] = None,
        morph: Optional[Any] = None,
        ent_type: Optional[str] = None,
        text: Optional[str] = None,
    ) -> GroundedLexicalConcept:
        """Constructs a fully-formed GroundedLexicalConcept using POS and grammatical morphology when ConceptNet lookup misses."""
        w_clean = lemma.strip().lower()
        pos_upper = (pos or "NOUN").upper()
        tag_upper = (tag or "").upper()
        morph_str = str(morph or "")
        active_slots: Dict[str, int] = {}

        if pos_upper in ("PUNCT", "PUNCTUATION") or w_clean in ('.', ',', ';', ':', '!', '?', '-', '--', '—', '"', "'", '`', '...', '(', ')', '[', ']', '{', '}', '/', '\\'):
            return GroundedLexicalConcept(
                lemma=w_clean,
                synset_name=f"punct:{w_clean}",
                definition=f"Punctuation delimiter '{w_clean}'",
                pos="punct",
                hypernym_path=[],
                active_slots={},
                vector=QuantaVector({}),
            )

        if pos_upper in ("VERB", "AUX", "V"):
            pos_norm = "v"
            anchor = f"cn:en:{w_clean} (v)"
            active_slots["TYPE_EVENT"] = 1
            active_slots["WN_ACT_ACTION"] = 1
            if tag_upper in ("VBD", "VBN") or "Tense=Past" in morph_str or w_clean in ("bit", "chased", "saw", "ran", "gave", "thought", "felt", "wanted", "pretended", "isolated", "verified", "retained", "prompted", "noted"):
                active_slots["LJB_PU_PAST_TENSE"] = 1
            elif tag_upper in ("VBP", "VBZ") or "Tense=Pres" in morph_str:
                active_slots["LJB_CA_PRESENT_TENSE"] = 1
            if tag_upper == "MD" or w_clean in ("must", "can", "could", "may", "might", "will", "would", "shall", "should"):
                if w_clean == "must":
                    active_slots["EPIST_DEONTIC_OBLIGATION"] = 1
                elif w_clean in ("can", "could"):
                    active_slots["EPIST_DEONTIC_PERMISSION"] = 1
                    active_slots["ROLE_AGENT_CAPABLE"] = 1
                elif w_clean in ("may", "might"):
                    active_slots["NSM_MAYBE"] = 3
                    active_slots["MODALITY_HYPOTHETICAL"] = 3
                elif w_clean in ("will", "shall"):
                    active_slots["LJB_BA_FUTURE_TENSE"] = 1
                elif w_clean == "would":
                    active_slots["MODALITY_HYPOTHETICAL"] = 3
        elif pos_upper in ("NOUN", "PROPN", "N"):
            pos_norm = "n"
            anchor = f"cn:en:{w_clean} (n)"
            if ent_type == "PERSON" or (text and text.istitle() and text.lower() not in ("the", "a", "an", "this", "that")):
                active_slots["TYPE_HUMAN"] = 1
                active_slots["TYPE_ANIMATE"] = 1
                active_slots["ROLE_AGENT_CAPABLE"] = 1
                active_slots["ROLE_SENTIENT"] = 1
            elif ent_type in ("GPE", "LOC", "FAC") or w_clean in ("airspace", "area", "garden", "house", "cell", "vessel", "room", "city"):
                active_slots["TYPE_SPATIAL_REGION"] = 1
                active_slots["WN_LOCATION_PLACE"] = 1
            elif ent_type == "ORG" or w_clean in ("council", "board", "commission", "director"):
                active_slots["TYPE_ORGANIZATION"] = 1
                active_slots["ROLE_AGENT_CAPABLE"] = 1
            elif ent_type in ("DATE", "TIME") or w_clean in ("dusk", "dawn", "afternoon", "hour", "hours", "minute", "day", "morning", "night"):
                active_slots["TYPE_TEMPORAL_INTERVAL"] = 1
            else:
                active_slots["TYPE_INANIMATE_PHYSICAL"] = 1
        elif pos_upper in ("ADJ", "A", "J"):
            pos_norm = "a"
            anchor = f"cn:en:{w_clean} (a)"
            active_slots["TYPE_ATTRIBUTE_PROPERTY"] = 1
            active_slots["WN_ATTRIBUTE_PROP"] = 1
            if tag_upper == "JJR" or "Degree=Cmp" in morph_str:
                active_slots["NSM_MORE"] = 1
            elif tag_upper == "JJS" or "Degree=Sup" in morph_str:
                active_slots["NSM_VERY"] = 1
        elif pos_upper in ("ADV", "R"):
            pos_norm = "r"
            anchor = f"cn:en:{w_clean} (r)"
            active_slots["VAL_MANNER_SLOT"] = 1
            if w_clean in ("rapidly", "fast", "quickly", "accelerating"):
                active_slots["NSM_ACCELERATING_RATE"] = 1
            elif w_clean in ("later", "earlier", "before", "soon", "immediately", "initially"):
                active_slots["VAL_TIME_SLOT"] = 1
        elif pos_upper in ("DET", "D"):
            pos_norm = "d"
            anchor = f"gram:det:{w_clean}"
            if w_clean in ("a", "an", "one", "1"):
                active_slots["NSM_ONE"] = 1
                active_slots["LJB_SUO_AT_LEAST_ONE"] = 1
            elif w_clean in ("the", "this", "that", "these", "those"):
                active_slots["NSM_THIS"] = 1
            elif w_clean in ("all", "every", "each"):
                active_slots["NSM_ALL"] = 1
                active_slots["LJB_RO_ALL_QUANT"] = 1
            elif w_clean in ("no", "neither"):
                active_slots["LJB_NO_NONE_QUANT"] = 1
                active_slots["LJB_NA_NEGATION"] = 2
        elif pos_upper in ("PRON", "P"):
            pos_norm = "p"
            anchor = f"gram:pron:{w_clean}"
            active_slots["GRAPH_ANAPHORA_TARGET"] = 1
            active_slots["GRAPH_COREF_BUNDLE"] = 1
            if w_clean in ("he", "she", "him", "her", "his", "they", "them", "their", "someone", "who"):
                active_slots["TYPE_HUMAN"] = 1
                active_slots["TYPE_ANIMATE"] = 1
                active_slots["ROLE_AGENT_CAPABLE"] = 1
            else:
                active_slots["TYPE_INANIMATE_PHYSICAL"] = 1
        elif pos_upper in ("ADP", "PREP"):
            pos_norm = "prep"
            anchor = f"gram:prep:{w_clean}"
            if w_clean in ("in", "inside", "within", "on", "into"):
                active_slots["SPATIAL_RCC_NON_TANG_PART"] = 1
                active_slots["NSM_INSIDE"] = 1
            elif w_clean in ("before", "prior"):
                active_slots["TEMP_ALLEN_BEFORE"] = 1
            elif w_clean in ("during", "throughout", "while"):
                active_slots["TEMP_ALLEN_DURING"] = 1
            elif w_clean in ("until", "till"):
                active_slots["LOGIC_TEMPORAL_UNTIL_U"] = 1
        elif pos_upper in ("PART", "NEG"):
            if w_clean in ("not", "n't", "never", "no"):
                pos_norm = "neg"
                anchor = "gram:neg:not"
                active_slots["LJB_NA_NEGATION"] = 2
            else:
                pos_norm = "part"
                anchor = f"gram:part:{w_clean}"
        elif pos_upper in ("CCONJ", "SCONJ", "C", "CONJ"):
            pos_norm = "conj"
            anchor = f"gram:conj:{w_clean}"
            if w_clean == "and":
                active_slots["LJB_JE_AND"] = 1
            elif w_clean == "or":
                active_slots["LJB_JA_OR"] = 1
            elif w_clean in ("if", "unless"):
                active_slots["LJB_GANAI_IF_THEN"] = 1
            elif w_clean in ("because", "since"):
                active_slots["CAUSAL_DIRECT_MECHANISM"] = 1
        else:
            pos_norm = "n"
            anchor = f"cn:en:{w_clean} (n)"
            active_slots["TYPE_INANIMATE_PHYSICAL"] = 1

        vec = QuantaVector(active_slots)
        return GroundedLexicalConcept(
            lemma=w_clean,
            synset_name=anchor,
            definition=f"Grammatically inferred concept {w_clean} ({pos_norm})",
            pos=pos_norm,
            hypernym_path=[],
            active_slots=active_slots,
            vector=vec,
        )

    def resolve_synset(self, word: str, pos: Optional[str] = None) -> Optional[str]:
        """Resolves word to a canonical ConceptNet anchor string (e.g. 'cn:en:dog (n)')."""
        concept = self.resolve_concept(word, pos=pos)
        if concept:
            return concept.synset_name
        return f"cn:en:{word.strip().lower()} ({pos or 'n'})"

    def ground_synset(self, synset_or_name: Any, pos: str = "n") -> GroundedLexicalConcept:
        """Grounds a synset name or concept key with grammatical fallback."""
        name = str(synset_or_name)
        concept = self.resolve_concept(name, pos=pos)
        if concept:
            return concept
        return self.infer_from_grammar(name, pos=pos)


# Canonical grounder alias
LexicalGrounder = ConceptNetLexicalGrounder


class WordNetLexicalGrounder:
    """Resolves lexical entities into canonical QUANTA ontological vectors using WordNet hypernym paths."""

    _default_instance: Optional[WordNetLexicalGrounder] = None

    def __init__(self, offline_cache_path: Optional[Union[str, Path]] = None):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.offline_cache_path = Path(offline_cache_path) if offline_cache_path else None

        if self.offline_cache_path and self.offline_cache_path.exists():
            self.load_cache(self.offline_cache_path)

    @classmethod
    def get_default(cls) -> WordNetLexicalGrounder:
        if cls._default_instance is None:
            default_db = find_quanta_data_file("wordnet_offline.db")
            cls._default_instance = cls(offline_cache_path=default_db if default_db and default_db.exists() else None)
        return cls._default_instance

    def load_cache(self, path: Union[str, Path]):
        """Loads offline precomputed grounding cache from JSON or SQLite database."""
        p = Path(path)
        if p.suffix in (".db", ".sqlite"):
            conn = sqlite3.connect(str(p))
            cur = conn.cursor()
            cur.execute("SELECT key, lemma, synset_name, definition, pos, hypernym_path, active_slots FROM synsets")
            for row in cur.fetchall():
                key, lemma, synset_name, definition, pos, hyp_path, active_slots = row
                self._cache[key] = {
                    "lemma": lemma,
                    "synset_name": synset_name,
                    "definition": definition,
                    "pos": pos,
                    "hypernym_path": json.loads(hyp_path) if isinstance(hyp_path, str) else hyp_path,
                    "active_slots": json.loads(active_slots) if isinstance(active_slots, str) else active_slots,
                }
            conn.close()
        else:
            with open(p, "r", encoding="utf-8") as f:
                self._cache = json.load(f)

    def save_cache(self, path: Union[str, Path]):
        """Saves current grounding cache to JSON or SQLite database for O(1) offline deployment."""
        p = Path(path)
        if p.suffix in (".db", ".sqlite"):
            conn = sqlite3.connect(str(p))
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS synsets (
                    key TEXT PRIMARY KEY,
                    lemma TEXT,
                    synset_name TEXT,
                    definition TEXT,
                    pos TEXT,
                    hypernym_path TEXT,
                    active_slots TEXT,
                    packed_bytes_hex TEXT
                )
            """)
            for key, entry in self._cache.items():
                slots = entry["active_slots"]
                vec = QuantaVector(slots)
                cur.execute("""
                    INSERT OR REPLACE INTO synsets (key, lemma, synset_name, definition, pos, hypernym_path, active_slots, packed_bytes_hex)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key,
                    entry["lemma"],
                    entry["synset_name"],
                    entry["definition"],
                    entry["pos"],
                    json.dumps(entry["hypernym_path"]),
                    json.dumps(slots),
                    vec.to_bytes().hex(),
                ))
            conn.commit()
            conn.close()
        else:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)

    def resolve_synset(self, word: str, pos: Optional[str] = None) -> Optional[str]:
        """Resolves a word and optional POS to a canonical WordNet synset ID (e.g. 'wn:dog.n.01')."""
        w_clean = word.strip()
        if w_clean.startswith("wn:"):
            return w_clean
        if "." in w_clean and len(w_clean.split(".")) == 3:
            return f"wn:{w_clean}"

        key = w_clean.lower().replace(" ", "_")
        if key in self._cache:
            return self._cache[key]["synset_name"]

        if not NLTK_WN_AVAILABLE:
            return None

        try:
            s = wn.synset(key)
            return f"wn:{s.name()}"
        except Exception:
            pass

        wn_pos = None
        if pos:
            pos_map = {
                "n": wn.NOUN, "noun": wn.NOUN,
                "v": wn.VERB, "verb": wn.VERB,
                "a": wn.ADJ, "adj": wn.ADJ, "s": wn.ADJ_SAT,
                "r": wn.ADV, "adv": wn.ADV,
            }
            wn_pos = pos_map.get(pos.lower(), pos)

        synsets = wn.synsets(key, pos=wn_pos) if wn_pos else wn.synsets(key)
        if synsets:
            return f"wn:{synsets[0].name()}"

        try:
            from parser.typo_normalizer import TypoNormalizer
            norm = TypoNormalizer.get_instance()
            corr = norm.normalize_text(w_clean, lang="en")
            corr_key = corr.lower().replace(" ", "_")
            if corr_key in self._cache:
                return self._cache[corr_key]["synset_name"]
            synsets = wn.synsets(corr_key, pos=wn_pos) if wn_pos else wn.synsets(corr_key)
            if synsets:
                return f"wn:{synsets[0].name()}"

            corr_w = norm.correct_word(w_clean, lang="en")
            corr_w_key = corr_w.lower().replace(" ", "_")
            if corr_w_key in self._cache:
                return self._cache[corr_w_key]["synset_name"]
            synsets = wn.synsets(corr_w_key, pos=wn_pos) if wn_pos else wn.synsets(corr_w_key)
            if synsets:
                return f"wn:{synsets[0].name()}"
        except Exception:
            pass

        return None

    def get_hypernym_path(self, synset_id: str) -> List[str]:
        """Returns the recursive hypernym path (ancestor synset IDs) for a synset."""
        key = synset_id[3:] if synset_id.startswith("wn:") else synset_id
        if key in self._cache and "hypernym_path" in self._cache[key]:
            return list(self._cache[key]["hypernym_path"])

        if not NLTK_WN_AVAILABLE:
            return []

        try:
            synset = wn.synset(key)
        except Exception:
            return []

        all_hypernyms: Set[str] = set()
        hypernym_path: List[str] = []

        def traverse(s):
            for h in s.hypernyms() + s.instance_hypernyms():
                name = h.name()
                if name not in all_hypernyms:
                    all_hypernyms.add(name)
                    hypernym_path.append(name)
                    traverse(h)

        traverse(synset)
        return hypernym_path

    def get_wordnet_root_category(self, synset_id: str) -> Optional[int]:
        """Maps a WordNet synset to its canonical Band 2 root category slot index (171-191)."""
        key = synset_id[3:] if synset_id.startswith("wn:") else synset_id

        if key in self._cache:
            for slot_name in LEXNAME_TO_WN_ROOT_SLOT.values():
                if self._cache[key].get("active_slots", {}).get(slot_name) == 1:
                    return get_slot_by_name(slot_name).index

        if not NLTK_WN_AVAILABLE:
            return None

        try:
            synset = wn.synset(key)
            lexname = synset.lexname()
            if lexname in LEXNAME_TO_WN_ROOT_SLOT:
                slot_name = LEXNAME_TO_WN_ROOT_SLOT[lexname]
                return get_slot_by_name(slot_name).index

            hyp_path = self.get_hypernym_path(key)
            for h_name in hyp_path:
                try:
                    h_syn = wn.synset(h_name)
                    h_lex = h_syn.lexname()
                    if h_lex in LEXNAME_TO_WN_ROOT_SLOT:
                        slot_name = LEXNAME_TO_WN_ROOT_SLOT[h_lex]
                        return get_slot_by_name(slot_name).index
                except Exception:
                    continue
        except Exception:
            pass

        return None

    def ground_synset(self, synset_or_name: Any) -> GroundedLexicalConcept:
        """Grounds a WordNet synset (e.g. 'dog.n.01' or wn.synset('dog.n.01')) to QUANTA slots."""
        if isinstance(synset_or_name, str):
            synset_name = synset_or_name
            if not synset_name.startswith("wn:"):
                key = synset_name
            else:
                key = synset_name[3:]
        else:
            key = synset_or_name.name()

        # Check offline cache first
        if key in self._cache:
            entry = self._cache[key]
            slots = entry["active_slots"]
            vec = QuantaVector(slots)
            s_name = entry.get("synset_name") or key
            if not s_name.startswith("wn:"):
                s_name = f"wn:{s_name}"
            return GroundedLexicalConcept(
                lemma=entry["lemma"],
                synset_name=s_name,
                definition=entry["definition"],
                pos=entry["pos"],
                hypernym_path=entry["hypernym_path"],
                active_slots=slots,
                vector=vec,
            )

        if not NLTK_WN_AVAILABLE:
            raise RuntimeError("WordNet is not available and synset is not in offline cache")

        # Resolve via nltk wordnet
        try:
            synset = wn.synset(key)
        except Exception:
            # Fallback: try finding synsets for lemma
            synsets = wn.synsets(key)
            if not synsets:
                # Try fuzzy typo correction
                from parser.typo_normalizer import TypoNormalizer
                norm = TypoNormalizer.get_instance()
                corr = norm.normalize_text(key.replace("_", " "), lang="en")
                corr_key = corr.lower().replace(" ", "_")
                synsets = wn.synsets(corr_key)
                if not synsets:
                    corr_w = norm.correct_word(key.replace("_", " "), lang="en")
                    corr_key = corr_w.lower().replace(" ", "_")
                    synsets = wn.synsets(corr_key)
                    if not synsets:
                        raise KeyError(f"WordNet synset not found for '{key}'")
                key = corr_key
            synset = synsets[0]
            key = synset.name()

        lemma_name = synset.lemmas()[0].name()
        definition = synset.definition()
        pos = synset.pos()

        # Extract all hypernyms recursively
        all_hypernyms: Set[str] = set()
        hypernym_path: List[str] = []

        def traverse(s):
            for h in s.hypernyms() + s.instance_hypernyms():
                name = h.name()
                if name not in all_hypernyms:
                    all_hypernyms.add(name)
                    hypernym_path.append(name)
                    traverse(h)

        traverse(synset)

        # Map hypernyms and lexnames to Band 2 slots
        active_slots: Dict[str, int] = {}
        lexname = synset.lexname()  # e.g., 'noun.animal', 'noun.person', 'verb.motion'

        # Lexname mapping
        lexname_to_wn_root = {
            "noun.act": "WN_ACT_ACTION",
            "noun.animal": "WN_ANIMAL_FAUNA",
            "noun.artifact": "WN_ARTIFACT_OBJECT",
            "noun.attribute": "WN_ATTRIBUTE_PROP",
            "noun.body": "WN_BODY_PART",
            "noun.cognition": "WN_COGNITION_THOUGHT",
            "noun.communication": "WN_COMMUNICATION_INFO",
            "noun.event": "WN_EVENT_OCCURRENCE",
            "noun.feeling": "WN_FEELING_EMOTION",
            "noun.food": "WN_FOOD_NUTRITION",
            "noun.group": "WN_GROUP_SOCIAL",
            "noun.location": "WN_LOCATION_PLACE",
            "noun.motive": "WN_MOTIVE_REASON",
            "noun.object": "WN_OBJECT_NATURAL",
            "noun.person": "WN_PERSON_HUMAN",
            "noun.phenomenon": "WN_PHENOMENON_NATURE",
            "noun.plant": "WN_PLANT_FLORA",
            "noun.possession": "WN_POSSESSION_ASSET",
            "noun.process": "WN_PROCESS_SERIES",
            "noun.quantity": "WN_QUANTITY_NUMBER",
            "noun.relation": "WN_RELATION_LINK",
        }

        if lexname in lexname_to_wn_root:
            active_slots[lexname_to_wn_root[lexname]] = 1

        # Ontological taxonomy checks
        hyp_str = " ".join(all_hypernyms)

        if "person.n.01" in all_hypernyms or "human.n.01" in all_hypernyms or lexname == "noun.person":
            active_slots["TYPE_HUMAN"] = 1
            active_slots["TYPE_ANIMATE"] = 1
            active_slots["ROLE_AGENT_CAPABLE"] = 1
            active_slots["ROLE_SENTIENT"] = 1
            active_slots["ROLE_COMMUNICATOR"] = 1
            active_slots["ROLE_MOVEABLE"] = 1
        elif "animal.n.01" in all_hypernyms or "organism.n.01" in all_hypernyms or lexname == "noun.animal":
            active_slots["TYPE_ANIMATE"] = 1
            active_slots["ROLE_AGENT_CAPABLE"] = 1
            active_slots["ROLE_SENTIENT"] = 1
            active_slots["ROLE_MOVEABLE"] = 1
        elif "artifact.n.01" in all_hypernyms or "instrument.n.01" in all_hypernyms or lexname == "noun.artifact":
            active_slots["TYPE_ARTIFACT"] = 1
            active_slots["TYPE_INANIMATE_PHYSICAL"] = 1
            active_slots["ROLE_INSTRUMENT_USABLE"] = 1
            if "container.n.01" in all_hypernyms:
                active_slots["ROLE_CONTAINER"] = 1
        elif "food.n.01" in all_hypernyms or "beverage.n.01" in all_hypernyms or lexname == "noun.food":
            active_slots["TYPE_SUBSTANCE_MASS"] = 1
            active_slots["ROLE_CONSUMABLE"] = 1
        elif "group.n.01" in all_hypernyms or "organization.n.01" in all_hypernyms or lexname == "noun.group":
            active_slots["TYPE_ORGANIZATION"] = 1
            active_slots["ROLE_AGENT_CAPABLE"] = 1
        elif "event.n.01" in all_hypernyms or "act.n.02" in all_hypernyms or pos == "v":
            active_slots["TYPE_EVENT"] = 1
        elif "state.n.02" in all_hypernyms or "condition.n.01" in all_hypernyms:
            active_slots["TYPE_STATE"] = 1
        elif "process.n.06" in all_hypernyms:
            active_slots["TYPE_PROCESS"] = 1
        elif "abstraction.n.06" in all_hypernyms or "concept.n.01" in all_hypernyms or lexname in ("noun.cognition", "noun.attribute"):
            active_slots["TYPE_ABSTRACT_CONCEPT"] = 1

        # Default literal modality
        active_slots["MODALITY_LITERAL"] = 1

        vec = QuantaVector(active_slots)

        # Cache result
        cached_entry = {
            "lemma": lemma_name,
            "synset_name": f"wn:{key}",
            "definition": definition,
            "pos": pos,
            "hypernym_path": hypernym_path[:10],
            "active_slots": active_slots,
        }
        self._cache[key] = cached_entry

        return GroundedLexicalConcept(
            lemma=lemma_name,
            synset_name=f"wn:{key}",
            definition=definition,
            pos=pos,
            hypernym_path=hypernym_path,
            active_slots=active_slots,
            vector=vec,
        )

    def create_node_for_concept(self, lemma_or_synset: str) -> QuantaNode:
        """Creates a QuantaNode with lexical anchor and grounded vector."""
        concept = self.ground_synset(lemma_or_synset)
        node = QuantaNode(
            vector=concept.vector,
            anchor=concept.synset_name,
        )
        return node


# Re-export validate_valency from core.valency for backward compatibility
# validate_valency = validate_valency



def resolve_synset(word: str, pos: Optional[str] = None, grounder: Optional[WordNetLexicalGrounder] = None) -> Optional[str]:
    """Module-level helper to resolve a word/lemma to a canonical WordNet synset ID."""
    g = grounder or WordNetLexicalGrounder.get_default()
    return g.resolve_synset(word, pos=pos)


def get_hypernym_path(synset_id: str, grounder: Optional[WordNetLexicalGrounder] = None) -> List[str]:
    """Module-level helper to retrieve the hypernym path for a synset."""
    g = grounder or WordNetLexicalGrounder.get_default()
    return g.get_hypernym_path(synset_id)


def get_wordnet_root_category(synset_id: str, grounder: Optional[WordNetLexicalGrounder] = None) -> Optional[int]:
    """Module-level helper to map a synset to its Band 2 WordNet root category slot index."""
    g = grounder or WordNetLexicalGrounder.get_default()
    return g.get_wordnet_root_category(synset_id)


@dataclass
class FrameNetTemplate:
    frame_name: str
    frame_id: str
    core_elements: List[str]
    slot_mapping: Dict[str, str]
    semantic_types: Dict[str, List[str]]
    lemmas: List[str] = field(default_factory=list)


class FrameNetValencyResolver:
    """Resolves verbs into FrameNet semantic frames and maps frame roles to Band 1 valency slots."""

    _instance: Optional[FrameNetValencyResolver] = None

    VERB_LEMMA_TO_FRAME: Dict[str, str] = {
        # Ingestion
        "bite": "Ingestion", "eat": "Ingestion", "drink": "Ingestion",
        "chew": "Ingestion", "swallow": "Ingestion", "taste": "Ingestion", "consume": "Ingestion",
        # Motion
        "move": "Motion", "run": "Motion", "walk": "Motion", "go": "Motion",
        "travel": "Motion", "fly": "Motion", "jump": "Motion", "chase": "Motion",
        "pursue": "Motion", "flee": "Motion", "arrive": "Motion", "leave": "Motion",
        "step": "Motion", "swim": "Motion", "climb": "Motion", "fall": "Motion",
        "enter": "Motion", "exit": "Motion", "cross": "Motion", "return": "Motion",
        # Statement
        "say": "Statement", "tell": "Statement", "speak": "Statement", "talk": "Statement",
        "claim": "Statement", "declare": "Statement", "state": "Statement", "report": "Statement",
        "announce": "Statement", "ask": "Statement", "answer": "Statement", "reply": "Statement",
        "explain": "Statement", "describe": "Statement", "whisper": "Statement", "shout": "Statement",
        # Perception_active
        "see": "Perception_active", "look": "Perception_active", "watch": "Perception_active",
        "observe": "Perception_active", "hear": "Perception_active", "listen": "Perception_active",
        "notice": "Perception_active", "smell": "Perception_active", "inspect": "Perception_active",
        # Causation
        "cause": "Causation", "make": "Causation", "force": "Causation", "trigger": "Causation",
        "produce": "Causation", "prevent": "Causation",
        # Giving
        "give": "Giving", "gift": "Giving", "offer": "Giving", "donate": "Giving", "provide": "Giving",
        "hand": "Giving", "pass": "Giving", "grant": "Giving",
        # Taking
        "take": "Taking", "grab": "Taking", "seize": "Taking", "acquire": "Taking",
        "catch": "Taking", "pick_up": "Taking",
        # Impact / Contact
        "touch": "Impact", "hit": "Impact", "strike": "Impact", "press": "Impact", "rub": "Impact",
        "slap": "Impact", "push": "Impact", "pull": "Impact",
        # Cognition / Mental
        "think": "Cogitation", "know": "Awareness", "want": "Desiring", "feel": "Feeling",
        "understand": "Cogitation", "believe": "Cogitation", "suppose": "Cogitation",
        "imagine": "Cogitation", "remember": "Cogitation", "forget": "Cogitation",
        "learn": "Cogitation", "study": "Cogitation", "need": "Desiring", "wish": "Desiring",
        "hope": "Desiring",
        # Placing / Putting
        "put": "Placing", "place": "Placing", "set": "Placing", "lay": "Placing",
        "drop": "Placing", "install": "Placing", "position": "Placing",
        # Removing
        "remove": "Removing", "extract": "Removing", "clear": "Removing",
        # Manipulation
        "hold": "Manipulation", "grasp": "Manipulation", "grip": "Manipulation", "carry": "Manipulation",
        # Commerce
        "buy": "Commerce_buy", "purchase": "Commerce_buy", "sell": "Commerce_buy", "pay": "Commerce_buy",
        # Social Interaction
        "meet": "Social_interaction", "visit": "Social_interaction", "greet": "Social_interaction",
        "hug": "Social_interaction", "kiss": "Social_interaction",
        # Seeking
        "search": "Seeking", "seek": "Seeking", "hunt": "Seeking", "find": "Seeking",
        # Assistance
        "help": "Assistance", "assist": "Assistance", "support": "Assistance", "aid": "Assistance",
        # Intentionally Affect / Destroy / Repair
        "kill": "Intentionally_affect", "destroy": "Intentionally_affect", "damage": "Intentionally_affect",
        "break": "Intentionally_affect", "repair": "Intentionally_affect", "fix": "Intentionally_affect",
        # Creating
        "create": "Creating", "build": "Creating", "construct": "Creating", "write": "Creating",
        "design": "Creating", "generate": "Creating",
    }

    BUILTIN_FRAMES: Dict[str, Dict[str, Any]] = {
        "Motion": {
            "frame_id": "frame:Motion.01",
            "core_elements": ["Theme", "Source", "Goal", "Path", "Carrier", "Manner", "Distance"],
            "slot_mapping": {
                "Theme": "VAL_X1_AGENT",
                "Source": "VAL_X4_SOURCE",
                "Goal": "VAL_X3_DESTINATION",
                "Manner": "VAL_MANNER_SLOT",
            },
            "semantic_types": {
                "Theme": ["TYPE_ANIMATE", "TYPE_INANIMATE_PHYSICAL"],
                "Source": ["TYPE_SPATIAL_REGION", "TYPE_INANIMATE_PHYSICAL"],
                "Goal": ["TYPE_SPATIAL_REGION", "TYPE_INANIMATE_PHYSICAL"],
            },
        },
        "Statement": {
            "frame_id": "frame:Statement.01",
            "core_elements": ["Speaker", "Message", "Topic", "Addressee", "Medium"],
            "slot_mapping": {
                "Speaker": "VAL_X1_AGENT",
                "Message": "VAL_X2_PATIENT",
                "Addressee": "VAL_EXPERIENCER",
            },
            "semantic_types": {
                "Speaker": ["TYPE_HUMAN", "TYPE_ORGANIZATION", "ROLE_COMMUNICATOR"],
                "Message": ["TYPE_COMMUNICATION_MSG", "TYPE_PROPOSITION"],
                "Addressee": ["TYPE_HUMAN", "ROLE_SENTIENT"],
            },
        },
        "Perception_active": {
            "frame_id": "frame:Perception_active.01",
            "core_elements": ["Perceiver_agentive", "Phenomenon", "Body_part", "Direction"],
            "slot_mapping": {
                "Perceiver_agentive": "VAL_X1_AGENT",
                "Phenomenon": "VAL_X2_PATIENT",
                "Body_part": "VAL_X5_INSTRUMENT",
            },
            "semantic_types": {
                "Perceiver_agentive": ["TYPE_HUMAN", "TYPE_ANIMATE", "ROLE_SENTIENT"],
                "Phenomenon": ["TYPE_EVENT", "TYPE_INANIMATE_PHYSICAL", "TYPE_ANIMATE"],
            },
        },
        "Ingestion": {
            "frame_id": "frame:Ingestion.01",
            "core_elements": ["Ingestor", "Ingestibles", "Manner", "Instrument"],
            "slot_mapping": {
                "Ingestor": "VAL_X1_AGENT",
                "Ingestibles": "VAL_X2_PATIENT",
                "Instrument": "VAL_X5_INSTRUMENT",
            },
            "semantic_types": {
                "Ingestor": ["TYPE_ANIMATE", "ROLE_AGENT_CAPABLE"],
                "Ingestibles": ["TYPE_SUBSTANCE_MASS", "ROLE_CONSUMABLE"],
                "Instrument": ["ROLE_INSTRUMENT_USABLE"],
            },
        },
        "Causation": {
            "frame_id": "frame:Causation.01",
            "core_elements": ["Cause", "Effect", "Actor", "Affected"],
            "slot_mapping": {
                "Cause": "VAL_X1_AGENT",
                "Effect": "VAL_RESULT_SLOT",
                "Affected": "VAL_X2_PATIENT",
            },
            "semantic_types": {
                "Cause": ["TYPE_EVENT", "TYPE_PROCESS", "ROLE_AGENT_CAPABLE"],
                "Effect": ["TYPE_EVENT", "TYPE_STATE"],
            },
        },
        "Giving": {
            "frame_id": "frame:Giving.01",
            "core_elements": ["Donor", "Theme", "Recipient"],
            "slot_mapping": {
                "Donor": "VAL_X1_AGENT",
                "Theme": "VAL_X2_PATIENT",
                "Recipient": "VAL_EXPERIENCER",
            },
            "semantic_types": {
                "Donor": ["TYPE_HUMAN", "TYPE_ORGANIZATION", "ROLE_AGENT_CAPABLE"],
                "Theme": ["TYPE_INANIMATE_PHYSICAL", "TYPE_ARTIFACT"],
                "Recipient": ["TYPE_HUMAN", "ROLE_SENTIENT"],
            },
        },
        "Taking": {
            "frame_id": "frame:Taking.01",
            "core_elements": ["Agent", "Theme", "Source"],
            "slot_mapping": {
                "Agent": "VAL_X1_AGENT",
                "Theme": "VAL_X2_PATIENT",
                "Source": "VAL_X4_SOURCE",
            },
            "semantic_types": {
                "Agent": ["TYPE_ANIMATE", "ROLE_AGENT_CAPABLE"],
                "Theme": ["TYPE_INANIMATE_PHYSICAL", "TYPE_ARTIFACT"],
                "Source": ["TYPE_HUMAN", "TYPE_SPATIAL_REGION"],
            },
        },
        "Impact": {
            "frame_id": "frame:Impact.01",
            "core_elements": ["Impactor", "Impactee", "Instrument"],
            "slot_mapping": {
                "Impactor": "VAL_X1_AGENT",
                "Impactee": "VAL_X2_PATIENT",
                "Instrument": "VAL_X5_INSTRUMENT",
            },
            "semantic_types": {
                "Impactor": ["TYPE_ANIMATE", "TYPE_INANIMATE_PHYSICAL"],
                "Impactee": ["TYPE_INANIMATE_PHYSICAL", "TYPE_ANIMATE"],
                "Instrument": ["ROLE_INSTRUMENT_USABLE"],
            },
        },
        "Cogitation": {
            "frame_id": "frame:Cogitation.01",
            "core_elements": ["Cognizer", "Topic"],
            "slot_mapping": {
                "Cognizer": "VAL_X1_AGENT",
                "Topic": "VAL_X2_PATIENT",
            },
            "semantic_types": {
                "Cognizer": ["TYPE_HUMAN", "ROLE_SENTIENT"],
                "Topic": ["TYPE_ABSTRACT_CONCEPT", "TYPE_PROPOSITION"],
            },
        },
        "Desiring": {
            "frame_id": "frame:Desiring.01",
            "core_elements": ["Experiencer", "Event"],
            "slot_mapping": {
                "Experiencer": "VAL_X1_AGENT",
                "Event": "VAL_X2_PATIENT",
            },
            "semantic_types": {
                "Experiencer": ["TYPE_HUMAN", "TYPE_ANIMATE", "ROLE_SENTIENT"],
                "Event": ["TYPE_EVENT", "TYPE_STATE", "TYPE_ABSTRACT_CONCEPT"],
            },
        },
        "Placing": {
            "frame_id": "frame:Placing.01",
            "core_elements": ["Agent", "Theme", "Goal"],
            "slot_mapping": {
                "Agent": "VAL_X1_AGENT",
                "Theme": "VAL_X2_PATIENT",
                "Goal": "VAL_X3_DESTINATION",
            },
            "semantic_types": {
                "Agent": ["TYPE_ANIMATE", "ROLE_AGENT_CAPABLE"],
                "Theme": ["TYPE_INANIMATE_PHYSICAL", "TYPE_ARTIFACT"],
                "Goal": ["TYPE_SPATIAL_REGION", "TYPE_INANIMATE_PHYSICAL"],
            },
        },
        "Removing": {
            "frame_id": "frame:Removing.01",
            "core_elements": ["Agent", "Theme", "Source"],
            "slot_mapping": {
                "Agent": "VAL_X1_AGENT",
                "Theme": "VAL_X2_PATIENT",
                "Source": "VAL_X4_SOURCE",
            },
            "semantic_types": {
                "Agent": ["TYPE_ANIMATE", "ROLE_AGENT_CAPABLE"],
                "Theme": ["TYPE_INANIMATE_PHYSICAL", "TYPE_ARTIFACT"],
                "Source": ["TYPE_SPATIAL_REGION", "TYPE_INANIMATE_PHYSICAL"],
            },
        },
        "Manipulation": {
            "frame_id": "frame:Manipulation.01",
            "core_elements": ["Agent", "Entity", "Instrument"],
            "slot_mapping": {
                "Agent": "VAL_X1_AGENT",
                "Entity": "VAL_X2_PATIENT",
                "Instrument": "VAL_X5_INSTRUMENT",
            },
            "semantic_types": {
                "Agent": ["TYPE_ANIMATE", "ROLE_AGENT_CAPABLE"],
                "Entity": ["TYPE_INANIMATE_PHYSICAL", "TYPE_ARTIFACT"],
            },
        },
        "Commerce_buy": {
            "frame_id": "frame:Commerce_buy.01",
            "core_elements": ["Buyer", "Goods", "Seller", "Money"],
            "slot_mapping": {
                "Buyer": "VAL_X1_AGENT",
                "Goods": "VAL_X2_PATIENT",
                "Seller": "VAL_X4_SOURCE",
                "Money": "VAL_X5_INSTRUMENT",
            },
            "semantic_types": {
                "Buyer": ["TYPE_HUMAN", "TYPE_ORGANIZATION", "ROLE_AGENT_CAPABLE"],
                "Goods": ["TYPE_INANIMATE_PHYSICAL", "TYPE_ARTIFACT", "TYPE_SUBSTANCE_MASS"],
            },
        },
        "Social_interaction": {
            "frame_id": "frame:Social_interaction.01",
            "core_elements": ["Agent", "Co_agent", "Topic"],
            "slot_mapping": {
                "Agent": "VAL_X1_AGENT",
                "Co_agent": "VAL_X2_PATIENT",
            },
            "semantic_types": {
                "Agent": ["TYPE_HUMAN", "TYPE_ANIMATE", "ROLE_SENTIENT"],
                "Co_agent": ["TYPE_HUMAN", "TYPE_ANIMATE", "ROLE_SENTIENT"],
            },
        },
        "Seeking": {
            "frame_id": "frame:Seeking.01",
            "core_elements": ["Cognizer_agent", "Sought_entity"],
            "slot_mapping": {
                "Cognizer_agent": "VAL_X1_AGENT",
                "Sought_entity": "VAL_X2_PATIENT",
            },
            "semantic_types": {
                "Cognizer_agent": ["TYPE_HUMAN", "TYPE_ANIMATE", "ROLE_SENTIENT"],
                "Sought_entity": ["TYPE_INANIMATE_PHYSICAL", "TYPE_ANIMATE", "TYPE_HUMAN"],
            },
        },
        "Assistance": {
            "frame_id": "frame:Assistance.01",
            "core_elements": ["Helper", "Benefited_party"],
            "slot_mapping": {
                "Helper": "VAL_X1_AGENT",
                "Benefited_party": "VAL_X2_PATIENT",
            },
            "semantic_types": {
                "Helper": ["TYPE_HUMAN", "TYPE_ORGANIZATION", "ROLE_AGENT_CAPABLE"],
                "Benefited_party": ["TYPE_HUMAN", "TYPE_ANIMATE", "ROLE_SENTIENT"],
            },
        },
        "Intentionally_affect": {
            "frame_id": "frame:Intentionally_affect.01",
            "core_elements": ["Agent", "Patient", "Means"],
            "slot_mapping": {
                "Agent": "VAL_X1_AGENT",
                "Patient": "VAL_X2_PATIENT",
                "Means": "VAL_X5_INSTRUMENT",
            },
            "semantic_types": {
                "Agent": ["TYPE_ANIMATE", "ROLE_AGENT_CAPABLE"],
                "Patient": ["TYPE_INANIMATE_PHYSICAL", "TYPE_ANIMATE", "TYPE_HUMAN"],
            },
        },
        "Creating": {
            "frame_id": "frame:Creating.01",
            "core_elements": ["Creator", "Created_entity"],
            "slot_mapping": {
                "Creator": "VAL_X1_AGENT",
                "Created_entity": "VAL_X2_PATIENT",
            },
            "semantic_types": {
                "Creator": ["TYPE_HUMAN", "TYPE_ORGANIZATION", "ROLE_AGENT_CAPABLE"],
                "Created_entity": ["TYPE_ARTIFACT", "TYPE_ABSTRACT_CONCEPT", "TYPE_PROPOSITION"],
            },
        },
    }

    def __init__(self, templates_path: Optional[Union[str, Path]] = None):
        self._templates: Dict[str, FrameNetTemplate] = {}
        path = Path(templates_path) if templates_path else find_quanta_data_file("framenet_valency.json")
        if path and path.exists():
            self.load_templates(path)
        else:
            self._init_builtins()

    @classmethod
    def get_instance(cls) -> FrameNetValencyResolver:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_builtins(self):
        for frame_name, f_data in self.BUILTIN_FRAMES.items():
            self._templates[frame_name] = FrameNetTemplate(
                frame_name=frame_name,
                frame_id=f_data.get("frame_id", f"frame:{frame_name}.01"),
                core_elements=f_data.get("core_elements", []),
                slot_mapping=f_data.get("slot_mapping", {}),
                semantic_types=f_data.get("semantic_types", {}),
                lemmas=[v for v, fr in self.VERB_LEMMA_TO_FRAME.items() if fr == frame_name],
            )

    def load_templates(self, path: Union[str, Path]):
        """Loads FrameNet valency and role template matrices from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            frames = data.get("frames", {})
            for frame_name, f_data in frames.items():
                self._templates[frame_name] = FrameNetTemplate(
                    frame_name=frame_name,
                    frame_id=f_data.get("frame_id", f"frame:{frame_name}.01"),
                    core_elements=f_data.get("core_elements", []),
                    slot_mapping=f_data.get("slot_mapping", {}),
                    semantic_types=f_data.get("semantic_types", {}),
                    lemmas=[v for v, fr in self.VERB_LEMMA_TO_FRAME.items() if fr == frame_name],
                )
        # Ensure built-ins fill any missing frames
        for frame_name in self.BUILTIN_FRAMES:
            if frame_name not in self._templates:
                f_data = self.BUILTIN_FRAMES[frame_name]
                self._templates[frame_name] = FrameNetTemplate(
                    frame_name=frame_name,
                    frame_id=f_data.get("frame_id", f"frame:{frame_name}.01"),
                    core_elements=f_data.get("core_elements", []),
                    slot_mapping=f_data.get("slot_mapping", {}),
                    semantic_types=f_data.get("semantic_types", {}),
                    lemmas=[v for v, fr in self.VERB_LEMMA_TO_FRAME.items() if fr == frame_name],
                )

    def get_frame(self, frame_name: str) -> Optional[FrameNetTemplate]:
        return self._templates.get(frame_name)

    def resolve_frame_roles(self, verb_lemma: str) -> Dict[str, str]:
        """Maps a verb lemma to its FrameNet roles and corresponding Band 1 valency slots."""
        lemma = verb_lemma.strip().lower()
        frame_name = self.VERB_LEMMA_TO_FRAME.get(lemma)
        if not frame_name or frame_name not in self._templates:
            return {
                "Agent": "VAL_X1_AGENT",
                "Patient": "VAL_X2_PATIENT",
            }

        template = self._templates[frame_name]
        role_map = dict(template.slot_mapping)

        # Synthesize standard Agent and Patient aliases
        for role, slot in template.slot_mapping.items():
            if slot == "VAL_X1_AGENT":
                role_map["Agent"] = "VAL_X1_AGENT"
            elif slot == "VAL_X2_PATIENT":
                role_map["Patient"] = "VAL_X2_PATIENT"

        if frame_name == "Motion":
            role_map["Agent"] = "VAL_X1_AGENT"

        return role_map


def resolve_frame_roles(verb_lemma: str, resolver: Optional[FrameNetValencyResolver] = None) -> Dict[str, str]:
    """Module-level helper to resolve a verb lemma to its FrameNet role-to-Band1-slot mapping."""
    r = resolver or FrameNetValencyResolver.get_instance()
    return r.resolve_frame_roles(verb_lemma)


__all__ = [
    "GroundedLexicalConcept",
    "ConceptNetLexicalGrounder",
    "LexicalGrounder",
    "WordNetLexicalGrounder",
    "FrameNetTemplate",
    "FrameNetValencyResolver",
    "resolve_frame_roles",
    "NLTK_WN_AVAILABLE",
]


