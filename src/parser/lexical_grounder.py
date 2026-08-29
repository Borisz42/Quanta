"""WordNet lexical grounder for mapping lemmas and synsets to Band 2 Ontological Signatures."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from nltk.corpus import wordnet as wn
    NLTK_WN_AVAILABLE = True
except ImportError:
    wn = None
    NLTK_WN_AVAILABLE = False

from core.slots import get_slot_by_name
from core.types import QuantaVector, QuaternaryValue
from core.asg import QuantaNode


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
            default_db = Path("data/wordnet_offline.db")
            cls._default_instance = cls(offline_cache_path=default_db if default_db.exists() else None)
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
            return GroundedLexicalConcept(
                lemma=entry["lemma"],
                synset_name=f"wn:{key}",
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


def validate_valency(
    parent_node_or_vector: Union[QuantaNode, QuantaVector],
    relation_slot: str,
    child_node_or_vector: Union[QuantaNode, QuantaVector],
) -> bool:
    """Validates compile-time type constraints and valency signatures.
    
    Checks whether a child argument satisfies the ontological requirements of the predicate slot.
    Returns True if valid, False if rejected by constraint violation.
    """
    p_vec = parent_node_or_vector.vector if isinstance(parent_node_or_vector, QuantaNode) else parent_node_or_vector
    c_vec = child_node_or_vector.vector if isinstance(child_node_or_vector, QuantaNode) else child_node_or_vector

    # Figurative language suspends strict physical valency constraints
    if p_vec["MODALITY_FIGURATIVE"] == QuaternaryValue.TRUE or c_vec["MODALITY_FIGURATIVE"] == QuaternaryValue.TRUE:
        return True

    # Variable bindings (e.g. FOL x, y) bypass static valency checks
    if c_vec["GRAPH_VARIABLE_BIND"] == QuaternaryValue.TRUE:
        return True

    # Agent slot validation
    if relation_slot == "VAL_X1_AGENT":
        # If predicate is a mental/cognitive action (THINK, KNOW, WANT, FEEL), agent MUST be sentient/animate
        is_mental = (
            p_vec["NSM_THINK"] == QuaternaryValue.TRUE
            or p_vec["NSM_KNOW"] == QuaternaryValue.TRUE
            or p_vec["NSM_WANT"] == QuaternaryValue.TRUE
            or p_vec["NSM_FEEL"] == QuaternaryValue.TRUE
        )
        if is_mental:
            is_sentient = (
                c_vec["ROLE_SENTIENT"] == QuaternaryValue.TRUE
                or c_vec["TYPE_HUMAN"] == QuaternaryValue.TRUE
                or c_vec["TYPE_ANIMATE"] == QuaternaryValue.TRUE
                or c_vec["WN_ANIMAL_FAUNA"] == QuaternaryValue.TRUE
                or c_vec["WN_PERSON_HUMAN"] == QuaternaryValue.TRUE
            )
            if not is_sentient:
                return False

        # General Agent check: Must be animate, human, organization, or explicitly agent-capable
        is_agent_capable = (
            c_vec["ROLE_AGENT_CAPABLE"] == QuaternaryValue.TRUE
            or c_vec["TYPE_HUMAN"] == QuaternaryValue.TRUE
            or c_vec["TYPE_ANIMATE"] == QuaternaryValue.TRUE
            or c_vec["TYPE_ORGANIZATION"] == QuaternaryValue.TRUE
            or c_vec["WN_ANIMAL_FAUNA"] == QuaternaryValue.TRUE
            or c_vec["WN_PERSON_HUMAN"] == QuaternaryValue.TRUE
        )
        if not is_agent_capable:
            # Inanimate physical or abstract concept cannot be an agent in literal context
            if (
                c_vec["TYPE_INANIMATE_PHYSICAL"] == QuaternaryValue.TRUE
                or c_vec["TYPE_ABSTRACT_CONCEPT"] == QuaternaryValue.TRUE
            ):
                return False

    # Experiencer slot validation
    elif relation_slot == "VAL_EXPERIENCER":
        is_sentient = (
            c_vec["ROLE_SENTIENT"] == QuaternaryValue.TRUE
            or c_vec["TYPE_HUMAN"] == QuaternaryValue.TRUE
            or c_vec["TYPE_ANIMATE"] == QuaternaryValue.TRUE
            or c_vec["WN_ANIMAL_FAUNA"] == QuaternaryValue.TRUE
            or c_vec["WN_PERSON_HUMAN"] == QuaternaryValue.TRUE
        )
        if not is_sentient:
            return False

    # Temporal slot validation
    elif relation_slot == "VAL_TIME_SLOT":
        is_temporal = (
            c_vec["TYPE_TEMPORAL_INTERVAL"] == QuaternaryValue.TRUE
            or c_vec["NSM_TIME"] == QuaternaryValue.TRUE
        )
        if not is_temporal and c_vec["TYPE_ANIMATE"] == QuaternaryValue.TRUE:
            return False

    return True


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

