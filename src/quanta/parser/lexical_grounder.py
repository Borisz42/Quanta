"""WordNet lexical grounder for mapping lemmas and synsets to Band 2 Ontological Signatures."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from nltk.corpus import wordnet as wn
    NLTK_WN_AVAILABLE = True
except ImportError:
    wn = None
    NLTK_WN_AVAILABLE = False

from quanta.core.slots import get_slot_by_name
from quanta.core.types import QuantaVector, QuaternaryValue
from quanta.core.asg import QuantaNode


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

    def __init__(self, offline_cache_path: Optional[Union[str, Path]] = None):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.offline_cache_path = Path(offline_cache_path) if offline_cache_path else None

        if self.offline_cache_path and self.offline_cache_path.exists():
            self.load_cache(self.offline_cache_path)

    def load_cache(self, path: Union[str, Path]):
        """Loads offline precomputed grounding cache from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            self._cache = json.load(f)

    def save_cache(self, path: Union[str, Path]):
        """Saves current grounding cache to JSON for O(1) offline deployment."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

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
                raise KeyError(f"WordNet synset not found for '{key}'")
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
