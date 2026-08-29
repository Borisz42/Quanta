"""Script to build and export the SQLite WordNet offline cache and JSON cache for QUANTA.

Grounds core vocabulary and reasoning dataset synsets into canonical 256-dimensional
ontological vectors and stores them for O(1) offline deployment.
"""

from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import sys

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from nltk.corpus import wordnet as wn
    NLTK_WN_AVAILABLE = True
except ImportError:
    wn = None
    NLTK_WN_AVAILABLE = False

from parser.lexical_grounder import WordNetLexicalGrounder
from parser.typo_normalizer import TypoNormalizer


def build_wordnet_cache(
    db_path: Path = Path("data/wordnet_offline.db"),
    json_path: Path = Path("data/wordnet_offline.json"),
):
    if not NLTK_WN_AVAILABLE:
        print("Error: NLTK WordNet is required to build the offline cache.")
        sys.exit(1)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"=== QUANTA WordNet Offline Cache Builder ===")
    print(f"Target DB:   {db_path.resolve()}")
    print(f"Target JSON: {json_path.resolve()}\n")

    grounder = WordNetLexicalGrounder()

    # Seed with core English lexicon words and common reasoning concepts
    seed_words = list(TypoNormalizer.ENGLISH_CORE_LEXICON.keys()) + [
        # FOLIO & Logic concepts
        "entity", "object", "concept", "proposition", "premise", "conclusion", "fact", "rule",
        "axiom", "proof", "theorem", "hypothesis", "assumption", "implication", "negation",
        # bAbI multi-hop & locations
        "hallway", "kitchen", "garden", "bedroom", "bathroom", "office", "park", "school",
        "football", "milk", "apple", "box", "chest", "container", "cup", "bottle",
        "north", "south", "east", "west", "up", "down", "left", "right", "inside", "outside",
        # ProofWriter traits & animals
        "rough", "smooth", "quiet", "loud", "smart", "clever", "dull", "cold", "warm", "hot",
        "red", "green", "blue", "yellow", "white", "black", "fur", "young", "old", "round",
        "lion", "tiger", "bear", "squirrel", "cat", "dog", "mouse", "eagle", "bald_eagle",
        # CLUTRR Kinship
        "father", "mother", "son", "daughter", "brother", "sister", "grandfather", "grandmother",
        "grandson", "granddaughter", "uncle", "aunt", "nephew", "niece", "cousin", "husband",
        "wife", "parent", "child", "sibling", "ancestor", "descendant", "spouse", "relative",
        # Natural Semantic Metalanguage (NSM) Concepts
        "someone", "something", "people", "body", "kind", "part", "this", "the_same", "other",
        "else", "one", "two", "much", "many", "little", "few", "all", "good", "bad", "big",
        "small", "think", "know", "want", "feel", "see", "hear", "say", "words", "true",
        "do", "happen", "move", "touch", "be", "there_is", "have", "live", "die", "when",
        "time", "now", "before", "after", "a_long_time", "a_short_time", "for_some_time",
        "moment", "where", "place", "here", "above", "below", "far", "near", "side", "inside",
        "not", "maybe", "can", "because", "if", "very", "more", "like",
        # Physical & Interpersonal Actions
        "put", "place", "set", "lay", "drop", "hold", "grasp", "grip", "carry", "buy", "sell",
        "pay", "meet", "visit", "greet", "search", "seek", "hunt", "find", "help", "assist",
        "kill", "destroy", "damage", "repair", "fix", "create", "build", "write", "learn", "study",
        "step", "swim", "climb", "fall", "enter", "exit", "cross", "return", "ask", "answer",
    ]

    # Collect distinct synsets
    synset_keys = set()
    for word in seed_words:
        w_clean = word.strip().lower().replace(" ", "_")
        try:
            syns = wn.synsets(w_clean)
            if syns:
                for s in syns[:3]:  # Top 3 synsets for each word
                    synset_keys.add(s.name())
                synset_keys.add(w_clean)
        except Exception:
            continue

    print(f"1. Grounding {len(synset_keys)} synsets and lemmas...")
    success_count = 0
    for key in sorted(synset_keys):
        try:
            concept = grounder.ground_synset(key)
            success_count += 1
        except Exception:
            continue

    print(f"   Successfully grounded: {success_count} entries")

    # Save to SQLite Database
    print(f"\n2. Writing to SQLite database at {db_path.name}...")
    grounder.save_cache(db_path)

    # Save to JSON Cache
    print(f"3. Writing to JSON cache at {json_path.name}...")
    grounder.save_cache(json_path)

    print(f"\n=== Offline WordNet cache build complete! ===")


if __name__ == "__main__":
    db_out = REPO_ROOT / "data" / "wordnet_offline.db"
    json_out = REPO_ROOT / "data" / "wordnet_offline.json"
    build_wordnet_cache(db_out, json_out)
