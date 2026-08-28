"""Tests for WordNet lexical grounding and ontological slot assignment."""

import pytest
from quanta.parser.lexical_grounder import WordNetLexicalGrounder, NLTK_WN_AVAILABLE


@pytest.mark.skipif(not NLTK_WN_AVAILABLE, reason="NLTK WordNet not available")
def test_ground_animal_concept():
    grounder = WordNetLexicalGrounder()
    concept = grounder.ground_synset("dog.n.01")

    assert concept.lemma == "dog"
    assert concept.synset_name == "wn:dog.n.01"
    assert concept.active_slots.get("TYPE_ANIMATE") == 1
    assert concept.active_slots.get("ROLE_AGENT_CAPABLE") == 1
    assert concept.active_slots.get("ROLE_SENTIENT") == 1
    assert concept.active_slots.get("WN_ANIMAL_FAUNA") == 1
    assert concept.vector["TYPE_ANIMATE"] == 1


@pytest.mark.skipif(not NLTK_WN_AVAILABLE, reason="NLTK WordNet not available")
def test_ground_human_concept():
    grounder = WordNetLexicalGrounder()
    concept = grounder.ground_synset("mailman.n.01")

    assert concept.synset_name == "wn:mailman.n.01"
    assert concept.active_slots.get("TYPE_HUMAN") == 1
    assert concept.active_slots.get("TYPE_ANIMATE") == 1
    assert concept.active_slots.get("ROLE_AGENT_CAPABLE") == 1
    assert concept.active_slots.get("ROLE_COMMUNICATOR") == 1
    assert concept.active_slots.get("WN_PERSON_HUMAN") == 1


@pytest.mark.skipif(not NLTK_WN_AVAILABLE, reason="NLTK WordNet not available")
def test_ground_artifact_concept():
    grounder = WordNetLexicalGrounder()
    concept = grounder.ground_synset("hammer.n.02")

    assert concept.active_slots.get("TYPE_ARTIFACT") == 1
    assert concept.active_slots.get("TYPE_INANIMATE_PHYSICAL") == 1
    assert concept.active_slots.get("ROLE_INSTRUMENT_USABLE") == 1


@pytest.mark.skipif(not NLTK_WN_AVAILABLE, reason="NLTK WordNet not available")
def test_cache_serialization(tmp_path):
    grounder = WordNetLexicalGrounder()
    # Ground some concepts
    grounder.ground_synset("dog.n.01")
    grounder.ground_synset("cat.n.01")

    cache_file = tmp_path / "wordnet_offline.json"
    grounder.save_cache(cache_file)
    assert cache_file.exists()

    # Load into a new grounder
    new_grounder = WordNetLexicalGrounder(offline_cache_path=cache_file)
    concept = new_grounder.ground_synset("dog.n.01")
    assert concept.lemma == "dog"
    assert concept.active_slots.get("TYPE_ANIMATE") == 1
