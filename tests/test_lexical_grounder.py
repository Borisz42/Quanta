"""Tests for WordNet lexical grounding and ontological slot assignment."""

import pytest
from parser.lexical_grounder import WordNetLexicalGrounder, NLTK_WN_AVAILABLE


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

    # JSON serialization
    json_cache = tmp_path / "wordnet_offline.json"
    grounder.save_cache(json_cache)
    assert json_cache.exists()

    new_grounder = WordNetLexicalGrounder(offline_cache_path=json_cache)
    concept = new_grounder.ground_synset("dog.n.01")
    assert concept.lemma == "dog"
    assert concept.active_slots.get("TYPE_ANIMATE") == 1

    # SQLite serialization
    db_cache = tmp_path / "wordnet_offline.db"
    grounder.save_cache(db_cache)
    assert db_cache.exists()

    sqlite_grounder = WordNetLexicalGrounder(offline_cache_path=db_cache)
    concept_db = sqlite_grounder.ground_synset("cat.n.01")
    assert concept_db.lemma == "cat"
    assert concept_db.active_slots.get("TYPE_ANIMATE") == 1


@pytest.mark.skipif(not NLTK_WN_AVAILABLE, reason="NLTK WordNet not available")
def test_resolve_synset_and_hypernyms():
    grounder = WordNetLexicalGrounder()

    # 5A.2: resolve_synset
    syn_dog = grounder.resolve_synset("dog", pos="n")
    assert syn_dog == "wn:dog.n.01"

    syn_think = grounder.resolve_synset("think", pos="v")
    assert syn_think is not None and syn_think.startswith("wn:think.v")

    # Typo fallback
    syn_typo = grounder.resolve_synset("glden retreiver")
    assert syn_typo == "wn:golden_retriever.n.01"

    # 5A.3: get_hypernym_path
    hyp_path = grounder.get_hypernym_path("dog.n.01")
    assert any("animal.n.01" in h for h in hyp_path)
    assert any("canine.n.02" in h or "domestic_animal.n.01" in h for h in hyp_path)

    # 5A.4: get_wordnet_root_category mapping to Band 2 slot index
    # WN_ANIMAL_FAUNA index is 172
    cat_idx = grounder.get_wordnet_root_category("dog.n.01")
    assert cat_idx == 172

    # WN_PERSON_HUMAN index is 185
    person_idx = grounder.get_wordnet_root_category("mailman.n.01")
    assert person_idx == 185

    # WN_ARTIFACT_OBJECT index is 173
    artifact_idx = grounder.get_wordnet_root_category("hammer.n.02")
    assert artifact_idx == 173

