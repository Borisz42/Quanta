import pytest
from parser.lexical_grounder import (
    WordNetLexicalGrounder,
    ConceptNetLexicalGrounder,
    LexicalGrounder,
    NLTK_WN_AVAILABLE,
    FrameNetValencyResolver,
    resolve_frame_roles,
)


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

    # 5A.4: get_wordnet_root_category mapping to Band 3 slot index
    from core.slots import get_slot_by_name
    cat_idx = grounder.get_wordnet_root_category("dog.n.01")
    assert cat_idx == get_slot_by_name("WN_ANIMAL_FAUNA").index

    person_idx = grounder.get_wordnet_root_category("mailman.n.01")
    assert person_idx == get_slot_by_name("WN_PERSON_HUMAN").index

    artifact_idx = grounder.get_wordnet_root_category("hammer.n.02")
    assert artifact_idx == get_slot_by_name("WN_ARTIFACT_OBJECT").index


def test_framenet_valency_resolution():
    resolver = FrameNetValencyResolver.get_instance()

    # 5B.3: Test: "bite" maps to frame with Agent, Patient roles → VAL_X1_AGENT, VAL_X2_PATIENT
    bite_roles = resolver.resolve_frame_roles("bite")
    assert bite_roles["Agent"] == "VAL_X1_AGENT"
    assert bite_roles["Patient"] == "VAL_X2_PATIENT"
    assert bite_roles["Ingestor"] == "VAL_X1_AGENT"
    assert bite_roles["Ingestibles"] == "VAL_X2_PATIENT"
    assert bite_roles["Instrument"] == "VAL_X5_INSTRUMENT"

    # Module helper check
    bite_helper = resolve_frame_roles("bite")
    assert bite_helper == bite_roles

    # Motion: "run"
    run_roles = resolver.resolve_frame_roles("run")
    assert run_roles["Agent"] == "VAL_X1_AGENT"
    assert run_roles["Source"] == "VAL_X4_SOURCE"
    assert run_roles["Goal"] == "VAL_X3_DESTINATION"
    assert run_roles["Manner"] == "VAL_MANNER_SLOT"

    # Statement: "say"
    say_roles = resolver.resolve_frame_roles("say")
    assert say_roles["Agent"] == "VAL_X1_AGENT"
    assert say_roles["Patient"] == "VAL_X2_PATIENT"
    assert say_roles["Speaker"] == "VAL_X1_AGENT"
    assert say_roles["Message"] == "VAL_X2_PATIENT"
    assert say_roles["Addressee"] == "VAL_EXPERIENCER"

    # Perception: "see"
    see_roles = resolver.resolve_frame_roles("see")
    assert see_roles["Agent"] == "VAL_X1_AGENT"
    assert see_roles["Patient"] == "VAL_X2_PATIENT"
    assert see_roles["Perceiver_agentive"] == "VAL_X1_AGENT"
    assert see_roles["Phenomenon"] == "VAL_X2_PATIENT"

    # Causation: "cause"
    cause_roles = resolver.resolve_frame_roles("cause")
    assert cause_roles["Agent"] == "VAL_X1_AGENT"
    assert cause_roles["Patient"] == "VAL_X2_PATIENT"
    assert cause_roles["Cause"] == "VAL_X1_AGENT"
    assert cause_roles["Effect"] == "VAL_RESULT_SLOT"

    # Placing: "put"
    put_roles = resolver.resolve_frame_roles("put")
    assert put_roles["Agent"] == "VAL_X1_AGENT"
    assert put_roles["Theme"] == "VAL_X2_PATIENT"
    assert put_roles["Goal"] == "VAL_X3_DESTINATION"

    # Commerce: "buy"
    buy_roles = resolver.resolve_frame_roles("buy")
    assert buy_roles["Agent"] == "VAL_X1_AGENT"
    assert buy_roles["Goods"] == "VAL_X2_PATIENT"
    assert buy_roles["Buyer"] == "VAL_X1_AGENT"

    # Assistance: "help"
    help_roles = resolver.resolve_frame_roles("help")
    assert help_roles["Agent"] == "VAL_X1_AGENT"
    assert help_roles["Helper"] == "VAL_X1_AGENT"
    assert help_roles["Benefited_party"] == "VAL_X2_PATIENT"

    # Creating: "create"
    create_roles = resolver.resolve_frame_roles("create")
    assert create_roles["Agent"] == "VAL_X1_AGENT"
    assert create_roles["Creator"] == "VAL_X1_AGENT"
    assert create_roles["Created_entity"] == "VAL_X2_PATIENT"


@pytest.mark.skipif(not NLTK_WN_AVAILABLE, reason="NLTK WordNet not available")
def test_ground_extended_domain_concepts():
    grounder = WordNetLexicalGrounder()

    # Kinship
    father_concept = grounder.ground_synset("father.n.01")
    assert father_concept.active_slots.get("TYPE_HUMAN") == 1
    assert father_concept.active_slots.get("WN_PERSON_HUMAN") == 1

    # Animals
    lion_concept = grounder.ground_synset("lion.n.01")
    assert lion_concept.active_slots.get("TYPE_ANIMATE") == 1
    assert lion_concept.active_slots.get("WN_ANIMAL_FAUNA") == 1

    # Places
    kitchen_concept = grounder.ground_synset("kitchen.n.01")
    assert kitchen_concept.active_slots.get("WN_ARTIFACT_OBJECT") == 1 or kitchen_concept.active_slots.get("WN_LOCATION_PLACE") == 1


def test_conceptnet_grounder_aliasing_and_singleton():
    """Verify LexicalGrounder aliases ConceptNetLexicalGrounder and singleton access."""
    assert LexicalGrounder is ConceptNetLexicalGrounder
    g1 = ConceptNetLexicalGrounder.get_default()
    g2 = LexicalGrounder.get_default()
    assert g1 is g2


def test_conceptnet_grounder_multi_pos_and_normalization():
    """5B.1 & 5B.2: Verify multi-POS resolution, underscore/space normalization, and binary hex unpacking."""
    grounder = ConceptNetLexicalGrounder.get_default()

    # Noun resolution
    dog_n = grounder.resolve_concept("dog", pos="n")
    assert dog_n is not None
    assert dog_n.lemma == "dog"
    assert dog_n.pos == "n"
    assert dog_n.synset_name == "cn:en:dog (n)"
    assert dog_n.vector["CN_Q011_ANIMAL"] == 1

    # Verb resolution
    bite_v = grounder.resolve_concept("bite", pos="v")
    assert bite_v is not None
    assert bite_v.lemma == "bite"
    assert bite_v.pos == "v"

    # Multiword expression with space and underscore
    gr_space = grounder.resolve_concept("golden retriever", pos="n")
    gr_under = grounder.resolve_concept("golden_retriever", pos="n")
    assert gr_space is not None
    assert gr_under is not None
    assert gr_space.lemma == gr_under.lemma

    # In-memory LRU cache hit test (5B.3)
    cached_dog = grounder.resolve_concept("dog", pos="n")
    assert cached_dog is dog_n

    # Typo tolerance fallback (5D.1, 5D.3)
    typo_concept = grounder.resolve_concept("glden retreiver", pos="n")
    assert typo_concept is not None
    assert "retriever" in typo_concept.lemma


def test_conceptnet_resolve_and_ground_synset_helpers():
    """Verify resolve_synset and ground_synset helper methods."""
    grounder = ConceptNetLexicalGrounder.get_default()

    anchor = grounder.resolve_synset("mailman", pos="n")
    assert anchor == "cn:en:mailman (n)"

    grounded = grounder.ground_synset("cn:en:mailman (n)")
    assert grounded.lemma == "mailman"
    assert grounded.pos == "n"



