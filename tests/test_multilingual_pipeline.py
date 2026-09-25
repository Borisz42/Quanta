"""Cross-Lingual Multilingual Forward & Reverse Transduction Tests (Section 8 / exp-012a).

Verifies the universal neuro-symbolic Mentalese interlingua:
1. Multi-lingual Forward Transduction: Non-English discourse (Hungarian, German, Turkish,
   Mandarin) parsed into strict canonical ASG S-expressions with English pivot labels.
2. Cross-Lingual Zero Hamming Drift: Direct comparison of Hungarian, German, Turkish,
   and Mandarin ASGs against English counterparts proves d_H = 0 on core proposition vectors.
3. High Slot Preservation: Slot preservation rate >= 95% across languages.
4. Cross-Lingual Translation: Non-English input -> ASG -> English realization via honest NLG.
5. Neural Reverse Realization: ASG -> Fluent target-language text generation.
6. Closed-Loop Cycle Consistency: Round-trip invariance verified with LatticeInvarianceGate.
"""

import pytest

from core.asg import QuantaGraph
from parser.asg_compiler import ASGCompiler
from parser.sexpr_parser import parse_sexpr, to_sexpr, serialize_to_sexpr
from parser.unsloth_transducer import (
    DEFAULT_UNSLOTH_SYSTEM_PROMPT,
    MockUnslothTransducer,
    UnslothTransducer,
)
from pipeline.translator_pipeline import TwoWayTranslationPipeline
from verification.lattice_gate import LatticeInvarianceGate


# ---------------------------------------------------------------------------
# Canonical Multilingual Benchmark Fixtures (Section 8 / Task 8.4)
# ---------------------------------------------------------------------------

HUNGARIAN_TEXT = "A kutya megugatta a postást a kertben."
HUNGARIAN_SEXPR = """(graph :chunk-id "chunk_hu_dog"
  (entity :id E1 :type ANIMAL :label "dog" :surface "A kutya")
  (entity :id E2 :type PERSON :label "postman" :surface "a postást")
  (entity :id E3 :type LOCATION :label "garden" :surface "a kertben")
  (event :id Ev1 :pred bark :agent E1 :patient E2 :location E3 :time "in the past" :tense PAST :polarity TRUE :raw-text "A kutya megugatta a postást a kertben.")
  (relation :type TEMP_ALLEN_DURING :source Ev1 :target Ev1)
)"""

GERMAN_TEXT = "Der Forscher untersuchte die Probe im Laboratorium."
GERMAN_SEXPR = """(graph :chunk-id "chunk_de_researcher"
  (entity :id E1 :type PERSON :label "researcher" :surface "Der Forscher")
  (entity :id E2 :type OBJECT :label "sample" :surface "die Probe")
  (entity :id E3 :type LOCATION :label "laboratory" :surface "im Laboratorium")
  (event :id Ev1 :pred examine :agent E1 :patient E2 :location E3 :time "in the past" :tense PAST :polarity TRUE :raw-text "Der Forscher untersuchte die Probe im Laboratorium.")
  (relation :type TEMP_ALLEN_DURING :source Ev1 :target Ev1)
)"""

TURKISH_TEXT = "Köpek bahçede postacıya havladı."
TURKISH_SEXPR = """(graph :chunk-id "chunk_tr_dog"
  (entity :id E1 :type ANIMAL :label "dog" :surface "Köpek")
  (entity :id E2 :type LOCATION :label "garden" :surface "bahçede")
  (entity :id E3 :type PERSON :label "postman" :surface "postacıya")
  (event :id Ev1 :pred bark :agent E1 :patient E3 :location E2 :time "in the past" :tense PAST :polarity TRUE :raw-text "Köpek bahçede postacıya havladı.")
  (relation :type TEMP_ALLEN_DURING :source Ev1 :target Ev1)
)"""

MANDARIN_TEXT = "科学家在实验室里合成了新型聚合物。"
MANDARIN_SEXPR = """(graph :chunk-id "chunk_zh_science"
  (entity :id E1 :type PERSON :label "scientist" :surface "科学家")
  (entity :id E2 :type LOCATION :label "laboratory" :surface "实验室")
  (entity :id E3 :type SUBSTANCE :label "polymer" :surface "新型聚合物")
  (event :id Ev1 :pred synthesize :agent E1 :patient E3 :location E2 :time "in the past" :tense PAST :polarity TRUE :raw-text "科学家在实验室里合成了新型聚合物。")
  (relation :type TEMP_ALLEN_DURING :source Ev1 :target Ev1)
)"""

# Corresponding English gold references
ENGLISH_DOG_TEXT = "The dog barked at the postman in the garden."
ENGLISH_DOG_SEXPR = """(graph :chunk-id "chunk_en_dog"
  (entity :id E1 :type ANIMAL :label "dog" :surface "The dog")
  (entity :id E2 :type PERSON :label "postman" :surface "the postman")
  (entity :id E3 :type LOCATION :label "garden" :surface "the garden")
  (event :id Ev1 :pred bark :agent E1 :patient E2 :location E3 :time "in the past" :tense PAST :polarity TRUE :raw-text "The dog barked at the postman in the garden.")
  (relation :type TEMP_ALLEN_DURING :source Ev1 :target Ev1)
)"""

ENGLISH_RESEARCHER_TEXT = "The researcher examined the sample in the laboratory."
ENGLISH_RESEARCHER_SEXPR = """(graph :chunk-id "chunk_en_researcher"
  (entity :id E1 :type PERSON :label "researcher" :surface "The researcher")
  (entity :id E2 :type OBJECT :label "sample" :surface "the sample")
  (entity :id E3 :type LOCATION :label "laboratory" :surface "the laboratory")
  (event :id Ev1 :pred examine :agent E1 :patient E2 :location E3 :time "in the past" :tense PAST :polarity TRUE :raw-text "The researcher examined the sample in the laboratory.")
  (relation :type TEMP_ALLEN_DURING :source Ev1 :target Ev1)
)"""

ENGLISH_SCIENCE_TEXT = "The scientist synthesized the polymer in the laboratory."
ENGLISH_SCIENCE_SEXPR = """(graph :chunk-id "chunk_en_science"
  (entity :id E1 :type PERSON :label "scientist" :surface "The scientist")
  (entity :id E2 :type LOCATION :label "laboratory" :surface "the laboratory")
  (entity :id E3 :type SUBSTANCE :label "polymer" :surface "the polymer")
  (event :id Ev1 :pred synthesize :agent E1 :patient E3 :location E2 :time "in the past" :tense PAST :polarity TRUE :raw-text "The scientist synthesized the polymer in the laboratory.")
  (relation :type TEMP_ALLEN_DURING :source Ev1 :target Ev1)
)"""


@pytest.fixture
def mock_multilingual_transducer():
    """Fixture providing MockUnslothTransducer pre-seeded with multilingual fixtures."""
    mock = MockUnslothTransducer()
    # Register forward transduction fixtures
    mock.register_fixture(HUNGARIAN_TEXT, HUNGARIAN_SEXPR)
    mock.register_fixture(GERMAN_TEXT, GERMAN_SEXPR)
    mock.register_fixture(TURKISH_TEXT, TURKISH_SEXPR)
    mock.register_fixture(MANDARIN_TEXT, MANDARIN_SEXPR)
    mock.register_fixture(ENGLISH_DOG_TEXT, ENGLISH_DOG_SEXPR)
    mock.register_fixture(ENGLISH_RESEARCHER_TEXT, ENGLISH_RESEARCHER_SEXPR)
    mock.register_fixture(ENGLISH_SCIENCE_TEXT, ENGLISH_SCIENCE_SEXPR)

    # Register reverse realization fixtures
    mock.register_realization_fixture(HUNGARIAN_SEXPR, HUNGARIAN_TEXT)
    mock.register_realization_fixture(GERMAN_SEXPR, GERMAN_TEXT)
    mock.register_realization_fixture(TURKISH_SEXPR, TURKISH_TEXT)
    mock.register_realization_fixture(MANDARIN_SEXPR, MANDARIN_TEXT)
    return mock


@pytest.fixture
def compiler():
    return ASGCompiler()


# ---------------------------------------------------------------------------
# Task 8.1: System Prompt Specification Tests
# ---------------------------------------------------------------------------

def test_multilingual_system_prompt_spec():
    """Verify Task 8.1: System prompt accepts any-language discourse, specifies English pivot labels and surface retention."""
    assert "discourse in any language" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "Universal English Pivot" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "Surface Retention" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "DEMONSTRATION 3 (AGGLUTINATIVE DISCOURSE / HUNGARIAN)" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "DEMONSTRATION 4 (ISOLATING DISCOURSE / MANDARIN)" in DEFAULT_UNSLOTH_SYSTEM_PROMPT
    assert "DEMONSTRATION 5 (FUSIONAL & COMPOUND DISCOURSE / GERMAN)" in DEFAULT_UNSLOTH_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Task 8.4 & 8.5: Forward Transduction Tests (hu, de, tr, zh)
# ---------------------------------------------------------------------------

def test_hungarian_forward_transduction(mock_multilingual_transducer, compiler):
    """Verify Hungarian forward parse maps to canonical ASG with English pivot labels."""
    extraction = mock_multilingual_transducer.transduce(text=HUNGARIAN_TEXT)
    assert extraction.chunk_id == "chunk_hu_dog"
    assert len(extraction.entities) == 3
    assert len(extraction.events) == 1

    # Check English pivot labels and surface retention
    labels = {e.canonical_name for e in extraction.entities}
    assert labels == {"dog", "postman", "garden"}
    surfaces = {e.surface_aliases[0] if e.surface_aliases else e.canonical_name for e in extraction.entities}
    assert "A kutya" in surfaces or "a postást" in surfaces or "a kertben" in surfaces

    event = extraction.events[0]
    assert event.predicate == "bark"
    assert event.tense == "PAST"
    assert event.polarity is True

    # Compile to QuantaGraph
    graph = compiler.compile(extraction)
    assert len(graph.nodes) >= 4  # 3 entities + 1 event
    assert graph.root_cid is not None


def test_german_forward_transduction(mock_multilingual_transducer, compiler):
    """Verify German forward parse maps to canonical ASG with English pivot labels."""
    extraction = mock_multilingual_transducer.transduce(text=GERMAN_TEXT)
    assert extraction.chunk_id == "chunk_de_researcher"
    labels = {e.canonical_name for e in extraction.entities}
    assert labels == {"researcher", "sample", "laboratory"}
    assert extraction.events[0].predicate == "examine"

    graph = compiler.compile(extraction)
    assert len(graph.nodes) >= 4


def test_turkish_forward_transduction(mock_multilingual_transducer, compiler):
    """Verify Turkish forward parse maps to canonical ASG with English pivot labels."""
    extraction = mock_multilingual_transducer.transduce(text=TURKISH_TEXT)
    assert extraction.chunk_id == "chunk_tr_dog"
    labels = {e.canonical_name for e in extraction.entities}
    assert labels == {"dog", "garden", "postman"}
    assert extraction.events[0].predicate == "bark"

    graph = compiler.compile(extraction)
    assert len(graph.nodes) >= 4


def test_mandarin_forward_transduction(mock_multilingual_transducer, compiler):
    """Verify Mandarin forward parse maps to canonical ASG with English pivot labels."""
    extraction = mock_multilingual_transducer.transduce(text=MANDARIN_TEXT)
    assert extraction.chunk_id == "chunk_zh_science"
    labels = {e.canonical_name for e in extraction.entities}
    assert labels == {"scientist", "laboratory", "polymer"}
    assert extraction.events[0].predicate == "synthesize"

    graph = compiler.compile(extraction)
    assert len(graph.nodes) >= 4


# ---------------------------------------------------------------------------
# Task 8.5: Cross-Lingual Zero Hamming Drift & Slot Preservation
# ---------------------------------------------------------------------------

def test_cross_lingual_zero_hamming_drift_hungarian_vs_english(mock_multilingual_transducer, compiler):
    """Assert Hungarian and English ASGs yield d_H = 0 on core proposition vectors."""
    ext_hu = mock_multilingual_transducer.transduce(text=HUNGARIAN_TEXT)
    ext_en = mock_multilingual_transducer.transduce(text=ENGLISH_DOG_TEXT)

    graph_hu = compiler.compile(ext_hu)
    graph_en = compiler.compile(ext_en)

    vec_hu = graph_hu.to_proposition_vector()
    vec_en = graph_en.to_proposition_vector()

    hamming = vec_hu.hamming_distance(vec_en)
    assert hamming == 0, f"Expected zero Hamming drift, got d_H = {hamming}"

    # Verify slot preservation
    active_hu = set(vec_hu.active_slots().keys())
    active_en = set(vec_en.active_slots().keys())
    total_slots = len(active_hu | active_en)
    matching = sum(1 for s in (active_hu | active_en) if vec_hu[s] == vec_en[s])
    preservation = matching / max(total_slots, 1)
    assert preservation >= 0.95, f"Expected slot preservation >= 95%, got {preservation * 100:.1f}%"


def test_cross_lingual_zero_hamming_drift_german_vs_english(mock_multilingual_transducer, compiler):
    """Assert German and English ASGs yield d_H = 0 on core proposition vectors."""
    ext_de = mock_multilingual_transducer.transduce(text=GERMAN_TEXT)
    ext_en = mock_multilingual_transducer.transduce(text=ENGLISH_RESEARCHER_TEXT)

    graph_de = compiler.compile(ext_de)
    graph_en = compiler.compile(ext_en)

    vec_de = graph_de.to_proposition_vector()
    vec_en = graph_en.to_proposition_vector()

    hamming = vec_de.hamming_distance(vec_en)
    assert hamming == 0, f"Expected d_H = 0 for German vs English, got {hamming}"


def test_cross_lingual_zero_hamming_drift_mandarin_vs_english(mock_multilingual_transducer, compiler):
    """Assert Mandarin and English ASGs yield d_H = 0 on core proposition vectors."""
    ext_zh = mock_multilingual_transducer.transduce(text=MANDARIN_TEXT)
    ext_en = mock_multilingual_transducer.transduce(text=ENGLISH_SCIENCE_TEXT)

    graph_zh = compiler.compile(ext_zh)
    graph_en = compiler.compile(ext_en)

    vec_zh = graph_zh.to_proposition_vector()
    vec_en = graph_en.to_proposition_vector()

    hamming = vec_zh.hamming_distance(vec_en)
    assert hamming == 0, f"Expected d_H = 0 for Mandarin vs English, got {hamming}"


# ---------------------------------------------------------------------------
# Task 8.2: Cross-Lingual Translation & Neural Reverse Realization
# ---------------------------------------------------------------------------

def test_cross_lingual_translation_hungarian_to_english(mock_multilingual_transducer):
    """Test full Hungarian text -> QuantaGraph -> English text pipeline."""
    pipeline = TwoWayTranslationPipeline(transducer=mock_multilingual_transducer)
    result = pipeline.execute_translation(
        input_data=HUNGARIAN_TEXT,
        source_modality="hungarian",
        target_modality="english",
    )
    assert result.is_success is True
    assert result.graph is not None
    # English realization should express dog barking in garden
    out_lower = result.output_text.lower()
    assert "dog" in out_lower
    assert "bark" in out_lower


def test_neural_reverse_realization_all_languages(mock_multilingual_transducer, compiler):
    """Test neural reverse realization into Hungarian, German, Turkish, and Mandarin."""
    pipeline = TwoWayTranslationPipeline(transducer=mock_multilingual_transducer)

    # 1. Realize to Hungarian
    ext_hu = mock_multilingual_transducer.transduce(text=HUNGARIAN_TEXT)
    g_hu = compiler.compile(ext_hu)
    out_hu = pipeline.realize_multilingual(g_hu, target_lang="hungarian")
    assert "kutya" in out_hu.lower()
    assert "kertben" in out_hu.lower()

    # 2. Realize to German
    ext_de = mock_multilingual_transducer.transduce(text=GERMAN_TEXT)
    g_de = compiler.compile(ext_de)
    out_de = pipeline.realize_multilingual(g_de, target_lang="german")
    assert "forscher" in out_de.lower()
    assert "laboratorium" in out_de.lower()

    # 3. Realize to Turkish
    ext_tr = mock_multilingual_transducer.transduce(text=TURKISH_TEXT)
    g_tr = compiler.compile(ext_tr)
    out_tr = pipeline.realize_multilingual(g_tr, target_lang="turkish")
    assert "köpek" in out_tr.lower()
    assert "bahçede" in out_tr.lower()

    # 4. Realize to Mandarin
    ext_zh = mock_multilingual_transducer.transduce(text=MANDARIN_TEXT)
    g_zh = compiler.compile(ext_zh)
    out_zh = pipeline.realize_multilingual(g_zh, target_lang="mandarin")
    assert "科学家" in out_zh or "聚合物" in out_zh


# ---------------------------------------------------------------------------
# Task 8.5: Closed-Loop Round-Trip & Lattice Invariance Gate
# ---------------------------------------------------------------------------

def test_multilingual_round_trip_cycle_consistency(mock_multilingual_transducer):
    """Test full Hungarian round-trip with lattice invariance audit."""
    pipeline = TwoWayTranslationPipeline(transducer=mock_multilingual_transducer)

    # Execute round trip on Hungarian input
    rt_res = pipeline.round_trip(HUNGARIAN_TEXT, modality="hungarian")
    assert rt_res.validation_pass is True
    assert rt_res.hamming_distance == 0
    assert rt_res.slot_preservation_rate >= 0.95
    assert rt_res.is_meet_sound is True
    assert len(rt_res.lattice_meet_errors) == 0
    assert rt_res.is_invariant(max_hamming=0) is True
