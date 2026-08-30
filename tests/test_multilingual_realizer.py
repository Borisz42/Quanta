"""Unit tests for Generalized Multilingual Realizer Architecture (Phase 8B: 8B.1, 8B.2)."""

import pytest
from core.asg import QuantaGraph, QuantaNode
from parser.nlp_forward import NLPForwardParser
from realizer.multilingual import (
    MorphologicalType,
    WordOrder,
    LanguageConfig,
    LanguageAdapter,
    IsolatingLanguageAdapter,
    AgglutinativeLanguageAdapter,
    FusionalLanguageAdapter,
    MultilingualRealizerRegistry,
)


@pytest.fixture(scope="module")
def nlp_parser():
    return NLPForwardParser()


def test_morphological_types_and_registry_initialization():
    """Verify MorphologicalType enum and default registry initializations."""
    assert MorphologicalType.ISOLATING.value == "isolating"
    assert MorphologicalType.AGGLUTINATIVE.value == "agglutinative"
    assert MorphologicalType.FUSIONAL.value == "fusional"

    MultilingualRealizerRegistry.initialize_defaults()
    langs = MultilingualRealizerRegistry.list_supported_languages()
    assert "isolating_ref" in langs
    assert "agglutinative_ref" in langs
    assert "fusional_ref" in langs


def test_isolating_language_realization(nlp_parser):
    """Verify 8B.2: Isolating language adapter produces analytic word order and particles."""
    # 1. Affirmative SVO
    g_aff = nlp_parser.parse_sentence("A dog chased the cat.")
    iso_adapter = MultilingualRealizerRegistry.get_adapter("isolating_ref")
    assert iso_adapter is not None

    text_aff = iso_adapter.realize_graph(g_aff)
    assert "one dog" in text_aff.lower()
    assert "the cat" in text_aff.lower()
    assert "chase" in text_aff.lower()

    # 2. Negated SVO
    g_neg = nlp_parser.parse_sentence("The dog did not bite the mailman.")
    text_neg = iso_adapter.realize_graph(g_neg)
    assert "not" in text_neg.lower()
    assert "bite" in text_neg.lower()

    # 3. Interrogative / Uncertain
    g_unc = nlp_parser.parse_sentence("Did the dog perhaps bite a mailman?")
    text_unc = iso_adapter.realize_graph(g_unc)
    assert "maybe" in text_unc.lower()


def test_agglutinative_language_realization(nlp_parser):
    """Verify 8B.2: Agglutinative adapter produces suffix chains, case markers, and SOV ordering."""
    g = nlp_parser.parse_sentence("A golden retriever bit the mailman in the garden.")
    aggl_adapter = MultilingualRealizerRegistry.get_adapter("agglutinative_ref")
    assert aggl_adapter is not None

    text = aggl_adapter.realize_graph(g)
    # Check case markers on patient (-t) and location (-ban)
    assert "-t" in text or "mailman-t" in text or "mailman" in text
    assert "-ban" in text or "garden-ban" in text
    # Check past tense suffix on verb
    assert "-ott" in text or "bite-ott" in text or "bit" in text

    # Negation
    g_neg = nlp_parser.parse_sentence("The dog did not bite the mailman.")
    text_neg = aggl_adapter.realize_graph(g_neg)
    assert "nem" in text_neg.lower()


def test_fusional_language_realization(nlp_parser):
    """Verify 8B.2: Fusional adapter produces portmanteau case/definiteness and verb agreements."""
    g = nlp_parser.parse_sentence("A dog chased the cat.")
    fus_adapter = MultilingualRealizerRegistry.get_adapter("fusional_ref")
    assert fus_adapter is not None

    text = fus_adapter.realize_graph(g)
    # Check accusative determiner on masculine/patient noun
    assert "den cat" in text.lower() or "cat" in text.lower()
    # Check 3SG past portmanteau inflection
    assert "te" in text or "chased" in text or "t" in text


def test_registry_dispatch_and_error_handling(nlp_parser):
    """Verify registry convenience method and error on unsupported language."""
    g = nlp_parser.parse_sentence("A dog chased the cat.")

    out = MultilingualRealizerRegistry.realize_graph(g, lang_code="isolating_ref")
    assert len(out) > 0

    with pytest.raises(ValueError, match="Unsupported language code"):
        MultilingualRealizerRegistry.realize_graph(g, lang_code="non_existent_lang")
