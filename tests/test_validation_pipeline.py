"""Integration tests for QUANTA TwoWayTranslationPipeline and MUC gating."""

import pytest
from core.asg import QuantaGraph, QuantaNode
from pipeline.translator_pipeline import TwoWayTranslationPipeline


@pytest.fixture(scope="module")
def pipeline():
    return TwoWayTranslationPipeline()


def test_valid_pipeline_execution(pipeline):
    # Valid input translation
    res = pipeline.execute_translation("A golden retriever chased a ball.", target_modality="english")
    assert res.is_success
    assert res.output_text != ""
    assert res.merkle_root is not None
    assert res.validation.is_valid


def test_flawed_proposition_trapped_by_muc(pipeline):
    # Construct an ungrounded proposition where an abstract concept acts as agent without figurative modality
    invalid_event = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1, "MODALITY_LITERAL": 1},
        anchor="wn:bite.v.01",
    )
    invalid_agent = QuantaNode(
        vector={"TYPE_ABSTRACT_CONCEPT": 1},
        anchor="wn:democracy.n.01",
    )

    graph = QuantaGraph()
    e_cid = graph.add_node(invalid_event, set_as_root=True)
    a_cid = graph.add_node(invalid_agent)
    graph.add_edge(e_cid, "VAL_X1_AGENT", a_cid)

    # Validate through pipeline validator
    val_res = pipeline.validator.validate_graph(graph)
    assert not val_res.is_valid
    assert len(val_res.errors) > 0


def test_auto_modality_detection(pipeline):
    assert pipeline.detect_modality("A dog chased a cat.") == "english"
    assert pipeline.detect_modality("\\forall x (P(x) -> Q(x))") == "fol"
    assert pipeline.detect_modality("def calculate(n):\n    return n * 2") == "python"
