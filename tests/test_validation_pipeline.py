"""Integration tests for QUANTA TwoWayTranslationPipeline, TranslatorPipeline, and MUC gating."""

import pytest
from core.asg import QuantaGraph, QuantaNode
from pipeline.translator_pipeline import TwoWayTranslationPipeline, TranslatorPipeline, StageLog


@pytest.fixture(scope="module")
def pipeline():
    return TwoWayTranslationPipeline()


def test_translator_pipeline_alias():
    """Verify TranslatorPipeline is an alias of TwoWayTranslationPipeline."""
    assert TranslatorPipeline is TwoWayTranslationPipeline
    p = TranslatorPipeline()
    assert isinstance(p, TwoWayTranslationPipeline)


def test_valid_pipeline_execution(pipeline):
    # Valid input translation
    res = pipeline.execute_translation("A golden retriever chased a ball.", target_modality="english")
    assert res.is_success
    assert res.output_text != ""
    assert res.merkle_root is not None
    assert res.validation.is_valid

    # Phase 12.4: Verify structured per-stage telemetry & timings
    assert isinstance(res.stage_timings, dict)
    assert "modality_detection" in res.stage_timings
    assert "forward_parse" in res.stage_timings
    assert "validation_gate" in res.stage_timings
    assert "merkle_addressing" in res.stage_timings
    assert "reverse_realization" in res.stage_timings
    assert res.total_duration_ms > 0

    assert len(res.stage_logs) == 5
    for stage_log in res.stage_logs:
        assert isinstance(stage_log, StageLog)
        assert stage_log.status == "success"
        assert stage_log.duration_ms >= 0.0
        log_dict = stage_log.to_dict()
        assert "stage_name" in log_dict
        assert "duration_ms" in log_dict


def test_round_trip_telemetry_and_logging(pipeline):
    """Phase 12.4: Verify round_trip stage timings and log emission."""
    res = pipeline.round_trip("A golden retriever chased a ball.", modality="english")
    assert res.validation_pass
    assert res.slot_preservation_rate > 0.8
    assert res.total_duration_ms > 0

    assert "forward_pass_1" in res.stage_timings
    assert "reverse_realization" in res.stage_timings
    assert "forward_pass_2_reparse" in res.stage_timings
    assert "invariance_audit" in res.stage_timings

    assert len(res.stage_logs) == 4
    for s in res.stage_logs:
        assert s.status == "success"
        assert s.duration_ms >= 0.0


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

