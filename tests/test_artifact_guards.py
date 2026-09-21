"""Unit tests for QUANTA fail-fast artifact verification gates."""

import pytest
from pathlib import Path

from core.artifacts import (
    ArtifactCriticality,
    ArtifactSpec,
    MissingArtifactError,
    require_artifacts,
    audit_artifacts,
    resolve_artifact_path,
    ARTIFACT_REGISTRY,
)


def test_registry_contains_critical_artifacts():
    """Verify that the central registry includes all primary critical assets."""
    rel_paths = {spec.rel_path for spec in ARTIFACT_REGISTRY}
    assert "data/conceptnet_offline.db" in rel_paths
    assert "data/conceptnet_slots.json" in rel_paths
    assert "data/wordnet_offline.db" in rel_paths
    assert "data/wordnet_offline.json" in rel_paths
    assert "data/framenet_valency.json" in rel_paths
    assert "data/grammar/quanta_asg.gbnf" in rel_paths
    assert "output/canonical_slots_layout.json" in rel_paths


def test_existing_artifacts_pass_guard():
    """Verify that require_artifacts succeeds when required files exist."""
    # These files are known to exist in the repository
    require_artifacts(
        "output/canonical_slots_layout.json",
        "data/conceptnet_slots.json",
        component="TestPassSuite",
    )


def test_missing_artifact_raises_missing_artifact_error():
    """Verify that requiring a non-existent file immediately raises MissingArtifactError."""
    with pytest.raises(MissingArtifactError) as exc_info:
        require_artifacts("data/imaginary_database_12345.db", component="TestFailureSuite")

    err_msg = str(exc_info.value)
    assert "imaginary_database_12345.db" in err_msg
    assert "TestFailureSuite" in err_msg
    assert "dev.ps1 init" in err_msg or "init_quanta.py" in err_msg


def test_audit_artifacts_report():
    """Verify that audit_artifacts returns structured inventory with critical checks."""
    report = audit_artifacts(critical_only=True)
    assert "all_critical_present" in report
    assert "missing_critical" in report
    assert "artifacts" in report
    assert report["all_critical_present"] is True
    assert len(report["missing_critical"]) == 0
