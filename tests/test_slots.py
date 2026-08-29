"""Tests for Canonical Slot Layout (Phase 2A).

Verifies:
- 256 canonical slot definitions and module-level integer constants
- No duplicate indices or names
- Strict 64-slot band partitioning (Bands 0, 1, 2, 3)
- JSON export integrity of output/canonical_slots_layout.json
"""

import json
from pathlib import Path
import pytest

import core.slots as slots
from core.slots import (
    SlotBand,
    SlotDefinition,
    BAND_0_SLOTS,
    BAND_1_SLOTS,
    BAND_2_SLOTS,
    BAND_3_SLOTS,
    CANONICAL_SLOTS,
    SLOT_NAME_TO_INDEX,
    SLOT_INDEX_TO_NAME,
    get_slot_by_name,
    get_slot_by_index,
    get_slot_names,
    export_canonical_slots_layout,
)


def test_slot_count_and_uniqueness():
    """Verify exactly 256 slots, unique indices, and unique names."""
    assert len(CANONICAL_SLOTS) == 256
    assert len(SLOT_NAME_TO_INDEX) == 256
    assert len(SLOT_INDEX_TO_NAME) == 256

    indices = [slot.index for slot in CANONICAL_SLOTS]
    assert indices == list(range(256)), "Slot indices must be sequential from 0 to 255"

    names = [slot.name for slot in CANONICAL_SLOTS]
    assert len(set(names)) == 256, "All slot names must be unique"


def test_band_boundaries_and_sizes():
    """Verify each band contains exactly 64 slots within expected index intervals."""
    assert len(BAND_0_SLOTS) == 64
    assert len(BAND_1_SLOTS) == 64
    assert len(BAND_2_SLOTS) == 64
    assert len(BAND_3_SLOTS) == 64

    for i, s in enumerate(BAND_0_SLOTS):
        assert s.index == i
        assert s.band == SlotBand.BAND_0_NSM_KINEMATICS
        assert 0 <= s.index < 64

    for i, s in enumerate(BAND_1_SLOTS):
        assert s.index == 64 + i
        assert s.band == SlotBand.BAND_1_VALENCIES_TOPOLOGY
        assert 64 <= s.index < 128

    for i, s in enumerate(BAND_2_SLOTS):
        assert s.index == 128 + i
        assert s.band == SlotBand.BAND_2_ONTOLOGY_MODALITY
        assert 128 <= s.index < 192

    for i, s in enumerate(BAND_3_SLOTS):
        assert s.index == 192 + i
        assert s.band == SlotBand.BAND_3_EPISTEMIC_METARULES
        assert 192 <= s.index < 256


def test_module_level_slot_constants():
    """Verify that all 256 slot names are exposed as module-level integer constants."""
    for slot in CANONICAL_SLOTS:
        assert hasattr(slots, slot.name), f"slots module missing constant {slot.name}"
        assert getattr(slots, slot.name) == slot.index

    # Test key constants from each band
    assert slots.NSM_I == 0
    assert slots.NSM_DO == 32
    assert slots.NSM_MAYBE == 63
    assert slots.VAL_X1_AGENT == 64
    assert slots.LJB_NA_NEGATION == 75
    assert slots.GRAPH_IMMUTABLE_HASH_LOCK == 127
    assert slots.TYPE_ANIMATE == 128
    assert slots.ROLE_AGENT_CAPABLE == 148
    assert slots.WN_RELATION_LINK == 191
    assert slots.EPIST_DIRECT_OBSERVATION == 192
    assert slots.SOLVER_CWA_CLOSED_WORLD == 208
    assert slots.SPATIAL_RCC_DISCONNECTED == 232
    assert slots.LOGIC_TEMPORAL_UNTIL_U == 255


def test_get_slot_helpers():
    """Verify get_slot_by_name, get_slot_by_index, and get_slot_names."""
    s0 = get_slot_by_index(0)
    assert s0.name == "NSM_I"

    s_agent = get_slot_by_name("VAL_X1_AGENT")
    assert s_agent is not None
    assert s_agent.index == 64

    assert get_slot_by_name("NON_EXISTENT_SLOT") is None

    with pytest.raises(IndexError):
        get_slot_by_index(256)

    with pytest.raises(IndexError):
        get_slot_by_index(-1)

    all_names = get_slot_names()
    assert len(all_names) == 256
    assert all_names[0] == "NSM_I"
    assert all_names[255] == "LOGIC_TEMPORAL_UNTIL_U"


def test_canonical_slots_json_export(tmp_path):
    """Verify export_canonical_slots_layout generates valid JSON and matches definitions."""
    test_export_path = tmp_path / "canonical_slots_layout.json"
    result_path = export_canonical_slots_layout(test_export_path)

    assert result_path.exists()
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 256
    for i, entry in enumerate(data):
        assert entry["index"] == i
        assert entry["name"] == CANONICAL_SLOTS[i].name
        assert entry["band_id"] == int(CANONICAL_SLOTS[i].band)
        assert entry["band_name"] == CANONICAL_SLOTS[i].band.name
        assert entry["category"] == CANONICAL_SLOTS[i].category
        assert entry["description"] == CANONICAL_SLOTS[i].description


def test_output_canonical_slots_layout_file_exists():
    """Verify output/canonical_slots_layout.json exists and is valid."""
    output_file = Path("output/canonical_slots_layout.json")
    assert output_file.exists(), "output/canonical_slots_layout.json must exist"

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 256
    for i, item in enumerate(data):
        assert item["index"] == i
        assert item["name"] == CANONICAL_SLOTS[i].name
