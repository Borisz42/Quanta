"""Tests for Canonical Slot Layout (1024 Dimensions).

Verifies:
- 1024 canonical slot definitions and module-level integer constants
- No duplicate indices or names
- Strict 128-slot band partitioning (8 Bands: Bands 0 to 7)
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
    BAND_4_SLOTS,
    BAND_5_SLOTS,
    BAND_6_SLOTS,
    BAND_7_SLOTS,
    CANONICAL_SLOTS,
    SLOT_NAME_TO_INDEX,
    SLOT_INDEX_TO_NAME,
    get_slot_by_name,
    get_slot_by_index,
    get_slot_names,
    export_canonical_slots_layout,
)


def test_slot_count_and_uniqueness():
    """Verify exactly 1024 slots, unique indices, and unique names."""
    assert len(CANONICAL_SLOTS) == 1024
    assert len(SLOT_INDEX_TO_NAME) == 1024
    assert len(SLOT_NAME_TO_INDEX) >= 1024

    indices = [slot.index for slot in CANONICAL_SLOTS]
    assert indices == list(range(1024)), "Slot indices must be sequential from 0 to 1023"

    names = [slot.name for slot in CANONICAL_SLOTS]
    assert len(set(names)) == 1024, "All slot names must be unique"


def test_band_boundaries_and_sizes():
    """Verify each of the 8 bands contains exactly 128 slots within expected index intervals."""
    all_bands = [
        (BAND_0_SLOTS, 0, SlotBand.BAND_0_NSM_KINEMATICS),
        (BAND_1_SLOTS, 1, SlotBand.BAND_1_VALENCIES_TOPOLOGY),
        (BAND_2_SLOTS, 2, SlotBand.BAND_2_LOGIC_VARIABLES),
        (BAND_3_SLOTS, 3, SlotBand.BAND_3_ONTOLOGY_STRUCTURES),
        (BAND_4_SLOTS, 4, SlotBand.BAND_4_AFFORDANCES_OPERATIONS),
        (BAND_5_SLOTS, 5, SlotBand.BAND_5_TOM_PRAGMATICS),
        (BAND_6_SLOTS, 6, SlotBand.BAND_6_PROOF_DEONTICS),
        (BAND_7_SLOTS, 7, SlotBand.BAND_7_SPATIOTEMPORAL_CAUSAL),
    ]

    for band_slots, band_idx, expected_band in all_bands:
        assert len(band_slots) == 128, f"Band {band_idx} must contain 128 slots, got {len(band_slots)}"
        start_idx = band_idx * 128
        end_idx = start_idx + 128
        for i, s in enumerate(band_slots):
            assert s.index == start_idx + i
            assert s.band == expected_band
            assert start_idx <= s.index < end_idx


def test_module_level_slot_constants():
    """Verify that all 1024 slot names are exposed as module-level integer constants."""
    for slot in CANONICAL_SLOTS:
        assert hasattr(slots, slot.name), f"slots module missing constant {slot.name}"
        assert getattr(slots, slot.name) == slot.index

    # Test key constants from each band
    assert slots.NSM_I == 0
    assert slots.NSM_DO == 32
    assert slots.NSM_MAYBE == 63
    assert slots.VAL_X1_AGENT == 128
    assert slots.GRAPH_IMMUTABLE_HASH_LOCK == 197
    assert slots.QUANT_UNIVERSAL_FORALL == 256
    assert slots.VAR_SLOT_X0 == 280
    assert slots.CN_Q001_COMPUTING == 384
    assert slots.CN_Q129_LEAVE == 512
    assert slots.TOM_FIRST_ORDER_BELIEF == 640
    assert slots.EPIST_DIRECT_OBSERVATION == 768
    assert slots.SOLVER_CWA_CLOSED_WORLD == 832
    assert slots.TEMP_ALLEN_BEFORE == 896
    assert slots.MODEL_CHECK_PROBABILISTIC_PRISM == 1023

    # Test legacy aliases
    assert slots.TYPE_ANIMATE == slots.CN_Q011_ANIMAL
    assert slots.TYPE_HUMAN == slots.CN_Q015_PERSON
    assert slots.AFFORD_INCISED_CUTTING == slots.CN_Q108_CUT
    assert slots.WN_ANIMAL_FAUNA == slots.CN_Q011_ANIMAL


def test_get_slot_helpers():
    """Verify get_slot_by_name, get_slot_by_index, and get_slot_names."""
    s0 = get_slot_by_index(0)
    assert s0.name == "NSM_I"

    s_agent = get_slot_by_name("VAL_X1_AGENT")
    assert s_agent is not None
    assert s_agent.index == 128

    assert get_slot_by_name("NON_EXISTENT_SLOT") is None

    with pytest.raises(IndexError):
        get_slot_by_index(1024)

    with pytest.raises(IndexError):
        get_slot_by_index(-1)

    all_names = get_slot_names()
    assert len(all_names) == 1024
    assert all_names[0] == "NSM_I"
    assert all_names[1023] == "MODEL_CHECK_PROBABILISTIC_PRISM"


def test_canonical_slots_json_export(tmp_path):
    """Verify export_canonical_slots_layout generates valid JSON and matches definitions."""
    test_export_path = tmp_path / "canonical_slots_layout.json"
    result_path = export_canonical_slots_layout(test_export_path)

    assert result_path.exists()
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 1024
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

    assert len(data) == 1024
    for i, item in enumerate(data):
        assert item["index"] == i
        assert item["name"] == CANONICAL_SLOTS[i].name
