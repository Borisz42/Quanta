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
    assert slots.CN_Q129_LOGIC == 512
    assert slots.TOM_FIRST_ORDER_BELIEF == 640
    assert slots.EPIST_DIRECT_OBSERVATION == 768
    assert slots.SOLVER_CWA_CLOSED_WORLD == 832
    assert slots.TEMP_ALLEN_BEFORE == 896
    assert slots.MODEL_CHECK_PROBABILISTIC_PRISM == 1023

    # Test legacy aliases
    assert slots.TYPE_ANIMATE == slots.CN_Q011_ANIMAL
    assert slots.TYPE_HUMAN == slots.CN_Q007_PERSON
    assert slots.AFFORD_INCISED_CUTTING == slots.CN_Q074_CUT
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


def test_band_first_and_last_slot_indices():
    """2A.11: Verify exact first and last slot indices for each of the 8 bands."""
    band_boundaries = [
        (BAND_0_SLOTS,  0,   127,  "NSM_I",                    "PROP_PHASE_TRANSITION_TEMP"),
        (BAND_1_SLOTS,  128, 255,  "VAL_X1_AGENT",             "OS_IO_URING_RING_BUFFER"),
        (BAND_2_SLOTS,  256, 383,  "QUANT_UNIVERSAL_FORALL",   None),  # last name varies
        (BAND_3_SLOTS,  384, 511,  "CN_Q001_COMPUTING",        None),
        (BAND_4_SLOTS,  512, 639,  "CN_Q129_LOGIC",            None),
        (BAND_5_SLOTS,  640, 767,  "TOM_FIRST_ORDER_BELIEF",   None),
        (BAND_6_SLOTS,  768, 895,  "EPIST_DIRECT_OBSERVATION", None),
        (BAND_7_SLOTS,  896, 1023, "TEMP_ALLEN_BEFORE",        "MODEL_CHECK_PROBABILISTIC_PRISM"),
    ]
    for band_slots, expected_first, expected_last, expected_first_name, expected_last_name in band_boundaries:
        assert band_slots[0].index == expected_first, (
            f"First slot of band starting at {expected_first} should be {expected_first}, got {band_slots[0].index}"
        )
        assert band_slots[-1].index == expected_last, (
            f"Last slot of band ending at {expected_last} should be {expected_last}, got {band_slots[-1].index}"
        )
        if expected_first_name:
            assert band_slots[0].name == expected_first_name, (
                f"First slot name mismatch: expected {expected_first_name}, got {band_slots[0].name}"
            )
        if expected_last_name:
            assert band_slots[-1].name == expected_last_name, (
                f"Last slot name mismatch: expected {expected_last_name}, got {band_slots[-1].name}"
            )


def test_legacy_alias_bridge_integrity():
    """2A.9: Verify LEGACY_ONTOLOGY_ALIASES maps all known legacy constants to canonical CN_Q* slot names."""
    from core.slots import LEGACY_ONTOLOGY_ALIASES

    assert isinstance(LEGACY_ONTOLOGY_ALIASES, dict)
    assert len(LEGACY_ONTOLOGY_ALIASES) > 0

    # All alias values must be valid slot name strings resolvable to a SlotDefinition
    for alias_name, target_name in LEGACY_ONTOLOGY_ALIASES.items():
        assert isinstance(target_name, str), f"Alias {alias_name} must map to a slot name string"
        resolved = get_slot_by_name(target_name)
        assert resolved is not None, (
            f"Alias {alias_name} -> '{target_name}' does not resolve to any SlotDefinition"
        )
        # The resolved slot must be in Band 3 or Band 4 (CN_Q* slots are indices 384–639)
        assert 384 <= resolved.index <= 639, (
            f"Alias {alias_name} -> {target_name} (index {resolved.index}) must map to Band 3 or 4 (384-639)"
        )

    # Spot-check critical semantic aliases resolve to the right canonical names
    assert "TYPE_ANIMATE" in LEGACY_ONTOLOGY_ALIASES
    assert LEGACY_ONTOLOGY_ALIASES["TYPE_ANIMATE"] == "CN_Q011_ANIMAL"
    assert "TYPE_HUMAN" in LEGACY_ONTOLOGY_ALIASES
    assert LEGACY_ONTOLOGY_ALIASES["TYPE_HUMAN"] == "CN_Q007_PERSON"
    assert "TYPE_INANIMATE_PHYSICAL" in LEGACY_ONTOLOGY_ALIASES
    assert "ROLE_AGENT_CAPABLE" in LEGACY_ONTOLOGY_ALIASES
    assert "ROLE_SENTIENT" in LEGACY_ONTOLOGY_ALIASES

    # Module-level aliases should point to the same index as the canonical slot
    assert slots.TYPE_ANIMATE == slots.CN_Q011_ANIMAL
    assert slots.TYPE_HUMAN == slots.CN_Q007_PERSON
    assert slots.WN_ANIMAL_FAUNA == slots.CN_Q011_ANIMAL
    assert slots.WN_PERSON_HUMAN == slots.CN_Q007_PERSON



def test_no_duplicate_slot_indices_across_all_bands():
    """2A.11: Verify zero duplicate slot indices across the full 1024-slot canonical layout."""
    all_slots = BAND_0_SLOTS + BAND_1_SLOTS + BAND_2_SLOTS + BAND_3_SLOTS + \
                BAND_4_SLOTS + BAND_5_SLOTS + BAND_6_SLOTS + BAND_7_SLOTS

    assert len(all_slots) == 1024, f"Expected 1024 total slots, got {len(all_slots)}"

    all_indices = [s.index for s in all_slots]
    all_names = [s.name for s in all_slots]

    # No duplicate indices
    assert len(set(all_indices)) == 1024, "Duplicate slot indices detected across bands"
    # No duplicate names
    assert len(set(all_names)) == 1024, "Duplicate slot names detected across bands"
    # Indices are a complete range [0, 1023]
    assert set(all_indices) == set(range(1024)), "Slot indices must span exactly {0, 1, ..., 1023}"


def test_conceptnet_slots_json_sync():
    """5A.5: Verify data/conceptnet_slots.json exactly matches canonical Band 3 and Band 4 slots."""
    cn_slots_file = Path("data/conceptnet_slots.json")
    assert cn_slots_file.exists(), "data/conceptnet_slots.json must exist"

    with open(cn_slots_file, "r", encoding="utf-8") as f:
        cn_slots = json.load(f)

    assert len(cn_slots) == 256, f"Expected 256 ConceptNet slots, got {len(cn_slots)}"

    for i, item in enumerate(cn_slots):
        expected_idx = 384 + i
        assert item["index"] == expected_idx, f"Slot index mismatch at position {i}: expected {expected_idx}, got {item['index']}"
        canon_slot = CANONICAL_SLOTS[expected_idx]
        assert item["name"] == canon_slot.name, f"Slot name mismatch at {expected_idx}: {item['name']} != {canon_slot.name}"
        assert item["description"] == canon_slot.description, f"Description mismatch at {expected_idx}"
        expected_band = 3 if expected_idx < 512 else 4
        assert item["band"] == expected_band

