"""Tests for quaternary logic, vector bit-packing, and PyTorch interop."""

import numpy as np
import pytest
import torch

from core.types import (
    QuantaVector,
    QuaternaryValue,
    pack_quaternary_array,
    unpack_quaternary_bytes,
)
from core.slots import SlotBand, CANONICAL_SLOTS, get_slot_by_name, get_slot_by_index


def test_quaternary_enum_values():
    assert QuaternaryValue.IRRELEVANT == 0
    assert QuaternaryValue.TRUE == 1
    assert QuaternaryValue.FALSE == 2
    assert QuaternaryValue.UNKNOWN == 3

    assert QuaternaryValue.TRUE.negate() == QuaternaryValue.FALSE
    assert QuaternaryValue.FALSE.negate() == QuaternaryValue.TRUE
    assert QuaternaryValue.UNKNOWN.negate() == QuaternaryValue.UNKNOWN
    assert QuaternaryValue.IRRELEVANT.negate() == QuaternaryValue.IRRELEVANT


def test_packing_and_unpacking_roundtrip():
    # Generate random 1024 quaternary array
    rng = np.random.default_rng(123)
    orig_arr = rng.integers(0, 4, size=1024, dtype=np.uint8)

    packed_bytes = pack_quaternary_array(orig_arr)
    assert len(packed_bytes) == 256

    unpacked_arr = unpack_quaternary_bytes(packed_bytes)
    assert len(unpacked_arr) == 1024
    assert np.array_equal(orig_arr, unpacked_arr)


def test_quanta_vector_initialization_and_access():
    vec = QuantaVector.zeros()
    assert len(vec) == 1024
    assert all(vec[i] == QuaternaryValue.IRRELEVANT for i in range(1024))

    # Set by index
    vec[42] = QuaternaryValue.TRUE
    vec[100] = 2
    vec[200] = QuaternaryValue.UNKNOWN
    vec[800] = QuaternaryValue.TRUE

    assert vec[42] == QuaternaryValue.TRUE
    assert vec[100] == QuaternaryValue.FALSE
    assert vec[200] == QuaternaryValue.UNKNOWN
    assert vec[800] == QuaternaryValue.TRUE

    active = vec.active_slots()
    assert len(active) == 4
    assert active[42] == QuaternaryValue.TRUE
    assert active[100] == QuaternaryValue.FALSE
    assert active[200] == QuaternaryValue.UNKNOWN
    assert active[800] == QuaternaryValue.TRUE


def test_quanta_vector_pytorch_interop():
    vec = QuantaVector.zeros()
    vec[0] = 1
    vec[10] = 2
    vec[20] = 3
    vec[900] = 1

    t = vec.to_tensor(dtype=torch.uint8)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (1024,)
    assert t[0].item() == 1
    assert t[10].item() == 2
    assert t[20].item() == 3
    assert t[900].item() == 1

    # From tensor roundtrip
    vec2 = QuantaVector.from_tensor(t)
    assert vec == vec2


def test_band_slicing():
    vec = QuantaVector.zeros()
    # Set Band 0 (0-127)
    vec[0] = 1
    vec[127] = 2
    # Set Band 1 (128-255)
    vec[128] = 3
    # Set Band 2 (256-383)
    vec[256] = 1
    # Set Band 7 (896-1023)
    vec[896] = 2

    band0 = vec.get_band(0)
    assert len(band0) == 128
    assert band0[0] == 1
    assert band0[127] == 2

    band1 = vec.get_band(1)
    assert len(band1) == 128
    assert band1[0] == 3

    band2 = vec.get_band(2)
    assert len(band2) == 128
    assert band2[0] == 1

    band7 = vec.get_band(7)
    assert len(band7) == 128
    assert band7[0] == 2


def test_hamming_distance_and_similarity():
    v1 = QuantaVector.zeros()
    v2 = QuantaVector.zeros()

    assert v1.hamming_distance(v2) == 0
    assert v1.similarity(v2) == 1.0

    v1[10] = 1
    v1[20] = 2
    assert v1.hamming_distance(v2) == 2
    assert v1.similarity(v2) == 1.0 - (2.0 / 1024.0)


def test_canonical_slots_integrity():
    assert len(CANONICAL_SLOTS) == 1024
    for i in range(1024):
        slot = get_slot_by_index(i)
        assert slot.index == i
        assert get_slot_by_name(slot.name) == slot
        expected_band_idx = i // 128
        assert int(slot.band) == expected_band_idx
