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


def test_uniform_and_edge_case_vectors():
    """1B.6: Test zero vector, full-TRUE, full-FALSE, full-UNKNOWN, and mixed block vectors."""
    # Zero vector: all IRRELEVANT
    v_zero = QuantaVector.zeros()
    assert len(v_zero) == 1024
    assert all(v_zero[i] == QuaternaryValue.IRRELEVANT for i in range(1024))
    assert v_zero.to_bytes() == bytes(256)

    # Full TRUE vector
    v_true = QuantaVector([1] * 1024)
    assert all(v_true[i] == QuaternaryValue.TRUE for i in range(1024))
    # Round-trip serialization
    assert QuantaVector.from_bytes(v_true.to_bytes()) == v_true

    # Full FALSE vector
    v_false = QuantaVector([2] * 1024)
    assert all(v_false[i] == QuaternaryValue.FALSE for i in range(1024))
    assert QuantaVector.from_bytes(v_false.to_bytes()) == v_false

    # Full UNKNOWN vector
    v_unknown = QuantaVector([3] * 1024)
    assert all(v_unknown[i] == QuaternaryValue.UNKNOWN for i in range(1024))
    assert QuantaVector.from_bytes(v_unknown.to_bytes()) == v_unknown

    # Mixed block pattern: bands alternate between all values
    mixed = [i % 4 for i in range(1024)]
    v_mixed = QuantaVector(mixed)
    assert QuantaVector.from_bytes(v_mixed.to_bytes()) == v_mixed

    # Verify join with zero vector is identity: v join 0 = v
    assert v_true.join(v_zero) == v_true
    assert v_false.join(v_zero) == v_false
    assert v_unknown.join(v_zero) == v_unknown

    # Verify meet with UNKNOWN vector is identity: v meet UNKNOWN = v
    assert v_true.meet(v_unknown) == v_true
    assert v_false.meet(v_unknown) == v_false
    assert v_zero.meet(v_unknown) == v_zero


def test_hamming_metric_space_properties():
    """1B.7: Verify Hamming distance satisfies all three metric space axioms.

    Axiom 1 (Identity of indiscernibles): d(u, v) = 0 iff u == v
    Axiom 2 (Symmetry): d(u, v) = d(v, u)
    Axiom 3 (Triangle inequality): d(u, w) <= d(u, v) + d(v, w)
    """
    rng = np.random.default_rng(seed=42)

    # --- Axiom 1: Identity of Indiscernibles ---
    v_a = QuantaVector(rng.integers(0, 4, size=1024, dtype=np.uint8))
    v_b = QuantaVector(v_a.to_numpy())  # Identical copy

    # Same content -> distance 0
    assert v_a.hamming_distance(v_b) == 0, "d(u, v) must be 0 when u == v"
    assert v_a == v_b

    # Distance 0 implies equality (converse)
    v_c = QuantaVector(rng.integers(0, 4, size=1024, dtype=np.uint8))
    if v_a != v_c:
        assert v_a.hamming_distance(v_c) > 0, "d(u, v) > 0 when u != v"

    # Self-distance is always 0
    v_zero = QuantaVector.zeros()
    assert v_zero.hamming_distance(v_zero) == 0
    assert v_a.hamming_distance(v_a) == 0

    # Mutating a single slot gives distance exactly 1
    v_single = QuantaVector.zeros()
    v_offset = QuantaVector.zeros()
    v_offset[512] = 1
    assert v_single.hamming_distance(v_offset) == 1
    assert v_single != v_offset

    # --- Axiom 2: Symmetry ---
    for _ in range(10):
        u = QuantaVector(rng.integers(0, 4, size=1024, dtype=np.uint8))
        v = QuantaVector(rng.integers(0, 4, size=1024, dtype=np.uint8))
        assert u.hamming_distance(v) == v.hamming_distance(u), "Hamming distance must be symmetric"

    # Specific cross-value cases
    v1 = QuantaVector([1] * 1024)
    v2 = QuantaVector([2] * 1024)
    assert v1.hamming_distance(v2) == v2.hamming_distance(v1)

    v3 = QuantaVector([0] * 512 + [3] * 512)
    v4 = QuantaVector([1] * 256 + [2] * 256 + [0] * 512)
    assert v3.hamming_distance(v4) == v4.hamming_distance(v3)

    # --- Axiom 3: Triangle Inequality ---
    for _ in range(20):
        u = QuantaVector(rng.integers(0, 4, size=1024, dtype=np.uint8))
        v = QuantaVector(rng.integers(0, 4, size=1024, dtype=np.uint8))
        w = QuantaVector(rng.integers(0, 4, size=1024, dtype=np.uint8))
        d_uw = u.hamming_distance(w)
        d_uv = u.hamming_distance(v)
        d_vw = v.hamming_distance(w)
        assert d_uw <= d_uv + d_vw, (
            f"Triangle inequality violated: d(u,w)={d_uw} > d(u,v)={d_uv} + d(v,w)={d_vw}"
        )

    # Degenerate: u == w -> d(u, w)=0 <= d(u, v) + d(v, u) = 2*d(u, v)
    for _ in range(5):
        u = QuantaVector(rng.integers(0, 4, size=1024, dtype=np.uint8))
        v = QuantaVector(rng.integers(0, 4, size=1024, dtype=np.uint8))
        assert u.hamming_distance(u) <= u.hamming_distance(v) + v.hamming_distance(u)

    # --- Boundary / Adversarial Cases ---
    # Max distance: all TRUE vs all FALSE -> 1024 differing slots
    v_all_true = QuantaVector([1] * 1024)
    v_all_false = QuantaVector([2] * 1024)
    assert v_all_true.hamming_distance(v_all_false) == 1024

    # Partial overlap
    v_half_true = QuantaVector([1] * 512 + [0] * 512)
    v_half_false = QuantaVector([0] * 512 + [2] * 512)
    assert v_half_true.hamming_distance(v_half_false) == 1024
    assert v_half_true.hamming_distance(v_half_true) == 0

    # Similarity is 1 - (distance / dim)
    assert v_all_true.similarity(v_all_false) == 0.0
    assert v_all_true.similarity(v_all_true) == 1.0
    assert v_half_true.similarity(QuantaVector.zeros()) == 1.0 - (512.0 / 1024.0)
