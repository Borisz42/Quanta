"""Tests for Belnap four-valued lattice algebra and QuaternaryValue operations."""

import pytest
from src.core.types import QuaternaryValue

def test_knowledge_ordering():
    """Verify knowledge ordering: 0 ≤_k {1, 2} ≤_k 3."""
    q0 = QuaternaryValue.IRRELEVANT
    q1 = QuaternaryValue.TRUE
    q2 = QuaternaryValue.FALSE
    q3 = QuaternaryValue.UNKNOWN

    # 0 is bottom in knowledge order (least knowledge)
    assert q0.knowledge_le(q0)
    assert q0.knowledge_le(q1)
    assert q0.knowledge_le(q2)
    assert q0.knowledge_le(q3)

    # 1 and 2 are intermediate
    assert q1.knowledge_le(q1)
    assert q1.knowledge_le(q3)
    assert not q1.knowledge_le(q0)
    assert not q1.knowledge_le(q2)  # Incomparable in knowledge order

    assert q2.knowledge_le(q2)
    assert q2.knowledge_le(q3)
    assert not q2.knowledge_le(q0)
    assert not q2.knowledge_le(q1)  # Incomparable in knowledge order

    # 3 is top in knowledge order (most knowledge / conflict)
    assert q3.knowledge_le(q3)
    assert not q3.knowledge_le(q0)
    assert not q3.knowledge_le(q1)
    assert not q3.knowledge_le(q2)


def test_truth_ordering():
    """Verify truth ordering: 2 ≤_t {0, 3} ≤_t 1."""
    q0 = QuaternaryValue.IRRELEVANT
    q1 = QuaternaryValue.TRUE
    q2 = QuaternaryValue.FALSE
    q3 = QuaternaryValue.UNKNOWN

    # 2 (FALSE) is bottom in truth order
    assert q2.truth_le(q2)
    assert q2.truth_le(q0)
    assert q2.truth_le(q3)
    assert q2.truth_le(q1)

    # 0 and 3 are intermediate and incomparable
    assert q0.truth_le(q0)
    assert q0.truth_le(q1)
    assert not q0.truth_le(q2)
    assert not q0.truth_le(q3)

    assert q3.truth_le(q3)
    assert q3.truth_le(q1)
    assert not q3.truth_le(q2)
    assert not q3.truth_le(q0)

    # 1 (TRUE) is top in truth order
    assert q1.truth_le(q1)
    assert not q1.truth_le(q2)
    assert not q1.truth_le(q0)
    assert not q1.truth_le(q3)


def test_all_16_join_meet_combinations():
    """Test exact outputs for all 16 pair combinations of join and meet."""
    # Expected join truth table (knowledge LUB):
    #       0  1  2  3
    # 0  [  0, 1, 2, 3 ]
    # 1  [  1, 1, 3, 3 ]
    # 2  [  2, 3, 2, 3 ]
    # 3  [  3, 3, 3, 3 ]
    expected_join = {
        (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3,
        (1, 0): 1, (1, 1): 1, (1, 2): 3, (1, 3): 3,
        (2, 0): 2, (2, 1): 3, (2, 2): 2, (2, 3): 3,
        (3, 0): 3, (3, 1): 3, (3, 2): 3, (3, 3): 3,
    }

    # Expected meet truth table (knowledge GLB):
    #       0  1  2  3
    # 0  [  0, 0, 0, 0 ]
    # 1  [  0, 1, 0, 1 ]
    # 2  [  0, 0, 2, 2 ]
    # 3  [  0, 1, 2, 3 ]
    expected_meet = {
        (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0,
        (1, 0): 0, (1, 1): 1, (1, 2): 0, (1, 3): 1,
        (2, 0): 0, (2, 1): 0, (2, 2): 2, (2, 3): 2,
        (3, 0): 0, (3, 1): 1, (3, 2): 2, (3, 3): 3,
    }

    for (a, b), exp_j in expected_join.items():
        qa = QuaternaryValue(a)
        qb = QuaternaryValue(b)
        res_j = qa.join(qb)
        assert res_j == exp_j, f"Join failed for ({a}, {b}): got {res_j}, expected {exp_j}"
        assert (qa | qb) == exp_j

    for (a, b), exp_m in expected_meet.items():
        qa = QuaternaryValue(a)
        qb = QuaternaryValue(b)
        res_m = qa.meet(qb)
        assert res_m == exp_m, f"Meet failed for ({a}, {b}): got {res_m}, expected {exp_m}"
        assert (qa & qb) == exp_m


def test_lattice_algebraic_properties():
    """Verify commutativity, associativity, idempotency, and absorption across all 4 values."""
    values = [QuaternaryValue(i) for i in range(4)]

    # Idempotency: a ⊔ a = a, a ⊓ a = a
    for a in values:
        assert (a | a) == a
        assert (a & a) == a

    # Commutativity: a ⊔ b = b ⊔ a, a ⊓ b = b ⊓ a
    for a in values:
        for b in values:
            assert (a | b) == (b | a)
            assert (a & b) == (b & a)

    # Associativity: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c), (a ⊓ b) ⊓ c = a ⊓ (b ⊓ c)
    for a in values:
        for b in values:
            for c in values:
                assert ((a | b) | c) == (a | (b | c))
                assert ((a & b) & c) == (a & (b & c))

    # Absorption: a ⊔ (a ⊓ b) = a, a ⊓ (a ⊔ b) = a
    for a in values:
        for b in values:
            assert (a | (a & b)) == a
            assert (a & (a | b)) == a
