"""Quaternary logic types and 1024-dimensional vector representations for QUANTA."""

from __future__ import annotations
import enum
from typing import Dict, Iterable, List, Optional, Sequence, Union
import numpy as np
import torch

DEFAULT_DIMENSION = 1024


class QuaternaryValue(enum.IntEnum):
    """Epistemic 4-valued logic states (FOUR).
    
    0 - IRRELEVANT (00_2): Absent, not applicable, default.
    1 - TRUE (01_2): Affirmed, verified true, positive existence.
    2 - FALSE (10_2): Negated, affirmed false, positive absence/contradiction.
    3 - UNKNOWN (11_2): Epistemic uncertainty, maybe, indeterminate.
    """
    IRRELEVANT = 0
    TRUE = 1
    FALSE = 2
    UNKNOWN = 3

    @classmethod
    def from_bool(cls, val: Optional[bool]) -> QuaternaryValue:
        if val is True:
            return cls.TRUE
        elif val is False:
            return cls.FALSE
        return cls.UNKNOWN

    def negate(self) -> QuaternaryValue:
        """Epistemic negation."""
        if self == QuaternaryValue.TRUE:
            return QuaternaryValue.FALSE
        elif self == QuaternaryValue.FALSE:
            return QuaternaryValue.TRUE
        return self

    def knowledge_le(self, other: Union[QuaternaryValue, int]) -> bool:
        """Knowledge ordering comparator: 0 ≤_k {1, 2} ≤_k 3.
        
        Returns True if self is less than or equal to other in information/knowledge order.
        """
        other_val = QuaternaryValue(other)
        if self == other_val or self == QuaternaryValue.IRRELEVANT:
            return True
        if other_val == QuaternaryValue.UNKNOWN:
            return True
        return False

    def truth_le(self, other: Union[QuaternaryValue, int]) -> bool:
        """Truth ordering comparator: 2 ≤_t {0, 3} ≤_t 1.
        
        Returns True if self is less than or equal to other in truth order.
        """
        other_val = QuaternaryValue(other)
        if self == other_val:
            return True
        if self == QuaternaryValue.FALSE:
            return True
        if other_val == QuaternaryValue.TRUE:
            return True
        return False

    def join(self, other: Union[QuaternaryValue, int]) -> QuaternaryValue:
        """Lattice join (⊔_k) operation in knowledge order (least upper bound).
        
        0 ⊔ x = x
        x ⊔ x = x
        1 ⊔ 2 = 3 (contradiction / overdetermined knowledge)
        3 ⊔ x = 3
        """
        other_val = QuaternaryValue(other)
        if self == other_val:
            return self
        if self == QuaternaryValue.IRRELEVANT:
            return other_val
        if other_val == QuaternaryValue.IRRELEVANT:
            return self
        return QuaternaryValue.UNKNOWN

    def meet(self, other: Union[QuaternaryValue, int]) -> QuaternaryValue:
        """Lattice meet (⊓_k) operation in knowledge order (greatest lower bound).
        
        3 ⊓ x = x
        x ⊓ x = x
        1 ⊓ 2 = 0 (no common information between TRUE and FALSE)
        0 ⊓ x = 0
        """
        other_val = QuaternaryValue(other)
        if self == other_val:
            return self
        if self == QuaternaryValue.UNKNOWN:
            return other_val
        if other_val == QuaternaryValue.UNKNOWN:
            return self
        return QuaternaryValue.IRRELEVANT

    def __or__(self, other: Union[QuaternaryValue, int]) -> QuaternaryValue:
        return self.join(other)

    def __and__(self, other: Union[QuaternaryValue, int]) -> QuaternaryValue:
        return self.meet(other)


_LATTICE_JOIN_TABLE = np.array([
    [0, 1, 2, 3],
    [1, 1, 3, 3],
    [2, 3, 2, 3],
    [3, 3, 3, 3]
], dtype=np.uint8)

_LATTICE_MEET_TABLE = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 2, 2],
    [0, 1, 2, 3]
], dtype=np.uint8)


def pack_quaternary_array(arr: Union[Sequence[int], np.ndarray]) -> bytes:
    """Packs quaternary values (in 0..3) into a packed bytes payload (4 slots per byte).
    
    Each byte encodes 4 slots:
      - slot 4*i + 0: bits 0-1
      - slot 4*i + 1: bits 2-3
      - slot 4*i + 2: bits 4-5
      - slot 4*i + 3: bits 6-7
    """
    n_slots = len(arr)
    if n_slots % 4 != 0:
        raise ValueError(f"Array length must be divisible by 4, got {n_slots}")
    
    num_bytes = n_slots // 4
    buf = bytearray(num_bytes)
    for i in range(num_bytes):
        s0 = int(arr[4 * i + 0]) & 0x03
        s1 = int(arr[4 * i + 1]) & 0x03
        s2 = int(arr[4 * i + 2]) & 0x03
        s3 = int(arr[4 * i + 3]) & 0x03
        buf[i] = s0 | (s1 << 2) | (s2 << 4) | (s3 << 6)
    return bytes(buf)


def unpack_quaternary_bytes(data: bytes) -> np.ndarray:
    """Unpacks a packed bytes payload into a numpy array with values in {0, 1, 2, 3} (4 slots per byte)."""
    num_slots = len(data) * 4
    out = np.empty(num_slots, dtype=np.uint8)
    for i in range(len(data)):
        b = data[i]
        out[4 * i + 0] = b & 0x03
        out[4 * i + 1] = (b >> 2) & 0x03
        out[4 * i + 2] = (b >> 4) & 0x03
        out[4 * i + 3] = (b >> 6) & 0x03
    return out


class QuantaVector:
    """Quaternary semantic vector in {0, 1, 2, 3}^D (default D = 1024)."""

    __slots__ = ("_data",)

    def __init__(
        self,
        data: Optional[Union[Sequence[int], np.ndarray, torch.Tensor, bytes, Dict[Union[int, str], int]]] = None,
        dim: Optional[int] = None,
    ):
        target_dim = dim if dim is not None else DEFAULT_DIMENSION

        if data is None:
            self._data = np.zeros(target_dim, dtype=np.uint8)
        elif isinstance(data, bytes):
            self._data = unpack_quaternary_bytes(data)
        elif isinstance(data, dict):
            from core.slots import get_slot_by_name
            self._data = np.zeros(target_dim, dtype=np.uint8)
            for k, val in data.items():
                if isinstance(k, str):
                    slot = get_slot_by_name(k)
                    if slot is None:
                        raise KeyError(f"Unknown slot name: {k}")
                    idx = slot.index
                else:
                    idx = int(k)
                if 0 <= idx < len(self._data):
                    self._data[idx] = int(val) & 0x03
                else:
                    raise IndexError(f"Slot index {idx} out of range [0, {len(self._data) - 1}]")
        elif isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy().astype(np.uint8).flatten()
            self._data = arr & 0x03
        elif isinstance(data, np.ndarray):
            arr = data.astype(np.uint8).flatten()
            self._data = arr & 0x03
        elif isinstance(data, (list, tuple, Sequence)):
            self._data = np.array([int(v) & 0x03 for v in data], dtype=np.uint8)
        else:
            raise TypeError(f"Unsupported data type for QuantaVector: {type(data)}")

    @classmethod
    def zeros(cls, dim: int = DEFAULT_DIMENSION) -> QuantaVector:
        """Returns a vector filled with 0 (IRRELEVANT)."""
        return cls(dim=dim)

    @classmethod
    def from_bytes(cls, data: bytes) -> QuantaVector:
        """Instantiates a QuantaVector from packed bytes."""
        return cls(data)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> QuantaVector:
        """Instantiates a QuantaVector from a PyTorch tensor."""
        return cls(tensor)

    def to_bytes(self) -> bytes:
        """Packs the quaternary slots into bytes."""
        return pack_quaternary_array(self._data)

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        """Returns the vector as a numpy ndarray of uint8 dtype."""
        return self._data.copy() if copy else self._data

    def to_tensor(self, device: Optional[Union[str, torch.device]] = None, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        """Converts the vector to a 1D PyTorch tensor."""
        t = torch.from_numpy(self._data).to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
        return t

    def get_band(self, band: int, slots_per_band: Optional[int] = None) -> np.ndarray:
        """Extracts a slice for a specific band index."""
        n_slots = len(self._data)
        if slots_per_band is None:
            # For 1024: 8 bands of 128 slots. For 256: 4 bands of 64 slots.
            if n_slots == 1024:
                slots_per_band = 128
            elif n_slots == 256:
                slots_per_band = 64
            else:
                slots_per_band = n_slots // 8 if n_slots % 8 == 0 else n_slots // 4

        num_bands = n_slots // slots_per_band
        if not (0 <= band < num_bands):
            raise ValueError(f"Band index must be in [0, {num_bands - 1}], got {band}")
        start = band * slots_per_band
        end = start + slots_per_band
        return self._data[start:end]

    def set_band(self, band: int, values: Union[Sequence[int], np.ndarray], slots_per_band: Optional[int] = None):
        """Sets the slots for a specific band."""
        n_slots = len(self._data)
        if slots_per_band is None:
            if n_slots == 1024:
                slots_per_band = 128
            elif n_slots == 256:
                slots_per_band = 64
            else:
                slots_per_band = n_slots // 8 if n_slots % 8 == 0 else n_slots // 4

        num_bands = n_slots // slots_per_band
        if not (0 <= band < num_bands):
            raise ValueError(f"Band index must be in [0, {num_bands - 1}], got {band}")
        if len(values) != slots_per_band:
            raise ValueError(f"Expected {slots_per_band} values for band {band}, got {len(values)}")
        start = band * slots_per_band
        end = start + slots_per_band
        self._data[start:end] = np.array([int(v) & 0x03 for v in values], dtype=np.uint8)

    def active_slots(self) -> Dict[int, QuaternaryValue]:
        """Returns non-zero (active) slot indices and their values."""
        active = {}
        for idx in range(len(self._data)):
            val = self._data[idx]
            if val != 0:
                active[idx] = QuaternaryValue(val)
        return active

    def hamming_distance(self, other: QuantaVector) -> int:
        """Computes the Hamming distance (count of differing slots) between two vectors."""
        min_len = min(len(self._data), len(other._data))
        diff = int(np.count_nonzero(self._data[:min_len] != other._data[:min_len]))
        len_diff = abs(len(self._data) - len(other._data))
        return diff + len_diff

    def similarity(self, other: QuantaVector) -> float:
        """Computes slot-wise agreement ratio in [0.0, 1.0]."""
        max_len = max(len(self._data), len(other._data))
        if max_len == 0:
            return 1.0
        return 1.0 - (self.hamming_distance(other) / float(max_len))

    def copy(self) -> QuantaVector:
        return QuantaVector(self._data.copy())

    def _resolve_slot_idx(self, key: Union[int, str]) -> int:
        if isinstance(key, str):
            from core.slots import get_slot_by_name
            sd = get_slot_by_name(key)
            if sd is None:
                raise KeyError(f"Unknown slot name: {key}")
            return sd.index
        idx = int(key)
        if not (0 <= idx < len(self._data)):
            raise IndexError(f"Slot index {idx} out of range [0, {len(self._data) - 1}]")
        return idx

    def __getitem__(self, key: Union[int, str]) -> QuaternaryValue:
        idx = self._resolve_slot_idx(key)
        return QuaternaryValue(int(self._data[idx]))

    def __setitem__(self, key: Union[int, str], value: Union[int, QuaternaryValue]):
        idx = self._resolve_slot_idx(key)
        self._data[idx] = int(value) & 0x03

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantaVector):
            return False
        return bool(np.array_equal(self._data, other._data))

    def join(self, other: QuantaVector) -> QuantaVector:
        """Computes element-wise lattice join (⊔_k) across all dimensions."""
        if not isinstance(other, QuantaVector):
            raise TypeError(f"Cannot join QuantaVector with {type(other)}")
        if len(self._data) != len(other._data):
            max_len = max(len(self._data), len(other._data))
            d1 = np.zeros(max_len, dtype=np.uint8)
            d2 = np.zeros(max_len, dtype=np.uint8)
            d1[:len(self._data)] = self._data
            d2[:len(other._data)] = other._data
            joined_data = _LATTICE_JOIN_TABLE[d1, d2]
        else:
            joined_data = _LATTICE_JOIN_TABLE[self._data, other._data]
        return QuantaVector(joined_data)

    def meet(self, other: QuantaVector) -> QuantaVector:
        """Computes element-wise lattice meet (⊓_k) across all dimensions."""
        if not isinstance(other, QuantaVector):
            raise TypeError(f"Cannot meet QuantaVector with {type(other)}")
        if len(self._data) != len(other._data):
            max_len = max(len(self._data), len(other._data))
            d1 = np.zeros(max_len, dtype=np.uint8)
            d2 = np.zeros(max_len, dtype=np.uint8)
            d1[:len(self._data)] = self._data
            d2[:len(other._data)] = other._data
            met_data = _LATTICE_MEET_TABLE[d1, d2]
        else:
            met_data = _LATTICE_MEET_TABLE[self._data, other._data]
        return QuantaVector(met_data)

    def __or__(self, other: QuantaVector) -> QuantaVector:
        return self.join(other)

    def __and__(self, other: QuantaVector) -> QuantaVector:
        return self.meet(other)

    def __hash__(self) -> int:
        return hash(self.to_bytes())

    def __repr__(self) -> str:
        active = self.active_slots()
        active_str = ", ".join(f"{k}:{v.name}" for k, v in list(active.items())[:8])
        if len(active) > 8:
            active_str += f", ... ({len(active)} active)"
        return f"QuantaVector(dim={len(self._data)}, {active_str if active else 'EMPTY'})"


__all__ = [
    "DEFAULT_DIMENSION",
    "QuaternaryValue",
    "QuantaVector",
    "pack_quaternary_array",
    "unpack_quaternary_bytes",
]
