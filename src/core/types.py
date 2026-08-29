"""Quaternary logic types and 256-dimensional vector representations for QUANTA."""

from __future__ import annotations
import enum
from typing import Dict, Iterable, List, Optional, Sequence, Union
import numpy as np
import torch

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


def pack_quaternary_array(arr: Union[Sequence[int], np.ndarray]) -> bytes:
    """Packs 256 quaternary values (in 0..3) into a 64-byte bytes payload.
    
    Each byte encodes 4 slots:
      - slot 4*i + 0: bits 0-1
      - slot 4*i + 1: bits 2-3
      - slot 4*i + 2: bits 4-5
      - slot 4*i + 3: bits 6-7
    """
    if len(arr) != 256:
        raise ValueError(f"Expected array of length 256, got {len(arr)}")
    
    buf = bytearray(64)
    for i in range(64):
        s0 = int(arr[4 * i + 0]) & 0x03
        s1 = int(arr[4 * i + 1]) & 0x03
        s2 = int(arr[4 * i + 2]) & 0x03
        s3 = int(arr[4 * i + 3]) & 0x03
        buf[i] = s0 | (s1 << 2) | (s2 << 4) | (s3 << 6)
    return bytes(buf)


def unpack_quaternary_bytes(data: bytes) -> np.ndarray:
    """Unpacks a 64-byte payload into a (256,) numpy array with values in {0, 1, 2, 3}."""
    if len(data) != 64:
        raise ValueError(f"Expected 64 bytes, got {len(data)}")
    
    out = np.empty(256, dtype=np.uint8)
    for i in range(64):
        b = data[i]
        out[4 * i + 0] = b & 0x03
        out[4 * i + 1] = (b >> 2) & 0x03
        out[4 * i + 2] = (b >> 4) & 0x03
        out[4 * i + 3] = (b >> 6) & 0x03
    return out


class QuantaVector:
    """256-dimensional quaternary semantic vector in {0, 1, 2, 3}^256."""

    __slots__ = ("_data",)

    def __init__(self, data: Optional[Union[Sequence[int], np.ndarray, torch.Tensor, bytes, Dict[Union[int, str], int]]] = None):
        if data is None:
            self._data = np.zeros(256, dtype=np.uint8)
        elif isinstance(data, bytes):
            self._data = unpack_quaternary_bytes(data)
        elif isinstance(data, dict):
            from core.slots import get_slot_by_name
            self._data = np.zeros(256, dtype=np.uint8)
            for k, val in data.items():
                if isinstance(k, str):
                    slot = get_slot_by_name(k)
                    if slot is None:
                        raise KeyError(f"Unknown slot name: {k}")
                    idx = slot.index
                else:
                    idx = int(k)
                if 0 <= idx < 256:
                    self._data[idx] = int(val) & 0x03
                else:
                    raise IndexError(f"Slot index {idx} out of range [0, 255]")
        elif isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy().astype(np.uint8).flatten()
            if len(arr) != 256:
                raise ValueError(f"Expected 256 elements in tensor, got {len(arr)}")
            self._data = arr & 0x03
        elif isinstance(data, np.ndarray):
            arr = data.astype(np.uint8).flatten()
            if len(arr) != 256:
                raise ValueError(f"Expected 256 elements in array, got {len(arr)}")
            self._data = arr & 0x03
        elif isinstance(data, (list, tuple, Sequence)):
            if len(data) != 256:
                raise ValueError(f"Expected 256 elements, got {len(data)}")
            self._data = np.array([int(v) & 0x03 for v in data], dtype=np.uint8)
        else:
            raise TypeError(f"Unsupported data type for QuantaVector: {type(data)}")

    @classmethod
    def zeros(cls) -> QuantaVector:
        """Returns a vector filled with 0 (IRRELEVANT)."""
        return cls()

    @classmethod
    def from_bytes(cls, data: bytes) -> QuantaVector:
        """Instantiates a QuantaVector from 64 packed bytes."""
        return cls(data)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> QuantaVector:
        """Instantiates a QuantaVector from a PyTorch tensor."""
        return cls(tensor)

    def to_bytes(self) -> bytes:
        """Packs the 256 quaternary slots into 64 bytes."""
        return pack_quaternary_array(self._data)

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        """Returns the vector as a numpy ndarray of shape (256,) with uint8 dtype."""
        return self._data.copy() if copy else self._data

    def to_tensor(self, device: Optional[Union[str, torch.device]] = None, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        """Converts the vector to a 1D PyTorch tensor."""
        t = torch.from_numpy(self._data).to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
        return t

    def get_band(self, band: int) -> np.ndarray:
        """Extracts a 64-dimensional slice for a specific band (0, 1, 2, or 3)."""
        if not (0 <= band <= 3):
            raise ValueError(f"Band index must be in [0, 3], got {band}")
        start = band * 64
        end = start + 64
        return self._data[start:end]

    def set_band(self, band: int, values: Union[Sequence[int], np.ndarray]):
        """Sets the 64 slots for a specific band."""
        if not (0 <= band <= 3):
            raise ValueError(f"Band index must be in [0, 3], got {band}")
        if len(values) != 64:
            raise ValueError(f"Expected 64 values for band {band}, got {len(values)}")
        start = band * 64
        end = start + 64
        self._data[start:end] = np.array([int(v) & 0x03 for v in values], dtype=np.uint8)

    def active_slots(self) -> Dict[int, QuaternaryValue]:
        """Returns non-zero (active) slot indices and their values."""
        active = {}
        for idx in range(256):
            val = self._data[idx]
            if val != 0:
                active[idx] = QuaternaryValue(val)
        return active

    def hamming_distance(self, other: QuantaVector) -> int:
        """Computes the Hamming distance (count of differing slots) between two vectors."""
        return int(np.count_nonzero(self._data != other._data))

    def similarity(self, other: QuantaVector) -> float:
        """Computes slot-wise agreement ratio in [0.0, 1.0]."""
        return 1.0 - (self.hamming_distance(other) / 256.0)

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
        if not (0 <= idx < 256):
            raise IndexError(f"Slot index {idx} out of range [0, 255]")
        return idx

    def __getitem__(self, key: Union[int, str]) -> QuaternaryValue:
        idx = self._resolve_slot_idx(key)
        return QuaternaryValue(int(self._data[idx]))

    def __setitem__(self, key: Union[int, str], value: Union[int, QuaternaryValue]):
        idx = self._resolve_slot_idx(key)
        self._data[idx] = int(value) & 0x03

    def __len__(self) -> int:
        return 256

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantaVector):
            return False
        return bool(np.array_equal(self._data, other._data))

    def __repr__(self) -> str:
        active = self.active_slots()
        active_str = ", ".join(f"{k}:{v.name}" for k, v in list(active.items())[:8])
        if len(active) > 8:
            active_str += f", ... ({len(active)} active)"
        return f"QuantaVector({active_str if active else 'EMPTY'})"
