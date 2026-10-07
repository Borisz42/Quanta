"""Fixed 128-byte C-compatible binary struct layout and BinaryNodeTable for QUANTA SVM.

Implements QuantaSemanticNodeStruct with exact 128-byte layout (64-byte aligned),
NumPy vector-aligned structured dtypes, Belnap 4-valued lattice mapping,
and memory-mapped BinaryNodeTable for microsecond SIMD filtering.
"""

from __future__ import annotations

import ctypes
from enum import IntEnum
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np


class BelnapValue(IntEnum):
    """Belnap 4-valued logic lattice states (2-bit packed representation).
    
    Belnap lattice: B_4 = {00_2, 01_2, 10_2, 11_2}
    - 00_2 (0): Contradiction / Both (True and False) / Bottom
    - 01_2 (1): True / Epistemically Verified
    - 10_2 (2): False / Epistemically Refuted
    - 11_2 (3): Unknown / Neither / Maybe / Unverified
    """
    CONTRADICTION = 0b00  # 0
    TRUE = 0b01          # 1
    FALSE = 0b10         # 2
    UNKNOWN = 0b11       # 3

    @classmethod
    def from_str(cls, val: Optional[Union[str, int, "BelnapValue"]]) -> int:
        """Converts string or int to Belnap lattice integer."""
        if val is None:
            return cls.UNKNOWN
        if isinstance(val, int):
            return int(val) & 0x03
        norm = str(val).strip().upper()
        if norm in ("TRUE", "FACT", "1", "01", "T"):
            return cls.TRUE
        if norm in ("FALSE", "REFUTED", "2", "10", "F"):
            return cls.FALSE
        if norm in ("CONTRADICTION", "CONTRA", "BOTH", "BOTTOM", "0", "00"):
            return cls.CONTRADICTION
        if norm in ("UNKNOWN", "MAYBE", "NEITHER", "UNVERIFIED", "IRRELEVANT", "3", "11", "U"):
            return cls.UNKNOWN
        return cls.UNKNOWN

    @classmethod
    def to_str(cls, val: int) -> str:
        """Converts Belnap lattice integer to canonical name."""
        code = val & 0x03
        if code == cls.TRUE:
            return "TRUE"
        if code == cls.FALSE:
            return "FALSE"
        if code == cls.CONTRADICTION:
            return "CONTRADICTION"
        return "UNKNOWN"


class SpeechActIntent(IntEnum):
    """Band 5 Theory of Mind / Speech-Act Intent enumeration."""
    NONE = 0
    INFORMATIVE = 1      # Assertive / Informative statement
    DIRECTIVE = 2        # Command, request, instruction
    COMMISSIVE = 3       # Promise, commitment, obligation
    EXPRESSIVE = 4       # Evaluative, irony, emotional stance
    DECLARATIVE = 5      # Institutional declaration, naming

    @classmethod
    def from_str(cls, val: Optional[Union[str, int]]) -> int:
        if val is None:
            return cls.NONE
        if isinstance(val, int):
            return int(val) & 0xFF
        norm = str(val).strip().upper()
        if "INFORM" in norm or "ASSERT" in norm:
            return cls.INFORMATIVE
        if "DIRECT" in norm or "COMMAND" in norm or "REQUEST" in norm:
            return cls.DIRECTIVE
        if "COMMIS" in norm or "PROMISE" in norm or "OBLIGAT" in norm:
            return cls.COMMISSIVE
        if "EXPRESS" in norm or "IRONY" in norm or "SARCASM" in norm:
            return cls.EXPRESSIVE
        if "DECLAR" in norm:
            return cls.DECLARATIVE
        return cls.NONE

    @classmethod
    def from_node(cls, node: Any) -> int:
        """Extracts speech-act intent from a QuantaNode's active slots."""
        if hasattr(node, "get_slot"):
            if node.get_slot("INTENT_IRONY_SARCASM") == 1 or node.get_slot("ROLE_SARCASM_IRONY") == 1:
                return cls.EXPRESSIVE
            if node.get_slot("EPIST_DEONTIC_OBLIGATION") == 1:
                return cls.COMMISSIVE
            if node.get_slot("EPIST_DEONTIC_PROHIBITION") == 1:
                return cls.DIRECTIVE
            if node.get_slot("SOLVER_PROOF_VALIDATED") == 1 or node.get_slot("NSM_TRUE") == 1:
                return cls.INFORMATIVE
        return cls.INFORMATIVE


class EpistemicSource(IntEnum):
    """Band 6 Epistemic Evidence Source enumeration."""
    NONE = 0
    DIRECT_OBSERVATION = 1   # Direct empirical observation / visual / perceptual
    DEDUCTION = 2            # Logical derivation / proof / math
    HEARSAY = 3              # Indirect communication / quotation / testimony
    CONJECTURE = 4            # Speculation / hypothesis / assumption
    KNOWLEDGE = 5            # Established background knowledge
    FACT = 6                 # Verified objective ground truth
    HYPOTHESIS = 7           # Falsifiable research hypothesis

    @classmethod
    def from_str(cls, val: Optional[Union[str, int]]) -> int:
        if val is None:
            return cls.NONE
        if isinstance(val, int):
            return int(val) & 0xFF
        norm = str(val).strip().upper()
        if "DIRECT" in norm or "OBSERV" in norm:
            return cls.DIRECT_OBSERVATION
        if "DEDUC" in norm or "PROOF" in norm or "INFER" in norm:
            return cls.DEDUCTION
        if "HEARSAY" in norm or "REPORT" in norm or "TESTIMONY" in norm:
            return cls.HEARSAY
        if "CONJECT" in norm or "SPECULAT" in norm or "DOUBT" in norm:
            return cls.CONJECTURE
        if "KNOW" in norm:
            return cls.KNOWLEDGE
        if "FACT" in norm:
            return cls.FACT
        if "HYPOTH" in norm:
            return cls.HYPOTHESIS
        return cls.DIRECT_OBSERVATION


# ---------------------------------------------------------------------------
# 128-Byte C-Compatible Binary Struct Layout
# ---------------------------------------------------------------------------

class QuantaSemanticNodeStruct(ctypes.Structure):
    """128-byte C-compatible binary struct for microsecond SIMD filtering.
    
    Memory layout (128 bytes total, 64-byte cache-line aligned):
    - node_id: uint32 (4 bytes)
    - passage_id: uint32 (4 bytes)
    - span_start: uint16 (2 bytes)
    - span_end: uint16 (2 bytes)
    - concept_code: uint16 (2 bytes)
    - belnap_lattice: uint8 (1 byte: 00=Contra, 01=True, 10=False, 11=Unknown)
    - confidence: uint8 (1 byte: 0-255 -> 0.0-1.0)
    - intent_band5: uint8 (1 byte: Speech-act enum)
    - epistemic_band6: uint8 (1 byte: Evidence source enum)
    - outgoing_edges: uint32[4] (16 bytes)
    - pad: uint8[94] (94 bytes padding -> exactly 128 bytes total)
    """
    _pack_ = 1
    _fields_ = [
        ("node_id", ctypes.c_uint32),            # 4 bytes  (offset 0)
        ("passage_id", ctypes.c_uint32),         # 4 bytes  (offset 4)
        ("span_start", ctypes.c_uint16),         # 2 bytes  (offset 8)
        ("span_end", ctypes.c_uint16),           # 2 bytes  (offset 10)
        ("concept_code", ctypes.c_uint16),       # 2 bytes  (offset 12)
        ("belnap_lattice", ctypes.c_uint8),      # 1 byte   (offset 14)
        ("confidence", ctypes.c_uint8),          # 1 byte   (offset 15)
        ("intent_band5", ctypes.c_uint8),        # 1 byte   (offset 16)
        ("epistemic_band6", ctypes.c_uint8),     # 1 byte   (offset 17)
        ("outgoing_edges", ctypes.c_uint32 * 4), # 16 bytes (offset 18)
        ("pad", ctypes.c_uint8 * 94),            # 94 bytes (offset 34) -> 128 bytes
    ]

    NODE_SIZE = 128

    @property
    def confidence_float(self) -> float:
        """Returns confidence as a float in [0.0, 1.0]."""
        return float(self.confidence) / 255.0

    @confidence_float.setter
    def confidence_float(self, val: float):
        """Sets confidence from a float in [0.0, 1.0]."""
        self.confidence = int(round(max(0.0, min(1.0, float(val))) * 255))

    @property
    def belnap_status(self) -> str:
        """Returns Belnap lattice status as a canonical string."""
        return BelnapValue.to_str(self.belnap_lattice)

    @belnap_status.setter
    def belnap_status(self, val: Union[str, int, BelnapValue]):
        """Sets Belnap lattice state."""
        self.belnap_lattice = BelnapValue.from_str(val)

    @property
    def intent_name(self) -> str:
        """Returns speech-act intent enum name."""
        try:
            return SpeechActIntent(self.intent_band5).name
        except ValueError:
            return f"UNKNOWN_{self.intent_band5}"

    @property
    def epistemic_name(self) -> str:
        """Returns epistemic source enum name."""
        try:
            return EpistemicSource(self.epistemic_band6).name
        except ValueError:
            return f"UNKNOWN_{self.epistemic_band6}"

    @property
    def edges_list(self) -> List[int]:
        """Returns non-zero outgoing edge target node IDs."""
        return [int(target) for target in self.outgoing_edges if target != 0]

    def add_edge(self, target_node_id: int) -> bool:
        """Appends a target node ID to outgoing_edges if slot available.
        
        Returns:
            True if added, False if all 4 outgoing edge slots are already filled.
        """
        for i in range(4):
            if self.outgoing_edges[i] == 0:
                self.outgoing_edges[i] = int(target_node_id) & 0xFFFFFFFF
                return True
        return False

    def to_bytes(self) -> bytes:
        """Serializes struct to exactly 128 bytes."""
        return bytes(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> QuantaSemanticNodeStruct:
        """Deserializes struct from exactly 128 bytes."""
        if len(data) != cls.NODE_SIZE:
            raise ValueError(f"Expected {cls.NODE_SIZE} bytes for QuantaSemanticNodeStruct, got {len(data)}")
        return cls.from_buffer_copy(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes struct attributes to dictionary."""
        return {
            "node_id": int(self.node_id),
            "passage_id": int(self.passage_id),
            "span_start": int(self.span_start),
            "span_end": int(self.span_end),
            "concept_code": int(self.concept_code),
            "belnap_lattice": int(self.belnap_lattice),
            "belnap_status": self.belnap_status,
            "confidence": int(self.confidence),
            "confidence_float": round(self.confidence_float, 4),
            "intent_band5": int(self.intent_band5),
            "intent_name": self.intent_name,
            "epistemic_band6": int(self.epistemic_band6),
            "epistemic_name": self.epistemic_name,
            "outgoing_edges": self.edges_list,
        }

    @classmethod
    def create(
        cls,
        node_id: int = 0,
        passage_id: int = 0,
        span_start: int = 0,
        span_end: int = 0,
        concept_code: int = 0,
        belnap_lattice: int = BelnapValue.TRUE,
        confidence: int = 255,
        intent_band5: int = 0,
        epistemic_band6: int = 1,
    ) -> QuantaSemanticNodeStruct:
        """Fast microsecond node constructor."""
        s = cls()
        s.node_id = node_id
        s.passage_id = passage_id
        s.span_start = span_start
        s.span_end = span_end
        s.concept_code = concept_code
        s.belnap_lattice = belnap_lattice
        s.confidence = confidence
        s.intent_band5 = intent_band5
        s.epistemic_band6 = epistemic_band6
        return s

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuantaSemanticNodeStruct:
        """Constructs struct from dictionary."""
        node_id = int(data.get("node_id", 0))
        passage_id = int(data.get("passage_id", 0))
        span_start = int(data.get("span_start", 0))
        span_end = int(data.get("span_end", 0))
        concept_code = int(data.get("concept_code", 0))

        if "belnap_lattice" in data:
            belnap = int(data["belnap_lattice"]) & 0x03
        else:
            belnap = BelnapValue.from_str(data.get("belnap_status", "TRUE"))

        if "confidence_float" in data:
            conf_u8 = int(round(max(0.0, min(1.0, float(data["confidence_float"]))) * 255))
        elif "confidence" in data:
            conf_val = data["confidence"]
            if isinstance(conf_val, float) and 0.0 <= conf_val <= 1.0:
                conf_u8 = int(round(conf_val * 255))
            else:
                conf_u8 = int(conf_val) & 0xFF
        else:
            conf_u8 = 255

        intent = SpeechActIntent.from_str(data.get("intent_name", data.get("intent_band5", 0)))
        epistemic = EpistemicSource.from_str(data.get("epistemic_name", data.get("epistemic_band6", 1)))

        raw_edges = data.get("outgoing_edges", [])
        edge_array = (ctypes.c_uint32 * 4)()
        for i, edge in enumerate(raw_edges[:4]):
            edge_array[i] = int(edge) & 0xFFFFFFFF

        return cls(
            node_id=node_id,
            passage_id=passage_id,
            span_start=span_start,
            span_end=span_end,
            concept_code=concept_code,
            belnap_lattice=belnap,
            confidence=conf_u8,
            intent_band5=intent,
            epistemic_band6=epistemic,
            outgoing_edges=edge_array,
        )

    def __repr__(self) -> str:
        return (
            f"QuantaSemanticNode(node_id={self.node_id}, passage_id={self.passage_id}, "
            f"span=[{self.span_start}:{self.span_end}], belnap='{self.belnap_status}', "
            f"conf={self.confidence_float:.2f}, edges={self.edges_list})"
        )


# Class alias for master plan naming parity
QuantaSemanticNode = QuantaSemanticNodeStruct


# ---------------------------------------------------------------------------
# NumPy Structured Data Type
# ---------------------------------------------------------------------------

NODE_DTYPE = np.dtype([
    ("node_id", np.uint32),
    ("passage_id", np.uint32),
    ("span_start", np.uint16),
    ("span_end", np.uint16),
    ("concept_code", np.uint16),
    ("belnap_lattice", np.uint8),
    ("confidence", np.uint8),
    ("intent_band5", np.uint8),
    ("epistemic_band6", np.uint8),
    ("outgoing_edges", np.uint32, (4,)),
    ("pad", np.uint8, (94,)),
])

assert NODE_DTYPE.itemsize == 128, f"NumPy structured dtype must be 128 bytes, got {NODE_DTYPE.itemsize}"
assert ctypes.sizeof(QuantaSemanticNodeStruct) == 128, f"Struct size must be 128 bytes, got {ctypes.sizeof(QuantaSemanticNodeStruct)}"


# ---------------------------------------------------------------------------
# Binary Node Table
# ---------------------------------------------------------------------------

class BinaryNodeTable:
    """Memory-mapped table of 128-byte QuantaSemanticNode structs.
    
    Supports O(1) random access, zero-copy slice iteration, and vector-aligned SIMD scans.
    Backed in-memory by bytearray or on-disk by np.memmap.
    """

    def __init__(
        self,
        initial_capacity: int = 0,
        buffer: Optional[Union[bytearray, np.ndarray]] = None,
    ):
        if buffer is not None:
            if isinstance(buffer, np.ndarray):
                self._buffer = buffer
                self._is_mmap = True
            elif isinstance(buffer, (bytearray, bytes)):
                self._buffer = bytearray(buffer)
                self._is_mmap = False
            else:
                raise TypeError(f"Unsupported buffer type: {type(buffer)}")
        else:
            self._buffer = bytearray(initial_capacity * QuantaSemanticNodeStruct.NODE_SIZE)
            self._is_mmap = False

        self.cid_to_node_id: Dict[str, int] = {}
        self.node_id_to_cid: Dict[int, str] = {}
        self.passage_id_map: Dict[str, int] = {}

    def register_node_cid(self, node_id: int, cid: str):
        """Associates a 32-bit integer node_id with its 256-bit BLAKE3 CID."""
        self.cid_to_node_id[cid] = int(node_id)
        self.node_id_to_cid[int(node_id)] = cid

    def get_cid(self, node_id: int) -> Optional[str]:
        """Resolves CID from node_id."""
        return self.node_id_to_cid.get(int(node_id))

    def get_node_id(self, cid: str) -> Optional[int]:
        """Resolves node_id from CID."""
        return self.cid_to_node_id.get(cid)

    def __len__(self) -> int:
        """Returns the number of 128-byte node structs in the table."""
        if self._is_mmap:
            return len(self._buffer)
        return len(self._buffer) // QuantaSemanticNodeStruct.NODE_SIZE

    def __getitem__(self, index: Union[int, slice]) -> Union[QuantaSemanticNodeStruct, List[QuantaSemanticNodeStruct]]:
        """O(1) random access to a 128-byte node struct."""
        n = len(self)
        if isinstance(index, slice):
            indices = range(*index.indices(n))
            return [self[i] for i in indices]

        idx = int(index)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"BinaryNodeTable index {index} out of bounds for size {n}")

        offset = idx * QuantaSemanticNodeStruct.NODE_SIZE
        if self._is_mmap:
            node_bytes = self._buffer[idx].tobytes()
            return QuantaSemanticNodeStruct.from_bytes(node_bytes)
        return QuantaSemanticNodeStruct.from_buffer(self._buffer, offset)

    def __setitem__(self, index: int, node: Union[QuantaSemanticNodeStruct, bytes]):
        """O(1) in-place overwrite of a 128-byte node struct."""
        idx = int(index)
        n = len(self)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"BinaryNodeTable index {index} out of bounds for size {n}")

        offset = idx * QuantaSemanticNodeStruct.NODE_SIZE
        raw_bytes = bytes(node) if isinstance(node, QuantaSemanticNodeStruct) else bytes(node)
        if len(raw_bytes) != QuantaSemanticNodeStruct.NODE_SIZE:
            raise ValueError(f"Expected 128 bytes, got {len(raw_bytes)}")

        if self._is_mmap:
            rec = np.frombuffer(raw_bytes, dtype=NODE_DTYPE)[0]
            self._buffer[idx] = rec
        else:
            self._buffer[offset : offset + QuantaSemanticNodeStruct.NODE_SIZE] = raw_bytes

    def append(self, node: Union[QuantaSemanticNodeStruct, bytes, Dict[str, Any]]) -> int:
        """Appends a 128-byte node struct to the table.
        
        Returns:
            The 0-based integer index of the appended node.
        """
        if self._is_mmap:
            raise RuntimeError("Cannot append to a read-only or memory-mapped table directly; re-open in write mode")

        if isinstance(node, dict):
            node_struct = QuantaSemanticNodeStruct.from_dict(node)
            raw = bytes(node_struct)
        elif isinstance(node, QuantaSemanticNodeStruct):
            raw = bytes(node)
        elif isinstance(node, (bytes, bytearray)):
            if len(node) != QuantaSemanticNodeStruct.NODE_SIZE:
                raise ValueError(f"Expected 128 bytes, got {len(node)}")
            raw = bytes(node)
        else:
            raise TypeError(f"Unsupported node type: {type(node)}")

        idx = len(self)
        self._buffer.extend(raw)
        return idx

    def extend(self, nodes: Iterable[Union[QuantaSemanticNodeStruct, bytes, Dict[str, Any]]]) -> List[int]:
        """Appends multiple nodes to the table."""
        indices = []
        for n in nodes:
            indices.append(self.append(n))
        return indices

    def as_numpy(self) -> np.ndarray:
        """Returns zero-copy NumPy structured array view (dtype=NODE_DTYPE)."""
        if self._is_mmap:
            return self._buffer
        return np.frombuffer(self._buffer, dtype=NODE_DTYPE)

    def filter_by_passage(self, passage_id: int) -> np.ndarray:
        """Vectorized scan returning row indices matching passage_id."""
        arr = self.as_numpy()
        return np.where(arr["passage_id"] == np.uint32(passage_id))[0]

    def filter_by_belnap(self, belnap_lattice: Union[int, str, BelnapValue]) -> np.ndarray:
        """Vectorized scan returning row indices matching Belnap lattice code."""
        code = BelnapValue.from_str(belnap_lattice)
        arr = self.as_numpy()
        return np.where(arr["belnap_lattice"] == np.uint8(code))[0]

    def filter_by_confidence(self, min_confidence: Union[float, int]) -> np.ndarray:
        """Vectorized scan returning row indices with confidence >= threshold.
        
        Args:
            min_confidence: Either a float in [0.0, 1.0] or a uint8 in [0, 255].
        """
        if isinstance(min_confidence, float) and 0.0 <= min_confidence <= 1.0:
            threshold = int(round(min_confidence * 255))
        else:
            threshold = int(min_confidence) & 0xFF
        arr = self.as_numpy()
        return np.where(arr["confidence"] >= np.uint8(threshold))[0]

    def find_by_node_id(self, node_id: int) -> Optional[QuantaSemanticNodeStruct]:
        """Fast lookup of node struct by node_id."""
        arr = self.as_numpy()
        matches = np.where(arr["node_id"] == np.uint32(node_id))[0]
        if len(matches) == 0:
            return None
        return self[matches[0]]

    def to_bytes(self) -> bytes:
        """Returns raw bytes of the entire binary table."""
        if self._is_mmap:
            return self._buffer.tobytes()
        return bytes(self._buffer)

    @classmethod
    def from_bytes(cls, data: bytes) -> BinaryNodeTable:
        """Constructs an in-memory BinaryNodeTable from raw bytes."""
        if len(data) % QuantaSemanticNodeStruct.NODE_SIZE != 0:
            raise ValueError(
                f"Data length ({len(data)}) must be a multiple of {QuantaSemanticNodeStruct.NODE_SIZE}"
            )
        return cls(buffer=bytearray(data))

    def save(self, file_path: Union[str, Path]):
        """Persists the binary table to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.to_bytes())

    @classmethod
    def load(cls, file_path: Union[str, Path], mmap: bool = False) -> BinaryNodeTable:
        """Loads a BinaryNodeTable from disk.
        
        Args:
            file_path: File path to read from.
            mmap: If True, uses np.memmap for zero-copy read-only streaming.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"BinaryNodeTable file not found: {path}")

        file_size = path.stat().st_size
        if file_size % QuantaSemanticNodeStruct.NODE_SIZE != 0:
            raise ValueError(
                f"File size ({file_size}) must be a multiple of {QuantaSemanticNodeStruct.NODE_SIZE}"
            )

        if mmap:
            mmap_arr = np.memmap(path, dtype=NODE_DTYPE, mode="r")
            return cls(buffer=mmap_arr)
        else:
            with open(path, "rb") as f:
                data = f.read()
            return cls.from_bytes(data)

    def close(self):
        """Closes memory map or releases buffer resources."""
        if getattr(self, "_is_mmap", False) and hasattr(self, "_buffer") and self._buffer is not None:
            if hasattr(self._buffer, "_mmap") and self._buffer._mmap is not None:
                try:
                    self._buffer._mmap.close()
                except Exception:
                    pass
            self._buffer = None

    def __enter__(self) -> BinaryNodeTable:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


__all__ = [
    "BelnapValue",
    "SpeechActIntent",
    "EpistemicSource",
    "QuantaSemanticNodeStruct",
    "QuantaSemanticNode",
    "NODE_DTYPE",
    "BinaryNodeTable",
]
