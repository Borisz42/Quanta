"""Compile ConceptNet Codebook into Zero-Copy Memory-Mapped Binary Array and Index.

Ingests `data/concept_codebook.csv.gz` and exports:
1. `data/concept_codebook.bin`: Packed 256-byte binary vectors (N x 256 bytes = N x 32 uint64).
2. `data/concept_codebook_index.json`: Fast string offset table for O(1) index resolution.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compile_mmap_codebook")


def parse_concept_string(raw_concept: str) -> Tuple[str, Optional[str]]:
    """Parse concept string such as 'slang (n)' into ('slang', 'n')."""
    raw = raw_concept.strip()
    match = re.match(r"^(.*?)\s*\(([a-z]+)\)$", raw, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).lower()
    return raw, None


def generate_key_variants(raw_concept: str) -> List[str]:
    """Generate canonical variations for robust dictionary lookup."""
    lemma, pos = parse_concept_string(raw_concept)
    raw_clean = raw_concept.strip().lower()
    lemma_lower = lemma.lower()
    lemma_space = lemma_lower.replace("_", " ")
    lemma_under = lemma_lower.replace(" ", "_")

    variants: List[str] = [raw_clean]

    # Qualified variants with POS
    if pos:
        for lem in (lemma_space, lemma_under):
            variants.append(f"{lem} ({pos})")
            variants.append(f"cn:{lem} ({pos})")
            variants.append(f"cn:en:{lem} ({pos})")
            variants.append(f"{lem}:{pos}")

    # Unqualified variants
    for lem in (lemma_space, lemma_under):
        variants.append(lem)
        variants.append(f"cn:{lem}")
        variants.append(f"cn:en:{lem}")

    return list(dict.fromkeys(variants))


def compile_codebook(
    csv_gz_path: Path,
    output_bin_path: Path,
    output_index_path: Path,
    max_concepts: int = 50000,
) -> Tuple[int, int]:
    """Compile compressed CSV codebook to contiguous binary array and index table.

    Returns:
        Tuple of (num_concepts_compiled, num_index_keys).
    """
    logger.info("Reading up to %d concepts from %s...", max_concepts, csv_gz_path)
    t0 = time.time()

    df = pd.read_csv(csv_gz_path, nrows=max_concepts)
    n_concepts = len(df)
    logger.info("Read %d concepts in %.2f seconds.", n_concepts, time.time() - t0)

    concept_labels = [str(x).strip() for x in df.iloc[:, 0].values]

    # Extract 256 question values
    logger.info("Vectorizing 256 quaternary question values...")
    q_values = df.iloc[:, 1:257].values.astype(np.uint8)
    if q_values.shape[1] != 256:
        raise ValueError(f"Expected 256 question columns, got {q_values.shape[1]}")

    # Pack 256 values into 64 bytes (4 values per byte)
    s0 = q_values[:, 0::4] & 0x03
    s1 = q_values[:, 1::4] & 0x03
    s2 = q_values[:, 2::4] & 0x03
    s3 = q_values[:, 3::4] & 0x03
    band_bytes = (s0 | (s1 << 2) | (s2 << 4) | (s3 << 6)).astype(np.uint8)

    # 1024-D Quanta vector packed into 256 bytes (32 uint64s):
    # Slots 0..383 (384 slots) -> bytes 0..95 (96 zeros)
    # Slots 384..639 (256 slots) -> bytes 96..159 (band_bytes)
    # Slots 640..1023 (384 slots) -> bytes 160..255 (96 zeros)
    full_packed = np.zeros((n_concepts, 256), dtype=np.uint8)
    full_packed[:, 96:160] = band_bytes

    # Write binary file
    output_bin_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing binary codebook to %s...", output_bin_path)
    full_packed.tofile(output_bin_path)
    bin_size_mb = output_bin_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %.2f MB to %s.", bin_size_mb, output_bin_path)

    # Build Index Table
    logger.info("Building lookup index...")
    index_table: Dict[str, int] = {}
    for idx, raw_label in enumerate(concept_labels):
        variants = generate_key_variants(raw_label)
        for var in variants:
            if var not in index_table:
                index_table[var] = idx

    index_payload = {
        "version": 1,
        "num_concepts": n_concepts,
        "vector_bytes": 256,
        "num_slots": 1024,
        "concepts": concept_labels,
        "index": index_table,
    }

    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing index JSON to %s (%d keys)...", output_index_path, len(index_table))
    with open(output_index_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, separators=(",", ":"))

    idx_size_mb = output_index_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Codebook compilation complete: %d concepts, %d index keys (bin: %.2f MB, index: %.2f MB) in %.2fs.",
        n_concepts,
        len(index_table),
        bin_size_mb,
        idx_size_mb,
        time.time() - t0,
    )
    return n_concepts, len(index_table)


def main():
    parser = argparse.ArgumentParser(description="Compile ConceptNet Codebook into Zero-Copy Mmap Binary")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/concept_codebook.csv.gz"),
        help="Path to input concept_codebook.csv.gz",
    )
    parser.add_argument(
        "--output-bin",
        type=Path,
        default=Path("data/concept_codebook.bin"),
        help="Path to output concept_codebook.bin",
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        default=Path("data/concept_codebook_index.json"),
        help="Path to output concept_codebook_index.json",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=50000,
        help="Maximum concepts to compile (default: 50,000)",
    )
    args = parser.parse_args()

    compile_codebook(
        csv_gz_path=args.input,
        output_bin_path=args.output_bin,
        output_index_path=args.output_index,
        max_concepts=args.max_concepts,
    )


if __name__ == "__main__":
    main()
