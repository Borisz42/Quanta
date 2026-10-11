"""Offline Archetype Pre-Computation Compiler for ANVC OOV Grounding (§Session 3).

Embeds all 256 ConceptNet Band 3 and Band 4 taxonomic slot descriptions
from `data/conceptnet_slots.json` using EmbeddingGemma 2 into a contiguous
memory-mapped binary array `data/archetype_embeddings.bin` at the calibrated
Matryoshka dimension D (default D=64 from MultiScaleConfig).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config.multi_scale_config import MultiScaleConfig
from server.embedding_manager import EmbeddingManager, MockEmbeddingManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("precompute_archetypes")


def precompute_archetype_embeddings(
    slots_path: Path,
    output_bin_path: Path,
    meta_path: Optional[Path] = None,
    matryoshka_dim: Optional[int] = None,
    device: Optional[str] = None,
    allow_mock: bool = False,
    batch_size: int = 32,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generates and exports normalized archetype embeddings for all 256 ConceptNet slots."""
    cfg = MultiScaleConfig()
    dim = int(matryoshka_dim or cfg.get("embedding.matryoshka_dim", 64))
    dev = device or cfg.get("embedding.device", "cuda")

    if not slots_path.exists():
        raise FileNotFoundError(f"ConceptNet slots JSON not found at: {slots_path}")

    with open(slots_path, "r", encoding="utf-8") as f:
        slots_data = json.load(f)

    if len(slots_data) != 256:
        logger.warning("Expected 256 slot descriptions, found %d", len(slots_data))

    slot_names = [s.get("name", f"CN_Q{i+1:03d}") for i, s in enumerate(slots_data)]
    slot_texts = [s.get("description", s.get("name", "")) for s in slots_data]
    slot_indices = [int(s.get("index", 384 + i)) for i, s in enumerate(slots_data)]

    logger.info("Initializing embedding engine (dim=%d, device=%s, mock=%s)...", dim, dev, allow_mock)
    if allow_mock:
        manager = MockEmbeddingManager(matryoshka_dim=dim)
    else:
        manager = EmbeddingManager(
            backend="auto",
            matryoshka_dim=dim,
            device=dev,
            allow_mock=False,
        )

    t0 = time.perf_counter()
    logger.info("Embedding %d archetype slots with EmbeddingGemma 2...", len(slot_texts))
    matrix = manager.embed_documents(
        texts=slot_texts,
        titles=slot_names,
        matryoshka_dim=dim,
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - t0

    if matrix.shape != (len(slot_texts), dim):
        raise RuntimeError(
            f"Embedding matrix shape mismatch: expected ({len(slot_texts)}, {dim}), got {matrix.shape}"
        )

    # Ensure strictly float32 and contiguous
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    # Verify L2 normalization
    norms = np.linalg.norm(matrix, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        logger.warning("Archetype embeddings not strictly unit length. Re-normalizing...")
        matrix = matrix / np.maximum(norms[:, np.newaxis], 1e-12)

    # Export contiguous binary array
    output_bin_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.tofile(output_bin_path)
    file_bytes = output_bin_path.stat().st_size
    expected_bytes = len(slot_texts) * dim * 4

    logger.info(
        "Successfully wrote archetype embeddings binary to %s (%d bytes, %d slots x %d dims in %.2f s)",
        output_bin_path,
        file_bytes,
        len(slot_texts),
        dim,
        elapsed,
    )

    if file_bytes != expected_bytes:
        raise IOError(f"Written binary size {file_bytes} != expected {expected_bytes} bytes")

    meta: Dict[str, Any] = {
        "num_archetypes": len(slot_texts),
        "dim": dim,
        "model_id": getattr(manager, "model_id", "google/embeddinggemma-2"),
        "backend": getattr(manager, "active_backend_name", "unknown"),
        "device": getattr(manager, "device", dev),
        "dtype": "float32",
        "byte_size": file_bytes,
        "generated_at": datetime.now().isoformat(),
        "slots_path": str(slots_path),
        "output_bin_path": str(output_bin_path),
        "slot_indices": slot_indices,
        "slot_names": slot_names,
    }

    if meta_path:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved archetype metadata to %s", meta_path)

    return matrix, meta


def main():
    parser = argparse.ArgumentParser(
        description="Precompute EmbeddingGemma 2 archetype vectors for ConceptNet Bands 3 & 4."
    )
    parser.add_argument(
        "--slots",
        type=str,
        default=str(repo_root / "data" / "conceptnet_slots.json"),
        help="Path to conceptnet_slots.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(repo_root / "data" / "archetype_embeddings.bin"),
        help="Path to export contiguous float32 binary matrix",
    )
    parser.add_argument(
        "--meta-output",
        type=str,
        default=str(repo_root / "data" / "archetype_embeddings_meta.json"),
        help="Path to export metadata JSON",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Matryoshka truncation dimension D (default from MultiScaleConfig)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="Compute device for EmbeddingGemma 2 (default from MultiScaleConfig)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="live",
        choices=["live", "auto", "mock"],
        help="Execution mode (live forces real EmbeddingGemma 2, mock for CI double)",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow deterministic mock embedding engine for offline CI testing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )

    args = parser.parse_args()
    allow_mock = args.allow_mock or (args.mode == "mock")

    slots_path = Path(args.slots)
    output_bin = Path(args.output)
    meta_path = Path(args.meta_output) if args.meta_output else None

    precompute_archetype_embeddings(
        slots_path=slots_path,
        output_bin_path=output_bin,
        meta_path=meta_path,
        matryoshka_dim=args.dim,
        device=args.device,
        allow_mock=allow_mock,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
