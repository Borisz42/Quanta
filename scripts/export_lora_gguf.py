#!/usr/bin/env python3
"""GGUF Quantization Export Script for QUANTA S-Expression Transducer (Phase 6.4).

Fuses fine-tuned LoRA adapters with the base model and exports quantized GGUF
binaries (Q4_K_M, Q8_0) for local low-latency Unsloth Desktop and llama.cpp serving
at http://localhost:8888/v1.

Usage:
    python scripts/export_lora_gguf.py --adapter-dir models/quanta-slm-lora --dry-run
    python scripts/export_lora_gguf.py --adapter-dir models/quanta-slm-lora --quantization-methods q4_k_m,q8_0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure Windows stdout handles UTF-8 cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)

SUPPORTED_QUANT_METHODS = [
    "q4_k_m",
    "q8_0",
    "q5_k_m",
    "q4_0",
    "f16",
]


def export_gguf(
    adapter_dir: Union[str, Path] = "models/quanta-slm-lora",
    output_dir: Union[str, Path] = "models/quanta-slm-gguf",
    quantization_methods: Optional[Sequence[str]] = None,
    max_seq_length: int = 2048,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Fuses LoRA adapter weights and exports quantized GGUF models."""
    t0 = time.perf_counter()
    adapt_path = Path(adapter_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if quantization_methods is None:
        methods = ["q4_k_m", "q8_0"]
    else:
        methods = [m.strip().lower() for m in quantization_methods if m.strip()]

    # Validate quantization formats
    for m in methods:
        if m not in SUPPORTED_QUANT_METHODS:
            raise ValueError(
                f"Unsupported quantization method '{m}'. Supported methods: {SUPPORTED_QUANT_METHODS}"
            )

    result_summary: Dict[str, Any] = {
        "adapter_dir": str(adapt_path),
        "output_dir": str(out_path),
        "quantization_methods": methods,
        "max_seq_length": max_seq_length,
        "dry_run": dry_run,
        "exported_files": [],
        "status": "pending",
    }

    if dry_run:
        # Simulate export
        simulated_files = []
        for m in methods:
            simulated_file = out_path / f"quanta-slm-{m}.gguf"
            simulated_files.append(str(simulated_file))

        config_path = out_path / "export_config.json"
        result_summary["status"] = "dry_run_success"
        result_summary["simulated_files"] = simulated_files
        result_summary["elapsed_time"] = time.perf_counter() - t0

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(result_summary, f, indent=2)

        return result_summary

    # Ensure adapter path exists for actual export
    if not adapt_path.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapt_path}. Please run scripts/train_unsloth_lora.py first."
        )

    from unsloth import FastLanguageModel

    logger.info(f"Loading adapter from '{adapt_path}' for GGUF fusion...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapt_path),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,  # Float16 / BFloat16 required for clean weight fusion
    )

    exported_files = []
    for q_method in methods:
        logger.info(f"Fusing LoRA weights and exporting '{q_method}' GGUF binary...")
        target_name = f"quanta-slm-{q_method}.gguf"
        target_file = out_path / target_name
        model.save_pretrained_gguf(
            str(out_path),
            tokenizer,
            quantization_method=q_method,
        )
        exported_files.append(str(target_file))

    result_summary["status"] = "export_success"
    result_summary["exported_files"] = exported_files
    result_summary["elapsed_time"] = time.perf_counter() - t0

    config_path = out_path / "export_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)

    return result_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse LoRA adapters and export quantized GGUF models for QUANTA local serving (Phase 6.4)."
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="models/quanta-slm-lora",
        help="Directory containing trained LoRA adapters (default: models/quanta-slm-lora)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/quanta-slm-gguf",
        help="Destination directory for GGUF binaries (default: models/quanta-slm-gguf)",
    )
    parser.add_argument(
        "-q", "--quantization-methods",
        type=str,
        default="q4_k_m,q8_0",
        help="Comma-separated list of GGUF quantization formats (default: q4_k_m,q8_0)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform configuration check and simulation without loading model weights",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    methods = [m.strip().lower() for m in args.quantization_methods.split(",") if m.strip()]

    print("=" * 70)
    print("QUANTA LoRA to GGUF Quantization Exporter (Phase 6.4)")
    print("=" * 70)
    print(f"  Adapter Directory:    {args.adapter_dir}")
    print(f"  Output Directory:     {args.output_dir}")
    print(f"  Quantization Methods: {', '.join(methods)}")
    print(f"  Max Sequence Length:  {args.max_seq_length}")
    print(f"  Dry-Run Mode:         {args.dry_run}")
    print("=" * 70)

    try:
        res = export_gguf(
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            quantization_methods=methods,
            max_seq_length=args.max_seq_length,
            dry_run=args.dry_run,
        )
        print("\nGGUF export procedure finished successfully!")
        print(f"  Status:               {res.get('status')}")
        print(f"  Target formats:       {', '.join(res.get('quantization_methods', []))}")
        if args.dry_run:
            print("  Simulated files:")
            for f in res.get("simulated_files", []):
                print(f"    - {f}")
        else:
            print("  Exported files:")
            for f in res.get("exported_files", []):
                print(f"    - {f}")
        print(f"  Elapsed Time:         {res.get('elapsed_time', 0):.2f}s")
        print("=" * 70)
        return 0
    except Exception as e:
        logger.error(f"GGUF export failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
