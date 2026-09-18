#!/usr/bin/env python3
"""Synthetic MUC Repair Dataset Generator CLI for QUANTA (Phase 6.2).

Synthesizes error-injected S-expressions paired with corrected target S-expressions
conditioned on diagnostic hints ([REPAIR REQUEST]) from Clingo formal verification.

Covers:
- Ontological violations (abstract/inanimate entity acting as volitional agent)
- Temporal violations (inverted Allen intervals or conflicting ordering relations)
- Causal violations (simultaneous direct mechanism link and preventive block)

Usage:
    python scripts/generate_muc_dataset.py --output data/training/muc_repair_train.jsonl -n 500
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

# Ensure Windows stdout handles UTF-8 characters cleanly
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

from data.sexpr_dataset_generator import SExprDatasetGenerator
from parser.sexpr_parser import parse_sexpr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic MUC repair conditioning dataset for QUANTA LoRA (Phase 6.2)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/training/muc_repair_train.jsonl",
        help="Target output JSONL path (default: data/training/muc_repair_train.jsonl)",
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=500,
        help="Number of MUC repair examples to synthesize (default: 500)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip strict S-expression validation",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 70)
    print("QUANTA Synthetic MUC Repair Dataset Generator (Phase 6.2)")
    print("=" * 70)
    print(f"  Target output:       {args.output}")
    print(f"  Count to synthesize: {args.count}")
    print(f"  Validate S-expr:     {not args.no_validate}")
    print(f"  Random seed:         {args.seed}")
    print("=" * 70)

    t0 = time.perf_counter()
    generator = SExprDatasetGenerator(seed=args.seed)
    samples = generator.synthesize_muc_repair_samples(count=args.count)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            if not args.no_validate:
                try:
                    parse_sexpr(s["output"])
                    valid_count += 1
                except Exception as e:
                    logger.warning(f"Invalid S-expression syntax in sample {s['id']}: {e}")
                    continue
            else:
                valid_count += 1
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0

    print("\nMUC repair dataset generation completed successfully!")
    print(f"  Output file:        {out_path}")
    print(f"  Total valid lines:  {valid_count}")
    print(f"  Elapsed time:       {elapsed:.2f}s ({valid_count/elapsed:.1f} samples/sec)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
