#!/usr/bin/env python3
"""S-Expression Dataset Generator CLI for QUANTA LoRA Fine-Tuning (Phase 6.1 & 6.2).

Generates instruction-tuning datasets of (Discourse, S-Expression) pairs across
FOLIO, ProofWriter, bAbI, CLUTRR, Python Code ASTs, and episodic narratives.
Includes 15% synthetic MUC repair conditioning examples.

Usage:
    python scripts/generate_sexpr_dataset.py --output data/training/sexpr_train.jsonl -n 1000
    python scripts/generate_sexpr_dataset.py --domains folio,babi,narrative -n 50 --muc-ratio 0.15
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate S-Expression instruction fine-tuning dataset for QUANTA LoRA (Phase 6)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/training/sexpr_train.jsonl",
        help="Target output JSONL path (default: data/training/sexpr_train.jsonl)",
    )
    parser.add_argument(
        "-n", "--samples-per-domain",
        type=int,
        default=1000,
        help="Number of samples to extract per domain (default: 1000)",
    )
    parser.add_argument(
        "-d", "--domains",
        type=str,
        default="folio,proofwriter,babi,clutrr,code_ast,narrative",
        help="Comma-separated list of domains (default: folio,proofwriter,babi,clutrr,code_ast,narrative)",
    )
    parser.add_argument(
        "-m", "--muc-ratio",
        type=float,
        default=0.15,
        help="Proportion of MUC repair error-recovery conditioning samples (default: 0.15)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional ceiling on total samples written",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip strict GBNF S-expression syntax validation during export",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw",
        help="Directory where raw benchmark datasets are cached (default: data/raw)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode using local cached files and rich synthetic domain templates",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    selected_domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    print("=" * 70)
    print("QUANTA S-Expression Dataset Generator (Phase 6)")
    print("=" * 70)
    print(f"  Target output:       {args.output}")
    print(f"  Samples per domain:  {args.samples_per_domain}")
    print(f"  Selected domains:    {', '.join(selected_domains)}")
    print(f"  MUC repair ratio:    {args.muc_ratio * 100:.1f}%")
    print(f"  Max samples ceiling: {args.max_samples or 'None'}")
    print(f"  Validate S-expr:     {not args.no_validate}")
    print(f"  Offline mode:        {args.offline}")
    print(f"  Random seed:         {args.seed}")
    print("=" * 70)

    t0 = time.perf_counter()
    generator = SExprDatasetGenerator(
        cache_dir=args.cache_dir,
        seed=args.seed,
        offline=args.offline,
    )

    summary = generator.generate_dataset(
        output_path=args.output,
        samples_per_domain=args.samples_per_domain,
        domains=selected_domains,
        muc_ratio=args.muc_ratio,
        max_samples=args.max_samples,
        validate_sexpr=not args.no_validate,
    )
    elapsed = time.perf_counter() - t0

    print("\nDataset generation completed successfully!")
    print(f"  Output file:        {summary['output_path']}")
    print(f"  Total valid lines:  {summary['total_samples']}")
    print(f"  MUC repair count:   {summary['muc_repair_count']}")
    print(f"  Syntax valid:       {summary['syntax_valid_count']}")
    print("  Per-domain breakdown:")
    for dom, count in summary["domain_counts"].items():
        print(f"    - {dom:15s}: {count:6d}")
    print(f"  Elapsed time:       {elapsed:.2f}s ({summary['total_samples']/elapsed:.1f} samples/sec)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
