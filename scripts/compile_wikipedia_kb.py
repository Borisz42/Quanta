"""CLI Utility to compile Wikidata dumps or 100k synthetic slices into wikipedia_quanta.db.

Usage:
    # Compile offline 100,000-entity synthetic slice for development and testing:
    python scripts/compile_wikipedia_kb.py --synthetic --num-entities 100000 --output data/wikipedia_quanta.db

    # Ingest a real Wikidata JSONL / compressed dump:
    python scripts/compile_wikipedia_kb.py --dump path/to/latest-all.json.gz --output data/wikipedia_quanta.db --max-entities 500000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

# Ensure src is on Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.wikidata_ingester import (
    WikidataSqliteCompiler,
    generate_synthetic_wikidata_slice,
    stream_wikidata_dump,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compile_wikipedia_kb")


def main():
    parser = argparse.ArgumentParser(
        description="Compile Wikidata dump or synthetic slice into QUANTA wikipedia_quanta.db"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic multi-hop slice instead of reading a file",
    )
    parser.add_argument(
        "--num-entities",
        type=int,
        default=100_000,
        help="Number of synthetic entities to generate (default: 100,000)",
    )
    parser.add_argument(
        "--dump",
        type=str,
        default=None,
        help="Path to real Wikidata JSON / JSONL / .gz / .bz2 dump file",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help="Maximum entities to process from dump",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wikipedia_quanta.db",
        help="Output SQLite database path (default: data/wikipedia_quanta.db)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Batch size for bulk insertion (default: 10,000)",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compiler = WikidataSqliteCompiler()

    def progress_callback(count: int, elapsed: float):
        rate = count / elapsed if elapsed > 0 else 0
        print(f"\rCompiled {count:,} entities in {elapsed:.1f}s ({rate:,.0f} entities/s)...", end="", flush=True)

    if args.synthetic or not args.dump:
        print(f"Generating and compiling {args.num_entities:,} synthetic Wikidata entities into {output_path}...")
        stream = generate_synthetic_wikidata_slice(num_entities=args.num_entities)
    else:
        print(f"Streaming and compiling entities from {args.dump} into {output_path}...")
        stream = stream_wikidata_dump(args.dump, max_entities=args.max_entities, filter_salient=True)

    t0 = time.perf_counter()
    stats = compiler.compile_database(
        entities=stream,
        db_path=output_path,
        batch_size=args.batch_size,
        progress_callback=progress_callback,
    )
    print()  # newline after progress

    elapsed = stats["elapsed_seconds"]
    size_mb = stats["file_size_bytes"] / (1024 * 1024)
    print("=" * 70)
    print(f"Compilation Complete!")
    print(f"  Target DB:     {stats['db_path']}")
    print(f"  Total Nodes:   {stats['total_nodes']:,}")
    print(f"  Total Aliases: {stats['total_aliases']:,}")
    print(f"  Total Triples: {stats['total_triples']:,}")
    print(f"  File Size:     {size_mb:.2f} MB")
    print(f"  Elapsed Time:  {elapsed:.2f} s")
    print(f"  Throughput:    {stats['nodes_per_second']:,.0f} nodes/sec")
    print("=" * 70)


if __name__ == "__main__":
    main()
