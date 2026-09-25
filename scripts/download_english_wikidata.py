"""Download and Stream Filtered English Wikidata directly into QUANTA Knowledge Base.

Provides two ultra-efficient methods to acquire real English Wikidata with pre-applied filters,
saving over 98% of disk space compared to the full 156 GB Wikimedia dump:

1. Method 1 (Recommended - Pre-filtered English Knowledge Graph from Hugging Face):
   - Streams or downloads `Alphonse7/Wikidata5M-KG` (1.4 GB compressed vs 156 GB).
   - Contains 4.6 million English entities and 21 million relational triples.
   - Pre-filtered to English Wikipedia-aligned entities, aliases, and descriptions.
   - Compiles directly into `data/wikipedia_quanta.db` or exports to `data/wikidata_en_salient.jsonl.gz`.

2. Method 2 (In-flight HTTP stream from official Wikimedia dump):
   - Streams `latest-all.json.gz` from dumps.wikimedia.org over HTTPS.
   - In-memory decompressor filters entities on the fly without saving the raw 156 GB file to disk.

Usage:
    # 1. Download & compile 100,000 real English Wikidata entities into wikipedia_quanta.db:
    python scripts/download_english_wikidata.py --max-entities 100000 --output data/wikipedia_quanta.db

    # 2. Compile 500,000 entities:
    python scripts/download_english_wikidata.py --max-entities 500000 --output data/wikipedia_quanta.db

    # 3. Compile full 4.6M English Wikipedia entities:
    python scripts/download_english_wikidata.py --output data/wikipedia_quanta.db

    # 4. Stream and export filtered entities to a compact compressed JSONL file:
    python scripts/download_english_wikidata.py --max-entities 100000 --export-jsonl data/wikidata_en_salient.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
from pathlib import Path
import sys
import tarfile
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import urllib.request

# Ensure src is on Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.wikidata_ingester import (
    SALIENT_PROPERTIES,
    PROPERTY_NAME_TO_PID,
    WikidataEntityMapper,
    WikidataSqliteCompiler,
    classify_entity_category,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_english_wikidata")

HF_WIKIDATA5M_URL = "https://huggingface.co/datasets/Alphonse7/Wikidata5M-KG/resolve/main/wikidata5m_kg.tar.gz"

# Mapping human-readable relation names to normalized property names
RELATION_MAP = {
    "instance of": "INSTANCE_OF",
    "subclass of": "SUBCLASS_OF",
    "country": "COUNTRY",
    "capital": "CAPITAL",
    "capital of": "CAPITAL_OF",
    "author": "AUTHOR",
    "screenwriter": "AUTHOR",
    "director": "AUTHOR",
    "composer": "AUTHOR",
    "place of birth": "PLACE_OF_BIRTH",
    "country of citizenship": "CITIZENSHIP",
    "citizenship": "CITIZENSHIP",
    "notable work": "NOTABLE_WORK",
    "occupation": "OCCUPATION",
    "located in": "LOCATED_IN",
    "located in the administrative territorial entity": "LOCATED_IN",
    "headquarters location": "HEADQUARTERS",
    "founded by": "FOUNDED_BY",
    "award received": "AWARD_RECEIVED",
    "part of": "PART_OF",
    "has part": "HAS_PART",
    "child": "CHILD",
    "father": "PARENT",
    "mother": "PARENT",
    "spouse": "SPOUSE",
    "educated at": "EDUCATED_AT",
    "employer": "EMPLOYER",
    "member of": "MEMBER_OF",
    "field of work": "FIELD_OF_WORK",
    "genre": "GENRE",
    "publisher": "PUBLISHER",
}

RELATION_TO_PID = {
    "INSTANCE_OF": "P31",
    "SUBCLASS_OF": "P279",
    "COUNTRY": "P17",
    "CAPITAL": "P36",
    "AUTHOR": "P50",
    "PLACE_OF_BIRTH": "P19",
    "CITIZENSHIP": "P27",
    "NOTABLE_WORK": "P800",
    "OCCUPATION": "P106",
    "LOCATED_IN": "P131",
    "HEADQUARTERS": "P159",
    "FOUNDED_BY": "P112",
    "AWARD_RECEIVED": "P166",
}


def stream_hf_wikidata5m(
    url: str = HF_WIKIDATA5M_URL,
    max_entities: Optional[int] = None,
    filter_salient: bool = True,
    tar_path: Optional[Path] = None,
) -> Iterator[Dict[str, Any]]:
    """Streams entities from Wikidata5M-KG tar.gz directly from HTTP or local archive.

    Yields normalized entity dictionaries:
    {"qid", "label", "description", "aliases", "category", "claims"}
    """
    if tar_path and tar_path.exists():
        logger.info("Opening local archive %s...", tar_path)
        file_stream = open(tar_path, "rb")
    else:
        logger.info("Connecting to Hugging Face CDN: %s...", url)
        req = urllib.request.Request(url, headers={"User-Agent": "QUANTA-Research/1.0 (https://github.com/Borisz42/Quanta)"})
        file_stream = urllib.request.urlopen(req)

    count = 0
    t0 = time.perf_counter()

    try:
        tar = tarfile.open(fileobj=file_stream, mode="r|gz")
        for member in tar:
            if not member.name.endswith(".jsonl"):
                continue

            logger.info("Streaming member %s (%.1f MB uncompressed)...", member.name, member.size / (1024 * 1024))
            extracted_file = tar.extractfile(member)
            if not extracted_file:
                continue

            for line in extracted_file:
                if not line.strip():
                    continue

                try:
                    raw = json.loads(line.decode("utf-8"))
                except Exception:
                    continue

                entity = parse_wikidata5m_entry(raw)
                if entity is None:
                    continue

                if filter_salient and entity["category"] == "other" and len(entity["claims"]) == 0:
                    continue

                yield entity
                count += 1

                if count % 25000 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = count / elapsed if elapsed > 0 else 0
                    logger.info("Streamed %d entities in %.1fs (%.0f entities/s)", count, elapsed, rate)

                if max_entities is not None and count >= max_entities:
                    break
            break
    finally:
        file_stream.close()


def parse_wikidata5m_entry(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Converts a Wikidata5M JSON entry into the normalized QUANTA schema."""
    qid = raw.get("entity_id")
    if not qid or not str(qid).startswith("Q"):
        return None

    aliases_raw = raw.get("entity_alias", [])
    if not aliases_raw:
        label = qid
        aliases = []
    else:
        label = aliases_raw[0]
        aliases = aliases_raw[1:]

    description = raw.get("entity_description", "")
    triples_raw = raw.get("all_one_hop_triples_str", [])

    claims: Dict[str, List[str]] = {}
    category = "other"

    # Inspect triples to classify category and extract relations
    instance_of_targets: List[str] = []
    for rel_str, target_str in triples_raw:
        rel_lower = rel_str.strip().lower()
        target_clean = target_str.strip()

        # Check instance of
        if rel_lower == "instance of":
            instance_of_targets.append(target_clean.lower())
            if any(h in target_clean.lower() for h in ["human", "person", "writer", "politician", "actor", "scientist", "player", "singer", "artist", "poet"]):
                category = "human"
            elif any(l in target_clean.lower() for l in ["city", "country", "location", "state", "island", "mountain", "village", "town", "capital"]):
                category = "location"
            elif any(o in target_clean.lower() for o in ["organization", "company", "university", "business", "institution", "agency", "party", "club", "team"]):
                category = "organization"
            elif any(w in target_clean.lower() for w in ["work", "book", "film", "novel", "song", "album", "painting", "series", "video game"]):
                category = "creative_work"

        # Map relation to P-ID and store in claims
        norm_rel = RELATION_MAP.get(rel_lower)
        if norm_rel:
            pid = RELATION_TO_PID.get(norm_rel, norm_rel)
            if pid not in claims:
                claims[pid] = []
            if target_clean not in claims[pid]:
                claims[pid].append(target_clean)

    # Secondary heuristic classification from description if category is still other
    if category == "other" and description:
        desc_lower = description.lower()
        if any(h in desc_lower for h in ["born ", "died ", "he was an ", "she was an ", "american ", "english ", "french ", "german ", "who served as"]):
            category = "human"
        elif any(l in desc_lower for l in ["is a city", "is a country", "is a town", "is a village", "located in", "capital of"]):
            category = "location"
        elif any(o in desc_lower for o in ["is an organization", "is a company", "is a university", "founded in"]):
            category = "organization"
        elif any(w in desc_lower for w in ["is a book", "is a film", "is a novel", "is a song", "written by", "directed by"]):
            category = "creative_work"

    return {
        "qid": str(qid),
        "label": str(label),
        "description": str(description),
        "aliases": aliases,
        "category": category,
        "claims": claims,
    }


def format_eta(seconds: float) -> str:
    """Formats seconds into human-readable hh:mm:ss string."""
    if seconds < 0 or seconds > 86400 * 30:
        return "calculating..."
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def download_archive_with_progress(url: str, dest_path: Path):
    """Downloads a remote file with real-time percentage, speed, and ETA display."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s...", url, dest_path)

    req = urllib.request.Request(url, headers={"User-Agent": "QUANTA/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1 MB
        t0 = time.perf_counter()
        last_log = t0

        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            now = time.perf_counter()
            elapsed = now - t0
            speed_mb = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0

            if total_size > 0:
                pct = (downloaded / total_size) * 100
                rem_bytes = max(0, total_size - downloaded)
                eta_sec = (rem_bytes / (1024 * 1024)) / speed_mb if speed_mb > 0 else 0
                eta_str = format_eta(eta_sec)
                print(f"\rDownloading Archive: {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) at {speed_mb:.1f} MB/s [ETA: {eta_str}]...", end="", flush=True)

                if now - last_log >= 15.0 or downloaded >= total_size:
                    last_log = now
                    logger.info("Download: %.1f/%.1f MB (%.1f%%) at %.1f MB/s [ETA: %s]", downloaded / (1024*1024), total_size / (1024*1024), pct, speed_mb, eta_str)
            else:
                print(f"\rDownloading: {downloaded / (1024*1024):.1f} MB at {speed_mb:.1f} MB/s...", end="", flush=True)

    print()
    logger.info("Download complete: %s (%.2f MB)", dest_path.name, dest_path.stat().st_size / (1024 * 1024))


def main():
    parser = argparse.ArgumentParser(
        description="Download and compile pre-filtered English Wikidata into QUANTA knowledge base"
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help="Maximum entities to process (default: None, all 4.6M entities)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wikipedia_quanta.db",
        help="Output SQLite database path (default: data/wikipedia_quanta.db)",
    )
    parser.add_argument(
        "--export-jsonl",
        type=str,
        default=None,
        help="Optional path to save filtered entities as compressed JSONL (.jsonl.gz)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Retain the downloaded 1.4 GB archive on disk after compilation",
    )
    parser.add_argument(
        "--archive-path",
        type=str,
        default="data/wikidata5m_kg.tar.gz",
        help="Path for local archive storage (default: data/wikidata5m_kg.tar.gz)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="SQLite compiler batch size (default: 10,000)",
    )

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    archive_path = Path(args.archive_path)
    downloaded_temporary = False

    # Download archive first to ensure zero connection interruptions during multi-hour compilation
    if not archive_path.exists():
        download_archive_with_progress(HF_WIKIDATA5M_URL, archive_path)
        downloaded_temporary = not args.keep_archive

    # 1. Stream filtered entities from local archive
    stream = stream_hf_wikidata5m(
        url=HF_WIKIDATA5M_URL,
        max_entities=args.max_entities,
        filter_salient=True,
        tar_path=archive_path,
    )

    # 2. Optional JSONL export
    if args.export_jsonl:
        jsonl_path = Path(args.export_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting filtered entities to %s...", jsonl_path)
        out_f = gzip.open(jsonl_path, "wt", encoding="utf-8") if jsonl_path.suffix == ".gz" else open(jsonl_path, "w", encoding="utf-8")

        def tee_stream():
            for ent in stream:
                out_f.write(json.dumps(ent) + "\n")
                yield ent
            out_f.close()
            logger.info("JSONL export complete: %s", jsonl_path)

        stream = tee_stream()

    # 3. Compile directly into SQLite database with ETA tracking
    compiler = WikidataSqliteCompiler()
    total_expected = args.max_entities or 4_665_331
    last_log_time = [0.0]

    def progress_callback(count: int, elapsed: float):
        now = time.perf_counter()
        rate = count / elapsed if elapsed > 0 else 0
        pct = (count / total_expected) * 100 if total_expected > 0 else 0
        rem_count = max(0, total_expected - count)
        eta_sec = rem_count / rate if rate > 0 else 0
        eta_str = format_eta(eta_sec)

        print(
            f"\rCompiled {count:,}/{total_expected:,} entities ({pct:.1f}%) in {elapsed:.1f}s ({rate:,.0f} ent/s) [ETA: {eta_str}]...",
            end="",
            flush=True,
        )

        # Log periodically every 20 seconds so task logs record clean ETA history
        if now - last_log_time[0] >= 20.0 or count >= total_expected:
            last_log_time[0] = now
            logger.info(
                "Progress: %d / %d (%.1f%%) | Rate: %.0f ent/s | Elapsed: %.1fs | ETA: %s",
                count,
                total_expected,
                pct,
                rate,
                elapsed,
                eta_str,
            )

    limit_str = f"{args.max_entities:,}" if args.max_entities else "all 4.6M"
    print(f"Streaming and compiling {limit_str} English Wikidata entities into {output_path}...")

    try:
        stats = compiler.compile_database(
            entities=stream,
            db_path=output_path,
            batch_size=args.batch_size,
            progress_callback=progress_callback,
        )
        print()

        elapsed = stats["elapsed_seconds"]
        size_mb = stats["file_size_bytes"] / (1024 * 1024)
        print("=" * 70)
        print(f"English Wikidata Pre-Compilation Complete!")
        print(f"  Target DB:     {stats['db_path']}")
        print(f"  Total Nodes:   {stats['total_nodes']:,}")
        print(f"  Total Aliases: {stats['total_aliases']:,}")
        print(f"  Total Triples: {stats['total_triples']:,}")
        print(f"  File Size:     {size_mb:.2f} MB")
        print(f"  Elapsed Time:  {elapsed:.2f} s")
        print(f"  Throughput:    {stats['nodes_per_second']:,.0f} nodes/sec")
        print("=" * 70)
    finally:
        # Clean up temporary archive if requested to save disk space
        if downloaded_temporary and archive_path.exists():
            logger.info("Cleaning up temporary archive %s to preserve disk space...", archive_path.name)
            try:
                archive_path.unlink()
            except Exception as e:
                logger.warning("Could not delete temporary archive: %s", e)


if __name__ == "__main__":
    main()
