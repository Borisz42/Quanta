#!/usr/bin/env python3
"""QUANTA Artifact Gathering & Verification Suite.

Audits, gathers, and generates all offline databases, semantic registries,
benchmark datasets, validation corpora, and training files required to execute
the QUANTA neuro-symbolic Mentalese architecture and SLM pipelines.

Usage:
    python scripts/gather_artifacts.py
    python scripts/gather_artifacts.py --force-regenerate
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure Windows stdout handles UTF-8 cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


class ArtifactItem:
    def __init__(
        self,
        rel_path: str,
        category: str,
        description: str,
        generator_fn: Optional[str] = None,
        is_critical: bool = True,
    ):
        self.rel_path = rel_path
        self.category = category
        self.description = description
        self.generator_fn = generator_fn
        self.is_critical = is_critical
        self.path = REPO_ROOT / rel_path

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def size_bytes(self) -> int:
        if self.path.exists():
            return self.path.stat().st_size
        return 0

    @property
    def size_str(self) -> str:
        size = self.size_bytes
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.2f} GB"


REQUIRED_ARTIFACTS: List[ArtifactItem] = [
    # 1. Slot Registries
    ArtifactItem(
        "output/canonical_slots_layout.json",
        "Slot Registries",
        "Canonical 1024-dimension slot definitions across 8 bands",
        generator_fn="src/scripts/export_slots.py",
    ),
    ArtifactItem(
        "data/conceptnet_slots.json",
        "Slot Registries",
        "Canonical Band 3 & 4 ConceptNet 5.7.0 slot definitions",
        generator_fn="scripts/concept_solver.py",
    ),
    # 2. Offline Databases & Codebooks
    ArtifactItem(
        "data/conceptnet_offline.db",
        "Offline Databases",
        "ConceptNet 5.7.0 403k-concept packed 256-byte quaternary vector DB",
        generator_fn="scripts/concept_solver.py",
    ),
    ArtifactItem(
        "data/concept_codebook.csv.gz",
        "Offline Databases",
        "ConceptNet 5.7.0 compressed concept codebook",
        generator_fn="scripts/concept_solver.py",
    ),
    ArtifactItem(
        "data/wordnet_offline.db",
        "Offline Databases",
        "WordNet offline SQLite taxonomy database",
        generator_fn="src/scripts/build_wordnet_cache.py",
    ),
    ArtifactItem(
        "data/wordnet_offline.json",
        "Offline Databases",
        "WordNet offline JSON synset/hypernym cache",
        generator_fn="src/scripts/build_wordnet_cache.py",
    ),
    ArtifactItem(
        "data/framenet_valency.json",
        "Offline Databases",
        "FrameNet semantic roles and valency mappings",
        generator_fn="src/scripts/export_framenet_valency.py",
    ),
    ArtifactItem(
        "data/wikipedia_quanta.db",
        "Offline Databases",
        "Phase 10 Global Encyclopedic Knowledge Base (4.6M English Wikipedia entities SQLite DB)",
        generator_fn="scripts/download_english_wikidata.py",
        is_critical=False,
    ),
    # 3. Grammars
    ArtifactItem(
        "data/grammar/quanta_asg.gbnf",
        "Grammars",
        "Context-free GBNF grammar for S-expression transduction",
    ),
    # 4. Multi-Domain Benchmark Datasets
    ArtifactItem(
        "data/raw/folio_train.jsonl",
        "Raw Benchmarks",
        "FOLIO First-Order Logic train benchmark",
        generator_fn="download_raw_benchmarks",
    ),
    ArtifactItem(
        "data/raw/folio_val.jsonl",
        "Raw Benchmarks",
        "FOLIO First-Order Logic validation benchmark",
        generator_fn="download_raw_benchmarks",
    ),
    ArtifactItem(
        "data/raw/proofwriter_train.jsonl",
        "Raw Benchmarks",
        "ProofWriter logical deduction benchmark (Hugging Face)",
        generator_fn="download_raw_benchmarks",
    ),
    ArtifactItem(
        "data/raw/babi_train.jsonl",
        "Raw Benchmarks",
        "bAbI multi-hop QA & spatial reasoning (Hugging Face)",
        generator_fn="download_raw_benchmarks",
    ),
    ArtifactItem(
        "data/raw/clutrr_train.jsonl",
        "Raw Benchmarks",
        "CLUTRR kinship reasoning benchmark (Hugging Face)",
        generator_fn="download_raw_benchmarks",
    ),
    ArtifactItem(
        "data/raw/python_ast_samples.jsonl",
        "Raw Benchmarks",
        "Python Code AST semantic graphs",
        generator_fn="download_raw_benchmarks",
    ),
    # 5. Validation Corpora
    ArtifactItem(
        "data/validation_corpus/validation_corpus.jsonl",
        "Validation Corpus",
        "5,000-sample multi-domain validation corpus",
        generator_fn="src/scripts/export_profiler_artifacts.py",
    ),
    ArtifactItem(
        "data/validation_corpus/validation_corpus_tensors.npz",
        "Validation Corpus",
        "Packed quaternary tensor representations of validation corpus",
        generator_fn="src/scripts/export_profiler_artifacts.py",
    ),
    # 6. Transducer Training & Repair Datasets
    ArtifactItem(
        "data/training/sexpr_train.jsonl",
        "Training Datasets",
        "S-expression instruction tuning dataset for LoRA",
        generator_fn="scripts/generate_sexpr_dataset.py",
    ),
    ArtifactItem(
        "data/training/muc_repair_train.jsonl",
        "Training Datasets",
        "Synthetic MUC repair error-recovery conditioning dataset",
        generator_fn="scripts/generate_muc_dataset.py",
    ),
    # 7. Profiler & Verification Artifacts
    ArtifactItem(
        "output/candidate_pool.json",
        "Profiler Artifacts",
        "2,048-candidate feature pool for mRMR selection",
        generator_fn="src/scripts/export_profiler_artifacts.py",
    ),
    ArtifactItem(
        "output/optimal_dimensions.json",
        "Profiler Artifacts",
        "Top 1,024 mRMR optimal dimensions",
        generator_fn="src/scripts/export_profiler_artifacts.py",
    ),
    ArtifactItem(
        "output/optimal_1024_dimensions.json",
        "Profiler Artifacts",
        "Top 1,024 mRMR optimal dimensions (canonical duplicate)",
        generator_fn="src/scripts/export_profiler_artifacts.py",
    ),
    ArtifactItem(
        "output/verifier_report.json",
        "Profiler Artifacts",
        "Semantic vector verifier soundness & cycle-consistency report",
        generator_fn="src/scripts/export_profiler_artifacts.py",
    ),
    ArtifactItem(
        "output/information_profiler_report.json",
        "Profiler Artifacts",
        "Entropy, mutual information, and total correlation report",
        generator_fn="src/scripts/export_profiler_artifacts.py",
    ),
    ArtifactItem(
        "output/dimension_sweep_results.json",
        "Profiler Artifacts",
        "Dimension sweep empirical scaling results",
        generator_fn="src/scripts/run_dimension_sweep.py",
    ),
    # 8. Translation Graphs & Examples
    ArtifactItem(
        "output/examples_eng_eng.json",
        "Translation Examples",
        "Deterministic English -> QuantaGraph -> English round-trip examples",
        generator_fn="src/scripts/generate_translation_examples.py",
    ),
    ArtifactItem(
        "output/complex_examples_eng_eng.json",
        "Translation Examples",
        "Complex narrative round-trip translation examples",
        generator_fn="src/scripts/generate_complex_translation_examples.py",
    ),
]


def ensure_directory_structure():
    dirs = [
        "data/raw",
        "data/grammar",
        "data/validation_corpus",
        "data/virtual_page_table",
        "data/training",
        "output",
    ]
    for d in dirs:
        (REPO_ROOT / d).mkdir(parents=True, exist_ok=True)


def run_py_script(script_rel_path: str, *extra_args: str) -> bool:
    full_path = REPO_ROOT / script_rel_path
    cmd = [sys.executable, str(full_path)] + list(extra_args)
    print(f"[*] Running generator: {script_rel_path} {' '.join(extra_args)}...", flush=True)
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return res.returncode == 0


def gather_missing_artifacts(force: bool = False):
    ensure_directory_structure()

    # 1. Slot layout
    layout_item = next(a for a in REQUIRED_ARTIFACTS if a.rel_path == "output/canonical_slots_layout.json")
    if force or not layout_item.exists:
        run_py_script("src/scripts/export_slots.py")

    # 2. WordNet offline cache
    wn_db = next(a for a in REQUIRED_ARTIFACTS if a.rel_path == "data/wordnet_offline.db")
    if force or not wn_db.exists:
        run_py_script("src/scripts/build_wordnet_cache.py")

    # 3. FrameNet valency
    fn_val = next(a for a in REQUIRED_ARTIFACTS if a.rel_path == "data/framenet_valency.json")
    if force or not fn_val.exists:
        run_py_script("src/scripts/export_framenet_valency.py")

    # 4. Raw benchmark datasets
    raw_benchmarks = [a for a in REQUIRED_ARTIFACTS if a.category == "Raw Benchmarks"]
    if force or any(not a.exists for a in raw_benchmarks):
        print("[*] Downloading / caching raw benchmark datasets...", flush=True)
        try:
            from data.real_loader import RealDatasetLoader
            loader = RealDatasetLoader()
            loader.download_all_raw_datasets(samples_to_cache=1000)
        except Exception as e:
            print(f"[!] Warning: Failed to download raw benchmark datasets: {e}", flush=True)

    # 5. Profiler, validation corpus & optimal dimensions
    profiler_items = [a for a in REQUIRED_ARTIFACTS if a.category in ("Validation Corpus", "Profiler Artifacts")]
    if force or any(not a.exists for a in profiler_items if a.rel_path.startswith("output/candidate_pool")):
        run_py_script("src/scripts/export_profiler_artifacts.py")

    # 6. Translation examples
    trans_item = next(a for a in REQUIRED_ARTIFACTS if a.rel_path == "output/examples_eng_eng.json")
    if force or not trans_item.exists:
        run_py_script("src/scripts/generate_translation_examples.py")

    complex_item = next(a for a in REQUIRED_ARTIFACTS if a.rel_path == "output/complex_examples_eng_eng.json")
    if force or not complex_item.exists:
        run_py_script("src/scripts/generate_complex_translation_examples.py", "--offline")

    # 7. Transducer training datasets
    sexpr_item = next(a for a in REQUIRED_ARTIFACTS if a.rel_path == "data/training/sexpr_train.jsonl")
    if force or not sexpr_item.exists:
        run_py_script("scripts/generate_sexpr_dataset.py", "-n", "100", "--output", "data/training/sexpr_train.jsonl")

    muc_item = next(a for a in REQUIRED_ARTIFACTS if a.rel_path == "data/training/muc_repair_train.jsonl")
    if force or not muc_item.exists:
        run_py_script("scripts/generate_muc_dataset.py", "-n", "100", "--output", "data/training/muc_repair_train.jsonl")


def print_audit_report():
    print("\n" + "=" * 95)
    print(f" {'QUANTA Neuro-Symbolic Runtime Artifact Audit & Inventory':^93} ")
    print("=" * 95)
    print(f" {'Status':<8} | {'Artifact Path':<48} | {'Size':<10} | {'Category':<20}")
    print("-" * 95)

    all_present = True
    total_bytes = 0

    current_cat = ""
    for item in REQUIRED_ARTIFACTS:
        if item.category != current_cat:
            current_cat = item.category
            print(f"--- {current_cat} " + "-" * (89 - len(current_cat)))

        status = "[OK]" if item.exists else "[MISSING]"
        if not item.exists:
            all_present = False
        total_bytes += item.size_bytes
        print(f" {status:<8} | {item.rel_path:<48} | {item.size_str:<10} | {item.description[:35]}")

    print("=" * 95)
    size_mb = total_bytes / (1024 * 1024)
    print(f" Total Artifact Footprint: {size_mb:.2f} MB across {len(REQUIRED_ARTIFACTS)} tracked runtime artifacts.")

    # Environment check
    env_file = REPO_ROOT / ".env"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("\nEnvironment & Credentials:")
    if env_file.exists():
        print(f"  [OK] Local .env file detected at {env_file}")
    else:
        print("  [!] .env file missing. Create one from .env.example")

    if hf_token:
        masked = hf_token[:6] + "..." + hf_token[-4:] if len(hf_token) > 10 else "***"
        print(f"  [OK] Hugging Face Token active: {masked}")
    else:
        print("  [!] Hugging Face Token not detected in environment.")

    if all_present:
        print("\nAll generated artifacts are present and ready for execution.")
    else:
        print("\nSome artifacts are missing. Run: python scripts/gather_artifacts.py --force-regenerate")
    print("=" * 95 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and gather all QUANTA generated artifacts.")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force re-generation of exportable artifacts even if they exist",
    )
    args = parser.parse_args()

    ensure_directory_structure()
    gather_missing_artifacts(force=args.force_regenerate)
    print_audit_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
