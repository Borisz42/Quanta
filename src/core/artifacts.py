"""QUANTA Central Artifact Registry & Pre-Flight Verification Gatekeeper.

Enforces strict fail-fast validation for all offline databases, semantic registries,
grammars, and datasets. Prevents partial execution and cryptic errors when required
data assets are missing locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


class ArtifactCriticality(Enum):
    CRITICAL = "critical"      # Must exist for runtime inference, parsing, or grounding
    OPTIONAL = "optional"      # Benchmark, evaluation, or auxiliary artifacts
    RAW_SOURCE = "raw_source"  # Raw dumps (e.g. ConceptNet 34M assertions csv.gz)


class MissingArtifactError(RuntimeError):
    """Raised when a required QUANTA runtime artifact is missing locally."""

    def __init__(
        self,
        rel_path: str,
        component: Optional[str] = None,
        remediation_hint: Optional[str] = None,
    ):
        self.rel_path = rel_path
        self.component = component or "QUANTA Runtime"
        hint = remediation_hint or (
            "Run 'python scripts/init_quanta.py' or '.\\scripts\\dev.ps1 init' "
            "to automatically download and initialize missing artifacts from Hugging Face (Borisz42/QUANTA)."
        )
        msg = (
            f"\n[QUANTA FAIL-FAST ERROR] Missing required artifact for {self.component}:\n"
            f"  Expected Path: {REPO_ROOT / rel_path}\n"
            f"  Remediation:   {hint}\n"
        )
        super().__init__(msg)


@dataclass(frozen=True)
class ArtifactSpec:
    rel_path: str
    category: str
    criticality: ArtifactCriticality
    description: str
    generator_script: Optional[str] = None
    min_size_bytes: int = 100

    @property
    def local_path(self) -> Path:
        return REPO_ROOT / self.rel_path

    @property
    def exists(self) -> bool:
        return self.local_path.exists() and self.local_path.stat().st_size >= self.min_size_bytes

    @property
    def size_bytes(self) -> int:
        return self.local_path.stat().st_size if self.local_path.exists() else 0


ARTIFACT_REGISTRY: List[ArtifactSpec] = [
    # 1. Slot Registries
    ArtifactSpec(
        "output/canonical_slots_layout.json",
        "Slot Registries",
        ArtifactCriticality.CRITICAL,
        "Canonical 1024-dimension slot definitions across 8 modular bands",
        generator_script="src/scripts/export_slots.py",
        min_size_bytes=50_000,
    ),
    ArtifactSpec(
        "data/conceptnet_slots.json",
        "Slot Registries",
        ArtifactCriticality.CRITICAL,
        "Canonical Band 3 & 4 ConceptNet 5.7.0 slot definitions",
        generator_script="scripts/concept_solver.py",
        min_size_bytes=20_000,
    ),
    # 2. Offline Databases & Codebooks
    ArtifactSpec(
        "data/conceptnet_offline.db",
        "Offline Databases",
        ArtifactCriticality.CRITICAL,
        "ConceptNet 5.7.0 403k-concept packed 256-byte quaternary vector SQLite DB",
        generator_script="scripts/concept_solver.py",
        min_size_bytes=100_000_000,
    ),
    ArtifactSpec(
        "data/concept_codebook.csv.gz",
        "Offline Databases",
        ArtifactCriticality.OPTIONAL,
        "ConceptNet 5.7.0 compressed concept codebook",
        generator_script="scripts/concept_solver.py",
        min_size_bytes=1_000_000,
    ),
    ArtifactSpec(
        "data/concept_codebook.bin",
        "Offline Databases",
        ArtifactCriticality.OPTIONAL,
        "Zero-copy memory-mapped quaternary concept codebook binary",
        generator_script="scripts/compile_mmap_codebook.py",
        min_size_bytes=10_000_000,
    ),
    ArtifactSpec(
        "data/concept_codebook_index.json",
        "Offline Databases",
        ArtifactCriticality.OPTIONAL,
        "Fast offset index for memory-mapped concept codebook",
        generator_script="scripts/compile_mmap_codebook.py",
        min_size_bytes=5_000_000,
    ),
    ArtifactSpec(
        "data/wordnet_offline.db",
        "Offline Databases",
        ArtifactCriticality.CRITICAL,
        "WordNet offline SQLite taxonomy database",
        generator_script="src/scripts/build_wordnet_cache.py",
        min_size_bytes=100_000,
    ),
    ArtifactSpec(
        "data/wordnet_offline.json",
        "Offline Databases",
        ArtifactCriticality.CRITICAL,
        "WordNet offline JSON synset/hypernym cache",
        generator_script="src/scripts/build_wordnet_cache.py",
        min_size_bytes=100_000,
    ),
    ArtifactSpec(
        "data/framenet_valency.json",
        "Offline Databases",
        ArtifactCriticality.CRITICAL,
        "FrameNet semantic roles and valency mappings",
        generator_script="src/scripts/export_framenet_valency.py",
        min_size_bytes=5_000,
    ),
    # 3. Grammars
    ArtifactSpec(
        "data/grammar/quanta_asg.gbnf",
        "Grammars",
        ArtifactCriticality.CRITICAL,
        "Context-free GBNF grammar for S-expression transduction",
        min_size_bytes=1_000,
    ),
    # 4. Multi-Domain Benchmark Datasets
    ArtifactSpec(
        "data/raw/folio_train.jsonl",
        "Raw Benchmarks",
        ArtifactCriticality.OPTIONAL,
        "FOLIO First-Order Logic train benchmark",
        min_size_bytes=50_000,
    ),
    ArtifactSpec(
        "data/raw/folio_val.jsonl",
        "Raw Benchmarks",
        ArtifactCriticality.OPTIONAL,
        "FOLIO First-Order Logic validation benchmark",
        min_size_bytes=20_000,
    ),
    ArtifactSpec(
        "data/raw/proofwriter_train.jsonl",
        "Raw Benchmarks",
        ArtifactCriticality.OPTIONAL,
        "ProofWriter logical deduction benchmark",
        min_size_bytes=100_000,
    ),
    ArtifactSpec(
        "data/raw/babi_train.jsonl",
        "Raw Benchmarks",
        ArtifactCriticality.OPTIONAL,
        "bAbI multi-hop QA & spatial reasoning benchmark",
        min_size_bytes=100_000,
    ),
    ArtifactSpec(
        "data/raw/clutrr_train.jsonl",
        "Raw Benchmarks",
        ArtifactCriticality.OPTIONAL,
        "CLUTRR kinship reasoning benchmark",
        min_size_bytes=50_000,
    ),
    ArtifactSpec(
        "data/raw/python_ast_samples.jsonl",
        "Raw Benchmarks",
        ArtifactCriticality.OPTIONAL,
        "Python Code AST semantic graphs",
        min_size_bytes=10_000,
    ),
    # 5. Validation Corpora
    ArtifactSpec(
        "data/validation_corpus/validation_corpus.jsonl",
        "Validation Corpus",
        ArtifactCriticality.OPTIONAL,
        "5,000-sample multi-domain validation corpus",
        generator_script="src/scripts/export_profiler_artifacts.py",
        min_size_bytes=500_000,
    ),
    ArtifactSpec(
        "data/validation_corpus/validation_corpus_tensors.npz",
        "Validation Corpus",
        ArtifactCriticality.OPTIONAL,
        "Packed quaternary tensor representations of validation corpus",
        generator_script="src/scripts/export_profiler_artifacts.py",
        min_size_bytes=100_000,
    ),
    # 6. Transducer Training & Repair Datasets
    ArtifactSpec(
        "data/training/sexpr_train.jsonl",
        "Training Datasets",
        ArtifactCriticality.OPTIONAL,
        "S-expression instruction tuning dataset for LoRA",
        generator_script="scripts/generate_sexpr_dataset.py",
        min_size_bytes=50_000,
    ),
    ArtifactSpec(
        "data/training/muc_repair_train.jsonl",
        "Training Datasets",
        ArtifactCriticality.OPTIONAL,
        "Synthetic MUC repair error-recovery conditioning dataset",
        generator_script="scripts/generate_muc_dataset.py",
        min_size_bytes=10_000,
    ),
    # 7. Profiler & Verification Artifacts
    ArtifactSpec(
        "output/candidate_pool.json",
        "Profiler Artifacts",
        ArtifactCriticality.OPTIONAL,
        "2,048-candidate feature pool for mRMR selection",
        generator_script="src/scripts/export_profiler_artifacts.py",
        min_size_bytes=50_000,
    ),
    ArtifactSpec(
        "output/optimal_dimensions.json",
        "Profiler Artifacts",
        ArtifactCriticality.OPTIONAL,
        "Top 1,024 mRMR optimal dimensions",
        generator_script="src/scripts/export_profiler_artifacts.py",
        min_size_bytes=50_000,
    ),
    ArtifactSpec(
        "output/optimal_1024_dimensions.json",
        "Profiler Artifacts",
        ArtifactCriticality.OPTIONAL,
        "Top 1,024 mRMR optimal dimensions (canonical duplicate)",
        generator_script="src/scripts/export_profiler_artifacts.py",
        min_size_bytes=50_000,
    ),
    ArtifactSpec(
        "output/verifier_report.json",
        "Profiler Artifacts",
        ArtifactCriticality.OPTIONAL,
        "Semantic vector verifier soundness & cycle-consistency report",
        generator_script="src/scripts/export_profiler_artifacts.py",
        min_size_bytes=50_000,
    ),
    ArtifactSpec(
        "output/information_profiler_report.json",
        "Profiler Artifacts",
        ArtifactCriticality.OPTIONAL,
        "Entropy, mutual information, and total correlation report",
        generator_script="src/scripts/export_profiler_artifacts.py",
        min_size_bytes=50_000,
    ),
    ArtifactSpec(
        "output/dimension_sweep_results.json",
        "Profiler Artifacts",
        ArtifactCriticality.OPTIONAL,
        "Dimension sweep empirical scaling results",
        generator_script="src/scripts/run_dimension_sweep.py",
        min_size_bytes=1_000,
    ),
    # 8. Translation Examples
    ArtifactSpec(
        "output/examples_eng_eng.json",
        "Translation Examples",
        ArtifactCriticality.OPTIONAL,
        "Deterministic English -> QuantaGraph -> English round-trip examples",
        generator_script="src/scripts/generate_translation_examples.py",
        min_size_bytes=5_000,
    ),
    ArtifactSpec(
        "output/complex_examples_eng_eng.json",
        "Translation Examples",
        ArtifactCriticality.OPTIONAL,
        "Complex narrative round-trip translation examples",
        generator_script="src/scripts/generate_complex_translation_examples.py",
        min_size_bytes=2_000,
    ),
    # 9. Raw Source Data
    ArtifactSpec(
        "conceptnet-assertions-5.7.0.csv.gz",
        "Raw Source Data",
        ArtifactCriticality.RAW_SOURCE,
        "ConceptNet 5.7.0 full 34M assertion source dump",
        min_size_bytes=100_000_000,
    ),
]


def normalize_rel_path(p: Union[str, Path]) -> str:
    norm = str(p).replace("\\", "/")
    if norm.startswith("./"):
        norm = norm[2:]
    return norm


def find_artifact_spec(rel_path: Union[str, Path]) -> Optional[ArtifactSpec]:
    target = normalize_rel_path(rel_path)
    for spec in ARTIFACT_REGISTRY:
        if spec.rel_path == target or spec.rel_path.endswith("/" + target) or target.endswith("/" + spec.rel_path):
            return spec
    return None


def resolve_artifact_path(rel_path: Union[str, Path]) -> Optional[Path]:
    """Resolves an artifact path across local repo, cwd, worktrees, or QUANTA_DATA_DIR."""
    target_rel = normalize_rel_path(rel_path)
    filename = Path(target_rel).name

    candidates: List[Path] = [
        REPO_ROOT / target_rel,
        Path("data") / filename,
        Path("output") / filename,
        REPO_ROOT / "data" / filename,
        REPO_ROOT / "output" / filename,
    ]

    env_data = os.environ.get("QUANTA_DATA_DIR")
    if env_data:
        candidates.append(Path(env_data) / filename)
        candidates.append(Path(env_data) / target_rel)

    # Worktree parent support
    try:
        git_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if git_dir:
            common_root = Path(git_dir).resolve().parent
            candidates.append(common_root / target_rel)
            candidates.append(common_root / "data" / filename)
    except Exception:
        pass

    for cand in candidates:
        if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
            return cand.resolve()

    return None


def require_artifacts(*rel_paths: Union[str, Path], component: Optional[str] = None) -> None:
    """Fail-fast gatekeeper. Verifies all required artifacts exist locally.

    Raises:
        MissingArtifactError: If any specified artifact is missing or empty.
    """
    for rp in rel_paths:
        target_str = normalize_rel_path(rp)
        resolved = resolve_artifact_path(target_str)
        if resolved is None:
            raise MissingArtifactError(target_str, component=component)


def audit_artifacts(critical_only: bool = False) -> Dict[str, Any]:
    """Audits all tracked artifacts and returns a comprehensive status dict."""
    results: List[Dict[str, Any]] = []
    missing_critical: List[str] = []

    for spec in ARTIFACT_REGISTRY:
        if critical_only and spec.criticality != ArtifactCriticality.CRITICAL:
            continue

        resolved = resolve_artifact_path(spec.rel_path)
        exists = resolved is not None
        size = resolved.stat().st_size if exists else 0

        if not exists and spec.criticality == ArtifactCriticality.CRITICAL:
            missing_critical.append(spec.rel_path)

        results.append({
            "rel_path": spec.rel_path,
            "category": spec.category,
            "criticality": spec.criticality.value,
            "description": spec.description,
            "exists": exists,
            "resolved_path": str(resolved) if exists else None,
            "size_bytes": size,
            "min_size_bytes": spec.min_size_bytes,
        })

    return {
        "all_critical_present": len(missing_critical) == 0,
        "missing_critical": missing_critical,
        "total_artifacts": len(results),
        "artifacts": results,
    }
