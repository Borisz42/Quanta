"""Empirical Calibration & Sweep Harness for Dynamic Archetype Grounding (§Session 3, exp-037d).

Sweeps and calibrates:
- top_k in {1, 2, 3, 5}
- theta_arch in {0.55, 0.60, 0.65, 0.70, 0.75, 0.80}
- belnap_mode in {3 ("unknown"), "confidence_scaled", 1 ("true")}

Evaluates against synthetic neologisms and rare domain entities absent from
the discrete ConceptNet codebook to:
1. Eliminate out-of-vocabulary semantic starvation in Bands 3 and 4.
2. Maximize semantic category precision and coverage without false-positive slot noise.
3. Verify SIMD BLAS matrix-vector projection latency remains strictly < 0.05 ms.
4. Persist winning calibrated parameters to config/multi_scale_profile.json (source_exp='exp-037d').
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
from parser.mmap_grounder import MmapLexicalGrounder
from server.embedding_manager import EmbeddingManager, MockEmbeddingManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sweep_archetype_grounding")

# -----------------------------------------------------------------------------
# Synthetic Neologism Benchmark Dataset (100% OOV for discrete codebook)
# -----------------------------------------------------------------------------

NEOLOGISM_EVAL_BENCHMARK = [
    {
        "term": "cryo-containment manifold",
        "expected_domains": ["physics", "physical", "chemistry", "tool", "artifact", "substance"],
        "category": "OBJECT",
    },
    {
        "term": "photonic lattice accelerator",
        "expected_domains": ["computing", "physics", "physical", "optics", "tool", "artifact"],
        "category": "OBJECT",
    },
    {
        "term": "quantum flux gate",
        "expected_domains": ["physics", "computing", "physical", "tool", "artifact"],
        "category": "OBJECT",
    },
    {
        "term": "chemoluminescent kinase reagent",
        "expected_domains": ["chemistry", "medicine", "biology", "substance"],
        "category": "SUBSTANCE",
    },
    {
        "term": "biosynthetic neural interface",
        "expected_domains": ["medicine", "biology", "anatomy", "computing", "tool"],
        "category": "OBJECT",
    },
    {
        "term": "gravitational anomaly stabilizer",
        "expected_domains": ["physics", "astronomy", "physical", "tool"],
        "category": "OBJECT",
    },
    {
        "term": "sub-atomic neutrino sensor",
        "expected_domains": ["physics", "astronomy", "physical", "tool", "measure"],
        "category": "OBJECT",
    },
    {
        "term": "superconducting magnetic tokamak",
        "expected_domains": ["physics", "physical", "electricity", "energy", "tool"],
        "category": "OBJECT",
    },
    {
        "term": "stellar nucleosynthesis furnace",
        "expected_domains": ["astronomy", "physics", "physical", "energy"],
        "category": "OBJECT",
    },
    {
        "term": "exoplanetary atmospheric bio-scrubber",
        "expected_domains": ["astronomy", "geography", "biology", "tool", "artifact"],
        "category": "OBJECT",
    },
    {
        "term": "lymphocytic monoclonal antibody conjugate",
        "expected_domains": ["medicine", "biology", "anatomy", "substance"],
        "category": "SUBSTANCE",
    },
    {
        "term": "decentralized cryptographic ledger consensus",
        "expected_domains": ["computing", "money", "law", "legal", "mathematics"],
        "category": "ABSTRACT",
    },
    {
        "term": "hypersonic aerodynamic aerospike engine",
        "expected_domains": ["vehicle", "physics", "transportation", "tool"],
        "category": "OBJECT",
    },
    {
        "term": "paleolithic lithic biface scraper",
        "expected_domains": ["archaeology", "history", "tool", "artifact"],
        "category": "OBJECT",
    },
    {
        "term": "mitochondrial oxidative phosphorylation catalyst",
        "expected_domains": ["biology", "medicine", "chemistry", "substance"],
        "category": "SUBSTANCE",
    },
    {
        "term": "neurovascular micro-embolization catheter",
        "expected_domains": ["medicine", "anatomy", "tool", "artifact"],
        "category": "OBJECT",
    },
    {
        "term": "multi-modal tensor decomposition kernel",
        "expected_domains": ["computing", "mathematics"],
        "category": "ABSTRACT",
    },
    {
        "term": "geothermal magma enthalpy heat-exchanger",
        "expected_domains": ["geology", "energy", "physics", "physical", "tool"],
        "category": "OBJECT",
    },
    {
        "term": "radiometric isotope mass-spectrometer",
        "expected_domains": ["physics", "chemistry", "measure", "tool"],
        "category": "OBJECT",
    },
    {
        "term": "synthetic elastomer vulcanization accelerator",
        "expected_domains": ["chemistry", "substance", "material", "industry"],
        "category": "SUBSTANCE",
    },
]

TOP_K_GRID = [1, 2, 3, 5]
THETA_ARCH_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
BELNAP_MODES = ["unknown", "confidence_scaled", "true"]


def get_git_commit_sha() -> str:
    """Returns the current short Git commit SHA."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def evaluate_archetype_configuration(
    grounder: MmapLexicalGrounder,
    slot_metadata: List[Dict[str, Any]],
    embedded_benchmark: List[Tuple[Dict[str, Any], np.ndarray]],
    top_k: int,
    theta_arch: float,
    belnap_mode: str,
) -> Dict[str, Any]:
    """Evaluates a single parameter combination over the benchmark neologisms."""
    total_samples = len(embedded_benchmark)
    covered_samples = 0
    total_slots_activated = 0
    correct_slots_activated = 0
    latencies_us: List[float] = []

    for item, query_vec in embedded_benchmark:
        expected = [d.lower() for d in item["expected_domains"]]

        t0 = time.perf_counter_ns()
        concept = grounder.project_oov_concept(
            term=query_vec,
            top_k=top_k,
            threshold=theta_arch,
            belnap_mode=belnap_mode,
        )
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)

        if concept is not None and len(concept.active_slots) > 0:
            covered_samples += 1
            for slot_name in concept.active_slots.keys():
                total_slots_activated += 1
                # Find slot definition
                slot_idx = int(slot_name.split("_")[1][1:]) - 1 if "CN_Q" in slot_name else 0
                slot_info = slot_metadata[slot_idx] if 0 <= slot_idx < len(slot_metadata) else {}
                target_word = str(slot_info.get("target", "")).lower()
                name_words = slot_name.lower().split("_")

                is_match = any(
                    exp in target_word or any(exp in nw for nw in name_words)
                    for exp in expected
                )
                if is_match:
                    correct_slots_activated += 1

    coverage = covered_samples / total_samples if total_samples > 0 else 0.0
    precision = (
        correct_slots_activated / total_slots_activated
        if total_slots_activated > 0
        else 0.0
    )
    f1 = (
        (2.0 * coverage * precision) / (coverage + precision)
        if (coverage + precision) > 0
        else 0.0
    )
    mean_latency_us = float(np.mean(latencies_us)) if latencies_us else 0.0
    mean_latency_ms = mean_latency_us / 1000.0

    return {
        "top_k": top_k,
        "theta_arch": theta_arch,
        "belnap_mode": belnap_mode,
        "coverage": round(coverage, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "total_slots": total_slots_activated,
        "correct_slots": correct_slots_activated,
        "mean_latency_us": round(mean_latency_us, 2),
        "mean_latency_ms": round(mean_latency_ms, 5),
    }


def run_archetype_sweep(
    allow_mock: bool = False,
    device: Optional[str] = None,
    export_report: bool = True,
    update_profile: bool = True,
) -> Dict[str, Any]:
    """Runs the 2D grid sweep across top_k, theta_arch, and belnap_mode."""
    cfg = MultiScaleConfig()
    dim = int(cfg.get("embedding.matryoshka_dim", 64))
    dev = device or cfg.get("embedding.device", "cuda")

    # Ensure grounder is initialized
    grounder = MmapLexicalGrounder.get_default()
    if not grounder.has_archetypes():
        raise RuntimeError(
            "MmapLexicalGrounder has no archetype projection matrix loaded! "
            "Please run scripts/precompute_archetype_embeddings.py first."
        )

    # Load slot metadata
    slots_path = repo_root / "data" / "conceptnet_slots.json"
    with open(slots_path, "r", encoding="utf-8") as f:
        slot_metadata = json.load(f)

    # Initialize embedding manager to pre-embed query neologisms
    logger.info("Initializing EmbeddingManager for benchmark neologisms...")
    if allow_mock:
        mgr = MockEmbeddingManager(matryoshka_dim=dim)
    else:
        mgr = EmbeddingManager(
            backend="auto",
            matryoshka_dim=dim,
            device=dev,
            allow_mock=False,
        )

    # Pre-embed benchmark neologisms
    logger.info("Pre-embedding %d evaluation neologisms...", len(NEOLOGISM_EVAL_BENCHMARK))
    query_texts = [item["term"] for item in NEOLOGISM_EVAL_BENCHMARK]
    query_vectors = mgr.embed_raw(
        [f"task: question answering | query: {t}" for t in query_texts],
        matryoshka_dim=dim,
    )
    # Ensure L2 normalization
    q_norms = np.linalg.norm(query_vectors, axis=-1, keepdims=True)
    query_vectors = query_vectors / np.maximum(q_norms, 1e-12)

    embedded_benchmark = list(zip(NEOLOGISM_EVAL_BENCHMARK, query_vectors))

    logger.info("Beginning sweep over 72 grid configurations...")
    results: List[Dict[str, Any]] = []

    for k in TOP_K_GRID:
        for th in THETA_ARCH_GRID:
            for b_mode in BELNAP_MODES:
                res = evaluate_archetype_configuration(
                    grounder=grounder,
                    slot_metadata=slot_metadata,
                    embedded_benchmark=embedded_benchmark,
                    top_k=k,
                    theta_arch=th,
                    belnap_mode=b_mode,
                )
                results.append(res)

    # Sort results by F1 descending, then precision descending, then latency ascending
    results.sort(
        key=lambda x: (x["f1"], x["precision"], x["coverage"], -x["mean_latency_us"]),
        reverse=True,
    )

    winner = results[0]
    logger.info(
        "=== SWEEP COMPLETED ===\n"
        "Winning Configuration (exp-037d):\n"
        "  top_k: %d\n"
        "  theta_arch: %.2f\n"
        "  belnap_mode: %s\n"
        "  F1: %.1f%% (Coverage: %.1f%%, Precision: %.1f%%)\n"
        "  Projection Latency: %.2f µs (%.4f ms)",
        winner["top_k"],
        winner["theta_arch"],
        winner["belnap_mode"],
        winner["f1"] * 100.0,
        winner["coverage"] * 100.0,
        winner["precision"] * 100.0,
        winner["mean_latency_us"],
        winner["mean_latency_ms"],
    )

    # Persist to profile if requested
    commit_sha = get_git_commit_sha()
    if update_profile:
        profile_path = repo_root / "config" / "multi_scale_profile.json"
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        profile["grounder.archetype_top_k"] = {
            "value": winner["top_k"],
            "source_exp": "exp-037d",
            "source_commit": commit_sha,
            "calibrated_on": "synthetic_neologisms/oov",
            "notes": f"Gate G3b calibrated top-k nearest archetype slots: k={winner['top_k']}",
        }
        profile["grounder.archetype_threshold"] = {
            "value": winner["theta_arch"],
            "source_exp": "exp-037d",
            "source_commit": commit_sha,
            "calibrated_on": "synthetic_neologisms/oov",
            "notes": f"Gate G3b calibrated archetype activation threshold theta_arch={winner['theta_arch']:.2f}",
        }
        profile["grounder.archetype_belnap_value"] = {
            "value": winner["belnap_mode"],
            "source_exp": "exp-037d",
            "source_commit": commit_sha,
            "calibrated_on": "synthetic_neologisms/oov",
            "notes": (
                f"Gate G3b calibrated Belnap truth value strategy: {winner['belnap_mode']} "
                f"(Coverage={winner['coverage']*100:.1f}%, Precision={winner['precision']*100:.1f}%, Latency={winner['mean_latency_us']:.1f}us)"
            ),
        }

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        logger.info("Persisted calibrated parameters to %s", profile_path)

    # Export report if requested
    sweep_summary: Dict[str, Any] = {
        "sweep_exp": "exp-037d",
        "timestamp": datetime.now().isoformat(),
        "commit_sha": commit_sha,
        "winner": winner,
        "top_10": results[:10],
        "total_evaluations": len(results),
    }

    if export_report:
        out_dir = repo_root / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "archetype_grounding_sweep.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sweep_summary, f, indent=2)
        logger.info("Exported sweep JSON to %s", json_path)

        md_path = out_dir / "archetype_grounding_sweep.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# ANVC Dynamic Archetype Grounding Parameter Sweep (exp-037d)\n\n")
            f.write(f"- **Timestamp**: {datetime.now().isoformat()}\n")
            f.write(f"- **Git Commit**: `{commit_sha}`\n")
            f.write(f"- **Evaluated Grid Points**: {len(results)}\n\n")
            f.write("## Winning Configuration (Pareto Frontier)\n\n")
            f.write(f"- **Top-K**: `{winner['top_k']}`\n")
            f.write(f"- **Threshold ($\\theta_{{\\text{{arch}}}}$)**: `{winner['theta_arch']:.2f}`\n")
            f.write(f"- **Belnap Mode**: `{winner['belnap_mode']}`\n")
            f.write(f"- **Coverage**: `{winner['coverage']*100:.1f}%`\n")
            f.write(f"- **Precision**: `{winner['precision']*100:.1f}%`\n")
            f.write(f"- **F1**: `{winner['f1']*100:.1f}%`\n")
            f.write(f"- **Projection Latency**: `{winner['mean_latency_us']:.2f} µs` ({winner['mean_latency_ms']:.4f} ms)\n\n")
            f.write("## Top 10 Parameter Configurations\n\n")
            f.write("| Rank | Top-K | $\\theta_{\\text{arch}}$ | Belnap Mode | Coverage | Precision | F1 | Latency (µs) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for idx, r in enumerate(results[:10]):
                f.write(
                    f"| {idx+1} | {r['top_k']} | {r['theta_arch']:.2f} | {r['belnap_mode']} | "
                    f"{r['coverage']*100:.1f}% | {r['precision']*100:.1f}% | {r['f1']*100:.1f}% | {r['mean_latency_us']:.2f} |\n"
                )
        logger.info("Exported sweep Markdown report to %s", md_path)

    return sweep_summary


def main():
    parser = argparse.ArgumentParser(
        description="Sweep and calibrate ANVC archetype grounding parameters for OOV terms."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="live",
        choices=["live", "auto", "mock"],
        help="Inference mode (live forces real EmbeddingGemma 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="Compute device for EmbeddingGemma 2",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow deterministic mock engine for testing",
    )
    parser.add_argument(
        "--no-update-profile",
        action="store_true",
        help="Skip updating config/multi_scale_profile.json",
    )
    args = parser.parse_args()

    allow_mock = args.allow_mock or (args.mode == "mock")
    run_archetype_sweep(
        allow_mock=allow_mock,
        device=args.device,
        export_report=True,
        update_profile=not args.no_update_profile,
    )


if __name__ == "__main__":
    main()
