"""Consolidated Artifact Exporter for QUANTA 1024-Dimension Architecture.

Evaluates translator grounding against gold paired benchmarks, audits symbolic soundness,
measures causal necessity, computes entropy and total correlation, runs multi-objective mRMR dimension
selection over real and synthetic reasoning benchmarks, and exports:
- output/candidate_pool.json
- output/canonical_slots_layout.json
- output/verifier_report.json
- output/information_profiler_report.txt
- output/information_profiler_report.json
- output/optimal_dimensions.json & .csv (and optimal_1024_dimensions.json/.csv)
- output/validation_corpus.jsonl & validation_corpus_tensors.npz
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Optional
import numpy as np

# Ensure src modules are importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.asg import QuantaNode
from core.slots import CANONICAL_SLOTS, get_slot_by_index
from data.corpus_generator import ValidationCorpusGenerator
from data.gold_corpus import GoldCorpusLoader
from data.real_loader import RealDatasetLoader
from profiler.candidate_pool import build_candidate_pool, export_candidate_pool
from profiler.info_profiler import QuantaInformationProfiler
from profiler.mrmr_selector import MRMRSelector, export_optimal_dimensions
from profiler.verifier import SemanticVectorVerifier


def run_profiler_export(
    output_dir: Path = Path("output"),
    num_samples: int = 5000,
    num_dims: int = 1024,
    use_real_data: bool = True,
    samples_per_domain: int = 1000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_val_dir = Path("data/validation_corpus")
    data_val_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("      QUANTA 1024-Dimension Neuro-Symbolic Artifact & Profiler Exporter        ")
    print("================================================================================")
    print(f"Target Output Directory: {output_dir.resolve()}")
    print(f"Target Dimensions:       {num_dims}")
    print(f"Dataset Mode:            {'Real Multi-Domain Benchmarks' if use_real_data else 'Synthetic Multi-Domain Generator'}\n")

    # 1. Export Clean Candidate Pool (2048 Candidates)
    print("1. Compiling and exporting clean candidate pool...", flush=True)
    candidates = build_candidate_pool()
    pool_path = output_dir / "candidate_pool.json"
    export_candidate_pool(candidates, pool_path)
    print(f"   [OK] Saved {len(candidates)} candidates to {pool_path.name}\n", flush=True)

    # 2. Export Canonical 1024 Slots Layout (8 Modular Bands)
    print("2. Exporting canonical 1024 slots layout...", flush=True)
    slots_layout_path = output_dir / "canonical_slots_layout.json"
    slots_layout = [
        {
            "index": s.index,
            "name": s.name,
            "band_id": int(s.band),
            "band_name": s.band.name,
            "category": s.category,
            "description": s.description,
        }
        for s in CANONICAL_SLOTS
    ]
    with open(slots_layout_path, "w", encoding="utf-8") as f:
        json.dump(slots_layout, f, indent=2)
    print(f"   [OK] Saved canonical 8-band layout to {slots_layout_path.name}\n", flush=True)

    # 3. Run Semantic Vector Verifier on Gold Paired Benchmarks
    print("3. Running Semantic Vector Verifier against gold formal ground truths...", flush=True)
    gold_loader = GoldCorpusLoader()
    gold_pairs = gold_loader.build_gold_validation_suite(max_total=500)
    print(f"   Compiled {len(gold_pairs)} gold-standard validation pairs (FOLIO, Code AST).", flush=True)

    verifier = SemanticVectorVerifier()
    verification_summary = verifier.run_full_verification(gold_pairs, dead_threshold=0.01)

    verifier_json_path = output_dir / "verifier_report.json"
    with open(verifier_json_path, "w", encoding="utf-8") as f:
        json.dump(verification_summary.to_dict(), f, indent=2)
    print(f"   Translator Macro F1 Score:  {verification_summary.gold_alignment.macro_f1:.3f}")
    print(f"   Cycle-Consistency Fidelity: {verification_summary.cycle_consistency_score:.3f}")
    print(f"   Symbolic Soundness Rate:    {verification_summary.symbolic_soundness_rate*100:.1f}%")
    print(f"   [OK] Saved verification report to {verifier_json_path.name}\n", flush=True)

    # 4. Ingest/Generate Validation Corpus
    if use_real_data:
        print(f"4. Ingesting real reasoning benchmarks ({samples_per_domain} samples per domain across 5 domains)...", flush=True)
        loader = RealDatasetLoader()
        corpus = loader.build_real_corpus(samples_per_domain=samples_per_domain)
    else:
        print(f"4. Generating {num_samples} validation corpus propositions...", flush=True)
        gen = ValidationCorpusGenerator(seed=42)
        corpus = gen.generate_corpus(num_samples=num_samples)

    num_total = len(corpus.labels)
    print(f"   Total Samples: {num_total}")
    print(f"   Canonical Matrix Shape: {corpus.canonical_matrix.shape}")
    print(f"   Candidate Matrix Shape: {corpus.candidate_matrix.shape}", flush=True)

    # 5. Export Corpus JSONL and NPZ Tensors
    corpus_jsonl_path = output_dir / "validation_corpus.jsonl"
    data_jsonl_path = data_val_dir / "validation_corpus.jsonl"
    print(f"\n5. Exporting propositions and packed vectors to {corpus_jsonl_path.name}...", flush=True)
    domain_map = {0: "FOLIO", 1: "ProofWriter", 2: "bAbI", 3: "CLUTRR", 4: "CodeAST"}

    with open(corpus_jsonl_path, "w", encoding="utf-8") as f_out, open(data_jsonl_path, "w", encoding="utf-8") as f_data:
        for idx in range(num_total):
            vec = corpus.canonical_matrix[idx]
            active_slots = {get_slot_by_index(i).name: int(vec[i]) for i in range(len(vec)) if vec[i] != 0}
            node = QuantaNode(vector=active_slots, literal=corpus.propositions[idx])
            record = {
                "sample_id": idx,
                "domain": domain_map.get(int(corpus.labels[idx]), "Unknown"),
                "text": corpus.propositions[idx],
                "active_slots": active_slots,
                "node_cid": node.cid,
                "packed_bytes_hex": node.vector.to_bytes().hex(),
            }
            line = json.dumps(record) + "\n"
            f_out.write(line)
            f_data.write(line)

    npz_path = output_dir / "validation_corpus_tensors.npz"
    data_npz_path = data_val_dir / "validation_corpus_tensors.npz"
    for p in [npz_path, data_npz_path]:
        np.savez_compressed(
            p,
            canonical_matrix=corpus.canonical_matrix,
            candidate_matrix=corpus.candidate_matrix,
            labels=corpus.labels,
        )
    print(f"   [OK] Saved compressed NumPy tensors to {npz_path.name}\n", flush=True)

    # 6. Run Information Profiler with Verifier Diagnostics
    print("6. Running Information Profiler (Entropies, Total Correlation, Collision Rate)...", flush=True)
    profiler = QuantaInformationProfiler(
        data_matrix=corpus.canonical_matrix,
        verification_summary=verification_summary,
    )
    report = profiler.run_diagnostic_suite(dead_threshold=0.01, redundancy_threshold=0.7, verbose=True)

    report_txt_path = output_dir / "information_profiler_report.txt"
    report_json_path = output_dir / "information_profiler_report.json"
    profiler.export_report(txt_path=report_txt_path, json_path=report_json_path, report=report)
    print(f"   [OK] Saved profiler report to {report_txt_path.name} and {report_json_path.name}\n", flush=True)

    # 7. Run Multi-Objective mRMR Dimension Selection
    print(f"7. Running Multi-Objective mRMR Dimension Selection over {len(candidates)} candidates (target: {num_dims} dims)...", flush=True)
    candidate_names = [c.name for c in candidates]
    selector = MRMRSelector(candidate_names)

    selected_dims = selector.select_dimensions_multi_objective(
        X=corpus.candidate_matrix,
        f1_scores=verification_summary.gold_alignment.f1_score,
        causal_necessity=verification_summary.causal_necessity_scores,
        num_to_select=num_dims,
        w_f1=0.35,
        w_entropy=0.25,
        w_necessity=0.20,
        w_redundancy=0.15,
        w_violation=0.05,
    )

    for prefix in ["optimal_dimensions", f"optimal_{num_dims}_dimensions"]:
        dims_json_path = output_dir / f"{prefix}.json"
        dims_csv_path = output_dir / f"{prefix}.csv"
        export_optimal_dimensions(selected_dims, candidates, json_path=dims_json_path, csv_path=dims_csv_path)

    print(f"   [OK] Saved optimal {num_dims} dimensions to {output_dir / 'optimal_dimensions.json'} and .csv\n", flush=True)
    print("================================================================================")
    print(f"  All QUANTA 1024-D artifacts successfully exported to: {output_dir.resolve()}")
    print("================================================================================")


def main():
    parser = argparse.ArgumentParser(description="Export all QUANTA 1024-dimension evaluation artifacts.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save generated artifacts")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of synthetic samples (if synthetic)")
    parser.add_argument("--samples-per-domain", type=int, default=1000, help="Samples per domain (if real data)")
    parser.add_argument("--num-dims", type=int, default=1024, help="Target optimal dimensions to select")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic generator instead of real benchmarks")

    args = parser.parse_args()
    run_profiler_export(
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        num_dims=args.num_dims,
        use_real_data=not args.synthetic,
        samples_per_domain=args.samples_per_domain,
    )


if __name__ == "__main__":
    main()

