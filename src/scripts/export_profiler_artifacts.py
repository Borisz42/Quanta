"""Fast export runner for Phase 10: Information Profiler & Dimension Optimization.

Generates 5000 validation propositions, computes entropy, total correlation,
runs mRMR dimension selection over 512+ candidate pool, and exports:
- output/candidate_pool.json
- output/canonical_slots_layout.json
- output/information_profiler_report.txt
- output/information_profiler_report.json
- output/optimal_256_dimensions.json
- output/optimal_256_dimensions.csv
"""

from __future__ import annotations
import json
from pathlib import Path
import sys

# Ensure src modules are importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.slots import CANONICAL_SLOTS
from data.corpus_generator import ValidationCorpusGenerator
from profiler.candidate_pool import build_candidate_pool, export_candidate_pool
from profiler.info_profiler import QuantaInformationProfiler
from profiler.mrmr_selector import MRMRSelector, export_optimal_dimensions


def run_profiler_export(output_dir: Path = Path("output"), num_samples: int = 5000):
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=== QUANTA Information Profiler & Dimension Optimization Exporter ===")
    print(f"Target output directory: {output_dir.resolve()}\n")

    # 1. Export Candidate Pool (512+ Candidates) (10A.6)
    print("1. Compiling and exporting over-complete candidate pool (512+ candidates)...")
    candidates = build_candidate_pool()
    pool_path = output_dir / "candidate_pool.json"
    export_candidate_pool(candidates, pool_path)
    print(f"   Saved {len(candidates)} candidates to {pool_path.name}")

    # 2. Export Canonical 256 Slots Layout
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
    print(f"\n2. Saved canonical 4-band layout to {slots_layout_path.name}")

    # 3. Generate Multi-Domain 5000 Validation Corpus (Phase 11 / Phase 10)
    print(f"\n3. Generating {num_samples} validation corpus propositions across FOLIO, ProofWriter, bAbI, CLUTRR, Code...")
    gen = ValidationCorpusGenerator(seed=42)
    corpus = gen.generate_corpus(num_samples=num_samples)
    print(f"   Canonical Matrix: {corpus.canonical_matrix.shape}")
    print(f"   Candidate Matrix: {corpus.candidate_matrix.shape}")

    # 4. Run Information Profiler on Canonical Matrix & Export (10B.1 - 10B.6)
    print(f"\n4. Running Information Profiler (Entropies, Total Correlation, Collision Rate)...")
    profiler = QuantaInformationProfiler(corpus.canonical_matrix)
    report = profiler.run_diagnostic_suite(dead_threshold=0.01, redundancy_threshold=0.7, verbose=True)

    report_txt_path = output_dir / "information_profiler_report.txt"
    report_json_path = output_dir / "information_profiler_report.json"
    profiler.export_report(txt_path=report_txt_path, json_path=report_json_path, report=report)
    print(f"   Saved profiler report to {report_txt_path.name} and {report_json_path.name}")

    # 5. Run mRMR Selection over 512+ Candidates & Export (10C.1 - 10C.3)
    print(f"\n5. Running greedy mRMR dimension selection over {corpus.candidate_matrix.shape[1]} candidates...")
    candidate_names = [c.name for c in candidates]
    selector = MRMRSelector(candidate_names)
    selected_dims = selector.select_dimensions(
        X=corpus.candidate_matrix,
        y=corpus.labels,
        num_to_select=256,
        alpha_redundancy=0.5,
    )

    dims_json_path = output_dir / "optimal_256_dimensions.json"
    dims_csv_path = output_dir / "optimal_256_dimensions.csv"
    export_optimal_dimensions(selected_dims, candidates, json_path=dims_json_path, csv_path=dims_csv_path)
    print(f"   Saved optimal 256 dimensions to {dims_json_path.name} and {dims_csv_path.name}")

    print(f"\n=== All Phase 10 artifacts successfully exported to {output_dir.resolve()} ===")


if __name__ == "__main__":
    out_dir = Path("output")
    num_s = 5000
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        num_s = int(sys.argv[2])
    run_profiler_export(out_dir, num_samples=num_s)
