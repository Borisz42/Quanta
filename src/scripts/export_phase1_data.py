"""Export script to ingest real reasoning benchmark datasets (FOLIO, ProofWriter, bAbI, CLUTRR, Python ASTs),
run mRMR dimension selection, and export optimal dimensions, profiler reports, candidate pool, and datasets.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import numpy as np

# Ensure src modules are importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.asg import QuantaNode
from core.slots import CANONICAL_SLOTS, get_slot_by_index, get_slot_names
from data.real_loader import RealDatasetLoader
from profiler.candidate_pool import build_candidate_pool, export_candidate_pool
from profiler.info_profiler import QuantaInformationProfiler
from profiler.mrmr_selector import MRMRSelector, export_optimal_dimensions


def export_phase1_artifacts(output_dir: Path, samples_per_domain: int = 500):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== QUANTA Real Data & Dimension Exporter ===")
    print(f"Target output directory: {output_dir.resolve()}\n")

    # 1. Export Candidate Pool (512+ Candidates)
    print("1. Building and exporting over-complete candidate pool...")
    candidates = build_candidate_pool()
    pool_path = output_dir / "candidate_pool.json"
    export_candidate_pool(candidates, pool_path)
    print(f"   Saved {len(candidates)} candidates to {pool_path.name}")

    # 2. Ingest Real Multi-Domain Datasets
    print(f"\n2. Ingesting real reasoning benchmarks ({samples_per_domain} samples per domain)...")
    loader = RealDatasetLoader()
    corpus = loader.build_real_corpus(samples_per_domain=samples_per_domain)
    num_total = len(corpus.labels)
    print(f"   Total Real Samples Loaded: {num_total}")
    print(f"   Canonical Matrix: {corpus.canonical_matrix.shape}")
    print(f"   Candidate Matrix: {corpus.candidate_matrix.shape}")

    # 3. Save Validation Corpus (.jsonl and .npz)
    corpus_jsonl_path = output_dir / "validation_corpus.jsonl"
    print(f"\n3. Exporting real corpus propositions to {corpus_jsonl_path.name}...")
    domain_map = {0: "FOLIO", 1: "ProofWriter", 2: "bAbI", 3: "CLUTRR"}

    with open(corpus_jsonl_path, "w", encoding="utf-8") as f:
        for idx in range(num_total):
            vec = corpus.canonical_matrix[idx]
            active_slots = {get_slot_by_index(i).name: int(vec[i]) for i in range(256) if vec[i] != 0}
            node = QuantaNode(vector=active_slots, literal=corpus.propositions[idx])
            record = {
                "sample_id": idx,
                "domain": domain_map.get(int(corpus.labels[idx]), "Unknown"),
                "text": corpus.propositions[idx],
                "active_slots": active_slots,
                "node_cid": node.cid,
                "packed_bytes_hex": node.vector.to_bytes().hex(),
            }
            f.write(json.dumps(record) + "\n")

    npz_path = output_dir / "validation_corpus_tensors.npz"
    np.savez_compressed(
        npz_path,
        canonical_matrix=corpus.canonical_matrix,
        candidate_matrix=corpus.candidate_matrix,
        labels=corpus.labels,
    )
    print(f"   Saved compressed NumPy tensors to {npz_path.name}")

    # 4. Run mRMR Selection over Candidate Pool using Real Data
    print(f"\n4. Running mRMR dimension selection over {corpus.candidate_matrix.shape[1]} candidates on real data...")
    candidate_names = [c.name for c in candidates]

    selector = MRMRSelector(candidate_names)
    selected_dims = selector.select_dimensions(
        X=corpus.candidate_matrix,
        y=corpus.labels,
        num_to_select=256,
        alpha_redundancy=0.5,
    )

    # Save optimal 256 dimensions to JSON and CSV
    dims_json_path = output_dir / "optimal_256_dimensions.json"
    dims_csv_path = output_dir / "optimal_256_dimensions.csv"
    export_optimal_dimensions(selected_dims, candidates, json_path=dims_json_path, csv_path=dims_csv_path)
    print(f"   Saved optimal 256 dimensions to {dims_json_path.name} and {dims_csv_path.name}")

    # 5. Save Canonical 256 Slots Layout
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
    print(f"\n5. Saved canonical 4-band layout to {slots_layout_path.name}")

    # 6. Run Information Profiler on Real Data & Export Diagnostics
    print(f"\n6. Running Information Profiler on real canonical matrix...")
    profiler = QuantaInformationProfiler(corpus.canonical_matrix)
    report = profiler.run_diagnostic_suite(dead_threshold=0.01, redundancy_threshold=0.7, verbose=False)

    report_json_path = output_dir / "information_profiler_report.json"
    report_txt_path = output_dir / "information_profiler_report.txt"
    profiler.export_report(txt_path=report_txt_path, json_path=report_json_path, report=report)
    print(f"   Saved profiler metrics to {report_json_path.name} and {report_txt_path.name}")

    print(f"\n=== All Phase 10 artifacts successfully exported to {output_dir.resolve()} ===")


if __name__ == "__main__":
    out_dir = Path("output")
    samples_per = 500
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        samples_per = int(sys.argv[2])
    export_phase1_artifacts(out_dir, samples_per_domain=samples_per)

