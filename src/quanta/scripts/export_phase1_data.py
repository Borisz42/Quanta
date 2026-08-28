"""Export script to ingest real reasoning benchmark datasets (FOLIO, ProofWriter, bAbI, CLUTRR, Python ASTs),

run mRMR dimension selection, and export optimal dimensions, profiler reports, and datasets.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import numpy as np

# Ensure quanta package is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quanta.core.asg import QuantaNode
from quanta.core.slots import CANONICAL_SLOTS, get_slot_by_index, get_slot_names
from quanta.data.real_loader import RealDatasetLoader
from quanta.profiler.candidate_pool import build_candidate_pool
from quanta.profiler.info_profiler import QuantaInformationProfiler
from quanta.profiler.mrmr_selector import MRMRSelector


def export_phase1_artifacts(output_dir: Path, samples_per_domain: int = 500):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== QUANTA Real Data & Dimension Exporter ===")
    print(f"Target output directory: {output_dir.resolve()}\n")

    # 1. Ingest Real Multi-Domain Datasets
    print(f"1. Ingesting real reasoning benchmarks ({samples_per_domain} samples per domain)...")
    loader = RealDatasetLoader()
    corpus = loader.build_real_corpus(samples_per_domain=samples_per_domain)
    num_total = len(corpus.labels)
    print(f"   Total Real Samples Loaded: {num_total}")
    print(f"   Canonical Matrix: {corpus.canonical_matrix.shape}")
    print(f"   Candidate Matrix: {corpus.candidate_matrix.shape}")

    # 2. Save Validation Corpus (.jsonl and .npz)
    corpus_jsonl_path = output_dir / "validation_corpus.jsonl"
    print(f"\n2. Exporting real corpus propositions to {corpus_jsonl_path.name}...")
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

    # 3. Run mRMR Selection over Candidate Pool using Real Data
    print(f"\n3. Running mRMR dimension selection over {corpus.candidate_matrix.shape[1]} candidates on real data...")
    candidates = build_candidate_pool()
    candidate_names = [c.name for c in candidates]
    cand_by_name = {c.name: c for c in candidates}

    selector = MRMRSelector(candidate_names)
    selected_dims = selector.select_dimensions(
        X=corpus.candidate_matrix,
        y=corpus.labels,
        num_to_select=256,
        alpha_redundancy=0.5,
    )

    # Save optimal 256 dimensions to JSON
    dims_json_path = output_dir / "optimal_256_dimensions.json"
    dims_data = []
    for rank, (cand_idx, name, score) in enumerate(selected_dims, start=1):
        cand_obj = cand_by_name[name]
        dims_data.append({
            "rank": rank,
            "id": cand_obj.id,
            "name": cand_obj.name,
            "source": cand_obj.source,
            "category": cand_obj.category,
            "description": cand_obj.description,
            "mrmr_score": float(score),
        })

    with open(dims_json_path, "w", encoding="utf-8") as f:
        json.dump(dims_data, f, indent=2)
    print(f"   Saved optimal 256 dimensions to {dims_json_path.name}")

    # Save optimal 256 dimensions to CSV
    dims_csv_path = output_dir / "optimal_256_dimensions.csv"
    with open(dims_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Candidate_ID", "Name", "Source", "Category", "MRMR_Score", "Description"])
        for item in dims_data:
            writer.writerow([
                item["rank"],
                item["id"],
                item["name"],
                item["source"],
                item["category"],
                f"{item['mrmr_score']:.6f}",
                item["description"],
            ])
    print(f"   Saved optimal 256 dimensions table to {dims_csv_path.name}")

    # 4. Save Canonical 256 Slots Layout
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
    print(f"\n4. Saved canonical 4-band layout to {slots_layout_path.name}")

    # 5. Run Information Profiler on Real Data & Export Diagnostics
    print(f"\n5. Running Information Profiler on real canonical matrix...")
    profiler = QuantaInformationProfiler(corpus.canonical_matrix)
    report = profiler.run_diagnostic_suite(dead_threshold=0.01, redundancy_threshold=0.7, verbose=False)

    report_json_path = output_dir / "information_profiler_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"   Saved profiler metrics to {report_json_path.name}")

    report_txt_path = output_dir / "information_profiler_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=== QUANTA REAL DATA INFORMATION PROFILER REPORT ===\n")
        f.write(f"Total Real Samples Evaluated: {report['num_samples']}\n")
        f.write(f"Average Dimension Entropy: {report['mean_entropy']:.3f} / 2.000 bits\n")
        f.write(f"Concept Collision Rate: {report['collision_rate']:.6f}\n\n")
        f.write("--- Band-wise Mean Entropy ---\n")
        for band_name, val in report["band_entropies"].items():
            f.write(f"  {band_name}: {val:.3f} bits\n")

        f.write(f"\n--- Under-Utilized / Dead Dimensions (< 0.01 bits) ---\n")
        if not report["dead_slots"]:
            f.write("  None! All dimensions meet minimum entropy requirements.\n")
        else:
            for idx, label, h in report["dead_slots"]:
                f.write(f"  Slot {idx:03d} [{label}]: Entropy = {h:.4f} bits\n")

        f.write(f"\n--- High Redundancy Pairs (MI > 0.7 bits) ---\n")
        if not report["high_redundancy_pairs"]:
            f.write("  None! No pairs exceed the redundancy threshold.\n")
        else:
            for dim_a, dim_b, mi in report["high_redundancy_pairs"]:
                f.write(f"  [{dim_a}] <---> [{dim_b}]: MI = {mi:.4f} bits\n")
    print(f"   Saved human-readable report to {report_txt_path.name}")

    print(f"\n=== All Phase 1 real data outputs successfully exported to {output_dir.resolve()} ===")


if __name__ == "__main__":
    out_dir = Path("output")
    samples_per = 500
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        samples_per = int(sys.argv[2])
    export_phase1_artifacts(out_dir, samples_per_domain=samples_per)
