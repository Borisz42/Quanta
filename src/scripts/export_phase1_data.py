"""Export script to ingest real reasoning benchmark datasets (FOLIO, ProofWriter, bAbI, CLUTRR, Python ASTs),
run mRMR dimension selection, and export optimal dimensions, profiler reports, candidate pool, and datasets.
"""

from __future__ import annotations
from pathlib import Path
import sys

# Ensure src modules are importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_profiler_artifacts import run_profiler_export


def export_phase1_artifacts(output_dir: Path = Path("output"), samples_per_domain: int = 1000):
    """Maintains backward compatibility while executing the unified 1024-dimension artifact export."""
    run_profiler_export(
        output_dir=output_dir,
        num_dims=1024,
        use_real_data=True,
        samples_per_domain=samples_per_domain,
    )


if __name__ == "__main__":
    out_dir = Path("output")
    samples_per = 1000
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        samples_per = int(sys.argv[2])
    export_phase1_artifacts(out_dir, samples_per_domain=samples_per)

