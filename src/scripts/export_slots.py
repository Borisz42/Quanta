"""Script to export all 1024 canonical slot definitions to output/canonical_slots_layout.json."""

from __future__ import annotations
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.slots import export_canonical_slots_layout, CANONICAL_SLOTS


def main():
    target_path = Path("output/canonical_slots_layout.json")
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])

    exported = export_canonical_slots_layout(target_path)
    print(f"Successfully exported {len(CANONICAL_SLOTS)} canonical slot definitions to {exported.resolve()}")


if __name__ == "__main__":
    main()
