"""Script to export FrameNet frames, thematic roles, and Band 1 valency slot templates.

Exports data/framenet_valency.json used for compile-time predicate valency resolution.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from parser.lexical_grounder import FrameNetValencyResolver


def export_framenet_valency(output_path: Path = Path("data/framenet_valency.json")):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"=== QUANTA FrameNet Valency Template Exporter ===")
    print(f"Target file: {output_path.resolve()}\n")

    frames_dict = {}
    for frame_name, f_data in FrameNetValencyResolver.BUILTIN_FRAMES.items():
        frames_dict[frame_name] = {
            "frame_id": f_data["frame_id"],
            "core_elements": f_data["core_elements"],
            "slot_mapping": f_data["slot_mapping"],
            "semantic_types": f_data["semantic_types"],
            "sample_verbs": [
                v for v, fr in FrameNetValencyResolver.VERB_LEMMA_TO_FRAME.items()
                if fr == frame_name
            ],
        }

    payload = {
        "description": "QUANTA FrameNet Valency and Thematic Role Template Matrices",
        "version": "1.0",
        "frames": frames_dict,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully exported {len(frames_dict)} FrameNet valency templates to {output_path.name}!")


if __name__ == "__main__":
    out = REPO_ROOT / "data" / "framenet_valency.json"
    export_framenet_valency(out)
