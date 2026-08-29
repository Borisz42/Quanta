"""Script to generate, benchmark, and export comprehensive translation examples.

Generates:
- eng-eng (English -> Mentalese -> English)
- eng-hu  (English -> Mentalese -> Hungarian)
- hu-eng  (Hungarian -> Mentalese -> English)
- hu-hu   (Hungarian -> Mentalese -> Hungarian)
Saved to the output/ directory in JSON and Markdown formats with full ASG node graph ASCII art.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys

# Ensure src is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.slots import get_slot_by_index
from pipeline.translator_pipeline import TwoWayTranslationPipeline
from visualization.asg_visualizer import ASGVisualizer


def generate_all_examples(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = TwoWayTranslationPipeline()

    print("==========================================================")
    print("QUANTA TWO-WAY TRANSLATION & VERIFICATION PIPELINE RUNNER")
    print(f"Target Output Directory: {output_dir.resolve()}")
    print("==========================================================\n")

    all_md_blocks = []

    # ------------------------------------------------------------------
    # 1. English -> English (Round-Trip Invariance)
    # ------------------------------------------------------------------
    eng_sentences = [
        "A dog chased a cat.",
        "A golden retriever bit the mailman in the garden.",
        "A person must touch a rock.",
        "A person saw a garden.",
        "A person did not see a cat.",
        "A dog ran rapidly into the house.",
        "A mailman gave a book to a person.",
        "A human thinks.",
        "Every person wants a house.",
        "A dog was not big.",
    ]

    print("1. Generating English -> English (eng-eng) round-trip examples...")
    eng_eng_records = []
    eng_eng_md = ["# English -> English Round-Trip Translation Graphs\n\n"]

    for i, sent in enumerate(eng_sentences, 1):
        res = pipeline.execute_translation(sent, target_modality="english", source_modality="english")
        rt = pipeline.round_trip(sent, modality="english")
        vec = res.graph.to_proposition_vector()
        active_slots = {get_slot_by_index(k).name: int(v) for k, v in vec.active_slots().items()}
        record = {
            "id": i,
            "input_text": sent,
            "input_modality": "english",
            "target_modality": "english",
            "output_text": res.output_text,
            "merkle_root": res.merkle_root,
            "is_valid": res.validation.is_valid,
            "slot_preservation_rate": rt.slot_preservation_rate,
            "hamming_distance": rt.hamming_distance,
            "active_slots": active_slots,
        }
        eng_eng_records.append(record)

        md_block = ASGVisualizer.format_translation_block(
            example_id=i,
            source_text=sent,
            target_text=res.output_text,
            source_modality="english",
            target_modality="english",
            graph=res.graph,
            merkle_root=res.merkle_root,
            is_valid=res.validation.is_valid,
            preservation_rate=rt.slot_preservation_rate,
            as_markdown=True,
        )
        eng_eng_md.append(md_block)
        all_md_blocks.append(md_block)

        print(f"   [{i:02d}] IN:  {sent}")
        print(f"        OUT: {res.output_text} (Preservation: {rt.slot_preservation_rate:.1%})")

    with open(output_dir / "examples_eng_eng.json", "w", encoding="utf-8") as f:
        json.dump(eng_eng_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "translation_graphs_eng_eng.md", "w", encoding="utf-8") as f:
        f.write("".join(eng_eng_md))

    # ------------------------------------------------------------------
    # 2. English -> Hungarian (eng-hu)
    # ------------------------------------------------------------------
    print("\n2. Generating English -> Hungarian (eng-hu) cross-lingual examples...")
    eng_hu_records = []
    eng_hu_md = ["# English -> Hungarian Cross-Lingual Translation Graphs\n\n"]

    for i, sent in enumerate(eng_sentences, 1):
        res = pipeline.execute_translation(sent, target_modality="hungarian", source_modality="english")
        vec = res.graph.to_proposition_vector()
        active_slots = {get_slot_by_index(k).name: int(v) for k, v in vec.active_slots().items()}
        record = {
            "id": i,
            "input_text": sent,
            "input_modality": "english",
            "target_modality": "hungarian",
            "output_text": res.output_text,
            "merkle_root": res.merkle_root,
            "is_valid": res.validation.is_valid,
            "active_slots": active_slots,
        }
        eng_hu_records.append(record)

        md_block = ASGVisualizer.format_translation_block(
            example_id=i,
            source_text=sent,
            target_text=res.output_text,
            source_modality="english",
            target_modality="hungarian",
            graph=res.graph,
            merkle_root=res.merkle_root,
            is_valid=res.validation.is_valid,
            as_markdown=True,
        )
        eng_hu_md.append(md_block)
        all_md_blocks.append(md_block)

        print(f"   [{i:02d}] EN: {sent}")
        print(f"        HU: {res.output_text}")

    with open(output_dir / "examples_eng_hu.json", "w", encoding="utf-8") as f:
        json.dump(eng_hu_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "translation_graphs_eng_hu.md", "w", encoding="utf-8") as f:
        f.write("".join(eng_hu_md))

    # ------------------------------------------------------------------
    # 3. Hungarian -> English (hu-eng)
    # ------------------------------------------------------------------
    hu_sentences = [
        "A kutya kergette a macskát.",
        "A golden retriever a kertben megharapta a postást.",
        "Az ember látta a kertet.",
        "A kutya nem látta a macskát.",
        "A kutya a házba futott.",
        "A postás adott egy könyvet az embernek.",
        "Az ember gondolkodik.",
        "Minden ember akar egy házat.",
        "A kutya nem volt nagy.",
        "A kutya látott egy kutyát.",
    ]

    print("\n3. Generating Hungarian -> English (hu-eng) cross-lingual examples...")
    hu_eng_records = []
    hu_eng_md = ["# Hungarian -> English Cross-Lingual Translation Graphs\n\n"]

    for i, sent in enumerate(hu_sentences, 1):
        res = pipeline.execute_translation(sent, target_modality="english", source_modality="hungarian")
        vec = res.graph.to_proposition_vector()
        active_slots = {get_slot_by_index(k).name: int(v) for k, v in vec.active_slots().items()}
        record = {
            "id": i,
            "input_text": sent,
            "input_modality": "hungarian",
            "target_modality": "english",
            "output_text": res.output_text,
            "merkle_root": res.merkle_root,
            "is_valid": res.validation.is_valid,
            "active_slots": active_slots,
        }
        hu_eng_records.append(record)

        md_block = ASGVisualizer.format_translation_block(
            example_id=i,
            source_text=sent,
            target_text=res.output_text,
            source_modality="hungarian",
            target_modality="english",
            graph=res.graph,
            merkle_root=res.merkle_root,
            is_valid=res.validation.is_valid,
            as_markdown=True,
        )
        hu_eng_md.append(md_block)
        all_md_blocks.append(md_block)

        print(f"   [{i:02d}] HU: {sent}")
        print(f"        EN: {res.output_text}")

    with open(output_dir / "examples_hu_eng.json", "w", encoding="utf-8") as f:
        json.dump(hu_eng_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "translation_graphs_hu_eng.md", "w", encoding="utf-8") as f:
        f.write("".join(hu_eng_md))

    # ------------------------------------------------------------------
    # 4. Hungarian -> Hungarian (hu-hu Round-Trip Invariance)
    # ------------------------------------------------------------------
    print("\n4. Generating Hungarian -> Hungarian (hu-hu) round-trip examples...")
    hu_hu_records = []
    hu_hu_md = ["# Hungarian -> Hungarian Round-Trip Translation Graphs\n\n"]

    for i, sent in enumerate(hu_sentences, 1):
        res = pipeline.execute_translation(sent, target_modality="hungarian", source_modality="hungarian")
        rt = pipeline.round_trip(sent, modality="hungarian")
        vec = res.graph.to_proposition_vector()
        active_slots = {get_slot_by_index(k).name: int(v) for k, v in vec.active_slots().items()}
        record = {
            "id": i,
            "input_text": sent,
            "input_modality": "hungarian",
            "target_modality": "hungarian",
            "output_text": res.output_text,
            "merkle_root": res.merkle_root,
            "is_valid": res.validation.is_valid,
            "slot_preservation_rate": rt.slot_preservation_rate,
            "hamming_distance": rt.hamming_distance,
            "active_slots": active_slots,
        }
        hu_hu_records.append(record)

        md_block = ASGVisualizer.format_translation_block(
            example_id=i,
            source_text=sent,
            target_text=res.output_text,
            source_modality="hungarian",
            target_modality="hungarian",
            graph=res.graph,
            merkle_root=res.merkle_root,
            is_valid=res.validation.is_valid,
            preservation_rate=rt.slot_preservation_rate,
            as_markdown=True,
        )
        hu_hu_md.append(md_block)
        all_md_blocks.append(md_block)

        print(f"   [{i:02d}] IN:  {sent}")
        print(f"        OUT: {res.output_text} (Preservation: {rt.slot_preservation_rate:.1%})")

    with open(output_dir / "examples_hu_hu.json", "w", encoding="utf-8") as f:
        json.dump(hu_hu_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "translation_graphs_hu_hu.md", "w", encoding="utf-8") as f:
        f.write("".join(hu_hu_md))

    # ------------------------------------------------------------------
    # 5. Combined Master Graph Markdown File
    # ------------------------------------------------------------------
    with open(output_dir / "all_translation_graphs.md", "w", encoding="utf-8") as f:
        f.write("# QUANTA Translation Examples — Complete ASG Node Graphs & ASCII Art\n\n")
        f.write("This document compiles the complete Abstract Syntax Graph (ASG) node hierarchies, relation edges, active quaternary semantic slots, and Mermaid diagrams for all 40 benchmark translation examples across English and Hungarian.\n\n")
        f.write("---\n\n")
        f.write("".join(all_md_blocks))

    # ------------------------------------------------------------------
    # 6. Generate Comprehensive Markdown Summary
    # ------------------------------------------------------------------
    md_path = output_dir / "translation_examples_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# QUANTA Two-Way Translation & Cross-Lingual Benchmarks\n\n")
        f.write("This document presents authentic input-output translation pairs generated via the Content-Addressed Neuro-Symbolic **QUANTA ASG** representation over $\\Sigma^{256}$.\n\n")
        f.write("### Graph Visualizations\n")
        f.write("Detailed structured ASCII graph trees, Mermaid diagrams, and node slot specifications are available in:\n")
        f.write("- **Complete Graphs:** [`all_translation_graphs.md`](all_translation_graphs.md)\n")
        f.write("- **English Round-Trip:** [`translation_graphs_eng_eng.md`](translation_graphs_eng_eng.md)\n")
        f.write("- **English → Hungarian:** [`translation_graphs_eng_hu.md`](translation_graphs_eng_hu.md)\n")
        f.write("- **Hungarian → English:** [`translation_graphs_hu_eng.md`](translation_graphs_hu_eng.md)\n")
        f.write("- **Hungarian Round-Trip:** [`translation_graphs_hu_hu.md`](translation_graphs_hu_hu.md)\n\n")
        f.write("---\n\n")

        # Table 1: English -> English
        f.write("## 1. English $\\leftrightarrow$ English (Round-Trip Invariance)\n\n")
        f.write("| ID | Input Sentence | Reconstructed English | Preservation Rate | Merkle Root (BLAKE3) |\n")
        f.write("|:---|:---------------|:----------------------|:------------------|:---------------------|\n")
        for r in eng_eng_records:
            f.write(f"| {r['id']:02d} | `{r['input_text']}` | `{r['output_text']}` | **{r['slot_preservation_rate']:.1%}** | `{r['merkle_root'][:16]}...` |\n")

        f.write("\n---\n\n")

        # Table 2: Hungarian -> Hungarian
        f.write("## 2. Hungarian $\\leftrightarrow$ Hungarian (Round-Trip Invariance)\n\n")
        f.write("| ID | Input Sentence | Reconstructed Hungarian | Preservation Rate | Merkle Root (BLAKE3) |\n")
        f.write("|:---|:---------------|:------------------------|:------------------|:---------------------|\n")
        for r in hu_hu_records:
            f.write(f"| {r['id']:02d} | `{r['input_text']}` | `{r['output_text']}` | **{r['slot_preservation_rate']:.1%}** | `{r['merkle_root'][:16]}...` |\n")

        f.write("\n---\n\n")

        # Table 3: English -> Hungarian
        f.write("## 3. English $\\rightarrow$ Hungarian (Cross-Lingual Agglutinative Mapping)\n\n")
        f.write("| ID | Source English | Target Hungarian (Vowel Harmony + Suffixes) | Validation Gate | Merkle Root |\n")
        f.write("|:---|:---------------|:--------------------------------------------|:----------------|:------------|\n")
        for r in eng_hu_records:
            status = "✅ PASS" if r["is_valid"] else "❌ FAIL"
            f.write(f"| {r['id']:02d} | `{r['input_text']}` | `{r['output_text']}` | {status} | `{r['merkle_root'][:16]}...` |\n")

        f.write("\n---\n\n")

        # Table 4: Hungarian -> English
        f.write("## 4. Hungarian $\\rightarrow$ English (Morphological Parsing to SVO Realization)\n\n")
        f.write("| ID | Source Hungarian | Target English | Validation Gate | Merkle Root |\n")
        f.write("|:---|:-----------------|:---------------|:----------------|:------------|\n")
        for r in hu_eng_records:
            status = "✅ PASS" if r["is_valid"] else "❌ FAIL"
            f.write(f"| {r['id']:02d} | `{r['input_text']}` | `{r['output_text']}` | {status} | `{r['merkle_root'][:16]}...` |\n")

        f.write("\n---\n\n")
        f.write("### Detailed Sample Trace: English $\\to$ Mentalese ASG $\\to$ Hungarian\n\n")
        sample = eng_hu_records[1]  # Dog bit mailman in garden
        f.write(f"**Input:** `{sample['input_text']}`  \n")
        f.write(f"**Hungarian Output:** `{sample['output_text']}`  \n")
        f.write(f"**BLAKE3 Merkle Root:** `{sample['merkle_root']}`  \n\n")
        f.write("**Active Discrete Slots (Band 0–3):**\n```json\n")
        f.write(json.dumps(sample["active_slots"], indent=2))
        f.write("\n```\n")

    print(f"\n==========================================================")
    print(f"Successfully generated all translation benchmark examples!")
    print(f"Files created in {output_dir.resolve()}:")
    print(f"  - examples_eng_eng.json")
    print(f"  - examples_hu_hu.json")
    print(f"  - examples_eng_hu.json")
    print(f"  - examples_hu_eng.json")
    print(f"  - translation_graphs_eng_eng.md")
    print(f"  - translation_graphs_eng_hu.md")
    print(f"  - translation_graphs_hu_eng.md")
    print(f"  - translation_graphs_hu_hu.md")
    print(f"  - all_translation_graphs.md")
    print(f"  - translation_examples_summary.md")
    print("==========================================================")


if __name__ == "__main__":
    out = REPO_ROOT / "output"
    if len(sys.argv) > 1:
        out = Path(sys.argv[1]).resolve()
    generate_all_examples(out)
