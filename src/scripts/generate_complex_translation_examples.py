"""Script to generate, benchmark, and export complex translation examples and multi-sentence cohesive discourse paragraphs.

Generates:
- eng-eng (English -> Mentalese -> English)
- eng-hu  (English -> Mentalese -> Hungarian)
- hu-eng  (Hungarian -> Mentalese -> English)
- hu-hu   (Hungarian -> Mentalese -> Hungarian)

Includes:
- Complex sentences with multi-prepositional modifiers, compound coordination, conditionals, modals, and negation.
- A 6-sentence cohesive narrative discourse paragraph with cross-sentence anaphora, coreference bundles, and timeline sequencing.
Saved to the output/ directory in JSON and Markdown formats with full ASG node graph ASCII art and Mermaid flowcharts.
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


def generate_all_complex_examples(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = TwoWayTranslationPipeline()

    print("================================================================================")
    print("QUANTA COMPLEX TRANSLATION & DISCOURSE COHESION BENCHMARK GENERATOR")
    print(f"Target Output Directory: {output_dir.resolve()}")
    print("================================================================================\n")

    all_md_blocks = []

    # ------------------------------------------------------------------
    # Benchmark Corpora
    # ------------------------------------------------------------------
    eng_complex_sentences = [
        # Stress Test 1: Counterfactual Causal Reasoning with Sarcasm & Second-Order Theory of Mind
        "Had Alice not falsely pretended to know that Bob believed her investment was secure, the auditor wouldn't have sarcastically remarked that her due diligence was a stroke of genius.",
        # Stress Test 2: Mixed Temporal Intervals, Continuous Kinematics, and Spatial Mereotopology
        "While the drone was accelerating into the restricted airspace before dusk, the operator plausibly suspected, but could not deduce with certainty, that the left wingtip was tangentially touching the perimeter wire.",
        # Stress Test 3: Deep Quantifier Scope Ambiguity with Higher-Order Modal Logic
        "Every investigator who doubted that any suspect had necessarily committed every crime secretly wanted someone to prove the absolute impossibility of an accomplice's alibi.",
        # Stress Test 4: Metalogical Self-Reference and Deontic Causal Interventions
        "By declaring this very decree to be legally void, the council obligated the commissioner to prevent its future enforcement unless the clause could recursively validate its own origin.",
        # Multi-Sentence Cohesive Scientific Narrative Paragraph
        "Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn. She immediately noted that this specimen exhibited anomalous crystalline lattice expansion, which strongly suggested an unobserved phase transition. Although her supervisor initially doubted the validity of the discovery, Eleanor verified the hypothesis three hours later by replicating the transformation within the same vessel. The resulting polymer retained its structural integrity throughout the afternoon, prompting the laboratory director to prohibit all competing tests until her synthesis protocol could be formally audited.",
    ]

    hu_complex_sentences = [
        # Stress Test 1: Counterfactual Causal Reasoning with Sarcasm & Second-Order Theory of Mind (HU)
        "Ha Alice nem színlelte volna hamisan, hogy tudja, hogy Bob biztonságosnak hitte a befektetését, a könyvvizsgáló nem jegyezte volna meg gúnyosan, hogy az átvilágítása zseniális húzás volt.",
        # Stress Test 2: Mixed Temporal Intervals, Continuous Kinematics, and Spatial Mereotopology (HU)
        "Miközben a drón szürkület előtt a korlátozott légtérbe gyorsult, a kezelő valószínűsíthetően gyanította, de nem tudta bizonyossággal levezetni, hogy a bal szárnyvég érintőlegesen érintette a kerítésdrótot.",
        # Stress Test 3: Deep Quantifier Scope Ambiguity with Higher-Order Modal Logic (HU)
        "Minden nyomozó, aki kételkedett abban, hogy bármelyik gyanúsított szükségszerűen elkövetett minden bűncselekményt, titokban azt akarta, hogy valaki bebizonyítsa egy bűntárs alibijének abszolút lehetetlenségét.",
        # Stress Test 4: Metalogical Self-Reference and Deontic Causal Interventions (HU)
        "Azzal, hogy ezt a rendeletet jogilag semmisnek nyilvánította, a tanács kötelezte a biztost a jövőbeli végrehajtás megakadályozására, hacsak a záradék rekurzívan nem tudta igazolni saját eredetét.",
        # Multi-Sentence Cohesive Scientific Narrative Paragraph (HU)
        "Dr. Eleanor Vance hajnalban egy illékony szintetikus vegyületet izolált a kriogén tárolócellában. Azonnal megjegyezte, hogy ez a minta rendellenes kristályrács-tágulást mutatott, ami határozottan egy megfigyeletlen fázisátmenetre utalt. Bár a témavezetője kezdetben kételkedett a felfedezés érvényességében, Eleanor három órával később igazolta a hipotézist a transformáció ugyanazon tartályban történő megismétlésével. A keletkező polimer egész délután megőrizte szerkezeti integritását, ami arra késztette a laboratórium igazgatóját, hogy tiltsa meg az összes versengő tesztet, amíg a szintézis protokollját hivatalosan felül nem vizsgálják.",
    ]

    # ------------------------------------------------------------------
    # 1. English -> English (Round-Trip Invariance)
    # ------------------------------------------------------------------
    print("1. Generating English -> English (eng-eng) complex round-trip examples...")
    eng_eng_records = []
    eng_eng_md = ["# English -> English Complex Round-Trip Translation Graphs\n\n"]

    for i, sent in enumerate(eng_complex_sentences, 1):
        res = pipeline.execute_translation(sent, target_modality="english", source_modality="english")
        rt = pipeline.round_trip(sent, modality="english")
        vec = res.graph.to_proposition_vector()
        active_slots = {get_slot_by_index(k).name: int(v) for k, v in vec.active_slots().items()}
        is_paragraph = (i == len(eng_complex_sentences))

        record = {
            "id": i,
            "type": "paragraph" if is_paragraph else "sentence",
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

        label = "PARAGRAPH" if is_paragraph else f"EX {i:02d}"
        print(f"   [{label}] IN:  {sent[:70]}{'...' if len(sent) > 70 else ''}")
        print(f"            OUT: {res.output_text[:70]}{'...' if len(res.output_text) > 70 else ''} (Preservation: {rt.slot_preservation_rate:.1%})")

    with open(output_dir / "complex_examples_eng_eng.json", "w", encoding="utf-8") as f:
        json.dump(eng_eng_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "complex_translation_graphs_eng_eng.md", "w", encoding="utf-8") as f:
        f.write("".join(eng_eng_md))

    # ------------------------------------------------------------------
    # 2. English -> Hungarian (eng-hu)
    # ------------------------------------------------------------------
    print("\n2. Generating English -> Hungarian (eng-hu) complex cross-lingual examples...")
    eng_hu_records = []
    eng_hu_md = ["# English -> Hungarian Complex Cross-Lingual Translation Graphs\n\n"]

    for i, sent in enumerate(eng_complex_sentences, 1):
        res = pipeline.execute_translation(sent, target_modality="hungarian", source_modality="english")
        vec = res.graph.to_proposition_vector()
        active_slots = {get_slot_by_index(k).name: int(v) for k, v in vec.active_slots().items()}
        is_paragraph = (i == len(eng_complex_sentences))

        record = {
            "id": i,
            "type": "paragraph" if is_paragraph else "sentence",
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

        label = "PARAGRAPH" if is_paragraph else f"EX {i:02d}"
        print(f"   [{label}] EN: {sent[:70]}{'...' if len(sent) > 70 else ''}")
        print(f"            HU: {res.output_text[:70]}{'...' if len(res.output_text) > 70 else ''}")

    with open(output_dir / "complex_examples_eng_hu.json", "w", encoding="utf-8") as f:
        json.dump(eng_hu_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "complex_translation_graphs_eng_hu.md", "w", encoding="utf-8") as f:
        f.write("".join(eng_hu_md))

    # ------------------------------------------------------------------
    # 3. Hungarian -> English (hu-eng)
    # ------------------------------------------------------------------
    print("\n3. Generating Hungarian -> English (hu-eng) complex cross-lingual examples...")
    hu_eng_records = []
    hu_eng_md = ["# Hungarian -> English Complex Cross-Lingual Translation Graphs\n\n"]

    for i, sent in enumerate(hu_complex_sentences, 1):
        res = pipeline.execute_translation(sent, target_modality="english", source_modality="hungarian")
        vec = res.graph.to_proposition_vector()
        active_slots = {get_slot_by_index(k).name: int(v) for k, v in vec.active_slots().items()}
        is_paragraph = (i == len(hu_complex_sentences))

        record = {
            "id": i,
            "type": "paragraph" if is_paragraph else "sentence",
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

        label = "PARAGRAPH" if is_paragraph else f"EX {i:02d}"
        print(f"   [{label}] HU: {sent[:70]}{'...' if len(sent) > 70 else ''}")
        print(f"            EN: {res.output_text[:70]}{'...' if len(res.output_text) > 70 else ''}")

    with open(output_dir / "complex_examples_hu_eng.json", "w", encoding="utf-8") as f:
        json.dump(hu_eng_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "complex_translation_graphs_hu_eng.md", "w", encoding="utf-8") as f:
        f.write("".join(hu_eng_md))

    # ------------------------------------------------------------------
    # 4. Hungarian -> Hungarian (hu-hu Round-Trip Invariance)
    # ------------------------------------------------------------------
    print("\n4. Generating Hungarian -> Hungarian (hu-hu) complex round-trip examples...")
    hu_hu_records = []
    hu_hu_md = ["# Hungarian -> Hungarian Complex Round-Trip Translation Graphs\n\n"]

    for i, sent in enumerate(hu_complex_sentences, 1):
        res = pipeline.execute_translation(sent, target_modality="hungarian", source_modality="hungarian")
        rt = pipeline.round_trip(sent, modality="hungarian")
        vec = res.graph.to_proposition_vector()
        active_slots = {get_slot_by_index(k).name: int(v) for k, v in vec.active_slots().items()}
        is_paragraph = (i == len(hu_complex_sentences))

        record = {
            "id": i,
            "type": "paragraph" if is_paragraph else "sentence",
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

        label = "PARAGRAPH" if is_paragraph else f"EX {i:02d}"
        print(f"   [{label}] IN:  {sent[:70]}{'...' if len(sent) > 70 else ''}")
        print(f"            OUT: {res.output_text[:70]}{'...' if len(res.output_text) > 70 else ''} (Preservation: {rt.slot_preservation_rate:.1%})")

    with open(output_dir / "complex_examples_hu_hu.json", "w", encoding="utf-8") as f:
        json.dump(hu_hu_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "complex_translation_graphs_hu_hu.md", "w", encoding="utf-8") as f:
        f.write("".join(hu_hu_md))

    # ------------------------------------------------------------------
    # 5. Combined Master Complex Graph Markdown File
    # ------------------------------------------------------------------
    with open(output_dir / "all_complex_translation_graphs.md", "w", encoding="utf-8") as f:
        f.write("# QUANTA Complex Translation & Discourse Paragraph Graphs\n\n")
        f.write("This document compiles the complete Abstract Syntax Graph (ASG) node hierarchies, relation edges, active quaternary semantic slots, discourse coreference bundles, and Mermaid diagrams for all complex translation benchmark examples and multi-sentence cohesive discourse paragraphs across English and Hungarian.\n\n")
        f.write("---\n\n")
        f.write("".join(all_md_blocks))

    # ------------------------------------------------------------------
    # 6. Generate Comprehensive Markdown Summary
    # ------------------------------------------------------------------
    md_path = output_dir / "complex_translation_examples_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# QUANTA Complex Sentences & Discourse Cohesion Translation Benchmarks\n\n")
        f.write("This document presents authentic input-output translation benchmarks on **complex sentences** and **multi-sentence cohesive paragraphs** generated via the Content-Addressed Neuro-Symbolic **QUANTA ASG** representation over $\\Sigma^{256}$.\n\n")
        f.write("### Graph Visualizations\n")
        f.write("Detailed structured ASCII graph trees, Mermaid diagrams, and node slot specifications are available in:\n")
        f.write("- **Complete Graphs:** [`all_complex_translation_graphs.md`](all_complex_translation_graphs.md)\n")
        f.write("- **English Round-Trip:** [`complex_translation_graphs_eng_eng.md`](complex_translation_graphs_eng_eng.md)\n")
        f.write("- **English → Hungarian:** [`complex_translation_graphs_eng_hu.md`](complex_translation_graphs_eng_hu.md)\n")
        f.write("- **Hungarian → English:** [`complex_translation_graphs_hu_eng.md`](complex_translation_graphs_hu_eng.md)\n")
        f.write("- **Hungarian Round-Trip:** [`complex_translation_graphs_hu_hu.md`](complex_translation_graphs_hu_hu.md)\n\n")
        f.write("---\n\n")

        # Table 1: English -> English
        f.write("## 1. English $\\leftrightarrow$ English (Complex Round-Trip Invariance)\n\n")
        f.write("| ID | Type | Input Sentence / Paragraph | Reconstructed English | Preservation Rate | Merkle Root (BLAKE3) |\n")
        f.write("|:---|:-----|:---------------------------|:----------------------|:------------------|:---------------------|\n")
        for r in eng_eng_records:
            t_badge = "📖 Paragraph" if r["type"] == "paragraph" else "Sentence"
            f.write(f"| {r['id']:02d} | {t_badge} | `{r['input_text']}` | `{r['output_text']}` | **{r['slot_preservation_rate']:.1%}** | `{r['merkle_root'][:16]}...` |\n")

        f.write("\n---\n\n")

        # Table 2: Hungarian -> Hungarian
        f.write("## 2. Hungarian $\\leftrightarrow$ Hungarian (Complex Round-Trip Invariance)\n\n")
        f.write("| ID | Type | Input Sentence / Paragraph | Reconstructed Hungarian | Preservation Rate | Merkle Root (BLAKE3) |\n")
        f.write("|:---|:-----|:---------------------------|:------------------------|:------------------|:---------------------|\n")
        for r in hu_hu_records:
            t_badge = "📖 Paragraph" if r["type"] == "paragraph" else "Sentence"
            f.write(f"| {r['id']:02d} | {t_badge} | `{r['input_text']}` | `{r['output_text']}` | **{r['slot_preservation_rate']:.1%}** | `{r['merkle_root'][:16]}...` |\n")

        f.write("\n---\n\n")

        # Table 3: English -> Hungarian
        f.write("## 3. English $\\rightarrow$ Hungarian (Complex Agglutinative Mapping)\n\n")
        f.write("| ID | Type | Source English | Target Hungarian (Vowel Harmony + Suffixes) | Validation Gate | Merkle Root |\n")
        f.write("|:---|:-----|:---------------|:--------------------------------------------|:----------------|:------------|\n")
        for r in eng_hu_records:
            t_badge = "📖 Paragraph" if r["type"] == "paragraph" else "Sentence"
            status = "✅ PASS" if r["is_valid"] else "❌ FAIL"
            f.write(f"| {r['id']:02d} | {t_badge} | `{r['input_text']}` | `{r['output_text']}` | {status} | `{r['merkle_root'][:16]}...` |\n")

        f.write("\n---\n\n")

        # Table 4: Hungarian -> English
        f.write("## 4. Hungarian $\\rightarrow$ English (Complex Morphological Parsing to SVO Realization)\n\n")
        f.write("| ID | Type | Source Hungarian | Target English | Validation Gate | Merkle Root |\n")
        f.write("|:---|:-----|:-----------------|:---------------|:----------------|:------------|\n")
        for r in hu_eng_records:
            t_badge = "📖 Paragraph" if r["type"] == "paragraph" else "Sentence"
            status = "✅ PASS" if r["is_valid"] else "❌ FAIL"
            f.write(f"| {r['id']:02d} | {t_badge} | `{r['input_text']}` | `{r['output_text']}` | {status} | `{r['merkle_root'][:16]}...` |\n")

        f.write("\n---\n\n")
        f.write("### Detailed Sample Trace: Multi-Sentence Paragraph Discourse with Cohesion & Anaphoric Backreferencing\n\n")
        sample_para = eng_hu_records[-1]
        f.write(f"**English Source Paragraph:**\n> {sample_para['input_text']}\n\n")
        f.write(f"**Hungarian Target Paragraph:**\n> {sample_para['output_text']}\n\n")
        f.write(f"**BLAKE3 Merkle Root:** `{sample_para['merkle_root']}`  \n\n")
        f.write("**Discourse Active Slots (Band 0–3):**\n```json\n")
        f.write(json.dumps(sample_para["active_slots"], indent=2))
        f.write("\n```\n")

    print(f"\n================================================================================")
    print(f"Successfully generated all complex translation benchmark examples!")
    print(f"Files created in {output_dir.resolve()}:")
    print(f"  - complex_examples_eng_eng.json")
    print(f"  - complex_examples_hu_hu.json")
    print(f"  - complex_examples_eng_hu.json")
    print(f"  - complex_examples_hu_eng.json")
    print(f"  - complex_translation_graphs_eng_eng.md")
    print(f"  - complex_translation_graphs_eng_hu.md")
    print(f"  - complex_translation_graphs_hu_eng.md")
    print(f"  - complex_translation_graphs_hu_hu.md")
    print(f"  - all_complex_translation_graphs.md")
    print(f"  - complex_translation_examples_summary.md")
    print("================================================================================")


if __name__ == "__main__":
    out = REPO_ROOT / "output"
    if len(sys.argv) > 1:
        out = Path(sys.argv[1]).resolve()
    generate_all_complex_examples(out)
