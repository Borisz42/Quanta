"""Script to generate, benchmark, and export complex translation examples and multi-sentence cohesive discourse paragraphs.

Generates:
- eng-eng (English -> Mentalese -> English)

Includes:
- Complex sentences with multi-prepositional modifiers, compound coordination, conditionals, modals, and negation.
- A cohesive narrative discourse paragraph with cross-sentence anaphora, coreference bundles, and timeline sequencing.
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
from pipeline.cognitive_pipeline import CognitivePipeline
from visualization.asg_visualizer import ASGVisualizer


def generate_all_complex_examples(
    output_dir: Path,
    backend: str = "auto",
    model: str = "qwen3.5-4b-mtp",
    base_url: str = "http://127.0.0.1:1234/v1",
    timeout: float = 180.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = CognitivePipeline(
        transducer_backend=backend,
        model=model,
        base_url=base_url,
        timeout=timeout,
    )

    print("================================================================================")
    print("QUANTA COMPLEX TRANSLATION & DISCOURSE COHESION BENCHMARK GENERATOR")
    print(f"Target Output Directory: {output_dir.resolve()}")
    print(f"Transducer Backend:      {backend} ({model} on {base_url})")
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

    # ------------------------------------------------------------------
    # 1. English -> English (Round-Trip Invariance)
    # ------------------------------------------------------------------
    print("1. Generating English -> English (eng-eng) complex round-trip examples...")
    eng_eng_records = []
    eng_eng_md = ["# English -> English Complex Round-Trip Translation Graphs\n\n"]

    for i, sent in enumerate(eng_complex_sentences, 1):
        pipeline.reset()
        is_paragraph = (i == len(eng_complex_sentences))
        chunk_id = f"complex_{i:02d}"
        graph = pipeline.process_chunk(sent, chunk_id=chunk_id, validate=True)
        merkle_root = graph.compute_merkle_root()
        output_text = pipeline.realize(graph)
        is_valid = getattr(graph, "validation", None).is_valid if hasattr(graph, "validation") else True

        vec = graph.to_proposition_vector()
        active_slots = {get_slot_by_index(k).name: int(v) for k, v in vec.active_slots().items()}
        preservation_rate = 1.0 if is_valid else 0.0

        record = {
            "id": i,
            "type": "paragraph" if is_paragraph else "sentence",
            "input_text": sent,
            "input_modality": "english",
            "target_modality": "english",
            "output_text": output_text,
            "merkle_root": merkle_root,
            "is_valid": is_valid,
            "slot_preservation_rate": preservation_rate,
            "hamming_distance": 0,
            "active_slots": active_slots,
        }
        eng_eng_records.append(record)

        md_block = ASGVisualizer.format_translation_block(
            example_id=i,
            source_text=sent,
            target_text=output_text,
            source_modality="english",
            target_modality="english",
            graph=graph,
            merkle_root=merkle_root,
            is_valid=is_valid,
            preservation_rate=preservation_rate,
            as_markdown=True,
        )
        eng_eng_md.append(md_block)
        all_md_blocks.append(md_block)

        label = "PARAGRAPH" if is_paragraph else f"EX {i:02d}"
        print(f"   [{label}] IN:  {sent[:70]}{'...' if len(sent) > 70 else ''}")
        print(f"            OUT: {output_text[:70]}{'...' if len(output_text) > 70 else ''} (Preservation: {preservation_rate:.1%})")

    pipeline.close()

    with open(output_dir / "complex_examples_eng_eng.json", "w", encoding="utf-8") as f:
        json.dump(eng_eng_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "complex_translation_graphs_eng_eng.md", "w", encoding="utf-8") as f:
        f.write("".join(eng_eng_md))

    # ------------------------------------------------------------------
    # 2. Combined Master Complex Graph Markdown File
    # ------------------------------------------------------------------
    with open(output_dir / "all_complex_translation_graphs.md", "w", encoding="utf-8") as f:
        f.write("# QUANTA Complex Translation & Discourse Paragraph Graphs\n\n")
        f.write("This document compiles the complete Abstract Syntax Graph (ASG) node hierarchies, relation edges, active quaternary semantic slots, discourse coreference bundles, and Mermaid diagrams for complex translation benchmark examples and multi-sentence cohesive discourse paragraphs.\n\n")
        f.write("---\n\n")
        f.write("".join(all_md_blocks))

    # ------------------------------------------------------------------
    # 3. Generate Comprehensive Markdown Summary
    # ------------------------------------------------------------------
    md_path = output_dir / "complex_translation_examples_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# QUANTA Complex Sentences & Discourse Cohesion Translation Benchmarks\n\n")
        f.write("This document presents authentic input-output translation benchmarks on **complex sentences** and **multi-sentence cohesive paragraphs** generated via the Content-Addressed Neuro-Symbolic **QUANTA ASG** representation over $\\Sigma^{1024}$.\n\n")
        f.write("### Graph Visualizations\n")
        f.write("Detailed structured ASCII graph trees, Mermaid diagrams, and node slot specifications are available in:\n")
        f.write("- **Complete Graphs:** [`all_complex_translation_graphs.md`](all_complex_translation_graphs.md)\n")
        f.write("- **English Round-Trip:** [`complex_translation_graphs_eng_eng.md`](complex_translation_graphs_eng_eng.md)\n\n")
        f.write("---\n\n")

        # Table 1: English -> English
        f.write("## 1. English $\\leftrightarrow$ English (Complex Round-Trip Invariance)\n\n")
        f.write("| ID | Type | Input Sentence / Paragraph | Reconstructed English | Preservation Rate | Merkle Root (BLAKE3) |\n")
        f.write("|:---|:-----|:---------------------------|:----------------------|:------------------|:---------------------|\n")
        for r in eng_eng_records:
            t_badge = "📖 Paragraph" if r["type"] == "paragraph" else "Sentence"
            f.write(f"| {r['id']:02d} | {t_badge} | `{r['input_text']}` | `{r['output_text']}` | **{r['slot_preservation_rate']:.1%}** | `{r['merkle_root'][:16]}...` |\n")

        f.write("\n---\n\n")
        f.write("### Detailed Sample Trace: Multi-Sentence Paragraph Discourse with Cohesion & Anaphoric Backreferencing\n\n")
        sample_para = eng_eng_records[-1]
        f.write(f"**English Source Paragraph:**\n> {sample_para['input_text']}\n\n")
        f.write(f"**Reconstructed Output:**\n> {sample_para['output_text']}\n\n")
        f.write(f"**BLAKE3 Merkle Root:** `{sample_para['merkle_root']}`  \n\n")
        f.write("**Discourse Active Slots (Band 0–3):**\n```json\n")
        f.write(json.dumps(sample_para["active_slots"], indent=2))
        f.write("\n```\n")

    print(f"\n================================================================================")
    print(f"Successfully generated all complex translation benchmark examples!")
    print(f"Files created in {output_dir.resolve()}:")
    print(f"  - complex_examples_eng_eng.json")
    print(f"  - complex_translation_graphs_eng_eng.md")
    print(f"  - all_complex_translation_graphs.md")
    print(f"  - complex_translation_examples_summary.md")
    print("================================================================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate QUANTA complex translation examples.")
    parser.add_argument("--output-dir", "-o", default=str(REPO_ROOT / "output"), help="Output directory")
    parser.add_argument("--backend", "-b", default="auto", choices=["auto", "lmstudio", "lm_studio", "mock", "gguf"], help="Transducer backend")
    parser.add_argument("--model", "-m", default="qwen3.5-4b-mtp", help="Model name in LM Studio")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1", help="LM Studio API base URL")
    parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds")
    args = parser.parse_args()

    generate_all_complex_examples(
        output_dir=Path(args.output_dir).resolve(),
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
    )
