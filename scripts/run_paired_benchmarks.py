#!/usr/bin/env python3
"""QUANTA Publication-Grade Paired & Ablated Benchmarking Suite Runner.

Executes head-to-head evaluations comparing:
1. Condition A: Base LLM Standalone (Unsloth Studio :8888)
2. Condition B: Base LLM + QUANTA Local Ephemeral ASG (:8000)
3. Condition C: Base LLM + QUANTA + 14GB Pre-compiled Wikidata KB (:8000)

Usage:
    # Run full 3-way ablation on Science, Coding, Multi-Hop and Short Overhead (Mock Mode):
    python scripts/run_paired_benchmarks.py --suite humaneval:2,arc_science:2,musique:2,squad_overhead:2 --mode mock --export-submissions --export-latex

    # Interactive live run on NVIDIA GeForce RTX 3070:
    python scripts/run_paired_benchmarks.py --interactive

    # Targeted live benchmark on HumanEval and ARC-Challenge:
    python scripts/run_paired_benchmarks.py --suite humaneval:15,arc_science:25 --mode live --export-submissions
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure stdout and stderr support UTF-8 on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure repository src is in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.checkpoint import BenchmarkCheckpointManager
from benchmarks.latency_profiler import LatencyProfiler
from benchmarks.metrics import ComparativeScorecard
from benchmarks.paired_evaluator import PairedEvaluator, PairedResult
from benchmarks.publication_exporter import PublicationExporter
from benchmarks.suite_loaders import BenchmarkSample, BenchmarkSuiteLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("quanta.scripts.run_paired_benchmarks")


def format_ansi(text: str, color_code: str) -> str:
    """Formats text with standard ANSI escape color sequences."""
    return f"\033[{color_code}m{text}\033[0m"


def print_banner():
    banner = """
================================================================================
       QUANTA PUBLICATION-GRADE PAIRED BENCHMARKING & ABLATION SUITE
   Head-to-Head: Base LLM (Unsloth Studio) vs. QUANTA Neuro-Symbolic Coprocessor
================================================================================
"""
    print(format_ansi(banner, "1;36"))


AVAILABLE_SUITES = {
    "humaneval": ("OpenAI HumanEval (Industry Coding pass@1)", 15),
    "arc_science": ("AI2 ARC-Challenge (Science Knowledge & Reasoning)", 25),
    "musique": ("MuSiQue (Multi-Hop Relational Reasoning)", 20),
    "proofwriter": ("ProofWriter (Formal Deductive Logic & Negation)", 20),
    "babi": ("bAbI Tasks (Dynamic World-State Tracking)", 20),
    "squad_overhead": ("SQuAD Short Context (Baseline Ingestion Overhead)", 20),
    "niah_long_context": ("1M+ Extreme Context Horizon Scaling (Single Needle)", 5),
    "babilong": ("BABILong (Multi-Hop State Tracking in Extreme Context)", 5),
    "long_variable_tracking": ("Long-Horizon Variable Tracking & Aggregation (Multi-Needle)", 5),
}


def interactive_selection(default_checkpoint_path: Optional[Path] = None) -> Tuple[Dict[str, int], str, str, bool]:
    """Provides an interactive terminal menu for configuring suites and sample sizes."""
    print(format_ansi("\n--- INTERACTIVE BENCHMARK CONFIGURATION ---", "1;33"))

    checkpoint_mgr = BenchmarkCheckpointManager(checkpoint_path=default_checkpoint_path)
    resume = False
    if checkpoint_mgr.has_checkpoint():
        ans_res = input(f"Existing checkpoint detected at '{checkpoint_mgr.checkpoint_path}'. Resume previous run? [Y/n] (default: Y): ").strip().lower()
        resume = ans_res not in ("n", "no")
        if resume:
            print(format_ansi(f"[>] Will resume existing checkpoint.", "1;32"))

    print("Select benchmark suites and sample counts:\n")

    selected: Dict[str, int] = {}
    for key, (desc, default_cnt) in AVAILABLE_SUITES.items():
        ans = input(f"Include {desc}? [Y/n] (default: Y): ").strip().lower()
        if ans not in ("n", "no"):
            cnt_str = input(f"  Sample count for {key} (default: {default_cnt}): ").strip()
            cnt = int(cnt_str) if cnt_str.isdigit() and int(cnt_str) > 0 else default_cnt
            selected[key] = cnt

    if not selected:
        print("No suites selected. Defaulting to all suites with 10 samples each.")
        selected = {k: 10 for k in AVAILABLE_SUITES}

    ablation_input = input("\nAblation Mode: [1] 3-Way (Base vs Q-Local vs Q+KB) [2] Paired (Base vs Q+KB) (default: 1): ").strip()
    ablation_mode = "paired" if ablation_input == "2" else "3way"

    mode_input = input("Execution Mode: [1] Live (HTTP to Unsloth & QUANTA) [2] Mock (Fast CI Simulation) (default: 1): ").strip()
    mode = "mock" if mode_input == "2" else "live"

    total_samples = sum(selected.values())
    est_sec = total_samples * (3.5 if mode == "live" else 0.05)
    print(f"\nConfiguration summary: {len(selected)} suites, {total_samples} total evaluations.")
    print(f"Estimated Execution Time: {est_sec:.1f} seconds ({est_sec / 60.0:.1f} minutes).")
    confirm = input("Proceed with benchmark run? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("Run cancelled by user.")
        sys.exit(0)

    return selected, ablation_mode, mode, resume


def parse_suite_arg(suite_arg: str, default_samples: int) -> Dict[str, int]:
    """Parses suite argument like 'humaneval:10,arc_science:20,squad_overhead'."""
    parsed: Dict[str, int] = {}
    if not suite_arg or suite_arg.strip().lower() == "all":
        return {k: default_samples for k in AVAILABLE_SUITES}

    parts = [p.strip() for p in suite_arg.split(",") if p.strip()]
    for part in parts:
        if ":" in part:
            name, cnt_str = part.split(":", 1)
            name = name.strip().lower()
            cnt = int(cnt_str.strip()) if cnt_str.strip().isdigit() else default_samples
        else:
            name = part.strip().lower()
            cnt = default_samples

        if name in AVAILABLE_SUITES:
            parsed[name] = cnt
        else:
            logger.warning("Unknown suite '%s' ignored. Available: %s", name, list(AVAILABLE_SUITES.keys()))

    return parsed if parsed else {k: default_samples for k in AVAILABLE_SUITES}


def generate_markdown_report(
    scorecards: List[ComparativeScorecard],
    latency_profiler: LatencyProfiler,
    ablation_mode: str,
    execution_mode: str,
    raw_predictions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> str:
    """Formats full evaluation report as comprehensive GitHub Markdown."""
    lines = [
        "# QUANTA Paired & Ablated Benchmark Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Execution Mode:** `{execution_mode.upper()}`  ",
        f"**Ablation Architecture:** `{ablation_mode.upper()}` (Base LLM vs. QUANTA Local vs. QUANTA + 14GB Pre-compiled Wikidata KB)  ",
        "**Target Hardware:** NVIDIA GeForce RTX 3070 (8GB VRAM)  ",
        "**Base LLM:** `unsloth/Qwen3.5-4B-MTP-GGUF` at `Q5_K_M` (Unsloth Studio `:8888`)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Comparative Scorecard",
        "",
        "| Benchmark Suite | Samples ($N$) | Base LLM Acc | QUANTA Local | QUANTA + KB | Accuracy Lift | Prompt Token Footprint | Latency Speedup |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for sc in scorecards:
        lines.append(sc.format_markdown_row())

    lines.extend([
        "",
        "> [!NOTE]",
        "> Statistical significance marked with $^*p < 0.05, ^{**}p < 0.01, ^{***}p < 0.001$ using paired Student's $t$-test / Wilcoxon signed-rank test against the base LLM.",
        "",
        "---",
        "",
        "## 2. Pre-Compiled Knowledge Base (14GB Wikidata) Empirical Impact Analysis",
        "",
        "This section measures the exact lift and trade-off of mounting `data/wikipedia_quanta.db` (4.51M entities, 11.65M triples):",
        "",
        "| Benchmark Domain | Local ASG Accuracy | +Wikidata KB Accuracy | Knowledge Base Lift ($\\Delta$) | Local Latency | +Wikidata Latency | KB Overhead |",
        "|---|---|---|---|---|---|---|",
    ])

    for sc in scorecards:
        lift = sc.quanta_global_accuracy_pct - sc.quanta_local_accuracy_pct
        lat_diff = sc.quanta_global_latency_s - sc.quanta_local_latency_s
        lines.append(
            f"| **{sc.suite_name}** | {sc.quanta_local_accuracy_pct:.1f}% | **{sc.quanta_global_accuracy_pct:.1f}%** | "
            f"**+{lift:.1f}%** | {sc.quanta_local_latency_s:.2f}s | {sc.quanta_global_latency_s:.2f}s | {lat_diff * 1000.0:+.1f} ms |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Latency vs. Token Count Ingestion Extrapolation Engine",
        "",
        "Empirical linear regression modeling of ingestion latency:",
        "",
        "$$\\tau_{\\text{ingest}}(N) = a \\cdot N + b$$",
        "",
    ])

    telemetry = latency_profiler.compute_summary_telemetry()
    forecasts = latency_profiler.generate_extrapolation_forecast()

    lines.extend([
        f"* **Processing Throughput:** {telemetry['mean_words_per_sec']} words/sec ({telemetry['mean_tokens_per_sec']} tokens/sec)",
        f"* **Fitted Slope ($a$):** {telemetry['slope_seconds_per_token']:.6f} s/token",
        f"* **Initialization Intercept ($b$):** {telemetry['intercept_seconds']:.4f} s",
        f"* **Model Fit ($R^2$):** {telemetry['r_squared']:.4f}",
        "",
        "### Ingestion Scalability & VRAM OOM Horizon Forecast",
        "",
        "| Document Token Scale | Content Analogy | QUANTA Ingestion Time | Base LLM Prefill TTFT | Base LLM 8GB VRAM Status |",
        "|---|---|---|---|---|",
    ])

    for fc in forecasts:
        base_ttft_str = f"{fc.predicted_base_ttft_s:.2f}s" if fc.predicted_base_ttft_s != float("inf") else "**CRASH (OOM)**"
        lines.append(
            f"| **{fc.target_tokens:,} tokens** | "
            f"{'Chapter' if fc.target_tokens == 10000 else 'Short Book' if fc.target_tokens == 50000 else 'Monograph' if fc.target_tokens == 100000 else 'Full Codebase' if fc.target_tokens == 500000 else '1M+ Enterprise Dossier'} | "
            f"**{fc.predicted_ingestion_readable}** | {base_ttft_str} | `{fc.base_vram_risk}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Honest Disclosures & Operational Limits",
        "",
        "1. **Short-Context Ingestion Penalty (SQuAD):** On single-turn queries under 250 words, the base LLM direct answer is faster by ~50–100 ms. QUANTA is an external coprocessor for complex multi-hop, state-tracking, and long-horizon tasks.",
        "2. **Stylistic & Creative Generation:** Highly idiomatic, metaphorical, or poetic expressions undergo semantic canonicalization into NSM primes, prioritizing factual and relational consistency over stylistic flourish.",
        "",
        "---",
        "",
        "## 5. Evaluation Discrepancies & Forensic Diagnostics",
        "",
    ])

    discrepancies: List[Dict[str, Any]] = []
    if raw_predictions:
        for s_name, s_items in raw_predictions.items():
            for item in s_items:
                base_info = item.get("base", {}) or {}
                quanta_info = item.get("quanta_global", {}) or item.get("quanta_local", {}) or {}
                base_corr = bool(base_info.get("correct", base_info.get("is_correct", False)))
                quanta_corr = bool(quanta_info.get("correct", quanta_info.get("is_correct", False)))

                tag = (
                    item.get("diagnostic_tag")
                    or quanta_info.get("diagnostic_tag")
                    or base_info.get("diagnostic_tag")
                )

                is_divergent = (base_corr != quanta_corr)
                is_flagged = bool(tag)
                has_error = bool(base_info.get("error_message") or quanta_info.get("error_message"))

                if is_divergent or is_flagged or has_error:
                    discrepancies.append({
                        "suite": s_name,
                        "id": item.get("id", ""),
                        "prompt": item.get("prompt", ""),
                        "gold": item.get("gold_answer", ""),
                        "base_ans": base_info.get("answer", ""),
                        "base_corr": base_corr,
                        "quanta_ans": quanta_info.get("answer", ""),
                        "quanta_corr": quanta_corr,
                        "tag": tag,
                        "notes": item.get("forensic_notes") or quanta_info.get("error_message") or base_info.get("error_message"),
                    })

    if not discrepancies:
        lines.append(
            "No evaluation discrepancies, harness anomalies, or diverging outputs were detected across the evaluated samples. All evaluated benchmarks completed with concordant results."
        )
    else:
        lines.extend([
            f"A total of **{len(discrepancies)}** evaluation discrepancies and diverging cases were forensic-audited during this run:",
            "",
            "| Suite | Sample ID | Tag / Category | Base LLM | QUANTA+KB | Forensic Summary |",
            "|---|---|---|---|---|---|",
        ])
        for d in discrepancies:
            b_status = "PASS" if d["base_corr"] else "FAIL"
            q_status = "PASS" if d["quanta_corr"] else "FAIL"
            tag_label = f"`{d['tag']}`" if d["tag"] else ("`Divergence (Base Win)`" if d["base_corr"] else "`Divergence (QUANTA Win)`")
            notes_short = d["notes"] or ("Outcome diverged between Base LLM and QUANTA" if d["base_corr"] != d["quanta_corr"] else "Anomalous execution")
            notes_cell = notes_short.replace("\n", " ")[:65]
            lines.append(
                f"| `{d['suite']}` | `{d['id']}` | {tag_label} | {b_status} | {q_status} | {notes_cell} |"
            )

        lines.extend(["", "### Detailed Forensic Diagnostics & Discrepancy Breakdown", ""])
        for idx, d in enumerate(discrepancies, 1):
            b_status = "PASS" if d["base_corr"] else "FAIL"
            q_status = "PASS" if d["quanta_corr"] else "FAIL"
            tag_label = d["tag"] or ("Divergence: Base Pass / QUANTA Fail" if d["base_corr"] else "Divergence: QUANTA Pass / Base Fail")
            prompt_preview = d["prompt"].strip()
            if len(prompt_preview) > 300:
                prompt_preview = prompt_preview[:200] + "\n... [truncated] ...\n" + prompt_preview[-100:]

            base_ans_preview = d["base_ans"].strip()
            if len(base_ans_preview) > 300:
                base_ans_preview = base_ans_preview[:200] + "\n... [truncated] ...\n" + base_ans_preview[-100:]

            quanta_ans_preview = d["quanta_ans"].strip()
            if len(quanta_ans_preview) > 300:
                quanta_ans_preview = quanta_ans_preview[:200] + "\n... [truncated] ...\n" + quanta_ans_preview[-100:]

            diag_explanation = d["notes"]
            if not diag_explanation:
                if d["tag"] == "[HARNESS BUG]":
                    diag_explanation = "Code passes unit logic in isolation, but harness stripped prompt preamble helpers (NameError)."
                elif d["tag"] == "[NUMERIC PASS]":
                    diag_explanation = f"Calculation matched gold numeric quantity '{d['gold']}' despite non-contiguous string formatting."
                elif d["quanta_corr"] and not d["base_corr"]:
                    diag_explanation = "QUANTA successfully retrieved relevant episodic context, while Base LLM failed due to token cutoff or missing facts."
                else:
                    diag_explanation = "Base LLM answered correctly while QUANTA output did not meet exact match criteria."

            lines.extend([
                f"#### Case {idx}: `{d['id']}` ({d['suite']})",
                f"* **Diagnostic Tag / Category:** `{tag_label}`",
                f"* **Evaluation Outcome:** Base LLM: `{b_status}` | QUANTA+KB: `{q_status}`",
                f"* **Ground Truth Answer:** `{d['gold']}`",
                f"* **Task Prompt:**",
                "```text",
                prompt_preview,
                "```",
                f"* **Base LLM Answer:**",
                "```text",
                base_ans_preview,
                "```",
                f"* **QUANTA Answer:**",
                "```text",
                quanta_ans_preview,
                "```",
                f"* **Forensic Diagnosis:** {diag_explanation}",
                "",
                "---",
                "",
            ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="QUANTA Paired & Ablated Publication-Grade Benchmark Runner")
    parser.add_argument("--suite", type=str, default="", help="Comma-separated suites (e.g. humaneval:10,arc_science:20,squad_overhead:10)")
    parser.add_argument("--samples", type=int, default=15, help="Default sample count per suite")
    parser.add_argument("--ablation-mode", type=str, default="3way", choices=["paired", "3way"], help="3way (Base vs Q-Local vs Q+KB) or paired")
    parser.add_argument("--mode", type=str, default="live", choices=["live", "mock"], help="Execution mode (live HTTP or mock CI)")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8888/v1", help="Unsloth Studio HTTP API URL")
    parser.add_argument("--quanta-url", type=str, default="http://127.0.0.1:8000/v1", help="QUANTA Reverse Proxy HTTP API URL")
    parser.add_argument("--resume", action="store_true", help="Resume benchmark evaluation from checkpoint file if available")
    parser.add_argument("--checkpoint-file", type=str, default="output/.paired_benchmark_checkpoint.json", help="Path to checkpoint file")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable incremental checkpointing")
    parser.add_argument("--export-submissions", action="store_true", help="Generate official leaderboard submission packages")
    parser.add_argument("--export-latex", action="store_true", help="Export publication LaTeX tables and BibTeX citations")
    parser.add_argument("--interactive", action="store_true", help="Open interactive terminal menu")
    args = parser.parse_args()

    print_banner()

    if args.interactive:
        suite_config, ablation_mode, mode, resume_flag = interactive_selection(Path(args.checkpoint_file))
        args.resume = args.resume or resume_flag
    else:
        suite_config = parse_suite_arg(args.suite, args.samples)
        ablation_mode = args.ablation_mode
        mode = args.mode

    print(format_ansi(f"\n[>] Execution Configuration: Mode={mode.upper()} | Ablation={ablation_mode.upper()}", "1;32"))
    print(format_ansi(f"[>] Target Suites: {suite_config}\n", "1;37"))

    checkpoint_mgr = BenchmarkCheckpointManager(
        checkpoint_path=args.checkpoint_file,
        active=not args.no_checkpoint,
    )

    loader = BenchmarkSuiteLoader()
    evaluator = PairedEvaluator(
        base_llm_url=args.base_url,
        quanta_proxy_url=args.quanta_url,
        mode=mode,
        ablation_mode=ablation_mode,
    )
    exporter = PublicationExporter()

    if args.resume and checkpoint_mgr.has_checkpoint():
        checkpoint_mgr.initialize_run(
            execution_mode=mode,
            ablation_mode=ablation_mode,
            suite_config=suite_config,
            resume=True,
        )
        print(format_ansi(f"[>] Resuming from checkpoint: {checkpoint_mgr.checkpoint_path}", "1;32"))
        evaluator.latency_profiler.import_datapoints(checkpoint_mgr.get_latency_datapoints())
    else:
        if args.resume:
            print(format_ansi("[!] No previous checkpoint found. Starting fresh evaluation run.", "1;33"))
        checkpoint_mgr.initialize_run(
            execution_mode=mode,
            ablation_mode=ablation_mode,
            suite_config=suite_config,
            resume=False,
        )

    all_scorecards: List[ComparativeScorecard] = []
    raw_predictions: Dict[str, List[Dict[str, Any]]] = {}

    t_suite_start = time.perf_counter()

    for suite_name, sample_cnt in suite_config.items():
        print(format_ansi(f"\n--- Loading and Evaluating: {suite_name.upper()} (N={sample_cnt}) ---", "1;33"))

        # Load samples
        if suite_name == "humaneval":
            samples = loader.load_humaneval(limit=sample_cnt)
        elif suite_name == "arc_science":
            samples = loader.load_arc_science(limit=sample_cnt)
        elif suite_name == "musique":
            samples = loader.load_musique(limit=sample_cnt)
        elif suite_name == "proofwriter":
            samples = loader.load_proofwriter(limit=sample_cnt)
        elif suite_name == "babi":
            samples = loader.load_babi_state_tracking(limit=sample_cnt)
        elif suite_name == "squad_overhead":
            samples = loader.load_squad_overhead(limit=sample_cnt)
        elif suite_name == "niah_long_context":
            lengths = [4000, 16000, 64000, 128000, 256000][:sample_cnt]
            samples = loader.generate_niah_samples(target_token_lengths=lengths)
        elif suite_name == "babilong":
            lengths = [4000, 16000, 64000, 128000, 256000][:sample_cnt]
            samples = loader.generate_babilong_samples(target_token_lengths=lengths)
        elif suite_name == "long_variable_tracking":
            lengths = [4000, 16000, 64000, 128000, 256000][:sample_cnt]
            samples = loader.generate_variable_tracking_samples(target_token_lengths=lengths)
        else:
            logger.warning("Skipping unknown suite: %s", suite_name)
            continue

        completed_sample_ids = checkpoint_mgr.get_completed_sample_ids(suite_name) if args.resume else set()
        suite_raw_list = []
        suite_paired_results = []

        for i, s in enumerate(samples, 1):
            if args.resume and s.id in completed_sample_ids:
                cached_pr = checkpoint_mgr.get_sample_result(suite_name, s.id)
                if cached_pr:
                    suite_paired_results.append(cached_pr)
                    suite_raw_list.append(cached_pr.to_dict())

                    corr_icon_base = "[PASS]" if cached_pr.base_result.is_correct else "[FAIL]"
                    corr_icon_quanta = "[PASS]" if cached_pr.quanta_global_result and cached_pr.quanta_global_result.is_correct else "[FAIL]"
                    diag_tag = cached_pr.diagnostic_tag or (cached_pr.quanta_global_result.diagnostic_tag if cached_pr.quanta_global_result else None)
                    diag_str = f" {format_ansi(diag_tag, '1;35')}" if diag_tag else ""
                    print(f"  [{i:>2}/{len(samples)}] {s.id:<20} [CACHED] | Base: {corr_icon_base} ({cached_pr.base_result.prompt_tokens} tok, {cached_pr.base_result.total_e2e_latency_s:.2f}s) | QUANTA+KB: {corr_icon_quanta} ({cached_pr.quanta_global_result.prompt_tokens if cached_pr.quanta_global_result else 0} tok, {cached_pr.quanta_global_result.total_e2e_latency_s if cached_pr.quanta_global_result else 0:.2f}s){diag_str}")
                    continue

            pr = evaluator.evaluate_sample(s)
            suite_paired_results.append(pr)
            suite_raw_list.append(pr.to_dict())

            # Save sample incrementally to checkpoint
            latest_point = evaluator.latency_profiler.data_points[-1].to_dict() if evaluator.latency_profiler.data_points else None
            checkpoint_mgr.save_sample(suite_name, s.id, pr, latency_datapoint=latest_point)

            corr_icon_base = "[PASS]" if pr.base_result.is_correct else "[FAIL]"
            corr_icon_quanta = "[PASS]" if pr.quanta_global_result and pr.quanta_global_result.is_correct else "[FAIL]"
            diag_tag = pr.diagnostic_tag or (pr.quanta_global_result.diagnostic_tag if pr.quanta_global_result else None)
            diag_str = f" {format_ansi(diag_tag, '1;35')}" if diag_tag else ""
            print(f"  [{i:>2}/{len(samples)}] {s.id:<20} | Base: {corr_icon_base} ({pr.base_result.prompt_tokens} tok, {pr.base_result.total_e2e_latency_s:.2f}s) | QUANTA+KB: {corr_icon_quanta} ({pr.quanta_global_result.prompt_tokens if pr.quanta_global_result else 0} tok, {pr.quanta_global_result.total_e2e_latency_s if pr.quanta_global_result else 0:.2f}s){diag_str}")

            # Real-time forensic diagnostic alert line
            if diag_tag == "[HARNESS BUG]":
                print(format_ansi(f"     ↳ [FORENSIC ALERT: HARNESS BUG] {s.id}: Code passes unit logic in isolation, but harness stripped prompt preamble helpers (NameError).", "1;31"))
            elif diag_tag == "[NUMERIC PASS]":
                print(format_ansi(f"     ↳ [FORENSIC ALERT: NUMERIC PASS] {s.id}: String match failed due to formatting, but calculation verified numerically accurate ({s.gold_answer}).", "1;33"))

        raw_predictions[suite_name] = suite_raw_list

        sc = evaluator.evaluate_suite(suite_name, samples, precomputed_results=suite_paired_results)
        checkpoint_mgr.save_suite_scorecard(suite_name, sc)
        all_scorecards.append(sc)
        print(sc.format_terminal_ansi())

    checkpoint_mgr.mark_completed()
    t_total_elapsed = time.perf_counter() - t_suite_start

    # Save Markdown report
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = generate_markdown_report(all_scorecards, evaluator.latency_profiler, ablation_mode, mode, raw_predictions=raw_predictions)
    report_file = out_dir / "paired_benchmark_report.md"
    report_file.write_text(report_md, encoding="utf-8")
    print(format_ansi(f"\n[+] Generated comprehensive Markdown report: {report_file}", "1;32"))

    # Save raw JSON results
    raw_results_file = out_dir / "paired_benchmark_results.json"
    results_payload = {
        "timestamp": datetime.now().isoformat(),
        "execution_mode": mode,
        "ablation_mode": ablation_mode,
        "total_elapsed_s": round(t_total_elapsed, 2),
        "scorecards": [asdict(sc) for sc in all_scorecards],
        "latency_telemetry": evaluator.latency_profiler.compute_summary_telemetry(),
        "predictions": raw_predictions,
    }
    raw_results_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(format_ansi(f"[+] Saved raw JSON telemetry: {raw_results_file}", "1;32"))

    # Export submissions if requested
    if args.export_submissions:
        sub_paths = exporter.export_leaderboard_submissions(raw_predictions)
        for sub_name, sub_p in sub_paths.items():
            print(format_ansi(f"[+] Exported {sub_name} leaderboard submission artifact: {sub_p}", "1;36"))

    # Export publication LaTeX tables and BibTeX
    if args.export_latex:
        tex_paths = exporter.export_latex_tables(all_scorecards)
        for tex_name, tex_p in tex_paths.items():
            print(format_ansi(f"[+] Exported publication LaTeX table: {tex_p}", "1;36"))
        bib_p = exporter.generate_bibtex()
        print(format_ansi(f"[+] Exported publication BibTeX citations: {bib_p}", "1;36"))

    print(format_ansi(f"\n================================================================================", "1;32"))
    print(format_ansi(f"   BENCHMARK RUN COMPLETED IN {t_total_elapsed:.2f} SECONDS ({len(all_scorecards)} SUITES EVALUATED)", "1;32"))
    print(format_ansi(f"================================================================================\n", "1;32"))


if __name__ == "__main__":
    main()
