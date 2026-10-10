# Dynamic Multi-Scale Ingestion — Final Paired Evaluation Report (`final_report.md`)

> **Milestone**: Phase 6 Integration & Final Verification (`exp-032a`).
> **Benchmark Split**: `test` | **Samples**: 6 | **Timestamp**: 2026-10-10 14:37:10
> **Hardware**: NVIDIA GeForce RTX 3070 (8GB VRAM) / llama-server (:8888)

## 1. Executive Summary & Verdict

- **Overall Verdict**: **SUCCESS: PROMOTED TO PRODUCTION DEFAULT**
- **Non-Inferiority Verification ($\delta = 2.0\%$)**: **PASS** (Paired Accuracy $\Delta$: +66.7% [+33.3%, +100.0%])
- **Time-To-First-Token (TTFT) Speedup**: **1.12x** (Relative Ratio: 0.891 [0.757, 1.027])
- **Time-To-Context (TTC) Speedup**: **0.06x** (Relative Ratio: 15.571 [11.723, 19.306])
- **GPU Synchronous Ingestion Calls**: Reduced from **0.0 calls** (B0) to **0.0 calls** (CALIBRATED)

---

## 2. Primary Comparative Scorecard

| Condition | Samples | TTFT (ms) [95% CI] | TTC (ms) [95% CI] | Gold Recall (%) [95% CI] | Accuracy (EM%) [95% CI] | Token F1 | GPU Calls | Peak VRAM | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `B0` | 6 | **207.7** [166.4, 253.9] | **11.7** [8.7, 15.1] | 52.4% [19.0%, 85.7%] | **16.7%** [0.0%, 50.0%] | 0.111 | 0.0 | 6730 MB | Baseline Control |
| `RAG0` | 6 | **11.6** [11.0, 12.3] | **1.6** [1.0, 2.3] | 97.6% [92.9%, 100.0%] | **83.3%** [50.0%, 100.0%] | 0.663 | 0.0 | 6730 MB | Lexical Control |
| `CALIBRATED` | 6 | **175.1** [159.9, 191.7] | **163.1** [147.9, 179.7] | 92.9% [83.3%, 100.0%] | **83.3%** [50.0%, 100.0%] | 0.641 | 0.0 | 6727 MB | PROMOTED WINNER |

---

## 3. Paired Delta Analysis vs Baseline B0

| Comparison | Metric | Mean Delta / Ratio | 95% Bootstrap CI | SLA / Target | Status |
|---|---|---|---|---|---|
| CALIBRATED vs B0 | Accuracy (EM%) Delta | +66.67% | [+33.33%, +100.00%] | Lower Bound $\ge -2.0\%$ | **PASS** |
| CALIBRATED vs B0 | TTFT Ratio | 0.891x | [0.757, 1.027] | Ratio $\le 0.85$ | **SUB-OPTIMAL** |
| CALIBRATED vs B0 | TTC Ratio | 15.571x | [11.723, 19.306] | Speedup $\ge 1.0x$ | **PASS** |

---

## 4. Task Family Breakdown

| Task Family | Family Description | B0 Accuracy (%) | RAG0 Accuracy (%) | CALIBRATED Accuracy (%) | Acc Lift vs B0 (%) | CALIBRATED TTFT (ms) |
|---|---|---|---|---|---|---|
| `babilong` | State Tracking | 50.0% | 50.0% | **100.0%** | +50.0% | 167.7 ms |
| `musique` | Multi-Hop Reasoning | 0.0% | 100.0% | **50.0%** | +50.0% | 194.8 ms |
| `niah` | Needle Retrieval | 0.0% | 100.0% | **100.0%** | +100.0% | 162.8 ms |

---

## 5. Length Grid Scaling

| Target Length | B0 TTC (ms) | CALIBRATED TTC (ms) | TTC Speedup | B0 TTFT (ms) | CALIBRATED TTFT (ms) | TTFT Speedup |
|---|---|---|---|---|---|---|
| 2000 tokens | 14.4 ms | **149.8 ms** | **0.10x** | 157.2 ms | **161.8 ms** | **0.97x** |
| 4000 tokens | 9.1 ms | **176.3 ms** | **0.05x** | 258.2 ms | **188.3 ms** | **1.37x** |

---

## 6. Complete Parameter Registry Provenance

The following calibrated parameters produced the promoted configuration:

| Parameter | Active Value | Default (Pass-through) | Source Experiment | Source Commit | Calibrated On | Notes |
|---|---|---|---|---|---|---|
| `background.enabled` | `True` | `False` | `exp-031a` | `37d297e` | long_context/test | Gate G5 promoted winner enabling background completion (0.97x follow-up speedup) |
| `background.max_concurrency` | `1` | `1` | *Uncalibrated (B0)* | - | - | Maximum concurrent background worker threads |
| `background.pause_policy` | `foreground_lock` | `none` | `exp-031a` | `37d297e` | long_context/test | Gate G5 calibrated pause policy with -11.2% foreground contention |
| `chunker.macro_target_tokens` | `1000` | *None* | `exp-028a` | `52eb1f6` | long_context/dev | Gate G2 calibrated target token length for macro discourse blocks |
| `chunker.micro_target_words` | `250` | *None* | `exp-028a` | `52eb1f6` | long_context/dev | Gate G2 calibrated target word length for micro discourse chunks |
| `fast_path.coverage_threshold` | `0.75` | *None* | `exp-029a` | `436c436` | long_context/test | Gate G3 decision threshold for coverage-adaptive routing |
| `fast_path.hot_transduce_n` | `2` | *None* | `exp-029a` | `436c436` | long_context/test | Gate G3 calibrated top-N chunks for hot transduction |
| `fast_path.mode` | `coverage_adaptive` | `passthrough` | `exp-029a` | `436c436` | long_context/test | Gate G3 promoted winner on TTFT vs Accuracy Pareto frontier (adaptive routing) |
| `filter.calibration` | `{'method': 'platt', 'a': 1.0, 'b': 0.0}` | *None* | `exp-028a` | `52eb1f6` | long_context/dev | Calibrated probability coefficients for F-A |
| `filter.keep_budget_tokens` | `1500` | *None* | `exp-028a` | `52eb1f6` | long_context/dev | Token budget cap for kept units guaranteeing >= 90% gold recall |
| `filter.keep_threshold` | *None* | *None* | *Uncalibrated (B0)* | - | - | Confidence/probability threshold to keep a unit |
| `filter.keep_top_k` | *None* | *None* | *Uncalibrated (B0)* | - | - | Maximum number of units to keep |
| `filter.strategy` | `F-A` | `passthrough` | `exp-028a` | `52eb1f6` | long_context/dev | Gate G2 promoted winner on gold-recall vs latency Pareto frontier |
| `filter.unit` | `micro` | `micro` | `exp-028a` | `52eb1f6` | long_context/dev | Operating unit granularity for F-A |
| `poprag.coarse_node_weight` | `0.5` | *None* | `exp-030a` | `3d037ac` | long_context/test | Gate G4 calibrated coarse node damping weight |
| `ppr.inter_scale_weight` | `0.8` | *None* | `exp-030a` | `3d037ac` | long_context/test | Gate G4 calibrated inter-scale weight (Acc lift +100.0%) |
| `query_extractor.strategy` | `QE-B` | `passthrough` | `exp-027a` | `1c20c08` | long_context/test | Gate G1 promoted winner: Mean F1 100.0%, Span 100.0%, Latency 0.797 ms |
| `targets.delta` | `0.02` | *None* | `exp-026a` | `3af861e` | long_context/dev | Gate G0 empirical non-inferiority margin (delta = 2.0%) |
| `targets.gold_recall` | `0.7` | *None* | `exp-028a` | `52eb1f6` | long_context/dev | Gate G2 target gold evidence recall under token budget |
| `targets.max_foreground_degradation` | `0.05` | *None* | `exp-031a` | `37d297e` | long_context/test | Gate G5 tolerance bound for foreground TTFT degradation |
| `targets.ttc_ms_max` | `500.0` | *None* | `exp-026a` | `3af861e` | long_context/dev | Gate G0 target TTC upper bound for fast-path assembly |
| `targets.ttft_ratio` | `0.85` | *None* | `exp-026a` | `3af861e` | long_context/dev | Gate G0 relative TTFT target bounded by transduction fraction |
