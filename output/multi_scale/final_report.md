# Dynamic Multi-Scale Ingestion — Final Paired Evaluation Report (`final_report.md`)

> **Milestone**: Phase 6 Integration & Final Verification (`exp-032a`).
> **Benchmark Split**: `test` | **Samples**: 6 | **Timestamp**: 2026-10-10 14:51:51
> **Hardware**: NVIDIA GeForce RTX 3070 (8GB VRAM) / llama-server (:8888)

## 1. Executive Summary & Verdict

- **Overall Verdict**: **INCONCLUSIVE: REVIEW WITH USER**
- **Non-Inferiority Verification ($\delta = 2.0\%$)**: **FAIL** (Paired Accuracy $\Delta$: +16.7% [-33.3%, +66.7%])
- **Time-To-First-Token (TTFT) Speedup**: **2.84x** (Relative Ratio: 0.352 [0.249, 0.468])
- **Time-To-Context (TTC) Speedup**: **0.00x** (Relative Ratio: 648.917 [461.841, 839.915])
- **GPU Synchronous Ingestion Calls**: Reduced from **8.0 calls** (B0) to **3.0 calls** (CALIBRATED)

---

## 2. Primary Comparative Scorecard

| Condition | Samples | TTFT (ms) [95% CI] | TTC (ms) [95% CI] | Gold Recall (%) [95% CI] | Accuracy (EM%) [95% CI] | Token F1 | GPU Calls | Peak VRAM | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `B0` | 6 | **18383.6** [15432.7, 21576.6] | **8.4** [7.2, 10.4] | 67.9% [36.9%, 92.9%] | **66.7%** [33.3%, 100.0%] | 0.381 | 8.0 | 6911 MB | Baseline Control |
| `RAG0` | 6 | **862.5** [839.3, 881.2] | **1.1** [0.9, 1.3] | 97.6% [92.9%, 100.0%] | **83.3%** [50.0%, 100.0%] | 0.587 | 1.0 | 6896 MB | Lexical Control |
| `CALIBRATED` | 6 | **5959.9** [4999.3, 7030.4] | **5050.4** [4117.7, 6095.2] | 92.9% [83.3%, 100.0%] | **83.3%** [50.0%, 100.0%] | 0.437 | 3.0 | 6896 MB | PROMOTED WINNER |

---

## 3. Paired Delta Analysis vs Baseline B0

| Comparison | Metric | Mean Delta / Ratio | 95% Bootstrap CI | SLA / Target | Status |
|---|---|---|---|---|---|
| CALIBRATED vs B0 | Accuracy (EM%) Delta | +16.67% | [-33.33%, +66.67%] | Lower Bound $\ge -2.0\%$ | **FAIL** |
| CALIBRATED vs B0 | TTFT Ratio | 0.352x | [0.249, 0.468] | Ratio $\le 0.85$ | **PASS** |
| CALIBRATED vs B0 | TTC Ratio | 648.917x | [461.841, 839.915] | Speedup $\ge 1.0x$ | **PASS** |

---

## 4. Task Family Breakdown

| Task Family | Family Description | B0 Accuracy (%) | RAG0 Accuracy (%) | CALIBRATED Accuracy (%) | Acc Lift vs B0 (%) | CALIBRATED TTFT (ms) |
|---|---|---|---|---|---|---|
| `babilong` | State Tracking | 50.0% | 50.0% | **100.0%** | +50.0% | 4598.7 ms |
| `musique` | Multi-Hop Reasoning | 100.0% | 100.0% | **50.0%** | -50.0% | 7346.7 ms |
| `niah` | Needle Retrieval | 50.0% | 100.0% | **100.0%** | +50.0% | 5934.2 ms |

---

## 5. Length Grid Scaling

| Target Length | B0 TTC (ms) | CALIBRATED TTC (ms) | TTC Speedup | B0 TTFT (ms) | CALIBRATED TTFT (ms) | TTFT Speedup |
|---|---|---|---|---|---|---|
| 2000 tokens | 9.5 ms | **5591.0 ms** | **0.00x** | 14521.5 ms | **6496.0 ms** | **2.24x** |
| 4000 tokens | 7.4 ms | **4509.9 ms** | **0.00x** | 22245.7 ms | **5423.8 ms** | **4.10x** |

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
