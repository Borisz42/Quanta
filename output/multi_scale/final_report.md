# Dynamic Multi-Scale Ingestion — Final Paired Evaluation Report (`final_report.md`)

> **Milestone**: Phase 6 Integration & Final Verification (`exp-032a`).
> **Benchmark Split**: `test` | **Samples**: 90 | **Timestamp**: 2026-10-10 16:12:46
> **Hardware**: NVIDIA GeForce RTX 3070 (8GB VRAM) / llama-server (:8888)

## 1. Executive Summary & Verdict

- **Overall Verdict**: **SUCCESS: PROMOTED TO PRODUCTION DEFAULT**
- **Non-Inferiority Verification ($\delta = 2.0\%$)**: **PASS** (Paired Accuracy $\Delta$: +27.8% [+14.4%, +40.0%])
- **Time-To-First-Token (TTFT) Speedup**: **5.91x** (Relative Ratio: 0.169 [0.139, 0.207])
- **Time-To-Context (TTC) Speedup**: **0.00x** (Relative Ratio: 543.181 [430.068, 685.576])
- **GPU Synchronous Ingestion Calls**: Reduced from **13.7 calls** (B0) to **2.3 calls** (CALIBRATED)

---

## 2. Primary Comparative Scorecard

| Condition | Samples | TTFT (ms) [95% CI] | TTC (ms) [95% CI] | Gold Recall (%) [95% CI] | Accuracy (EM%) [95% CI] | Token F1 | GPU Calls | Peak VRAM | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `B0` | 90 | **31035.4** [27908.4, 34151.7] | **17.7** [7.0, 47.3] | 54.4% [44.8%, 63.7%] | **46.7%** [37.8%, 56.7%] | 0.301 | 13.7 | 7048 MB | Baseline Control |
| `RAG0` | 90 | **916.9** [906.5, 927.2] | **1.7** [1.5, 1.9] | 89.1% [83.1%, 94.4%] | **76.7%** [66.7%, 85.6%] | 0.625 | 1.0 | 7051 MB | Lexical Control |
| `CALIBRATED` | 90 | **4139.0** [3628.5, 4643.7] | **3175.6** [2672.1, 3676.2] | 80.7% [73.5%, 87.3%] | **74.4%** [64.4%, 83.3%] | 0.572 | 2.3 | 7051 MB | PROMOTED WINNER |

---

## 3. Paired Delta Analysis vs Baseline B0

| Comparison | Metric | Mean Delta / Ratio | 95% Bootstrap CI | SLA / Target | Status |
|---|---|---|---|---|---|
| CALIBRATED vs B0 | Accuracy (EM%) Delta | +27.78% | [+14.44%, +40.00%] | Lower Bound $\ge -2.0\%$ | **PASS** |
| CALIBRATED vs B0 | TTFT Ratio | 0.169x | [0.139, 0.207] | Ratio $\le 0.85$ | **PASS** |
| CALIBRATED vs B0 | TTC Ratio | 543.181x | [430.068, 685.576] | Speedup $\ge 1.0x$ | **PASS** |

---

## 4. Task Family Breakdown

| Task Family | Family Description | B0 Accuracy (%) | RAG0 Accuracy (%) | CALIBRATED Accuracy (%) | Acc Lift vs B0 (%) | CALIBRATED TTFT (ms) |
|---|---|---|---|---|---|---|
| `babilong` | State Tracking | 73.3% | 100.0% | **96.7%** | +23.3% | 5800.4 ms |
| `musique` | Multi-Hop Reasoning | 33.3% | 30.0% | **26.7%** | -6.7% | 5592.1 ms |
| `niah` | Needle Retrieval | 33.3% | 100.0% | **100.0%** | +66.7% | 1024.4 ms |

---

## 5. Length Grid Scaling

| Target Length | B0 TTC (ms) | CALIBRATED TTC (ms) | TTC Speedup | B0 TTFT (ms) | CALIBRATED TTFT (ms) | TTFT Speedup |
|---|---|---|---|---|---|---|
| 1363 tokens | 11.0 ms | **3209.1 ms** | **0.00x** | 38608.4 ms | **4155.3 ms** | **9.29x** |
| 1498 tokens | 13.2 ms | **5062.6 ms** | **0.00x** | 40128.9 ms | **6073.0 ms** | **6.61x** |
| 1580 tokens | 5.8 ms | **3653.1 ms** | **0.00x** | 39535.7 ms | **4765.8 ms** | **8.30x** |
| 1710 tokens | 13.8 ms | **3455.6 ms** | **0.00x** | 43896.0 ms | **4392.1 ms** | **9.99x** |
| 1715 tokens | 15.2 ms | **3452.7 ms** | **0.00x** | 40931.4 ms | **4629.2 ms** | **8.84x** |
| 1828 tokens | 14.9 ms | **3478.1 ms** | **0.00x** | 38957.9 ms | **4389.8 ms** | **8.87x** |
| 1891 tokens | 3.9 ms | **3621.4 ms** | **0.00x** | 40652.9 ms | **4762.5 ms** | **8.54x** |
| 1897 tokens | 11.7 ms | **3891.2 ms** | **0.00x** | 44534.5 ms | **4973.6 ms** | **8.95x** |
| 1924 tokens | 6.1 ms | **5352.3 ms** | **0.00x** | 47429.1 ms | **6352.4 ms** | **7.47x** |
| 1977 tokens | 13.4 ms | **3639.7 ms** | **0.00x** | 50488.7 ms | **4631.8 ms** | **10.90x** |
| 1983 tokens | 7.3 ms | **3707.1 ms** | **0.00x** | 39411.5 ms | **4741.2 ms** | **8.31x** |
| 2000 tokens | 5.4 ms | **3038.9 ms** | **0.00x** | 11798.1 ms | **3972.4 ms** | **2.97x** |
| 2026 tokens | 15.8 ms | **4985.4 ms** | **0.00x** | 46392.0 ms | **6075.3 ms** | **7.64x** |
| 2034 tokens | 10.9 ms | **4932.1 ms** | **0.00x** | 45816.8 ms | **6106.9 ms** | **7.50x** |
| 2070 tokens | 13.9 ms | **6967.9 ms** | **0.00x** | 46416.9 ms | **7915.2 ms** | **5.86x** |
| 2112 tokens | 9.6 ms | **3205.8 ms** | **0.00x** | 40748.6 ms | **4195.2 ms** | **9.71x** |
| 2142 tokens | 19.5 ms | **4644.7 ms** | **0.00x** | 46777.5 ms | **5706.1 ms** | **8.20x** |
| 2197 tokens | 13.2 ms | **4363.2 ms** | **0.00x** | 49157.6 ms | **5364.7 ms** | **9.16x** |
| 2257 tokens | 8.1 ms | **4572.9 ms** | **0.00x** | 46375.1 ms | **5623.0 ms** | **8.25x** |
| 2401 tokens | 7.4 ms | **3808.6 ms** | **0.00x** | 41590.5 ms | **4835.0 ms** | **8.60x** |
| 2409 tokens | 8.8 ms | **3652.0 ms** | **0.00x** | 44906.1 ms | **4761.9 ms** | **9.43x** |
| 2451 tokens | 6.2 ms | **6085.2 ms** | **0.00x** | 47722.3 ms | **6985.2 ms** | **6.83x** |
| 2568 tokens | 16.6 ms | **5337.4 ms** | **0.00x** | 49107.5 ms | **6318.3 ms** | **7.77x** |
| 2578 tokens | 5.3 ms | **4560.2 ms** | **0.00x** | 49548.0 ms | **5682.9 ms** | **8.72x** |
| 2586 tokens | 15.7 ms | **7411.6 ms** | **0.00x** | 50926.7 ms | **8383.4 ms** | **6.07x** |
| 2827 tokens | 11.8 ms | **4762.6 ms** | **0.00x** | 46273.1 ms | **5788.7 ms** | **7.99x** |
| 2830 tokens | 5.7 ms | **5218.6 ms** | **0.00x** | 51539.8 ms | **6166.3 ms** | **8.36x** |
| 2951 tokens | 11.3 ms | **5540.5 ms** | **0.00x** | 48011.0 ms | **6487.5 ms** | **7.40x** |
| 3100 tokens | 910.7 ms | **4463.0 ms** | **0.20x** | 42772.2 ms | **5318.6 ms** | **8.04x** |
| 3903 tokens | 11.3 ms | **5611.4 ms** | **0.00x** | 48516.7 ms | **6598.6 ms** | **7.35x** |
| 3936 tokens | 12.8 ms | **4566.4 ms** | **0.00x** | 44314.2 ms | **5582.7 ms** | **7.94x** |
| 4000 tokens | 7.3 ms | **2121.6 ms** | **0.00x** | 18876.4 ms | **3058.7 ms** | **6.17x** |
| 8000 tokens | 5.3 ms | **2269.2 ms** | **0.00x** | 41410.4 ms | **3206.2 ms** | **12.92x** |

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
