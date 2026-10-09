# QUANTA Phase 0 Baseline Report (`exp-026`) — Split: dev

- **Timestamp**: 2026-10-09T15:58:04.454430
- **Total evaluations**: 24

## 1. Primary Metrics by Condition (Mean ± 95% Bootstrap CI)

| Condition | Samples | TTFT (ms) | TTC (ms) | Full Ingest (s) | Throughput (w/s) | Gold Recall | Accuracy (EM%) | F1 Score |
|---|---|---|---|---|---|---|---|---|
| **B0_run1** | 6 | 1149.0 [839.5, 1588.1] | 223.9 [9.1, 646.2] | 31.520 [24.872, 38.814] | 79.4 [69.0, 94.3] | 36.1% [11.1%, 63.9%] | 16.7% [0.0%, 50.0%] | 0.133 [0.000, 0.400] |
| **B0_run2** | 6 | 472.2 [449.4, 503.6] | 9.6 [5.6, 13.8] | 30.671 [24.253, 38.058] | 81.9 [70.3, 97.9] | 36.1% [11.1%, 63.9%] | 16.7% [0.0%, 50.0%] | 0.133 [0.000, 0.400] |
| **FULLCTX0** | 6 | 990.0 [728.1, 1253.9] | 279.3 [255.8, 303.7] | 0.001 [0.001, 0.001] | 4990666.7 [3865333.3, 6128666.7] | 100.0% [100.0%, 100.0%] | 50.0% [16.7%, 83.3%] | 0.328 [0.083, 0.572] |
| **RAG0** | 6 | 891.6 [748.7, 992.5] | 1.2 [1.0, 1.4] | 0.001 [0.001, 0.001] | 2495333.3 [1932666.7, 3064333.3] | 80.6% [58.3%, 100.0%] | 100.0% [100.0%, 100.0%] | 0.656 [0.556, 0.756] |

## 2. B0 Run-to-Run Noise Analysis & delta Margin Calibration

Evaluating variance between B0_run1 and B0_run2 on identical inputs to inform the non-inferiority margin delta:

- **Accuracy Drift Mean (B0_run1 - B0_run2)**: `+0.00%`
- **95% Bootstrap CI of Drift**: `[+0.00%, +0.00%]`
- **Recommended delta (Non-inferiority margin)**: `delta >= 0.0%` (must not be smaller than run noise)
- **TTFT Run Noise Mean**: `+676.73 ms`

## 3. B0 Stage-Level Latency Breakdown

| Stage | Mean Duration (ms) | % of TTC | GPU Calls |
|---|---|---|---|
| `transduction` | 30936.10 ms | 26495.5% | 6.3 (mean) |
| `reader` | 796.88 ms | 682.5% | 1 |
| `PPR` | 112.38 ms | 96.2% | 0 |
| `chunking` | 17.59 ms | 15.1% | 0 |
| `context assembly` | 4.35 ms | 3.7% | 0 |
| `Kev` | 0.33 ms | 0.3% | 0 |
| `Clingo-DL` | 0.00 ms | 0.0% | 0 |

---

## 4. Parameter Registry Provenance

| Parameter | Active Value | Default (Pass-through) | Source Experiment | Source Commit | Calibrated On | Notes |
|---|---|---|---|---|---|---|
| `background.enabled` | `False` | `False` | *Uncalibrated (B0)* | - | - | Whether asynchronous background ingestion is enabled |
| `background.max_concurrency` | `1` | `1` | *Uncalibrated (B0)* | - | - | Maximum concurrent background worker threads |
| `background.pause_policy` | `none` | `none` | *Uncalibrated (B0)* | - | - | Background worker pause policy: none, foreground_lock, slot_polling |
| `chunker.macro_target_tokens` | *None* | *None* | *Uncalibrated (B0)* | - | - | Target token length for macro discourse blocks |
| `chunker.micro_target_words` | *None* | *None* | *Uncalibrated (B0)* | - | - | Target word length for micro discourse chunks |
| `fast_path.coverage_threshold` | *None* | *None* | *Uncalibrated (B0)* | - | - | Decision threshold for coverage check |
| `fast_path.hot_transduce_n` | *None* | *None* | *Uncalibrated (B0)* | - | - | Number of top-ranked units synchronously transduced |
| `fast_path.mode` | `passthrough` | `passthrough` | *Uncalibrated (B0)* | - | - | Fast-path mode: raw_only, hot_transduce, full, or passthrough |
| `filter.calibration` | *None* | *None* | *Uncalibrated (B0)* | - | - | Calibration mapping (temperature or Platt coefficients) |
| `filter.keep_budget_tokens` | *None* | *None* | *Uncalibrated (B0)* | - | - | Token budget cap for kept units |
| `filter.keep_threshold` | *None* | *None* | *Uncalibrated (B0)* | - | - | Confidence/probability threshold to keep a unit |
| `filter.keep_top_k` | *None* | *None* | *Uncalibrated (B0)* | - | - | Maximum number of units to keep |
| `filter.strategy` | `passthrough` | `passthrough` | *Uncalibrated (B0)* | - | - | Scorer variant: lexical, concept, kev_micro, kev_macro, kev_head, or passthrough |
| `filter.unit` | `micro` | `micro` | *Uncalibrated (B0)* | - | - | Scoring granularity unit: micro, macro, or macro_head |
| `poprag.coarse_node_weight` | *None* | *None* | *Uncalibrated (B0)* | - | - | PPR damping weight for coarse macro nodes |
| `ppr.inter_scale_weight` | *None* | *None* | *Uncalibrated (B0)* | - | - | Edge weight connecting micro chunks to coarse macro nodes |
| `query_extractor.strategy` | `passthrough` | `passthrough` | *Uncalibrated (B0)* | - | - | Strategy for isolating query from context (passthrough/QE-A control) |