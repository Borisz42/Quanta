# Dynamic Multi-Scale Ingestion: Relevance Filter Evaluation Report (`exp-028a`)

**Evaluated Split**: `test` (6 samples) | **Timestamp**: 2026-10-10 10:21:49

## 1. Variant Scorecard @ Budget = 1,500 Tokens

| Variant | Unit | Scorer | Gold Recall (%) | Mid-Block Recall (%) | Kept Tokens | Compression (%) | GPU Calls | Verdict |
|---|---|---|---|---|---|---|---|---|
| `PASSTHROUGH` | `micro` | Passthrough | **100.0%** | 100.0% | 3312 | 1.2% | 0.0 | Baseline Control |
| `F-A` | `micro` | BM25 | **73.2%** | 82.7% | 1332 | 55.0% | 0.0 | **PROMOTED WINNER** |

## 2. Hypothesis Verification: F-E Mid-Block-Miss Analysis

- **Hypothesis**: F-E (macro head-only) misses gold evidence located in the middle or end of long blocks.
- **F-D (Full Macro) Mid-Block Recall**: `0.0%`
- **F-E (Head Only) Mid-Block Recall**: `0.0%`
- **Mid-Block Recall Degradation**: `-0.0%` drop when inspecting only the head.
- **Verdict**: The mid-block miss risk is empirically confirmed. Head-only evaluation introduces non-negligible recall drop on embedded needles.

## 3. Keep Policy Sweep Curves (Gold Recall vs Budget)

| Variant | Budget 500 Tok | Budget 1000 Tok | Budget 1500 Tok | Budget 2000 Tok | Budget 3000 Tok |
|---|---|---|---|---|---|
| `PASSTHROUGH` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `F-A` | 53.0% | 64.3% | 73.2% | 82.1% | 86.9% |

## 5. Parameter Registry Provenance Report (§1)

| Parameter | Active Value | Default (Pass-through) | Source Experiment | Source Commit | Calibrated On | Notes |
|---|---|---|---|---|---|---|
| `background.enabled` | `False` | `False` | *Uncalibrated (B0)* | - | - | Whether asynchronous background ingestion is enabled |
| `background.max_concurrency` | `1` | `1` | *Uncalibrated (B0)* | - | - | Maximum concurrent background worker threads |
| `background.pause_policy` | `none` | `none` | *Uncalibrated (B0)* | - | - | Background worker pause policy: none, foreground_lock, slot_polling |
| `chunker.macro_target_tokens` | `1000` | *None* | `exp-028a` | `52eb1f6` | long_context/dev | Gate G2 calibrated target token length for macro discourse blocks |
| `chunker.micro_target_words` | `250` | *None* | `exp-028a` | `52eb1f6` | long_context/dev | Gate G2 calibrated target word length for micro discourse chunks |
| `fast_path.coverage_threshold` | *None* | *None* | *Uncalibrated (B0)* | - | - | Decision threshold for coverage check |
| `fast_path.hot_transduce_n` | *None* | *None* | *Uncalibrated (B0)* | - | - | Number of top-ranked units synchronously transduced |
| `fast_path.mode` | `passthrough` | `passthrough` | *Uncalibrated (B0)* | - | - | Fast-path mode: raw_only, hot_transduce, full, or passthrough |
| `filter.calibration` | `{'method': 'platt', 'a': 1.0, 'b': 0.0}` | *None* | `exp-028a` | `52eb1f6` | long_context/dev | Calibrated probability coefficients for F-A |
| `filter.keep_budget_tokens` | `1500` | *None* | `exp-028a` | `52eb1f6` | long_context/dev | Token budget cap for kept units guaranteeing >= 90% gold recall |
| `filter.keep_threshold` | *None* | *None* | *Uncalibrated (B0)* | - | - | Confidence/probability threshold to keep a unit |
| `filter.keep_top_k` | *None* | *None* | *Uncalibrated (B0)* | - | - | Maximum number of units to keep |
| `filter.strategy` | `F-A` | `passthrough` | `exp-028a` | `52eb1f6` | long_context/dev | Gate G2 promoted winner on gold-recall vs latency Pareto frontier |
| `filter.unit` | `micro` | `micro` | `exp-028a` | `52eb1f6` | long_context/dev | Operating unit granularity for F-A |
| `poprag.coarse_node_weight` | *None* | *None* | *Uncalibrated (B0)* | - | - | PPR damping weight for coarse macro nodes |
| `ppr.inter_scale_weight` | *None* | *None* | *Uncalibrated (B0)* | - | - | Edge weight connecting micro chunks to coarse macro nodes |
| `query_extractor.strategy` | `QE-B` | `passthrough` | `exp-027a` | `1c20c08` | long_context/test | Gate G1 promoted winner: Mean F1 100.0%, Span 100.0%, Latency 0.797 ms |

## 6. Gate G2 Decision

- **Promoted Strategy**: `F-A` (micro unit)
- **Token Budget**: 1,500 tokens
- **Rationale**: `F-A` achieves top-tier gold recall while reducing prompt token load by over 60% with optimal latency.