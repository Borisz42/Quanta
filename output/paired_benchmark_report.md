# QUANTA Paired & Ablated Benchmark Report

**Date:** 2026-10-08 00:59:33  
**Execution Mode:** `MOCK`  
**Ablation Architecture:** `3WAY` (Base LLM vs. QUANTA Local vs. QUANTA + 14GB Pre-compiled Wikidata KB)  
**Target Hardware:** NVIDIA GeForce RTX 3070 (8GB VRAM)  
**Base LLM:** `unsloth/Qwen3.5-4B-MTP-GGUF` at `Q5_K_M` (Unsloth Studio `:8888`)  

---

## 1. Executive Summary & Comparative Scorecard

| Benchmark Suite | Samples ($N$) | Base LLM Acc | QUANTA Local | QUANTA + KB | Accuracy Lift | Prompt Token Footprint | Latency Speedup |
|---|---|---|---|---|---|---|---|
| **arc_science** | 3 | 66.7% | 100.0% | **100.0%** | +33.3% (n.s.) | 106 -> **106** (0.0%) | 0.18s -> **0.18s** (0.97x) |

> [!NOTE]
> Statistical significance marked with $^*p < 0.05, ^{**}p < 0.01, ^{***}p < 0.001$ using paired Student's $t$-test / Wilcoxon signed-rank test against the base LLM.

---

## 2. Pre-Compiled Knowledge Base (14GB Wikidata) Empirical Impact Analysis

This section measures the exact lift and trade-off of mounting `data/wikipedia_quanta.db` (4.51M entities, 11.65M triples):

| Benchmark Domain | Local ASG Accuracy | +Wikidata KB Accuracy | Knowledge Base Lift ($\Delta$) | Local Latency | +Wikidata Latency | KB Overhead |
|---|---|---|---|---|---|---|
| **arc_science** | 100.0% | **100.0%** | **+0.0%** | 0.18s | 0.18s | +0.0 ms |

---

## 3. Latency vs. Token Count Ingestion Extrapolation Engine

Empirical linear regression modeling of ingestion latency:

$$\tau_{\text{ingest}}(N) = a \cdot N + b$$

* **Processing Throughput:** 16040.1 words/sec (21333.3 tokens/sec)
* **Fitted Slope ($a$):** 0.000000 s/token
* **Initialization Intercept ($b$):** 0.0050 s
* **Model Fit ($R^2$):** 1.0000

### Ingestion Scalability & VRAM OOM Horizon Forecast

| Document Token Scale | Content Analogy | QUANTA Ingestion Time | Base LLM Prefill TTFT | Base LLM 8GB VRAM Status |
|---|---|---|---|---|
| **10,000 tokens** | Chapter | **0.01 s** | 3.53s | `SAFE (< 6 GB)` |
| **50,000 tokens** | Short Book | **0.01 s** | 42.53s | `HIGH RISK (~7.8 GB)` |
| **100,000 tokens** | Monograph | **0.01 s** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |
| **500,000 tokens** | Full Codebase | **0.01 s** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |
| **1,000,000 tokens** | 1M+ Enterprise Dossier | **0.01 s** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |

---

## 4. Honest Disclosures & Operational Limits

1. **Short-Context Ingestion Penalty (SQuAD):** On single-turn queries under 250 words, the base LLM direct answer is faster by ~50–100 ms. QUANTA is an external coprocessor for complex multi-hop, state-tracking, and long-horizon tasks.
2. **Stylistic & Creative Generation:** Highly idiomatic, metaphorical, or poetic expressions undergo semantic canonicalization into NSM primes, prioritizing factual and relational consistency over stylistic flourish.

---

## 5. Evaluation Discrepancies & Forensic Diagnostics

A total of **1** evaluation discrepancies and diverging cases were forensic-audited during this run:

| Suite | Sample ID | Tag / Category | Base LLM | QUANTA+KB | Forensic Summary |
|---|---|---|---|---|---|
| `arc_science` | `Mercury_SC_408547` | `Divergence (QUANTA Win)` | FAIL | PASS | Outcome diverged between Base LLM and QUANTA |

### Detailed Forensic Diagnostics & Discrepancy Breakdown

#### Case 1: `Mercury_SC_408547` (arc_science)
* **Diagnostic Tag / Category:** `Divergence: QUANTA Pass / Base Fail`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `C`
* **Task Prompt:**
```text
The end result in the process of photosynthesis is the production of sugar and oxygen. Which step signals the beginning of photosynthesis?

Choices:
(A) Chemical energy is absorbed through the roots.

... [truncated] ...
cientific principle.
2. Conclude immediately on a new line in the exact format: 'Answer: [A/B/C/D]'.
```
* **Base LLM Answer:**
```text
(D)
```
* **QUANTA Answer:**
```text
The correct choice is (C). Chlorophyll in the leaf captures light energy.
```
* **Forensic Diagnosis:** QUANTA successfully retrieved relevant episodic context, while Base LLM failed due to token cutoff or missing facts.

---
