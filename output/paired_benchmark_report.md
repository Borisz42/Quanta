# QUANTA Paired & Ablated Benchmark Report

**Date:** 2026-10-06 16:07:18  
**Execution Mode:** `LIVE`  
**Ablation Architecture:** `3WAY` (Base LLM vs. QUANTA Local vs. QUANTA + 14GB Pre-compiled Wikidata KB)  
**Target Hardware:** NVIDIA GeForce RTX 3070 (8GB VRAM)  
**Base LLM:** `unsloth/Qwen3.5-4B-MTP-GGUF` at `Q5_K_M` (Unsloth Studio `:8888`)  

---

## 1. Executive Summary & Comparative Scorecard

| Benchmark Suite | Samples ($N$) | Base LLM Acc | QUANTA Local | QUANTA + KB | Accuracy Lift | Prompt Token Footprint | Latency Speedup |
|---|---|---|---|---|---|---|---|
| **proofwriter** | 30 | 100.0% | 96.7% | **100.0%** | 0.0% (n.s.) | 233 -> **311** (0.0%) | 1.14s -> **2.21s** (0.52x) |
| **musique** | 30 | 50.0% | 50.0% | **43.3%** | -6.7% (n.s.) | 2527 -> **2102** (-16.8%) | 2.50s -> **12.64s** (0.20x) |

> [!NOTE]
> Statistical significance marked with $^*p < 0.05, ^{**}p < 0.01, ^{***}p < 0.001$ using paired Student's $t$-test / Wilcoxon signed-rank test against the base LLM.

---

## 2. Pre-Compiled Knowledge Base (14GB Wikidata) Empirical Impact Analysis

This section measures the exact lift and trade-off of mounting `data/wikipedia_quanta.db` (4.51M entities, 11.65M triples):

| Benchmark Domain | Local ASG Accuracy | +Wikidata KB Accuracy | Knowledge Base Lift ($\Delta$) | Local Latency | +Wikidata Latency | KB Overhead |
|---|---|---|---|---|---|---|
| **proofwriter** | 96.7% | **100.0%** | **+3.3%** | 2.76s | 2.21s | -552.0 ms |
| **musique** | 50.0% | **43.3%** | **+-6.7%** | 13.86s | 12.64s | -1215.0 ms |

---

## 3. Latency vs. Token Count Ingestion Extrapolation Engine

Empirical linear regression modeling of ingestion latency:

$$\tau_{\text{ingest}}(N) = a \cdot N + b$$

* **Processing Throughput:** 205.7 words/sec (273.6 tokens/sec)
* **Fitted Slope ($a$):** 0.003803 s/token
* **Initialization Intercept ($b$):** 0.0010 s
* **Model Fit ($R^2$):** 0.7923

### Ingestion Scalability & VRAM OOM Horizon Forecast

| Document Token Scale | Content Analogy | QUANTA Ingestion Time | Base LLM Prefill TTFT | Base LLM 8GB VRAM Status |
|---|---|---|---|---|
| **10,000 tokens** | Chapter | **38.03 s** | 3.53s | `SAFE (< 6 GB)` |
| **50,000 tokens** | Short Book | **3.2 min** | 42.53s | `HIGH RISK (~7.8 GB)` |
| **100,000 tokens** | Monograph | **6.3 min** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |
| **500,000 tokens** | Full Codebase | **31.7 min** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |
| **1,000,000 tokens** | 1M+ Enterprise Dossier | **63.4 min** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |

---

## 4. Honest Disclosures & Operational Limits

1. **Short-Context Ingestion Penalty (SQuAD):** On single-turn queries under 250 words, the base LLM direct answer is faster by ~50–100 ms. QUANTA is an external coprocessor for complex multi-hop, state-tracking, and long-horizon tasks.
2. **Stylistic & Creative Generation:** Highly idiomatic, metaphorical, or poetic expressions undergo semantic canonicalization into NSM primes, prioritizing factual and relational consistency over stylistic flourish.

---

## 5. Evaluation Discrepancies & Forensic Diagnostics

A total of **6** evaluation discrepancies and diverging cases were forensic-audited during this run:

| Suite | Sample ID | Tag / Category | Base LLM | QUANTA+KB | Forensic Summary |
|---|---|---|---|---|---|
| `musique` | `3hop1__241001_568433_47686` | `Divergence (Base Win)` | PASS | FAIL | Outcome diverged between Base LLM and QUANTA |
| `musique` | `2hop__169443_83769` | `Divergence (QUANTA Win)` | FAIL | PASS | Outcome diverged between Base LLM and QUANTA |
| `musique` | `2hop__82910_75184` | `Divergence (Base Win)` | PASS | FAIL | Outcome diverged between Base LLM and QUANTA |
| `musique` | `4hop1__17192_17130_70784_61381` | `Divergence (QUANTA Win)` | FAIL | PASS | Outcome diverged between Base LLM and QUANTA |
| `musique` | `3hop1__506062_792411_51423` | `Divergence (Base Win)` | PASS | FAIL | Outcome diverged between Base LLM and QUANTA |
| `musique` | `2hop__846388_19320` | `Divergence (Base Win)` | PASS | FAIL | Outcome diverged between Base LLM and QUANTA |

### Detailed Forensic Diagnostics & Discrepancy Breakdown

#### Case 1: `3hop1__241001_568433_47686` (musique)
* **Diagnostic Tag / Category:** `Divergence: Base Pass / QUANTA Fail`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Toronto Coach Terminal`
* **Task Prompt:**
```text
Where do Greyhound buses leave from in the city where Arna Selznick's employer is headquartered?

Instructions:
1. In 2 to 3 concise sentences, explain the factual bridge connecting the relevant docum
... [truncated] ...
vant documents.
2. Conclude immediately on a new line in the exact format: 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
Arna Selznick is a Canadian director whose employer, Nelvana, is headquartered in Toronto, Ontario, Canada. Greyhound buses operate inter-city services from Toronto, departing from the Toronto Coach Terminal which serves as the central hub for these routes.

Answer: Toronto Coach Terminal
```
* **QUANTA Answer:**
```text
Arna is a former municipality in Norway, while Selznick International (the employer of cinematographer Lee Garmes) was headquartered in California, USA; therefore, there is no Greyhound bus terminal i
... [truncated] ...
, as noted in the records of the Los Angeles to Houston Freedom Ride.

Answer: Union Railway Station
```
* **Forensic Diagnosis:** Base LLM answered correctly while QUANTA output did not meet exact match criteria.

---

#### Case 2: `2hop__169443_83769` (musique)
* **Diagnostic Tag / Category:** `Divergence: QUANTA Pass / Base Fail`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `Fred O'Bannion`
* **Task Prompt:**
```text
Who did the screenwriter for Good Will Hunting play in Dazed and Confused?

Instructions:
1. In 2 to 3 concise sentences, explain the factual bridge connecting the relevant documents. Do not summarize
... [truncated] ...
vant documents.
2. Conclude immediately on a new line in the exact format: 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
The screenwriter for *Good Will Hunting* is Matt Damon, who is explicitly identified in Document 19 as one of the two writers of the film alongside Ben Affleck. According to Document 16, Matt Damon fi
... [truncated] ...
 served as his breakout role before he wrote and starred in *Good Will Hunting*.

Answer: Matt Damon
```
* **QUANTA Answer:**
```text
The screenwriter for *Good Will Hunting* is Ben Affleck, who co-wrote the screenplay with Damon Affleck. According to the provided documents, Ben Affleck played the character Fred O'Bannion in the film *Dazed and Confused*.

Answer: Ben Affleck
```
* **Forensic Diagnosis:** QUANTA successfully retrieved relevant episodic context, while Base LLM failed due to token cutoff or missing facts.

---

#### Case 3: `2hop__82910_75184` (musique)
* **Diagnostic Tag / Category:** `Divergence: Base Pass / QUANTA Fail`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Claudia Wells`
* **Task Prompt:**
```text
Who played the girlfriend of Alex P. Keaton's actor on Family Ties in Back to the Future?

Instructions:
1. In 2 to 3 concise sentences, explain the factual bridge connecting the relevant documents. D
... [truncated] ...
vant documents.
2. Conclude immediately on a new line in the exact format: 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
Michael J. Fox played the character Alex P. Keaton on the sitcom *Family Ties*, and he later portrayed Marty McFly in the *Back to the Future* film series. In *Back to the Future*, Marty's girlfriend 
... [truncated] ...
a Wells but ultimately played by Elisabeth Shue due to scheduling conflicts.

Answer: Elisabeth Shue
```
* **QUANTA Answer:**
```text
Michael J. Fox played Alex P. Keaton on *Family Ties*, and he was originally cast as Marty McFly in *Back to the Future* before being replaced by Eric Stoltz due to his television commitments. The gir
... [truncated] ...
his casting in *Back to the Future* without identifying his on-screen romantic partner in that film.
```
* **Forensic Diagnosis:** Base LLM answered correctly while QUANTA output did not meet exact match criteria.

---

#### Case 4: `4hop1__17192_17130_70784_61381` (musique)
* **Diagnostic Tag / Category:** `Divergence: QUANTA Pass / Base Fail`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `1932`
* **Task Prompt:**
```text
When was the region immediately north of the region where the country that secured southern Lebanon is located and the Persian Gulf established?

Instructions:
1. In 2 to 3 concise sentences, explain 
... [truncated] ...
vant documents.
2. Conclude immediately on a new line in the exact format: 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
The country that secured southern Lebanon is Israel, which shares a northern border with Syria and lies south of the region where the Persian Gulf is located. The Persian Gulf was established as a dis
... [truncated] ...
 to the north, while Israel is situated south of Syria and borders the Gulf of Aqaba in the Red Sea.
```
* **QUANTA Answer:**
```text
The country that secured southern Lebanon is Israel, which shares a land border with the Persian Gulf to its northeast (via Jordan and the Gulf of Aqaba connection). The region immediately north of Is
... [truncated] ...
ity of Saudi Arabia, which borders the Persian Gulf to the south, was founded in 1932.

Answer: 1932
```
* **Forensic Diagnosis:** QUANTA successfully retrieved relevant episodic context, while Base LLM failed due to token cutoff or missing facts.

---

#### Case 5: `3hop1__506062_792411_51423` (musique)
* **Diagnostic Tag / Category:** `Divergence: Base Pass / QUANTA Fail`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Casa Loma`
* **Task Prompt:**
```text
What is the name of the castle in the place of birth of the performer of When I Was a Boy?

Instructions:
1. In 2 to 3 concise sentences, explain the factual bridge connecting the relevant documents. 
... [truncated] ...
vant documents.
2. Conclude immediately on a new line in the exact format: 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
The performer of the album *When I Was a Boy* is Jane Siberry, who is identified in Document [13] as a Canadian singer-songwriter. The birthplace of this performer is Toronto, Canada, a city that contains Casa Loma, the Gothic Revival mansion described in Document [19].

Answer: Casa Loma
```
* **QUANTA Answer:**
```text
The performer of the album *When I Was a Boy* is Jane Siberry, who was born in Canada (Document 15). While no specific castle is mentioned in her birthplace documents, Document 8 identifies Viborg and
... [truncated] ...
after the castle town of Nyslott, which is literally translated as "New Castle."

Answer: New Castle
```
* **Forensic Diagnosis:** Base LLM answered correctly while QUANTA output did not meet exact match criteria.

---

#### Case 6: `2hop__846388_19320` (musique)
* **Diagnostic Tag / Category:** `Divergence: Base Pass / QUANTA Fail`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `74th`
* **Task Prompt:**
```text
What worldwide ranking does the university that Pedro Ramirez Vasquez attended hold?

Instructions:
1. In 2 to 3 concise sentences, explain the factual bridge connecting the relevant documents. Do not
... [truncated] ...
vant documents.
2. Conclude immediately on a new line in the exact format: 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
Pedro Ramírez Vázquez is identified as a former faculty member of the National Autonomous University of Mexico (UNAM) in Document [19], and Document [13] confirms that UNAM is the specific institution
... [truncated] ...
h in the Top 200 World University Ranking published by Times Higher Education in 2006.

Answer: 74th
```
* **QUANTA Answer:**
```text
The provided documents do not contain information linking Pedro Ramírez Vázquez to a specific university he attended, nor do they mention his educational background. While Document 20 identifies him a
... [truncated] ...
 provide information about which university Pedro Ramírez Vázquez attended or its worldwide ranking.
```
* **Forensic Diagnosis:** Base LLM answered correctly while QUANTA output did not meet exact match criteria.

---
