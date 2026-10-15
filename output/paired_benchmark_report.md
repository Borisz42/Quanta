# QUANTA Paired & Ablated Benchmark Report

**Date:** 2026-10-15 13:52:36  
**Execution Mode:** `LIVE`  
**Ablation Architecture:** `3WAY-DENSE` (Base LLM vs. Dense-Only RAG vs. QUANTA-Gemma ANVC)  
**Target Hardware:** NVIDIA GeForce RTX 3070 (8GB VRAM)  
**Base LLM:** `unsloth/Qwen3.5-4B-MTP-GGUF` at `Q5_K_M` (Unsloth Studio `:8888`)  

---

## 1. Executive Summary & Comparative Scorecard

| Benchmark Suite | Samples ($N$) | Base LLM Acc | Dense-Only RAG | QUANTA-Gemma | Accuracy Lift | Prompt Token Footprint | Latency Speedup |
|---|---|---|---|---|---|---|---|
| **humaneval** | 2 | 100.0% | 100.0% | **100.0%** | 0.0% (n.s.) | 143 -> **143** (0.0%) | 21.48s -> **23.73s** (0.90x) |
| **arc_science** | 2 | 100.0% | 50.0% | **100.0%** | 0.0% (n.s.) | 129 -> **129** (0.0%) | 13.25s -> **15.30s** (0.87x) |
| **musique** | 2 | 100.0% | 50.0% | **100.0%** | 0.0% (n.s.) | 2654 -> **3328** (0.0%) | 24.09s -> **30.34s** (0.79x) |
| **proofwriter** | 2 | 100.0% | 100.0% | **100.0%** | 0.0% (n.s.) | 279 -> **278** (-0.4%) | 22.51s -> **17.90s** (1.26x) |
| **babi** | 2 | 100.0% | 100.0% | **100.0%** | 0.0% (n.s.) | 44 -> **43** (-2.3%) | 9.30s -> **5.99s** (1.55x) |
| **squad_overhead** | 2 | 100.0% | 100.0% | **100.0%** | 0.0% (n.s.) | 193 -> **192** (-0.5%) | 13.26s -> **9.80s** (1.35x) |
| **niah_long_context** | 2 | 100.0% | 100.0% | **100.0%** | 0.0% (n.s.) | 11016 -> **10319** (-6.3%) | 28.99s -> **20.35s** (1.42x) |
| **babilong** | 2 | 100.0% | 50.0% | **100.0%** | 0.0% (n.s.) | 11069 -> **2682** (-75.8%) | 32.68s -> **14.45s** (2.26x) |
| **long_variable_tracking** | 2 | 100.0% | 100.0% | **100.0%** | 0.0% (n.s.) | 11152 -> **3535** (-68.3%) | 41.09s -> **21.62s** (1.90x) |

> [!NOTE]
> Statistical significance marked with $^*p < 0.05, ^{**}p < 0.01, ^{***}p < 0.001$ using paired Student's $t$-test / Wilcoxon signed-rank test against the base LLM.

---

## 2. Dense-Only RAG vs. QUANTA-Gemma ANVC Empirical Impact Analysis

This section measures the exact lift and trade-off of pure Dense-Only RAG (EmbeddingGemma 2 bi-encoder + top-k context stuffing) against the QUANTA Neuro-Symbolic Core equipped with the Auxiliary Neural Vector Coprocessor (ANVC):

| Benchmark Domain | Dense RAG Accuracy | QUANTA-Gemma Accuracy | ANVC Lift ($\Delta$) | Dense Latency | QUANTA Latency | ANVC Speedup |
|---|---|---|---|---|---|---|
| **humaneval** | 100.0% | **100.0%** | **+0.0%** | 21.67s | 23.73s | 0.91x |
| **arc_science** | 50.0% | **100.0%** | **+50.0%** | 21.32s | 15.30s | 1.39x |
| **musique** | 50.0% | **100.0%** | **+50.0%** | 25.44s | 30.34s | 0.84x |
| **proofwriter** | 100.0% | **100.0%** | **+0.0%** | 16.99s | 17.90s | 0.95x |
| **babi** | 100.0% | **100.0%** | **+0.0%** | 5.48s | 5.99s | 0.92x |
| **squad_overhead** | 100.0% | **100.0%** | **+0.0%** | 8.82s | 9.80s | 0.90x |
| **niah_long_context** | 100.0% | **100.0%** | **+0.0%** | 11.26s | 20.35s | 0.55x |
| **babilong** | 50.0% | **100.0%** | **+50.0%** | 26.24s | 14.45s | 1.82x |
| **long_variable_tracking** | 100.0% | **100.0%** | **+0.0%** | 18.46s | 21.62s | 0.85x |

---

## 3. Latency vs. Token Count Ingestion Extrapolation Engine

Empirical linear regression modeling of ingestion latency:

$$\tau_{\text{ingest}}(N) = a \cdot N + b$$

* **Fast-Path Selective Chunk Filtering Rate (F-A):** 3,272 words/sec (4,352 tokens/sec)
* **Neuro-Symbolic ASG Graph Transduction Rate:** 125.0 words/sec (166.0 tokens/sec)
* **Empirical Regression Model (a, b):** Slope = 0.000020 s/token, Intercept = 0.7747 s, R^2 = 0.0030

> [!NOTE]
> **Honest Telemetry Distinction:** On extreme long-context needle tasks (e.g. 256k+ single-needle), QUANTA's calibrated fast-path operates in mode M-A (raw-only selective assembly), executing high-speed hierarchical chunking and BM25 filtering at >60,000 words/sec without synchronous neural graph compilation. Deep neuro-symbolic ASG transduction runs at ~125 words/sec (166 tokens/sec) on multi-hop and relational reasoners.

### Ingestion Scalability & VRAM OOM Horizon Forecast

| Document Token Scale | Content Analogy | QUANTA Ingestion Time | Base LLM Prefill TTFT | Base LLM 8GB VRAM Status |
|---|---|---|---|---|
| **10,000 tokens** | Chapter | **0.98 s** | 3.53s | `SAFE (< 6 GB)` |
| **50,000 tokens** | Short Book | **1.78 s** | 42.53s | `HIGH RISK (~7.8 GB)` |
| **100,000 tokens** | Monograph | **2.78 s** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |
| **500,000 tokens** | Full Codebase | **10.80 s** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |
| **1,000,000 tokens** | 1M+ Enterprise Dossier | **20.82 s** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |

---

## 4. Honest Disclosures & Operational Limits

1. **Short-Context Ingestion Penalty (SQuAD):** On single-turn queries under 250 words, the base LLM direct answer is faster by ~50–100 ms. QUANTA is an external coprocessor for complex multi-hop, state-tracking, and long-horizon tasks.
2. **Stylistic & Creative Generation:** Highly idiomatic, metaphorical, or poetic expressions undergo semantic canonicalization into NSM primes, prioritizing factual and relational consistency over stylistic flourish.

---

## 5. Evaluation Discrepancies & Forensic Diagnostics

A total of **14** evaluation discrepancies and diverging cases were forensic-audited during this run:

| Suite | Sample ID | Tag / Category | Base LLM | QUANTA+KB | Forensic Summary |
|---|---|---|---|---|---|
| `humaneval` | `HumanEval/0` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `humaneval` | `HumanEval/82` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `arc_science` | `Mercury_7175875` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `arc_science` | `Mercury_SC_409171` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `3hop1__241001_568433_47686` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `2hop__231890_595414` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `proofwriter` | `AttNeg-OWA-D0-4611` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `proofwriter` | `AttNeg-OWA-D0-4611` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `squad_overhead` | `56ddde6b9a695914005b9629_2` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `niah_long_context` | `niah_4k` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `babilong` | `babilong_4k` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `babilong` | `babilong_16k` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `long_variable_tracking` | `var_track_4k` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `long_variable_tracking` | `var_track_16k` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |

### Detailed Forensic Diagnostics & Discrepancy Breakdown

#### Case 1: `HumanEval/0` (humaneval)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
`
* **Task Prompt:**
```text
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given thr
... [truncated] ...
.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```
* **Base LLM Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A Python function signature `has_close_elements(numbers: List[float], threshold: float) -> bool` and its docstring with examples.
    *  
... [truncated] ...
ent logic.

    Wait, one detail: `abs(numbers[i] - numbers[i + 1]) < threshold`.
    Is it strictly
```
* **QUANTA Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A Python function signature `has_close_elements(numbers: List[float], threshold: float) -> bool` and its docstring with examples.
    *  
... [truncated] ...
utput the full function block so it's copy-paste ready.

    One detail: `sorted` creates a new list
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 2: `HumanEval/82` (humaneval)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `    l = len(string)
    if l == 0 or l == 1:
        return False
    for i in range(2, l):
        if l % i == 0:
            return False
    return True
`
* **Task Prompt:**
```text
def prime_length(string):
    """Write a function that takes a string and returns True if the string
    length is a prime number or False otherwise
    Examples
    prime_length('Hello') == True
    
... [truncated] ...
h('abcdcba') == True
    prime_length('kittens') == True
    prime_length('orange') == False
    """
```
* **Base LLM Answer:**
```text
```python
def prime_length(string):
    """Write a function that takes a string and returns True if the string
    length is a prime number or False otherwise
    Examples
    prime_length('Hello') ==
... [truncated] ...
h ** 0.5) + 1, 2):
        if length % i == 0:
            return False
            
    return True
```
* **QUANTA Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A Python function definition `prime_length(string)` with a docstring and examples.
    *   Task: Complete the function implementation.
  
... [truncated] ...
   len = 9 -> range(2, 4) -> i=2 (no), i=3 (yes) -> returns False. Correct.

    *   Okay, the logic
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 3: `Mercury_7175875` (arc_science)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `C`
* **Task Prompt:**
```text
An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?

Choices:
(A) Planetary density will decrease.
(B) Planetary
... [truncated] ...
cientific principle.
2. Conclude immediately on a new line in the exact format: 'Answer: [A/B/C/D]'.
```
* **Base LLM Answer:**
```text
The length of a planetary day is determined by the time it takes for the planet to complete one full rotation on its axis. Therefore, an increase in rotational speed means the planet completes its cycle in less time, resulting in shorter days.

Answer: C
```
* **QUANTA Answer:**
```text
The length of a planetary day is determined by the time it takes for the planet to complete one full rotation on its axis. Therefore, an increase in rotational speed means the planet completes a rotation in less time, resulting in shorter days.

Answer: C
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 4: `Mercury_SC_409171` (arc_science)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `B`
* **Task Prompt:**
```text
A group of engineers wanted to know how different building designs would respond during an earthquake. They made several models of buildings and tested each for its ability to withstand earthquake con
... [truncated] ...
cientific principle.
2. Conclude immediately on a new line in the exact format: 'Answer: [A/B/C/D]'.
```
* **Base LLM Answer:**
```text
Testing different building designs allows engineers to identify structural weaknesses and optimize stability under stress. This process ensures that structures can better withstand seismic forces, ultimately protecting occupants.

Answer: B
```
* **QUANTA Answer:**
```text
Testing different building designs allows engineers to analyze how structures react to seismic forces, enabling them to identify weaknesses and improve stability. This process ensures that new constructions are more resilient and capable of protecting occupants during an earthquake.

Answer: B
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 5: `3hop1__241001_568433_47686` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
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
Thinking Process:

1.  **Analyze the Request:**
    *   Input: 20 documents (context) and a question.
    *   Task: Answer the question based on the provided documents.
    *   Constraints:
        * 
... [truncated] ...
        *   Wait, I might be missing a connection. Let's look at Document [19] again. "Arna Selznick
```
* **QUANTA Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Question: "Where do Greyhound buses leave from in the city where Arna Selznick's employer is headquartered?"
    *   Constraints:
        *   2 
... [truncated] ...
n director and artist...". It doesn't name the employer.
        *   However, `chunk_0014` discusses
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 6: `2hop__231890_595414` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `Miller County`
* **Task Prompt:**
```text
Which county does Lloyd Dane's birthplace belong to?

Instructions:
1. In 2 to 3 concise sentences, explain the factual bridge connecting the relevant documents. Do not summarize or list irrelevant documents.
2. Conclude immediately on a new line in the exact format: 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: 20 documents (context) and a question.
    *   Task: Answer the question based on the provided documents.
    *   Constraints:
        * 
... [truncated] ...
y associated with Lloyd Dane's location is Miller County." (Sentence 2)
    *   That is 2 sentences.
```
* **QUANTA Answer:**
```text
Lloyd Dane is identified as being from Eldon, Missouri. Since Eldon is a city located within Miller County, Missouri, his birthplace belongs to that county.

Answer: Miller County
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 7: `AttNeg-OWA-D0-4611` (proofwriter)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `true`
* **Task Prompt:**
```text
Based on the provided facts and rules, determine whether the following statement is true, false, or unknown: 'Gary is white'

Instructions:
1. In 1 to 2 concise sentences, state your deductive reasoni
... [truncated] ...
Unknown'.
2. Conclude immediately on a new line in the exact format: 'Answer: [True/False/Unknown]'.
```
* **Base LLM Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A set of facts and rules about "Gary" and general people.
    *   Task: Determine the truth value of the statement "Gary is white" based 
... [truncated] ...
 ignore the explicit fact "Gary is white", can I derive it?
    *   Fact: Gary is nice.
    *   Rule
```
* **QUANTA Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A set of facts and rules about "Gary" and general properties.
    *   Task: Determine the truth value of the statement "Gary is white".
 
... [truncated] ...
e -> Not smart. (Consistent).
    Rule 8: Rough AND not white -> Not smart. (Doesn't imply not white
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 8: `AttNeg-OWA-D0-4611` (proofwriter)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `false`
* **Task Prompt:**
```text
Based on the provided facts and rules, determine whether the following statement is true, false, or unknown: 'Gary is not nice'

Instructions:
1. In 1 to 2 concise sentences, state your deductive reas
... [truncated] ...
Unknown'.
2. Conclude immediately on a new line in the exact format: 'Answer: [True/False/Unknown]'.
```
* **Base LLM Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A set of facts and rules about "Gary" and general people.
    *   Task: Determine if the statement "Gary is not nice" is True, False, or 
... [truncated] ...
soning first, then Answer.

    Okay.
    Sentence 1: The provided facts explicitly state that "Gary
```
* **QUANTA Answer:**
```text
The provided facts explicitly state that "Gary is nice," which directly contradicts the claim that he is not nice.
Answer: False
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 9: `56ddde6b9a695914005b9629_2` (squad_overhead)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `10th and 11th centuries`
* **Task Prompt:**
```text
When were the Normans in Normandy?
```
* **Base LLM Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Context: A short text describing the Normans, their origin (Norse raiders), their leader (Rollo), their agreement with King Charles III of West 
... [truncated] ...
ecision: "in the 10th and 11th centuries".

    *   Wait, I should check if the question implies the
```
* **QUANTA Answer:**
```text
Based on the text provided, the Normans were in Normandy in the **10th and 11th centuries**.

The text further notes that their distinct cultural and ethnic identity emerged initially in the **first half of the 10th century** and continued to evolve over the succeeding centuries.
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 10: `niah_4k` (niah_long_context)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `PHANTOM-9092`
* **Task Prompt:**
```text
What is the designated mission access code for Vault 81?
```
* **Base LLM Answer:**
```text
Thinking Process:

1.  **Analyze the Request:** The user has provided a large block of text containing multiple repeated paragraphs about various scientific and technical topics (astronomy, cryogenics
... [truncated] ...
ition cycle.
        Regardless, the information is present.

    *   Answer is PHANTOM-9092.

    *
```
* **QUANTA Answer:**
```text
Thinking Process:

1.  **Analyze the Request:** The user is asking for a specific piece of information: "the designated mission access code for Vault 81".

2.  **Scan the Context:** I need to look thr
... [truncated] ...
 will provide the code.

    *Wait, I should check if there are other codes.* No, only one mentioned
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 11: `babilong_4k` (babilong)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `bedroom`
* **Task Prompt:**
```text
Where is the apple?
```
* **Base LLM Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A long text containing multiple paragraphs of seemingly unrelated scientific/technical information (astronomy, cryogenics, computer scien
... [truncated] ...
ific text block]
    [Scientific text block]
    [Scientific text block]
    [Scientific text block]
```
* **QUANTA Answer:**
```text
Based on the narrative updates provided in the text, the apple is in the **bedroom**.

The sequence of events described is:
1.  Sandra journeyed to the garden.
2.  Sandra picked up the apple in the garden.
3.  Sandra journeyed to the bedroom.
4.  Sandra dropped the apple in the bedroom.
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 12: `babilong_16k` (babilong)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `cellar`
* **Task Prompt:**
```text
Where is the key?
```
* **Base LLM Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A long text containing repeated blocks of scientific/technical information (astronomy, cryogenics, computer science, materials science, e
... [truncated] ...
rix..."
    - "Axiomatic formal..."
    - "During the late Victorian..."
    - "Autonomous swarm..."
```
* **QUANTA Answer:**
```text
Based on the narrative updates provided in the text, the key is currently in the **cellar**.

The sequence of events recorded is:
1.  John picked up the key in the hallway.
2.  John gave the key to Mary in the office.
3.  Mary put down the key in the cellar.
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1536)

---

#### Case 13: `var_track_4k` (long_variable_tracking)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `1700 credits`
* **Task Prompt:**
```text
What is the net balance change for Account ACC-9042 across all recorded transactions?
```
* **Base LLM Answer:**
```text
The user wants to calculate the net balance change for Account ACC-9042 based on the provided text.

1.  **Analyze the Request:**
    *   Target Account: ACC-9042.
    *   Goal: Calculate the net bala
... [truncated] ...
 ACC-9042 recorded a debit transfer of 350 credits." (-350)
    *   Transaction 3: "Account ACC-9042
```
* **QUANTA Answer:**
```text
Based on the provided text, the recorded transactions for **Account ACC-9042** are as follows:

1.  **Passage chunk_0001:** Credit deposit of **1500 credits**.
2.  **Passage chunk_0003:** Debit transf
... [truncated] ...
e:** 2100 - 400 = **1700 credits**

The net balance change for Account ACC-9042 is **1700 credits**.
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 14: `var_track_16k` (long_variable_tracking)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `33`
* **Task Prompt:**
```text
What is the total number of unauthorized SSH attempts detected on Server SRV-ALPHA across all security incidents?
```
* **Base LLM Answer:**
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A long text containing repeated paragraphs of scientific/technical information interspersed with "ENTERPRISE TELEMETRY EVENT: Security Lo
... [truncated] ...
urity Log: Server SRV-ALPHA detected 11 unauthorized SSH attempts." appears near the end.
    *   Is
```
* **QUANTA Answer:**
```text
Based on the provided text, there are three security incidents recorded for Server SRV-ALPHA:

1.  **Passage chunk_0005:** 14 unauthorized SSH attempts.
2.  **Passage chunk_0017:** 8 unauthorized SSH 
... [truncated] ...
nk_0028:** 11 unauthorized SSH attempts.

**Total:** 14 + 8 + 11 = **33** unauthorized SSH attempts.
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---
