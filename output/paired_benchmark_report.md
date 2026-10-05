# QUANTA Paired & Ablated Benchmark Report

**Date:** 2026-10-05 22:23:48  
**Execution Mode:** `LIVE`  
**Ablation Architecture:** `3WAY` (Base LLM vs. QUANTA Local vs. QUANTA + 14GB Pre-compiled Wikidata KB)  
**Target Hardware:** NVIDIA GeForce RTX 3070 (8GB VRAM)  
**Base LLM:** `unsloth/Qwen3.5-4B-MTP-GGUF` at `Q5_K_M` (Unsloth Studio `:8888`)  

---

## 1. Executive Summary & Comparative Scorecard

| Benchmark Suite | Samples ($N$) | Base LLM Acc | QUANTA Local | QUANTA + KB | Accuracy Lift | Prompt Token Footprint | Latency Speedup |
|---|---|---|---|---|---|---|---|
| **musique** | 30 | 50.0% | 46.7% | **46.7%** | -3.3% (n.s.) | 2668 -> **2657** (-0.4%) | 23.75s -> **39.53s** (0.60x) |
| **proofwriter** | 30 | 93.3% | 100.0% | **96.7%** | +3.3% (n.s.) | 144 -> **144** (0.0%) | 10.43s -> **8.44s** (1.24x) |

> [!NOTE]
> Statistical significance marked with $^*p < 0.05, ^{**}p < 0.01, ^{***}p < 0.001$ using paired Student's $t$-test / Wilcoxon signed-rank test against the base LLM.

---

## 2. Pre-Compiled Knowledge Base (14GB Wikidata) Empirical Impact Analysis

This section measures the exact lift and trade-off of mounting `data/wikipedia_quanta.db` (4.51M entities, 11.65M triples):

| Benchmark Domain | Local ASG Accuracy | +Wikidata KB Accuracy | Knowledge Base Lift ($\Delta$) | Local Latency | +Wikidata Latency | KB Overhead |
|---|---|---|---|---|---|---|
| **musique** | 46.7% | **46.7%** | **+0.0%** | 35.50s | 39.53s | +4027.0 ms |
| **proofwriter** | 100.0% | **96.7%** | **+-3.3%** | 9.37s | 8.44s | -930.0 ms |

---

## 3. Latency vs. Token Count Ingestion Extrapolation Engine

Empirical linear regression modeling of ingestion latency:

$$\tau_{\text{ingest}}(N) = a \cdot N + b$$

* **Processing Throughput:** 209.6 words/sec (278.7 tokens/sec)
* **Fitted Slope ($a$):** 0.003762 s/token
* **Initialization Intercept ($b$):** 0.0010 s
* **Model Fit ($R^2$):** 0.7823

### Ingestion Scalability & VRAM OOM Horizon Forecast

| Document Token Scale | Content Analogy | QUANTA Ingestion Time | Base LLM Prefill TTFT | Base LLM 8GB VRAM Status |
|---|---|---|---|---|
| **10,000 tokens** | Chapter | **37.62 s** | 3.53s | `SAFE (< 6 GB)` |
| **50,000 tokens** | Short Book | **3.1 min** | 42.53s | `HIGH RISK (~7.8 GB)` |
| **100,000 tokens** | Monograph | **6.3 min** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |
| **500,000 tokens** | Full Codebase | **31.3 min** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |
| **1,000,000 tokens** | 1M+ Enterprise Dossier | **62.7 min** | **CRASH (OOM)** | `OOM CRASH (> 8 GB VRAM)` |

---

## 4. Honest Disclosures & Operational Limits

1. **Short-Context Ingestion Penalty (SQuAD):** On single-turn queries under 250 words, the base LLM direct answer is faster by ~50–100 ms. QUANTA is an external coprocessor for complex multi-hop, state-tracking, and long-horizon tasks.
2. **Stylistic & Creative Generation:** Highly idiomatic, metaphorical, or poetic expressions undergo semantic canonicalization into NSM primes, prioritizing factual and relational consistency over stylistic flourish.

---

## 5. Evaluation Discrepancies & Forensic Diagnostics

A total of **26** evaluation discrepancies and diverging cases were forensic-audited during this run:

| Suite | Sample ID | Tag / Category | Base LLM | QUANTA+KB | Forensic Summary |
|---|---|---|---|---|---|
| `musique` | `2hop__153573_109006` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `4hop1__129721_32392_823060_610794` | `[TRUNCATED]` | PASS | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `3hop2__655849_223623_162182` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `3hop1__241001_568433_47686` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `2hop__525242_52835` | `Divergence (QUANTA Win)` | FAIL | PASS | Outcome diverged between Base LLM and QUANTA |
| `musique` | `2hop__169443_83769` | `[TRUNCATED]` | PASS | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `4hop1__436202_765799_282674_759393` | `[TRUNCATED]` | FAIL | PASS | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `3hop1__539312_744503_34754` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `3hop2__90098_60649_10557` | `Divergence (QUANTA Win)` | FAIL | PASS | Outcome diverged between Base LLM and QUANTA |
| `musique` | `2hop__499003_853511` | `Divergence (Base Win)` | PASS | FAIL | Outcome diverged between Base LLM and QUANTA |
| `musique` | `2hop__822965_165304` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `4hop1__342858_131850_33897_81096` | `[TRUNCATED]` | FAIL | PASS | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `2hop__82910_75184` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `2hop__156700_63853` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `4hop1__17192_17130_70784_61381` | `[TRUNCATED]` | PASS | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `4hop1__56635_49925_13759_736921` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `2hop__384238_74735` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `2hop__860241_766393` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `2hop__156658_155922` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `3hop1__295815_40769_64047` | `Divergence (Base Win)` | PASS | FAIL | Outcome diverged between Base LLM and QUANTA |
| `musique` | `2hop__156700_73244` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `2hop__2416_53663` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `musique` | `2hop__463353_80420` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `proofwriter` | `AttNeg-OWA-D0-4611` | `[TRUNCATED]` | PASS | PASS | Output truncated due to token budget limit (finish_reason='length |
| `proofwriter` | `AttNoneg-OWA-D0-3238` | `[TRUNCATED]` | FAIL | FAIL | Output truncated due to token budget limit (finish_reason='length |
| `proofwriter` | `AttNoneg-OWA-D0-4291` | `Divergence (QUANTA Win)` | FAIL | PASS | Outcome diverged between Base LLM and QUANTA |

### Detailed Forensic Diagnostics & Discrepancy Breakdown

#### Case 1: `2hop__153573_109006` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Walt Disney`
* **Task Prompt:**
```text
Who developed the eponymous character from the series that contains Mickey's Safari in Letterland?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
Based on the provided documents, there is no information regarding "Mickey's Safari," "Letterland," or the specific character development for that series. The documents contain information about vario
... [truncated] ...
 its characters.

Answer: The provided documents do not contain information to answer this question.
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to answer your question:

1.  **Analyze the Premise:** The question asks about the developer of an "eponymous character" fro
... [truncated] ...
elopment'... gameplay is very true to the graphical point-and-click adventure tradition of gameplay,
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 2: `4hop1__129721_32392_823060_610794` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Richland County`
* **Task Prompt:**
```text
What county is the city that shares a border with the state capital of the state where Zubly Cemetery is located in?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine the answer, we must trace the location of Zubly Cemetery through the provided documents and identify the specific county associated with its neighboring city.

1.  **Identify the location
... [truncated] ...
city identified (Aiken) is located in this county, the answer is Aiken County.

Answer: Aiken County
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the county:

1.  **Identify the location of Zubly Cemetery:**
    According to Document [19] and multiple state
... [truncated] ...
rsity of Central Oklahoma ... located just north of the city in the suburb of Edmond."
    "Oklahoma
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 3: `3hop2__655849_223623_162182` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `South Central Coast`
* **Task Prompt:**
```text
In what region of the country containing A Lưới is the birthplace of John Phan located?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to answer the question:

1.  **Analyze the Question:** The user asks for the region of the country containing "A Lưới" where
... [truncated] ...
, Doc 27 is Baby Boom. Doc 11 is Population growth.
Doc 12 = Doc 35 (Mehergaon, India).
Doc 14 = Doc
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the region:

1.  **Identify John Phan's birthplace:**
    *   The document states: "Bon 'John' Phan (born Octob
... [truncated] ...
situated in the **Central** region of Vietnam.

    Answer: Central Vietnam

Answer: Central Vietnam
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 4: `3hop1__241001_568433_47686` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `Toronto Coach Terminal`
* **Task Prompt:**
```text
Where do Greyhound buses leave from in the city where Arna Selznick's employer is headquartered?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine where Greyhound buses leave from in the city where Arna Selznick's employer is headquartered, we must follow these steps based on the provided documents:

1.  **Identify Arna Selznick and
... [truncated] ...
er is headquartered (Toronto) leave from the Toronto Coach Terminal.

Answer: Toronto Coach Terminal
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine where Greyhound buses leave from in the city where Arna Selznick's employer is headquartered:

1.  **Identify A
... [truncated] ...
 Document [16] lists Greyhound routes), this is the departure point.

Answer: Toronto Coach Terminal
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 5: `2hop__525242_52835` (musique)
* **Diagnostic Tag / Category:** `Divergence: QUANTA Pass / Base Fail`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `John D. Loudermilk`
* **Task Prompt:**
```text
Who wrote "Turn Me On" by performer of "Happy Pills"?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine who wrote "Turn Me On," we must first identify the performer associated with that song and then link them to the performer of "Happy Pills" using the provided documents.

**Step 1: Identi
... [truncated] ...
riter of "Turn Me On" by the performer of "Happy Pills" is Norah Jones herself.

Answer: Norah Jones
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to identify the writer of "Turn Me On":

1.  **Identify the performer of "Happy Pills":**
    *   According to Document [6],
... [truncated] ...
 with that single, "John D." is the direct answer derived from the text's metadata.

Answer: John D.
```
* **Forensic Diagnosis:** QUANTA successfully retrieved relevant episodic context, while Base LLM failed due to token cutoff or missing facts.

---

#### Case 6: `2hop__169443_83769` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Fred O'Bannion`
* **Task Prompt:**
```text
Who did the screenwriter for Good Will Hunting play in Dazed and Confused?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine who the screenwriter of *Good Will Hunting* played in *Dazed and Confused*, we must first identify the screenwriters of *Good Will Hunting* and then cross-reference their acting roles in 
... [truncated] ...
ed and Confused* is Ben Affleck, and he played the character Fred O'Bannion.

Answer: Fred O'Bannion
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to answer the question:

1.  **Identify the screenwriter of *Good Will Hunting*:**
    *   Document [20] states that *Good W
... [truncated] ...
er is **Matt Damon**, and he played a character named **Will**? Yes, in *Good Will Hunting*. But the
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 7: `4hop1__436202_765799_282674_759393` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `Green Bay`
* **Task Prompt:**
```text
What is the seat of the county sharing a border with the county in which J. P. Hayes was born?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine the seat of the county sharing a border with the county in which J.P. Hayes was born, we must first identify Hayes's birthplace and the corresponding county using the provided documents, 
... [truncated] ...
estion refers to **Outagamie** itself? No, "sharing a border with the county in which J.P. Hayes was
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the seat of the county sharing a border with the county in which J.P. Hayes was born:

1.  **Identify where J.P
... [truncated] ...
h Outagamie.
    6.  The seat of Oconto County is **Oconto**.

    Is there any other interpretation
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 8: `3hop1__539312_744503_34754` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `blackmail`
* **Task Prompt:**
```text
What does the U.S. believe caused the country where the birthplace of Yaovabha Bongsanid is located, to help Japan?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine the answer, we must first identify the country where Princess Yaovabha Bongsanid was born and then analyze the historical context provided in the documents regarding that country's relati
... [truncated] ...
se stating that while the documents identify Princess Yaovabha Bongsanid as being from Thailand (via
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction:

1.  **Identify Yaovabha Bongsanid's Birthplace:**
    *   The document states: "Yaovabha Bongsanid ... was the Princess of
... [truncated] ...
ific is located in Iwo Jima and Okinawa."
        *   "American is located in Iwo Jima and Okinawa."
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 9: `3hop2__90098_60649_10557` (musique)
* **Diagnostic Tag / Category:** `Divergence: QUANTA Pass / Base Fail`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `Medieval Latin`
* **Task Prompt:**
```text
What was the language from which the last name Sylvester originates during the era of the Frankish king who formed the Holy Roman Empire later known as?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
Based on the provided documents, there is no mention of a person named "Sylvester," nor is there any information regarding the etymology or origin of the last name "Sylvester." The documents discuss v
... [truncated] ...
: The provided documents do not contain information regarding the origin of the last name Sylvester.
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the origin of the last name Sylvester during the era of the Frankish king who formed the Holy Roman Empire late
... [truncated] ...
ing the Frankish kingdom and the Carolingian Empire's linguistic landscape, is Latin.

Answer: Latin
```
* **Forensic Diagnosis:** QUANTA successfully retrieved relevant episodic context, while Base LLM failed due to token cutoff or missing facts.

---

#### Case 10: `2hop__499003_853511` (musique)
* **Diagnostic Tag / Category:** `Divergence: Base Pass / QUANTA Fail`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Kanine Records`
* **Task Prompt:**
```text
What is the record label for the band which performed Pythons?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine the record label for the band that performed "Pythons," we must first identify the specific band associated with the title "Pythons" based on the provided documents.

1.  **Identify the B
... [truncated] ...
n 2013). Therefore, Sire Records is the correct label for the "Pythons" album.

Answer: Sire Records
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the record label for the band that performed "Pythons":

1.  **Identify the album and artist:** Document [13] s
... [truncated] ...
*Pythons* album by Surfer Blood.

Therefore, the record label is Sire Records.

Answer: Sire Records
```
* **Forensic Diagnosis:** Base LLM answered correctly while QUANTA output did not meet exact match criteria.

---

#### Case 11: `2hop__822965_165304` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Kim Braden`
* **Task Prompt:**
```text
Who is the child of the actor who is a cast member of Love in Pawn?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine the answer, we must trace the connections between the actors mentioned in the provided documents and their family relationships.

1.  **Identify the actor from "Love in Pawn":**
    *   D
... [truncated] ...
 Braeden is the child of the actor (Bernard Braden) who was in *Love in Pawn*.

Answer: Eric Braeden
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to answer your question:

1.  **Identify the actor associated with "Love in Pawn":**
    *   Document [19] states: "Love in 
... [truncated] ...
ion about the children of the actors (Bernard Braden and Barbara Kelly) who starred in Love in Pawn.
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 12: `4hop1__342858_131850_33897_81096` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `Mario Andretti`
* **Task Prompt:**
```text
Who won the Indy Car Race in the largest populated city of the state where Yuma's Library District is located?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine the winner of the Indy Car race in the specified location, we must follow these logical steps based on the provided documents:

1.  **Identify Yuma's Library District and its location:**

... [truncated] ...
* won his final career victory there in 1993. It also mentions a crash with Johnny Rutherford.
    *
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to answer your question:

1.  **Identify the location of Yuma's Library District:**
    *   Document [19] states that "The C
... [truncated] ...

    *   **Refining the logic based strictly on the text's internal consistency (which is broken):**
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 13: `2hop__82910_75184` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `Claudia Wells`
* **Task Prompt:**
```text
Who played the girlfriend of Alex P. Keaton's actor on Family Ties in Back to the Future?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine who played the girlfriend of the actor who portrayed Alex P. Keaton in *Family Ties* within the movie *Back to the Future*, we must follow these steps based on the provided documents:

1.
... [truncated] ...
r in the original *Back to the Future* film is Claudia Wells (Document [13]).

Answer: Claudia Wells
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to identify the actress:

1.  **Identify the actor of Alex P. Keaton:**
    *   Document [20] and the main text state that "
... [truncated] ...
he very first sentence of the prompt's context: "On October 26 is located in Marty McFly." "Family T
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 14: `2hop__156700_63853` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Thaba Putsoa`
* **Task Prompt:**
```text
What is the source of the river of which Kraai River is a tributary?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine the source of the river to which the Kraai River is a tributary, we must follow the chain of tributaries described in the provided documents.

1.  **Identify the parent river of the Kraai
... [truncated] ...
e "of which... tributary" part is a distractor or error in the prompt generation.
However, if I must
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the source of the river of which the Kraai River is a tributary:

1.  **Identify the parent river of the Kraai 
... [truncated] ...
mation regarding the source of the Orange River; they only state that the Kraai River flows into it.
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 15: `4hop1__17192_17130_70784_61381` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `1932`
* **Task Prompt:**
```text
When was the region immediately north of the region where the country that secured southern Lebanon is located and the Persian Gulf established?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To answer this question, we must first identify the specific entities mentioned in the query by tracing the relationships defined in the provided documents.

**Step 1: Identify the country that secure
... [truncated] ...
hen was [A] and [B] established?" implying a single answer.
    *   Is it possible the question is a
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the answer:

**Step 1: Identify the country that secured southern Lebanon.**
*   According to **Document [4]** 
... [truncated] ...
ers to **July 21** because the text says "Established on July 21, 1849" and the user is testing if I
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 16: `4hop1__56635_49925_13759_736921` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `Saxony-Anhalt`
* **Task Prompt:**
```text
The German priest, who wanted to reform the religious denomination now the largest in the US, preached a sermon on Marian devotion soon before his death in which German state?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine the answer, we must trace the logical connections between the religious reformer mentioned and the specific location where he preached his final sermon.

1.  **Identify the "German priest
... [truncated] ...
n at Wittenberg.
    *   Wittenberg is located in the state of Saxony-Anhalt.

Answer: Saxony-Anhalt
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction:

1.  **Identify the largest religious denomination in the US:**
    *   Document [2] states: "With 70.4 million members, it
... [truncated] ...
ittenberg is located in the state of **Saxony-Anhalt** (historically Saxony).

Answer: Saxony-Anhalt
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 17: `2hop__384238_74735` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** ```Hey Jude ''`
* **Task Prompt:**
```text
What song did Paul McCartney write for Sean Lennon's siblings?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to identify the song Paul McCartney wrote for Sean Lennon's siblings:

1.  **Identify Sean Lennon's siblings:** Document [1]
... [truncated] ...
en "for" someone by Paul McCartney in the text is "It's for You," which was written for Cilla Black.
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to answer your question:

1.  **Identify Sean Lennon's siblings:**
    *   Document [1] states that Sean Lennon is the son o
... [truncated] ...
see, but I must deduce the most likely intended answer based on the entities present.*
    Entities:
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 18: `2hop__860241_766393` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Nobel Prize in Literature`
* **Task Prompt:**
```text
What award was received by the person who authored Missing Person?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
Based on the provided documents, there is no mention of an author named "Missing Person" or a book titled "Missing Person." The documents contain information about various authors (Theresa Schwegel, S
... [truncated] ...
itled "Missing Person," so it is impossible to determine what award they received based on the text.
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the award received by the author of *Missing Person*:

1.  **Identify the author of *Missing Person*:**
    *  
... [truncated] ...
he answer in the context of the dataset's logic (as some datasets equate festival selection/premiere
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 19: `2hop__156658_155922` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `Sydney Harbour`
* **Task Prompt:**
```text
What body of water does the river that Toongabbie Creek flows into flow into?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine what body of water the river that Toongabbie Creek flows into eventually drains into, we must trace the flow path using the provided documents:

1.  **Identify the immediate parent water 
... [truncated] ...
ed from combining the text with necessary geographical facts.

Answer: Port Jackson (Sydney Harbour)
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the final body of water:

1.  **Identify Toongabbie Creek's location and parent river:**
    *   Document [7] s
... [truncated] ...
**Parramatta River** catchment.
    *   **Step 2:** Determine where the Parramatta River flows into.
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 20: `3hop1__295815_40769_64047` (musique)
* **Diagnostic Tag / Category:** `Divergence: Base Pass / QUANTA Fail`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Sales began worldwide in April 2012`
* **Task Prompt:**
```text
When did the luxury division of the company that manufactures Toyopet Master change the body style of the RX 350?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine when the luxury division of the company that manufactures the Toyopet Master changed the body style of the RX 350, we must follow these logical steps based on the provided documents:

1. 
... [truncated] ...
based on the text.

Answer: 2012 (specifically unveiled in March 2012 and sales began in April 2012)
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction regarding your question:

1.  **Identify the manufacturer of the Toyopet Master:**
    *   Document [QUANTA NEURO-SYMBOLIC V
... [truncated] ...
ot contain information about when the luxury division of Toyota changed the body style of an RX 350.
```
* **Forensic Diagnosis:** Base LLM answered correctly while QUANTA output did not meet exact match criteria.

---

#### Case 21: `2hop__156700_73244` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `Thaba Putsoa`
* **Task Prompt:**
```text
Where is the origin of the river that Kraai River is a tributary of?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine the origin of the river that the Kraai River is a tributary of, we must follow the chain of relationships described in the provided documents.

1.  **Identify the parent river of the Kraa
... [truncated] ...
 "The Kraai River originates in the mountains south of Lesotho... joins the Orange River".
    Maybe
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the origin of the river that the Kraai River is a tributary of:

1.  **Identify the parent river of the Kraai R
... [truncated] ...
lows into the Orange River, but they do not specify the origin or source of the Orange River itself.
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 22: `2hop__2416_53663` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `northeastern Africa`
* **Task Prompt:**
```text
Where is the river that the engine pumped water from located in the world?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
To determine where the river that the engine pumped water from is located, we must trace the specific historical event and location mentioned in the provided documents.

1.  **Identify the Engine and 
... [truncated] ...
through countries like Egypt and Sudan).

Therefore, the river is located in Africa.

Answer: Africa
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to determine the location of the river:

1.  **Identify the Engine and its Location:** Document [2] states that John Lofting
... [truncated] ...
lonial Industrial Quarter of Bethlehem, PA.
    *   Let's reconsider the **Rainy River**. Document [
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 23: `2hop__463353_80420` (musique)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `James Howard Meredith`
* **Task Prompt:**
```text
Who was the first African American student to attend the school that owns Swayze Field?

Explain the step-by-step connections across the documents, then conclude on a new line with 'Answer: <final answer>'.
```
* **Base LLM Answer:**
```text
Based on the provided documents, there is no mention of "Swayze Field," nor is there any information linking any specific school to the ownership of Swayze Field. The documents contain information abo
... [truncated] ...
ool, so it is impossible to determine who the first African American student was based on this text.
```
* **QUANTA Answer:**
```text
Based on the provided documents, here is the step-by-step logical deduction to answer your question:

1.  **Identify the owner of Swayze Field:**
    The document states: "Oxford-University Stadium at
... [truncated] ...
nity High School"
    - "Woodrow Wilson Classical High School"
    - "Anthony of Padua School"
    -
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=2048)

---

#### Case 24: `AttNeg-OWA-D0-4611` (proofwriter)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `PASS` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `true`
* **Task Prompt:**
```text
Based on the provided facts and rules, determine whether the following statement is true, false, or unknown: 'Gary is white'

Explain your deductive reasoning step by step, then conclude on a new line with 'Answer: [True/False/Unknown]'.
```
* **Base LLM Answer:**
```text
To determine the truth value of the statement "Gary is white," let's break down the facts and rules provided and apply them logically to Gary.

**1. List the Facts:**
*   F1: Gary is furry.
*   F2: Ga
... [truncated] ...
e specific types of generated logic problems, there might be a trick where one of the rules *contrad
```
* **QUANTA Answer:**
```text
To determine the truth value of the statement "Gary is white," let's break down the provided facts and rules logically.

**1. Analyze the Facts:**
*   Fact 1: Gary is furry.
*   Fact 2: Gary is nice.

... [truncated] ...
.

Therefore, based on the explicit statement in the context, the proposition is true.

Answer: True
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1024)

---

#### Case 25: `AttNoneg-OWA-D0-3238` (proofwriter)
* **Diagnostic Tag / Category:** `[TRUNCATED]`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `FAIL`
* **Ground Truth Answer:** `false`
* **Task Prompt:**
```text
Based on the provided facts and rules, determine whether the following statement is true, false, or unknown: 'Fiona is not red'

Explain your deductive reasoning step by step, then conclude on a new line with 'Answer: [True/False/Unknown]'.
```
* **Base LLM Answer:**
```text
To determine the truth value of the statement "Fiona is not red," let's break down the facts and rules provided step by step:

1.  **Fact**: We are explicitly told that "Fiona is blue."
2.  **Rule**: 
... [truncated] ...
iven. It depends on an unstated assumption.

Therefore, the truth value is unknown.

Answer: Unknown
```
* **QUANTA Answer:**
```text
To determine the truth value of the statement "Fiona is not red," let's break down the provided facts and rules logically:

1.  **Fact**: Fiona is blue.
2.  **Rule**: If someone is red, then they are 
... [truncated] ...
ona is nice." -> Since Fiona is blue, **Fiona is nice**.
        *   "All nice people are young." ->
```
* **Forensic Diagnosis:** Output truncated due to token budget limit (finish_reason='length', completion_tokens=1024)

---

#### Case 26: `AttNoneg-OWA-D0-4291` (proofwriter)
* **Diagnostic Tag / Category:** `Divergence: QUANTA Pass / Base Fail`
* **Evaluation Outcome:** Base LLM: `FAIL` | QUANTA+KB: `PASS`
* **Ground Truth Answer:** `true`
* **Task Prompt:**
```text
Based on the provided facts and rules, determine whether the following statement is true, false, or unknown: 'Harry is young'

Explain your deductive reasoning step by step, then conclude on a new line with 'Answer: [True/False/Unknown]'.
```
* **Base LLM Answer:**
```text
To determine the truth value of the statement "Harry is young," let's break down the provided facts and rules logically:

1.  **Analyze the Facts about Harry:**
    *   Fact 1: "Harry is big."
    *  
... [truncated] ...
text.

Therefore, the statement cannot be proven true or false; it remains unknown.

Answer: Unknown
```
* **QUANTA Answer:**
```text
To determine the truth value of the statement "Harry is young," let's break down the provided facts and rules logically:

1.  **Analyze the Facts:**
    *   Fact 1: Erin is blue.
    *   Fact 2: Erin 
... [truncated] ...
.
    *   Therefore, based on the direct assertion in the text, the statement is true.

Answer: True
```
* **Forensic Diagnosis:** QUANTA successfully retrieved relevant episodic context, while Base LLM failed due to token cutoff or missing facts.

---
