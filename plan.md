# QUANTA Benchmark Hardening & Architectural Improvement Plan

This plan outlines root causes, forensic diagnostics, and implementation strategies for resolving discrepancies and improving performance across the QUANTA publication-grade benchmark suite.

---

## Session Starter Commands

Copy-paste any of the following commands into a new Antigravity session to begin implementation on a specific section:

* `Start working on Section 1 in @plan.md: HumanEval Sandbox & Harness Scope Preservation`
* `Start working on Section 2 in @plan.md: Science Multiple-Choice Evaluation Hardening (ARC-Challenge)`
* `Start working on Section 3 in @plan.md: Numeric-Aware Evaluation & Unit Lenience (Variable Tracking)`
* `Start working on Section 4 in @plan.md: Multi-Hop Bridge Expansion & Graph Spreading Activation (MuSiQue)`
* `Start working on Section 5 in @plan.md: Submission Artifact Sanitization & Publication Exporter Polishing`
* `Start working on Section 6 in @plan.md: Benchmark Runner Resumption, Checkpointing & Telemetry`

---

## Section 1: HumanEval Sandbox & Harness Scope Preservation

### Context & Problem Statement
In the paired benchmark run on OpenAI HumanEval ($N=15$), Base LLM achieved 100.0% (15/15) while QUANTA achieved 93.3% (14/15). The single failure occurred on `HumanEval/10` (`make_palindrome`).

### Forensic Diagnosis
The task prompt for `HumanEval/10` defines a helper function prior to the target function:
```python
def is_palindrome(string: str) -> bool:
    """ Test if given string is a palindrome """
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string. ... """
```
In [`src/benchmarks/code_evaluator.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/code_evaluator.py#L97):
```python
if entry_point and f"def {entry_point}" in extracted_code:
    full_code = preamble + extracted_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
else:
    full_code = preamble + prompt.rstrip() + "\n" + extracted_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
```
When QUANTA generates a self-contained definition of `def make_palindrome(...)`, `f"def {entry_point}" in extracted_code` evaluates to `True`. As a result, the harness completely drops the original `prompt`, stripping `def is_palindrome(...)` from the test script and causing a runtime error:
```
NameError: name 'is_palindrome' is not defined
```
The Base LLM only passed because it redundantly appended `def is_palindrome(string: str) -> bool: ...` at the bottom of its markdown completion block.

### Key Files Involved
* [`src/benchmarks/code_evaluator.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/code_evaluator.py)
* [`tests/test_code_evaluator.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/tests/test_code_evaluator.py) (if present, or create dedicated test)

### Proposed Architectural Improvements
1. **Prompt Preamble Slicing:**
   In [`CodeEvaluator.evaluate_solution`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/code_evaluator.py):
   If `entry_point` is specified and present in `prompt`, extract all declarations in `prompt` that precede `def {entry_point}` (including helper functions, constants, type aliases, and custom imports).
   ```python
   idx = prompt.find(f"def {entry_point}")
   prompt_helpers = prompt[:idx].rstrip() if idx != -1 else ""
   ```
   Construct `full_code` using:
   `full_code = preamble + prompt_helpers + "\n\n" + extracted_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"`
2. **Import & Function Deduping:**
   Ensure helper functions from `prompt_helpers` do not conflict if the model completion also defines them.
3. **Verification Target:**
   Running `HumanEval/10` with both completions must yield `passed=True` without relying on accidental model redundancy.

---

## Section 2: Science Multiple-Choice Evaluation Hardening (ARC-Challenge)

### Context & Problem Statement
On the AI2 ARC-Challenge science benchmark ($N=15$), both Base LLM and QUANTA scored 46.7% (7/15 correct). The identical low score was driven by evaluation regex flaws and token cutoff during chain-of-thought explanations.

### Forensic Diagnosis
1. **Case-Insensitive Letter Extraction Collision:**
   In [`src/benchmarks/metrics.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/metrics.py#L74):
   ```python
   m = re.search(r"\b([A-D])\b", clean, re.IGNORECASE)
   if m:
       return m.group(1).upper()
   ```
   Because `re.IGNORECASE` was used, `\b([A-D])\b` matched the lowercase English article `"a"` (*"a planet"*, *"a meteorite"*, *"a distance"*), or algebraic variable names (*"d = 1.0 m"*), falsely extracting `"A"` or `"D"` from the very first sentence of the model's explanation.
2. **Truncated Chain-of-Thought:**
   The generation token budget was constrained to `max_tokens = 150` in [`src/benchmarks/paired_evaluator.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/paired_evaluator.py#L275). When the model generated detailed multi-step reasoning before stating the answer letter, it was truncated before reaching the closing choice sentence.

### Key Files Involved
* [`src/benchmarks/metrics.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/metrics.py)
* [`src/benchmarks/paired_evaluator.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/paired_evaluator.py)
* [`src/benchmarks/suite_loaders.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/suite_loaders.py)

### Proposed Architectural Improvements
1. **Closing-Answer Prioritization:**
   Prioritize explicit answer phrases at the end of the text:
   * Regex 1 (Explicit answer anchor): `r"(?:correct\s+(?:choice|answer)\s+is|therefore,?\s*(?:the\s+answer\s+is)?|answer\s*[:\-])\s*\(?([A-D])\)?"` (case-insensitive keyword, capturing A-D).
   * Regex 2 (Parenthesized choice anywhere in concluding lines): `r"\(([A-D])\)"`.
   * Regex 3 (Isolated uppercase choice on its own line): `r"^\s*([A-D])\s*$"`.
   * Avoid matching isolated lowercase `a` or `d` unless specifically designated as the choice key.
2. **Adaptive Generation Token Budget:**
   In [`PairedEvaluator.evaluate_sample`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/paired_evaluator.py): Increase `max_tokens` for `arc_science` from 150 to 300 to allow chain-of-thought to finish.
3. **Structured Prompt Formatting:**
   In [`BenchmarkSuiteLoader.load_arc_science`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/suite_loaders.py):
   Instruct the model: `"Conclude your reasoning with 'Answer: [A/B/C/D]'."`

---

## Section 3: Numeric-Aware Evaluation & Unit Lenience (Variable Tracking)

### Context & Problem Statement
On `long_variable_tracking`, QUANTA achieved 80.0% (4/5) on paper, but in reality its reasoning on the failed 256k sample (`var_track_256k`) was 100% mathematically and factually accurate.

### Forensic Diagnosis
* **Sample `var_track_256k`:**
  * **Ground Truth:** `"630 microprocessors"`
  * **QUANTA Output:**
    > "1. Event 1: Dispatched 120 microprocessors (-120)... 2. Event 2: Received 250 microprocessors (+250)... 3. Event 3: Received 500 microprocessors (+500)...  
    > **Calculation:** $-120 + 250 + 500 = 630$.  
    > The net change in microprocessor inventory at Warehouse WH-WEST is **630**."
* In [`src/benchmarks/paired_evaluator.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/paired_evaluator.py#L324):
  `corr = BenchmarkMetrics.exact_match_score(ans, gold) or (gold.lower() in ans.lower())`
  Because the number `630` and the noun `microprocessors` were separated by formatting and explanatory sentences, the literal substring `"630 microprocessors"` was not found contiguously.

### Key Files Involved
* [`src/benchmarks/metrics.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/metrics.py)
* [`src/benchmarks/paired_evaluator.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/paired_evaluator.py)

### Proposed Architectural Improvements
1. **Numeric + Quantity Verifier in `BenchmarkMetrics`:**
   Add `numeric_match_score(prediction: str, gold_answer: str) -> bool`:
   * Parse `gold_answer` for numeric quantity $V$ (integer or float) and optional unit/noun $U$ (e.g. `630` and `microprocessors`, `1700` and `credits`, `90` and `minutes`).
   * Check if $V$ appears in bold (`**630**`) or as the final calculated value in `prediction`.
   * If $U$ is present, check that the noun or its singular/plural stem appears in `prediction`.
   * If both conditions are satisfied, evaluate as `correct=True`.
2. **Integration in Evaluation Loop:**
   Update `evaluate_sample()` across all numeric tracking and aggregation tasks to utilize `numeric_match_score()`.

---

## Section 4: Multi-Hop Bridge Expansion & Graph Spreading Activation (MuSiQue)

### Context & Problem Statement
On the MuSiQue multi-hop reasoning benchmark ($N=15$), Base LLM scored 13.3% while QUANTA scored 6.7%. Although QUANTA achieved **71.3% prompt token compression** ($2,604 \to 748$ tokens), it lost critical intermediate documents in 3-hop and 4-hop reasoning chains.

### Forensic Diagnosis
* In 3-hop and 4-hop questions, only the start entity is explicitly stated in the question.
  * *Example:* "What county is the city that shares a border with the state capital of the state where Zubly Cemetery is located?"
  * The question mentions `"Zubly Cemetery"`.
  * Document [19] mentions: `"Zubly Cemetery near Beech Island, South Carolina"`.
  * The intermediate entities (*"South Carolina"*, *"Columbia"*, *"Richland County"*) are latent and unknown at query time.
* When QUANTA extracted initial seeds from the query text alone, it activated nodes around `"Zubly Cemetery"`, but when ranking candidate paragraphs, intermediate paragraphs without direct lexical overlap with the query were dropped under the aggressive 750-token budget ceiling.
* The 14GB pre-compiled Wikidata KB did not provide a lift (+0.0%) because entity triples alone lacked the specific textual paragraphs expected by MuSiQue distractors.

### Key Files Involved
* [`src/retrieval/spreading_activation.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/retrieval/spreading_activation.py)
* [`src/retrieval/page_table.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/retrieval/page_table.py)
* [`src/benchmarks/paired_evaluator.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/paired_evaluator.py)

### Proposed Architectural Improvements
1. **Dynamic Multi-Hop Activation Propagation:**
   In [`SpreadingActivationRetriever`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/retrieval/spreading_activation.py):
   * When query intent indicates multi-hop traversal (`"shares a border with"`, `"state capital of the state where"`, `"born in the city where"`), increase maximum hop depth from 2 to 3 or 4.
   * Introduce associative cross-paragraph entity bridges during episodic ASG ingestion so that documents mentioning common named entities share semantic edges (`CO_OCCURS`, `LOCATED_IN`).
2. **Adaptive Context Budgeting:**
   In [`paired_evaluator.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/paired_evaluator.py) for MuSiQue:
   * Allow up to 1,200 tokens (instead of hard-capping at ~700 tokens) when the source context contains $\ge 20$ multi-hop paragraphs, preserving 2–3 additional bridge paragraphs.
3. **Entity Alias & Disambiguation Resolution:**
   Utilize `WikidataEntityMapper` in `page_table.py` to bridge latent entities into the active canvas.

---

## Section 5: Submission Artifact Sanitization & Publication Exporter Polishing

### Context & Problem Statement
[`PublicationExporter`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/publication_exporter.py) generates submissions and LaTeX tables, but several artifacts require formatting sanitization for official benchmark leaderboards.

### Forensic Diagnosis
1. **HumanEval JSONL Submissions:**
   [`output/submissions/humaneval_predictions.jsonl`](file:///c:/Users/PC/Documents/GitHub/Quanta/output/submissions/humaneval_predictions.jsonl) wraps code completions inside markdown codeblocks (```` ```python ... ``` ````). Official EvalPlus and HumanEval runners expect bare Python code strings.
2. **MuSiQue CodaLab Submissions:**
   [`output/submissions/musique_codalab_submission.json`](file:///c:/Users/PC/Documents/GitHub/Quanta/output/submissions/musique_codalab_submission.json) includes full conversational sentences (*"Based on the documents provided..."*). Official CodaLab scoring scripts evaluate token F1 and exact match against short answer phrases (*"Walt Disney"*, *"Richland County"*).
3. **LaTeX Main Scorecard Table:**
   [`output/publication/tables/main_scorecard.tex`](file:///c:/Users/PC/Documents/GitHub/Quanta/output/publication/tables/main_scorecard.tex) formats negative zero percentages (`-0.0\%`) when token reduction is 0.

### Key Files Involved
* [`src/benchmarks/publication_exporter.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/publication_exporter.py)
* [`output/submissions/`](file:///c:/Users/PC/Documents/GitHub/Quanta/output/submissions/)
* [`output/publication/tables/`](file:///c:/Users/PC/Documents/GitHub/Quanta/output/publication/tables/)

### Proposed Architectural Improvements
1. **Clean Code Extraction for HumanEval:**
   In `PublicationExporter.export_leaderboard_submissions()`, run `CodeEvaluator.extract_python_code(completion, entry_point)` on all HumanEval outputs prior to writing `humaneval_predictions.jsonl`.
2. **Concise Answer Extraction for MuSiQue CodaLab:**
   Extract the target entity or bolded text (`\*\*(.*?)\*\*`), or apply a heuristic concise noun-phrase extractor to sanitize `musique_ans_preds` for CodaLab submission.
3. **LaTeX Formatting Polish:**
   * Format `prompt_reduction_pct` such that `abs(val) < 0.05` is rendered as `0.0\%` (suppressing negative zeroes).
   * Ensure standard `booktabs` rules, bold highlights for winning columns, and proper footnote notation for statistical significance.

---

## Section 6: Benchmark Runner Resumption, Checkpointing & Telemetry

### Context & Problem Statement
Executing a full 9-suite academic run ($N=105$) takes 15–30 minutes on live hardware. An unexpected interruption currently discards intermediate results. Furthermore, real-time error forensics during the run are minimal.

### Key Files Involved
* [`scripts/run_paired_benchmarks.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/scripts/run_paired_benchmarks.py)
* [`src/benchmarks/paired_evaluator.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/benchmarks/paired_evaluator.py)

### Proposed Architectural Improvements
1. **Incremental Checkpointing & `--resume` Support:**
   * Write intermediate results to `output/.paired_benchmark_checkpoint.json` after each sample or suite completion.
   * If `--resume` is supplied, load pre-computed predictions and skip completed samples.
2. **Real-Time Forensic Diagnostics:**
   * Log discrepancies immediately to stdout:
     * Flag `[HARNESS BUG]` when code passes unit logic but throws `NameError` due to missing prompt preamble.
     * Flag `[NUMERIC PASS]` when calculations match but string matching failed.
3. **Automated Diagnostic Section in `paired_benchmark_report.md`:**
   Extend `generate_markdown_report()` to include an automated "Evaluation Discrepancies & Forensics" section listing exact prompts, ground truth, and outputs for diverging cases.

---

## Verification Plan

### Automated Test Matrix
* **Unit Tests:**
  ```powershell
  pytest tests/test_code_evaluator.py tests/test_metrics.py
  ```
* **Targeted Verification Runs (Mock & Live Quick Runs):**
  ```powershell
  # Test HumanEval helper preservation and Science regex in mock mode
  python scripts/run_paired_benchmarks.py --suite humaneval:5,arc_science:5,long_variable_tracking:5 --mode mock --export-submissions --export-latex
  
  # Targeted live test on HumanEval/10 and var_track_256k
  python scripts/run_paired_benchmarks.py --suite humaneval:15,long_variable_tracking:5 --mode live
  ```
* **Full Academic Verification:**
  ```powershell
  python scripts/run_paired_benchmarks.py --suite all --samples 15 --mode live --export-submissions --export-latex
  ```
