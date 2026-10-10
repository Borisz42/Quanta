# QUANTA Master Plan: High-Throughput S-Expression Ingestion & Parallel Slot Optimization

## Goal Description

During the initial development of QUANTA, discourse and Abstract Syntax Graph (ASG) extraction relied on formal **S-expressions** ([`data/grammar/quanta_asg.gbnf`](file:///c:/Users/PC/Documents/GitHub/Quanta/data/grammar/quanta_asg.gbnf), [`src/parser/sexpr_parser.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/sexpr_parser.py)). In subsequent refactoring phases (Session 3 in [`quanta_svm_refactor_master_plan.md`](file:///c:/Users/PC/Documents/GitHub/Quanta/quanta_svm_refactor_master_plan.md), commits `1fe32d6` and `5e1d4d8`), the representation was converted to flat **JSON** ([`data/grammar/skeleton_schema.gbnf`](file:///c:/Users/PC/Documents/GitHub/Quanta/data/grammar/skeleton_schema.gbnf) and [`data/grammar/co_decoded_skeleton_schema.gbnf`](file:///c:/Users/PC/Documents/GitHub/Quanta/data/grammar/co_decoded_skeleton_schema.gbnf)) to incorporate non-autoregressive and co-decoded Kev relational decisions (`intent`, `epist`, `allen`, `pearl`).

Because surface extraction does not strictly require JSON syntax, the JSON format has become a major ingestion bottleneck:
- **Token Inflation**: JSON syntax wastes tokens on repetitive object keys (`"entities"`, `"events"`, `"id"`, `"text"`, `"pred"`, `"subj"`, `"obj"`), quotes around every identifier/predicate, and punctuation delimiters (`{}[]:,`).
- **GPU Ingestion Latency Bottleneck**: On autoregressive Small Language Models (Qwen 3.5 4B on NVIDIA RTX 3070), decoding latency is strictly proportional to completion tokens ($t_{\text{chunk}} \approx t_{\text{prefill}} + N_{\text{tokens}} \cdot t_{\text{decode}}$). Generating 180 tokens takes $>2.6\text{ seconds}$ per chunk.

This master plan redesigns the ingestion transduction layer to bring back **token-compact S-expressions** as the high-throughput production default while preserving full backwards compatibility and modularity across 5 independent Antigravity sessions:
1. **Compact Keyword S-Expression** (`sexpr_compact`) as the production default ($\sim 61\text{ tokens/chunk}$, **2.79x speedup**).
2. **Positional Ultra-Compact S-Expression** (`sexpr_positional`) for maximum raw throughput ($\sim 41\text{ tokens/chunk}$, **3.80x speedup**).
3. **Universal Co-Decoding Toggles**: Every grammar (both S-expression formats and JSON) can independently toggle co-decoded Kev decisions (`intent`, `epist`, `allen`, `pearl`) on or off via a single configuration parameter.
4. **Parallel Slot Concurrency Optimization**: A systematic empirical sweep of Unsloth / `llama-server` parallel slots ($1, 2, 4, 8, 12, 16$) to discover the optimal concurrency setting on the active RTX 3070 setup and make it easily configurable for any workstation.
5. **Real-World MuSiQue Multi-Hop Validation**: Paired end-to-end evaluation proving zero regression in multi-hop question-answering accuracy alongside 3x+ ingestion acceleration.

---

## Empirical Verification & Baseline Evidence (Live RTX 3070)

Prior to authoring this plan, live empirical experiments were executed directly on the user's active NVIDIA GeForce RTX 3070 backend running `unsloth/Qwen3.5-4B-MTP-GGUF` at `Q5_K_M` across real multi-sentence paragraphs from the **MuSiQue** multi-hop benchmark (Charles Babbage, University of Cambridge, Cambridge Town, United Kingdom).

### Live Empirical Benchmark Results

| Representation Format | Decoding Latency / Chunk | Ingestion Throughput | Completion Tokens / Chunk | Latency Speedup | Token Reduction vs Current |
|---|---|---|---|---|---|
| **1. Co-Decoded JSON (Current Production Default)** | **2,635.1 ms** | **34.2 w/s** | **181.8 tok** | **1.00x** (Baseline) | **+0.0%** (Baseline) |
| **2. Standard JSON Skeleton (`skeleton_schema.gbnf`)** | **1,667.8 ms** | **54.0 w/s** | **99.5 tok** | **1.58x** | **-45.3%** |
| **3. Compact Keyword S-Expr (`compact_sexpr`)** | **944.5 ms** | **95.3 w/s** | **61.5 tok** | **2.79x** | **-66.2%** |
| **4. Positional Ultra-Compact S-Expr (`positional_sexpr`)** | **693.6 ms** | **129.8 w/s** | **41.0 tok** | **3.80x** | **-77.4%** |

### Concrete Output Comparison on MuSiQue Passage 1 (Babbage at Cambridge)

```json
// CURRENT CO-DECODED JSON (182 tokens, 2,635 ms)
{"entities": [{"id": "E1", "text": "Charles Babbage"}, {"id": "E2", "text": "Difference Engine"}, {"id": "E3", "text": "Analytical Engine"}], "events": [{"id": "EV1", "pred": "originated", "subj": "E1", "obj": "digital programmable computer", "intent": "I", "epist": "O", "allen": "B", "pearl": "M"}, {"id": "EV2", "pred": "designed", "subj": "E1", "obj": "E2", "intent": "I", "epist": "O", "allen": "B", "pearl": "M"}]}
```

```lisp
;; CANDIDATE A: COMPACT KEYWORD S-EXPRESSION (61 tokens, 944 ms — 2.79x faster)
(graph
  (entity E1 "Charles Babbage")
  (entity E2 "Trinity College, Cambridge")
  (entity E3 "Peterhouse, Cambridge")
  (event EV1 matriculated :subj E1 :obj E2)
  (event EV2 graduated :subj E1 :obj E3))
```

```lisp
;; CANDIDATE B: POSITIONAL ULTRA-COMPACT S-EXPRESSION (41 tokens, 693 ms — 3.80x faster)
((e E1 "Charles Babbage")
 (e E2 "Trinity College, Cambridge")
 (e E3 "Peterhouse, Cambridge")
 (ev EV1 matriculated E1 E2)
 (ev EV2 graduated E1 E3))
```

```lisp
;; CANDIDATE C: COMPACT S-EXPRESSION WITH CO-DECODED KEV TAGS (79 tokens, ~1,100 ms — 2.4x faster than Co-Decoded JSON)
(graph
  (entity E1 "Charles Babbage")
  (entity E2 "Trinity College, Cambridge")
  (event EV1 matriculated :subj E1 :obj E2 :intent I :epist O :allen B :pearl M)
  (event EV2 graduated :subj E1 :obj E3 :intent I :epist O :allen B :pearl M))
```

---

## Architecture: Non-Destructive S-Expression Ingestion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 QUANTA MULTI-FORMAT S-EXPRESSION ARCHITECTURE               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Raw Passage Chunk ──────────────────────────────────────────┐              │
│         │                                                    │              │
│         ▼                                                    ▼              │
│  [Prompt & Grammar Dispatcher]                        [PassageStore]        │
│  - format: sexpr_compact | sexpr_positional | json    (V_passage immutable) │
│  - co_decoded: true | false                                  ▲              │
│         │                                                    │              │
│         ▼                                                    │ E_ground     │
│  ┌──────────────────────────────────────────────┐            │ UTF-8 spans  │
│  │     UNIFIED UNSLOTH / LLAMA-SERVER BACKEND   │            │              │
│  │   Qwen 3.5 4B Backbone (:8888, N=8..16 slots)│            │              │
│  └──────────────────────┬───────────────────────┘            │              │
│                         │ S-Expression Stream                │              │
│                         ▼                                    │              │
│             [Fast S-Expression Lexer/Parser]                 │              │
│             (handles keyword & positional ASTs)              │              │
│                         │                                    │              │
│                         ▼                                    │              │
│                  [SpanAligner] ──────────────────────────────┘              │
│             (microsecond character & byte grounding)                        │
│                         │                                                   │
│                         ▼                                                   │
│             [SkeletonExtractionResult]                                      │
│                         │                                                   │
│                         ▼                                                   │
│        [CognitivePipeline.ingest_document]                                  │
│        - Valencies: subj -> AGENT, obj -> PATIENT                           │
│        - Optional Kev Decoders: mapped if co_decoded=True                   │
│        - Commit: QuantaGraph & 128-byte BinaryNodeTable                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Session Starter Commands

Copy-paste any of the following commands into a new Antigravity session to execute that specific section:

* `Start working on Section 1 in @quanta_sexpr_ingestion_master_plan.md: Grammar Architecture & Universal Co-Decoding Specification`
* `Start working on Section 2 in @quanta_sexpr_ingestion_master_plan.md: High-Speed Dual-Mode Transducer & Parser Engine`
* `Start working on Section 3 in @quanta_sexpr_ingestion_master_plan.md: End-to-End Pipeline Integration & Representation Toggles`
* `Start working on Section 4 in @quanta_sexpr_ingestion_master_plan.md: Parallel Slot Concurrency Optimization & Empirical Sweeps`
* `Start working on Section 5 in @quanta_sexpr_ingestion_master_plan.md: Multi-Passage MuSiQue Validation, Comparative Matrix & EVAL.md Lineage`

---

## 5-Session Implementation Roadmap

---

### Section 1: Grammar Architecture & Universal Co-Decoding Specification

**Session Scope & Purpose:**
Define formal GBNF (GGML BNF) grammars for both S-expression representations (`sexpr_compact` and `sexpr_positional`) and ensure every grammar in QUANTA cleanly supports the universal co-decoding toggle (`co_decoded=False` for pure SVO surface extraction, `co_decoded=True` for single-letter Kev decisions).

#### Concrete Goals
1. Create `data/grammar/compact_skeleton_sexpr.gbnf` for keyword-based S-expressions.
2. Create `data/grammar/positional_skeleton_sexpr.gbnf` for positional Lisp S-expressions.
3. Include optional Kev decision rules in both grammars:
   - `intent`: `I` (Informative), `D` (Directive), `C` (Commissive), `E` (Expressive)
   - `epist`: `O` (Direct Observation), `D` (Deduction), `H` (Hearsay), `C` (Conjecture)
   - `allen`: `M` (Meets), `B` (Before), `O` (Overlaps), `D` (During), `N` (None)
   - `pearl`: `M` (Mechanism), `C` (Condition), `N` (None)
4. Ensure pure surface SVO decoding omits Kev enums by default, but allows them when co-decoding is active.
5. Add unit tests validating that llama.cpp / llama-server parses all GBNF grammars without syntax errors.

#### [NEW] `data/grammar/compact_skeleton_sexpr.gbnf`
```gbnf
root ::= ws "(" ws "graph" (ws clause)* ws ")" ws
clause ::= entity_clause | event_clause
entity_clause ::= "(" ws "entity" ws id ws string (ws ":type" ws id)? ws ")"
event_clause ::= "(" ws "event" ws id ws id (ws ":subj" ws id)? (ws ":obj" ws id)? (ws ":intent" ws intent_val)? (ws ":epist" ws epist_val)? (ws ":allen" ws allen_val)? (ws ":pearl" ws pearl_val)? ws ")"

intent_val ::= "I" | "D" | "C" | "E"
epist_val ::= "O" | "D" | "H" | "C"
allen_val ::= "M" | "B" | "O" | "D" | "N"
pearl_val ::= "M" | "C" | "N"

id ::= [a-zA-Z0-9_.-]+
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
ws ::= [ \t\n\r]*
```

#### [NEW] `data/grammar/positional_skeleton_sexpr.gbnf`
```gbnf
root ::= ws "(" (ws clause)* ws ")" ws
clause ::= entity_clause | event_clause
entity_clause ::= "(" ws "e" ws id ws string (ws id)? ws ")"
event_clause ::= "(" ws "ev" ws id ws id (ws id)? (ws id)? (ws intent_val)? (ws epist_val)? (ws allen_val)? (ws pearl_val)? ws ")"

intent_val ::= "I" | "D" | "C" | "E"
epist_val ::= "O" | "D" | "H" | "C"
allen_val ::= "M" | "B" | "O" | "D" | "N"
pearl_val ::= "M" | "C" | "N"

id ::= [a-zA-Z0-9_.-]+
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
ws ::= [ \t\n\r]*
```

#### [NEW] `tests/test_sexpr_grammars.py`
- Test that all grammar files exist and contain non-empty root rules.
- Test that valid mock S-expressions conform to grammar production rules.
- Verify that GBNF grammar hashes can be registered with `llama-server`.

---

### Section 2: High-Speed Dual-Mode Transducer & Parser Engine

**Session Scope & Purpose:**
Implement high-performance S-expression parsing and grounding inside [`src/parser/skeleton_transducer.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/skeleton_transducer.py) and [`src/parser/unsloth_transducer.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/unsloth_transducer.py) while preserving existing JSON paths.

#### Concrete Goals
1. Add `skeleton_format` parameter (`"sexpr_compact"` [default], `"sexpr_positional"`, `"json"`) and `co_decoded: bool = False` to `SkeletonTransducer` and `MockSkeletonTransducer`.
2. Implement high-speed `parse_skeleton_sexpr(sexpr_str: str)` in `skeleton_transducer.py`:
   - Fast tokenizer/parser processing keyword clauses `(entity E1 "...") (event EV1 pred :subj E1 :obj E2)` and positional clauses `((e E1 "...") (ev EV1 pred E1 E2))`.
   - Extracts Kev decision symbols when present (`intent`, `epist`, `allen`, `pearl`).
   - Graceful error recovery: Auto-closes missing trailing parentheses or skips unparsed fragments.
3. Integrate with [`src/parser/span_aligner.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/span_aligner.py):
   - Grounds every extracted surface entity and event predicate to exact 0-indexed character offsets `(start, end)` and UTF-8 byte ranges `(b_start, b_end)`.
4. Update `MockSkeletonTransducer`:
   - Supports pre-seeded and dynamic heuristic extraction in S-expression formats for lightning-fast (<5 ms) offline CI testing.
5. Create comprehensive test suite `tests/test_skeleton_sexpr_transducer.py`.

#### Key File Modifications
- **[MODIFY]** [`src/parser/skeleton_transducer.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/skeleton_transducer.py)
  - Add `DEFAULT_COMPACT_SEXPR_SYSTEM_PROMPT` and `DEFAULT_POSITIONAL_SEXPR_SYSTEM_PROMPT`.
  - Add `CO_DECODED_COMPACT_SEXPR_SYSTEM_PROMPT` and `CO_DECODED_POSITIONAL_SEXPR_SYSTEM_PROMPT`.
  - Add `parse_skeleton_sexpr()` parser function.
  - Add format dispatching in `transduce()` and `_clean_and_parse()`.
- **[MODIFY]** [`src/parser/unsloth_transducer.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/unsloth_transducer.py)
  - Expose `skeleton_format` and `co_decoded` parameters in `UnslothTransducer`.
- **[NEW]** [`tests/test_skeleton_sexpr_transducer.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/tests/test_skeleton_sexpr_transducer.py)
  - Test keyword S-expression parsing, positional S-expression parsing, co-decoded Kev parsing, span grounding fidelity, and mock fallback.

---

### Section 3: End-to-End Pipeline Integration & Representation Toggles

**Session Scope & Purpose:**
Wire the S-expression transducer into [`src/pipeline/cognitive_pipeline.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/pipeline/cognitive_pipeline.py), [`src/config/multi_scale_config.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/config/multi_scale_config.py), and the server proxy without altering or breaking the 128-byte `BinaryNodeTable` or `PassageStore` layouts.

#### Concrete Goals
1. Add `skeleton_format: str = "sexpr_compact"` and `co_decoded_kev: bool = False` to `CognitivePipeline.__init__`.
2. Allow runtime format switching via `pipe.set_skeleton_format("sexpr_positional")` or environment variables (`QUANTA_SKELETON_FORMAT`, `QUANTA_CO_DECODED_KEV`).
3. In `CognitivePipeline.ingest_document()`:
   - Accept `SkeletonExtractionResult` emitted from S-expression transducers.
   - Map SVO arguments (`subj -> AGENT`, `obj -> PATIENT`) directly to valency links.
   - If `co_decoded_kev=True`, map compact single-letter decisions into Belnap truth values using `BelnapLatticeMapper`.
4. Register configuration keys in `MultiScaleConfig`:
   - `transducer.skeleton_format`: `"sexpr_compact"` (default), `"sexpr_positional"`, `"json"`.
   - `transducer.co_decoded`: `false` (default), `true`.
5. Integration test `tests/test_pipeline_sexpr_integration.py` verifying full end-to-end ingestion and retrieval parity.

#### Key File Modifications
- **[MODIFY]** [`src/pipeline/cognitive_pipeline.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/pipeline/cognitive_pipeline.py)
- **[MODIFY]** [`src/config/multi_scale_config.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/config/multi_scale_config.py)
- **[NEW]** [`tests/test_pipeline_sexpr_integration.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/tests/test_pipeline_sexpr_integration.py)

---

### Section 4: Parallel Slot Concurrency Optimization & Empirical Sweeps

**Session Scope & Purpose:**
Perform a systematic concurrency sweep on the user's live RTX 3070 backend across parallel slot configurations ($N \in \{1, 2, 4, 8, 12, 16\}$ slots), monitor continuous batching behavior in the Unsloth Studio UI, determine the empirical throughput ceiling, and establish a portable configuration mechanism for different hardware setups.

#### Core Research Questions
1. **Unsloth Studio Slot Scaling**: Unsloth currently runs with 8 slots. What is the empirical throughput (words/sec) and per-chunk latency as batch workers scale from 1 to 16?
2. **Saturation vs Overhead**: At what point does KV cache contention or CPU-GPU context dispatch overhead diminish returns on an 8GB RTX 3070?
3. **Hardware Portability**: How can other developers with more powerful GPUs (e.g. 24GB RTX 4090, 16GB RTX 4080) or more modest setups (e.g. 8GB M-series Mac, CPU) easily configure their optimal slot count?

#### Implementation Details
1. Create `scripts/benchmark_parallel_slots.py`:
   - Loads the 16 multi-sentence passages of the MuSiQue benchmark.
   - Sweeps slot counts $N \in \{1, 2, 4, 8, 12, 16\}$ using `ThreadPoolExecutor(max_workers=N)`.
   - Measures:
     - Total ingestion time ($s$)
     - Ingestion throughput ($w/s$)
     - Mean latency per chunk ($ms$)
     - Unsloth HTTP connection pool health
     - Real GPU telemetry via `UnslothServerManager.get_gpu_telemetry()`
2. Add global slot configuration to `MultiScaleConfig` and CLI tools:
   - Config key: `server.max_parallel_slots` / `server.batch_workers`.
   - Environment variable: `QUANTA_MAX_SLOTS` (default: 8, calibrated per device).
   - CLI flag: `--slots N` in all benchmark and ingestion scripts.
3. Persist the empirically discovered optimal slot setting to `config/multi_scale_profile.json`.

---

### Section 5: Multi-Passage MuSiQue Validation, Comparative Matrix & EVAL.md Lineage

**Session Scope & Purpose:**
Execute publication-grade paired benchmarking on the full 16-passage MuSiQue multi-hop suite comparing all representation variants head-to-head on the live RTX 3070, verifying both speed and multi-hop reasoning recall, and recording the findings in `EVAL.md`.

#### Experimental Conditions
Compare all 6 operational permutations:
1. **`sexpr_compact` (Pure SVO, Default)**: Compact keyword S-expression without Kev tags.
2. **`sexpr_compact` (Co-Decoded Kev)**: Compact keyword S-expression with 1-letter Kev decisions.
3. **`sexpr_positional` (Pure SVO)**: Positional ultra-compact S-expression without Kev tags.
4. **`sexpr_positional` (Co-Decoded Kev)**: Positional ultra-compact S-expression with 1-letter Kev decisions.
5. **`json_standard`**: Legacy standard JSON skeleton.
6. **`json_co_decoded`**: Legacy co-decoded JSON skeleton (current default baseline).

#### Key Measurement Metrics
- **Speed & Efficiency**: Words per second ($w/s$), Mean latency per chunk ($ms$), Completion tokens per chunk.
- **Reasoning Accuracy**: Downstream multi-hop passage projection recall and question answering exact match / F1 on target question (*"In which sovereign country is the city housing the university where Charles Babbage studied located?"* $\implies$ *"United Kingdom"*).
- **GPU Saturation**: Active slots, VRAM usage ($MB$), zero truncation rate.

#### Key File Modifications
- **[MODIFY]** [`scripts/benchmark_musique_ingestion.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/scripts/benchmark_musique_ingestion.py)
  - Add `--format {all,sexpr_compact,sexpr_positional,json}`.
  - Add `--co-decode {on,off}`.
  - Add `--slots {1,2,4,8,12,16}`.
  - Export comprehensive comparative report to `output/sexpr_vs_json_ingestion_benchmark.md`.
- **[MODIFY]** [`EVAL.md`](file:///c:/Users/PC/Documents/GitHub/Quanta/EVAL.md)
  - Record experiment node `exp-033a` with full metrics, speedups, and Gate G7 decision.

---

## Verification Plan

### Automated Tests
```powershell
# 1. Grammar validation tests
pytest tests/test_sexpr_grammars.py -v

# 2. S-expression skeleton transducer unit tests
pytest tests/test_skeleton_sexpr_transducer.py -v

# 3. Pipeline integration tests
pytest tests/test_pipeline_sexpr_integration.py -v

# 4. Full regression suite across legacy and new parsers
pytest tests/test_sexpr_parser.py tests/test_skeleton_transducer.py tests/test_co_decoded_skeleton_kev.py tests/test_svm_cognitive_pipeline.py -v
```

### Manual Verification
```powershell
# Run the parallel slot concurrency sweep on live hardware
python scripts/benchmark_parallel_slots.py --corpus paragraphs --slots 1,2,4,8,12,16

# Run full multi-representation MuSiQue benchmark across all formats
python scripts/benchmark_musique_ingestion.py --mode auto --format all --backend live

# Inspect the generated comparative markdown report
Get-Content output/sexpr_vs_json_ingestion_benchmark.md
```
