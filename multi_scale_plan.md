# Dynamic Multi-Scale Ingestion & Task-Aware Kev Gating Master Plan (`multi_scale_plan.md`)

## Architectural Overview

This master plan extends the QUANTA Semantic Virtual Memory (SVM) architecture with **Dynamic Multi-Scale Ingestion**. Instead of processing massive input contexts uniformly through high-overhead micro-scale S-expression transduction, 3-pass Kev relational logprob scoring, and Clingo difference-logic verification, the ingestion engine introduces a **hierarchical coarse-to-fine gating pipeline**:

1. **Task Boundary & Intent Extraction:** Intercepts long prompts and identifies the user prompt's instruction/question target $\mathbf{q}$ from the head or tail boundaries before chunking.
2. **Two-Tier Chunk Hierarchy:** Context is parsed into coarse macro-blocks ($1{,}000\text{--}2{,}000$ tokens) mapped via zero-copy SIMD concept codebooks ($0.32\,\mu\text{s}$ per concept), alongside micro-blocks ($150\text{--}350$ words).


3. **Kev-4B Coarse Relevance Gating:** Evaluates each macro-block against $\mathbf{q}$ in a single non-autoregressive prefill logprob pass ($20\text{--}35\,\text{ms}$) over single-token letters `(A) CRITICAL`, `(B) BACKGROUND`, `(C) IRRELEVANT`.


4. **Selective High-Precision Transduction:** Chunks scoring `CRITICAL` undergo full micro-scale transduction (Skeleton Transduction, 3-pass Kev valency/causal scoring, and Clingo-DL verification). `BACKGROUND` chunks register solely as lightweight concept-annotated passage nodes ($V_{\text{passage}}$).


5. **Multi-Scale HippoRAG 2 & PoP-RAG Spreading Activation:** Personalized PageRank propagates activation seamlessly across both macro-concept nodes and micro-event DAGs, with PoP-RAG damping unverified paths.


6. **Speculative Early Answering & Background Queue:** The downstream model generates the initial answer immediately upon resolving hot subgraph paths. Full ingestion of remaining cold chunks proceeds asynchronously in the background (or terminates early in evaluation/test harnesses).

```
 Raw Discourse / Prompt (10k-100k tokens)
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 0: Task Boundary & Intent Scanner               │
  │  Extracts user instruction anchor q from Head / Tail   │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 1: Macro-Blocker & SIMD Mmap Concept Harvester  │
  │  1,000–2,000 token blocks + 0.32µs ConceptNet codebook │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 2: Kev-4B Coarse Relevance Gater (llama-server) │
  │  1-Pass Prefill Logprob Scoring: [A] CRIT / [B] BG / [C] IRR │
  └──────────┬─────────────────────────────────┬───────────┘
             │ [CRITICAL]                      │ [BACKGROUND / IRR]
             ▼                                 ▼
  ┌──────────────────────────────┐  ┌──────────────────────┐
  │ Stage 3: Full Micro Ingestion│  │ Coarse Registration  │
  │ - Skeleton Transducer (40tok)│  │ - V_passage Nodes    │
  │ - 3-Pass Kev-4B (Relational) │  │ - Concept Anchors    │
  │ - Clingo-DL Verification     │  │                      │
  └──────────────┬───────────────┘  └──────────┬───────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 4: Multi-Scale HippoRAG 2 + PoP-RAG PPR         │
  │  Spreading activation over hybrid Macro/Micro topology │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 5: Fast-Path Answer Generation (Low TTFT)       │
  │  Emit response using hot subgraph + pristine spans     │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 6: Asynchronous Ingestion Worker (Background)   │
  │  Processes remaining background chunks while idle      │
  └────────────────────────────────────────────────────────┘

```

---

## Session Starter Commands

Copy-paste any of the following commands into a new Antigravity session to execute that implementation phase sequentially:

* `Start working on Section 1 in @multi_scale_plan.md: Task Boundary Scanner & Intent Extraction Engine`
* `Start working on Section 2 in @multi_scale_plan.md: Hierarchical Macro-Chunker & Mmap Concept Profiler`
* `Start working on Section 3 in @multi_scale_plan.md: Kev-4B Coarse Relevance Gater & Single-Token Logprob Scorer`
* `Start working on Section 4 in @multi_scale_plan.md: Multi-Scale Dual-Node ASG & Macro/Micro Graph Topology`
* `Start working on Section 5 in @multi_scale_plan.md: Multi-Scale HippoRAG 2 PPR & PoP-RAG Activation Traversal`
* `Start working on Section 6 in @multi_scale_plan.md: Speculative Fast-Path Context Assembly & TTFT Bypass`
* `Start working on Section 7 in @multi_scale_plan.md: Asynchronous Background Ingestion Queue & Early Termination Guards`
* `Start working on Section 8 in @multi_scale_plan.md: End-to-End Pipeline Rewire, Proxy Telemetry & Paired Evaluation Suite`

---

## Hardware Budgets & Throughput Targets (RTX 3070 8GB)

| Ingestion Parameter | Legacy Monolithic Ingestion

 | Redesigned SVM Master Plan

 | Dynamic Multi-Scale Ingestion (This Plan) |
| --- | --- | --- | --- |
| **VRAM Consumption** | $\sim 6.3\,\text{GB}$<br> | $\sim 4.8\,\text{GB}$ (Shared 4B)

 | **$\sim 4.8\,\text{GB}$ (Strictly Shared 4B)** |
| **Initial TTFT (10k tokens)** | $8{,}500\text{--}12{,}000\,\text{ms}$ | $3{,}500\text{--}5{,}200\,\text{ms}$<br> | **$550\text{--}850\,\text{ms}$ ($6\times\text{ to }10\times$ faster)** |
| **Initial TTFT (50k tokens)** | $42{,}500\,\text{ms}$<br> | $17{,}500\text{--}26{,}000\,\text{ms}$ | **$1{,}200\text{--}1{,}900\,\text{ms}$ ($12\times\text{ to }15\times$ faster)** |
| **Prefill Ingestion Passes** | Full 3-pass Kev on all 150 chunks

 | Full 3-pass Kev on all 150 chunks

 | **5–8 macro prefill passes + 6–10 micro passes** |
| **Background Ingestion** | None (Synchronous blocking) | None (Synchronous blocking) | **Non-blocking background thread / event loop** |
| **Evaluation Mode Behavior** | Mandatory full ingestion | Mandatory full ingestion | **Immediate stop upon fast-path answer emission** |

---

### Section 1: Task Boundary Scanner & Intent Extraction Engine

**Session Scope & Purpose:**
In real-world prompts, long documents, and conversational turns, the active query or instruction almost never resides in the middle of a 20,000-word payload; it is situated either at the absolute beginning (pre-prompt directive) or the absolute end (closing query/instruction). This session builds a lightweight, sub-5ms heuristic and regex boundary parser that isolates the actual instruction target $\mathbf{q}$ and its epistemic focus before running any chunking or model inference.

#### [NEW] `src/parser/task_boundary_extractor.py`

* Implements `ExtractedTaskIntent` dataclass:
* `query_text: str`
* `boundary_location: Literal["HEAD", "TAIL", "ISOLATED"]`
* `char_span: Tuple[int, int]`
* `target_entities: List[str]`
* `epistemic_goal: Literal["FACTUAL", "TEMPORAL", "CAUSAL", "AGGREGATION", "ACTION"]`


* Implements `TaskBoundaryExtractor`:
* `scan_prompt_boundaries(text: str, head_token_limit: int = 500, tail_token_limit: int = 500) -> ExtractedTaskIntent`
* Heuristic detection:
* Scans trailing paragraphs for interrogative syntax (`?`, `"Explain..."`, `"What is..."`, `"Calculate..."`, `"Based on the above..."`).
* Scans leading sections for system role directives (`"You are...", "Your task is to..."`).
* If ambiguous, extracts candidate sentences with highest question-particle and imperative verb density.


* Fallback: If no localized question is found, falls back to the final non-empty block.



#### [NEW] `tests/test_task_boundary_extractor.py`

* Test prompt extraction with instructions located at the beginning (prompt framing).
* Test extraction with instructions appended after 50,000 characters of background text (tail query).
* Test extraction on standard benchmarks (MuSiQue question at tail, NIAH needle question, HumanEval docstring prompt).


* Benchmark boundary scanning latency ($< 3.0\,\text{ms}$ execution time across 100k characters).

---

### Section 2: Hierarchical Macro-Chunker & Mmap Concept Profiler

**Session Scope & Purpose:**
Establish a two-tier segmentation hierarchy. The existing `DiscourseChunker` segments text into $150\text{--}350$ word micro-chunks for fine-grained predicate extraction. This session implements a coarse `MacroChunker` that segments discourse into $1{,}000\text{--}2{,}000$ token macro-blocks, and pairs it with the zero-copy memory-mapped codebook (`data/concept_codebook.bin`) to harvest bag-of-concept vectors in sub-0.5ms without running neural models.

#### [NEW] `src/parser/multi_scale_chunker.py`

* Implements `MacroBlock` dataclass:
* `macro_id: str` (e.g. `"MB01"`)
* `char_span: Tuple[int, int]`
* `text: str`
* `concept_codes: List[int]` (ConceptNet 16-bit concept IDs extracted via mmap)


* `surface_entities: List[str]`
* `micro_chunks: List[DiscourseChunk]` (Child micro-chunks ready for deferred processing)


* Implements `HierarchicalChunker`:
* `segment_document(text: str, macro_target_tokens: int = 1500, micro_target_words: int = 250) -> List[MacroBlock]`
* Preserves hierarchical parent-child span alignment: each micro-chunk points to its parent `macro_id`.
* Integrates with `MmapLexicalGrounder` to profile every macro-block's salient concept signatures in $< 0.5\,\text{ms}$.





#### [MODIFY] `src/parser/chunker.py`

* Add parent reference pointer to `DiscourseChunk`:
* `parent_macro_id: Optional[str] = None`
* `is_ingested_full: bool = False`



#### [NEW] `tests/test_multi_scale_chunker.py`

* Test hierarchical parent-child span invariants across synthetic documents.
* Verify macro-chunk size distributions remain within $[1000, 2000]$ token limits.
* Benchmark SIMD mmap concept profiling across 20 macro-blocks ($< 5.0\,\text{ms}$ total CPU time).

---

### Section 3: Kev-4B Coarse Relevance Gater & Single-Token Logprob Scorer

**Session Scope & Purpose:**
Implement the coarse gating head in `KevDecisionEngine`. Given the isolated user query anchor $\mathbf{q}$ and a macro-block's concept profile/header, Kev-4B evaluates relevance non-autoregressively using a single prefill logprob pass ($n_{\text{predict}}=1$) via `llama-server` `/completion`. Using the atomic single-token option letter strategy (`A`, `B`, `C`), this evaluates relevance in $20\text{--}35\,\text{ms}$ per macro-block without generating tokens.

#### [MODIFY] `src/models/kev_engine.py`

* Add `score_macro_block_relevance`:
```python
async def score_macro_block_relevance(
    self,
    task_intent: ExtractedTaskIntent,
    macro_block: MacroBlock,
) -> Tuple[str, float]:
    """
    Scores macro-block relevance against query anchor using 1-token logprob prefill.
    Returns (label, calibrated_probability) where label in {"CRITICAL", "BACKGROUND", "IRRELEVANT"}.
    """

```


* Structured prefill template:
```text
Context Summary: {concept_keywords}
Passage Excerpt: {first_250_tokens_of_macro_block}
Question: {task_intent.query_text}

How relevant is this context to answering the question?
(A) CRITICAL: Contains direct evidence or causal mechanisms needed to answer.
(B) BACKGROUND: Contains contextual or tangential background information.
(C) IRRELEVANT: Completely unrelated topic or distractor.

Classification [A/B/C]:

```


* Reads normalized next-token logprobs for `' A'`, `' B'`, `' C'` and computes closed-form softmax probabilities.


* Threshold mapping:
* $P(\text{CRITICAL}) \ge 0.50 \implies \text{CRITICAL}$ (Triggers immediate micro-ingestion)
* $P(\text{BACKGROUND}) \ge 0.40$ or $0.20 \le P(\text{CRITICAL}) < 0.50 \implies \text{BACKGROUND}$ (Registers coarse-only)
* Otherwise $\implies \text{IRRELEVANT}$ (Skipped or registered as cold passage node)



#### [NEW] `scripts/evaluate_macro_gater.py`

* Test relevance classification accuracy, latency, and calibration on synthetic and MuSiQue distractor passages (gold bridge vs. distractor passages).


* Assert Macro F1 $\ge 90\%$ on distinguishing critical bridge context from unrelated distractors.

#### [NEW] `tests/test_kev_macro_gater.py`

* Test logprob extraction across simulated responses for `A`, `B`, `C` candidates.


* Verify that batch evaluation of 8 macro-blocks completes within $< 250\,\text{ms}$ on `llama-server`.



---

### Section 4: Multi-Scale Dual-Node ASG & Macro/Micro Graph Topology

**Session Scope & Purpose:**
Adapt the dual-node bipartite graph topology ($V_{\text{passage}}$, $V_{\text{semantic}}$) to support multi-scale representations. Cold chunks must not pollute the graph with millions of unneeded syntactic nodes, yet their information must remain reachable during spreading activation. This session implements a multi-scale ASG schema where macro-blocks exist as coarse passage nodes tied to concept clusters, while hot chunks unfold into fine-grained event-predicate DAGs.

#### [MODIFY] `src/memory/passage_store.py`

* Add `granularity` and `hierarchy` attributes to `PassageRecord`:


* `granularity: Literal["MACRO", "MICRO"] = "MICRO"`
* `parent_macro_id: Optional[str] = None`
* `concept_codes: List[int] = field(default_factory=list)`
* `ingestion_status: Literal["COARSE_ONLY", "PENDING_BACKGROUND", "FULLY_INGESTED"]`



#### [MODIFY] `src/core/asg.py`

* Support multi-scale node linking in `QuantaGraph`:


* Introduce `add_macro_concept_anchor(macro_passage_id: str, concept_code: int)`
* Allow `QuantaNode` instances to bind to both a micro-passage and its parent macro-passage.
* Implement dynamic subgraph unfolding: when a macro-block is later upgraded to full precision in the background, its child micro-nodes seamlessly merge into the active graph without invalidating existing edges.



#### [NEW] `tests/test_multi_scale_asg.py`

* Test graph construction with a mixture of coarse macro-passage anchors and fine-grained event DAGs.
* Test in-place graph expansion when an existing `COARSE_ONLY` passage is upgraded to `FULLY_INGESTED`.
* Validate binary table consistency and 128-byte struct generation for hybrid nodes.



---

### Section 5: Multi-Scale HippoRAG 2 PPR & PoP-RAG Activation Traversal

**Session Scope & Purpose:**
Extend Personalized PageRank (PPR) spreading activation to propagate energy across a hybrid multi-scale topology. When a query is issued, activation seeds spread through fine-grained Allen/Pearl causal edges on hot chunks, but can also traverse coarse concept-codebook bridges into background macro-blocks if multi-hop evidence requires it. PoP-RAG epistemic weights dampen unverified paths.

#### [MODIFY] `src/memory/hipporag_ppr.py`

* Update `HippoRAGRetriever` transition matrix construction:
* Add transition weights between macro-concept nodes and fine-grained entity nodes.
* Set inter-scale scaling factor $\gamma_{\text{scale}} = 0.65$: coarse connections transmit lower energy than validated micro-causal edges to prioritize verified facts while maintaining global connectivity.


* Vectorized power iteration over hybrid adjacency matrices:

$$\mathbf{p}^{(t+1)} = (1 - \alpha) \mathbf{W}_{\text{multi}} \mathbf{p}^{(t)} + \alpha \mathbf{p}^{(0)}$$



#### [MODIFY] `src/memory/poprag_gating.py`

* Modulate weights based on node ingestion status and Belnap truth lattice:
* `FULLY_INGESTED` + `TRUE` ($01_2$): nominal full weight ($1.0$)


* `COARSE_ONLY`: soft heuristic weight ($0.45$)
* `CONTRADICTION` ($00_2$): pruned to $0.0$




#### [NEW] `tests/test_multi_scale_hipporag.py`

* Test PPR convergence across a graph with 3 fully-ingested micro-clusters and 10 coarse macro-blocks.
* Verify that a multi-hop query successfully travels across coarse nodes to identify a distant relevant block.
* Assert traversal latency stays under $5.0\,\text{ms}$ on hybrid graphs of 50,000 nodes.



---

### Section 6: Speculative Fast-Path Context Assembly & TTFT Bypass

**Session Scope & Purpose:**
Eliminate Time-To-First-Token (TTFT) lag by decoupling answer generation from background ingestion completeness. If the initial hot subgraph + top macro-passages provide sufficient epistemic coverage for the user query $\mathbf{q}$, assemble the dual-stream context block immediately and trigger the downstream LLM generation, bypassing full document compilation.

#### [NEW] `src/memory/fast_path_assembler.py`

* Implements `FastPathCoverageEvaluator`:
* `check_coverage(query_seeds: Set[str], active_subgraph: QuantaGraph, top_passages: List[PassageRecord]) -> Tuple[bool, float]`
* Calculates epistemic confidence: if target query entities are grounded and causal/temporal chain satisfies the query constraint, returns `(True, confidence)`.


* Implements `FastPathContextAssembler`:
* Builds dual-stream prompt:
1. **Stream 1: Verified Active Subgraph Briefing** (Fine-grained facts from hot chunks)


2. **Stream 2: High-Ranking Verbatim Raw Spans** (From both hot micro-passages and top macro-passages)




* Returns context ready for immediate streaming completion.



#### [MODIFY] `src/memory/context_assembler.py`

* Bridge `DualStreamContextAssembler` to consume hybrid multi-scale ranking scores.



#### [NEW] `tests/test_fast_path_assembler.py`

* Test coverage evaluation on queries with complete hot paths vs queries requiring background expansion.
* Verify dual-stream context format strictly matches downstream reader requirements.


* Benchmark assembly time ($< 2.0\,\text{ms}$).

---

### Section 7: Asynchronous Background Ingestion Queue & Early Termination Guards

**Session Scope & Purpose:**
Implement the non-blocking background ingestion worker. While the host LLM generates the answer or waits for the user's next turn in chat mode, an asynchronous worker incrementally digests remaining `BACKGROUND` chunks into full micro-scale ASG nodes. Crucially, implement **Early Termination Guards** so that in testing, benchmarking, or CI environments (`QUANTA_BACKGROUND_INGESTION=0`), background processing is disabled entirely, guaranteeing minimal runtime and zero GPU resource waste.

#### [NEW] `src/pipeline/background_ingestor.py`

* Implements `BackgroundIngestionWorker`:
* Uses `asyncio.Queue` or a bounded worker thread pool.
* Priority queue ordered by macro relevance score ($P(\text{CRITICAL}) > P(\text{BACKGROUND})$).
* Background loop:
1. Pops next pending micro-chunk.
2. Executes `SkeletonTransducer` (30–45 tokens).


3. Executes `KevDecisionEngine` 3-pass prefill scoring.


4. Validates with `ClingoDLGate` and merges into global `QuantaGraph`.




* Graceful cancellation and pause triggers:
* Automatically pauses when incoming interactive user requests query `llama-server` to avoid GPU contention.
* Resumes when the server is idle.




* Implements `IngestionPolicyGuard`:
* Inspects environment flag `QUANTA_BACKGROUND_INGESTION`:
* `"0"` or `"off"`: Immediate stop after fast-path answer emission (for tests/benchmarks).
* `"1"` or `"on"`: Detaches background worker to finish full document indexing.





#### [NEW] `tests/test_background_ingestor.py`

* Test queuing, background task execution, and dynamic graph updating.
* Test cancellation guard when `QUANTA_BACKGROUND_INGESTION="0"`.
* Test queue prioritization: higher-scoring background chunks are ingested first.

---

### Section 8: End-to-End Pipeline Rewire, Proxy Telemetry & Paired Evaluation Suite

**Session Scope & Purpose:**
Wire all components into `CognitivePipeline`, update the reverse proxy (`:8000`) and MCP tools, and run a comprehensive head-to-head empirical evaluation against the monolithic SVM baseline on live RTX 3070 hardware. Measure TTFT reduction, total ingestion throughput, and accuracy parity on long-context benchmarks (MuSiQue, BABILong, NIAH).

#### [MODIFY] `src/pipeline/cognitive_pipeline.py`

* Rewire `ingest_and_query_streaming`:
1. Extract task intent $\mathbf{q}$ via `TaskBoundaryExtractor`.
2. Segment into macro-blocks via `HierarchicalChunker`.


3. Score macro relevance via `KevDecisionEngine.score_macro_block_relevance`.


4. Perform immediate micro-ingestion on `CRITICAL` chunks.


5. Register coarse anchors for `BACKGROUND` chunks.


6. Execute multi-scale HippoRAG 2 PPR.


7. If fast-path coverage holds, assemble context and return answer immediately.
8. If `QUANTA_BACKGROUND_INGESTION=1`, dispatch remaining chunks to `BackgroundIngestionWorker`.



#### [MODIFY] `src/server/proxy.py`

* Add control headers:
* `X-Quanta-Dynamic-Granularity: true` (default: true)
* `X-Quanta-Background-Ingest: false` (allows test client to toggle background worker)


* Expose telemetry in HTTP response headers:
* `X-Quanta-TTFT-Ms`: Time taken from request receipt to context assembly.
* `X-Quanta-Macro-Blocks`: Total macro-blocks scanned.
* `X-Quanta-Hot-Chunks`: Number of micro-chunks fully ingested before answer.
* `X-Quanta-Deferred-Chunks`: Number of micro-chunks queued for background.



#### [NEW] `scripts/evaluate_dynamic_ingestion.py`

* Paired live hardware benchmark suite:
* Compares Monolithic Full Ingestion vs Dynamic Multi-Scale Ingestion across:
1. **MuSiQue 20-passage tasks ($N=30$)**

2. **NIAH long context ($N=5$, 4k to 64k tokens)**

3. **BABILong long-horizon state tracking ($N=5$)**



* Collects TTFT (ms), End-to-End latency (s), Ingestion throughput (tok/s), Peak VRAM (GB), and QA Accuracy (%).


* Emits publication-ready Markdown scorecard: `output/dynamic_ingestion_ablation_report.md`.



#### [NEW] `tests/test_dynamic_ingestion_pipeline.py`

* End-to-end integration test of the full dynamic pipeline.
* Assert TTFT on a 10,000-token prompt drops below $850\,\text{ms}$ on RTX 3070 (mock/live).
* Assert accuracy parity: 100% agreement with full-ingestion on gold test assertions.

---

## Verification Plan

### Automated Regression & Unit Suite

Run sequentially in Windows PowerShell as each section is implemented:

```powershell
# Section 1: Task Boundary & Intent Scanner
pytest tests/test_task_boundary_extractor.py -v

# Section 2: Hierarchical Macro-Chunker & Mmap Profiler
pytest tests/test_multi_scale_chunker.py -v

# Section 3: Kev-4B Coarse Relevance Gater
pytest tests/test_kev_macro_gater.py -v
python scripts/evaluate_macro_gater.py --quick

# Section 4: Multi-Scale Dual-Node ASG
pytest tests/test_multi_scale_asg.py -v

# Section 5: Multi-Scale HippoRAG 2 PPR
pytest tests/test_multi_scale_hipporag.py -v

# Section 6: Speculative Fast-Path Context Assembler
pytest tests/test_fast_path_assembler.py -v

# Section 7: Background Ingestion Queue & Early Termination Guards
pytest tests/test_background_ingestor.py -v

# Section 8: End-to-End Pipeline Rewire & Paired Benchmark
pytest tests/test_dynamic_ingestion_pipeline.py -v
python scripts/evaluate_dynamic_ingestion.py --mode live --samples 10

```

### Manual Hardware & Telemetry Verification

1. **Time-To-First-Token (TTFT) Audit:** Submit a 20,000-token document containing a specific query at the end. Verify via proxy headers (`X-Quanta-TTFT-Ms`) that the first answer token streams back in $< 1{,}000\,\text{ms}$.
2. **VRAM Concurrency Audit:** Run `nvidia-smi` while the background worker is ingesting cold chunks during a stream completion to confirm memory remains $\le 4.8\,\text{GB}$ on the RTX 3070 without GPU OOM crashes.


3. **Early Termination Verification:** Run `pytest tests/ -v` with `QUANTA_BACKGROUND_INGESTION=0` and confirm that test suites finish in seconds without background ingestion worker thread leakage.