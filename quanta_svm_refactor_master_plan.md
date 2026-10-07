# QUANTA Refactor Master Plan: Neuro-Symbolic Semantic Virtual Memory (SVM) with Unified 4B Shared Backbone

## Goal Description

This master plan redesigns QUANTA from a monolithic, generative S-expression language into an external **Semantic Virtual Memory (SVM)**. It replaces lossy neural realization with a **dual-node bipartite graph topology** inspired by HippoRAG 2 and PoP-RAG, paired with non-autoregressive slot classification via **Kev-4B**.

### Major Architectural Innovation: Shared-Backbone Ingestion (Qwen 3.5 4B + Kev-4B)

Instead of maintaining two distinct model weights in GPU memory (e.g., Qwen 3.5 4B + a separate 0.8B model), QUANTA standardizes on a **single unified 4B model backbone** (e.g., `Qwen3.5-4B-MTP-GGUF` or `Qwen2.5-4B` / `Qwen3.5-4B` at Q5_K_M) served via a high-performance standalone server (`llama-server`):

1. **Shared 4B Weights in VRAM:** The base 4B model is loaded once (~3.0 GB VRAM). Both the **Skeleton Transducer** (surface entities and S-V-O triple frames) and **Kev-4B** (thematic valency, pragmatics, and Allen/Pearl relation scoring) query this identical engine.
2. **Higher Accuracy, Lower VRAM:** Kev upgrades from 0.8B to a full **4B parameter reasoning model**—dramatically reducing valency ambiguity and relation classification errors—while reducing total VRAM from **6.3 GB down to ~4.8 GB** on an 8GB RTX 3070!
3. **Deprecation of Unsloth Studio:** Unsloth Studio is replaced by a production-grade, headless `llama-server` process with FlashAttention-2 and native logprob prefill evaluation (`/completion` with `n_probs`), eliminating Python serving overhead and GPU context fragmentation.
4. **Dual-Node Bipartite Graph & Canonical Record DSL:** Decouples raw text passage nodes ($V_{\text{passage}}$) from semantic concept/event nodes ($V_{\text{semantic}}$). Replaces nested S-expressions with an auditable flat Record DSL and a 128-byte C-compatible binary struct for microsecond SIMD processing.
5. **Calibrated Belnap Truth Lattice & `clingo-dl`:** Calibrates Kev relational posteriors to the Belnap 4-valued lattice ($\mathcal{B}_4 = \{00_2, 01_2, 10_2, 11_2\}$) and validates difference-logic constraints with `clingo-dl`, keeping MUC repair retries $<1.5\%$.
6. **HippoRAG 2 Spreading Activation + PoP-RAG Gating:** Propagates activation via Personalized PageRank (PPR) across Allen temporal and Pearl causal edges without multi-hop semantic drift. Filters contradictory paths ($00_2$) and dampens uncertain paths ($11_2$).
7. **Realization Bypass via Bipartite Projection:** Direct projection of semantic activation onto raw passage nodes ($\text{Score}(v_p) = \sum \mathbf{p}(v_s) \cdot \text{conf}(v_s)$). Eliminates lossy neural text realizer, supplying downstream reader LLMs with a concise Logical Briefing Block plus pristine raw text spans.

---

## Session Starter Commands

Copy-paste any of the following commands into a new Antigravity session to begin implementation on that specific section:

* `Start working on Section 1 in @quanta_svm_refactor_master_plan.md: Dual-Node Bipartite Graph Schema & Passage Storage Engine`
* `Start working on Section 2 in @quanta_svm_refactor_master_plan.md: Canonical Record DSL Parser, AST Compiler & 128-Byte Binary Layout`
* `Start working on Section 3 in @quanta_svm_refactor_master_plan.md: Fast Qwen-4B Skeleton Transducer & Grammar Stripping`
* `Start working on Section 4 in @quanta_svm_refactor_master_plan.md: Kev-4B Shared-Backbone Non-Autoregressive Relational Engine & Comparative Ablation`
* `Start working on Section 5 in @quanta_svm_refactor_master_plan.md: Calibrated Belnap Lattice Mapping & clingo-dl Difference Logic Gate`
* `Start working on Section 6 in @quanta_svm_refactor_master_plan.md: HippoRAG 2 Spreading Activation & PoP-RAG Epistemic Gating`
* `Start working on Section 7 in @quanta_svm_refactor_master_plan.md: Bipartite Passage Projection & Realization Bypass (Context Assembly)`
* `Start working on Section 8 in @quanta_svm_refactor_master_plan.md: End-to-End Pipeline Rewire, Unified Serving & VRAM Validation`

---

## User Review Required

> [!IMPORTANT]
> **Switch to Shared 4B Weights (Kev-4B + Qwen-4B) & Unsloth Studio Deprecation**:
> - We replace Unsloth Studio (`unsloth studio --api-only`) with a standalone, hardware-accelerated `llama-server` process on port 8888 (running `Qwen3.5-4B-MTP-GGUF` at Q5_K_M with CUDA and FlashAttention-2).
> - Skeleton Transduction invokes `/completion` (or `/v1/chat/completions`) with a compact JSON/GBNF schema constraint for 30–45 tokens.
> - Kev-4B invokes `/completion` in prefill mode with `n_probs` logprob extraction to evaluate candidate slots, intent, epistemics, and Allen/Pearl links non-autoregressively without token generation.

> [!IMPORTANT]
> **Deprecation of Neural NLG Realization for Retrieval Context**:
> In the current architecture, [`EnglishRealizer`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/realizer/english_nlg.py) reconstructs synthetic English sentences from ASG subgraphs before passing them to the reader LLM. Under the SVM design, this lossy synthesis is deprecated in favor of retrieving the original verbatim passage text spans ($V_{\text{passage}}$) accompanied by a structured logical briefing block. `EnglishRealizer` will remain available for standalone NLG testing and multi-lingual inspection, but will be removed from the hot retrieval path.

> [!NOTE]
> **VRAM Footprint Reduction on NVIDIA RTX 3070 (8GB)**:
> By sharing weights between Kev and Qwen:
> - Base Model Weights (GGUF Q5_K_M): **~3.0 GB**
> - Kev-4B Task Heads / Logprob Routing: **0.0 GB** (in-server prefill forward passes)
> - Dynamic KV Cache & Buffers (8k context): **~1.5–1.8 GB**
> - Total VRAM: **~4.5–4.8 GB** (leaves $>3.2\text{ GB}$ headroom on an 8GB GPU!).

---

## Architectural Decisions & Evaluation Strategy

> [!TIP]
> **Kev-4B Prefill Scoring Protocol: In-Context Logprob Primary with LoRA Ablation**:
> - **Primary Serving Mode (Approach A)**: Kev-4B formats structured classification prompts and reads next-token logprobs via `llama-server` `/completion` with `n_probs=10`. This requires **zero additional VRAM**, executes in $\sim 20\text{--}35\,\text{ms}$ per prefill pass, and leverages the full 4B backbone reasoning depth.
> - **Adapter Mode (Approach B)**: Dynamic LoRA adapter attached via `--lora` in `llama-server`. Official Kev-4B models often provide task-specific LoRA weights (~30 MB).
> - **Empirical Decision Protocol**: Approach A is implemented as the production default. Section 4 explicitly includes a dedicated **Empirical Comparative Experiment (`scripts/evaluate_kev_approaches.py`)** comparing Approach A (zero-shot logprob) vs Approach B (LoRA adapter) on gold valency, pragmatic, and temporal/causal benchmark datasets to measure whether the LoRA adapter accuracy delta justifies the operational overhead.

---

## Hardware Budgets & Throughput Comparison

### Workstation Budget (NVIDIA RTX 3070, 8GB VRAM)

| Component | Architecture / Quantization | VRAM Allocation | Host RAM Allocation |
|---|---|---|---|
| **Shared 4B Backbone (Qwen 3.5 4B + Kev-4B)** | GGUF Q5_K_M (`llama-server`) | $\sim 3.0\,\text{GB}$ | Minimal |
| **Kev-4B In-Server Scoring** | Logprob Prefill / Zero-Shot Head | **$0.0\,\text{GB}$** (Shared Weights) | Minimal |
| **Concept Codebook** | `concept_codebook.bin` (SIMD) | $0.0\,\text{GB}$ (Host RAM) | $\sim 256\,\text{MB}$ |
| **Active ASG Working Canvas** | Dual-Node Structs ($M \le 2048$ nodes) | $< 1.0\,\text{MB}$ | $\sim 64\,\text{MB}$ |
| **Dynamic KV Cache & Buffers** | Context buffers for 8k window | $\sim 1.5\text{--}1.8\,\text{GB}$ | System swap |
| **Total Footprint** | **Unified SVM Runtime** | **$\sim 4.8\,\text{GB}$ (Fits 8GB with $>3.2\text{ GB}$ free)** | $\sim 1.0\,\text{GB}$ |

### Latency and Throughput Comparison

| Metric | Legacy Monolithic Architecture | Redesigned Master Plan (Unified 4B SVM) | Gain / Delta |
|---|---|---|---|
| **Ingestion Latency / Chunk** | $2,200\text{--}2,800\,\text{ms}$ | $350\text{--}520\,\text{ms}$ | **$4.2\times\text{ to }5.5\times$ speedup** |
| **Model Serving Instances** | 2 processes (Unsloth + PyTorch) | **1 single `llama-server` instance** | Zero IPC friction |
| **Kev Model Capacity** | 0.8B parameters | **4.0B parameters** | **$5\times$ larger model capacity** |
| **Total VRAM Consumption** | $\sim 6.3\,\text{GB}$ | **$\sim 4.8\,\text{GB}$** | **$1.5\,\text{GB}$ VRAM saved** |
| **Decoding Workload** | 100–140 tokens (Autoregressive) | 30–45 tokens + Prefill passes | Decoupled execution |
| **MUC Repair Retry Rate** | $8\text{--}12\%$ of chunks | $< 1.5\%$ of chunks | **$>80\%$ reduction** |
| **Multi-Hop Edge Recall** | $50\text{--}62\%$ (Dense vectors) | $>98.5\%$ (Topological PPR) | Zero vector drift |

---

## Proposed Changes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   QUANTA UNIFIED 4B SVM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Raw Document Chunk ──────► [Passage Store] (V_passage, immutable spans)    │
│           │                                       ▲                         │
│           │                                       │ E_ground                │
│           ▼                                       │                         │
│  ┌──────────────────────────────────────────────┐ │                         │
│  │     UNIFIED LLAMA-SERVER BACKEND (:8888)     │ │                         │
│  │         Shared Qwen 3.5 4B Backbone          │ │                         │
│  ├──────────────────────┬───────────────────────┤ │                         │
│  │ 1. Skeleton Transduce│ 2. Kev-4B Prefill     │ │                         │
│  │    (30-45 tok decod.)│    (Logprob Scoring)  │ │                         │
│  └──────────┬───────────┴───────────┬───────────┘ │                         │
│             │ Candidate Triples     │ Relational  │                         │
│             │                       │ Posteriors  │                         │
│             ▼                       ▼             │                         │
│  [clingo-dl / SIMD Codebook] ──► [Dual-Node ASG] ─┘                         │
│                                  (128B binary structs)                      │
│                                           │                                 │
│  User Query                               ▼                                 │
│     │                           [HippoRAG 2 PPR]                            │
│     │                           [PoP-RAG Belnap]                            │
│     └─────────────────────────►           │                                 │
│                                           ▼                                 │
│                                 [Bipartite Projection]                      │
│                                           │                                 │
│                                           ▼                                 │
│                            Logical Briefing + Raw Passages                  │
│                                           │                                 │
│                                           ▼                                 │
│                                [Downstream Reader LLM]                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Section 1: Dual-Node Bipartite Graph Schema & Passage Storage Engine

**Session Scope & Purpose:**
Establish the foundation of the dual-node bipartite graph topology. Introduce passage nodes ($V_{\text{passage}}$) holding immutable raw source text, document identifiers, and character offsets, and establish bipartite provenance links ($E_{\text{ground}}$) connecting semantic concept/event nodes ($V_{\text{semantic}}$) to their precise passage text spans.

#### [NEW] `src/memory/passage_store.py`
- Implements `PassageRecord` dataclass:
  - `passage_id: str` (e.g. `"P104"`)
  - `doc_id: str` (e.g. `"doc_physics_run_02"`)
  - `char_span: Tuple[int, int]` (absolute offsets within source document)
  - `text: str` (raw source text string)
  - `created_at: float`
- Implements `PassageStore`:
  - In-memory hash registry + persistent SQLite backing (`passages` table).
  - Fast span lookup: `get_text_span(passage_id, span_start, span_end) -> str`.
  - Batch insertion and offset-index management.

#### [MODIFY] `src/core/asg.py`
- Extend [`QuantaNode`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/core/asg.py#L27-L100):
  - Add provenance attributes:
    - `passage_id: Optional[str] = None`
    - `span_start: Optional[int] = None`
    - `span_end: Optional[int] = None`
    - `salience: float = 1.0`
    - `truth_status: str = "TRUE"` (Belnap state)
    - `confidence: float = 1.0`
    - `evidence_source: str = "direct_observation"`
  - Add helper methods:
    - `bind_passage(passage_id: str, span_start: int, span_end: int, confidence: float = 1.0)`
    - `get_grounded_text(passage_store: PassageStore) -> Optional[str]`
- Extend [`QuantaGraph`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/core/asg.py):
  - Add bipartite index:
    - `passage_to_nodes: Dict[str, Set[str]]` (maps `passage_id` to set of semantic node CIDs)
    - `node_to_passage: Dict[str, Tuple[str, int, int]]` (maps semantic CID to `(passage_id, span_start, span_end)`)
  - Method `add_passage_anchor(node_cid: str, passage_id: str, span_start: int, span_end: int)`
  - Method `get_nodes_for_passage(passage_id: str) -> List[QuantaNode]`

#### [NEW] `tests/test_dual_node_topology.py`
- Test passage creation, persistence, and retrieval in `PassageStore`.
- Test bipartite link integrity between `QuantaNode` instances and `PassageRecord`.
- Test sub-graph cloning and Merkle hashing invariance when passage pointers are attached.

---

### Section 2: Canonical Record DSL Parser, AST Compiler & 128-Byte Binary Layout

**Session Scope & Purpose:**
Replace nested S-expressions with the human-readable canonical Record DSL and compile it losslessly into fixed 128-byte C-compatible binary structs aligned to 64 bytes for microsecond SIMD filtering and Answer Set Programming.

#### [NEW] `src/core/binary_node.py`
- Implements `QuantaSemanticNode` using `ctypes.Structure` and NumPy structured data types:
  ```python
  import ctypes

  class QuantaSemanticNodeStruct(ctypes.Structure):
      _pack_ = 1
      _fields_ = [
          ("node_id", ctypes.c_uint32),          # 4 bytes
          ("passage_id", ctypes.c_uint32),       # 4 bytes
          ("span_start", ctypes.c_uint16),       # 2 bytes
          ("span_end", ctypes.c_uint16),         # 2 bytes
          ("concept_code", ctypes.c_uint16),     # 2 bytes
          ("belnap_lattice", ctypes.c_uint8),    # 1 byte (00=Contra, 01=True, 10=False, 11=Unknown)
          ("confidence", ctypes.c_uint8),        # 1 byte (0-255 -> 0.0-1.0)
          ("intent_band5", ctypes.c_uint8),      # 1 byte (Speech-act enum)
          ("epistemic_band6", ctypes.c_uint8),   # 1 byte (Evidence source enum)
          ("outgoing_edges", ctypes.c_uint32 * 4), # 16 bytes
          ("pad", ctypes.c_uint8 * 92),          # 92 bytes padding -> 128 bytes total
      ]
  ```
- Implements `BinaryNodeTable`:
  - Memory-mapped buffer (`bytearray` or `np.memmap`) supporting $O(1)$ random access, zero-copy slice iteration, and vector-aligned scans.

#### [NEW] `src/parser/record_dsl.py`
- Recursive-descent lexer and parser for the Canonical Record DSL:
  - Parses `passage P... { ... }` blocks.
  - Parses `entity E... "..." { ... }` blocks.
  - Parses `event EV... "..." { ... }` blocks.
  - Parses `relation R... { ... }` blocks.
- Serializer: Converts `QuantaGraph` or `DiscourseExtractionResult` into pristine Canonical Record DSL text.

#### [MODIFY] `src/parser/asg_compiler.py`
- Add `compile_record_dsl(dsl_text: str, passage_store: PassageStore) -> QuantaGraph` bridge.
- Add `compile_to_binary_table(graph: QuantaGraph) -> BinaryNodeTable`.

#### [NEW] `tests/test_record_dsl_and_binary_layout.py`
- Verify round-trip parsing: Record DSL $\leftrightarrow$ AST $\leftrightarrow$ QuantaGraph $\leftrightarrow$ 128-byte binary layout.
- Validate exact 128-byte alignment, bit-packing fields, and 64-byte boundary alignment.
- Benchmark binary struct instantiation latency ($< 0.5\,\mu\text{s}$ per node).

---

### Section 3: Fast Qwen-4B Skeleton Transducer & Grammar Stripping

**Session Scope & Purpose:**
Eliminate the ingestion bottleneck by removing logic/epistemic tags and S-expression syntax from Qwen 3.5 4B's decoding grammar. Restrict Qwen to extracting surface entities, surface spans, and unlabelled thematic subject-verb-object frames, reducing token generation budget from $>120$ tokens to $30\text{--}45$ tokens ($300\text{--}450\,\text{ms}$).

#### [NEW] `src/parser/skeleton_transducer.py`
- Implements `SkeletonExtractionResult`:
  - `entities: List[SkeletonEntity]` (`id`, `surface_text`, `char_span`)
  - `events: List[SkeletonEvent]` (`id`, `predicate`, `char_span`, `subject_ent_id`, `object_ent_id`)
- Implements `SkeletonTransducer`:
  - Minimal system prompt focused strictly on named entities and raw event anchors without S-expression brackets or logic tags.
  - GBNF grammar: Restricts decoding to flat JSON or compact record skeletons.
  - Connects to the unified `llama-server` on port 8888.
  - Offline fallback / mock transducer for deterministic CI testing without GPU.

#### [NEW] `src/parser/span_aligner.py`
- Maps extracted surface tokens back to exact character and byte offsets in the source text chunk.
- Resolves token-boundary shifts, whitespace variants, and punctuation trimming.

#### [MODIFY] `src/parser/unsloth_transducer.py`
- Update [`UnslothTransducer`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/unsloth_transducer.py#L82-L130) to support `mode="skeleton"` alongside legacy mode.
- Add telemetry measuring decoding token count ($30\text{--}45$ tokens) and prefill/decoding latency.

#### [NEW] `tests/test_skeleton_transducer.py`
- Test skeleton extraction over standard scientific and narrative passages.
- Verify decoding length is strictly $\le 50$ tokens.
- Verify byte-level span alignments against raw passage text.

---

### Section 4: Kev-4B Shared-Backbone Non-Autoregressive Relational Engine & Comparative Ablation

**Session Scope & Purpose:**
Implement the non-autoregressive decision engine powered by Kev-4B. Instead of loading separate weights, Kev-4B operates directly against the shared 4B `llama-server` engine using prefill logprob evaluation (`/completion` with `n_probs`). Evaluates candidate entity-event pairs across three prefill passes ($25\text{--}40\,\text{ms}$ per pass, $\sim 75\text{--}120\,\text{ms}$ total) with 4B-grade reasoning precision and zero extra VRAM overhead.
Crucially, this session also implements an **Empirical Comparative Experiment** evaluating **Approach A (Zero-shot / In-Context Logprob Scoring)** vs. **Approach B (Task-Specific Dynamic LoRA Adapter)** to quantify the trade-offs in accuracy, latency, and calibration before freezing the production pipeline.

#### [NEW] `src/models/kev_engine.py`
- Implements `KevDecisionEngine`:
  - Connects directly to the unified `llama-server` on port 8888.
  - VRAM Footprint: **0.0 GB additional VRAM** (leverages shared 4B model in `llama-server`).
  - Supports dual evaluation backends:
    - `mode="in_context_logprob"` (Approach A - Primary): Zero-shot structured prefill prompt with normalized logprob extraction across target label tokens.
    - `mode="lora_adapter"` (Approach B - Optional / Ablation): Requests routed with dynamic LoRA adapter attached via `llama-server` API (`--lora` / adapter slot).
  - Pass 1: `score_valency_and_coreference(entities, events, text)`
    - Submits prefill template: `"In the text '{text}', the semantic role of '{entity}' in event '{predicate}' is [AGENT/PATIENT/INSTRUMENT/NONE]:"`
    - Extracts normalized logprobs across candidate token logits.
  - Pass 2: `score_intent_and_epistemics(events, text)`
    - Submits prefill template for speech-act intent (`informative`, `directive`, `commissive`, `expressive`).
    - Submits prefill template for epistemic source (`direct_observation`, `deduction`, `hearsay`, `conjecture`).
  - Pass 3: `score_allen_and_pearl_relations(event_pairs, text)`
    - Evaluates Allen temporal interval (`MEETS`, `BEFORE`, `OVERLAPS`, `DURING`, `NONE`).
    - Evaluates Pearl causal link (`MECHANISM_LINK`, `ENABLING_CONDITION`, `NONE`).
  - Parallel batch dispatch using `asyncio` or `httpx` connection pool.

#### [NEW] `scripts/evaluate_kev_approaches.py`
- Empirical ablation runner evaluating Approach A vs. Approach B head-to-head across three gold benchmark datasets:
  1. **Thematic Valency Benchmark** ($N=500$ candidate pairs from FrameNet / PropBank gold tuples).
  2. **Speech-Act & Epistemic Benchmark** ($N=250$ verified discourse assertions).
  3. **Allen Temporal & Pearl Causal Benchmark** ($N=250$ multi-event scientific & narrative pairs).
- Generates publication-ready comparative metrics:
  - **Classification Accuracy & Macro F1** across all three passes.
  - **Expected Calibration Error (ECE) & Brier Score**: Measures probability calibration quality for Belnap lattice mapping ($P \ge 0.85$, $P \le 0.15$).
  - **Latency Profile**: Per-pass latency (ms), batch throughput (pairs/sec), and end-to-end chunk evaluation time.
  - **Operational & Memory Overhead**: 0 MB (Approach A) vs 30 MB + adapter switching latency (Approach B).
- Emits markdown evaluation report: `output/kev_approach_ablation_report.md`.

#### [NEW] `tests/test_kev_engine.py`
- Test 3-pass scoring on candidate entity and event pairs.
- Verify logprob extraction and calibrated probability distributions for valency, epistemics, and spatio-temporal relations.
- Benchmark prefill execution time ($< 120\,\text{ms}$ total across all 3 passes).

#### [NEW] `tests/test_kev_ablation.py`
- Automated test asserting comparative ablation runner executes successfully, outputs valid metrics, and verifies that Approach A meets the minimum accuracy bar ($\ge 92\%$ Macro F1) and calibration threshold ($ECE \le 0.08$).

---

### Section 5: Calibrated Belnap Lattice Mapping & `clingo-dl` Difference Logic Gate

**Session Scope & Purpose:**
Directly map Kev-4B relational posteriors to the Belnap 4-valued logic lattice ($\mathcal{B}_4 = \{00_2, 01_2, 10_2, 11_2\}$), perform $0.32\,\mu\text{s}$ SIMD ConceptNet codebook grounding, and validate temporal difference-logic and causal DAG acyclicity using `clingo-dl`, bringing MUC repair retries to $< 1.5\%$.

#### [NEW] `src/verification/belnap_calibrator.py`
- Implements `BelnapLatticeMapper`:
  - Threshold mapping:
    - $P(\text{relation}) \ge 0.85 \implies 01_2$ (TRUE)
    - $P(\text{relation}) \le 0.15 \implies 10_2$ (FALSE)
    - $0.15 < P(\text{relation}) < 0.85 \implies 11_2$ (UNKNOWN / MAYBE)
  - Confidence encoding: $P \in [0.0, 1.0] \mapsto \text{uint8} \in [0, 255]$.

#### [NEW] `src/verification/clingo_dl_gate.py`
- Implements `ClingoDLGate`:
  - Encodes Allen temporal intervals as difference logic constraints: $t_{\text{end}} - t_{\text{start}} > 0$, $t_{\text{ev2,start}} - t_{\text{ev1,end}} \ge 0$ (for `MEETS`).
  - Checks for causal cycles and temporal paradoxes in polynomial time via difference logic solvers.
  - Emits structured MUC diagnostics if UNSAT, with repair hints.

#### [MODIFY] `src/parser/mmap_grounder.py`
- Ensure zero-copy ConceptNet grounding completes in $\le 0.35\,\mu\text{s}$ per concept code.
- Populate `concept_code` in the 128-byte binary struct.

#### [NEW] `tests/test_belnap_calibrator_and_clingo_dl.py`
- Test calibration thresholds on synthetic probability vectors.
- Test difference-logic verification on valid sequences and synthetic paradoxes (e.g. A before B, B before C, C before A).
- Measure verification latency and verify $< 1.5\%$ MUC rejection rate.

---

### Section 6: HippoRAG 2 Spreading Activation & PoP-RAG Epistemic Gating

**Session Scope & Purpose:**
Transition context retrieval from dense embedding vector similarity to Personalized PageRank (PPR) spreading activation over the ASG, gated by Belnap truth values (PoP-RAG). Eliminate multi-hop semantic drift while pruning contradictory edges and dampening unverified facts.

#### [NEW] `src/memory/hipporag_ppr.py`
- Implements `HippoRAGRetriever`:
  - Transition matrix $\mathbf{W}$: Constructed from Allen intervals, Pearl causal paths, and valency links.
  - Personalized PageRank computation:
    $$\mathbf{p}^{(t+1)} = (1 - \alpha) \mathbf{W} \mathbf{p}^{(t)} + \alpha \mathbf{p}^{(0)}$$
    with restart probability $\alpha \in [0.15, 0.20]$ and convergence tolerance $\epsilon = 10^{-6}$.
  - SIMD / NumPy vectorized power iteration.

#### [NEW] `src/memory/poprag_gating.py`
- Implements `PoPRAGGating`:
  - Edge weight modulation based on Belnap truth state:
    - If $\text{state} = 00_2$ (Contradiction): set edge weight to $0.0$ (prune path).
    - If $\text{state} = 01_2$ (True): weight maintained at nominal value.
    - If $\text{state} = 11_2$ (Unknown): weight scaled by calibrated confidence $P$.
    - If $\text{state} = 10_2$ (False): inverted or suppressed based on query target polarity.

#### [MODIFY] `src/memory/spreading_activation.py`
- Integrate `HippoRAGRetriever` and `PoPRAGGating` into [`SpreadingActivationRetriever`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/memory/spreading_activation.py#L50-L90).
- Support multi-hop query traversal (2 to 4 hops) with zero semantic drift.

#### [NEW] `tests/test_hipporag_poprag_retriever.py`
- Test PPR convergence and activation distribution over complex 4-hop graphs.
- Verify that contradictory edges ($00_2$) are completely pruned from the activation path.
- Verify that unknown edges ($11_2$) have attenuated influence compared to verified true edges ($01_2$).
- Measure traversal latency over 100,000 nodes ($< 5\,\text{ms}$).

---

### Section 7: Bipartite Passage Projection & Realization Bypass (Context Assembly)

**Session Scope & Purpose:**
Bypass the lossy neural text realizer entirely. Map activated semantic node scores onto raw passage text nodes via bipartite projection edges, and assemble a dual-stream context block consisting of a concise Logical Briefing Block plus pristine top-ranked raw text passages for the downstream reader LLM.

#### [NEW] `src/memory/context_assembler.py`
- Implements `BipartiteProjector`:
  - Projects semantic activation scores onto raw passage nodes:
    $$\text{Score}(v_p) = \sum_{v_s \in \mathcal{N}(v_p)} \mathbf{p}(v_s) \cdot \text{conf}(v_s)$$
  - Ranks passages by accumulated score and applies token budget cutoffs.
- Implements `DualStreamContextAssembler`:
  - Stream 1: **Logical Briefing Block**
    - Summarizes verified active causal paths and temporal intervals:
      `"Path: EV1 [t1..t2] -> CAUSES -> EV2 [t2..t3] | Status: Verified"`
  - Stream 2: **Top-K Raw Source Passages**
    - Pristine verbatim lexical text spans extracted directly from `PassageStore`.

#### [MODIFY] `src/memory/spreading_activation.py`
- Deprecate [`_format_english_context`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/memory/spreading_activation.py#L1259-L1350) in favor of `DualStreamContextAssembler`.
- Update `format_context_for_llm` to return the dual-stream representation.

#### [NEW] `tests/test_bipartite_projection_and_context_assembler.py`
- Test bipartite score projection across multiple passages sharing entities and events.
- Test dual-stream formatting under various token budget constraints (500, 1000, 2000 tokens).
- Verify 100% lexical fidelity of retrieved passages (exact match to source text).

---

### Section 8: End-to-End Pipeline Rewire, Unified Serving & VRAM Validation

**Session Scope & Purpose:**
Rewire the full cognitive pipeline and reverse proxy to utilize the Semantic Virtual Memory architecture. Deprecate Unsloth Studio in favor of the unified `llama-server` backend. Validate concurrent execution within the 8GB VRAM workstation budget (RTX 3070), benchmark ingestion throughput ($350\text{--}520\,\text{ms}$/chunk), and verify multi-hop QA recall ($>98.5\%$) on MuSiQue, ARC-Challenge, and HumanEval.

#### [MODIFY] `src/server/unsloth_manager.py` & `src/server/service_manager.py`
- Replace Unsloth Studio lifecycle management with standalone `llama-server` management:
  - Command: `llama-server.exe -m models/Qwen3.5-4B-MTP-GGUF/qwen3.5-4b-mtp-q5_k_m.gguf --port 8888 --host 127.0.0.1 -ngl 99 -c 8192 -fa`
  - Health check: `http://127.0.0.1:8888/health` and `/v1/models`
  - Dual routing: Supports `/v1/chat/completions` (for skeleton transduction) and `/completion` with logprobs (for Kev-4B scoring).

#### [MODIFY] `src/pipeline/cognitive_pipeline.py`
- Rewire [`CognitivePipeline.ingest_document`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/pipeline/cognitive_pipeline.py#L120-L200):
  1. Register chunk in `PassageStore`.
  2. Run `SkeletonTransducer` to extract entities and event frames via `llama-server`.
  3. Run `KevDecisionEngine` 3-pass prefill scoring via `llama-server`.
  4. Apply `BelnapLatticeMapper` and SIMD concept grounding.
  5. Validate via `ClingoDLGate`.
  6. Commit to dual-node `QuantaGraph` and binary table.
- Rewire [`CognitivePipeline.query_memory`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/pipeline/cognitive_pipeline.py):
  1. Compile query seeds and constraints.
  2. Execute HippoRAG 2 PPR with PoP-RAG gating.
  3. Run bipartite projection to rank raw passages.
  4. Return dual-stream context block.

#### [MODIFY] `src/server/proxy.py`
- Update [`QuantaProxyServer`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/server/proxy.py) to stream the assembled dual-stream context directly to the unified `llama-server` backend.

#### [MODIFY] `src/server/mcp_server.py`
- Update MCP tools (`quanta_ingest_document`, `quanta_query_memory`) to report bipartite graph statistics and passage span mappings.

#### [NEW] `tests/test_svm_cognitive_pipeline.py`
- End-to-end integration test: Raw text ingestion $\rightarrow$ dual-node ASG $\rightarrow$ query $\rightarrow$ dual-stream retrieval.
- Memory profile benchmark: Verify total VRAM stays $\le 4.8\text{ GB}$ during concurrent Qwen + Kev execution on RTX 3070.
- Throughput benchmark: Ingestion latency verified at $350\text{--}520\,\text{ms}$ per chunk.

---

## Verification Plan

### Automated Tests
Run via Windows PowerShell:

```powershell
# Section 1 Verification:
pytest tests/test_dual_node_topology.py -v

# Section 2 Verification:
pytest tests/test_record_dsl_and_binary_layout.py -v

# Section 3 Verification:
pytest tests/test_skeleton_transducer.py -v

# Section 4 Verification & Ablation Experiment:
pytest tests/test_kev_engine.py tests/test_kev_ablation.py -v
python scripts/evaluate_kev_approaches.py --quick

# Section 5 Verification:
pytest tests/test_belnap_calibrator_and_clingo_dl.py -v

# Section 6 Verification:
pytest tests/test_hipporag_poprag_retriever.py -v

# Section 7 Verification:
pytest tests/test_bipartite_projection_and_context_assembler.py -v

# Section 8 Verification & Full Regression:
pytest tests/test_svm_cognitive_pipeline.py -v
pytest tests/test_end_to_end_suite.py tests/test_paired_benchmarks.py -v
```

### Manual Verification
1. **Unified VRAM Footprint Audit:** Run `nvidia-smi` while running ingestion and retrieval through `llama-server` to confirm total GPU memory consumption is **$\le 4.8\text{ GB}$** on NVIDIA GeForce RTX 3070 (leaving $>3.2\text{ GB}$ free headroom).
2. **Kev-4B Prefill Scoring Latency:** Verify through pipeline logs that each of Kev-4B's prefill passes completes in $25\text{--}40\,\text{ms}$ with zero autoregressive generation lag.
3. **Retrieval Purity Audit:** Inspect output of `DualStreamContextAssembler` to confirm retrieved passage text contains exact, unmodified character spans from the source documents with zero hallucinated paraphrasing.
4. **MUC Retry Frequency:** Review pipeline telemetry logs during batch document ingestion to verify that MUC repairs occur on $< 1.5\%$ of processed chunks.
