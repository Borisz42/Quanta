# Master Plan: EmbeddingGemma 2 as an Auxiliary Neural Vector Coprocessor (ANVC) in QUANTA

## Goal Description & Empirical Rationale

QUANTA's neuro-symbolic architecture relies on discrete 1024-dimension quaternary spaces ($\Sigma = \{0, 1, 2, 3\}^{1024}$), BLAKE3 Merkle folding, sub-microsecond memory-mapped concept grounding, and ASP / Difference-Logic verification gates (`clingo-dl`). These symbolic primitives guarantee tamper-evidence, $0.000000\%$ hallucination on transactional knowledge, and instant $O(1)$ lookup performance.

However, three specific bottlenecks persist in the current pipeline due to reliance on strict lexical and discrete representations:
1. **Vocabulary-Mismatch Ingestion Dropouts (Stage 12.8)**: Platt-calibrated BM25 (`F-A`) reduces irrelevant tokens by 60.2%, but strict token overlap causes it to discard gold evidence when queries use synonyms, abstract paraphrasing, or figurative terminology (Gold evidence recall caps at $80.7\%$).
2. **Out-of-Vocabulary (OOV) Semantic Starvation**: When an extracted entity is absent from `data/concept_codebook.bin` (403k entries) and Damerau-Levenshtein normalization misses, ConceptNet taxonomic slots (Band 3 [384..511] and Band 4 [512..639]) default to inactive zeros ($00_2$), losing domain context.
3. **Spreading Activation Seed Initialization**: The `SpreadingActivationRetriever` relies on discrete Hamming distance matching or exact keyword overlaps to select initial graph seed nodes ($E_0$). On conversational or indirect questions (*"What prevented the spacecraft from overheating?"* vs *`SUNSHIELD`*), discrete Hamming distance frequently misses entry nodes.

### The Objective: Auxiliary Neural Vector Coprocessor (ANVC)

Integrate **EmbeddingGemma 2** (`google/embeddinggemma-2`, non-gated 740M multimodal model with 270M text slice) as an **Auxiliary Neural Vector Coprocessor (ANVC)**.

The ANVC operates as a continuous geometric advisor to the discrete symbolic core:
- **Neural Advisory Role**: Provides continuous dense similarities for candidate filtering, continuous archetype projections for OOV grounding, and smooth energy distributions for spreading activation seeding.
- **Preserved Symbolic Invariants**: All downstream truth values remain strictly quantized into Belnap FOUR ($\mathcal{B}_4 = \{00_2, 01_2, 10_2, 11_2\}$); all knowledge graph nodes remain content-addressed via 256-bit BLAKE3 hashes; and all causal/temporal consistency checks continue to pass through `clingo-dl` difference logic gates.
- **Decoupled Architecture & Zero VRAM Collision**:
  - The primary generative LLM (`unsloth/Qwen3.5-4B-MTP-GGUF`) runs on GPU on `:8888`, consuming $\approx 7.3\text{ GB} / 8.0\text{ GB}$ VRAM on the RTX 3070.
  - EmbeddingGemma 2 is **strictly decoupled** from Unsloth. It runs in-process on CPU (`device="cpu"`) via SentenceTransformers, or on a dedicated standalone `llama-server` process on `:8889`.
  - On CPU, EmbeddingGemma 2 consumes **$0\text{ MB}$ GPU VRAM** and $\approx 1.4\text{ GB}$ system RAM, eliminating OOM risk entirely while delivering $\approx 100\text{--}150\text{ ms}$ inference latencies.
- **Strict Live Guard Policy**: No mock pseudo-random representations are allowed for research calibration, evaluation sweeps, or production runtime. An explicit live backend assertion (`assert_live_backend()`) halts execution with `RuntimeError` if a live EmbeddingGemma 2 engine is not online.

```mermaid
flowchart TD
    subgraph INGESTION["Input Stream & Task Extraction"]
        Q[User Query / Document Chunk] --> TBE[TaskBoundaryExtractor]
        TBE --> EMB_MGR[EmbeddingManager (In-Process CPU / Port :8889)<br/>Real google/embeddinggemma-2 (Configurable MRL Dim)]
    end

    subgraph DENSE_ADVISORY["Auxiliary Neural Vector Coprocessor (ANVC)"]
        EMB_MGR --> RRF_FILTER[Hybrid BM25 + Dense RRF Filter<br/>Calibrated Weights & Thresholds]
        EMB_MGR --> OOV_PROJ[Archetype Projection SIMD Engine<br/>archetype_embeddings.bin (Configurable Dim)]
        EMB_MGR --> SEED_ENERGY[Continuous Energy Seeding<br/>e_0 Dense Seed Vector (Calibrated Alpha)]
    end

    subgraph SYMBOLIC_CORE["QUANTA Discrete Neuro-Symbolic Core"]
        RRF_FILTER --> CHUNK[Hierarchical Discourse Chunker]
        OOV_PROJ --> MMAP[MmapLexicalGrounder<br/>Band 3/4 ConceptNet Slots]
        MMAP --> ASG[ASG Compiler & Binary Node Table]
        SEED_ENERGY --> SPREAD[SpreadingActivationRetriever & HippoRAG PPR]
        ASG --> BELNAP[Belnap Truth Lattice & clingo-dl Gates]
    end

    subgraph PROJECTION["Downstream Generation & Verification"]
        SPREAD --> DUAL[DualStreamContextAssembler]
        BELNAP --> DUAL
        DUAL --> LLM[Qwen 3.5 4B MTP (:8888)]
    end
```

---

## Zero-Hardcoded-Numbers Architecture: The Calibration-First Paradigm

> [!IMPORTANT]
> **Core Architectural Rule: No Hardcoded Magic Numbers**
> Every dimension, threshold, fusion weight, smoothing constant, seed count, and token budget is **strictly configurable** and governed by QUANTA's Calibrated Parameter Registry (`src/config/multi_scale_config.py` + `config/multi_scale_profile.json`).
>
> No values are assumed *a priori*. Each parameter is calibrated through dedicated sweep experiments across gold benchmark datasets (MuSiQue, HumanEval, BABILong, NIAH), selecting the winning configuration via Pareto-optimal frontiers (Accuracy vs. Latency vs. Memory Bandwidth), and writing calibrated profiles with explicit `source_exp` and `source_commit` metadata.

### Configuration Precedence Order
1. **Pass-through Defaults**: Uncalibrated fallback baseline (feature disabled or non-intrusive control).
2. **Profile Configuration**: `config/multi_scale_profile.json` (written strictly by automated sweep and calibration scripts).
3. **Environment Variables**: `QUANTA_EMBEDDING_*`, `QUANTA_FILTER_*`, `QUANTA_GROUNDER_*`, `QUANTA_SPREADING_*`.
4. **Per-Request Headers / Runtime Overrides**: Direct runtime overrides for dynamic ablation testing.

---

## Systematic Parameter Registry & Sweep Matrix

| Parameter Key | Description | Uncalibrated Default | Empirical Sweep Space | Dedicated Sweep Experiment |
|---|---|---|---|---|
| `embedding.matryoshka_dim` | Truncation dimension for MRL embeddings | `768` (Full Native) | $D \in \{64, 128, 256, 512, 768\}$ | `scripts/sweep_matryoshka_dimensions.py` |
| `embedding.batch_size` | Inference batch size for document chunking | `16` | $B \in \{4, 8, 16, 32, 64\}$ | `tests/test_embedding_manager.py` |
| `filter.strategy` | Relevance filter scoring strategy | `"passthrough"` | `{"F-A", "F-B", "F-F", "F-G"}` | `scripts/evaluate_relevance_filter.py` |
| `filter.rrf_k` | Reciprocal Rank Fusion smoothing constant $k$ | `60` | $k \in \{10, 20, 30, 60, 100\}$ | `scripts/calibrate_dense_filter.py` |
| `filter.weight_lexical` | RRF lexical BM25 component weight $w_{\text{lex}}$ | `0.5` | $w_{\text{lex}} \in [0.0, 1.0]$ (step 0.1) | `scripts/calibrate_dense_filter.py` |
| `filter.weight_dense` | RRF dense bi-encoder component weight $w_{\text{dense}}$ | `0.5` | $w_{\text{dense}} \in [0.0, 1.0]$ (step 0.1) | `scripts/calibrate_dense_filter.py` |
| `filter.dense_fast_path_threshold` | Dense similarity threshold $\theta_{\text{fast}}$ to bypass BM25 | `None` (Disabled) | $\theta_{\text{fast}} \in [0.70, 0.95]$ (step 0.05) | `scripts/calibrate_dense_filter.py` |
| `filter.keep_budget_tokens` | Token budget cap for kept units | `1500` | Budget $\in \{800, 1200, 1500, 2000, 3000\}$ | `scripts/calibrate_dense_filter.py` |
| `grounder.archetype_top_k` | Top-$k$ nearest archetype dimensions activated | `2` | $k \in \{1, 2, 3, 4, 5\}$ | `scripts/sweep_archetype_grounding.py` |
| `grounder.archetype_threshold` | Similarity threshold $\theta_{\text{arch}}$ for OOV slot activation | `0.70` | $\theta_{\text{arch}} \in [0.50, 0.85]$ (step 0.05) | `scripts/sweep_archetype_grounding.py` |
| `grounder.archetype_belnap_value` | Belnap truth state assigned to activated OOV slots | `3` (`UNKNOWN`) | $\{1 \text{ (TRUE)}, 3 \text{ (UNKNOWN)}, \text{"confidence_scaled"}\}$ | `scripts/sweep_archetype_grounding.py` |
| `spreading.dense_seed_threshold` | Dense cosine threshold $\theta_{\text{seed}}$ for seed injection | `0.65` | $\theta_{\text{seed}} \in [0.45, 0.85]$ (step 0.05) | `scripts/sweep_spreading_energy_seeding.py` |
| `spreading.hybrid_blend_alpha` | Blend ratio $\alpha$ (Dense vs Hamming seed energy) | `0.5` | $\alpha \in [0.0, 1.0]$ (step 0.1) | `scripts/sweep_spreading_energy_seeding.py` |
| `spreading.max_seed_nodes` | Maximum initial seed nodes $K_{\text{seed}}$ | `5` | $K_{\text{seed}} \in \{3, 5, 8, 12, 16\}$ | `scripts/sweep_spreading_energy_seeding.py` |
| `ast.code_similarity_threshold` | Similarity threshold $\theta_{\text{code}}$ for AST function search | `0.70` | $\theta_{\text{code}} \in [0.50, 0.85]$ (step 0.05) | `scripts/sweep_code_grounding.py` |
| `scene.visual_archetype_threshold`| Similarity threshold $\theta_{\text{vis}}$ for visual crop mapping | `0.65` | $\theta_{\text{vis}} \in [0.50, 0.85]$ (step 0.05) | `scripts/sweep_code_grounding.py` |

---

## 6-Session Implementation & Calibration Roadmap

---

### Session 1: Dedicated Local Embedding Engine & Matryoshka Dimension Sweep

**Goal**: Implement a lightweight, non-blocking local serving manager for EmbeddingGemma 2, establish configuration parameters in `MultiScaleConfig`, and execute a rigorous empirical sweep across Matryoshka truncation sizes ($D \in \{64, 128, 256, 512, 768\}$) to evaluate accuracy vs. memory bandwidth vs. latency.

#### Target Files
- `src/server/embedding_manager.py` [NEW]: Dedicated lifecycle and client manager for `google/embeddinggemma-2`. Supports dual backend (dedicated `llama-server` on `:8889` or in-process `sentence_transformers`/`transformers` on CPU/GPU), strict live guard (`assert_live_backend()`), and isolated `MockEmbeddingManager` for unit test doubles only.
- `src/config/multi_scale_config.py` [MODIFY]: Register embedding configuration parameters (`embedding.model_id` defaulting to `google/embeddinggemma-2`, `embedding.port`, `embedding.backend`, `embedding.matryoshka_dim`, `embedding.device` defaulting to `cpu`, `embedding.batch_size`).
- `src/server/service_manager.py` [MODIFY]: Integrate `EmbeddingManager` into unified startup (`--all`), health auditing, and process shutdown.
- `scripts/sweep_matryoshka_dimensions.py` [NEW]: Empirical dimension sweep harness evaluating $D \in \{64, 128, 256, 512, 768\}$ on representation entropy, retrieval accuracy (MRR / Recall@K), SIMD dot-product latency, and memory footprint. Strictly runs with live `google/embeddinggemma-2` by default.
- `tests/test_embedding_manager.py` [NEW]: Unit tests for embedding generation, task prefix formatting, arbitrary Matryoshka truncation, L2 normalization, and mock test double isolation.

#### Technical Specification
1. **Dynamic Matryoshka Representation Learning (MRL)**:
   - Given raw native 768-D vector $\mathbf{v} \in \mathbb{R}^{768}$.
   - Truncate dynamically to configured dimension $D \in [1, 768]$: $\mathbf{v}_D = \mathbf{v}[:D]$.
   - Apply $L_2$ normalization:
     $$\hat{\mathbf{v}}_D = \frac{\mathbf{v}_D}{\|\mathbf{v}_D\|_2}$$
2. **Dimension Sweep Design (`scripts/sweep_matryoshka_dimensions.py`)**:
   - Evaluates:
     - **Information Retention**: Cosine fidelity to full 768-D representation: $\rho(D) = \frac{1}{N} \sum_{i=1}^N (\hat{\mathbf{v}}_D^{(i)} \cdot \hat{\mathbf{v}}_{768}^{(i)})$.
     - **Retrieval Accuracy**: Gold passage retrieval Recall@5 and MRR on MuSiQue and NIAH samples.
     - **SIMD Dot-Product Latency**: Time per 10,000 vector similarity computations in host RAM.
     - **Memory Footprint**: RAM / VRAM per 100,000 vectors ($D \times 4\text{ bytes} \times 10^5$).
   - Generates publication curve and Pareto frontier: Accuracy vs. Microsecond Latency.
   - Automatically writes the winning Pareto dimension to `config/multi_scale_profile.json` under `embedding.matryoshka_dim`.
3. **Strict Live Guard Policy**:
   - `EmbeddingManager(allow_mock=False)` asserts the active engine is genuine `google/embeddinggemma-2`.
   - Silent fallbacks to pseudo-random mocks are strictly rejected and trigger a `RuntimeError`.
4. **Task Prefix Formatters**:
   - Asymmetric Query: `task: question answering | query: {query}` or `task: search result | query: {query}`
   - Asymmetric Document: `title: {title} | text: {chunk_text}` (default `title: none`)
   - Code Search: `task: code retrieval | query: {query}`

#### Verification & Test Suite
```powershell
pytest tests/test_embedding_manager.py tests/test_service_manager.py
python scripts/sweep_matryoshka_dimensions.py --samples 50 --mode live --export-report
```
- Assert single-query latency $\le 200\text{ ms}$ on CPU ($0\text{ MB}$ GPU VRAM impact).
- Assert batched throughput $\ge 1,000\text{ tokens/sec}$ on CPU batched evaluation.
- Confirm GPU VRAM for EmbeddingGemma is $0\text{ MB}$ (CPU execution) preserving all $8\text{ GB}$ for Unsloth/Qwen on `:8888`.
- Verify dimension sweep generates `output/matryoshka_dimension_sweep.json` and report using live embeddings.

#### Example Starting Prompt for Session 1
```text
Execute Session 1 of the ANVC Master Plan (quanta_embeddinggemma_anvc_master_plan.md).
Goal: Implement the Dedicated Local Embedding Engine & Matryoshka Dimension Sweep.
1. Create `src/server/embedding_manager.py` implementing `EmbeddingManager`:
   - Real model `google/embeddinggemma-2` executing on CPU or dedicated `:8889` llama-server.
   - Strict live guard enforcing `assert_live_backend()` and `allow_mock=False`.
   - Enforce task prefix formatters (asymmetric query, document, code).
   - Support arbitrary Matryoshka truncation D in [64..768] with L2 normalization (retrieving D from MultiScaleConfig).
2. Update `src/config/multi_scale_config.py` to register all embedding parameters with CPU default.
3. Update `src/server/service_manager.py` to orchestrate the embedding manager lifecycle.
4. Create `scripts/sweep_matryoshka_dimensions.py` to benchmark D in {64, 128, 256, 512, 768} with live backend.
5. Create `tests/test_embedding_manager.py` verifying vector shapes, L2 normalization, prefix injection, and guard enforcement.
Verify with `pytest tests/test_embedding_manager.py` and `python scripts/sweep_matryoshka_dimensions.py --mode live`.
```

---

### Session 2: Hybrid BM25 + Dense Bi-Encoder Relevance Filter & Fusion Calibration

**Goal**: Upgrade the multi-scale ingestion `RelevanceFilter` to eliminate vocabulary-mismatch dropouts on paraphrased or abstract queries without increasing synchronous SLM transduction calls, using calibrated Reciprocal Rank Fusion (RRF) and dynamic fast-path routing.

#### Target Files
- `src/retrieval/relevance_filter.py` [MODIFY]: Add Variant `F-G` (`dense_hybrid`) combining BM25 lexical scores with EmbeddingGemma 2 dense cosine similarities via configurable RRF.
- `src/parser/multi_scale_chunker.py` [MODIFY]: Ensure `MacroBlock` and `DiscourseChunk` store or cache dense embeddings on ingestion with configurable dimensionality.
- `src/memory/fast_path_assembler.py` [MODIFY]: Connect dense score hot-path: if $\text{score}_{\text{dense}} \ge \theta_{\text{fast}}$, bypass BM25 and immediately route to fast-path assembly.
- `scripts/calibrate_dense_filter.py` [NEW]: Empirical grid search and calibration harness sweeping $w_{\text{lex}}, w_{\text{dense}}, k, \theta_{\text{fast}}$, and token budgets.
- `tests/test_relevance_filter.py` [MODIFY]: Add unit tests for `F-G` dense hybrid filtering, parameter overrides, synonym resilience, and RRF rank merging.

#### Technical Specification
1. **Reciprocal Rank Fusion (RRF) Formulation**:
   - For candidate unit $d \in \mathcal{D}$:
     $$RRF(d) = \frac{w_{\text{lex}}}{k_{\text{rrf}} + \text{rank}_{\text{BM25}}(d)} + \frac{w_{\text{dense}}}{k_{\text{rrf}} + \text{rank}_{\text{dense}}(d)}$$
   - All components are parameterized:
     - $k_{\text{rrf}} = \text{cfg.get("filter.rrf_k", 60)}$
     - $w_{\text{lex}} = \text{cfg.get("filter.weight_lexical", 0.5)}$
     - $w_{\text{dense}} = \text{cfg.get("filter.weight_dense", 0.5)}$
2. **Fast-Path Dynamic Coupling**:
   - If $\max_d s_{\text{dense}}(d) \ge \theta_{\text{fast}}$ where $\theta_{\text{fast}} = \text{cfg.get("filter.dense_fast_path_threshold")}$:
     - Bypass lexical BM25 ranking and immediately route the top-ranked unit to `FastPathAssembler`.
3. **Calibration Script (`scripts/calibrate_dense_filter.py`)**:
   - Evaluates a 2D parameter grid:
     - $w_{\text{dense}} \in \{0.2, 0.4, 0.5, 0.6, 0.8\}$ (with $w_{\text{lex}} = 1.0 - w_{\text{dense}}$)
     - $k_{\text{rrf}} \in \{10, 20, 30, 60, 100\}$
     - $\theta_{\text{fast}} \in \{0.75, 0.80, 0.82, 0.85, 0.90\}$
   - Objective: Maximize Gold Evidence Recall on MuSiQue synonym distractors while keeping token footprint within budget.
   - Writes optimal configuration to `config/multi_scale_profile.json`.

#### Verification & Test Suite
```powershell
pytest tests/test_relevance_filter.py
python scripts/calibrate_dense_filter.py --samples 30
```
- Verify Gold Evidence Recall increases from $80.7\%$ to $\ge 92.0\%$ on paraphrase distractors.
- Confirm filter execution latency remains $< 10.0\text{ ms}$ per query.

#### Example Starting Prompt for Session 2
```text
Execute Session 2 of the ANVC Master Plan (quanta_embeddinggemma_anvc_master_plan.md).
Goal: Implement the Hybrid BM25 + Dense Bi-Encoder Relevance Filter & Fusion Calibration.
1. Modify `src/retrieval/relevance_filter.py` to add strategy `F-G` (`dense_hybrid`):
   - Ingest `filter.rrf_k`, `filter.weight_lexical`, `filter.weight_dense`, and `filter.dense_fast_path_threshold` from MultiScaleConfig.
   - Calculate cosine similarity of query against candidate macro-blocks/chunks.
   - Implement Reciprocal Rank Fusion (RRF) combining BM25 and dense ranks using configured weights.
   - Hot-route to fast-path when dense similarity >= theta_fast.
2. Update `src/parser/multi_scale_chunker.py` so `MacroBlock` can cache its dense embedding.
3. Wire fast-path trigger into `src/memory/fast_path_assembler.py`.
4. Create `scripts/calibrate_dense_filter.py` sweeping weights, k_rrf, and theta_fast to find the optimal fusion configuration and persist it to `config/multi_scale_profile.json`.
5. Extend `tests/test_relevance_filter.py` with test cases testing synonym queries and configurable weight behavior.
Verify with `pytest tests/test_relevance_filter.py`.
```

---

### Session 3: Out-of-Vocabulary (OOV) Grounding via Dynamic Archetype Projection

**Goal**: Replace string-only Levenshtein fallbacks with continuous semantic projection when grounding unseen terms into ConceptNet Band 3 and Band 4 taxonomic slots, calibrating top-$k$, threshold, and Belnap truth values.

#### Target Files
- `scripts/precompute_archetype_embeddings.py` [NEW]: Offline compiler that embeds all 256 ConceptNet slot descriptions from `data/conceptnet_slots.json` using EmbeddingGemma 2 into `data/archetype_embeddings.bin` at the calibrated dimension $D$.
- `scripts/sweep_archetype_grounding.py` [NEW]: Sweep harness evaluating top-$k \in \{1, 2, 3, 5\}$, $\theta_{\text{arch}} \in [0.50, 0.85]$, and Belnap truth assignment modes on OOV neologisms and rare entities.
- `src/parser/mmap_grounder.py` [MODIFY]: Integrate memory-mapped archetype projection matrix; if dictionary lookup fails, project entity embedding onto archetype dimensions via SIMD using calibrated parameters.
- `src/parser/asg_compiler.py` [MODIFY]: Assign projected Belnap truth values into QuantaVector Band 3 & Band 4 slots based on configuration.
- `tests/test_mmap_grounder_speed.py` [MODIFY]: Add benchmarks and tests verifying OOV term grounding without degrading $< 0.05\text{ ms}$ lookup speed.

#### Technical Specification
1. **Dynamic Archetype Pre-Computation**:
   - Read 256 slots from `data/conceptnet_slots.json` (Slot 384 `CN_Q001_COMPUTING` through Slot 639 `CN_Q256_IMAGE`).
   - Format: `title: {name} | text: {description}`.
   - Generate normalized $D$-dimensional vectors where $D$ is read from `MultiScaleConfig` (`embedding.matryoshka_dim`).
   - Export contiguous binary array `data/archetype_embeddings.bin` ($256 \times D \times 4\text{ bytes}$).
2. **Archetype Sweep Harness (`scripts/sweep_archetype_grounding.py`)**:
   - Tests synthetic neologisms (e.g. *"cryo-containment manifold"*, *"photonic lattice accelerator"*, *"quantum flux gate"*).
   - Sweeps:
     - $k \in \{1, 2, 3, 5\}$
     - $\theta_{\text{arch}} \in \{0.55, 0.60, 0.65, 0.70, 0.75, 0.80\}$
     - Belnap mapping mode:
       - Fixed `UNKNOWN` ($11_2 = 3$)
       - Fixed `TRUE` ($01_2 = 1$)
       - Similarity-scaled: If $\cos \ge 0.85 \implies \text{TRUE}$, else if $\cos \ge \theta_{\text{arch}} \implies \text{UNKNOWN}$
   - Measures semantic category precision and downstream classification accuracy.
   - Writes calibrated parameters (`grounder.archetype_top_k`, `grounder.archetype_threshold`, `grounder.archetype_belnap_value`) to `config/multi_scale_profile.json`.
3. **Runtime SIMD Projection**:
   - When `MmapLexicalGrounder._lookup_row(term)` returns `None`:
     1. Obtain $D$-dimensional query vector $\mathbf{v}_{\text{term}}$.
     2. Compute projection $\mathbf{p} = \mathbf{A} \cdot \mathbf{v}_{\text{term}}$ via AVX2/AVX-512 BLAS matrix-vector product ($< 0.05\text{ ms}$).
     3. Select top-$k$ entries exceeding $\theta_{\text{arch}}$ and populate Band 3/4 slots with calibrated Belnap values.

#### Verification & Test Suite
```powershell
python scripts/precompute_archetype_embeddings.py
python scripts/sweep_archetype_grounding.py
pytest tests/test_mmap_grounder_speed.py
```
- Verify synthetic neologisms activate appropriate slots (e.g. `CN_Q001_COMPUTING`, `CN_Q156_PHYSICS`).
- Verify lookup latency remains $< 0.05\text{ ms}$ on host memory.

#### Example Starting Prompt for Session 3
```text
Execute Session 3 of the ANVC Master Plan (quanta_embeddinggemma_anvc_master_plan.md).
Goal: Implement Out-of-Vocabulary (OOV) Grounding via Dynamic Archetype Projection.
1. Create `scripts/precompute_archetype_embeddings.py`:
   - Read embedding.matryoshka_dim D from MultiScaleConfig.
   - Load 256 slot descriptions from `data/conceptnet_slots.json`.
   - Embed each slot using EmbeddingGemma 2 truncated to D dimensions.
   - Export normalized (256, D) float32 matrix to `data/archetype_embeddings.bin`.
2. Create `scripts/sweep_archetype_grounding.py`:
   - Benchmark synthetic neologisms across k in {1..5}, theta_arch in [0.50..0.85], and Belnap mapping strategies.
   - Select optimal parameters and persist to `config/multi_scale_profile.json`.
3. Modify `src/parser/mmap_grounder.py`:
   - Memory-map `archetype_embeddings.bin` with shape (256, D).
   - Add `project_oov_concept(term, embedding_manager)` consuming calibrated parameters.
4. Update `src/parser/asg_compiler.py` to assign calibrated Belnap truth values into Band 3/4 slots.
5. Extend `tests/test_mmap_grounder_speed.py` verifying neologism grounding and latency SLA (< 0.05 ms).
Verify with `python scripts/precompute_archetype_embeddings.py` and `pytest tests/test_mmap_grounder_speed.py`.
```

---

### Session 4: Dense Energy Seeding & Spreading Activation Sweep

**Goal**: Seed the `SpreadingActivationRetriever` with continuous semantic energy vectors on informal or conversational queries where formal predicate unification targets ($?X$) cannot be synthesized, calibrating seed thresholds and hybrid blending ratios.

#### Target Files
- `src/core/asg.py` [MODIFY]: Allow `QuantaNode` or entity cluster representations to cache compact dense vector references.
- `src/memory/spreading_activation.py` [MODIFY]: Add dense seed vector injection $\mathbf{e}_0$ to `find_seed_nodes()` and `traverse_hipporag()` using configurable parameters (`spreading.dense_seed_threshold`, `spreading.hybrid_blend_alpha`, `spreading.max_seed_nodes`).
- `scripts/sweep_spreading_energy_seeding.py` [NEW]: Empirical sweep evaluating seed thresholds $\theta_{\text{seed}} \in [0.45, 0.85]$, blend ratios $\alpha \in [0.0, 1.0]$, and seed caps $K_{\text{seed}} \in [3, 16]$.
- `tests/test_spreading_activation_retrieval.py` [MODIFY]: Extend test cases with abstract/paraphrased queries and configurable blending.

#### Technical Specification
1. **Continuous Seed Energy Vector**:
   - Embed user query $\mathbf{q} \in \mathbb{R}^D$.
   - Calculate cosine similarities against active entity cluster nodes: $s_i = \cos(\mathbf{q}, \mathbf{c}_i)$.
   - Filter by threshold:
     $$\mathbf{e}_{0,\text{dense}}[i] = \begin{cases} s_i, & \text{if } s_i \ge \theta_{\text{seed}} \\ 0.0, & \text{otherwise} \end{cases}$$
2. **Hybrid Blending Formulation**:
   - Combine dense cosine energy with discrete SIMD Hamming distance energy:
     $$\mathbf{e}_0 = \alpha \cdot \mathbf{e}_{0,\text{dense}} + (1 - \alpha) \cdot \mathbf{e}_{0,\text{hamming}}$$
   - Retain top $K_{\text{seed}}$ nodes with non-zero energy.
3. **Sweep Harness (`scripts/sweep_spreading_energy_seeding.py`)**:
   - Evaluates multi-hop QA recall on MuSiQue indirect queries (*"What prevented the spacecraft from overheating?"* vs *`SUNSHIELD`*).
   - Grid sweep:
     - $\theta_{\text{seed}} \in \{0.50, 0.60, 0.65, 0.70, 0.75\}$
     - $\alpha \in \{0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0\}$
     - $K_{\text{seed}} \in \{3, 5, 8, 12\}$
   - Measures multi-hop gold bridge recall and spreading activation latency.
   - Persists Pareto winner to `config/multi_scale_profile.json`.

#### Verification & Test Suite
```powershell
python scripts/sweep_spreading_energy_seeding.py
pytest tests/test_spreading_activation_retrieval.py
```
- Benchmark multi-hop retrieval recall on MuSiQue with abstract paraphrases.
- Verify total spreading activation latency remains strictly $< 5.0\text{ ms}$.

#### Example Starting Prompt for Session 4
```text
Execute Session 4 of the ANVC Master Plan (quanta_embeddinggemma_anvc_master_plan.md).
Goal: Implement Dense Energy Seeding & Spreading Activation Sweep.
1. Modify `src/core/asg.py` to allow `QuantaNode` to store an optional dense vector.
2. Create `scripts/sweep_spreading_energy_seeding.py`:
   - Benchmark multi-hop retrieval over indirect questions.
   - Sweep theta_seed in [0.50..0.75], alpha in [0.0..1.0], and K_seed in [3..12].
   - Select winning parameters ensuring latency < 5.0 ms and persist to `config/multi_scale_profile.json`.
3. Modify `src/memory/spreading_activation.py`:
   - Read calibrated `spreading.dense_seed_threshold`, `spreading.hybrid_blend_alpha`, and `spreading.max_seed_nodes` from MultiScaleConfig.
   - In `find_seed_nodes()`, compute hybrid blended seed energy e_0.
   - Pass e_0 directly into HippoRAG PPR (`traverse_hipporag`).
4. Extend `tests/test_spreading_activation_retrieval.py` with test cases verifying indirect query matching.
Verify with `pytest tests/test_spreading_activation_retrieval.py`.
```

---

### Session 5: Code AST & Multimodal Alignment Sweep

**Goal**: Leverage EmbeddingGemma 2's native code and vision alignment to connect Python/Java AST topologies and diagram visual features directly to QUANTA's discrete Band 1 AST slots and RCC-8 spatial predicates, calibrating linking thresholds through empirical code retrieval sweeps.

#### Target Files
- `src/parser/ast_parser.py` [MODIFY]: Use `task: code retrieval | query: {code_signature}` to anchor function identifiers and docstrings to semantic clusters in the global knowledge base with configurable threshold $\theta_{\text{code}}$.
- `src/parser/scene_graph_grounder.py` [NEW]: Multimodal bridge mapping visual perception crops directly into ConceptNet archetypes with configurable threshold $\theta_{\text{vis}}$.
- `scripts/sweep_code_grounding.py` [NEW]: Empirical sweep on HumanEval problem descriptions evaluating prefix strategies and similarity thresholds for code AST grounding.
- `tests/test_ast_parser.py` [MODIFY]: Unit tests verifying code-to-NL cross-modal retrieval and configurable thresholds.

#### Technical Specification
1. **Code AST Semantic Anchoring**:
   - In `ASTForwardParser`, embed function declarations, arguments, and docstrings using code prefix:
     `task: code retrieval | query: def {func_name}({args}): {docstring}`
   - Attach the resulting $D$-dimensional representation to the function root node in the ASG.
   - Enables natural language queries (*"find sorting algorithms with early termination"*) to ground directly to `bubble_sort` AST subtrees without keyword overlap.
2. **Code Grounding Sweep (`scripts/sweep_code_grounding.py`)**:
   - Tests across 82 HumanEval problem docstrings and code solutions.
   - Evaluates:
     - Prefix styles: `task: code retrieval | query: ...` vs raw docstring vs signature-only.
     - Thresholds: $\theta_{\text{code}} \in \{0.55, 0.60, 0.65, 0.70, 0.75, 0.80\}$.
   - Measures retrieval pass@1 and identifier independence.
   - Persists winning configuration to `config/multi_scale_profile.json`.

#### Verification & Test Suite
```powershell
python scripts/sweep_code_grounding.py
pytest tests/test_ast_parser.py
```
- Verify cross-modal alignment between docstrings and function bodies on HumanEval problem definitions.
- Verify visual crop embeddings map to correct WordNet noun synsets with $\ge 85\%$ accuracy.

#### Example Starting Prompt for Session 5
```text
Execute Session 5 of the ANVC Master Plan (quanta_embeddinggemma_anvc_master_plan.md).
Goal: Implement the Code AST & Multimodal Alignment Sweep.
1. Create `scripts/sweep_code_grounding.py`:
   - Evaluate HumanEval problem docstrings against function implementations.
   - Sweep prefix strategies and similarity thresholds theta_code in [0.55..0.80].
   - Select winning parameters and persist to `config/multi_scale_profile.json`.
2. Modify `src/parser/ast_parser.py`:
   - Ingest calibrated `ast.code_similarity_threshold` from MultiScaleConfig.
   - Embed function definitions, signatures, and docstrings using the calibrated prefix.
   - Attach dense representations to AST root nodes in QuantaGraph.
3. Create `src/parser/scene_graph_grounder.py`:
   - Implement `SceneGraphGrounder` mapping visual bounding box embeddings to ConceptNet slots using calibrated `scene.visual_archetype_threshold`.
4. Update `tests/test_ast_parser.py` to verify natural language code search over AST graphs.
Verify with `pytest tests/test_ast_parser.py`.
```

---

### Session 6: Head-to-Head Paired Ablation & Gate G10 Promotion

**Goal**: Run the full publication-grade paired benchmarking harness on the NVIDIA RTX 3070 with all calibrated parameters active, comparing Base LLM vs Dense RAG vs QUANTA-Gemma ANVC across the full benchmark matrix, establishing Gate G10 promotion criteria.

#### Target Files
- `scripts/run_paired_benchmarks.py` [MODIFY]: Add `--enable-embeddinggemma` and `--ablation-mode 3way-dense` flags.
- `src/benchmarks/paired_evaluator.py` [MODIFY]: Add `EvaluationCondition.DENSE_RAG` and `EvaluationCondition.QUANTA_GEMMA`.
- `EVAL.md` [MODIFY]: Record experimental run `exp-037a` and establish Gate G10 promotion verdict.
- `docs/publication.md` [MODIFY]: Update Section 12 with final comparative scorecards.
- `output/publication/tables/embeddinggemma_ablation.tex` [NEW]: Auto-generated LaTeX table comparing Base LLM vs Dense-Only RAG vs QUANTA-Gemma.

#### Experimental Matrix ($N=135+$ Samples, RTX 3070 Live Execution)
- **Condition A (Base LLM)**: Qwen 3.5 4B MTP standalone.
- **Condition B (Dense-Only RAG)**: Pure vector search (EmbeddingGemma 2) + top-$k$ context stuffing into Qwen 3.5 4B.
- **Condition C (QUANTA-Gemma ANVC)**: Full QUANTA Neuro-Symbolic Core + EmbeddingGemma 2 ANVC (Calibrated Hybrid RRF Filter + Calibrated OOV Archetype Projection + Calibrated Dense Energy Seeding).

#### Gate G10 Promotion Criteria
1. **Combined Peak VRAM**: Must remain strictly $\le 4.2\text{ GB}$ (leaving $\ge 3.8\text{ GB}$ free for batch queues on RTX 3070).
2. **MuSiQue Multi-Hop Accuracy**: Lift from current $40.0\%$ to **$\ge 48.0\%$** through improved bridge-evidence retention.
3. **Ingestion TTFT**: Must maintain $\le 4,500\text{ ms}$ on 8k-token documents (no regressions against Gate G6).
4. **Strict Symbolic Guard**: Hallucination rate on transactional queries must remain strictly **$0.000000\%$**.

#### Example Starting Prompt for Session 6
```text
Execute Session 6 of the ANVC Master Plan (quanta_embeddinggemma_anvc_master_plan.md).
Goal: Run Head-to-Head Paired Ablation & Gate G10 Promotion.
1. Update `src/benchmarks/paired_evaluator.py` to support `Condition DENSE_RAG` and `Condition QUANTA_GEMMA`.
2. Update `scripts/run_paired_benchmarks.py` with flags `--enable-embeddinggemma` and `--dense-hybrid`.
3. Execute live paired benchmark across MuSiQue, BABILong, NIAH, and HumanEval using calibrated configuration from `config/multi_scale_profile.json` (mock strictly disallowed).
4. Auto-generate LaTeX table `output/publication/tables/embeddinggemma_ablation.tex`.
5. Record run `exp-037a` in `EVAL.md` and evaluate Gate G10 promotion criteria.
```

---

## Verification Plan

### Automated Unit & Regression Tests
- **Session 1**: `pytest tests/test_embedding_manager.py tests/test_service_manager.py`
- **Session 2**: `pytest tests/test_relevance_filter.py tests/test_fast_path_assembler.py`
- **Session 3**: `pytest tests/test_mmap_grounder_speed.py tests/test_asg_compiler.py`
- **Session 4**: `pytest tests/test_spreading_activation_retrieval.py`
- **Session 5**: `pytest tests/test_ast_parser.py`
- **Full System Regression**: `pytest tests/` (100% pass across all tests)

### Live Workstation Telemetry Verification (RTX 3070)
- VRAM Monitoring: `nvidia-smi --query-gpu=memory.used,memory.free --format=csv`
- Combined VRAM constraint: $\le 4,200\text{ MB}$ under 8-slot concurrent load.
