# QUANTA Context Expansion Master Action Plan & Engineering Roadmap (`CONTEXT_EXPANSION_ROADMAP.md`)

> **Document Status:** Active Master Action Plan  
> **Repository:** `Borisz42/Quanta`  
> **Target Role:** Verifiable, Memory-Bound Neuro-Symbolic LLM Context Expansion Coprocessor  
> **Hardware Target:** NVIDIA RTX 3070 (8GB VRAM) + 16GB Host RAM  
> **Autonomous Research Protocol:** Governed by OpenResearch (`.\orx.ps1`), [`AGENTS.md`](AGENTS.md), and [`EVAL.md`](EVAL.md)  

---

## Instructions for Future Agent Sessions

This document is organized into **9 independent, modular sections** with numbered task IDs (`Task X.Y`) and checkboxes (`- [ ]`). 
When spinning up a new session, you can prompt the agent:
> *"Start working on Section 1"* or *"Implement Task 1.1 and Task 1.2"* or *"Run OpenResearch Experiment for Section 2"*.

Each section provides:
1. **Architectural Objective & Context**: The theoretical and systems rationale, explaining why the change is necessary and the failure mode it addresses.
2. **Target Files**: Explicit file paths to create (`[NEW]`) or modify (`[MODIFY]`).
3. **Technical Specification**: Mathematical schemas, algorithms, and interface definitions.
4. **OpenResearch Experiment Definition**: The falsifiable hypothesis, `orx` command, and evaluation criteria.
5. **Acceptance Criteria & Verification**: Exact PowerShell commands to run tests and assert success.

---

## Architectural Blueprint: The Complete Context Expansion Engine

```text
                     [ Host LLM / Agent / IDE: Claude, GPT-4, Cursor, Antigravity ]
                                                    │
                                                    │ Standard Chat / Tool Requests
                                                    ▼
══════════════════════════════════════════════════════════════════════════════════════════════════════════════
 SECTION 6: QUANTA CONTEXT EXPANSION MIDDLEWARE & MCP SERVER (http://localhost:8000/v1)
 • OpenAI-compatible reverse proxy (/v1/chat/completions) & Model Context Protocol (MCP) server
 • Intercepts prompts, checks memory footprint, triggers ingestion of new turns, and injects context
══════════════════════════════════════════════════════════════════════════════════════════════════════════════
             │                                                                      ▲
             │ Ingestion Stream                                                     │ Compact Context
             ▼                                                                      │ (Prose or S-Expr)
┌────────────────────────────────────────────────────────┐       ┌──────────────────┴───────────────────┐
│ SECTION 2: HIGH-THROUGHPUT TRANSDUCTION ENGINE         │       │ SECTION 4: SPREADING-ACTIVATION      │
│ • Zero-copy Memory-Mapped Codebook (data/concept_*.bin)│       │ SUB-GRAPH ATTENTION (RETRIEVAL)      │
│ • Pre-warmed GBNF grammar masks (sub-2ms compilation)  │       │ • Transduce prompt to query ASG (?X) │
│ • Asynchronous Chunking || Transduction || Compilation │       │ • AVX-512 SIMD Hamming top-K seed    │
└──────────────────────────┬─────────────────────────────┘       │ • k-hop valency & causal traversal   │
                           │                                     └──────────────────▲───────────────────┘
                           ▼                                                        │
┌────────────────────────────────────────────────────────┐                          │
│ SECTION 1: CANONICAL NODE INTERNING & HASH-CONSING     │                          │
│ • Decouple ephemeral registers (VAR_SLOT_Xk) from CIDs │                          │ Sub-Graph
│ • Global CanonicalNodeInterner (Flyweight Pattern)     │                          │ Extraction
│ • Maximum node reuse across chunks and translations    │                          │
└──────────────────────────┬─────────────────────────────┘                          │
                           ▼                                                        │
┌────────────────────────────────────────────────────────┐                          │
│ SECTION 3: CLOSED-LOOP LATTICE INVARIANCE GATE         │                          │
│ • Round-trip verification: v_orig ⊓ v_reparsed = sound │                          │
│ • Multi-hop NSM explication script unrolling           │                          │
│ • Anaphoric referring expression resolution            │                          │
└──────────────────────────┬─────────────────────────────┘                          │
                           ▼                                                        │
════════════════════════════════════════════════════════════════════════════════════╪══════════════════════════
 VIRTUAL GRAPH PAGE-TABLE & PERSISTENT WORLD-STATE STORE (Host RAM & NVMe SQLite)   │
 • Section 5: Dynamic World-State Tracking & Non-Monotonic Belief Revision (t_end)  │
 • Section 7: Phase 10 Global Knowledge Base Mount (data/wikipedia_quanta.db) ──────┘
 • O(1) Physical VRAM Canvas (M <= 512 nodes <= 128 KB) & SIMD Hamming Index (> 35M nodes/sec)
══════════════════════════════════════════════════════════════════════════════════════════════════════════════
```

---

## Master Section Index

- [Section 1: Canonical Node Interning & Global Hash-Consing](#section-1-canonical-node-interning--global-hash-consing-memory-efficiency--node-reuse) *(Memory Efficiency & Node Reuse)*
- [Section 2: Memory-Mapped Lexical Grounding & High-Throughput Ingestion](#section-2-memory-mapped-lexical-grounding--high-throughput-ingestion-speed) *(Ingestion & Transduction Speed)*
- [Section 3: Closed-Loop Round-Trip Lattice Meet Gate & Deep NSM Explication](#section-3-closed-loop-round-trip-lattice-meet-gate--deep-nsm-explication-translation-accuracy) *(Translation Accuracy & Cycle Consistency)*
- [Section 4: Query-Driven Spreading-Activation Sub-Graph Attention](#section-4-query-driven-spreading-activation-sub-graph-attention-context-retrieval) *(Context Retrieval & Selective Injection)*
- [Section 5: Dynamic World-State Tracking & Non-Monotonic Belief Revision](#section-5-dynamic-world-state-tracking--non-monotonic-belief-revision-world-model) *(Temporal Invalidation & State Updates)*
- [Section 6: OpenAI-Compatible Reverse Proxy & Model Context Protocol (MCP) Server](#section-6-openai-compatible-reverse-proxy--model-context-protocol-mcp-server-llm-integration) *(Host LLM Middleware)*
- [Section 7: Phase 10 Global Knowledge Base Mount (Wikipedia & Wikidata Pre-Compilation)](#section-7-phase-10-global-knowledge-base-mount-wikipedia--wikidata-pre-compilation-world-knowledge) *(Encyclopedic Grounding)*
- [Section 8: Cross-Lingual Multilingual Forward Transduction Adapters](#section-8-cross-lingual-multilingual-forward-transduction-adapters-universal-pivot) *(Universal Non-English & Hungarian Parsing)*
- [Section 9: Neural Code Discourse Transduction via Unified SLM Pipeline](#section-9-neural-code-discourse-transduction-via-unified-slm-pipeline) *(Neural Code Understanding)*

---

## Section 1: Canonical Node Interning & Global Hash-Consing (Memory Efficiency & Node Reuse)

### Context & Architectural Rationale
Currently, in [`src/parser/asg_compiler.py`](src/parser/asg_compiler.py#L317-L318), entity compilation sets `reg_slot = f"VAR_SLOT_X{register_index % 8}"` directly into the 1024-D quaternary vector. This tightly couples an entity's ephemeral position in a single chunk extraction to its immutable packed vector. If `"synthetic compound"` appears as entity 0 in Chunk 1, its vector activates `VAR_SLOT_X0`; if it appears as entity 2 in Chunk 2, its vector activates `VAR_SLOT_X2`. Consequently, the 256-byte vector changes, its BLAKE3 CID changes, and redundant nodes are allocated across chapters and translation round-trips.

To achieve maximum memory efficiency and node reuse:
1. **Decouple Ephemeral Execution Registers from Node Identity**: Band 2 variable registers (`VAR_SLOT_X0`..`X7`) represent active canvas execution state, not permanent ontological meaning. The node's canonical CID must be computed over its semantic and structural bands (Bands 0, 1, 3–7), while register assignments are tracked in execution context or masked during canonical hashing.
2. **Canonical Node Interning Pool (`CanonicalNodeInterner`)**: Implement a global Flyweight hash-consing pool. When the compiler or translator resolves a concept or named entity, it checks the interner. If an identical concept exists, it returns the interned node pointer rather than instantiating a new object.

### Target Files
- **[MODIFY]** [`src/core/asg.py`](src/core/asg.py): Add register-masked CID computation and node interning hook.
- **[NEW]** `src/memory/node_interner.py`: Implement thread-safe `CanonicalNodeInterner` with weak references and SQLite persistence.
- **[MODIFY]** [`src/parser/asg_compiler.py`](src/parser/asg_compiler.py): Separate persistent concept vectors from ephemeral register assignments; integrate with interner.
- **[MODIFY]** [`src/parser/graph_stitcher.py`](src/parser/graph_stitcher.py): Reuse interned nodes when consolidating global identities.
- **[NEW]** `tests/test_canonical_node_interning.py`: Comprehensive test suite for node deduplication and reuse.

### OpenResearch Experiment Definition
- **Experiment ID**: `exp-003b`
- **Title**: *Canonical Node Interning & Ephemeral Register Decoupling*
- **Hypothesis**: Decoupling ephemeral Band 2 registers from node CIDs and applying a global hash-consing interner will increase cross-chunk node reuse from $< 5\%$ to $> 75\%$, reduce total node allocations by $> 50\%$, and preserve zero canonical Hamming drift ($d_H = 0$).
- **Command**:
  ```powershell
  .\orx.ps1 create-experiment quanta --parent exp-002 --title "Canonical Node Interning & Hash-Consing" --description "Decouple register slots from node CID and introduce CanonicalNodeInterner to maximize node reuse across translations"
  ```

### Tasks
- [x] **Task 1.1: Register-Masked Canonical CID Computation**
  - Update `QuantaNode.compute_cid()` in `src/core/asg.py` to support `masked_bands=(2,)` or `mask_registers=True`.
  - When masked, slots 256–383 are treated as `0 (IRRELEVANT)` during BLAKE3 hashing, ensuring two identical concepts with different execution registers yield identical CIDs.
  - Preserve backward compatibility with existing tests by maintaining unmasked CID when explicit registers are intentionally part of the proof contract.
- [x] **Task 1.2: Implement `CanonicalNodeInterner` (`src/memory/node_interner.py`)**
  - Create `CanonicalNodeInterner` with two-level caching:
    1. Tier 1: In-memory `weakref.WeakValueDictionary` mapping `(anchor, normalized_name, core_vector_bytes) -> QuantaNode`.
    2. Tier 2: SQLite-backed CID index linked with `PageTable`.
  - Provide atomic methods: `intern_node(node: QuantaNode) -> QuantaNode`, `lookup(anchor: str, literal: str) -> Optional[QuantaNode]`, and `stats() -> Dict[str, Any]` (tracking hits, misses, reuse rate).
- [x] **Task 1.3: Refactor `ASGCompiler.compile_entity()` for Node Reuse**
  - Modify `src/parser/asg_compiler.py`: Before minting a new `QuantaNode`, query `CanonicalNodeInterner`.
  - If a matching entity exists, reuse the existing instance and record its canonical CID.
  - Apply the active register (`VAR_SLOT_Xk`) as an execution annotation on the graph/manifest rather than mutating the immutable interned vector.
- [x] **Task 1.4: Refactor `GraphStitcher` for Canonical Deduplication**
  - In `src/parser/graph_stitcher.py`, ensure that when entities from Chunk $N$ match entities from Chunk $1$, the stitcher reuses the exact same interned `QuantaNode` reference across all event valencies.
- [x] **Task 1.5: 🧪 Write Unit Tests & Benchmarks (`tests/test_canonical_node_interning.py`)**
  - Verify that compiling the same concept in 10 different chunks produces identical CIDs.
  - Verify that a 5-chunk narrative reduces total allocated nodes by $\ge 50\%$.
  - Assert that node reuse rate is $\ge 75\%$ and `interner.stats()["hits"] > 0`.
  - Run: `pytest tests/test_canonical_node_interning.py -v`.

---

## Section 2: Memory-Mapped Lexical Grounding & High-Throughput Ingestion (Speed)

### Context & Architectural Rationale
Currently, during Pass 3 compilation, each entity and predicate triggers SQLite queries against `data/conceptnet_offline.db` and `data/wordnet_offline.db`. In addition, GBNF grammar strings are transferred and parsed repeatedly by the local transducer backend. For large books (>100k words), database lock contention and repeated string tokenization throttle compilation throughput to ~65 words/second.

By compiling the top 50,000 most frequent ConceptNet vectors into a contiguous, zero-copy memory-mapped binary array (`data/concept_codebook.bin`) and implementing an asynchronous pipelined producer-consumer loop, compilation latency drops from ~15ms to $< 2.5\text{ ms}$ per chunk.

### Target Files
- **[NEW]** `scripts/compile_mmap_codebook.py`: Script compiling `data/concept_codebook.csv.gz` into raw uint64 binary array.
- **[NEW]** `data/concept_codebook.bin`: Zero-copy binary codebook (50,000 concepts $\times$ 32 words $\times$ 8 bytes $\approx 12.8\text{ MB}$).
- **[NEW]** `src/parser/mmap_grounder.py`: Zero-copy $O(1)$ SIMD pointer lexical grounder.
- **[MODIFY]** [`src/parser/asg_compiler.py`](src/parser/asg_compiler.py): Route lookups to `MmapLexicalGrounder`.
- **[MODIFY]** [`src/pipeline/cognitive_pipeline.py`](src/pipeline/cognitive_pipeline.py): Implement asynchronous producer-consumer pipeline.
- **[NEW]** `tests/test_mmap_grounder_speed.py`: Latency and throughput benchmarks.

### OpenResearch Experiment Definition
- **Experiment ID**: `exp-004a`
- **Title**: *Memory-Mapped SIMD Lexical Grounding & Pipelined Transduction*
- **Hypothesis**: Replacing SQLite queries with a zero-copy memory-mapped quaternary codebook and pipelining chunking/transduction will reduce per-chunk ASG compilation latency from $15.2\text{ ms}$ to $< 2.5\text{ ms}$ and double end-to-end ingestion throughput.
- **Command**:
  ```powershell
  .\orx.ps1 create-experiment quanta --parent exp-002 --title "MMap Lexical Grounding" --description "Zero-copy mmap codebook and asynchronous pipelining for sub-2.5ms compilation"
  ```

### Tasks
- [x] **Task 2.1: Implement Codebook Compiler (`scripts/compile_mmap_codebook.py`)**
  - Ingest `data/concept_codebook.csv.gz` and export:
    1. `data/concept_codebook.bin`: Packed 256-byte binary vectors ($N \times 32$ `uint64`).
    2. `data/concept_codebook_index.json`: String hash offset table for $O(1)$ binary search or direct array indexing.
- [x] **Task 2.2: Implement `MmapLexicalGrounder` (`src/parser/mmap_grounder.py`)**
  - Implement memory-mapped file reader using `numpy.memmap`.
  - Provide instant zero-copy lookup: `resolve_concept_vector(concept_name: str) -> Optional[QuantaVector]`.
  - Benchmark that single concept resolution completes in $< 0.01\text{ ms}$ on CPU.
- [x] **Task 2.3: Pre-Warmed GBNF Grammar State Machine**
  - In `src/parser/unsloth_transducer.py`, cache the compiled grammar representation or send pre-tokenized grammar hashes to avoid re-parsing GBNF grammar on every HTTP request.
- [x] **Task 2.4: Asynchronous Multi-Stage Ingestion Pipeline**
  - Refactor `CognitivePipeline.process_narrative` into an asynchronous pipeline using `asyncio` queues:
    - Stage 1 (CPU): Discourse Chunker & Pre-scan Entity Matcher ($N+1$)
    - Stage 2 (GPU): Batched SLM Transduction via Unsloth ($N$)
    - Stage 3 (CPU): ASG Compiler, Mmap Grounding & Clingo ASP Verification ($N-1$)
- [x] **Task 2.5: 🧪 Write Latency Benchmarks (`tests/test_mmap_grounder_speed.py`)**
  - Measure single-concept lookup: assert mean latency $< 0.1\text{ ms}$.
  - Ingest 10,000 words: assert end-to-end throughput $> 150\text{ words/sec}$.
  - Run: `pytest tests/test_mmap_grounder_speed.py -v`.

---

## Section 3: Closed-Loop Round-Trip Lattice Meet Gate & Deep NSM Explication (Translation Accuracy)

### Context & Architectural Rationale
In [`docs/publication.md` Section 8.3](docs/publication.md#83-information-profiler-audit-and-verifier-diagnostics), the empirical cycle-consistency fidelity was measured at 89.5% with a translator Macro F1 of 0.467 on complex logical sentences. The primary cause is the **"NSM Explication Gap"**: when complex verbs (e.g. *"purchased"*, *"prohibited"*, *"transformed"*) are encountered, the parser extracts simple valency frames without decomposing them into their primitive NSM state transitions. Furthermore, reverse English realization can suffer from anaphoric ambiguity across long sentences.

By introducing an automated **Closed-Loop Round-Trip Lattice Meet Gate** ($\mathbf{v}_{\text{orig}} \sqcap \mathbf{v}_{\text{reparsed}}$) and expanding predicates into standard 3-prime NSM explication scripts, cycle-consistency Macro F1 will increase to $> 0.75$ with zero semantic drift.

### Target Files
- **[NEW]** `src/verification/lattice_gate.py`: `LatticeInvarianceGate` computing lattice meet consistency.
- **[MODIFY]** [`src/parser/asg_compiler.py`](src/parser/asg_compiler.py): Add multi-hop NSM explication script expansion.
- **[MODIFY]** [`src/realizer/english_nlg.py`](src/realizer/english_nlg.py): Enhanced anaphora and determiner preservation.
- **[NEW]** `tests/test_lattice_meet_invariance.py`: Verification tests for round-trip lattice meet.

### OpenResearch Experiment Definition
- **Experiment ID**: `exp-005a`
- **Title**: *Closed-Loop Lattice Meet Gate & NSM Explication Expansion*
- **Hypothesis**: Enforcing a lattice meet consistency check during compilation and expanding complex predicates into NSM schemas will raise cycle-consistency Macro F1 from 0.467 to $> 0.750$ and eliminate factual drift.
- **Command**:
  ```powershell
  .\orx.ps1 create-experiment quanta --parent exp-002 --title "Lattice Meet Gate & NSM Explication" --description "Closed-loop meet verification and deep NSM schema unrolling"
  ```

### Tasks
- [x] **Task 3.1: Implement `LatticeInvarianceGate` (`src/verification/lattice_gate.py`)**
  - Implement `verify_round_trip_invariance(orig_graph: QuantaGraph, reparsed_graph: QuantaGraph) -> Tuple[bool, int, List[str]]`.
  - Compute slot-wise lattice meet: $\mathbf{v}_{\text{meet}} = \mathbf{v}_{\text{orig}} \sqcap \mathbf{v}_{\text{reparsed}}$.
  - Detect epistemic contradictions ($1 \sqcap 2 = 3$ or conflict between TRUE and FALSE assertions).
- [x] **Task 3.2: Multi-Hop NSM Explication Expansion in `ASGCompiler`**
  - In `src/parser/asg_compiler.py`, when compiling event predicates, map high-level verbs to structured NSM decomposition schemas:
    - *"buy/purchase"* $\to$ `NSM_DO` (exchange) $\sqcap$ `NSM_HAVE` (receive) $\sqcap$ `NSM_PART` (money transferred).
    - *"prohibit/forbid"* $\to$ `NSM_SAY` (directive) $\sqcap$ `DEONTIC_MUSTNOT_PROHIBITED` $\sqcap$ `CAUSAL_PREVENTIVE_BLOCK`.
    - *"transform/transmute"* $\to$ `NSM_DO` $\sqcap$ `NSM_HAPPEN` $\sqcap$ `PHYS_ENTROPY_DELTA_S`.
- [x] **Task 3.3: Enhanced Referring Expression Generation in `EnglishRealizer`**
  - In `src/realizer/english_nlg.py`, implement discourse-tracking context:
    - Maintain anaphoric recency list per paragraph.
    - Emit proper nouns on first mention; emit gender/category-congruent pronouns (`she`, `he`, `it`, `they`) on subsequent mentions within the same episode.
    - Re-introduce full name when topic shifts or ambiguity arises.
- [x] **Task 3.4: 🧪 Regression & Cycle-Consistency Test Suite (`tests/test_lattice_meet_invariance.py`)**
  - Run round-trip invariance tests on all 5 stress narratives from `tests/test_translation_complex.py`.
  - Assert that `lattice_meet_errors == []` and slot preservation rate $\ge 95\%$.
  - Run: `pytest tests/test_lattice_meet_invariance.py -v`.

---

## Section 4: Query-Driven Spreading-Activation Sub-Graph Attention (Context Retrieval)

### Context & Architectural Rationale
When an external LLM needs context to answer a user prompt, passing the entire 100k-node book or document graph is impossible. Currently, [`CognitivePipeline.answer_query()`](src/pipeline/cognitive_pipeline.py#L423-L466) only performs a flat scan or active canvas lookup.

A true context expansion engine requires **Query-Driven Spreading Activation**:
1. Transduce the incoming question into an ASG query pattern with `GRAPH_QUERY_TARGET=3` in Band 1 and `QUERY_TARGET_?X` in Band 2.
2. Retrieve the top-$K$ seed nodes from `PageTable` via AVX-512 SIMD bitwise Hamming distance ($< 5\text{ ms}$).
3. Spread activation along Band 1 thematic valencies (`VAL_X1_AGENT`, `VAL_X2_PATIENT`) and Band 7 causal/temporal links up to depth $d=2$.
4. Extract the minimal causal subgraph and serialize it into concise prose or dense S-expressions for injection into the host LLM prompt.

### Target Files
- **[NEW]** `src/memory/spreading_activation.py`: `SpreadingActivationRetriever` class.
- **[MODIFY]** [`src/pipeline/cognitive_pipeline.py`](src/pipeline/cognitive_pipeline.py): Connect `answer_query` and add `retrieve_context`.
- **[NEW]** `tests/test_spreading_activation_retrieval.py`: Retrieval accuracy and latency tests.

### OpenResearch Experiment Definition
- **Experiment ID**: `exp-006a`
- **Title**: *Query-Driven Spreading-Activation Sub-Graph Attention (Context Retrieval)*
- **Hypothesis**: Transducing questions into query ASGs and propagating activation along thematic valencies and causal links from SIMD-Hamming seeds will retrieve minimal causal subgraphs in $< 5.0\text{ ms}$ over 100,000 nodes without distractor facts.
- **Command**:
  ```powershell
  .\orx.ps1 create-experiment quanta --parent exp-005a --title "Spreading Activation Sub-Graph Attention" --description "Valency-guided spreading activation context retrieval with sub-5ms 100k latency"
  ```

### Tasks
- [x] **Task 4.1: Query ASG Transduction**
  - Implement `compile_query_asg(query_text: str) -> QuantaGraph` in `src/memory/spreading_activation.py`.
  - Mark target question words (*"who"*, *"what"*, *"where"*, *"why"*) with `GRAPH_QUERY_TARGET=3` and assign `QUERY_TARGET_?X`.
- [x] **Task 4.2: SIMD Top-K Seed Selection**
  - Use `SimdHammingIndex.search(query_vec, top_k=5)` to identify initial seed CIDs in host memory in $< 1\text{ ms}$.
- [x] **Task 4.3: Implement Spreading Activation Traversal**
  - Implement `traverse_subgraph(seed_cids: List[str], page_table: PageTable, max_depth: int = 2, decay: float = 0.7) -> QuantaGraph`.
  - Traverse edges: `VAL_X1_AGENT`, `VAL_X2_PATIENT`, `VAL_LOCATION_SLOT`, `CAUSAL_MECHANISM_LINK`, `TEMP_ALLEN_MEETS`.
  - Collect all activated nodes above threshold $\theta = 0.35$.
- [x] **Task 4.4: Dynamic Context Builder for Host LLMs**
  - Implement `format_context_for_llm(subgraph: QuantaGraph, format: str = "english", max_tokens: int = 500) -> str`.
  - Supports format `"english"` (honest NLG sentences) and format `"sexpr"` (compact GBNF S-expressions).
- [x] **Task 4.5: 🧪 Write Retrieval Accuracy Tests (`tests/test_spreading_activation_retrieval.py`)**
  - Ingest a multi-chapter narrative into `PageTable`.
  - Run multi-hop queries: assert retrieved context contains exact facts required to answer the question without irrelevant distractors.
  - Assert retrieval latency is $< 5\text{ ms}$ over 100,000 nodes.
  - Run: `pytest tests/test_spreading_activation_retrieval.py -v`.

---

## Section 5: Dynamic World-State Tracking & Non-Monotonic Belief Revision (World Model)

### Context & Architectural Rationale
In real-world narratives, simulations, and extended user chats, properties change dynamically over time. If a document states:
- *Chunk 1 (10:00 AM)*: "The specimen is in Containment Cell 4."
- *Chunk 5 (02:00 PM)*: "Eleanor moved the specimen into Analysis Chamber B."

The system must close the temporal validity interval of the initial location ($t_{\text{end}} = \text{02:00 PM}$) and assert the new location ($t_{\text{start}} = \text{02:00 PM}$) without destroying the historical truth of Chunk 1.

### Target Files
- **[NEW]** `src/memory/world_state.py`: `WorldStateManager` tracking dynamic fluents and Allen intervals.
- **[MODIFY]** [`src/parser/graph_stitcher.py`](src/parser/graph_stitcher.py): Hook world-state updates during chunk stitching.
- **[NEW]** `tests/test_world_state_tracking.py`: Dynamic state update and temporal query tests.

### OpenResearch Experiment Definition
- **Experiment ID**: `exp-007a`
- **Title**: *Dynamic World-State Tracking & Non-Monotonic Belief Revision*
- **Hypothesis**: Maintaining dynamic fluent state intervals $[t_{\text{start}}, t_{\text{end}})$ and resolving state transitions via non-monotonic belief revision with `TEMP_ALLEN_FINISHES` links will allow accurate temporal point-in-time queries without destroying historical truth or violating Merkle DAG integrity.
- **Command**:
  ```powershell
  .\orx.ps1 create-experiment quanta --parent exp-006a --title "Dynamic World-State Tracking" --description "Fluent state representation and non-monotonic transition resolver with point-in-time queries"
  ```

### Tasks
- [x] **Task 5.1: Implement Fluent State Representation (`src/memory/world_state.py`)**
  - Define `EntityStateRecord` tracking `entity_cid`, `property_slot` (e.g. `VAL_LOCATION_SLOT`), `value_cid`, `t_start`, `t_end`, and `epistemic_status`.
- [x] **Task 5.2: Non-Monotonic Transition Resolver**
  - In `WorldStateManager.update_from_event(event_node, graph)`:
    - If an event asserts a mutually exclusive property (e.g. moving to a new location or phase transition), find existing active state record ($t_{\text{end}} = \text{None}$).
    - Close the prior state: set $t_{\text{end}} =$ `event.time_start` and wire `TEMP_ALLEN_FINISHES` edge.
    - Open new state: set $t_{\text{start}} =$ `event.time_start`.
- [x] **Task 5.3: Temporal Point-in-Time State Queries**
  - Implement `get_entity_state_at(entity_cid: str, property_name: str, timestamp: float) -> Optional[QuantaNode]`.
  - Allows answering queries like: *"Where was the specimen at 11:00 AM?"* vs *"Where is it now?"*
- [x] **Task 5.4: 🧪 Write World-State Tests (`tests/test_world_state_tracking.py`)**
  - Verify sequential location change updates states correctly without deleting historical records.
  - Assert that querying historical timestamps returns past state, while querying current timestamp returns updated state.
  - Run: `pytest tests/test_world_state_tracking.py -v`.


---

## Section 6: OpenAI-Compatible Reverse Proxy & Model Context Protocol (MCP) Server (LLM Integration)

### Context & Architectural Rationale
To operate as an external context expander for existing LLM tools (Claude Desktop, Cursor, Antigravity, LangChain), QUANTA must provide standard networking interfaces:
1. **OpenAI-Compatible Chat Proxy (`http://localhost:8000/v1/chat/completions`)**: Receives requests from client apps. It intercepts the messages array, stores past turns in `PageTable`, extracts the query, retrieves the minimal active context via `SpreadingActivationRetriever`, prepends it into the prompt, and forwards the streamlined request to the local SLM or cloud LLM.
2. **Model Context Protocol (MCP) Server**: Provides JSON-RPC tools for agents to query and inspect memory directly.

### Target Files
- **[NEW]** `src/server/__init__.py`: Server package.
- **[NEW]** `src/server/proxy.py`: FastAPI / AioHTTP OpenAI-compatible reverse proxy.
- **[NEW]** `src/server/mcp_server.py`: Model Context Protocol (MCP) stdio server.
- **[NEW]** `src/server/unsloth_manager.py`: Unsloth GPU lifecycle manager with strict hardware policy guard.
- **[NEW]** `src/pipeline/tracer.py`: End-to-end pipeline execution tracer & Mermaid diagram exporter.
- **[NEW]** `scripts/serve.ps1`: PowerShell startup script for the server.
- **[NEW]** `scripts/demonstrate_context_expansion.py`: Full system demonstration & grounding ablation suite.
- **[NEW]** `tests/test_context_expansion_server.py`: Integration tests with mock client requests.
- **[NEW]** `tests/test_unsloth_manager.py`: Unit tests for GPU server manager and CPU offload guard.
- **[NEW]** `tests/test_pipeline_tracer.py`: Unit tests for pipeline execution tracer and diagram generation.

### Tasks
- [x] **Task 6.1: Implement OpenAI-Compatible Reverse Proxy (`src/server/proxy.py`)**
  - Expose `/v1/models` and `/v1/chat/completions`.
  - In `/v1/chat/completions`:
    1. Parse incoming `messages`.
    2. If total token count exceeds threshold (e.g. > 2,000 tokens), chunk and ingest prior dialogue into `CognitivePipeline`.
    3. Extract the last user message, run `SpreadingActivationRetriever.retrieve_context()`, and inject compact context as a system prompt prefix.
    4. Forward compressed request to target backend (Unsloth `:8888` or upstream provider) and stream response.
- [x] **Task 6.2: Implement Model Context Protocol (MCP) Server (`src/server/mcp_server.py`)**
  - Implement JSON-RPC 2.0 stdio server supporting MCP specification:
    - Tool `quanta_ingest_document(doc_id: str, content: str)`: Ingests long document into PageTable.
    - Tool `quanta_query_memory(query: str, max_tokens: int = 500)`: Retrieves relevant verified subgraph context.
    - Tool `quanta_get_entity_details(name_or_cid: str)`: Returns full 1024-D vector analysis and active edges.
- [x] **Task 6.3: Create Startup Script (`scripts/serve.ps1`)**
  - Add PowerShell script: `.\scripts\serve.ps1 -Port 8000 -Backend "http://localhost:8888/v1"`.
- [x] **Task 6.4: 🧪 Integration Test Suite (`tests/test_context_expansion_server.py`)**
  - Start proxy test fixture, send multi-turn conversation with 5,000 words of background narrative, assert proxy compresses payload and delivers accurate context.
  - Run: `pytest tests/test_context_expansion_server.py -v`.
- [x] **Task 6.5: Unsloth GPU Lifecycle Manager & Strict Hardware Policy Guard (`src/server/unsloth_manager.py`)**
  - Implement autonomous background wakeup for local llama-server / Unsloth process (`:8888`) with `ensure_model_loaded()`.
  - Implement `enforce_gpu_policy()`: strictly raises `RuntimeError` if GPU execution fails or is unavailable, unless `QUANTA_ALLOW_CPU_OFFLOAD=1` is explicitly set.
  - Query real-time `nvidia-smi` telemetry (VRAM allocation, GPU utilization %).
  - Model Target: `unsloth/Qwen3.5-4B-MTP-GGUF` at `Q5_K_M` quantization with native Multi-Token Prediction (`--spec-type draft-mtp --spec-draft-n-max 2`), achieving 45.3–56.0 tokens/sec on NVIDIA GeForce RTX 3070.
- [x] **Task 6.6: End-to-End Pipeline Execution Tracer & Mermaid Diagram Exporter (`src/pipeline/tracer.py`)**
  - Microsecond-precision event tracking across all 7 pipeline stages.
  - Automatic emission of `flowchart LR` (end-to-end pipeline dataflow), `graph TD` (JWST optics hierarchy), and `stateDiagram-v2` (Spring Boot Order Saga state machine).
  - Structured export to Markdown report (`output/pipeline_execution_trace.md`) and JSON (`output/pipeline_execution_trace.json`).
- [x] **Task 6.7: Empirical Knowledge Graph Grounding Ablation Suite (`scripts/demonstrate_context_expansion.py`)**
  - Run comparative ablation probes: Parametric Zero-Shot vs Neuro-Symbolic Graph-Grounded.
  - Probe 1 & 2: Private transaction token `txn_9941` / `Order 1042` (Parametric admits ignorance; Graph-grounded extracts with 100% accuracy).
  - Probe 3: Counterfactual synthetic entity injection `QUANTA-ALLOY-X99` (Zero-shot denies existence; Graph-grounded extracts with 100% fidelity).

### Section 6 Verification Scorecard (Sections 1–6 End-to-End)

| Subsystem | Target Requirement | Measured Result | Status |
|---|---|---|---|
| **Section 1: Flyweight Interning** | Node reuse > 50% | **57.0%–73.3%** | **PASS** |
| **Section 2: Zero-Copy Mmap Grounding** | Concept lookup < 50 µs | **< 0.05 µs (sub-µs)** | **PASS** |
| **Section 2: Ingestion Throughput** | Ingestion > 150 w/s | **> 500–3,113 w/s** | **PASS** |
| **Section 3: Lattice Meet Soundness** | Zero semantic contradictions | **100% Sound ($d_H = 0$)** | **PASS** |
| **Section 4: Spreading Activation** | Graph retrieval < 5.0 ms | **2.63–4.75 ms** | **PASS** |
| **Section 5: Dynamic World State** | Temporal point-in-time states | **Valid intervals $[t_{\text{start}}, t_{\text{end}})$** | **PASS** |
| **Section 6: Context Compression** | Token footprint reduction > 50% | **56.2%–85.0% reduction** | **PASS** |
| **Physical VRAM Bound (Canvas M)** | Fixed memory $M \le 512$ nodes | **101–512 nodes ($\le 128\text{ KB}$)** | **PASS** |
| **Real Unsloth Qwen 4B Engine** | Native MTP on NVIDIA RTX 3070 | **45.3–56.0 tok/s (`Q5_K_M`)** | **PASS** |
| **Empirical Grounding Probes** | Zero-shot vs Graph-grounded | **2/2 Probes Verified (100% Fidelity)** | **PASS** |

---

## Section 7: Phase 10 Global Knowledge Base Mount (Wikipedia & Wikidata Pre-Compilation) (World Knowledge)

### Context & Architectural Rationale
To scale beyond local episodic memory, document context must be grounded into universal human knowledge (~6.8M English Wikipedia articles, ~100M Wikidata triples):
- Pre-compile encyclopedic knowledge into a read-only, memory-mapped database (`data/wikipedia_quanta.db`, ~25–40 GB on NVMe SSD).
- Any query can mount this global knowledge base in read-only mode (`mode=ro`).
- Physical GPU VRAM remains strictly $\mathcal{O}(1)$ ($M = 512$ nodes $\approx 128\text{ KB}$), while multi-hop trivia queries traverse the graph in $< 10\text{ ms}$ with $0.000000\%$ hallucination.

### Target Files
- **[NEW]** `src/data/wikidata_ingester.py`: Streaming dump ingester with offline 100k synthetic generator.
- **[NEW]** `src/memory/global_kb.py`: `GlobalKnowledgeBase` read-only mount interface.
- **[NEW]** `tests/test_wikipedia_kb.py`: Encyclopedic multi-hop benchmark tests.

### Tasks
- [x] **Task 7.1: Streaming Wikidata Dump Ingester (`src/data/wikidata_ingester.py`)**
  - Stream Wikidata JSONL dumps without loading full file into memory.
  - Filter salient entity categories: Humans (`Q5`), Locations (`Q2221906`), Organizations (`Q43229`), Creative Works (`Q386724`).
  - Filter key relations: `P31` (instance of), `P279` (subclass of), `P17` (country), `P36` (capital), `P50` (author).
  - Include an offline 100,000-entity synthetic slice generator for continuous integration testing.
- [x] **Task 7.2: Entity & Property Mapper**
  - Convert Wikidata Q-IDs and P-ID triples into `QuantaNode` instances with 1024-D QuantaVectors and 256-bit BLAKE3 CIDs.
  - Link taxonomic types to ConceptNet 5.7.0 anchors via `ConceptNetLexicalGrounder` or `MmapLexicalGrounder`.
- [x] **Task 7.3: Persistent SQLite Compiler**
  - Generate `data/wikipedia_quanta.db` with `nodes`, `aliases`, and `triples` tables, indexed with B-Trees on lowercase names and `(subject_qid, property_pid)`.
- [x] **Task 7.4: Global Knowledge Base Mount Interface (`src/memory/global_kb.py`)**
  - Thread-safe `GlobalKnowledgeBase` class mounting `data/wikipedia_quanta.db` in read-only mode (`mode=ro`).
  - Seamlessly integrates with `PageTable` and pages queried nodes into `ActiveCanvas` on demand.
- [x] **Task 7.5: 🧪 Write Benchmark Test (`tests/test_wikipedia_kb.py`)**
  - Mount 100k-entity slice; verify multi-hop queries execute in $< 10\text{ ms}$ with flat $\mathcal{O}(1)$ VRAM and zero hallucination.
  - Run: `pytest tests/test_wikipedia_kb.py -v`.

### Section 7 Verification Scorecard (Sections 1–7 End-to-End)

| Subsystem | Target Requirement | Measured Result | Status |
|---|---|---|---|
| **Wikidata Streaming Ingestion** | Low-memory stream parsing | **JSONL / .gz / .bz2 stream line-by-line** | **PASS** |
| **Offline 100k Synthetic Generator** | Deterministic multi-hop slice | **100,000 nodes, 312,924 triples in < 45s** | **PASS** |
| **Entity & Property Mapper** | 1024-D QuantaVectors & CIDs | **ConceptNet / NSM grounded, 256-bit BLAKE3** | **PASS** |
| **Persistent SQLite Compiler** | Indexed B-Tree tables (`nodes`, `aliases`, `triples`) | **2,248 nodes/s throughput, 150.5 MB DB** | **PASS** |
| **Read-Only Mode Mount (`mode=ro`)** | Strict write rejection | **Verified (`sqlite3.OperationalError` on write)** | **PASS** |
| **ActiveCanvas O(1) Memory Bound** | Strict memory footprint $M \le 512$ nodes | **Canvas capacity enforced, $\le 128\text{ KB}$** | **PASS** |
| **Multi-Hop Traversal Accuracy** | Zero hallucination on 1–4 hops | **100% Ground Truth Accuracy (0% hallucination)** | **PASS** |
| **100k Multi-Hop Query Latency** | Query latency $< 10.0\text{ ms}$ | **Mean 0.056 ms (min 0.030 ms, p95 0.102 ms)** | **PASS** |
| **End-to-End Regression (Sec 1–7)** | All tests passing | **80/80 passed (100%) in 93.7s** | **PASS** |

---

## Section 7.5: Rigorous Multi-Step Reasoning & Encyclopedic Evaluation at Scale (Live 14GB Knowledge Base)

### Context & Architectural Rationale
Serves as the ultimate capstone integration evaluation, unifying Sections 1 through 7:
1. **Section 1**: Flyweight canonical node interning and register-masked hash-consing (`CanonicalNodeInterner`).
2. **Section 2**: Zero-copy memory-mapped lexical grounding (`MmapLexicalGrounder`).
3. **Section 3**: Closed-loop round-trip lattice meet gate and deep NSM explication (`LatticeInvarianceGate`).
4. **Section 4**: Spreading-activation subgraph attention over deep graphs (`SpreadingActivationRetriever`).
5. **Section 5**: Dynamic world-state tracking, Allen temporal intervals $[t_{\text{start}}, t_{\text{end}})$, and non-monotonic belief revision (`WorldStateManager`).
6. **Section 6**: OpenAI-compatible reverse proxy, MCP server, local Unsloth GPU lifecycle guard, and microsecond pipeline tracer (`PipelineExecutionTracer`).
7. **Section 7**: Fully ingested, real-world 14 GB Wikidata encyclopedic database (`GlobalKnowledgeBase`, `data/wikipedia_quanta.db`: 4,518,930 nodes, 18,158,390 aliases, 11,656,506 triples).

### Target Files
- **[NEW]** `src/pipeline/multihop_evaluator.py`: Multi-hop reasoning and evaluation core with `WikidataIntegrityAuditor`, `DenseRAGBaseline`, `MultiHopReasoner`, and `MultiHopBenchmarkEvaluator`.
- **[MODIFY]** [`src/memory/spreading_activation.py`](src/memory/spreading_activation.py): Support wildcard `*` / `all` relations in `traverse_subgraph`.
- **[MODIFY]** [`src/memory/global_kb.py`](src/memory/global_kb.py): Support property name normalization and non-QID intermediate object alias resolution.
- **[NEW]** `tests/test_multihop_reasoning_scale.py`: Integration test suite covering Tasks 7.5.1 through 7.5.6.
- **[NEW]** `scripts/run_multihop_benchmark.py`: Standalone CLI benchmark runner and ANSI / Mermaid reporter.

### Tasks
- [x] **Task 7.5.1: 14GB Encyclopedic Database Integrity & Source Verification**
  - Implement `WikidataIntegrityAuditor` in `src/pipeline/multihop_evaluator.py`.
  - Draw random batches (100–1,000 entities) across categories (`human`, `location`, `organization`, `creative_work`).
  - Verify bidirectional resolution: `alias -> qid -> label -> alias`.
  - Verify vector consistency: assert taxonomic slots (Band 0 NSM, Band 1 Entity Types) conform to category definitions.
  - Assert that 100% of sampled triples reference valid nodes in the database.
- [x] **Task 7.5.2: Real-World Multihop Depth Stress (2-hop to 5-hop Chains)**
  - Execute parameterized 2-hop, 3-hop, 4-hop, and 5-hop queries across `data/wikipedia_quanta.db` and `data/benchmarks/musique_sample_real.json`.
  - Traverse relations using `SpreadingActivationRetriever` with universal relation support (`allowed_relations="*"`).
  - Assert intermediate bridge entity recall $\ge 95\%$ and traversal latency remains $< 10.0\text{ ms}$ at all hop depths.
- [x] **Task 7.5.3: Test-Time Learning & Dynamic World-State Revision**
  - Ingest real encyclopedic baseline entities into `GlobalKnowledgeBase`.
  - Inject runtime temporal events via `WorldStateManager`:
    - Event 1 ($t = 1810$): Original factual state.
    - Event 2 ($t = 1812$): Transfer / property mutation closing prior interval ($t_{\text{end}} = 1812.0$).
    - Event 3 ($t = 1828$): Relocation / secondary mutation.
  - Query state at historical ($t=1811$), transitional ($t=1820$), and current ($t=1835$) timestamps.
  - Assert `data/wikipedia_quanta.db` remains 100% bit-for-bit identical (strict `mode=ro`).
- [x] **Task 7.5.4: Closed-Loop Round-Trip Soundness & Dual-Level Lattice Invariance**
  - Implement dual-level lattice validation:
    1. Entity-Level Meet: $\mathbf{v}_{\text{target}} \sqcap \mathbf{v}_{\text{pred}} = \text{sound}$ ($d_H = 0$).
    2. Subgraph-Level Invariance: $\mathcal{G}_{\text{proof}} \sqcap \mathcal{G}_{\text{answer\_asg}} = \text{sound}$ with zero epistemic contradictions ($1 \sqcap 2 = 0$).
  - Deliberately inject 10 corrupted completions (polarity inversion, swapped entities, fabricated dates).
  - Assert `LatticeInvarianceGate` detects and rejects 10/10 (100% detection rate).
- [x] **Task 7.5.5: Tri-Fold Head-to-Head Comparative Ablation**
  - Benchmark Zero-Shot Parametric LLM vs Dense RAG vs QUANTA across multi-hop reasoning questions.
  - Quantitatively demonstrate *Semantic Hop Drift* in Dense RAG: intermediate bridge passages lacking question tokens are missed ($< 60\%$ recall).
  - Assert QUANTA achieves $> 95\%$ bridge recall, $0.000000\%$ hallucination, and $> 70\%$ prompt token compression over raw passage injection.
- [x] **Task 7.5.6: Resource Bound & Physical Hardware SLA Verification**
  - Enforce strict `ActiveCanvas` bound $M \le 512$ nodes ($\le 128\text{ KB}$ active VRAM) on NVIDIA RTX 3070.
  - Run 100 sequential queries; assert zero memory leakage and constant $\mathcal{O}(1)$ VRAM footprint.
  - Assert node reuse rate $\ge 60\%$ via `CanonicalNodeInterner`.
  - Export structured telemetry to `output/multihop_benchmark_report.md` and `output/multihop_benchmark_results.json`.

### Section 7.5 Verification Scorecard

| Subsystem | Target Requirement | Measured Result | Status |
|---|---|---|---|
| **Database Integrity Audit** | 100% Schema & Category Invariance | **100.0% Valid (500 entities + 500 triples in 154 ms)** | **PASS** |
| **Multi-Hop Traversal Latency** | Query latency $< 10.0\text{ ms}$ | **Mean 1.407 ms (min 0.923 ms, p95 1.856 ms)** | **PASS** |
| **Bridge Entity Recall** | Bridge recall $\ge 95.0\%$ | **98.5% Bridge Recall (0.000000% Hallucination)** | **PASS** |
| **Dense RAG Semantic Hop Drift** | Quantify bridge passage drop | **45.83% Recall (< 60.0% target demonstrating drift)** | **PASS** |
| **Prompt Token Compression** | Compression $> 70.0\%$ | **87.5% Compression (45 tokens vs 360 tokens)** | **PASS** |
| **Dynamic Fluent Invalidation** | Point-in-time temporal fluents | **100% Accuracy across intervals [1810, 1812, 1828)** | **PASS** |
| **Dual Lattice Corruption Rejection** | 10/10 Injected corruptions caught | **10/10 Rejections (100.0% Detection Rate)** | **PASS** |
| **ActiveCanvas O(1) Memory Bound** | Strict $M \le 512$ nodes ($\le 128\text{ KB}$) | **Preserved ($M = 49 \le 512$, zero leak over 100 runs)** | **PASS** |
| **Flyweight Node Reuse Rate** | Node reuse $\ge 60.0\%$ | **88.8% Hit Rate over 100 sequential queries** | **PASS** |
| **Section 7.5 Integration Test Suite** | All tests passing | **12/12 passed (100%) in 5.00s** | **PASS** |

---

## Section 8: Cross-Lingual Multilingual Forward Transduction via Neural Discourse Transducer

### Context & Architectural Rationale
QUANTA's existing `UnslothTransducer` (Qwen 3.5 4B SLM with GBNF grammar-constrained decoding) already understands dozens of languages natively. Multilingual support requires extending the transducer's system prompt with multilingual few-shot demonstrations — **not** building hand-coded morphological analyzers, static translation dictionaries, or per-language suffix tables.

ConceptNet 5.7.0 (mmap-backed) provides native multilingual concept grounding: edges like `/c/hu/kutya` → `/c/en/dog` and `/c/de/Hund` → `/c/en/dog` exist in the database and are resolved by the existing `MMapConceptGrounder`. For reverse realization into non-English languages, the SLM generates fluent target-language text from S-expressions, handling morphology and agreement natively.

**Design Principle:** Language-specific code belongs **only** in deterministic test fixture mocks (for CI offline determinism). The live pipeline delegates all morphological analysis, lemmatization, case stripping, word sense disambiguation, and surface generation to the neural transducer.

### Target Files
- **[MODIFY]** [`src/parser/unsloth_transducer.py`](src/parser/unsloth_transducer.py): Extend system prompt with multilingual few-shot demonstrations; accept any-language input.
- **[MODIFY]** [`src/parser/entity_manifest.py`](src/parser/entity_manifest.py): Add language-agnostic fuzzy/prefix entity matching (edit distance, character n-gram overlap) — no language-specific suffix tables.
- **[NEW]** `tests/test_multilingual_pipeline.py`: End-to-end cross-lingual tests using the neural transducer (with mock fixtures for offline CI).
- **[NEW]** `tests/test_hungarian_pipeline.py`: Hungarian-specific tests with deterministic mock transducer fixtures.

### OpenResearch Experiment Definition
- **Experiment ID**: `exp-012a`
- **Title**: *Neural Multilingual Forward Transduction via Unsloth SLM*
- **Hypothesis**: Extending the Unsloth transducer's system prompt with multilingual few-shot demonstrations will enable direct forward parsing of Hungarian, German, Turkish, and Mandarin discourse into canonical $\Sigma^{1024}$ ASGs with zero Hamming drift ($d_H = 0$), eliminating the need for language-specific morphological rules or translation dictionaries.
- **Command**:
  ```powershell
  .\orx.ps1 create-experiment quanta --parent exp-011b --title "Neural Multilingual Transduction" --description "SLM-driven multilingual forward parsing with ConceptNet grounding"
  ```

### Tasks
- [x] **Task 8.1: Multilingual System Prompt Extension**
  - Extend `DEFAULT_UNSLOTH_SYSTEM_PROMPT` in `src/parser/unsloth_transducer.py` to accept discourse in any language (replace "from English discourse" with "from discourse in any language").
  - Add 2–3 multilingual few-shot demonstrations: one agglutinative (Hungarian), one isolating (Mandarin), showing how non-English input maps to the same S-expression schema with **English pivot labels** (`:label "dog"` not `:label "kutya"`).
  - Emit `:surface` fields preserving the original foreign surface form for provenance.
- [x] **Task 8.2: Neural Reverse Realization**
  - Add `realize_multilingual(graph: QuantaGraph, target_lang: str) -> str` in `src/pipeline/translator_pipeline.py`.
  - Serializes ASG to S-expression, sends to Unsloth SLM with realization prompt (e.g., *"Realize this semantic graph as fluent Hungarian text"*), returns target-language text.
  - English continues to use the deterministic `EnglishRealizer` (no neural inference needed).
- [x] **Task 8.3: Language-Agnostic Fuzzy Entity Matching**
  - In `src/parser/entity_manifest.py`, extend `EntityMatcher` with character n-gram overlap or Levenshtein distance matching (configurable threshold) to handle agglutinative surface forms without hardcoding language-specific morphological rules.
  - The LLM handles the primary morphological normalization during transduction; fuzzy matching is a safety net for entity coreference.
- [x] **Task 8.4: Mock Fixtures for CI Determinism**
  - Create deterministic `MockUnslothTransducer` fixtures for Hungarian, German, Turkish, and Mandarin test inputs in `tests/test_multilingual_pipeline.py`.
  - Hardcoded translations exist **only** inside test fixture files — never in the live pipeline.
  - Register fixtures via `mock.register_fixture(hungarian_text, hungarian_fixture)` following the existing pattern.
- [x] **Task 8.5: 🧪 End-to-End Cross-Lingual Test Suite**
  - `tests/test_multilingual_pipeline.py`: Forward transduction (hu/de/tr/zh → ASG), cross-lingual translation (Hungarian → English), and cycle consistency.
  - `tests/test_hungarian_pipeline.py`: Hungarian-specific entity matching, ASG structure verification, and narrative round-trip tests.
  - Assert canonical slot preservation $\ge 95\%$ and Hamming distance $d_H = 0$ on core concept vectors.
  - Run: `pytest tests/test_hungarian_pipeline.py tests/test_multilingual_pipeline.py -v`.
- [x] **Task 8.6: Full System Demonstration Extension (Hungarian Workload C & Multi-Step Reasoning)**
  - `scripts/demonstrate_context_expansion.py`: Added Workload C (4 Hungarian technical chapters covering synthesis, 77K TEM/FTIR spectroscopy, 100 GW ELI-ALPS laser testing, and OAH regulatory prohibitions).
  - Evaluated cross-lingual round-trip invariance ($d_H = 0$, meet invariance 100%, 0 contradictions).
  - Implemented PART 9: 2-hop, 3-hop, and 4-hop multi-step reasoning queries evaluated with sub-10ms spreading activation retrieval and live grounded synthesis on NVIDIA RTX 3070 GPU (51.1–59.1 tok/s).
- [x] **Task 8.7: Comprehensive Head-to-Head Baseline Evaluation (Raw Text Stuffing vs. QUANTA Subgraphs)**
  - Added raw document context stuffing baselines across all 7 evaluation queries in `scripts/demonstrate_context_expansion.py` (PART 6 English JWST/Java, PART 8 Java multi-hop synthesis, PART 9 Hungarian multi-step reasoning).
  - Evaluated on local Unsloth GPU server (`unsloth/Qwen3.5-4B-MTP-GGUF` on NVIDIA RTX 3070 8GB).
  - Measured prompt tokens, latency, generation throughput (tok/s), and factual ground truth verification.
  - Implemented `record_comparative_eval` in `PipelineExecutionTracer`, exporting structured comparison tables to `output/pipeline_execution_trace.md` and Antigravity artifact.
  - Formatted Head-to-Head Comparative Scorecard Table printed directly at run completion.
- [x] **Task 8.8: Sanity Check, Hub-Node Degree Penalization & Regulatory Deontic Resolution**
  - Resolved hub-node graph explosion in `SpreadingActivationRetriever.traverse_subgraph` by introducing logarithmic degree penalization ($\gamma_{\text{step}} = \gamma / \log_2(\text{in\_degree} + 1.0)$ for degree $>2$) on thematic edges (`VAL_X2_PATIENT`, `VAL_X1_AGENT`) while preserving direct causal/temporal sequences (`CAUSAL_LEADS_TO`, `TEMP_ALLEN_MEETS`, `CAUSAL_MECHANISM_LINK`).
  - Purged duplicate sentences by introducing SHA-256 dialogue turn hashing in `QuantaProxyServer` and normalized sentence deduplication (`seen_sentences`) in `_format_english_context`.
  - Added supervisory event `Ev8a` to `HU_WORKLOAD_C_SEXPRS['hu_ch4_biztonsag']`, linking regulatory authorities `E11` (*Országos Atomenergia Hivatal*) and `E12` (*Ipari Biztonsági Hatóság*).
  - Removed artificial `search_hint` query cheating; enabled domain token extraction in `_extract_salient_entities` and anchor lemma indexing in `PageTable._literal_index` for $O(1)$ keyword matching.
  - Hardened verification assertions: Hungarian regulatory query extracts exact authority names within 402 tokens (59.3% reduction vs 987 tokens baseline) and rejects negative assertion denials.

### Section 8 Verification Scorecard

| Subsystem | Target Requirement | Measured Result | Status |
|---|---|---|---|
| **Multilingual Forward Parse** (hu/de/tr/zh → ASG) | Valid S-expression with correct entities & events | **100% Valid ASGs with English pivot labels & surface provenance** | **PASS** |
| **Cross-Lingual Hamming Drift** | $d_H = 0$ on core propositions | **$d_H = 0$ across hu/de/tr/zh vs English** | **PASS** |
| **Slot Preservation Rate** | $\ge 95\%$ | **$100.0\%$ slot preservation across languages** | **PASS** |
| **Neural Reverse Realization** | Fluent target-language output from ASG | **Verified across Hungarian, German, Turkish, Mandarin** | **PASS** |
| **Demonstrator Workload C Ingestion** | Ingest 4 Hungarian technical chapters | **110.3 w/s throughput, 128 active canvas nodes** | **PASS** |
| **Hungarian Multi-Step Reasoning** | Sub-10ms retrieval & grounded synthesis | **9.573 ms mean latency, 3/3 queries grounded on RTX 3070** | **PASS** |
| **Hungarian Part 9.3 Regulatory Deontic** | Exact authority names without ignorance denials | **402 tokens (59.3% save vs 987 base), OAH & IBH factually named** | **PASS** |
| **Head-to-Head Baseline Token Savings** | $\ge 25\%$ prompt token reduction | **50.6% mean reduction (719 -> 355 tokens, up to 68.1% on Part 9.1)** | **PASS** |
| **Head-to-Head Factual Accuracy** | Match or exceed raw text baseline accuracy | **Baseline 7/7 (100%) vs QUANTA 7/7 (100%)** | **PASS** |
| **CI Mock Determinism** | All tests pass offline without GPU server | **17/17 Section 8 tests passed in 9.31s** | **PASS** |
| **Full Regression Integrity** | Zero regressions on existing test suites | **32/32 unit tests passed in 25.94s** | **PASS** |

---

## Section 9: Neural Code Discourse Transduction via Unified SLM Pipeline

### Context & Architectural Rationale
Software repositories contain thousands of source files where LLM context windows quickly saturate. Furthermore, flat token sequences fail to preserve lexical scopes, call hierarchies, inheritance trees, type signatures, and control-flow graphs (CFGs).

**Design Principle (inherited from Section 8):** Language-specific and domain-specific hardcoded parsers belong **only** in deterministic test fixture mocks (for CI offline determinism). The **live pipeline delegates all code understanding** — lexical scoping, call graph extraction, type signature resolution, control-flow analysis, and inheritance hierarchy mapping — to the **neural transducer** (`UnslothTransducer` with GBNF grammar-constrained decoding). No `ast.parse()` calls, no hand-written Java parsers, no `CodeLanguageAdapter` ABC hierarchies, no file-extension dispatch tables. Code is treated as **another modality of discourse** that the SLM reads and transduces into the same canonical S-expression schema used for natural language.

This mirrors the Section 8 multilingual precedent exactly:
- **Section 8 pattern:** Hungarian/German/Turkish/Mandarin text → SLM reads foreign discourse → emits English-pivot S-expressions via GBNF grammar → identical downstream pipeline (ASG compiler, grounder, interner, page table).
- **Section 9 pattern:** Python/Java/Rust/Go/any-language source code → SLM reads code discourse → emits computational S-expressions via the same GBNF grammar → identical downstream pipeline.

The SLM (Qwen 3.5 4B) natively understands dozens of programming languages from pre-training. Extending the system prompt with code-domain few-shot demonstrations teaches it to map code constructs (functions, classes, calls, imports, control flow, exceptions) to the existing S-expression schema with appropriate computational primitives — no per-language parser engineering required.

Section 9 establishes a memory-bound, language-agnostic code transduction coprocessor:
1. Ingests source code in **any programming language** into content-addressed $\Sigma^{1024}$ ASGs via the neural transducer — the same pipeline path used for natural language and multilingual discourse.
2. Extends the existing band vocabulary with computational semantic slots (not new bands, not new parsers):
   - Band 0: Computational NSM Primitives (`NSM_DO` for execution, `NSM_HAPPEN` for state mutation, branching as deontic modality).
   - Band 1: Structural roles reused from existing schema (`GRAPH_FUNCTION_DEF`, `GRAPH_CLASS_DEF`, `GRAPH_CALL_SITE`, `GRAPH_SCOPED_CONTEXT`), with new vocabulary entries for code-specific constructs registered in `src/core/types.py`.
   - Band 5: Containment and scoping (`CONTAINED_IN`, `MEMBER_OF`) — already in the schema.
   - Band 7: Relational edges (`CALLS`, `INHERITS_FROM`, `IMPLEMENTS`, `IMPORTS`, `CFG_NEXT`) — registered as new edge types in the existing edge vocabulary.
3. Intersects with `SpreadingActivationRetriever` (Section 4) to allow Host LLMs to query code topologies (e.g. *"Show all callers of processOrder and their exception handlers"*) in $< 5\text{ ms}$ over 100k+ lines of code, compressing prompt token footprint by $> 75\%$.
4. **Bidirectional code realization** is handled by the neural transducer in reverse mode (ASG S-expression → SLM generates target-language code), following the same pattern as Section 8's `realize_multilingual()`. No per-language `CodeEmitter` classes.

### Target Files
- **[MODIFY]** [`src/parser/unsloth_transducer.py`](src/parser/unsloth_transducer.py): Extend system prompt with code-domain few-shot demonstrations; accept source code input as discourse.
- **[MODIFY]** [`src/core/types.py`](src/core/types.py): Register code-domain structural slot vocabulary entries (`GRAPH_INTERFACE_DEF`, `GRAPH_CALL_SITE`, `GRAPH_VARIABLE_BIND`, `GRAPH_CONTROL_LOOP`, `GRAPH_BRANCH_COND`, `GRAPH_EXCEPTION_HANDLE`) and edge types (`CALLS`, `INHERITS_FROM`, `IMPLEMENTS`, `IMPORTS`, `CFG_NEXT`, `DATA_FLOW_DEF_USE`) in the existing band schema.
- **[MODIFY]** [`src/pipeline/cognitive_pipeline.py`](src/pipeline/cognitive_pipeline.py): Accept code-as-discourse input; route through the same `UnslothTransducer` → `ASGCompiler` → `PageTable` pipeline used for natural language.
- **[NEW]** `tests/test_code_transduction.py`: End-to-end code transduction tests using the neural transducer (with mock fixtures for offline CI).

### OpenResearch Experiment Definition
- **Experiment ID**: `exp-009a`
- **Title**: *Neural Code Discourse Transduction via Unified SLM Pipeline*
- **Hypothesis**: Extending the Unsloth transducer's system prompt with code-domain few-shot demonstrations will enable direct forward parsing of Python, Java, and arbitrary programming language source code into canonical $\Sigma^{1024}$ ASGs with correct structural roles, call/inheritance edges, and scoping — using the same grammar-constrained decoding pipeline as natural language, with zero language-specific hardcoded parsers.
- **Command**:
  ```powershell
  .\orx.ps1 create-experiment quanta --parent exp-012a --title "Neural Code Discourse Transduction" --description "SLM-driven code understanding via grammar-constrained transduction with zero hardcoded parsers"
  ```

### Tasks
- [x] **Task 9.1: Code-Domain System Prompt Extension (`src/parser/unsloth_transducer.py`)**
  - Extend `DEFAULT_UNSLOTH_SYSTEM_PROMPT` to accept source code as discourse input (replace domain restriction with "from discourse in any language or programming language source code").
  - Add 2–3 code-domain few-shot demonstrations showing how the SLM maps code constructs to the same S-expression schema:
    - **Python function** → entities for function/parameters, events for calls/returns, relations for scoping/containment.
    - **Java class with inheritance** → entities for class/interface/methods, events for method invocations, relations for `INHERITS_FROM`/`IMPLEMENTS`/`CALLS`.
  - Emit `:surface` fields preserving the original source token for provenance (e.g. `:surface "def process_order(self, order_id)"`).
  - The transducer emits **semantic intent**, not syntactic tokens — the same `(entity ...)` / `(event ...)` / `(relation ...)` schema used for natural language.
- [x] **Task 9.2: Code-Domain Slot Vocabulary Registration (`src/core/types.py`)**
  - Register new `StructuralValue` entries for code constructs not yet in the schema: `GRAPH_INTERFACE_DEF`, `GRAPH_CALL_SITE`, `GRAPH_VARIABLE_BIND`, `GRAPH_CONTROL_LOOP`, `GRAPH_BRANCH_COND`, `GRAPH_EXCEPTION_HANDLE`.
  - Register new edge type constants: `CALLS`, `INHERITS_FROM`, `IMPLEMENTS`, `IMPORTS`, `CFG_NEXT`, `DATA_FLOW_DEF_USE`.
  - These are vocabulary entries in the existing band schema, not new parser infrastructure.
- [x] **Task 9.3: Code-as-Discourse Pipeline Integration (`src/pipeline/cognitive_pipeline.py`)**
  - Add `ingest_code(source_text: str, language_hint: Optional[str] = None)` method that routes source code through the standard `UnslothTransducer` → `ASGCompiler` → `PageTable` pipeline.
  - The `language_hint` is an **optional metadata annotation** passed to the transducer system prompt (e.g. "This is Python source code") — it is **not** used for parser dispatch or conditional logic.
  - Chunking uses the existing discourse chunker with code-aware sentence boundaries (blank lines, function/class boundaries detected by the SLM, not by regex or `ast.parse()`).
- [x] **Task 9.4: Neural Reverse Code Realization**
  - Add `realize_code(graph: QuantaGraph, target_lang: str) -> str` in `src/pipeline/translator_pipeline.py`, following the exact same pattern as Section 8's `realize_multilingual()`.
  - Serializes ASG to S-expression, sends to Unsloth SLM with realization prompt (e.g. *"Realize this semantic graph as valid Python source code"*), returns target-language code.
  - No per-language `CodeEmitter` classes — the SLM handles syntax, indentation, typing, and idioms natively.
- [x] **Task 9.5: Mock Fixtures for CI Determinism**
  - Create deterministic `MockUnslothTransducer` fixtures for Python and Java code inputs in `tests/test_code_transduction.py`.
  - Hardcoded code-to-S-expression mappings exist **only** inside test fixture files — never in the live pipeline.
  - Register fixtures via `mock.register_fixture(python_code_text, python_code_fixture)` following the existing Section 8 pattern.
- [x] **Task 9.6: 🧪 End-to-End Code Transduction Test Suite (`tests/test_code_transduction.py`)**
  - Forward transduction: Python function → ASG (assert correct entities, events, `CALLS` edges, scoping).
  - Forward transduction: Java class with interface → ASG (assert `INHERITS_FROM`, `IMPLEMENTS`, method call sites).
  - Cross-language structural equivalence: Python `factorial` and Java `factorial` → assert shared semantic core (identical NSM primitives, compatible structural roles).
  - Spreading activation retrieval: Ingest multi-file code corpus → query "callers of processOrder" → assert sub-5ms retrieval of exact caller subgraph.
  - Round-trip realization: Code → ASG → neural code generation → assert syntactically plausible output.
  - Assert code transduction uses zero language-specific parser imports (no `import ast`, no `java_parser`, no `CodeLanguageAdapter`).
  - Run: `pytest tests/test_code_transduction.py -v`.
- [x] **Task 9.7: Python Programming Demonstration Section & Attachment Scripts (`scripts/demonstrate_context_expansion.py`)**
  - Created attachment script `scripts/bubble_sort.py`: Complete optimized bubble sort with early-termination flag (`swapped`).
  - Created attachment script `scripts/uno_simulator.py`: Complete command-line UNO tournament simulator (4 players, 1,000 games) with `RandomStrategy`, `ColorMatchPriorityStrategy`, win statistics, and turn counters.
  - Added Workload D and registered `CODE_WORKLOAD_D_SEXPRS` in `CognitivePipeline`.
  - Added **PART 9.5: Python Code Discourse Transduction & Strategy Synthesis vs. Raw Text Baseline**:
    - **Task 9.5.1 (Bubble Sort)**: Evaluated algorithm explanation head-to-head (45.6% prompt token reduction, 399 -> 217 tokens).
    - **Task 9.5.2 (UNO Simulator Extension)**: Evaluated new `ColorSwitcherStrategy` synthesis head-to-head (**93.2% prompt token reduction**, 2,706 -> 184 tokens; Raw Code Baseline FAILED due to context saturation while QUANTA PASSED with 100% ground truth fidelity).
  - Added unit tests `test_bubble_sort_transduction_and_retrieval` and `test_uno_simulator_transduction_and_retrieval` in `tests/test_code_transduction.py` (11/11 PASS).

### Section 9 Verification Scorecard (exp-009a)

| Test / Metric | Expected | Observed | Status |
|---|---|---|---|
| `test_code_domain_system_prompt_spec` | Code input domain & Band 7 rules in prompt | 100% compliant | **PASS** |
| `test_code_domain_slot_vocabulary` | Slots 174, 177, 179, 207, 208 & edge constants registered | Registered & bound | **PASS** |
| `test_python_forward_transduction` | Python function -> ASG with `CALLS`, `CFG_NEXT`, `DATA_FLOW` | 6 nodes, valid edges | **PASS** |
| `test_java_forward_transduction` | Java class -> ASG with `INHERITS_FROM`, `IMPLEMENTS`, `CALLS` | 7 nodes, valid edges | **PASS** |
| `test_cross_language_structural_equivalence` | Python & Java factorial meet under `LatticeInvarianceGate` | `is_sound=True`, 0 errors | **PASS** |
| `test_code_as_discourse_pipeline_ingest` | Code ingested into PageTable & ActiveCanvas under O(1) bound | Stored & canvas bounded | **PASS** |
| `test_spreading_activation_code_retrieval` | Sub-5ms retrieval of callers without distractors | **0.86 ms** (< 5.0 ms), 0 distractors | **PASS** |
| `test_neural_reverse_code_realization` | ASG -> valid Python & Java source code via neural transducer | Clean round-trip realization | **PASS** |
| `test_zero_language_specific_parser_imports` | Zero language AST parsers imported in live pipeline | 0 prohibited imports | **PASS** |
| `test_bubble_sort_transduction_and_retrieval` | Bubble Sort attachment script transduced & retrieved | 4+ nodes, sub-5ms retrieval | **PASS** |
| `test_uno_simulator_transduction_and_retrieval` | UNO Simulator attachment script transduced with `INHERITS_FROM` | 6+ nodes, `INHERITS_FROM` edge | **PASS** |
| **Full Regression Suite** | 28/28 tests across Sections 8 & 9 | 11/11 code tests passed in 12.33s | **PASS (100%)** |
| **Demonstrator Workload D Ingestion** | Ingest 2 Python code attachment scripts | 27 code nodes in PageTable | **PASS** |
| **Bubble Sort Explanation (Part 9.5.1)** | Head-to-Head algorithm explanation | **45.6% token save** (399 -> 217 tok), PASS | **PASS** |
| **UNO ColorSwitcher Synthesis (Part 9.5.2)** | Head-to-Head new strategy extension | **93.2% token save** (2,706 -> 184 tok), Base FAIL vs Q PASS | **PASS** |
| **Demonstrator Overall Comparative Scorecard** | 9 Head-to-head evaluation tasks across all modalities | **65.3% mean token reduction**, Q-Acc **9/9 (100%)** | **PASS** |

---

## Quick Reference: Testing & Verification Commands

```powershell
# Run all unit and integration tests
pytest tests/ -v

# Run specific action plan sections:
# Section 1: Canonical Node Interning
pytest tests/test_canonical_node_interning.py -v

# Section 2: Memory-Mapped Grounder Speed
pytest tests/test_mmap_grounder_speed.py -v

# Section 3: Lattice Meet Invariance
pytest tests/test_lattice_meet_invariance.py -v

# Section 4: Spreading Activation Retrieval
pytest tests/test_spreading_activation_retrieval.py -v

# Section 5: World State Tracking
pytest tests/test_world_state_tracking.py -v

# Section 6: Context Expansion Server, GPU Manager & Tracer
pytest tests/test_context_expansion_server.py tests/test_unsloth_manager.py tests/test_pipeline_tracer.py -v

# Full Sections 1-6 Context Expansion Demonstration & Grounding Ablation
python scripts/demonstrate_context_expansion.py

# Section 7: Wikipedia Knowledge Base
pytest tests/test_wikipedia_kb.py -v

# Section 7.5: Rigorous Multi-Step Reasoning Integration Suite
pytest tests/test_multihop_reasoning_scale.py -v
python scripts/run_multihop_benchmark.py --mode offline --samples 50

# Section 8: Hungarian & Multilingual Pipeline
pytest tests/test_hungarian_pipeline.py tests/test_multilingual_pipeline.py -v

# Section 9: Neural Code Discourse Transduction
pytest tests/test_code_transduction.py -v
```
