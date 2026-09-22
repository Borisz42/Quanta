# QUANTA Context Expansion Master Action Plan & Engineering Roadmap (`CONTEXT_EXPANSION_ROADMAP.md`)

> **Document Status:** Active Master Action Plan  
> **Repository:** `Borisz42/Quanta`  
> **Target Role:** Verifiable, Memory-Bound Neuro-Symbolic LLM Context Expansion Coprocessor  
> **Hardware Target:** NVIDIA RTX 3070 (8GB VRAM) + 16GB Host RAM  
> **Autonomous Research Protocol:** Governed by OpenResearch (`.\orx.ps1`), [`AGENTS.md`](file:///c:/Users/PC/Documents/GitHub/Quanta/AGENTS.md), and [`EVAL.md`](file:///c:/Users/PC/Documents/GitHub/Quanta/EVAL.md)  

---

## Instructions for Future Agent Sessions

This document is organized into **8 independent, modular sections** with numbered task IDs (`Task X.Y`) and checkboxes (`- [ ]`). 
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
- [Section 8: Cross-Lingual Multilingual Forward Transduction Adapters](#section-8-cross-lingual-multilingual-forward-transduction-adapters-universal-pivot) *(Universal Non-English Parsing)*

---

## Section 1: Canonical Node Interning & Global Hash-Consing (Memory Efficiency & Node Reuse)

### Context & Architectural Rationale
Currently, in [`src/parser/asg_compiler.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/asg_compiler.py#L317-L318), entity compilation sets `reg_slot = f"VAR_SLOT_X{register_index % 8}"` directly into the 1024-D quaternary vector. This tightly couples an entity's ephemeral position in a single chunk extraction to its immutable packed vector. If `"synthetic compound"` appears as entity 0 in Chunk 1, its vector activates `VAR_SLOT_X0`; if it appears as entity 2 in Chunk 2, its vector activates `VAR_SLOT_X2`. Consequently, the 256-byte vector changes, its BLAKE3 CID changes, and redundant nodes are allocated across chapters and translation round-trips.

To achieve maximum memory efficiency and node reuse:
1. **Decouple Ephemeral Execution Registers from Node Identity**: Band 2 variable registers (`VAR_SLOT_X0`..`X7`) represent active canvas execution state, not permanent ontological meaning. The node's canonical CID must be computed over its semantic and structural bands (Bands 0, 1, 3–7), while register assignments are tracked in execution context or masked during canonical hashing.
2. **Canonical Node Interning Pool (`CanonicalNodeInterner`)**: Implement a global Flyweight hash-consing pool. When the compiler or translator resolves a concept or named entity, it checks the interner. If an identical concept exists, it returns the interned node pointer rather than instantiating a new object.

### Target Files
- **[MODIFY]** [`src/core/asg.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/core/asg.py): Add register-masked CID computation and node interning hook.
- **[NEW]** `src/memory/node_interner.py`: Implement thread-safe `CanonicalNodeInterner` with weak references and SQLite persistence.
- **[MODIFY]** [`src/parser/asg_compiler.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/asg_compiler.py): Separate persistent concept vectors from ephemeral register assignments; integrate with interner.
- **[MODIFY]** [`src/parser/graph_stitcher.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/graph_stitcher.py): Reuse interned nodes when consolidating global identities.
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
- **[MODIFY]** [`src/parser/asg_compiler.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/asg_compiler.py): Route lookups to `MmapLexicalGrounder`.
- **[MODIFY]** [`src/pipeline/cognitive_pipeline.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/pipeline/cognitive_pipeline.py): Implement asynchronous producer-consumer pipeline.
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
In [`docs/publication.md` Section 8.3](file:///c:/Users/PC/Documents/GitHub/Quanta/docs/publication.md#83-information-profiler-audit-and-verifier-diagnostics), the empirical cycle-consistency fidelity was measured at 89.5% with a translator Macro F1 of 0.467 on complex logical sentences. The primary cause is the **"NSM Explication Gap"**: when complex verbs (e.g. *"purchased"*, *"prohibited"*, *"transformed"*) are encountered, the parser extracts simple valency frames without decomposing them into their primitive NSM state transitions. Furthermore, reverse English realization can suffer from anaphoric ambiguity across long sentences.

By introducing an automated **Closed-Loop Round-Trip Lattice Meet Gate** ($\mathbf{v}_{\text{orig}} \sqcap \mathbf{v}_{\text{reparsed}}$) and expanding predicates into standard 3-prime NSM explication scripts, cycle-consistency Macro F1 will increase to $> 0.75$ with zero semantic drift.

### Target Files
- **[NEW]** `src/verification/lattice_gate.py`: `LatticeInvarianceGate` computing lattice meet consistency.
- **[MODIFY]** [`src/parser/asg_compiler.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/asg_compiler.py): Add multi-hop NSM explication script expansion.
- **[MODIFY]** [`src/realizer/english_nlg.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/realizer/english_nlg.py): Enhanced anaphora and determiner preservation.
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
When an external LLM needs context to answer a user prompt, passing the entire 100k-node book or document graph is impossible. Currently, [`CognitivePipeline.answer_query()`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/pipeline/cognitive_pipeline.py#L423-L466) only performs a flat scan or active canvas lookup.

A true context expansion engine requires **Query-Driven Spreading Activation**:
1. Transduce the incoming question into an ASG query pattern with `GRAPH_QUERY_TARGET=3` in Band 1 and `QUERY_TARGET_?X` in Band 2.
2. Retrieve the top-$K$ seed nodes from `PageTable` via AVX-512 SIMD bitwise Hamming distance ($< 5\text{ ms}$).
3. Spread activation along Band 1 thematic valencies (`VAL_X1_AGENT`, `VAL_X2_PATIENT`) and Band 7 causal/temporal links up to depth $d=2$.
4. Extract the minimal causal subgraph and serialize it into concise prose or dense S-expressions for injection into the host LLM prompt.

### Target Files
- **[NEW]** `src/memory/spreading_activation.py`: `SpreadingActivationRetriever` class.
- **[MODIFY]** [`src/pipeline/cognitive_pipeline.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/pipeline/cognitive_pipeline.py): Connect `answer_query` and add `retrieve_context`.
- **[NEW]** `tests/test_spreading_activation_retrieval.py`: Retrieval accuracy and latency tests.

### Tasks
- [ ] **Task 4.1: Query ASG Transduction**
  - Implement `compile_query_asg(query_text: str) -> QuantaGraph` in `src/memory/spreading_activation.py`.
  - Mark target question words (*"who"*, *"what"*, *"where"*, *"why"*) with `GRAPH_QUERY_TARGET=3` and assign `QUERY_TARGET_?X`.
- [ ] **Task 4.2: SIMD Top-K Seed Selection**
  - Use `SimdHammingIndex.search(query_vec, top_k=5)` to identify initial seed CIDs in host memory in $< 1\text{ ms}$.
- [ ] **Task 4.3: Implement Spreading Activation Traversal**
  - Implement `traverse_subgraph(seed_cids: List[str], page_table: PageTable, max_depth: int = 2, decay: float = 0.7) -> QuantaGraph`.
  - Traverse edges: `VAL_X1_AGENT`, `VAL_X2_PATIENT`, `VAL_LOCATION_SLOT`, `CAUSAL_MECHANISM_LINK`, `TEMP_ALLEN_MEETS`.
  - Collect all activated nodes above threshold $\theta = 0.35$.
- [ ] **Task 4.4: Dynamic Context Builder for Host LLMs**
  - Implement `format_context_for_llm(subgraph: QuantaGraph, format: str = "english", max_tokens: int = 500) -> str`.
  - Supports format `"english"` (honest NLG sentences) and format `"sexpr"` (compact GBNF S-expressions).
- [ ] **Task 4.5: 🧪 Write Retrieval Accuracy Tests (`tests/test_spreading_activation_retrieval.py`)**
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
- **[MODIFY]** [`src/parser/graph_stitcher.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/graph_stitcher.py): Hook world-state updates during chunk stitching.
- **[NEW]** `tests/test_world_state_tracking.py`: Dynamic state update and temporal query tests.

### Tasks
- [ ] **Task 5.1: Implement Fluent State Representation (`src/memory/world_state.py`)**
  - Define `EntityStateRecord` tracking `entity_cid`, `property_slot` (e.g. `VAL_LOCATION_SLOT`), `value_cid`, `t_start`, `t_end`, and `epistemic_status`.
- [ ] **Task 5.2: Non-Monotonic Transition Resolver**
  - In `WorldStateManager.update_from_event(event_node, graph)`:
    - If an event asserts a mutually exclusive property (e.g. moving to a new location or phase transition), find existing active state record ($t_{\text{end}} = \text{None}$).
    - Close the prior state: set $t_{\text{end}} =$ `event.time_start` and wire `TEMP_ALLEN_FINISHES` edge.
    - Open new state: set $t_{\text{start}} =$ `event.time_start`.
- [ ] **Task 5.3: Temporal Point-in-Time State Queries**
  - Implement `get_entity_state_at(entity_cid: str, property_name: str, timestamp: float) -> Optional[QuantaNode]`.
  - Allows answering queries like: *"Where was the specimen at 11:00 AM?"* vs *"Where is it now?"*
- [ ] **Task 5.4: 🧪 Write World-State Tests (`tests/test_world_state_tracking.py`)**
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
- **[NEW]** `scripts/serve.ps1`: PowerShell startup script for the server.
- **[NEW]** `tests/test_context_expansion_server.py`: Integration tests with mock client requests.

### Tasks
- [ ] **Task 6.1: Implement OpenAI-Compatible Reverse Proxy (`src/server/proxy.py`)**
  - Expose `/v1/models` and `/v1/chat/completions`.
  - In `/v1/chat/completions`:
    1. Parse incoming `messages`.
    2. If total token count exceeds threshold (e.g. > 2,000 tokens), chunk and ingest prior dialogue into `CognitivePipeline`.
    3. Extract the last user message, run `SpreadingActivationRetriever.retrieve_context()`, and inject compact context as a system prompt prefix.
    4. Forward compressed request to target backend (Unsloth `:8888` or upstream provider) and stream response.
- [ ] **Task 6.2: Implement Model Context Protocol (MCP) Server (`src/server/mcp_server.py`)**
  - Implement JSON-RPC 2.0 stdio server supporting MCP specification:
    - Tool `quanta_ingest_document(doc_id: str, content: str)`: Ingests long document into PageTable.
    - Tool `quanta_query_memory(query: str, max_tokens: int = 500)`: Retrieves relevant verified subgraph context.
    - Tool `quanta_get_entity_details(name_or_cid: str)`: Returns full 1024-D vector analysis and active edges.
- [ ] **Task 6.3: Create Startup Script (`scripts/serve.ps1`)**
  - Add PowerShell script: `.\scripts\serve.ps1 -Port 8000 -Backend "http://localhost:8888/v1"`.
- [ ] **Task 6.4: 🧪 Integration Test Suite (`tests/test_context_expansion_server.py`)**
  - Start proxy test fixture, send multi-turn conversation with 5,000 words of background narrative, assert proxy compresses payload and delivers accurate context.
  - Run: `pytest tests/test_context_expansion_server.py -v`.

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
- [ ] **Task 7.1: Streaming Wikidata Dump Ingester (`src/data/wikidata_ingester.py`)**
  - Stream Wikidata JSONL dumps without loading full file into memory.
  - Filter salient entity categories: Humans (`Q5`), Locations (`Q2221906`), Organizations (`Q43229`), Creative Works (`Q386724`).
  - Filter key relations: `P31` (instance of), `P279` (subclass of), `P17` (country), `P36` (capital), `P50` (author).
  - Include an offline 100,000-entity synthetic slice generator for continuous integration testing.
- [ ] **Task 7.2: Entity & Property Mapper**
  - Convert Wikidata Q-IDs and P-ID triples into `QuantaNode` instances with 1024-D QuantaVectors and 256-bit BLAKE3 CIDs.
  - Link taxonomic types to ConceptNet 5.7.0 anchors via `ConceptNetLexicalGrounder`.
- [ ] **Task 7.3: Persistent SQLite Compiler**
  - Generate `data/wikipedia_quanta.db` with `nodes`, `aliases`, and `triples` tables, indexed with B-Trees on lowercase names and `(subject_qid, property_pid)`.
- [ ] **Task 7.4: Global Knowledge Base Mount Interface (`src/memory/global_kb.py`)**
  - Thread-safe `GlobalKnowledgeBase` class mounting `data/wikipedia_quanta.db` in read-only mode (`mode=ro`).
  - Seamlessly integrates with `PageTable` and pages queried nodes into `ActiveCanvas` on demand.
- [ ] **Task 7.5: 🧪 Write Benchmark Test (`tests/test_wikipedia_kb.py`)**
  - Mount 100k-entity slice; verify multi-hop queries execute in $< 10\text{ ms}$ with flat $\mathcal{O}(1)$ VRAM and zero hallucination.
  - Run: `pytest tests/test_wikipedia_kb.py -v`.

---

## Section 8: Cross-Lingual Multilingual Forward Transduction Adapters (Universal Pivot)

### Context & Architectural Rationale
[`docs/publication.md` Section 7](file:///c:/Users/PC/Documents/GitHub/Quanta/docs/publication.md#7-bidirectional-translation-and-typological-multilingual-realization) specifies reverse realization into Isolating, Agglutinative, and Fusional languages. However, forward parsing currently relies on English spaCy models and English prompt demonstrations. To serve as a universal language-agnostic context expander, forward transduction must accept queries in German, Turkish, or Mandarin Chinese and compile them into the same canonical $\Sigma^{1024}$ ASG.

### Target Files
- **[NEW]** `src/parser/multilingual_transducer.py`: Multilingual prompt templates and language-specific GBNF extensions.
- **[MODIFY]** [`src/parser/entity_manifest.py`](file:///c:/Users/PC/Documents/GitHub/Quanta/src/parser/entity_manifest.py): Support agglutinative case-inflected entity aliases.
- **[NEW]** `tests/test_multilingual_pipeline.py`: End-to-end cross-lingual translation invariance tests.

### Tasks
- [ ] **Task 8.1: Multilingual S-Expression Transduction Prompts**
  - Author prompt templates in `src/parser/multilingual_transducer.py` with demonstrations for German, Turkish, and Mandarin.
  - Ground non-English verbs directly to universal NSM primes (Band 0) and ConceptNet multi-lingual concept IDs (`/c/de/...`, `/c/zh/...`, `/c/tr/...`).
- [ ] **Task 8.2: Inflected Entity Alias Resolution**
  - In `src/parser/entity_manifest.py`, extend `EntityMatcher` with lightweight stemming/lemmatization to match agglutinative case suffixes (e.g. Turkish `-da/-de`, `-a/-e`).
- [ ] **Task 8.3: 🧪 Cross-Lingual Round-Trip Tests (`tests/test_multilingual_pipeline.py`)**
  - Ingest German paragraph $\to$ compile to ASG $\to$ realize in English.
  - Assert that canonical slot preservation is $\ge 95\%$ and Hamming distance is 0 on core concepts.
  - Run: `pytest tests/test_multilingual_pipeline.py -v`.

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

# Section 6: Context Expansion Server
pytest tests/test_context_expansion_server.py -v

# Section 7: Wikipedia Knowledge Base
pytest tests/test_wikipedia_kb.py -v

# Section 8: Multilingual Pipeline
pytest tests/test_multilingual_pipeline.py -v
```
