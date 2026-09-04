# QUANTA Semantic Graph Generation & Long-Horizon Context Architecture: Master Roadmap (`graph_todo.md`)

> [!IMPORTANT]
> **Mission Statement:**
> Transform QUANTA from an overfitted syntactic word-tree builder into a production-grade, verifiable **Neuro-Symbolic Cognitive Engine**.
> By combining a local **Neural Discourse Transducer** (e.g. 4-bit Qwen-4B via LM Studio) for open-domain English extraction with QUANTA's **Symbolic ASG Compiler** (ConceptNet 5.7.0, 1024-dim quaternary vectors, BLAKE3 Merkle CIDs, and Clingo/s(CASP) gates), QUANTA achieves:
> 1. **100% General Extraction on ANY English Text** (zero hardcoded keywords, regexes, or manual verb lists).
> 2. **Effectively Unlimited Context Scaling via Streaming Discourse Chunking & Working Memory Manifests**.
> 3. **Strict $O(1)$ Physical VRAM Execution ($M=512$) via Virtual Page-Table Attention & Hierarchical Merkle Sub-Graph Folding**.

---

## Architectural Architecture: The Neuro-Symbolic Pipeline

```text
                                [ ANY English Input: Single Sentence, Narrative, or Entire Book ]
                                                              │
                                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: STREAMING DISCOURSE CHUNKER (150–400 words per chunk: Paragraph / Scene Boundaries)                           │
└─────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                              │
                              Dynamic Entity Manifest (~150 tok)│ (Active working memory: E1..Ek)
                              Paging dormant entities from Host RAM▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: NEURAL DISCOURSE TRANSDUCER (Local Qwen-4B @ Q4 via LM Studio / GGUF)                                         │
│ • Input: Current Chunk + Active Entity Manifest                                                                        │
│ • Output: Normalized Intermediate JSON (Entities with mentions, Events with E-IDs, Allen Intervals, Causal Links)     │
└─────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                              │ Normalized JSON-ASG
                                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: QUANTA SYMBOLIC ASG COMPILER (Deterministic Grounding & Merkle Topology)                                      │
│ • ConceptNet 5.7.0 (403,503 concepts) & WordNet lexical grounding                                                      │
│ • 1024-Dimension Quaternary Vector Synthesis (Bands 0–7, 256 bytes per node)                                            │
│ • Band 1 Thematic Valencies (VAL_X1_AGENT, VAL_X2_PATIENT...) & Band 2 Variable Registers (X0..X7)                     │
│ • Band 7 Spatio-Temporal Calculi (Allen Intervals) & Pearl Causal DAG Links                                            │
│ • 256-bit BLAKE3 Cryptographic CID Hashing for all Nodes                                                               │
└─────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                              │
                    ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
                    ▼                                                                                   ▼
┌───────────────────────────────────────┐                                   ┌───────────────────────────────────────────┐
│ Clingo / s(CASP) Verification Gate    │                                   │ PHASE 5 & 6: VIRTUAL PAGE-TABLE ATTENTION │
│ • Ontological type constraint checks  │                                   │ • Bounded GPU VRAM Canvas (M = 512 nodes) │
│ • Temporal consistency (no A < B & B < A)                                 │ • Hierarchical Merkle Sub-Graph Folding   │
│ • Minimal Unsatisfiable Core (MUC)    │                                   │ • Host RAM / SQLite 1M+ Node Storage      │
└───────────────────────────────────────┘                                   │ • Sub-10ms Bitwise SIMD Hamming Retrieval │
                                                                            └─────────────────────┬─────────────────────┘
                                                                                                  │
                                                                                                  ▼
                                                                            ┌───────────────────────────────────────────┐
                                                                            │ PHASE 7: HONEST SEMANTIC REALIZER (NLG)   │
                                                                            │ • Compositional English unrolling from DAG│
                                                                            │ • O(1) VRAM Question Answering over Graph │
                                                                            └───────────────────────────────────────────┘
```

---

## Phase 0: Documentation Audit, Clean-Up & Architectural Alignment

### Context & Architectural Rationale
The repository contains outdated scratchpads, superseded design drafts (e.g. references to early 256-dimension experiments prior to the 1024-dimension 8-band layout), and duplicate benchmark walk-throughs. Before building the new pipeline, the documentation must be audited, sanitized, and updated so contributors and AI agents have an unambiguous source of truth.

### Implementation Guidelines
- Remove redundant, scratch, and misspelled markdown files (`plan.md`, `empriricalsweepresults.md`).
- Update `README.md` to reflect the Neuro-Symbolic Transducer + Symbolic Compiler architecture.
- Update `todo.md` to align Phase 6 and Phase 8 with the entity-event DAG model.
- Retain high-value architectural defense docs in `docs/`: `dim1024.md`, `conceptnetDimensions.md`, `QUANTA_NeuroSymbolic_Architecture_Defense.md`, `DimensionEval.md`.

### Checklist
- [x] **0.1** Audit all repository markdown files and identify outdated/redundant artifacts.
- [x] **0.2** Remove redundant scratch files: `plan.md` (superseded by `graph_todo.md`) and `empriricalsweepresults.md` (redundant with `output/dimension_sweep_report.md` and contains filename typo).
- [x] **0.3** Update `README.md`:
  - Replace the description of the forward parser to reflect the two-stage **Neural Discourse Transducer (Qwen-4B)** + **Symbolic ASG Compiler**.
  - Document the **Entity-Event DAG schema** (purging word-token/punctuation trees).
  - Detail the **Streaming Discourse Chunker & Active Entity Manifest** for unlimited context horizons.
- [x] **0.4** Update `todo.md` to mark completed scaffolding and align Phase 6/8 with the generalized entity-event paradigm.
- [x] **0.5** 🧪 Run verification check to ensure all links across `README.md`, `docs/`, and `graph_todo.md` resolve cleanly without broken file references.

---

## Phase 1: Streaming Discourse Chunker & Segmentation Engine

### Context & Architectural Rationale
Feeding an entire 100,000-word book or continuous chat history into any LLM causes attention degradation and memory exhaustion. Natural human discourse is structured into paragraphs—self-contained thematic episodes (150–400 words, ~200–500 tokens). Chunking text along paragraph/scene boundaries ensures the neural model operates in its optimal attention sweet spot with near-instant inference latency (~2–3s per chunk).

### Implementation Guidelines
- Implement `DiscourseChunker` in `src/parser/chunker.py`.
- Support multiple boundary granularities:
  - Paragraph boundaries (double newlines, indentation).
  - Dialogue speaker turns.
  - Markdown headings / section breaks / scene breaks (`***`, `---`).
- Maintain a running global token and character offset index so all chunks remain anchored to source provenance.

### Checklist
- [x] **1.1** Implement `DiscourseChunk` dataclass tracking `chunk_id`, `text`, `sentence_spans`, `paragraph_index`, `chapter_id`, and `global_offset`.
- [x] **1.2** Implement `DiscourseChunker` in `src/parser/chunker.py` with configurable window limits (min 100 words, max 400 words) and sentence-boundary preservation.
- [x] **1.3** Implement structural chapter/section delimiter detection for long-form books and documents.
- [x] **1.4** 🧪 Write unit tests in `tests/test_discourse_chunker.py` verifying clean chunking across multi-paragraph narratives, dialogues, and multi-chapter markdown books without splitting mid-sentence.

---

## Phase 2: Working Memory Entity Manifest & Paging Engine

### Context & Architectural Rationale
To maintain global coreference across 100 chapters without passing past text tokens:
1. An **Active Entity Manifest** is maintained in working memory (Host RAM).
2. For each incoming chunk, the manifest (the ~10 currently active characters, places, and objects) is injected into the prompt.
3. The neural model reuses existing IDs (`E1`, `E2`...) rather than minting duplicate entities.
4. If an entity hasn't appeared for 50 chapters, it is safely evicted to the Host-RAM Page Table (SQLite). When its name reappears, the Page Table **pages it back into the active manifest** via fast sub-2ms lookup.

### Implementation Guidelines
- Implement `EntityManifest` and `EntityPagingEngine` in `src/parser/entity_manifest.py`.
- Entity records track: `canonical_id` (`E1`), `canonical_name`, `category`, `surface_aliases` (`["Eleanor Vance", "Dr. Vance", "Eleanor", "she"]`), `last_seen_chunk`, and `salience_score`.
- Pre-chunk surface scan: Use fast regex / spaCy token match against the entity registry to page in dormant entities prior to calling the neural transducer.

### Checklist
- [x] **2.1** Implement `EntityRecord` dataclass in `src/parser/entity_manifest.py` with alias list, category, register binding, and recency tracking.
- [x] **2.2** Implement `ActiveEntityManifest` managing the in-memory working set (target size: 5–15 entities) with LRU eviction to persistent storage.
- [x] **2.3** Implement fast pre-scan entity matcher: scans chunk text in $< 1\text{ ms}$ to resurrect dormant entities from the Host-RAM SQLite Page Table.
- [x] **2.4** Implement prompt formatter that renders the active manifest into a compact ~100-token prompt block:
  ```text
  ACTIVE ENTITIES:
  - E1: Dr. Eleanor Vance (aliases: Eleanor, Vance)
  - E2: synthetic compound (aliases: polymer, specimen)
  ```
- [x] **2.5** 🧪 Write unit tests in `tests/test_entity_manifest.py`: Simulate a 5-chunk story; verify entity `E1` introduced in Chunk 1 is correctly paged into the manifest and reused in Chunk 5.

---

## Phase 3: Local Neural Discourse Transducer (`src/parser/transducer.py`)

### Context & Architectural Rationale
Handwritten regexes cannot generalize to open-domain English. We employ a quantized local model (Qwen-4B via LM Studio on `http://localhost:1234/v1` or direct in-repo GGUF) strictly as an open-domain **Semantic Transducer**. By supplying a strict JSON schema and a 1-shot in-context demonstration, Qwen-4B extracts entities, coreference backreferences, event predicates, Allen temporal relations, and causal links with high fidelity in 2–4 seconds per chunk.

### Implementation Guidelines
- Create `src/parser/transducer.py` supporting:
  - `LMStudioTransducer`: REST client connecting to local OpenAI-compatible endpoint (`http://localhost:1234/v1/chat/completions`) with temperature 0.0 and JSON response mode.
  - `LocalGGUFTransducer`: Fallback runner via `llama-cpp-python` loading `.gguf` directly.
  - `MockTransducer`: Deterministic mock for fast offline unit tests in CI.
- System prompt enforces the canonical intermediate schema:
  - Physical entities only (no abstract propositions as entities).
  - Explicit entity ID pointers in event arguments (`agent_id: "E1"`, `patient_id: "E2"`).
  - Exact Allen interval relations and causal mechanism strings.

### Checklist
- [x] **3.1** Define typed intermediate schema in `src/parser/schema.py`: `DiscourseExtractionResult`, `ExtractedEntity`, `ExtractedEvent`, `ExtractedRelation`, `ExtractedProposition`.
- [x] **3.2** Implement `LMStudioTransducer` in `src/parser/transducer.py` with connection health-check, retry logic, and strict JSON parsing.
- [x] **3.3** Implement `LocalGGUFTransducer` using `llama-cpp-python` as a direct in-process alternative.
- [x] **3.4** Implement `MockTransducer` returning pre-recorded fixture outputs for offline regression testing.
- [x] **3.5** Optimize system prompt with a 1-shot demonstration enforcing entity ID foreign keys (`agent_id: "E1"`) and synonym consolidation.
- [x] **3.6** 🧪 Write integration tests in `tests/test_transducer.py` verifying that parsing the Eleanor Vance paragraph produces the validated 5-entity, 6-event schema.

---

## Phase 4: QUANTA Symbolic ASG Compiler (`src/parser/asg_compiler.py`)

### Context & Architectural Rationale
The Neural Transducer outputs normalized JSON. The **Symbolic ASG Compiler** is the mathematical core of QUANTA: it turns those JSON tuples into 1024-dimension quaternary vectors ($\Sigma^{1024}$), grounds lemmas to offline ConceptNet 5.7.0 and WordNet, assigns Band 2 registers, wires Band 1 thematic valency edges, and computes deterministic 256-bit BLAKE3 Merkle CIDs.

### Implementation Guidelines
- Implement `ASGCompiler` in `src/parser/asg_compiler.py`.
- Concept Grounding:
  - Query `data/conceptnet_offline.db` (403,503 concepts) for canonical concept anchors and 256-D taxonomy vectors (Bands 3 & 4).
  - Apply WordNet synset fallback.
- Vector Compilation:
  - Band 0: Map event verb lemmas to universal NSM primes (`NSM_DO`, `NSM_MOVE`, `NSM_THINK`, `NSM_TRUE`).
  - Band 1: Attach thematic valency edges (`VAL_X1_AGENT`, `VAL_X2_PATIENT`, `VAL_LOCATION_SLOT`, `VAL_X5_INSTRUMENT`) pointing to child entity CIDs. Set tense slots (`LJB_PU_PAST_TENSE`).
  - Band 2: Allocate Variable Binding Registers (`VAR_SLOT_X0` to `VAR_SLOT_X7`) with `01_2` (`BOUND_LOCAL`).
  - Band 5 & 6: Set Theory of Mind belief states, epistemic observation markers, and deontic prohibition slots.
  - Band 7: Wire directed Allen interval edges (`TEMP_ALLEN_MEETS`, `TEMP_ALLEN_AFTER`) and Pearl causal DAG edges (`CAUSAL_MECHANISM_LINK`).
- Cryptographic CID calculation:
  - Compute BLAKE3 hashes bottom-up: Entity CIDs first $\to$ Event CIDs incorporating entity CIDs $\to$ Proposition CIDs.

### Checklist
- [x] **4.1** Implement `ASGCompiler` class in `src/parser/asg_compiler.py`.
- [x] **4.2** Implement entity compilation: convert `ExtractedEntity` $\to$ canonical `QuantaNode` with ConceptNet grounding, Band 2 register assignment, and deterministic CID computation.
- [x] **4.3** Implement event compilation: convert `ExtractedEvent` $\to$ predicate `QuantaNode` with Band 0 NSM primes, Band 1 valency edges to entity CIDs, and Band 5/6 epistemic/deontic slots.
- [x] **4.4** Implement spatio-temporal & causal edge wiring: compile Allen interval relations and Pearl causal mechanisms into Band 7 graph edges.
- [x] **4.5** Integrate with `ValidatorGate`: automatically execute Clingo ASP integrity validation on the compiled graph and report any MUC conflicts.
- [x] **4.6** 🧪 Write comprehensive tests in `tests/test_asg_compiler.py`: Compile the Eleanor Vance JSON; assert resulting graph has exactly 5 entity nodes, 6 event nodes, 0 punctuation nodes, correct BLAKE3 CIDs, and 100% Clingo validation pass rate.

---

## Phase 5: Hierarchical Merkle Sub-Graph Folding (Book-Scale Memory)

### Context & Architectural Rationale
In long-form documents (500 pages), maintaining every individual event node in active memory is unnecessary. Once a paragraph/chunk is compiled and validated, its event DAG is sealed into a **Sub-Graph Merkle CID** (`GRAPH_MERKLE_FOLD_POINT`). At chapter boundaries, chunk CIDs fold into Chapter CIDs, and chapters fold into the Book Merkle Root.

```text
                                Book Root CID (32 Bytes)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          Chapter 1 Merkle CID                          Chapter 2 Merkle CID
                    │                                             │
           ┌────────┴────────┐                           ┌────────┴────────┐
           ▼                 ▼                           ▼                 ▼
     Paragraph 1 CID   Paragraph 2 CID             Paragraph 1 CID   Paragraph 2 CID
           │
    [Folded Event Sub-Graph: Ev1..Ev6]
```

### Implementation Guidelines
- Extend `src/core/asg.py` and `src/core/page_table.py` with hierarchical sub-graph folding operations.
- Folding replaces the internal nodes of an episode with a single 32-byte Merkle CID node while keeping external entity references intact.
- Unfolding dynamically restores the full sub-graph from disk/database when needed.

### Checklist
- [x] **5.1** Implement `fold_discourse_episode(graph, chunk_id) -> QuantaNode` producing a 32-byte Merkle fold node with aggregate quaternary vector.
- [x] **5.2** Implement hierarchical folding: combine chunk fold nodes into Chapter Merkle roots and Book Merkle roots.
- [x] **5.3** Implement dynamic unfolding: `unfold_subgraph(cid, storage) -> QuantaGraph` verifying exact cryptographic SHA-256/BLAKE3 hash integrity.
- [x] **5.4** 🧪 Write unit tests in `tests/test_merkle_folding.py`: Fold a 10-chunk narrative into a single Book CID; verify any tamper in Chunk 3 invalidates the Book Merkle root.

---

## Phase 6: Virtual Page-Table Attention & $O(1)$ VRAM Memory Offloading

### Context & Architectural Rationale
Decouples physical GPU execution memory from context length. The GPU operates on a fixed-size canvas buffer ($M = 512$ nodes $\approx 128\text{ KB}$ of vectors). The rest of the document resides in Host RAM / SQLite. When an active node requires past context, host CPU threads perform **AVX-512 / SIMD bitwise Hamming scans** over 1024-dim quaternary keys at **$51.3\text{ M nodes/sec}$**, paging only the relevant sub-graph into the GPU canvas in $< 5\text{ ms}$.

### Implementation Guidelines
- Implement `PageTable` in `src/memory/page_table.py` backed by SQLite / memory-mapped files.
- Implement vectorized SWAR / SIMD Hamming search over 256-byte quaternary vectors.
- Implement `ActiveCanvas` with LRU eviction and automatic page-fault handling.

### Checklist
- [ ] **6.1** Implement disk-backed `PageTable` in `src/memory/page_table.py` storing `QuantaNode` instances and string interning tables.
- [ ] **6.2** Implement SIMD-accelerated bitwise Hamming distance top-$K$ search over stored 1024-dim quaternary keys.
- [ ] **6.3** Implement `ActiveCanvas` fixed-buffer manager ($M=64$ to $512$ nodes) with LRU eviction policy.
- [ ] **6.4** Implement semantic page-fault handler: when an entity or past event is queried, page it from `PageTable` into `ActiveCanvas`.
- [ ] **6.5** 🧪 Benchmark in `tests/test_page_table_scaling.py`: Ingest a 100,000-node graph into `PageTable`; verify active memory stays strictly $\le 512$ nodes and top-$K$ retrieval completes in $< 5\text{ ms}$.

---

## Phase 7: Honest Semantic Reverse Realizer (NLG) Overhaul

### Context & Architectural Rationale
The previous realizer cheated round-trip evaluation by collecting raw literal strings from all token nodes and rejoining them. In this phase:
1. The token-bypass function `_format_token_sequence` is **completely removed**.
2. Generation is performed strictly from the semantic graph: traversing event predicates, realizing thematic arguments (`VAL_X1_AGENT`, `VAL_X2_PATIENT`, `VAL_LOCATION_SLOT`), utilizing ConceptNet lemmas, and generating discourse-coherent pronouns/anaphora based on the entity registry.

### Implementation Guidelines
- Refactor `EnglishRealizer` in `src/realizer/english_nlg.py`.
- Event traversal: follow Band 7 temporal sequence edges (`TEMP_ALLEN_MEETS`, `TEMP_ALLEN_AFTER`).
- Referring Expression Generation:
  - First mention of $E_i$ in discourse $\to$ Full canonical noun phrase ("Dr. Eleanor Vance", "a volatile synthetic compound").
  - Subsequent mentions in same scope $\to$ Pronoun ("she", "it") or definite descriptor ("the specimen").
- Graph-Native Question Answering:
  - Support query prompts (e.g. *"What did Eleanor Vance verify?"*): traverse the graph directly to the target event and emit the concise factual answer.

### Checklist
- [ ] **7.1** Permanently delete `_collect_descendant_tokens` and `_format_token_sequence` from `src/realizer/english_nlg.py`.
- [ ] **7.2** Implement compositional clause realization: unroll `VAL_X1_AGENT + Predicate + VAL_X2_PATIENT + Modifiers` directly from ConceptNet anchors and Band 1 tense slots.
- [ ] **7.3** Implement discourse-aware anaphoric referring expression generator (first mention full name, subsequent mention pronoun/definite NP).
- [ ] **7.4** Implement discourse narrative sequencing: traverse events along Band 7 temporal edges and insert fluent transition phrases ("immediately", "three hours later").
- [ ] **7.5** Implement graph-query answer generator for zero-attention-cost question answering over Host-RAM graphs.
- [ ] **7.6** 🧪 Write unit tests in `tests/test_honest_realizer.py`: Realize the Eleanor Vance ASG; verify generated text is grammatically fluent, preserves all facts, and uses proper pronouns without inspecting verbatim token lists.

---

## Phase 8: End-to-End Multi-Chapter & Book Benchmark Suite

### Context & Architectural Rationale
Demonstrate the end-to-end cognitive cycle on both complex single paragraphs and multi-chapter documents, proving that the architecture achieves zero hallucination, 100% coreference integrity, constant $O(1)$ VRAM usage, and sub-10ms query execution.

### Checklist
- [ ] **8.1** Re-run the Dr. Eleanor Vance narrative through the complete pipeline:
  - Input text $\to$ Qwen-4B Transducer $\to$ ASG Compiler $\to$ Clingo Verification $\to$ Honest English Realizer.
  - Verify graph contains exactly 5 entities, 6 events, 0 token nodes.
- [ ] **8.2** Multi-Chunk Continuity Test: Run a 3-chunk continuous story through the pipeline; verify entities introduced in Chunk 1 are correctly reused in Chunk 3 without passing Chunk 1 text tokens.
- [ ] **8.3** Long-Context Book Benchmark: Ingest a multi-chapter text ($>10,000$ words) into `PageTable`; verify:
  - VRAM usage remains flat ($\le 512$ nodes).
  - Merkle root is deterministically computed.
  - Questions about Chapter 1 asked after Chapter 10 are answered in $< 10\text{ ms}$ with zero hallucination.
- [ ] **8.4** Regenerate `output/complex_translation_graphs_eng_eng.md` and `output/all_complex_translation_graphs.md` with authentic, deduplicated ASG ASCII hierarchies, Mermaid diagrams, and realizer traces.
