# QUANTA Implementation Roadmap & AI Agent Engineering Manual

> [!IMPORTANT]
> **QUANTA (Quaternary Universal Abstract Natural Topology Architecture)** is a neuro-symbolic cognitive architecture that replaces unconstrained continuous token streams with a discrete, strongly-typed semantic metalanguage (*Mentalese*).
> 
> This document serves as the comprehensive implementation roadmap, technical specification, and verification manual for AI agents and human contributors. Every milestone item is scoped to fit a single atomic git commit. Items marked with 🧪 include an explicit testing or evaluation step.
>
> **Core Architectural Pillars:**
> 1. **Discrete Quaternary Vector Space ($\Sigma^{1024} = \{0, 1, 2, 3\}^{1024}$):** Epistemic 4-valued logic ($\mathcal{B}_4$) preventing continuous floating-point noise drift and representation collapse across 8 isolated 128-slot bands.
> 2. **Universal Semantic & Syntactic Grounding:** Natural Semantic Metalanguage (NSM) primes, ConceptNet 5.7.0 256-D data-driven ontological & affordance grounding, WordNet root categories, FrameNet valencies, and Lojban construct grammar.
> 3. **Cryptographic Merkle ASG Topology:** BLAKE3 sub-graph Content Identifiers (CIDs) enabling recursive graph folding and $\mathcal{O}(1)$ VRAM context scaling.
> 4. **Strict Neuro-Symbolic Verification:** Logic Tensor Networks (LTNs) and top-down coinductive $s(\text{CASP})$ / Answer Set Programming (ASP) with Minimal Unsatisfiable Core (MUC) extraction for closed-loop repair (zero structural hallucination).
> 5. **Non-Autoregressive Discrete Diffusion Proposer (Fast-dLLM v2):** Parallel block denoising bypassing the autoregressive $\mathcal{O}(N)$ generation bottleneck.
> 6. **Virtual Graph Page-Table Attention:** Decouples active GPU execution canvas ($M=64\text{ to }512$) from host RAM/NVMe storage, scaling context horizons to millions of nodes.

---

## Phase 0: Project Scaffolding & Tooling

### Context & Architectural Rationale
QUANTA requires a deterministic, highly modular Python runtime environment capable of integrating high-performance cryptographic hashing (BLAKE3), discrete tensor operations (NumPy), symbolic logic solvers (PyClingo / $s(\text{CASP})$), and linguistic parsing tools (spaCy, WordNet). Clean isolation between core algebraic types, parsers, solvers, profilers, realizers, and neural models is essential to prevent cyclic dependencies and enable independent sub-system testing.

### Implementation Guidelines
- Configure `pyproject.toml` using standard PEP 621 packaging metadata.
- Structure `src/` into distinct domain subpackages: `core`, `parser`, `solver`, `profiler`, `realizer`, `pipeline`, `models`, `data`, and `scripts`.
- Ensure directory layouts under `data/` support raw corpora, processed validation sets, and offline SQLite/JSON caches.

### Verification & Acceptance Criteria
- Running `pytest` discovers and executes the test suite cleanly without import errors.
- Code formatting and linting targets (`lint`, `format`) enforce strict type annotations and PEP 8 compliance.

### Checklist
- [x] **0.1** Create Python project skeleton with `pyproject.toml`, `src/` package structure, and `tests/` directory
- [x] **0.2** Set up dev dependencies: `pytest`, `numpy`, `blake3`, `spacy`, `nltk` (WordNet corpus), `clingo`, `torch`
- [x] **0.3** Add a `Makefile` / `justfile` with targets: `lint`, `test`, `format`
- [x] **0.4** Add `.gitignore` for Python, data caches, and model checkpoints
- [x] **0.5** Create `data/` directory structure: `data/raw/`, `data/validation_corpus/`, `data/virtual_page_table/`
- [x] **0.6** Create `output/` directory structure for profiler reports and dimension exports

---

## Phase 1: Core Data Types & Quaternary Algebra

### Context & Architectural Rationale
Standard autoregressive transformers operate over continuous vector embeddings ($\mathbb{R}^d$), where floating-point approximation errors accumulate across recursive reasoning steps, causing semantic drift and representation collapse. QUANTA grounds its internal states in Belnap’s 4-valued epistemic logic ($\mathcal{B}_4$ or $\mathcal{FOUR}$), treating vectors as discrete algebraic lattices.

$$\Sigma = \{0, 1, 2, 3\}^{1024}$$

The four values represent precise epistemic states:
- `0 (IRRELEVANT / INACTIVE)`: Feature unasserted or structurally non-applicable.
- `1 (TRUE / AFFIRMED)`: Confirmed positive assertion.
- `2 (FALSE / NEGATED)`: Explicit epistemic negation or contradictory property.
- `3 (UNKNOWN / MODAL / QUERY)`: Epistemic uncertainty, interrogative query target, or hypothetical conjecture.

The discrete lattice is governed by two partial orders:
- **Knowledge Order ($\le_k$):** $0 \le_k \{1, 2\} \le_k 3$ (reflecting increasing degrees of information or conflict).
- **Truth Order ($\le_t$):** $2 \le_t \{0, 3\} \le_t 1$ (standard logical lattice from false to true).

Lattice operations:
- **Join ($\sqcup$):** Slot-wise supremum in knowledge order (aggregating evidence across sub-graphs: $\mathbf{u} \sqcup \mathbf{v}$).
- **Meet ($\sqcap$):** Slot-wise infimum in knowledge order (extracting shared asserted knowledge: $\mathbf{u} \sqcap \mathbf{v}$).

Connecting to Finite Scalar Quantization (FSQ), discretizing each slot independently prevents continuous noise drift and functions as a semantic error-correcting code.

### Implementation Guidelines
- `QuaternaryValue` enum must define `IRRELEVANT=0`, `TRUE=1`, `FALSE=2`, `UNKNOWN=3` with fast lookup tables for $\sqcup$ and $\sqcap$.
- `QuantaVector` encapsulates a 1024-element `numpy.uint8` array (with configurable dimension support, default $D=1024$) bounded strictly to $\{0, 1, 2, 3\}$.
- Compact serialization packs 4 quaternary values per byte (2 bits each), compressing each 1024-dimensional vector into exactly 256 bytes (aligned to 4 CPU cache lines / 4 AVX-512 registers).
- Hamming distance is computed as the number of non-matching dimensions: $d_H(\mathbf{u}, \mathbf{v}) = \sum_{i=0}^{1023} \mathbb{I}(u_i \neq v_i)$.

### Verification & Acceptance Criteria
- Verify all 16 pair combinations of $\{0, 1, 2, 3\}$ for $\sqcup$ and $\sqcap$ match the theoretical Belnap $\mathcal{B}_4$ lattice table.
- Algebraic properties: Prove commutativity ($a \sqcup b = b \sqcup a$), associativity ($(a \sqcup b) \sqcup c = a \sqcup (b \sqcup c)$), idempotency ($a \sqcup a = a$), and absorption ($a \sqcup (a \sqcap b) = a$).
- 256-byte binary serialization round-trip: `QuantaVector.from_bytes(v.to_bytes()) == v` for any arbitrary 1024-dimensional vector.
- Metric properties: $d_H(\mathbf{u}, \mathbf{v}) = 0 \iff \mathbf{u} = \mathbf{v}$, $d_H(\mathbf{u}, \mathbf{v}) = d_H(\mathbf{v}, \mathbf{u})$, and triangle inequality holds.

### Checklist
#### 1A — QuaternaryValue enum & lattice operations
- [x] **1A.1** Define `QuaternaryValue` enum: `IRRELEVANT=0`, `TRUE=1`, `FALSE=2`, `UNKNOWN=3`
- [x] **1A.2** Implement knowledge ordering comparator: `0 ≤_k {1,2} ≤_k 3`
- [x] **1A.3** Implement truth ordering comparator: `2 ≤_t {0,3} ≤_t 1`
- [x] **1A.4** Implement lattice `join (⊔)` operation (slot-wise, element-wise max in knowledge order)
- [x] **1A.5** Implement lattice `meet (⊓)` operation (slot-wise, element-wise min in knowledge order)
- [x] **1A.6** 🧪 Write unit tests for all lattice algebra properties: commutativity, associativity, idempotency, absorption
- [x] **1A.7** 🧪 Write tests verifying `join`/`meet` produce correct results for all 16 pair combinations of `{0,1,2,3}`

#### 1B — QuantaVector (1024-dimension quaternary array)
- [x] **1B.1** Define `QuantaVector` class wrapping a 1024-element `numpy.uint8` array (values constrained to `{0,1,2,3}`)
- [x] **1B.2** Implement slot-wise `join` and `meet` between two `QuantaVector` instances
- [x] **1B.3** Implement Hamming distance computation between two `QuantaVector` instances
- [x] **1B.4** Implement `to_bytes()` / `from_bytes()` serialization (256-byte compact form: 2 bits per value, 4 cache lines)
- [x] **1B.5** Implement `__eq__`, `__hash__`, and `__repr__` for `QuantaVector`
- [x] **1B.6** 🧪 Write unit tests: zero vector, full-TRUE vector, mixed vectors, round-trip serialization
- [x] **1B.7** 🧪 Test that Hamming distance is 0 only for identical vectors and symmetric

---

## Phase 2: Canonical Slot Layout (1024 Dimensions - 8 Bands)

### Context & Architectural Rationale
To guarantee zero cross-domain representational drift and deterministic indexing across all cognitive modalities, the 1024 dimensions are strictly partitioned into eight isolated bands of 128 slots each:
- **Band 0 (Slots 000–127):** Universal NSM Primes, Classical Kinematics & Continuous Physics (000–063: core NSM primes, 064–095: physical trajectories & continuous forces, 096–127: vector fields & material states).
- **Band 1 (Slots 128–255):** Structural Valencies, Grammatical Tense/Aspect, Code AST & Concurrency Topologies (128–143: Lojban valencies $x_1 \dots x_7$, 144–167: aspect & tense, 168–215: ASG & code AST topologies, 216–255: concurrency & OS topologies).
- **Band 2 (Slots 256–383):** Logic Quantifiers ($\forall, \exists, \exists!$), Variable Binding & Query Registers ($X_0 \dots X_7$, query targets $?X, ?Y, ?Z$), Lambda Binders, and Sequent Calculus Derivations.
- **Band 3 (Slots 384–511):** ConceptNet 5.7.0 Ontological Taxonomies & Scientific Domains (`CN_Q001_COMPUTING` .. `CN_Q128_MANNER`) derived via Usage-Weighted Ontological Density Scoring (U-ODS) and Hopcroft partition refinement across 34M assertions.
- **Band 4 (Slots 512–639):** ConceptNet 5.7.0 Cyber-Physical Tool Affordances, Mechanical Actions & Functional Capabilities (`CN_Q129_LEAVE` .. `CN_Q256_WORTHY`), enabling 2-tier vector decoding (23,383 singletons in $<10\text{ ms}$ SIMD + SQLite category basin fallback).
- **Band 5 (Slots 640–767):** Theory of Mind (1st, 2nd, 3rd-order nested multi-agent beliefs, shared common ground), Teleological Goals & Planning, Affective Drives, and Pragmatic Discourse Intent.
- **Band 6 (Slots 768–895):** Epistemic Proof Solvers, $s(\text{CASP})$ Invariants (MUC, abducibles, coinduction loops), and Deontic Normative Logic (obligatory, permissible, prohibited, liability).
- **Band 7 (Slots 896–1023):** Spatio-Temporal Mereotopology (13 Allen interval relations, 8 RCC-8 spatial relations), Pearl Causal Counterfactual DAGs, and Modal/LTL/CTL Temporal Logic.

Strongly-typed valency signatures and the `LEGACY_ONTOLOGY_ALIASES` semantic bridge layer reconcile legacy symbolic constants (`TYPE_ANIMATE`, `TYPE_HUMAN`, `AFFORD_INCISED_CUTTING`) with canonical `CN_Q*` slots, ensuring compile-time semantic typing and 100% solver rule compatibility.

### Implementation Guidelines
- Define immutable integer constants for all 1024 slots organized by 8 bands in `src/core/slots.py`.
- Export the complete slot taxonomy and metadata to `output/canonical_slots_layout.json` via `src/scripts/generate_slots_registry.py`.
- Maintain `LEGACY_ONTOLOGY_ALIASES` mapping legacy symbolic constants to canonical `CN_Q*` slot indices.
- Implement a type constraint registry mapping predicate argument valencies to required ontological capabilities (e.g., `VAL_X1_AGENT` requires `ROLE_AGENT_CAPABLE=1` and `TYPE_ANIMATE=1`).

### Verification & Acceptance Criteria
- Validation test confirms exactly 1024 unique indices, zero duplicate slot assignments, and proper band interval boundaries (8 bands × 128 slots: 0–127, 128–255, 256–383, 384–511, 512–639, 640–767, 768–895, 896–1023).
- `validate_valency()` strictly accepts valid types (e.g., `THINK(human)`) and rejects type violations (e.g., `THINK(rock)` where entity has `TYPE_INANIMATE_PHYSICAL=1`).

### Checklist
#### 2A — Slot definitions
- [x] **2A.1** Define Band 0 slot constants (0–127): Universal NSM Primes, Kinematics, Continuous Physics
- [x] **2A.2** Define Band 1 slot constants (128–255): Structural Valencies, Grammatical Tense/Aspect, Code AST & Concurrency
- [x] **2A.3** Define Band 2 slot constants (256–383): Logic Quantifiers, Variable Binding Registers ($X_0 \dots X_7$), Query Heads
- [x] **2A.4** Define Band 3 slot constants (384–511): ConceptNet Ontological Taxonomies & Domains (`CN_Q001` .. `CN_Q128`)
- [x] **2A.5** Define Band 4 slot constants (512–639): ConceptNet Cyber-Physical Tool Affordances & Actions (`CN_Q129` .. `CN_Q256`)
- [x] **2A.6** Define Band 5 slot constants (640–767): Theory of Mind (3-Tier Beliefs), Teleological Goals, Discourse Intent
- [x] **2A.7** Define Band 6 slot constants (768–895): Epistemic Solvers, $s(\text{CASP})$ Invariants, Deontic Logic
- [x] **2A.8** Define Band 7 slot constants (896–1023): Spatio-Temporal Calculi (Allen, RCC-8), Pearl Causal DAGs, LTL/CTL
- [x] **2A.9** Implement `LEGACY_ONTOLOGY_ALIASES` semantic bridge layer in `src/core/slots.py`
- [x] **2A.10** Export all 1024 slot definitions to `output/canonical_slots_layout.json`
- [x] **2A.11** 🧪 Write test verifying no duplicate slot indices, all 1024 indices covered, alias integrity, and 8-band boundaries are correct

#### 2B — Strongly-typed valency signatures
- [x] **2B.1** Define type constraint registry mapping predicate argument slots to required ontological types (e.g., `VAL_X1_AGENT` → `+ANIMATE_AGENT`)
- [x] **2B.2** Implement `validate_valency(node, slot, child_vector) → bool` checking type constraints
- [x] **2B.3** 🧪 Write test: `THINK(rock)` should be rejected (rock is `INANIMATE`), `THINK(human)` should pass

---

## Phase 3: ASG Node & Graph Structure

### Context & Architectural Rationale
Concepts in QUANTA are modeled not as isolated vector embeddings, but as Abstract Syntax Graphs (ASGs). 
- An **atomic node** represents a localized semantic concept (e.g., the action *"bit"* or entity *"dog"*).
- The **whole proposition** is encoded by the entire graph hierarchy.
- The **whole-tree proposition vector** is computed via quaternary lattice join across all nodes in the tree:
  $$\mathbf{v}_{\text{tree}} = \bigsqcup_{u \in \text{Tree}} \mathbf{v}_u$$
- Every node receives a deterministic 256-bit BLAKE3 Content Identifier (CID) calculated over its vector, literal payload, and sorted child edge pointers:
  $$\text{CID}(u) = \text{BLAKE3}\Big(\mathbf{v}_u \,\|\, \text{Payload}(u) \,\|\, \bigoplus_{e=(u,v)} \big(\text{Type}(e) \,\|\, \text{CID}(v)\big)\Big)$$

### Implementation Guidelines
- `QuantaNode` stores `node_cid` (32 bytes), `parent_cid`, `semantic_vector` (`QuantaVector`), `concept_label`, `literal`, edge table `list[(relation_type, child_cid)]`, and anchor metadata (WordNet synset, FrameNet role).
- `QuantaGraph` provides graph container operations, CID indexing, topological traversals, and JSON serialization.

### Verification & Acceptance Criteria
- Hashing invariant: Identical node contents produce identical CIDs; modifying a single vector slot, label, or child edge changes the CID.
- Implement and test canonical examples from README:
  - **Example A (Affirmative):** *"A golden retriever bit the mailman in the garden."* $\to$ aggregate vector matches documented values (`NSM_DO=1`, `NSM_TOUCH=1`, `VAL_X1_AGENT=1`, `TYPE_ANIMATE=1`, `SPATIAL_RCC_NON_TANG_PART=1`).
  - **Example B (Negated):** *"The dog did not bite the mailman."* $\to$ `NSM_DO=2`, `LJB_NA_NEGATION=2`.
  - **Example C (Uncertain):** *"Did the dog perhaps bite a mailman?"* $\to$ `NSM_DO=3`, `GRAPH_QUERY_TARGET=3`.

### Checklist
#### 3A — QuantaNode
- [x] **3A.1** Define `QuantaNode` dataclass: `node_cid: bytes`, `parent_cid: bytes | None`, `semantic_vector: QuantaVector`, `concept_label: str`, `literal: str`
- [x] **3A.2** Add edge table to `QuantaNode`: list of `(relation_type: str, child_cid: bytes)` tuples
- [x] **3A.3** Add anchor/literal payload field: optional WordNet synset ID, FrameNet role, or raw literal value
- [x] **3A.4** Implement BLAKE3 CID computation: `CID(node) = BLAKE3(vector ∥ payload ∥ concat(edge_type ∥ child_CID))`
- [x] **3A.5** 🧪 Write test: same node content → same CID; any bit flip → different CID

#### 3B — QuantaGraph (ASG container)
- [x] **3B.1** Define `QuantaGraph` class: stores `dict[bytes, QuantaNode]` keyed by CID, with a `root_cid` field
- [x] **3B.2** Implement `add_node()` that computes CID and inserts into graph
- [x] **3B.3** Implement `get_children(node_cid) → list[QuantaNode]`
- [x] **3B.4** Implement tree-aggregate vector: `v_tree = ⊔ (join) of all node vectors in the graph`
- [x] **3B.5** Implement graph serialization to JSON and deserialization
- [x] **3B.6** 🧪 Build the "golden retriever bit the mailman" example graph from README Example A; verify aggregate vector matches documented values
- [x] **3B.7** 🧪 Build the negated example (Example B: "The dog did not bite the mailman"); verify `NSM_DO=2`, `LJB_NA_NEGATION=2` in aggregate
- [x] **3B.8** 🧪 Build the uncertainty example (Example C: "Did the dog perhaps bite a mailman?"); verify `NSM_DO=3`, `GRAPH_QUERY_TARGET=3` in aggregate

---

## Phase 4: Merkle-Tree Sub-Graph Folding

### Context & Architectural Rationale
In autoregressive LLMs, scaling context horizons incurs quadratic memory overhead ($\mathcal{O}(N^2)$). In QUANTA, complex multi-node sub-graphs (e.g., a past conversation turn, a sub-clause, or a library function definition) are folded into a single 256-bit BLAKE3 Merkle Content Identifier (CID). When referenced elsewhere, the system passes the compact 32-byte CID pointer rather than re-expanding the entire token sequence.

$$\text{CID}(\mathcal{G}_{\text{sub}}) = \text{BLAKE3}\left( \bigoplus_{v \in \mathcal{V}} \mathbf{v}_v \;\Big\|\; \text{Adj}(\mathcal{G}_{\text{sub}}) \right)$$

### Implementation Guidelines
- Implement recursive sub-graph hashing where child sub-graphs are replaced with Merkle fold point nodes (`GRAPH_MERKLE_FOLD_POINT=1`).
- `fold_subgraph()` prunes the subtree and inserts a pointer node containing the sub-graph's aggregate vector and folded CID.
- `unfold_subgraph()` dynamically restores the original graph structure from disk/memory storage.

### Verification & Acceptance Criteria
- Exact round-trip restoration: `unfold_subgraph(fold_subgraph(G))` produces an isomorphic graph with identical node CIDs.
- Cryptographic tamper detection: Modifying any attribute or child in the sub-graph alters the folded root CID.

### Checklist
- [x] **4.1** Implement recursive sub-graph CID folding: `CID(G_sub) = BLAKE3(⊕ v_node ∥ Adj(G_sub))`
- [x] **4.2** Implement `fold_subgraph(graph, subtree_root_cid) → folded_cid` that replaces a subtree with a CID pointer node
- [x] **4.3** Implement `unfold_subgraph(folded_cid, storage) → QuantaGraph` that restores the subtree from storage
- [x] **4.4** 🧪 Test fold → unfold round-trip: CID matches, restored graph is identical to original
- [x] **4.5** 🧪 Test that modifying any node in the subtree changes the folded CID (tamper detection)

---

## Phase 5: Offline Data, ConceptNet 5.7.0 & Lexical Resources

### Context & Architectural Rationale
To ensure $\mathcal{O}(1)$ deterministic lexical anchoring with rich real-world common-sense, physical affordances, and cross-lingual capability without runtime network calls:
1. **ConceptNet 5.7.0 Knowledge Base:** Ingests 34,074,917 assertions, runs depth $d=2$ BLAS sparse matrix transitive propagation ($M_{\text{inherited}} = M + TM + T^2M$ expanding to 54.61M non-zero connections), and executes the Hopcroft Inverted-Index Partition Refinement Solver across 72,930 unique semantic archetypes to produce 256 globally optimal discriminative dimensions (Bands 3 & 4).
2. **ConceptNet SQLite & Codebook Artifacts:**
   - `data/conceptnet_slots.json`: 256 canonical slot definitions for Bands 3 & 4.
   - `data/conceptnet_offline.db`: 403,503 concepts (713,784 multi-POS entries) with pre-packed 256-byte quaternary vectors.
   - `data/concept_codebook.csv.gz`: Dense $403,503 \times 256$ binary codebook matrix containing 23,383 singletons for instant SIMD decoding.
3. **ConceptNet Lexical Grounder (`ConceptNetLexicalGrounder`):** High-speed in-memory cached SQLite lookup supporting `cn:en:<lemma> (<pos>)`, space/underscore normalization, and fallback multi-POS resolution.
4. **WordNet & FrameNet Offline Fallback:** `data/wordnet_offline.db` (synsets, hypernyms) and `data/framenet_valency.json` (valency place templates).
5. **Typo-Tolerant Lexical Grounding:** Damerau-Levenshtein fuzzy matching (`TypoNormalizer`) and compound word healing to map misspelled surface inputs (e.g., *"glden retreiver"*) to canonical concepts (`cn:en:golden retriever (n)`) without semantic corruption.

### Implementation Guidelines
- `scripts/concept_solver.py`: Ingestion, U-ODS scoring, Hopcroft partition refinement, and SQLite compilation.
- `src/parser/lexical_grounder.py`: Implement `ConceptNetLexicalGrounder` (aliased to `LexicalGrounder`) with quaternary vector unpacking and singleton cache.
- Maintain `WordNetLexicalGrounder` and `FrameNetValencyResolver` for backward-compatible fallback.
- Implement fuzzy matching with maximum edit distance threshold ($d \le 2$) and multi-word token healing heuristics.

### Verification & Acceptance Criteria
- "dog" resolves via ConceptNet to `cn:en:dog (n)` with active `CN_Q011_ANIMAL=1` and `TYPE_ANIMATE=1`.
- "bite" resolves to verb concept `cn:en:bite (v)` and FrameNet frame with Agent and Patient mappings.
- "golden retriever" resolves to multiword concept `cn:en:golden retriever (n)`.
- Typo test suite: `"glden retreiver"` $\to$ `golden retriever`, `"maileman"` $\to$ `mailman`, `"graden"` $\to$ `garden`, generating the exact same Merkle CID as clean input.

### Checklist
#### 5A — ConceptNet 5.7.0 ingestion & partition solver
- [x] **5A.1** Stream and parse 34.07M assertions from ConceptNet 5.7.0; filter for clean English concepts
- [x] **5A.2** Implement depth $d=2$ BLAS sparse matrix transitive inheritance propagation ($M_{\text{inherited}} = M + TM + T^2M$)
- [x] **5A.3** Implement Usage-Weighted Ontological Density Scoring (U-ODS) combining degree, relation entropy, affordances, and Zipf frequencies
- [x] **5A.4** Implement Inverted-Index Hopcroft Partition Refinement Solver to select 256 optimal dimensions across 72,930 archetypes
- [x] **5A.5** Export `data/conceptnet_slots.json` (256 slot definitions for Bands 3 & 4)
- [x] **5A.6** Compile `data/conceptnet_offline.db` (403,503 concepts with pre-packed 256-byte quaternary vectors)
- [x] **5A.7** Export `data/concept_codebook.csv.gz` ($403,503 \times 256$ matrix with 23,383 singletons)

#### 5B — ConceptNet Lexical Grounder
- [x] **5B.1** Implement `ConceptNetLexicalGrounder` in `src/parser/lexical_grounder.py` with multi-POS and space/underscore lookup
- [x] **5B.2** Implement binary quaternary hex unpacking from SQLite `packed_bytes_hex` into `QuantaVector`
- [x] **5B.3** Implement in-memory concept LRU cache for sub-millisecond lexical grounding
- [x] **5B.4** 🧪 Write unit tests for `ConceptNetLexicalGrounder` resolving nouns, verbs, and multiword compounds

#### 5C — WordNet & FrameNet fallback caches
- [x] **5C.1** Build SQLite cache of WordNet synsets, hypernym paths, and top-level root categories (`data/wordnet_offline.db`)
- [x] **5C.2** Export FrameNet frames and roles to `data/framenet_valency.json`
- [x] **5C.3** Implement `resolve_frame_roles(verb_lemma) → dict[role_name, Band1_slot]` mapping
- [x] **5C.4** 🧪 Test: "bite" maps to frame with Agent, Patient roles → `VAL_X1_AGENT`, `VAL_X2_PATIENT`

#### 5D — Typo-tolerant lexical grounding
- [x] **5D.1** Implement fuzzy Damerau-Levenshtein matcher for surface tokens against concept lemma index
- [x] **5D.2** Implement compound word healing (split/merge heuristics for multi-word expressions)
- [x] **5D.3** 🧪 Test: "glden retreiver" → `cn:en:golden retriever (n)`, "maileman" → `cn:en:mailman (n)`, "graden" → `cn:en:garden (n)`

---

## Phase 6: Forward Parser (NL / FOL / AST → Mentalese ASG)

### Context & Architectural Rationale
The forward parser transforms heterogeneous surface expressions (natural language sentences, First-Order Logic formulas, and Python source code) into canonical Quanta Abstract Syntax Graphs ($\Sigma^{1024}$).
- **Natural Language Parsing:** Uses spaCy dependency trees + `ConceptNetLexicalGrounder` (with WordNet/FrameNet fallback) to ground nominal and verbal nodes with `cn:en:<lemma> (<pos>)` anchors, extract thematic valency structures, tenses, determiners, negations (setting polarity `2`), and questions/modals (setting polarity `3`). Multiword compound expressions (e.g. *"golden retriever"*) are grounded as unified concepts before defaulting to head noun + descriptor unrolling.
- **FOL Formula Parsing:** Converts quantified logic expressions ($\forall x, \exists x, \land, \lor, \rightarrow, \neg$) into propositional AST graphs.
- **Python AST Parsing:** Converts Python AST nodes (`FunctionDef`, `If`, `Return`, `For`/`While`, recursive calls) into graph topologies with cyclic self-CID links for recursive routines.

### Implementation Guidelines
- `nlp_forward.py`: Convert spaCy dependency trees into Quanta ASG topologies using `ConceptNetLexicalGrounder`. Handle determiners ("a" $\to$ `NSM_ONE=1`, "the" $\to$ `NSM_THIS=1`, "every" $\to$ `NSM_ALL=1`). Negations set `LJB_NA_NEGATION=2` and flip action primes (`NSM_DO=2`). Interrogatives set `GRAPH_QUERY_TARGET=3` and modal primes (`NSM_MAYBE=3`).
- `fol_parser.py`: Tokenize and recursively parse FOL strings into structured ASGs.
- `ast_parser.py`: Walk Python `ast` nodes, mapping control flow, variable bindings, and self-referential recursive calls (`GRAPH_RECURSIVE_REF=1` pointing to root function CID).

### Verification & Acceptance Criteria
- Parse Example A, B, C sentences and verify node and aggregate vectors match README definitions with `cn:en:...` anchors.
- FOL test: $\forall x (\text{Dog}(x) \rightarrow \text{Animal}(x))$ generates the canonical implication ASG (Example E).
- Code test: `def factorial(n): ...` generates the canonical recursive AST topology (Example F).
- Run test item **6B.11** for epistemic uncertainty query parsing.
- Implement and test **6D.6** (recursion detection), **6D.8** (exception handling), and **6D.9** (`factorial(n)` AST generation).

### Checklist
#### 6A — spaCy NLP pipeline integration
- [x] **6A.1** Download and configure spaCy English model (`en_core_web_sm` or `en_core_web_md`)
- [x] **6A.2** Implement `parse_dependency_tree(text) → spaCy Doc` wrapper
- [x] **6A.3** Implement `extract_subject_verb_object(doc) → (subject, verb, object, modifiers)` from dependency parse
- [x] **6A.4** 🧪 Test: "A golden retriever bit the mailman in the garden" extracts correct SVO + location

#### 6B — spaCy → QuantaGraph forward mapping
- [x] **6B.1** Implement root predicate node creation from main verb via `ConceptNetLexicalGrounder` (`cn:en:<verb> (v)`) with WordNet fallback
- [x] **6B.2** Implement agent child node creation from subject + multiword compound ConceptNet grounding (`cn:en:<noun> (n)`)
- [x] **6B.3** Implement patient child node creation from object + multiword compound ConceptNet grounding
- [x] **6B.4** Implement modifier/location/temporal/destination child node creation from prepositional phrases
- [x] **6B.5** Implement tense detection and mapping to Band 1 tense slots (`LJB_PU_PAST_TENSE`, etc.)
- [x] **6B.6** Implement negation detection (`not`, `n't`) → set `LJB_NA_NEGATION=2` and flip relevant Band 0 primes to `2`
- [x] **6B.7** Implement question/uncertainty detection → set `GRAPH_QUERY_TARGET=3`, relevant primes to `3`
- [x] **6B.8** Implement determiner resolution: "a" → `NSM_ONE=1`, "the" → `NSM_THIS=1`, "every" → `NSM_ALL=1`
- [x] **6B.9** 🧪 Full forward-parse test for Example A ("golden retriever bit the mailman in the garden") — verify all node vectors and anchors match
- [x] **6B.10** 🧪 Full forward-parse test for Example B ("The dog did not bite the mailman") — verify negation slots
- [x] **6B.11** 🧪 Full forward-parse test for Example C ("Did the dog perhaps bite a mailman?") — verify query/uncertainty slots

#### 6C — FOL formula parser
- [x] **6C.1** Implement tokenizer for FOL strings: `∀`, `∃`, `→`, `∧`, `∨`, `¬`, `(`, `)`, predicates, variables
- [x] **6C.2** Implement recursive descent parser producing an AST of FOL expressions
- [x] **6C.3** Implement FOL AST → QuantaGraph conversion (universal quantifier → `LJB_RO_ALL_QUANT=1`, implication → `LJB_GANAI_IF_THEN=1`, etc.)
- [x] **6C.4** 🧪 Test: `∀x(Dog(x) → Animal(x))` → QuantaGraph matching README Example E

#### 6D — Python AST parser
- [x] **6D.1** Implement Python source → `ast.parse()` → walk AST nodes
- [x] **6D.2** Map `FunctionDef` → `GRAPH_FUNCTION_DEF=1`, `GRAPH_SCOPED_CONTEXT=1`
- [x] **6D.3** Map `If` → `GRAPH_BRANCH_COND`, `GRAPH_BRANCH_THEN`, `GRAPH_BRANCH_ELSE`
- [x] **6D.4** Map `Return` → `GRAPH_RETURN_VALUE=1`
- [x] **6D.5** Map function arguments → `GRAPH_VARIABLE_BIND=1`, `GRAPH_ARGUMENT_LIST=1`
- [x] **6D.6** Detect recursive calls → `GRAPH_RECURSIVE_REF=1` with self-CID pointer
- [x] **6D.7** Map loops → `GRAPH_CONTROL_LOOP=1`
- [x] **6D.8** Map exception handling → `GRAPH_EXCEPTION_HANDLE=1`
- [x] **6D.9** 🧪 Test: `factorial(n)` function → QuantaGraph matching README Example F

---

## Phase 7: Neuro-Symbolic Verification Gate

### Context & Architectural Rationale
The Neuro-Symbolic Verification Gate acts as QUANTA's "System 2" cognitive compiler. Proposed graphs are audited by symbolic constraint solvers to guarantee formal correctness.
1. **PyClingo / ASP Invariants with Semantic Alias Bridging:** Encodes ontological domain/range constraints, RCC-8 spatial mereotopology (e.g., `DISCONNECTED` and `NON_TANG_PART` are mutually exclusive), Allen interval temporal logic, and Pearl causal hierarchy rules. Integrates `LEGACY_ONTOLOGY_ALIASES` so that distributed ConceptNet semantic fingerprints (`CN_Q001` .. `CN_Q256`) project cleanly onto symbolic domain rules (`TYPE_ANIMATE`, `ROLE_AGENT_CAPABLE`, etc.) without requiring combinatorial 1-hot mutual exclusion.
2. **Minimal Unsatisfiable Core (MUC) Extraction:** When a constraint is violated, the solver isolates the smallest set of conflicting graph nodes and slot assertions. This diagnostic vector enables closed-loop targeted repair rather than discarding the entire generation.
3. **$s(\text{CASP})$ Coinductive ASP:** Evaluates logic programs top-down without exhaustive grounding, retaining logical variables and resolving cyclic dependencies (such as recursive function ASTs) that cause grounding explosion in standard ASP solvers.

### Implementation Guidelines
- `scasp_rules.lp`: Write Clingo integrity constraints (`:- ...`) enforcing ontological typing (e.g., inanimate entities cannot be volitional agents unless figurative).
- Implement spatial consistency rules (RCC-8) and temporal ordering rules (Allen intervals).
- `validator_gate.py`: Execute solver over converted graph facts and mapped `LEGACY_ONTOLOGY_ALIASES`. If unsatisfiable, compute MUC using assumption literals.
- `scasp_rules.pl`: Formulate equivalent top-down coinductive $s(\text{CASP})$ rules for SWI-Prolog bridge with alias expansion.

### Verification & Acceptance Criteria
- "The rock thinks" triggers an immediate constraint violation; MUC identifies the rock entity node as the conflict source.
- "The human thinks" and "The dog bites the mailman" pass validation with zero violations across ConceptNet-grounded nodes.
- Spatial/temporal contradiction tests: A graph asserting both $A \text{ before } B$ and $B \text{ before } A$ is flagged as invalid.
- Coinduction test: Recursive ASG graph is validated without infinite loop or grounding timeout.

### Checklist
#### 7A — Clingo / PyClingo ASP rules
- [x] **7A.1** Install `clingo` Python bindings
- [x] **7A.2** Write ontological integrity rules in `scasp_rules.lp`: animate-only agents, entity-level typing constraints with figurative modality bypass
- [x] **7A.3** Write domain/range validation rules: e.g., `ABSTRACT_CONCEPT` cannot have `AGENT_CAPABLE=1` unless figurative
- [x] **7A.4** 🧪 Test: "The rock thinks" graph triggers constraint violation; "The human thinks" passes

#### 7B — Extended invariant rules
- [x] **7B.1** Write RCC-8 spatial mereotopology consistency rules (e.g., `DISCONNECTED` and `NON_TANG_PART` are mutually exclusive)
- [x] **7B.2** Write Allen interval temporal calculus consistency rules (e.g., `BEFORE` and `AFTER` are inverses)
- [x] **7B.3** Write Pearl causal hierarchy exclusivity rules
- [x] **7B.4** Write Band 1 structural topology rules (e.g., `GRAPH_ROOT_NODE` and `GRAPH_LEAF` are mutually exclusive on same node)
- [x] **7B.5** 🧪 Test each rule category with valid and invalid graph inputs

#### 7C — Validator gate & MUC extraction
- [x] **7C.1** Implement `ValidatorGate` class that accepts a `QuantaGraph`, emits canonical and aliased facts, and runs Clingo solving
- [x] **7C.2** Implement MUC (Minimal Unsatisfiable Core) extraction: identify the smallest set of conflicting nodes/axioms
- [x] **7C.3** Implement `validate(graph) → ValidationResult` returning validity, errors, MUC slots, and MUC node CIDs
- [x] **7C.4** 🧪 Test MUC extraction: create a graph with one invalid node among valid ones; verify MUC pinpoints exactly the invalid node
- [x] **7C.5** 🧪 Test validation with ConceptNet-grounded graphs and verify alias bridge functionality

#### 7D — s(CASP) / SWI-Prolog integration (optional advanced path)
- [x] **7D.1** Install SWI-Prolog with s(CASP) pack
- [x] **7D.2** Write s(CASP) Prolog equivalents of the Clingo rules in `scasp_rules.pl` with alias expansion
- [x] **7D.3** Implement Python↔Prolog subprocess bridge for s(CASP) invocation (`scasp_bridge.py`)
- [x] **7D.4** 🧪 Test coinductive reasoning: validate a cyclic graph (recursive function ASG) that standard grounding would reject

---

## Phase 8: Reverse Realizers (Mentalese → Surface Forms)

### Context & Architectural Rationale
To prove that Mentalese is a complete, lossless semantic pivot, verified Quanta ASGs must be deterministically unrolled into diverse human and computational target languages:
- **English Realizer (`EnglishRealizer`):** Unrolls thematic roles into natural SVO syntax with tense inflection, determiner selection, preposition routing, and **2-Tier Vector Decoding (`ConceptVectorDecoder`)**:
  - *Tier 1 (In-Memory SIMD)*: Instant decoding across 23,383 singletons ($<10\text{ ms}$) loaded from `data/concept_codebook.csv.gz`.
  - *Tier 2 (Category Basin Fallback)*: Category basin lookup in `data/conceptnet_offline.db`.
  - Parses canonical `cn:en:<lemma> (<pos>)`, `cn:...`, and legacy `wn:...` anchors, as well as anchor-free semantic vector decoding.
- **FOL Emitter (`FOLEmitter`):** Emits standard First-Order Logic syntax with correct operator precedence.
- **Code Emitter (`CodeEmitter`):** Emits executable Python source code from AST topologies.
- *(Note: Specific language realizers like Hungarian have been removed in favor of a future generalized multilingual architecture).*

### Implementation Guidelines
- `english_nlg.py`: Construct grammatical English prose from ASG traversal and instantiate `ConceptVectorDecoder`.
- `fol_emitter.py`: Traverse quantified sub-expressions and format formulas ($\forall x (\text{Dog}(x) \rightarrow \text{Animal}(x))$).
- `code_emitter.py`: Format Python functions, loops, and recursive calls with proper indentation.

### Verification & Acceptance Criteria
- Example A graph $\to$ English: *"A golden retriever bit the mailman in the garden."*
- Example B graph $\to$ English: *"The dog did not bite the mailman."*
- Example C graph $\to$ English: *"Did the dog perhaps bite a mailman?"*
- Example E graph $\to$ FOL string: `∀x(Dog(x) → Animal(x))`.
- Example F graph $\to$ Executable Python code; running `factorial(5)` returns `120`.
- 2-Tier vector decoding test: Pure 256-D vector with zero anchors resolves correctly to "dog" via Tier 1 codebook table.

### Checklist
#### 8A — English Realizer & ConceptVectorDecoder
- [x] **8A.1** Implement ASG traversal: root predicate → verb, X1 → subject, X2 → object
- [x] **8A.2** Implement `ConceptVectorDecoder` in `src/realizer/english_nlg.py` with 2-tier search (23,383 singletons SIMD table + SQLite category basin fallback)
- [x] **8A.3** Implement anchor parsing for `cn:en:<lemma> (<pos>)`, `cn:...`, and `wn:...` formats
- [x] **8A.4** Implement tense inflection from Band 1 tense slots (past/present/future)
- [x] **8A.5** Implement negation surface form: `LJB_NA_NEGATION=2` → "did not [verb]"
- [x] **8A.6** Implement determiner generation: `NSM_ONE=1` → "a", `NSM_THIS=1` → "the", `NSM_ALL=1` → "every"
- [x] **8A.7** Implement prepositional phrase generation from location/temporal/destination child nodes
- [x] **8A.8** 🧪 Test: Example A graph → "A golden retriever bit the mailman in the garden."
- [x] **8A.9** 🧪 Test: Example B graph → "The dog did not bite the mailman."
- [x] **8A.10** 🧪 Test: Example C graph → "Did the dog perhaps bite a mailman?"
- [x] **8A.11** 🧪 Test: Anchor-free 2-tier vector decoding (< 10 ms SIMD lookup) on 256-D ConceptNet vector

#### 8B — Generalized Multilingual Realizer Architecture
- [x] **8B.1** Plan generalized cross-lingual morphology & syntax mapping framework
- [x] **8B.2** Design typologically diverse language adapters (agglutinative, fusional, isolating)

#### 8C — FOL Emitter
- [x] **8C.1** Implement ASG → FOL string conversion: quantifiers, connectives, predicates, variables
- [x] **8C.2** Implement operator precedence and parenthesization
- [x] **8C.3** 🧪 Test: Example E graph → `∀x(Dog(x) → Animal(x))`
- [x] **8C.4** 🧪 Test round-trip: FOL string → parse → ASG → emit → same FOL string

#### 8D — Code Emitter (Python)
- [x] **8D.1** Implement ASG → Python source: function def, arguments, return, branches, loops
- [x] **8D.2** Implement recursive call reconstruction from `GRAPH_RECURSIVE_REF` edges
- [x] **8D.3** Implement indentation and formatting for readable output
- [x] **8D.4** 🧪 Test: Example F graph → syntactically valid `factorial(n)` Python code
- [x] **8D.5** 🧪 Test: emitted code is executable and `factorial(5) == 120`

---

## Phase 9: Round-Trip Invariance Testing 🧪

### Context & Architectural Rationale
Round-trip invariance is the definitive test of representational adequacy in QUANTA:
$$\text{Input} \xrightarrow{\text{Forward Parser}} \mathcal{G} \in \Sigma^{1024} \xrightarrow{\text{Reverse Realizer}} \text{Target}$$
If Mentalese captures true invariant semantics, transforming surface text into an ASG and unrolling it must preserve 100% of logical meaning, maintain Hamming distance $d_H(\mathbf{v}_{\text{orig}}, \mathbf{v}_{\text{rt}}) = 0$ over canonical slots, and preserve execution semantics for code.

### Implementation Guidelines
- Construct round-trip test suites in `tests/test_round_trip.py` covering natural language (English), FOL formulas, and Python source code.
- Compute quaternary Hamming distance across round-tripped ASG vectors to verify zero semantic slot drift.

### Verification & Acceptance Criteria
- FOL round-trip: String $\to$ ASG $\to$ String produces semantically and syntactically identical formula.
- Python round-trip: Executing generated code yields identical return values across all test inputs.
- Negation and modal round-trip: Affirmation, negation, and uncertainty polarities are perfectly preserved.
- Complete test **9.7** (uncertainty round-trip).

### Checklist
- [x] **9.1** 🧪 NL round-trip: English sentence → forward parse → ASG → English realizer → compare semantic equivalence
- [x] **9.2** 🧪 FOL round-trip: FOL string → forward parse → ASG → FOL emitter → exact string match
- [x] **9.3** 🧪 Code round-trip: Python source → AST parser → ASG → Code emitter → exec both, compare output
- [x] **9.4** 🧪 Quaternary Hamming distance test: round-tripped ASG vectors must have Hamming distance = 0 on canonical slots
- [x] **9.5** 🧪 Cross-lingual test: Verify Mentalese ASG unrolls across Isolating, Agglutinative, and Fusional adapters with zero semantic drift
- [x] **9.6** 🧪 Negation round-trip: "X did not Y" → ASG → English → verify negation preserved
- [x] **9.7** 🧪 Uncertainty round-trip: "Did X perhaps Y?" → ASG → English → verify question/modal preserved
- [x] **9.8** 🧪 ConceptNet translation suite: dedicated round-trip & vector decoding verification (`tests/test_conceptnet_translation.py`)

---

## Phase 10: Information Profiler & Dimension Optimization

### Context & Architectural Rationale
We frame the QUANTA discrete semantic space as a **Discrete Information Bottleneck** optimization problem. An optimal discrete semantic alphabet $\mathbf{D} = (D_0, \dots, D_{1023}) \in \{0, 1, 2, 3\}^{1024}$ must satisfy:
1. **Maximal Channel Utilization:** Slot entropy $H(D_i) \ge 0.35\text{ bits}$ (target $\ge 1.2\text{ bits}$ on benchmark corpus). Dead slots ($H(D_i) \approx 0$) are diagnosed and refactored into symbolic $s(\text{CASP})$ inference rules.
   $$H(D_i) = -\sum_{s \in \{0,1,2,3\}} P(D_i = s) \log_2 P(D_i = s)$$
2. **Minimal Redundancy:** Pairwise mutual information $I(D_i; D_j) < 0.15\text{ bits}$ computed via vectorized one-hot BLAS GEMM matrix operations.
   $$I(D_i; D_j) = \sum_{s_i, s_j} P(D_i=s_i, D_j=s_j) \log_2 \frac{P(D_i=s_i, D_j=s_j)}{P(D_i=s_i)P(D_j=s_j)}$$
3. **Zero Semantic Collisions:** Collision rate $R_{\text{collision}} = 0$ for all ontological and concept propositions.
4. **Empirical Pareto Dimension Optimization ($d^* = 1024$):** Sweep experiments across $d \in \{64, 128, 256, 512, 1024, 2048\}$ prove that 1024 dimensions (256 packed bytes = 4 CPU cache lines / 4 AVX-512 registers) represents the global sweet spot—achieving 0.000000% collision, entropy saturation, $<5\text{ ms}$ ASP solver grounding, and $51.3\text{ M nodes/sec}$ SIMD scanning.

### Implementation Guidelines
- `candidate_pool.py`: Aggregate candidates from NSM, WordNet root categories, FrameNet roles, and formal logic operators into an over-complete pool ($K = 2048$ candidates).
- `info_profiler.py`: Compute entropy, pairwise MI matrix, total correlation $\text{TC}(\mathbf{D})$, and collision rates over parsed validation tensors.
- `mrmr_selector.py`: Run forward greedy mRMR ranking using multi-objective neuro-symbolic utility (F1 alignment, entropy, causal necessity, redundancy penalty, constraint violation penalty).
- `scripts/run_dimension_sweep.py`: Benchmark dimension configurations ($d \in \{64, 128, 256, 512, 1024, 2048\}$) across collision rates, entropy, ASP solver latency, and SIMD throughput.
- Export results to `output/canonical_slots_layout.json`, `output/candidate_pool.json`, `output/dimension_sweep_results.json`, `output/dimension_sweep_report.md`, and `output/information_profiler_report.txt`.

### Verification & Acceptance Criteria
- All 1024 canonical dimensions satisfy $H(D_i) \ge 0.35\text{ bits}$ and $I(D_i; D_j) < 0.15\text{ bits}$.
- Collision rate $R_{\text{collision}} = 0.000000\%$ across 10,000 distinct concept propositions in the benchmark corpus.
- Dimension sweep empirically verifies that $d^* = 1024$ balances collision-free representation, $<5\text{ ms}$ solver grounding, and 4-cache-line hardware alignment.

### Checklist
#### 10A — Candidate pool builder
- [x] **10A.1** Enumerate all NSM primes (~65) as candidate dimensions
- [x] **10A.2** Enumerate WordNet base synsets (top-level hypernym categories) as candidates
- [x] **10A.3** Enumerate FrameNet thematic roles as candidates
- [x] **10A.4** Enumerate formal modal/temporal/spatial operators as candidates
- [x] **10A.5** Combine into over-complete pool (`K = 2048` candidates)
- [x] **10A.6** Export pool to `output/candidate_pool.json`

#### 10B — Information Profiler
- [x] **10B.1** Implement slot entropy computation: `H(D_i) = -Σ P(D_i=s) log₂ P(D_i=s)` over parsed corpus vectors
- [x] **10B.2** Implement pairwise mutual information computation: `I(D_i; D_j)` for all slot pairs using fast BLAS matrix operations
- [x] **10B.3** Implement total correlation computation: `TC(D) = Σ H(D_i) - H(D_0,...,D_1023)`
- [x] **10B.4** Implement collision rate computation: count distinct concept pairs mapping to identical vectors
- [x] **10B.5** Implement report generator: flag slots with `H(D_i) < 0.1 bits` (dead), pairs with `I(D_i;D_j) > 0.5 bits` (redundant)
- [x] **10B.6** Export report to `output/information_profiler_report.txt` and `.json`

#### 10C — mRMR dimension selector & dimension sweep
- [x] **10C.1** Implement greedy forward mRMR selection: iteratively pick dimension maximizing multi-objective utility
- [x] **10C.2** Run mRMR on candidate pool against validation corpus
- [x] **10C.3** Export selected optimal dimensions to `output/optimal_dimensions.json` and `output/optimal_dimensions.csv`
- [x] **10C.4** 🧪 Implement and run dimension sweep across `d ∈ {64, 128, 256, 512, 1024, 2048}`; verify $d^* = 1024$ achieves $R_{\text{collision}} = 0$, entropy saturation, $<5\text{ ms}$ ASP latency, and 4-cache-line SIMD throughput
- [x] **10C.5** Export dimension sweep results to `output/dimension_sweep_results.json` and `output/dimension_sweep_report.md`

#### 10D — ConceptNet 256-D Partition Solver & Information Optimization
- [x] **10D.1** 🧪 Formulate and execute Hopcroft Inverted-Index Partition Refinement Solver on ConceptNet 5.7.0 sparse matrix ($d=2$ transitive expansion)
- [x] **10D.2** 🧪 Evaluate entropy and collision rates across 72,930 unique semantic archetypes; select 256 globally optimal dimensions (Bands 3 & 4)
- [x] **10D.3** 🧪 Verify singleton resolvability: exactly 23,383 concepts (32.1% of archetypes, >95% conversational coverage) uniquely identified in Tier 1 codebook
- [x] **10D.4** Export empirical benchmark findings and solver documentation to `docs/conceptnetDimensions.md` and `docs/DimensionEval.md`

---

## Phase 11: Validation Corpus & Benchmark Data

### Context & Architectural Rationale
To rigorously profile dimension entropy and train/evaluate neuro-symbolic reasoning without synthetic distribution bias, QUANTA ingests 5,000 diverse propositions across five established reasoning benchmarks:
1. **FOLIO:** First-Order Logic reasoning over natural language claims.
2. **ProofWriter:** Multi-step rule induction and proof tree generation.
3. **bAbI Tasks:** Multi-hop spatial, temporal, and positional state tracking.
4. **CLUTRR:** Inductive relational and kinematic reasoning over family graphs.
5. **Python AST Corpus:** ~200 algorithmic functions (recursion, loops, conditionals).

### Implementation Guidelines
- `data/raw/`: Store downloaded benchmark raw files.
- `corpus_generator.py`: Ingest and sample 5,000 balanced propositions.
- Forward-parse all 5,000 propositions into Quanta quaternary tensors and audit via Information Profiler.

### Verification & Acceptance Criteria
- Ingestion scripts parse 100% of benchmark samples without crashes.
- Forward parser achieves $>95\%$ valid parse rate across the 5,000 validation propositions.
- Information Profiler audit confirms absence of dead dimensions across the ingested corpus.

### Checklist
#### 11A — Corpus generation
- [x] **11A.1** Download FOLIO dataset and write ingestion script to `data/raw/`
- [x] **11A.2** Download ProofWriter dataset and write ingestion script
- [x] **11A.3** Download bAbI tasks dataset and write ingestion script
- [x] **11A.4** Download CLUTRR dataset and write ingestion script
- [x] **11A.5** Collect ~200 Python functions of varying complexity (factorial, sort, search, etc.)
- [x] **11A.6** Write `corpus_generator.py` that samples 5,000 diverse propositions across all datasets
- [x] **11A.7** 🧪 Run forward parser on all 5,000 propositions; report parse success rate

#### 11B — Corpus-driven profiler validation
- [x] **11B.1** 🧪 Forward-parse entire validation corpus into quaternary tensors
- [x] **11B.2** 🧪 Run Information Profiler on parsed tensors; generate report
- [x] **11B.3** 🧪 Identify and document any dead slots or highly redundant pairs
- [x] **11B.4** 🧪 If dead/redundant slots found, propose refactoring rules (move to s(CASP) inference rules)
- [x] **11B.5** 🧪 Re-run profiler after refactoring; confirm all slots pass thresholds

---

## Phase 12: Unified Translation Pipeline

### Context & Architectural Rationale
The `TranslatorPipeline` orchestrates the complete two-way cognitive cycle:
$$\text{Input (NL / FOL / Code)} \longrightarrow \text{ConceptNet Forward Parser} \longrightarrow \text{Validation Gate (ASP + Aliases)} \longrightarrow \text{Merkle Address} \longrightarrow \text{Reverse Realizer (2-Tier Decoder)}$$
It guarantees end-to-end type safety, structured logging, performance telemetry, and graceful error reporting with Minimal Unsatisfiable Core (MUC) diagnostics when invalid inputs are rejected.

### Implementation Guidelines
- `translator_pipeline.py`: Build orchestration engine supporting `target_format ∈ {english, fol, python}` and ConceptNet lexical grounding.
- Emit structured JSON log entries tracking stage latencies and memory usage.
- Return explicit MUC diagnostic payloads upon validation rejection.

### Verification & Acceptance Criteria
- Valid sentences, FOL formulas, and code snippets round-trip cleanly across all target formats.
- Invalid input (*"The rock thinks"*) is rejected at the gate and returns an actionable MUC error report.
- Complete task **12.4** (structured logging with per-stage timing).

### Checklist
- [x] **12.1** Implement `TranslatorPipeline` class orchestrating: Input → Forward Parser → ASP Validation Gate → Merkle Addressing → Reverse Realizer
- [x] **12.2** Add pipeline mode selection: `target_format ∈ {english, fol, python}`
- [x] **12.3** Implement error handling: parser failure, validation rejection (return MUC diagnostic)
- [x] **12.4** Add pipeline logging: every stage emits structured log entry with timing
- [x] **12.5** 🧪 End-to-end pipeline test: "A golden retriever bit the mailman" → English round-trip
- [x] **12.6** 🧪 End-to-end pipeline test: `∀x(Dog(x) → Animal(x))` → FOL round-trip
- [x] **12.7** 🧪 End-to-end pipeline test: `def factorial(n)` → Python code round-trip
- [x] **12.8** 🧪 End-to-end pipeline test: invalid input ("The rock thinks") → validation rejection with MUC
- [x] **12.9** 🧪 End-to-end pipeline test: ConceptNet-grounded sentences with 2-tier vector decoding

---

## Phase 13: Virtual Page-Table Attention (Memory Offloading)

### Context & Architectural Rationale
Autoregressive LLM attention mechanisms scale quadratically ($\mathcal{O}(N^2)$) with token sequence length, exhausting GPU VRAM. QUANTA decouples working context from physical GPU memory by storing long-term history as discrete quaternary bit vectors and Merkle CIDs in host system memory (RAM/NVMe).
- **Constant Physical GPU Canvas ($M = 64\text{ to }512$ active nodes):** The GPU operates on a fixed-size buffer ($\mathcal{O}(1)$ VRAM scaling: $512 \times 256\text{ bytes} = 128\text{ KB}$).
- **SIMD-Accelerated Bitwise Hamming Lookup:** When an active node encounters an external variable or CID pointer, host CPU threads execute AVX-512 / Neon SIMD bitwise distance lookups over 1024-dim quaternary keys (256 bytes = exactly 4 cache lines / 4 AVX-512 registers) to find relevant schemas.
- **Dynamic Paging & LRU Eviction:** Missing sub-graphs are paged into the GPU canvas dynamically on semantic page faults.

### Implementation Guidelines
- `PageTable`: Disk-backed / memory-mapped key-value store mapping `CID → QuantaNode`.
- SIMD / vectorized NumPy kernel computing batch Hamming distances across millions of 1024-dim quaternary keys.
- `ActiveCanvas`: Bounded in-memory node buffer with LRU eviction and page fault handling.

### Verification & Acceptance Criteria
- Load a 100,000-node graph; verify that active memory usage stays strictly constant ($\le 512$ nodes).
- Benchmark lookup latency: Top-K retrieval across 100K stored quaternary keys executes in $< 5\text{ ms}$ on CPU (51.3 M nodes/sec throughput).
- Verify zero corruption when swapping nodes between `PageTable` and `ActiveCanvas`.

### Checklist
#### 13A — Host-side graph storage
- [ ] **13A.1** Implement `PageTable` class: dictionary-based storage mapping `CID → QuantaNode` serialized to disk (SQLite or memory-mapped file)
- [ ] **13A.2** Implement `store(node) → CID` and `fetch(CID) → QuantaNode`
- [ ] **13A.3** Implement batch store/fetch for sub-graphs

#### 13B — SIMD-accelerated Hamming lookup (software path)
- [ ] **13B.1** Implement brute-force Hamming distance scan over stored 1024-dim quaternary keys using NumPy / AVX vectorized operations
- [ ] **13B.2** Implement top-K nearest CID retrieval by Hamming distance
- [ ] **13B.3** 🧪 Benchmark: measure lookup latency for 1K, 10K, 100K, 1M stored nodes (1M nodes = 256 MB RAM)

#### 13C — Active canvas paging simulation
- [ ] **13C.1** Implement `ActiveCanvas` class: fixed-size buffer of `M=64 to 512` nodes in-memory
- [ ] **13C.2** Implement page fault handler: when a CID pointer is encountered that isn't in canvas, fetch from `PageTable`
- [ ] **13C.3** Implement LRU eviction policy for canvas when full
- [ ] **13C.4** 🧪 Test: load a graph of 1000 nodes, verify canvas never exceeds buffer limit in memory
- [ ] **13C.5** 🧪 Test: access patterns hitting cold/hot nodes → measure page fault rates

---

## Phase 14: Mentalese Language Testing 🧪

### Context & Architectural Rationale
This phase establishes an exhaustive validation battery for the Mentalese language design itself, verifying that the 1024-dimension quaternary vector space is mathematically expressive, unambiguous, and semantically faithful across epistemic states, near-synonyms, formal calculi, tool affordances, theory of mind, and complex linguistic edge cases.

### Implementation Guidelines
- Test suites in `tests/test_mentalese_language.py`:
  - **14A:** Epistemic differentiation (affirmative `1`, negated `2`, uncertain `3`; `0` only on non-applicable slots; $\mathbf{u}_{\text{true}} \sqcup \mathbf{u}_{\text{false}}$ yields `3` on conflicting slots).
  - **14B:** Semantic collision & near-synonym separation ("happy" vs. "joyful" differ in at least one canonical dimension; ontological class separation).
  - **14C:** Structural band coverage (declarative $\to$ Bands 0-3; FOL $\to$ Band 2 connectives & variables; code $\to$ Band 1 AST; affordances $\to$ Band 4; ToM $\to$ Band 5; s(CASP) $\to$ Band 6; spatial $\to$ Band 7 RCC-8; temporal $\to$ Band 7 Allen; causal $\to$ Band 7 Pearl).
  - **14D:** Adversarial edge cases (empty inputs, 50+ word sentences, structural ambiguities, double negations, multi-clause conjunctions).

### Verification & Acceptance Criteria
- All tests pass with $100\%$ determinism.
- Zero semantic collisions observed between distinct non-synonymous concepts ($R_{\text{collision}} = 0$).
- Parser gracefully handles adversarial inputs without uncaught exceptions.

### Checklist
#### 14A — Epistemic value differentiation
- [ ] **14A.1** 🧪 Create test suite: same sentence in affirmative (value 1), negated (value 2), and uncertain (value 3) forms → verify distinct aggregate vectors with correct epistemic slots
- [ ] **14A.2** 🧪 Test that `IRRELEVANT=0` only appears in slots genuinely non-applicable (e.g., `SPATIAL_RCC_*` for non-spatial propositions)
- [ ] **14A.3** 🧪 Test composition: join of an affirmed and negated proposition produces `UNKNOWN=3` in the correct slots

#### 14B — Semantic collision testing
- [ ] **14B.1** 🧪 Test distinct concepts produce distinct vectors: "dog" ≠ "cat" ≠ "car" ≠ "tree" across 256 ConceptNet dimensions
- [ ] **14B.2** 🧪 Test near-synonym differentiation: "happy" vs "joyful" → vectors differ in at least one canonical slot
- [ ] **14B.3** 🧪 Test ontological class separation: all animals share `CN_Q011_ANIMAL=1` and `TYPE_ANIMATE=1` but differ in specific discriminative dimensions
- [ ] **14B.4** 🧪 Test 23,383 singleton concepts in Tier 1 codebook have 0% collision with one another

#### 14C — Structural coverage testing
- [ ] **14C.1** 🧪 Test that simple declarative sentences activate Band 0 + Band 1 + Band 3/4 (ConceptNet `CN_Q*` slots) but leave non-applicable bands at `0`
- [ ] **14C.2** 🧪 Test that FOL formulas activate Band 2 quantifier/connective/variable slots
- [ ] **14C.3** 🧪 Test that code ASTs activate Band 1 AST topology slots (`GRAPH_FUNCTION_DEF`, `GRAPH_CONTROL_LOOP`, etc.)
- [ ] **14C.4** 🧪 Test that spatial propositions ("X is inside Y") activate Band 7 RCC-8 slots
- [ ] **14C.5** 🧪 Test that temporal propositions ("X happened before Y") activate Band 7 Allen Temporal slots
- [ ] **14C.6** 🧪 Test that causal propositions ("X caused Y") activate Band 7 Pearl Causal slots

#### 14D — Edge case & adversarial testing
- [ ] **14D.1** 🧪 Test empty/null input → graceful failure, not crash
- [ ] **14D.2** 🧪 Test very long sentences (50+ words) → parser still produces valid graph
- [ ] **14D.3** 🧪 Test ambiguous sentences ("I saw the man with the telescope") → verify parser makes a deterministic choice
- [ ] **14D.4** 🧪 Test sentences with multiple negations ("I did not not go") → verify correct epistemic value
- [ ] **14D.5** 🧪 Test type violation sentences ("The idea ran quickly") → verify validator rejects or flags
- [ ] **14D.6** 🧪 Test multi-clause sentences ("The cat sat on the mat and the dog barked") → verify correct multi-root or conjunction structure

---

## Phase 15: Synthetic Data Generation (Training Pipeline Stage 1)

### Context & Architectural Rationale
Following neuro-symbolic dataset synthesis literature (e.g., FormalGeo, VERUS-LM), Stage 1 of the training roadmap converts the benchmark corpus (FOLIO, ProofWriter, bAbI, CLUTRR, Python ASTs) into normalized, paired training representations:
$$(\text{Input Context} \longleftrightarrow \text{Target Mentalese ASG Canvas } \mathbf{X}_0 \in \{0, 1, 2, 3\}^{M \times 1024})$$
Entities and numerical literals are normalized into symbolic placeholders (`ENT_1`, `ENT_2`, `NUM_1`) to force the neural model to learn invariant structural reasoning rather than memorizing surface tokens.

### Implementation Guidelines
- Implement abstraction normalizers for named entities and numeric quantities.
- Stream propositions through the forward parser to generate clean ground-truth ASG canvases.
- Export balanced, shuffled datasets in JSON Lines or Parquet format for Stage 2/3 training.

### Verification & Acceptance Criteria
- Dataset spot-check: 50 randomly sampled pairs verify 100% syntactic validity and correct placeholder substitution.
- Exported dataset contains $>100,000$ validated training pairs.

### Checklist
- [ ] **15.1** Implement abstraction normalizer: replace entity names with normalized placeholders (`ENT_1`, `ENT_2`)
- [ ] **15.2** Implement numeric value normalizer: replace specific numbers with symbolic placeholders
- [ ] **15.3** Process FOLIO dataset through forward parser → generate `(NL_input, ASG_target)` pairs
- [ ] **15.4** Process ProofWriter dataset → generate `(rule_chain, ASG_target)` pairs
- [ ] **15.5** Process bAbI tasks → generate `(story_fragment, ASG_target)` pairs
- [ ] **15.6** Process CLUTRR dataset → generate `(relation_chain, ASG_target)` pairs
- [ ] **15.7** Process Python AST corpus → generate `(code_text, ASG_target)` pairs
- [ ] **15.8** Balance and shuffle synthetic dataset; export to training-ready format (JSON lines or parquet)
- [ ] **15.9** 🧪 Spot-check 50 random pairs for correctness (manual inspection or automated sanity checks)

---

## Phase 16: Discrete Diffusion Backbone (Fast-dLLM v2)

### Context & Architectural Rationale
QUANTA generates graph structures using non-autoregressive discrete diffusion over categorical state spaces (derived from SEDD, LLaDA, and Fast-dLLM):
- **Forward Corruption Process:** Replaces quaternary slots in target canvas $\mathbf{X}_0 \in \{0, 1, 2, 3\}^{M \times 1024}$ with a special `MASK` token according to continuous-time noise schedule $t \in [0, 1]$.
- **Bidirectional Transformer Backbone:** Employs full self-attention (non-causal) to predict clean slot distributions $p_\theta(\mathbf{X}_0 \mid \mathbf{X}_t, \mathbf{c})$ given prompt context $\mathbf{c}$.
- **Parallel Iterative Sampling:** Starts at $t=1$ (fully masked) and iteratively unmasks high-confidence slots in parallel across $K$ denoising steps ($K \ll N$), bypassing the sequential $\mathcal{O}(N)$ bottleneck of autoregressive generation.

```text
Canvas (t=1.0): [ MASK MASK MASK ... MASK ]  (100% Masked)
      │
      ▼ Denoise Step 1 (Parallel confidence prediction)
Canvas (t=0.5): [   1     0   MASK ...   2   ]  (High-confidence slots locked)
      │
      ▼ Denoise Step 2 (Contextual unmasking)
Canvas (t=0.0): [   1     0     3  ...   2   ]  (Fully formed Quanta ASG)
```

### Implementation Guidelines
- `fast_dllm.py`: Implement PyTorch bidirectional transformer module with quaternary token embedding layer (states: $\{0, 1, 2, 3, \text{MASK}\}$).
- Target-Side Masking (ddm-SFT): Prompts remain unmasked; noise schedule applies exclusively to target ASG blocks.
- Iterative sampling loop with confidence-based remasking and configurable step count $K \in \{8, 16, 32, 64\}$.

### Verification & Acceptance Criteria
- Forward corruption sanity check: At $t=0$ canvas is clean; at $t=1$ canvas is $100\%$ masked.
- Training loss on synthetic subset decreases monotonically across training epochs.
- Parallel generation throughput targets $180\text{--}450$ text-equivalent tokens/second on an RTX 3070 GPU.

### Checklist
#### 16A — Model architecture
- [ ] **16A.1** Implement bidirectional transformer backbone: multi-head self-attention with full (non-causal) attention mask
- [ ] **16A.2** Implement input embedding for quaternary tokens: embed each of `{0,1,2,3,MASK}` per slot position
- [ ] **16A.3** Implement canvas representation: `M × 1024` grid input (M nodes, each 1024 quaternary dimensions)
- [ ] **16A.4** Implement output head: predict probability distribution over `{0,1,2,3}` for each masked slot ($1024 \times 4 = 4096$ logits per node block)
- [ ] **16A.5** Implement conditioning input pathway: encode prompt/context vectors as additional attention tokens

#### 16B — Forward noise process
- [ ] **16B.1** Implement forward corruption kernel: `q(X_t | X_0)` replacing slots with `MASK` token at rate controlled by noise schedule
- [ ] **16B.2** Implement continuous-time noise schedule: `t ∈ [0,1]`, linear or cosine masking rate
- [ ] **16B.3** 🧪 Test: at `t=0` canvas is clean, at `t=1` canvas is fully masked, intermediate `t` has partial masking

#### 16C — Reverse denoising & training
- [ ] **16C.1** Implement training loop: sample `t ~ U(0,1)`, corrupt canvas, predict clean state, compute cross-entropy loss
- [ ] **16C.2** Implement training data loader for synthetic `(input, ASG_target)` pairs from Phase 15
- [ ] **16C.3** Implement mixed-precision training (BF16/FP16)
- [ ] **16C.4** 🧪 Train on small synthetic subset (1K examples); verify loss decreases

#### 16D — Inference sampling
- [ ] **16D.1** Implement iterative parallel decoding: at each step, predict all slots, unmask highest-confidence slots
- [ ] **16D.2** Implement confidence-based remasking: low-confidence predictions stay masked for next iteration
- [ ] **16D.3** Implement step-count scheduler: configurable number of denoising steps `K` (e.g., 8, 16, 32, 64)
- [ ] **16D.4** 🧪 Test: given a partially corrupted canvas from training data, model recovers a valid ASG
- [ ] **16D.5** 🧪 Measure generation throughput (nodes/second) for `M ∈ {16, 32, 64}`

---

## Phase 17: Logic Tensor Network (LTN) Loss

### Context & Architectural Rationale
While discrete diffusion generates candidate graph tokens, unguided neural sampling can propose syntactically illegal or ontologically contradictory vectors. Logic Tensor Networks (LTNs) provide differentiable neuro-symbolic guidance during Stage 3 training (ddm-SFT). First-Order Logic domain axioms are compiled into PyTorch computation graphs using real-logic t-norms (e.g., Łukasiewicz t-norm), computing an auxiliary rule satisfaction loss:
$$\mathcal{L}_{\text{LTN}} = 1 - \text{SatAgg}(\phi_1, \dots, \phi_m)$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{LTN}}$$
This loss directly penalizes type and valency violations during early denoising steps, shaping the diffusion trajectory toward valid structural regions.

### Implementation Guidelines
- `ltn_loss.py`: Define differentiable predicate grounding mapping quaternary slot probabilities to LTN truth values $[0, 1]$.
- Formulate core ontological rules (e.g., $\forall x (\text{AGENT}(x) \rightarrow \text{ANIMATE}(x))$) as differentiable LTN formulas.
- Integrate $\mathcal{L}_{\text{LTN}}$ into the Stage 3 diffusion training loop.

### Verification & Acceptance Criteria
- Denoising models trained with auxiliary LTN loss exhibit $>80\%$ fewer type and valency violations on unconstrained generation compared to pure cross-entropy baselines.

### Checklist
- [ ] **17.1** Install `ltn` Python package (Logic Tensor Networks)
- [ ] **17.2** Define grounding: map quaternary vector slots to LTN variables
- [ ] **17.3** Compile core domain axioms (type constraints, valency rules) into LTN formulas with Łukasiewicz t-norms
- [ ] **17.4** Implement `SatAgg` aggregation: `L_LTN = 1 - SatAgg(φ_1, ..., φ_m)`
- [ ] **17.5** Integrate `L_LTN` as auxiliary loss term during diffusion training (Phase 16C)
- [ ] **17.6** 🧪 Test: training with LTN loss produces fewer type-violating graphs than without

---

## Phase 18: Closed-Loop Repair Integration

### Context & Architectural Rationale
To fulfill the **zero structural and logical hallucination guarantee**, proposed candidate graphs must pass the $s(\text{CASP})$ symbolic verification gate. When a violation is detected:
1. The solver extracts the **Minimal Unsatisfiable Core (MUC)**—the smallest set of conflicting graph nodes or axioms.
2. The MUC is converted into a binary mask over the canvas.
3. The diffusion backbone executes a **targeted re-denoising pass** exclusively over the corrupted slots while keeping valid nodes locked.
4. Outputs are released to user-facing realizers only after achieving $100\%$ formal proof verification.

```text
               ┌──────────────────────────────────────────────┐
               │    Fast-dLLM Discrete Diffusion Proposer     │
               └──────────────────────┬───────────────────────┘
                                      │ Proposed ASG Canvas
                                      ▼
               ┌──────────────────────────────────────────────┐
               │   s(CASP) / PyClingo ASP Verification Gate   │
               └──────────────┬───────────────────────────────┘
                              │
             ┌────────────────┴────────────────┐
             │                                 │
     [Pass / Valid]                   [Conflict / Violation]
             │                                 │
             ▼                                 ▼
┌───────────────────────────┐    ┌───────────────────────────┐
│ Commit to Graph Memory /  │    │ Extract Minimal           │
│ Render Deterministic NLG  │    │ Unsatisfiable Core (MUC)  │
└───────────────────────────┘    └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ Re-mask Invalid Nodes &   │
                                 │ Re-Denoise (Targeted Fix) │
                                 └─────────────┬─────────────┘
                                               │
                                               └─► (Feedback Loop)
```

### Implementation Guidelines
- Implement closed-loop repair manager orchestrating: Propose $\to$ Validate $\to$ MUC Mask $\to$ Re-Denoise $\to$ Re-Validate.
- Maintain a maximum retry limit (e.g., 5 iterations) with diagnostic telemetry.

### Verification & Acceptance Criteria
- Injected error test: Inject an invalid node (e.g., rock agent) into a valid 10-node graph; repair loop isolates the node via MUC and repairs it within 1–3 iterations without disturbing valid nodes.
- Multiple conflict test: Resolves complex multi-node contradictions or reports a formal diagnostic proof trace upon unrecoverable failure.

### Checklist
- [ ] **18.1** Implement repair loop: diffusion proposer → validator gate → if MUC → re-mask invalid nodes → re-denoise
- [ ] **18.2** Implement re-masking: convert MUC node list to binary mask over canvas slots
- [ ] **18.3** Implement targeted re-denoising: run diffusion backbone only on re-masked slots (keep valid slots locked)
- [ ] **18.4** Implement max-retry limit (e.g., 5 iterations) with fallback error report
- [ ] **18.5** 🧪 Test: inject one invalid node into a valid canvas → repair loop fixes it within 1–3 iterations
- [ ] **18.6** 🧪 Test: inject multiple conflicting nodes → repair loop resolves all conflicts or reports failure

---

## Phase 19: SEPO RL Fine-Tuning (Stage 4)

### Context & Architectural Rationale
Because logic solvers produce discrete, non-differentiable rewards ($+1$ for valid proof, $-1$ for contradiction), standard policy gradients fail on discrete diffusion models. Following Score Entropy Policy Optimization (SEPO), reinforcement learning fine-tunes the discrete diffusion proposer over non-differentiable solver feedback, directly maximizing the likelihood of proposing structurally sound, proof-valid ASGs on the initial denoising pass.

### Implementation Guidelines
- Implement binary and MUC-shaped reward functions.
- Formulate SEPO policy gradient estimators tailored for discrete masked diffusion transitions.
- Execute RL fine-tuning on consumer/cloud GPU setup.

### Verification & Acceptance Criteria
- Initial-pass verification pass rate on complex reasoning benchmarks improves significantly ($>25\%$ relative boost) after SEPO fine-tuning compared to pre-RL checkpoints.

### Checklist
- [ ] **19.1** Implement reward function: `+1` for s(CASP)-validated graph, `-1` for MUC violation, shaped by MUC size
- [ ] **19.2** Implement Score Entropy Policy Optimization (SEPO) gradient estimator for discrete diffusion backbone
- [ ] **19.3** Implement training loop: sample → propose canvas → validate → compute SEPO gradient → update
- [ ] **19.4** 🧪 Test: after RL fine-tuning, validation pass rate improves compared to pre-RL model

---

## Phase 20: Inference Acceleration & Deployment (Stage 5)

### Context & Architectural Rationale
To make QUANTA deployable on standard consumer workstations (RTX 3070 8GB VRAM) and cloud edge instances:
- **Step-Schedule Distillation:** Distills reverse diffusion sampling from $K=64$ steps down to $8\text{--}16$ steps without loss of proof accuracy.
- **Block-Wise KV Caching:** Reuses bidirectional attention key-value states for unmasked context tokens across sampling steps.
- **Virtual Page-Table Integration:** Seamlessly streams external graph nodes from host RAM into the active execution canvas during reverse sampling.

### Implementation Guidelines
- Train step-distilled student checkpoints.
- Implement block KV caching for PyTorch attention modules.
- Profile memory allocation across context horizons ($10^3 \to 10^6$ nodes).

### Verification & Acceptance Criteria
- Inference latency benchmarks confirm generation throughput of $180\text{--}450$ text-equivalent tokens/second.
- Physical GPU VRAM usage remains flat ($\le 5.0\text{ GB}$) while context scales to $1,000,000$ nodes in host RAM (256 MB).

### Checklist
- [ ] **20.1** Implement step-schedule distillation: train student model to match teacher in fewer steps (64 → 16 → 8)
- [ ] **20.2** Implement block-wise KV caching for the bidirectional transformer
- [ ] **20.3** Implement Virtual Page-Table integration with diffusion inference (page-in external context during denoising)
- [ ] **20.4** 🧪 Benchmark: throughput (tok/s equivalent) at step counts 8, 16, 32, 64
- [ ] **20.5** 🧪 Benchmark: VRAM usage stays constant as context grows from 1K to 100K to 1M nodes (256 MB in host RAM)

---

## Phase 21: Benchmark Evaluation 🧪

### Context & Architectural Rationale
Rigorous empirical benchmarking across formal deduction, multi-hop reasoning, efficiency, and information-theoretic metrics to empirically validate the claims of the QUANTA architecture.

| Benchmark Dataset | Domain / Target | Baseline LLMs | QUANTA Target | Primary Failure Mode Mitigated |
| :--- | :--- | :--- | :--- | :--- |
| **FOLIO** | First-Order Logic deduction | 65%–78% | **100% Proof Accuracy** | Fallacious implication / quantifier errors |
| **ProofWriter** | Multi-step rule induction ($D \le 10$) | Degrades at $D \ge 5$ | **100% Execution Correctness** | Depth limit collapse / reasoning drift |
| **bAbI Tasks** | Spatial, temporal, positional tracking | 85%–92% | **100% Task Completion** | State tracking memory failure / entity drift |
| **CLUTRR** | Inductive kinematic family reasoning | Degrades on long chains | **100% Kinematic Accuracy** | Relational composition collapse |
| **AR-LSAT** | Analytical constraint satisfaction | 45%–60% (GPT-4) | **> 92% Solver Pass Rate** | Constraint satisfaction failure |

### Implementation Guidelines
- Build automated evaluation harnesses in `tests/` for FOLIO, ProofWriter, bAbI, CLUTRR, and AR-LSAT.
- Measure generation throughput (tok/s), VRAM scaling across context horizons ($10^3 \dots 10^6$ nodes), and semantic losslessness ($d_H = 0$).
- Run Information Profiler over full benchmark outputs to verify $H(D_i) \ge 1.2\text{ bits}$, $I(D_i; D_j) < 0.8\text{ bits}$, and $CR = 0$.

### Verification & Acceptance Criteria
- All target benchmark metrics are achieved and documented in summary tables and charts.

### Checklist
#### 21A — Logical deduction benchmarks
- [ ] **21A.1** 🧪 Run FOLIO benchmark: measure proof accuracy (target: 100%)
- [ ] **21A.2** 🧪 Run ProofWriter benchmark at depths D=1,2,3,5,10: measure execution correctness (target: 100%)
- [ ] **21A.3** 🧪 Run bAbI tasks (all 20): measure task completion rate (target: 100%)
- [ ] **21A.4** 🧪 Run CLUTRR with varying chain lengths: measure kinematic accuracy (target: 100%)
- [ ] **21A.5** 🧪 Run AR-LSAT: measure solver pass rate (target: >92%)

#### 21B — Efficiency & scaling benchmarks
- [ ] **21B.1** 🧪 Measure generation throughput across block sizes `M ∈ {16, 32, 64}` — target: 180–450 text-equiv tok/s
- [ ] **21B.2** 🧪 Measure physical VRAM usage across context horizons: 10³, 10⁴, 10⁵, 10⁶ nodes — target: O(1) VRAM
- [ ] **21B.3** 🧪 Run round-trip reversibility benchmark: Hamming distance comparison on 1000 round-tripped propositions — target: distance = 0

#### 21C — Information-theoretic slot validation
- [ ] **21C.1** 🧪 Run Information Profiler on full benchmark corpus: verify `H(D_i) ≥ 1.2 bits` for all slots
- [ ] **21C.2** 🧪 Verify `I(D_i; D_j) < 0.8 bits` for all slot pairs
- [ ] **21C.3** 🧪 Verify collision rate `CR = 0` for fundamental ontological classes

---

## Phase 22: Multimodal Vision Integration (Future)

### Context & Architectural Rationale
Vision-Language Models (VLMs) suffer from severe visual hallucinations, spatial misjudgments, and object counting errors. QUANTA integrates vision by converting raw sensor inputs (images, point clouds) into Spatio-Temporal Scene Graphs (STSGs) grounded in formal mereotopology:
- Lightweight perception backbones (e.g., InternVL) extract bounding boxes and 3D coordinates.
- Spatial relations are mapped strictly to RCC-8 primitives (`SPATIAL_RCC_NON_TANG_PART`, `SPATIAL_RCC_EXT_CONNECTED`).
- Temporal interactions map to Allen Interval operators (`TEMP_ALLEN_DURING`).
- Visual Question Answering (VQA) executes deterministic graph search and $s(\text{CASP})$ spatial logic over the scene graph, guaranteeing zero visual hallucination.

### Implementation Guidelines
- Integrate visual scene graph generator emitting Quanta ASG topologies.
- Implement spatial query evaluation over RCC-8 and Allen interval slots.

### Verification & Acceptance Criteria
- Image of *"cat on table"* generates graph with `SPATIAL_RCC_NON_TANG_PART=1` between cat and table surface.
- Counting and spatial relation queries achieve $100\%$ precision via deterministic graph search.

### Checklist
- [ ] **22.1** Integrate lightweight vision backbone (e.g., InternVL) for object detection and attribute extraction
- [ ] **22.2** Implement Scene Graph Generator: bounding boxes + spatial relations → Visual Scene Graph (VSG)
- [ ] **22.3** Map spatial relations to RCC-8 primitives (Band 7 slots)
- [ ] **22.4** Map temporal relations to Allen Interval operators (Band 7 slots)
- [ ] **22.5** Implement VSG → QuantaGraph conversion
- [ ] **22.6** 🧪 Test: image of "cat on table" → graph with `SPATIAL_RCC_NON_TANG_PART=1` between cat and table
- [ ] **22.7** 🧪 Test zero-hallucination VQA: "How many cats are in the image?" → exact count via graph search

---

## Phase 23: Publication Writing

### Context & Architectural Rationale
Strategic dissemination across two high-impact conference targets:
1. **Primary Submission (NeSy / ACL):** *The Mentalese Paradigm: Epistemic 4-Valued Metalanguage for Verifiable Neuro-Symbolic Reasoning*
   - Focus: Theoretical grounding, 1024-dimension layout (8 bands of 128 slots), ConceptNet 5.7.0 256-D data-driven grounding, Belnap logic, top-down coinductive $s(\text{CASP})$ verification with alias bridging, MUC repair, 2-tier polyglot surface realization, and 100% accuracy on FOLIO/ProofWriter.
2. **Secondary Submission (NeurIPS / EMNLP):** *Non-Autoregressive Discrete Graph Diffusion over Strongly-Typed Quaternary Abstract Syntax Graphs*
   - Focus: Fast-dLLM v2 discrete diffusion engine, Finite Scalar Quantization (FSQ) link, parallel block decoding throughput, and Virtual Page-Table Attention ($\mathcal{O}(1)$ VRAM context scaling).

### Implementation Guidelines
- Structure LaTeX manuscripts with clear narrative flow, formal definitions, empirical benchmark tables, and supplementary proof traces.
- Prepare reproducible code artifact repositories.

### Verification & Acceptance Criteria
- Complete draft manuscripts, high-resolution figures, proof traces, and slot layout tables ready for peer review submission.

### Checklist
- [ ] **23.1** Draft primary paper abstract and introduction (NeSy / ACL target)
- [ ] **23.2** Write representational grounding section (1024-dim layout, 8-band taxonomy, ConceptNet 5.7.0 256-D U-ODS derivation, Belnap 4-valued logic, FSQ connection, empirical Pareto dimension sweep proof)
- [ ] **23.3** Write neuro-symbolic gate section (LTN + s(CASP) + alias bridging + MUC repair)
- [ ] **23.4** Write bidirectional surface realization section (English NLG with 2-tier SIMD decoding, FOL, Code)
- [ ] **23.5** Compile benchmark results tables (FOLIO, ProofWriter, bAbI, CLUTRR, AR-LSAT)
- [ ] **23.6** Write conclusion and AGI alignment discussion
- [ ] **23.7** Draft secondary paper (NeurIPS / EMNLP target): Fast-dLLM engine + FSQ + Page-Table Attention
- [ ] **23.8** Prepare supplementary materials: slot layout tables, ConceptNet 256-D solver derivations, proof trace examples, code listings
