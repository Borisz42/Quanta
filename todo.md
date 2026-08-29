# QUANTA Implementation TODO

> [!NOTE]
> Every item below is scoped to fit a single git commit. Items marked with 🧪 include an explicit testing/evaluation step for the Mentalese language or system behavior.

---

## Phase 0: Project Scaffolding & Tooling

- [x] **0.1** Create Python project skeleton with `pyproject.toml`, `src/quanta/` package, and `tests/` directory
- [x] **0.2** Set up dev dependencies: `pytest`, `numpy`, `blake3`, `spacy`, `nltk` (WordNet corpus)
- [x] **0.3** Add a `Makefile` / `justfile` with targets: `lint`, `test`, `format`
- [x] **0.4** Add `.gitignore` for Python, data caches, and model checkpoints
- [x] **0.5** Create `data/` directory structure: `data/raw/`, `data/validation_corpus/`, `data/virtual_page_table/`
- [x] **0.6** Create `output/` directory structure for profiler reports and dimension exports

---

## Phase 1: Core Data Types & Quaternary Algebra

### 1A — QuaternaryValue enum & lattice operations

- [x] **1A.1** Define `QuaternaryValue` enum: `IRRELEVANT=0`, `TRUE=1`, `FALSE=2`, `UNKNOWN=3`
- [ ] **1A.2** Implement knowledge ordering comparator: `0 ≤_k {1,2} ≤_k 3`
- [ ] **1A.3** Implement truth ordering comparator: `2 ≤_t {0,3} ≤_t 1`
- [ ] **1A.4** Implement lattice `join (⊔)` operation (slot-wise, element-wise max in knowledge order)
- [ ] **1A.5** Implement lattice `meet (⊓)` operation (slot-wise, element-wise min in knowledge order)
- [ ] **1A.6** 🧪 Write unit tests for all lattice algebra properties: commutativity, associativity, idempotency, absorption
- [ ] **1A.7** 🧪 Write tests verifying `join`/`meet` produce correct results for all 16 pair combinations of `{0,1,2,3}`

### 1B — QuantaVector (256-dimension quaternary array)

- [x] **1B.1** Define `QuantaVector` class wrapping a 256-element `numpy.uint8` array (values constrained to `{0,1,2,3}`)
- [ ] **1B.2** Implement slot-wise `join` and `meet` between two `QuantaVector` instances
- [x] **1B.3** Implement Hamming distance computation between two `QuantaVector` instances
- [x] **1B.4** Implement `to_bytes()` / `from_bytes()` serialization (64-byte compact form: 2 bits per value)
- [ ] **1B.5** Implement `__eq__`, `__hash__`, and `__repr__` for `QuantaVector`
- [x] **1B.6** 🧪 Write unit tests: zero vector, full-TRUE vector, mixed vectors, round-trip serialization
- [x] **1B.7** 🧪 Test that Hamming distance is 0 only for identical vectors and symmetric

---

## Phase 2: Canonical Slot Layout (256 Dimensions)

### 2A — Slot definitions

- [x] **2A.1** Define Band 0 slot constants (0–63): NSM Primes, Actions, Descriptors, Kinematics, Space
- [x] **2A.2** Define Band 1 slot constants (64–127): Structural Valencies, Lojban Connectives, ASG/AST Topology
- [x] **2A.3** Define Band 2 slot constants (128–191): Ontological Signatures, Theory of Mind, WordNet Root Categories
- [x] **2A.4** Define Band 3 slot constants (192–255): Epistemic Bounds, s(CASP) Proof Solver, Allen Temporal, RCC-8 Spatial, Pearl Causal, Higher-Order Modal/LTL
- [x] **2A.5** Export all 256 slot definitions to `output/canonical_slots_layout.json`
- [x] **2A.6** 🧪 Write test verifying no duplicate slot indices, all 256 indices covered, and band boundaries are correct

### 2B — Strongly-typed valency signatures

- [x] **2B.1** Define type constraint registry mapping predicate argument slots to required ontological types (e.g., `VAL_X1_AGENT` → `+ANIMATE_AGENT`)
- [ ] **2B.2** Implement `validate_valency(node, slot, child_vector) → bool` checking type constraints
- [x] **2B.3** 🧪 Write test: `THINK(rock)` should be rejected (rock is `INANIMATE`), `THINK(human)` should pass

---

## Phase 3: ASG Node & Graph Structure

### 3A — QuantaNode

- [x] **3A.1** Define `QuantaNode` dataclass: `node_cid: bytes`, `parent_cid: bytes | None`, `semantic_vector: QuantaVector`, `concept_label: str`, `literal: str`
- [x] **3A.2** Add edge table to `QuantaNode`: list of `(relation_type: str, child_cid: bytes)` tuples
- [x] **3A.3** Add anchor/literal payload field: optional WordNet synset ID, FrameNet role, or raw literal value
- [x] **3A.4** Implement BLAKE3 CID computation: `CID(node) = BLAKE3(vector ∥ payload ∥ concat(edge_type ∥ child_CID))`
- [x] **3A.5** 🧪 Write test: same node content → same CID; any bit flip → different CID

### 3B — QuantaGraph (ASG container)

- [x] **3B.1** Define `QuantaGraph` class: stores `dict[bytes, QuantaNode]` keyed by CID, with a `root_cid` field
- [x] **3B.2** Implement `add_node()` that computes CID and inserts into graph
- [ ] **3B.3** Implement `get_children(node_cid) → list[QuantaNode]`
- [x] **3B.4** Implement tree-aggregate vector: `v_tree = ⊔ (join) of all node vectors in the graph`
- [x] **3B.5** Implement graph serialization to JSON and deserialization
- [x] **3B.6** 🧪 Build the "golden retriever bit the mailman" example graph from README Example A; verify aggregate vector matches documented values
- [x] **3B.7** 🧪 Build the negated example (Example B: "The dog did not bite the mailman"); verify `NSM_DO=2`, `LJB_NA_NEGATION=2` in aggregate
- [ ] **3B.8** 🧪 Build the uncertainty example (Example C: "Did the dog perhaps bite a mailman?"); verify `NSM_DO=3`, `GRAPH_QUERY_TARGET=3` in aggregate

---

## Phase 4: Merkle-Tree Sub-Graph Folding

- [x] **4.1** Implement recursive sub-graph CID folding: `CID(G_sub) = BLAKE3(⊕ v_node ∥ Adj(G_sub))`
- [ ] **4.2** Implement `fold_subgraph(graph, subtree_root_cid) → folded_cid` that replaces a subtree with a CID pointer node
- [ ] **4.3** Implement `unfold_subgraph(folded_cid, storage) → QuantaGraph` that restores the subtree from storage
- [ ] **4.4** 🧪 Test fold → unfold round-trip: CID matches, restored graph is identical to original
- [x] **4.5** 🧪 Test that modifying any node in the subtree changes the folded CID (tamper detection)

---

## Phase 5: Offline Data & Lexical Resources

### 5A — WordNet offline cache

- [x] **5A.1** Write script to build SQLite cache of WordNet synsets, hypernym paths, and top-level root categories
- [x] **5A.2** Implement `resolve_synset(word, pos) → synset_id` lookup function
- [x] **5A.3** Implement `get_hypernym_path(synset_id) → list[synset_id]` for ontological classification
- [x] **5A.4** Implement `get_wordnet_root_category(synset_id) → Band2 slot index` mapping synsets to Band 2 WordNet root slots
- [x] **5A.5** 🧪 Test: "dog" → `wn:dog.n.01`, hypernym path includes `animal.n.01` → `WN_ANIMAL_FAUNA` slot

### 5B — FrameNet valency templates

- [x] **5B.1** Write script to export FrameNet frames and roles to `data/framenet_valency.json`
- [ ] **5B.2** Implement `resolve_frame_roles(verb_lemma) → dict[role_name, Band1_slot]` mapping
- [ ] **5B.3** 🧪 Test: "bite" maps to frame with Agent, Patient roles → `VAL_X1_AGENT`, `VAL_X2_PATIENT`

### 5C — Typo-tolerant lexical grounding

- [x] **5C.1** Implement fuzzy Damerau-Levenshtein matcher for surface tokens against WordNet lemma index
- [x] **5C.2** Implement compound word healing (split/merge heuristics for multi-word expressions)
- [x] **5C.3** 🧪 Test: "glden retreiver" → `wn:golden_retriever.n.01`, "maileman" → `wn:mailman.n.01`, "graden" → `wn:garden.n.01` (README Example D)

---

## Phase 6: Forward Parser (NL → Mentalese ASG)

### 6A — spaCy NLP pipeline integration

- [x] **6A.1** Download and configure spaCy English model (`en_core_web_sm` or `en_core_web_md`)
- [x] **6A.2** Implement `parse_dependency_tree(text) → spaCy Doc` wrapper
- [x] **6A.3** Implement `extract_subject_verb_object(doc) → (subject, verb, object, modifiers)` from dependency parse
- [x] **6A.4** 🧪 Test: "A golden retriever bit the mailman in the garden" extracts correct SVO + location

### 6B — spaCy → QuantaGraph forward mapping

- [x] **6B.1** Implement root predicate node creation from main verb + WordNet synset resolution
- [x] **6B.2** Implement agent child node creation from subject + ontological type resolution
- [x] **6B.3** Implement patient child node creation from object + ontological type resolution
- [x] **6B.4** Implement modifier/location/temporal child node creation from prepositional phrases
- [x] **6B.5** Implement tense detection and mapping to Band 1 tense slots (`LJB_PU_PAST_TENSE`, etc.)
- [x] **6B.6** Implement negation detection (`not`, `n't`) → set `LJB_NA_NEGATION=2` and flip relevant Band 0 primes to `2`
- [x] **6B.7** Implement question/uncertainty detection → set `GRAPH_QUERY_TARGET=3`, relevant primes to `3`
- [x] **6B.8** Implement determiner resolution: "a" → `NSM_ONE=1`, "the" → `NSM_THIS=1`, "every" → `NSM_ALL=1`
- [x] **6B.9** 🧪 Full forward-parse test for Example A ("golden retriever bit the mailman in the garden") — verify all node vectors match README
- [x] **6B.10** 🧪 Full forward-parse test for Example B ("The dog did not bite the mailman") — verify negation slots
- [ ] **6B.11** 🧪 Full forward-parse test for Example C ("Did the dog perhaps bite a mailman?") — verify query/uncertainty slots

### 6C — FOL formula parser

- [x] **6C.1** Implement tokenizer for FOL strings: `∀`, `∃`, `→`, `∧`, `∨`, `¬`, `(`, `)`, predicates, variables
- [x] **6C.2** Implement recursive descent parser producing an AST of FOL expressions
- [x] **6C.3** Implement FOL AST → QuantaGraph conversion (universal quantifier → `LJB_RO_ALL_QUANT=1`, implication → `LJB_GANAI_IF_THEN=1`, etc.)
- [x] **6C.4** 🧪 Test: `∀x(Dog(x) → Animal(x))` → QuantaGraph matching README Example E

### 6D — Python AST parser

- [x] **6D.1** Implement Python source → `ast.parse()` → walk AST nodes
- [x] **6D.2** Map `FunctionDef` → `GRAPH_FUNCTION_DEF=1`, `GRAPH_SCOPED_CONTEXT=1`
- [x] **6D.3** Map `If` → `GRAPH_BRANCH_COND`, `GRAPH_BRANCH_THEN`, `GRAPH_BRANCH_ELSE`
- [x] **6D.4** Map `Return` → `GRAPH_RETURN_VALUE=1`
- [x] **6D.5** Map function arguments → `GRAPH_VARIABLE_BIND=1`, `GRAPH_ARGUMENT_LIST=1`
- [ ] **6D.6** Detect recursive calls → `GRAPH_RECURSIVE_REF=1` with self-CID pointer
- [x] **6D.7** Map loops → `GRAPH_CONTROL_LOOP=1`
- [ ] **6D.8** Map exception handling → `GRAPH_EXCEPTION_HANDLE=1`
- [ ] **6D.9** 🧪 Test: `factorial(n)` function → QuantaGraph matching README Example F

---

## Phase 7: Neuro-Symbolic Verification Gate

### 7A — Clingo / PyClingo ASP rules

- [x] **7A.1** Install `clingo` Python bindings
- [x] **7A.2** Write basic ontological integrity rules in `scasp_rules.lp`: animate-only agents, type exclusivity constraints
- [x] **7A.3** Write domain/range validation rules: e.g., `ABSTRACT_CONCEPT` cannot have `AGENT_CAPABLE=1`
- [x] **7A.4** 🧪 Test: "The rock thinks" graph triggers constraint violation; "The human thinks" passes

### 7B — Extended invariant rules

- [x] **7B.1** Write RCC-8 spatial mereotopology consistency rules (e.g., `DISCONNECTED` and `NON_TANG_PART` are mutually exclusive)
- [x] **7B.2** Write Allen interval temporal calculus consistency rules (e.g., `BEFORE` and `AFTER` are inverses)
- [ ] **7B.3** Write Pearl causal hierarchy exclusivity rules
- [ ] **7B.4** Write Band 1 structural topology rules (e.g., `GRAPH_ROOT_NODE` and `GRAPH_LEAF` are mutually exclusive on same node)
- [x] **7B.5** 🧪 Test each rule category with valid and invalid graph inputs

### 7C — Validator gate & MUC extraction

- [x] **7C.1** Implement `ValidatorGate` class that accepts a `QuantaGraph` and runs Clingo solving
- [x] **7C.2** Implement MUC (Minimal Unsatisfiable Core) extraction: identify the smallest set of conflicting nodes/axioms
- [x] **7C.3** Implement `validate(graph) → (is_valid: bool, muc: list[node_cid] | None)` API
- [x] **7C.4** 🧪 Test MUC extraction: create a graph with one invalid node among valid ones; verify MUC pinpoints exactly the invalid node

### 7D — s(CASP) / SWI-Prolog integration (optional advanced path)

- [ ] **7D.1** Install SWI-Prolog with s(CASP) pack
- [ ] **7D.2** Write s(CASP) Prolog equivalents of the Clingo rules in `scasp_rules.pl`
- [ ] **7D.3** Implement Python↔Prolog subprocess bridge for s(CASP) invocation
- [ ] **7D.4** 🧪 Test coinductive reasoning: validate a cyclic graph (recursive function ASG) that standard grounding would reject

---

## Phase 8: Reverse Realizers (Mentalese → Surface Forms)

### 8A — English Realizer

- [x] **8A.1** Implement ASG traversal: root predicate → verb, X1 → subject, X2 → object
- [x] **8A.2** Implement tense inflection from Band 1 tense slots (past/present/future)
- [x] **8A.3** Implement negation surface form: `LJB_NA_NEGATION=2` → "did not [verb]"
- [x] **8A.4** Implement determiner generation: `NSM_ONE=1` → "a", `NSM_THIS=1` → "the", `NSM_ALL=1` → "every"
- [x] **8A.5** Implement prepositional phrase generation from location/temporal child nodes
- [x] **8A.6** 🧪 Test: Example A graph → "A golden retriever bit the mailman in the garden."
- [x] **8A.7** 🧪 Test: Example B graph → "The dog did not bite the mailman."
- [ ] **8A.8** 🧪 Test: Example C graph → "Did the dog perhaps bite a mailman?"

### 8B — Hungarian Realizer

- [x] **8B.1** Implement vowel harmony classifier: back (`a,á,o,ó,u,ú`) vs front (`e,é,i,í,ö,ő,ü,ű`)
- [x] **8B.2** Implement accusative suffix: `-t` / `-ot` / `-et` / `-öt` based on vowel harmony and final consonant
- [x] **8B.3** Implement inessive suffix: `-ban` (back) / `-ben` (front)
- [x] **8B.4** Implement instrumental suffix: `-val` (back) / `-vel` (front) with consonant assimilation
- [x] **8B.5** Implement definite/indefinite conjugation selection based on determiner slots
- [ ] **8B.6** Implement verbal prefix handling (`meg-`, `el-`, `ki-`) with negation splitting (`nem harapta meg`)
- [x] **8B.7** 🧪 Test: Example A graph → "A golden retriever a kertben megharapta a postást."
- [ ] **8B.8** 🧪 Test: Example B graph → "A kutya nem harapta meg a postást."
- [ ] **8B.9** 🧪 Test: Example C graph → "Vajon a kutya megharapott egy postást?"

### 8C — FOL Emitter

- [x] **8C.1** Implement ASG → FOL string conversion: quantifiers, connectives, predicates, variables
- [x] **8C.2** Implement operator precedence and parenthesization
- [x] **8C.3** 🧪 Test: Example E graph → `∀x(Dog(x) → Animal(x))`
- [x] **8C.4** 🧪 Test round-trip: FOL string → parse → ASG → emit → same FOL string

### 8D — Code Emitter (Python)

- [x] **8D.1** Implement ASG → Python source: function def, arguments, return, branches, loops
- [ ] **8D.2** Implement recursive call reconstruction from `GRAPH_RECURSIVE_REF` edges
- [x] **8D.3** Implement indentation and formatting for readable output
- [x] **8D.4** 🧪 Test: Example F graph → syntactically valid `factorial(n)` Python code
- [ ] **8D.5** 🧪 Test: emitted code is executable and `factorial(5) == 120`

---

## Phase 9: Round-Trip Invariance Testing 🧪

- [x] **9.1** 🧪 NL round-trip: English sentence → forward parse → ASG → English realizer → compare semantic equivalence
- [x] **9.2** 🧪 FOL round-trip: FOL string → forward parse → ASG → FOL emitter → exact string match
- [x] **9.3** 🧪 Code round-trip: Python source → AST parser → ASG → Code emitter → exec both, compare output
- [x] **9.4** 🧪 Quaternary Hamming distance test: round-tripped ASG vectors must have Hamming distance = 0 on canonical slots
- [x] **9.5** 🧪 Cross-lingual test: English → ASG → Hungarian realizer → verify Hungarian output is semantically equivalent (manual gold set of 20 sentences)
- [x] **9.6** 🧪 Negation round-trip: "X did not Y" → ASG → English → verify negation preserved
- [ ] **9.7** 🧪 Uncertainty round-trip: "Did X perhaps Y?" → ASG → English → verify question/modal preserved

---

## Phase 10: Information Profiler & Dimension Optimization

### 10A — Candidate pool builder

- [x] **10A.1** Enumerate all NSM primes (~65) as candidate dimensions
- [x] **10A.2** Enumerate WordNet base synsets (top-level hypernym categories) as candidates
- [x] **10A.3** Enumerate FrameNet thematic roles as candidates
- [x] **10A.4** Enumerate formal modal/temporal/spatial operators as candidates
- [x] **10A.5** Combine into over-complete pool (`K = 512–1024` candidates)
- [ ] **10A.6** Export pool to `output/candidate_pool.json`

### 10B — Information Profiler

- [x] **10B.1** Implement slot entropy computation: `H(D_i) = -Σ P(D_i=s) log₂ P(D_i=s)` over parsed corpus vectors
- [x] **10B.2** Implement pairwise mutual information computation: `I(D_i; D_j)` for all slot pairs
- [ ] **10B.3** Implement total correlation computation: `TC(D) = Σ H(D_i) - H(D_0,...,D_255)`
- [x] **10B.4** Implement collision rate computation: count distinct concept pairs mapping to identical vectors
- [x] **10B.5** Implement report generator: flag slots with `H(D_i) < 0.1 bits` (dead), pairs with `I(D_i;D_j) > 0.5 bits` (redundant)
- [x] **10B.6** Export report to `output/information_profiler_report.txt`

### 10C — mRMR dimension selector

- [x] **10C.1** Implement greedy forward mRMR selection: iteratively pick dimension maximizing `relevance - redundancy`
- [x] **10C.2** Run mRMR on candidate pool against validation corpus
- [x] **10C.3** Export selected optimal 256 dimensions to `output/optimal_256_dimensions.json` and `.csv`
- [x] **10C.4** 🧪 Verify selected dimensions satisfy: `H(D_i) ≥ 0.35 bits` for all i, `I(D_i;D_j) < 0.15 bits` for all i≠j, collision rate = 0

---

## Phase 11: Validation Corpus & Benchmark Data

### 11A — Corpus generation

- [x] **11A.1** Download FOLIO dataset and write ingestion script to `data/raw/`
- [x] **11A.2** Download ProofWriter dataset and write ingestion script
- [x] **11A.3** Download bAbI tasks dataset and write ingestion script
- [x] **11A.4** Download CLUTRR dataset and write ingestion script
- [x] **11A.5** Collect ~200 Python functions of varying complexity (factorial, sort, search, etc.)
- [x] **11A.6** Write `corpus_generator.py` that samples 5,000 diverse propositions across all datasets
- [x] **11A.7** 🧪 Run forward parser on all 5,000 propositions; report parse success rate

### 11B — Corpus-driven profiler validation

- [x] **11B.1** 🧪 Forward-parse entire validation corpus into quaternary tensors
- [x] **11B.2** 🧪 Run Information Profiler on parsed tensors; generate report
- [x] **11B.3** 🧪 Identify and document any dead slots or highly redundant pairs
- [x] **11B.4** 🧪 If dead/redundant slots found, propose refactoring rules (move to s(CASP) inference rules)
- [x] **11B.5** 🧪 Re-run profiler after refactoring; confirm all slots pass thresholds

---

## Phase 12: Unified Translation Pipeline

- [x] **12.1** Implement `TranslatorPipeline` class orchestrating: Input → Forward Parser → ASP Validation Gate → Merkle Addressing → Reverse Realizer
- [x] **12.2** Add pipeline mode selection: `target_format ∈ {english, hungarian, fol, python}`
- [x] **12.3** Implement error handling: parser failure, validation rejection (return MUC diagnostic)
- [ ] **12.4** Add pipeline logging: every stage emits structured log entry with timing
- [x] **12.5** 🧪 End-to-end pipeline test: "A golden retriever bit the mailman" → English round-trip
- [x] **12.6** 🧪 End-to-end pipeline test: `∀x(Dog(x) → Animal(x))` → FOL round-trip
- [x] **12.7** 🧪 End-to-end pipeline test: `def factorial(n)` → Python code round-trip
- [x] **12.8** 🧪 End-to-end pipeline test: invalid input ("The rock thinks") → validation rejection with MUC

---

## Phase 13: Virtual Page-Table Attention (Memory Offloading)

### 13A — Host-side graph storage

- [ ] **13A.1** Implement `PageTable` class: dictionary-based storage mapping `CID → QuantaNode` serialized to disk (SQLite or memory-mapped file)
- [ ] **13A.2** Implement `store(node) → CID` and `fetch(CID) → QuantaNode`
- [ ] **13A.3** Implement batch store/fetch for sub-graphs

### 13B — SIMD-accelerated Hamming lookup (software path)

- [ ] **13B.1** Implement brute-force Hamming distance scan over stored quaternary keys using NumPy vectorized operations
- [ ] **13B.2** Implement top-K nearest CID retrieval by Hamming distance
- [ ] **13B.3** 🧪 Benchmark: measure lookup latency for 1K, 10K, 100K, 1M stored nodes

### 13C — Active canvas paging simulation

- [ ] **13C.1** Implement `ActiveCanvas` class: fixed-size buffer of `M=64` nodes in-memory
- [ ] **13C.2** Implement page fault handler: when a CID pointer is encountered that isn't in canvas, fetch from `PageTable`
- [ ] **13C.3** Implement LRU eviction policy for canvas when full
- [ ] **13C.4** 🧪 Test: load a graph of 1000 nodes, verify canvas never exceeds 64 nodes in memory
- [ ] **13C.5** 🧪 Test: access patterns hitting cold/hot nodes → measure page fault rates

---

## Phase 14: Mentalese Language Testing 🧪

> [!IMPORTANT]
> These tests specifically validate the Mentalese language design—ensuring the quaternary representation is expressive, unambiguous, and semantically faithful.

### 14A — Epistemic value differentiation

- [ ] **14A.1** 🧪 Create test suite: same sentence in affirmative (value 1), negated (value 2), and uncertain (value 3) forms → verify distinct aggregate vectors with correct epistemic slots
- [ ] **14A.2** 🧪 Test that `IRRELEVANT=0` only appears in slots genuinely non-applicable (e.g., `SPATIAL_RCC_*` for non-spatial propositions)
- [ ] **14A.3** 🧪 Test composition: join of an affirmed and negated proposition produces `UNKNOWN=3` in the correct slots

### 14B — Semantic collision testing

- [ ] **14B.1** 🧪 Test distinct concepts produce distinct vectors: "dog" ≠ "cat" ≠ "car" ≠ "tree"
- [ ] **14B.2** 🧪 Test near-synonym differentiation: "happy" vs "joyful" → vectors differ in at least one canonical slot
- [ ] **14B.3** 🧪 Test ontological class separation: all animals share `TYPE_ANIMATE=1` but differ in other dimensions

### 14C — Structural coverage testing

- [ ] **14C.1** 🧪 Test that simple declarative sentences activate Band 0 + Band 1 + Band 2 but leave most of Band 3 at `0`
- [ ] **14C.2** 🧪 Test that FOL formulas activate Band 1 quantifier/connective slots
- [ ] **14C.3** 🧪 Test that code ASTs activate Band 1 AST topology slots (`GRAPH_FUNCTION_DEF`, `GRAPH_CONTROL_LOOP`, etc.)
- [ ] **14C.4** 🧪 Test that spatial propositions ("X is inside Y") activate Band 3 RCC-8 slots
- [ ] **14C.5** 🧪 Test that temporal propositions ("X happened before Y") activate Band 3 Allen Temporal slots
- [ ] **14C.6** 🧪 Test that causal propositions ("X caused Y") activate Band 3 Pearl Causal slots

### 14D — Edge case & adversarial testing

- [ ] **14D.1** 🧪 Test empty/null input → graceful failure, not crash
- [ ] **14D.2** 🧪 Test very long sentences (50+ words) → parser still produces valid graph
- [ ] **14D.3** 🧪 Test ambiguous sentences ("I saw the man with the telescope") → verify parser makes a deterministic choice
- [ ] **14D.4** 🧪 Test sentences with multiple negations ("I did not not go") → verify correct epistemic value
- [ ] **14D.5** 🧪 Test type violation sentences ("The idea ran quickly") → verify validator rejects or flags
- [ ] **14D.6** 🧪 Test multi-clause sentences ("The cat sat on the mat and the dog barked") → verify correct multi-root or conjunction structure

---

## Phase 15: Synthetic Data Generation (Training Pipeline Stage 1)

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

### 16A — Model architecture

- [ ] **16A.1** Implement bidirectional transformer backbone: multi-head self-attention with full (non-causal) attention mask
- [ ] **16A.2** Implement input embedding for quaternary tokens: embed each of `{0,1,2,3,MASK}` per slot position
- [ ] **16A.3** Implement canvas representation: `M × 256` grid input (M nodes, each 256 quaternary dimensions)
- [ ] **16A.4** Implement output head: predict probability distribution over `{0,1,2,3}` for each masked slot
- [ ] **16A.5** Implement conditioning input pathway: encode prompt/context vectors as additional attention tokens

### 16B — Forward noise process

- [ ] **16B.1** Implement forward corruption kernel: `q(X_t | X_0)` replacing slots with `MASK` token at rate controlled by noise schedule
- [ ] **16B.2** Implement continuous-time noise schedule: `t ∈ [0,1]`, linear or cosine masking rate
- [ ] **16B.3** 🧪 Test: at `t=0` canvas is clean, at `t=1` canvas is fully masked, intermediate `t` has partial masking

### 16C — Reverse denoising & training

- [ ] **16C.1** Implement training loop: sample `t ~ U(0,1)`, corrupt canvas, predict clean state, compute cross-entropy loss
- [ ] **16C.2** Implement training data loader for synthetic `(input, ASG_target)` pairs from Phase 15
- [ ] **16C.3** Implement mixed-precision training (BF16/FP16)
- [ ] **16C.4** 🧪 Train on small synthetic subset (1K examples); verify loss decreases

### 16D — Inference sampling

- [ ] **16D.1** Implement iterative parallel decoding: at each step, predict all slots, unmask highest-confidence slots
- [ ] **16D.2** Implement confidence-based remasking: low-confidence predictions stay masked for next iteration
- [ ] **16D.3** Implement step-count scheduler: configurable number of denoising steps `K` (e.g., 8, 16, 32, 64)
- [ ] **16D.4** 🧪 Test: given a partially corrupted canvas from training data, model recovers a valid ASG
- [ ] **16D.5** 🧪 Measure generation throughput (nodes/second) for `M ∈ {16, 32, 64}`

---

## Phase 17: Logic Tensor Network (LTN) Loss

- [ ] **17.1** Install `ltn` Python package (Logic Tensor Networks)
- [ ] **17.2** Define grounding: map quaternary vector slots to LTN variables
- [ ] **17.3** Compile core domain axioms (type constraints, valency rules) into LTN formulas with Łukasiewicz t-norms
- [ ] **17.4** Implement `SatAgg` aggregation: `L_LTN = 1 - SatAgg(φ_1, ..., φ_m)`
- [ ] **17.5** Integrate `L_LTN` as auxiliary loss term during diffusion training (Phase 16C)
- [ ] **17.6** 🧪 Test: training with LTN loss produces fewer type-violating graphs than without

---

## Phase 18: Closed-Loop Repair Integration

- [ ] **18.1** Implement repair loop: diffusion proposer → validator gate → if MUC → re-mask invalid nodes → re-denoise
- [ ] **18.2** Implement re-masking: convert MUC node list to binary mask over canvas slots
- [ ] **18.3** Implement targeted re-denoising: run diffusion backbone only on re-masked slots (keep valid slots locked)
- [ ] **18.4** Implement max-retry limit (e.g., 5 iterations) with fallback error report
- [ ] **18.5** 🧪 Test: inject one invalid node into a valid canvas → repair loop fixes it within 1–3 iterations
- [ ] **18.6** 🧪 Test: inject multiple conflicting nodes → repair loop resolves all conflicts or reports failure

---

## Phase 19: SEPO RL Fine-Tuning (Stage 4)

- [ ] **19.1** Implement reward function: `+1` for s(CASP)-validated graph, `-1` for MUC violation, shaped by MUC size
- [ ] **19.2** Implement Score Entropy Policy Optimization (SEPO) gradient estimator for discrete diffusion backbone
- [ ] **19.3** Implement training loop: sample → propose canvas → validate → compute SEPO gradient → update
- [ ] **19.4** 🧪 Test: after RL fine-tuning, validation pass rate improves compared to pre-RL model

---

## Phase 20: Inference Acceleration & Deployment (Stage 5)

- [ ] **20.1** Implement step-schedule distillation: train student model to match teacher in fewer steps (64 → 16 → 8)
- [ ] **20.2** Implement block-wise KV caching for the bidirectional transformer
- [ ] **20.3** Implement Virtual Page-Table integration with diffusion inference (page-in external context during denoising)
- [ ] **20.4** 🧪 Benchmark: throughput (tok/s equivalent) at step counts 8, 16, 32, 64
- [ ] **20.5** 🧪 Benchmark: VRAM usage stays constant as context grows from 1K to 100K to 1M nodes

---

## Phase 21: Benchmark Evaluation 🧪

### 21A — Logical deduction benchmarks

- [ ] **21A.1** 🧪 Run FOLIO benchmark: measure proof accuracy (target: 100%)
- [ ] **21A.2** 🧪 Run ProofWriter benchmark at depths D=1,2,3,5,10: measure execution correctness (target: 100%)
- [ ] **21A.3** 🧪 Run bAbI tasks (all 20): measure task completion rate (target: 100%)
- [ ] **21A.4** 🧪 Run CLUTRR with varying chain lengths: measure kinematic accuracy (target: 100%)
- [ ] **21A.5** 🧪 Run AR-LSAT: measure solver pass rate (target: >92%)

### 21B — Efficiency & scaling benchmarks

- [ ] **21B.1** 🧪 Measure generation throughput across block sizes `M ∈ {16, 32, 64}` — target: 180–450 text-equiv tok/s
- [ ] **21B.2** 🧪 Measure physical VRAM usage across context horizons: 10³, 10⁴, 10⁵, 10⁶ nodes — target: O(1) VRAM
- [ ] **21B.3** 🧪 Run round-trip reversibility benchmark: Hamming distance comparison on 1000 round-tripped propositions — target: distance = 0

### 21C — Information-theoretic slot validation

- [ ] **21C.1** 🧪 Run Information Profiler on full benchmark corpus: verify `H(D_i) ≥ 1.2 bits` for all slots
- [ ] **21C.2** 🧪 Verify `I(D_i; D_j) < 0.8 bits` for all slot pairs
- [ ] **21C.3** 🧪 Verify collision rate `CR = 0` for fundamental ontological classes

---

## Phase 22: Multimodal Vision Integration (Future)

- [ ] **22.1** Integrate lightweight vision backbone (e.g., InternVL) for object detection and attribute extraction
- [ ] **22.2** Implement Scene Graph Generator: bounding boxes + spatial relations → Visual Scene Graph (VSG)
- [ ] **22.3** Map spatial relations to RCC-8 primitives (Band 3 slots)
- [ ] **22.4** Map temporal relations to Allen Interval operators (Band 3 slots)
- [ ] **22.5** Implement VSG → QuantaGraph conversion
- [ ] **22.6** 🧪 Test: image of "cat on table" → graph with `SPATIAL_RCC_NON_TANG_PART=1` between cat and table
- [ ] **22.7** 🧪 Test zero-hallucination VQA: "How many cats are in the image?" → exact count via graph search

---

## Phase 23: Publication Writing

- [ ] **23.1** Draft primary paper abstract and introduction (NeSy / ACL target)
- [ ] **23.2** Write representational grounding section (256-dim layout, Belnap 4-valued logic, FSQ connection)
- [ ] **23.3** Write neuro-symbolic gate section (LTN + s(CASP) + MUC repair)
- [ ] **23.4** Write bidirectional surface realization section (English, Hungarian, FOL, Code)
- [ ] **23.5** Compile benchmark results tables (FOLIO, ProofWriter, bAbI, CLUTRR, AR-LSAT)
- [ ] **23.6** Write conclusion and AGI alignment discussion
- [ ] **23.7** Draft secondary paper (NeurIPS / EMNLP target): Fast-dLLM engine + FSQ + Page-Table Attention
- [ ] **23.8** Prepare supplementary materials: slot layout tables, proof trace examples, code listings
