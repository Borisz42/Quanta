# QUANTA (Quaternary Universal Abstract Natural Topology Architecture)

QUANTA is a neuro-symbolic framework that replaces traditional continuous floating-point LLM embeddings with a cryptographically hash-addressed, strongly-typed semantic metalanguage. It bounds generative state spaces to eliminate representation collapse, hallucination, and continuous noise accumulation during multi-step logic.

The primary generative engine is a non-autoregressive block discrete diffusion model (Fast-dLLM v2) constrained by strict neuro-symbolic verification (Logic Tensor Networks and `s(CASP)` Answer Set Programming). Invalid states trigger targeted re-denoising via Minimal Unsatisfiable Core (MUC) extraction.

---

## 1. Core Architectural Philosophy

1. **Discrete State Space ($\Sigma^{256}$):** Rather than outputting English strings, the diffusion model generates 256-dimensional quaternary vectors mapping strictly to universal logical primitives and ontological bounds.
2. **Epistemic 4-Valued Logic ($\mathcal{FOUR}$):** Every vector slot evaluates strictly as `0 (Irrelevant)`, `1 (Yes/True)`, `2 (No/False)`, or `3 (Maybe/Unknown)`.
3. **Decoupled Opaque Anchors:** Encyclopedic knowledge and specific lexical identities (e.g., `wn:golden_retriever.n.01`) are stored as immutable literal payloads via an offline k-NN lookup, keeping the diffusion synthesis ultra-lightweight and VRAM-friendly.
4. **$O(1)$ Virtual Page-Table Context:** Extracted schemas, standard library functions, and historical context are compressed into Merkle-tree CIDs and stored in host RAM for instant bitwise Hamming lookup.

---

## 2. Node Anatomy & Storage Structure

A QUANTA concept is decomposed into Abstract Syntax Graphs (ASGs) where each atomic node contains both a logical signature and structural routing data. 

* **Node Header (Structural Metadata):**
  * **Node CID:** A 256-bit BLAKE3/SHA-256 Content Identifier hash of the node's payload and edges, used for Merkle-tree graph folding.
  * **Parent CID:** A pointer to the enclosing sub-graph.
* **Semantic Vector (The 256-Dimension Logical Contract):**
  * A 64-byte payload storing 256 exact semantic, syntactic, and ontological constraints in $\{0, 1, 2, 3\}$.
* **Edge Table & Literal Payloads (The Graph Topology):**
  * **Edges:** Directed pointers linking to child CIDs with defined relation types (e.g., `VAL_X1_AGENT_ACTOR` $\to$ `CID: 0x9A4...`).
  * **Anchors & Literals:** Pointers to external lexicons (WordNet synsets, FrameNet roles) or immutable strings/timestamps.

---

## 3. The 256-Dimension Canonical Slot Layout

The 256 dimensions are strictly partitioned into four isolated bands to guarantee deterministic indexing and zero cross-band representation drift.

### Band 0: Universal NSM Primes & Kinematics (0–63)
Anchors concepts to universal Natural Semantic Metalanguage (NSM) primes.
* `00–05`: Substantives (I, YOU, SOMEONE, SOMETHING, PEOPLE, BODY)
* `06–17`: Quantifiers & Determiners (THIS, SAME, OTHER, ONE, TWO, MUCH, LITTLE, SOME, ALL, MORE)
* `18–29`: Evaluators, Descriptors, & Mental (GOOD, BAD, BIG, SMALL, VERY, TRUE, THINK, KNOW, WANT, FEEL, SEE, HEAR)
* `30–42`: Actions, Events, & Vitality (SAY, WORDS, DO, HAPPEN, MOVE, TOUCH, BE_SOMEWHERE, LIVE, DIE, BORN)
* `43–59`: Time & Space (NOW, BEFORE, AFTER, MOMENT, HERE, ABOVE, BELOW, FAR, NEAR)
* `60–61`: Continuous Kinematics (`NSM_CONTINUOUS_RATE`, `NSM_ACCELERATING_RATE`)
* `62–63`: Logic (CAN, MAYBE)

### Band 1: Structural Valencies & Concurrency Topology (64–127)
Defines argument structures, logical operations, and asynchronous execution paths.
* `64–74`: Predicate Place Structures (`VAL_X1_AGENT`, `VAL_X2_PATIENT`, `VAL_X3_DESTINATION`, `VAL_EXPERIENCER`, `VAL_TIME_SLOT`)
* `75–95`: Logical Connectives / *cmavo* (Negation, AND, OR, XOR, If-Then, Past/Present/Future tense)
* `96–98`: Asynchronous Concurrency markers (`LJB_ASYNC_CONCURRENT`, `LJB_MUTEX_DEPENDENCY`, `LJB_RACE_CONDITION`)
* `99–127`: Graph Structural Topology (`GRAPH_ROOT_NODE`, `GRAPH_LEAF`, `GRAPH_RECURSIVE_REF`)

### Band 2: Ontological Signatures & Theory of Mind Modalities (128–191)
Enforces compile-time semantic typing via WordNet and FrameNet constraints.
* `128–147`: Entity Types (Animate, Human, Abstract_Concept, Proposition, Event, Temporal_Interval, Measure_Scalar)
* `148–154`: Semantic Roles (Agent_Capable, Sentient, Moveable, Communicator)
* `155–156`: Theory of Mind / Deception (`ROLE_DECEPTIVE_PROJECTION`, `ROLE_SARCASM_IRONY`)
* `157–158`: Modality Overlays (`MODALITY_LITERAL`, `MODALITY_FIGURATIVE`)
* `171–191`: WordNet Root Categories (Action_Perception, Action_Motion, State_Relation)

### Band 3: Epistemic Bounds, Probability & Logical Metarules (192–255)
Guides the `s(CASP)` constraint solver and quantitative likelihoods.
* `192–200`: Epistemic Context (Direct Observation, Deductive Inference, Hearsay, Axiomatic, Deontic Obligation)
* `201–204`: Probabilistic Weights (`EPIST_PROB_HIGH`, `EPIST_PROB_MARGINAL`, `EPIST_PROB_DISTRIBUTED`, `EPIST_STATISTICAL_EDGE`)
* `208–213`: `s(CASP)` Solver Flags (Closed World Assumption, MUC Targeted, Proof Validated)
* `224–255`: Extension Registers for domain-specific payloads.

---

## 4. The Two-Way Translation & Execution Pipeline

```text
[ Natural Language / Intent ]
         │
         ▼
1. Forward Parser (spaCy, FrameNet, WordNet offline DB)
   - Resolves thematic structure and lexical hypernyms.
   - Outputs unverified Quanta ASG.
         │
         ▼
2. Structural Validation Gate (s(CASP) / PyClingo)
   - Checks Domain/Range violations (e.g., +ABSTRACT_CONCEPT acting as +AGENT_CAPABLE).
   - Validates Graph Merkle integrity.
         │
         ▼
3. Virtual Page-Table Graph RAG
   - Bitwise Hamming distance matching on host CPU (FAISS).
   - Inlines schemas (e.g., importing `python.builtins.list.sort`).
         │
         ▼
4. Reverse Realizer (SimpleNLG / hunmorph / Code Emitter)
   - Deterministic unrolling of Merkle pointers.
   - Outputs human syntax or executable code (Python, C++).

```

---

## 5. Examples

### Example A: Natural Language Representation

*"A brown golden retriever dog bit the postman yesterday."*

* **Root (`BITE_EVENT`):** `NSM_DO=1`, `LJB_PU_PAST_TENSE=1`, `TYPE_EVENT=1`, `MODALITY_LITERAL=1`.
* **Agent (`DOG_ENTITY`):** `TYPE_ANIMATE=1`, `ROLE_AGENT_CAPABLE=1`. Anchor: `wn:golden_retriever.n.01`.
* **Patient (`POSTMAN_ENTITY`):** `TYPE_HUMAN=1`, `VAL_EXPERIENCER=1`. Anchor: `wn:mailman.n.01`.
* *Validation:* If the LLM assigned `TYPE_ABSTRACT_CONCEPT=1` to the Dog, `s(CASP)` flags an immediate MUC contradiction without querying WordNet.

### Example B: Code Execution (Factorial Function)

Code is stored via Abstract Syntax Tree topology, not text files.

* **Root (Function Def):** `GRAPH_ROOT_NODE=1`, `TYPE_PROCESS=1`, `TYPE_MEASURE_SCALAR=1`.
* **Base Case (Branching):** `NSM_IF=1`, `NSM_THE_SAME=1`, `LJB_DU_IDENTITY=1`.
* **Recursive Call:** `GRAPH_IS_SUB_EXP=1`, `GRAPH_RECURSIVE_REF=1` (Points to Root CID).

---

## 6. Directory Structure

```bash
quanta-core/
├── data/
│   ├── wordnet_offline.db        # O(1) Anchor lookup indices
│   ├── framenet_valency.json     # Role template matrices
│   └── virtual_page_table/       # Merkle-folded library schemas
├── src/
│   ├── parser/
│   │   ├── nlp_forward.py        # spaCy -> ASG parser
│   │   └── lexical_grounder.py   # Hypernym to Band 2 resolver
│   ├── solver/
│   │   ├── scasp_rules.pl        # Prolog / s(CASP) logical invariants
│   │   └── validator_gate.py     # Subprocess MUC extractor
│   ├── models/
│   │   ├── fast_dllm.py          # Discrete Diffusion v2 backbone
│   │   └── ltn_loss.py           # Logic Tensor Network differentiable loss
│   └── realizer/
│       ├── english_nlg.py        # ASG -> SVO English mapping
│       ├── hungarian_morph.py    # ASG -> Agglutinative mapping
│       └── code_emitter.py       # ASG -> Python/Java compiler
├── tests/
│   ├── test_round_trip.py        # English -> QUANTA -> English integrity
│   ├── test_merkle_hashes.py     # BLAKE3 topological mismatch testing
│   └── test_type_constraints.py  # s(CASP) contradiction catching
└── README.md

```

---

## 7. Implementation Roadmap & Milestones

* **Phase 1: Formalization & Core Symbolic Engine**
* Implement $\Sigma = \{0, 1, 2, 3\}^{256}$ tensor layouts in PyTorch.
* Build Python `pyclingo`/`s(CASP)` validator gate and offline WordNet schema loader.


* **Phase 2: Bidirectional Realizers & Benchmarking**
* Build strict `English -> Mentalese -> English` round-trip invariance tests.
* Ensure 100% Type Constraint validation (catching logic flaws prior to generation).


* **Phase 3: Diffusion Backbone & Compilation Loop**
* Train Fast-dLLM v2 discrete diffusion masking mechanism.
* Integrate Logic Tensor Network (LTN) loss for early denoising and MUC invalidation.


* **Phase 4: Empirical Evaluation**
* Benchmark ProofWriter, CLUTRR, and FOLIO for 0% hallucination verification.
* Test context scaling (1k to 128k nodes) on sub-8GB VRAM RTX 3070 environments.


* **Phase 5: Publication Writing**
* Target conferences: NeurIPS, NeSy, ACL.

