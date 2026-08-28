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

## 4. Discrete Information Bottleneck & Dimension Testing

We frame the QUANTA 256-dimension quaternary space as a **Discrete Information Bottleneck** problem.

In information theory, an optimal discrete semantic code $\mathbf{D} = (D_0, \dots, D_{255}) \in \{0, 1, 2, 3\}^{256}$ must satisfy three formal conditions:

1. **Maximal Channel Utilization (High Individual Entropy):** No dimension should be degenerate or constant.
2. **Minimal Redundancy (Maximum Orthogonality):** Pairwise mutual information across dimensions must be minimized.
3. **Sufficient Expressive Capacity (Zero Semantic Collisions):** The vector representation must preserve enough mutual information with the underlying semantic distribution to deterministically disambiguate concepts.

---

### 4.1. Information-Theoretic Evaluation Metrics

Given a diverse sample dataset of $N$ concepts/nodes $\mathcal{D} = \{\mathbf{v}^{(1)}, \mathbf{v}^{(2)}, \dots, \mathbf{v}^{(N)}\}$ where each $\mathbf{v}^{(k)} \in \{0, 1, 2, 3\}^{256}$:

#### Metric A: Slot Entropy & Utilization ($H(D_i)$)

Evaluates whether each individual dimension $i$ is actively conveying information across the alphabet $\Sigma = \{0, 1, 2, 3\}$.

$$H(D_i) = -\sum_{s \in \{0,1,2,3\}} P(D_i = s) \log_2 P(D_i = s)$$

* **Theoretical Maximum:** $\log_2(4) = 2.0\text{ bits}$.
* **Target:** $H(D_i) \ge 0.35\text{ bits}$.
* **Failure Mode:** If $H(D_i) \approx 0$, the dimension is dead weight (e.g., it is `0` in 99.9% of all concepts) and should be pruned or merged.

---

#### Metric B: Pairwise Redundancy / Total Correlation ($\text{TC}(\mathbf{D})$)

Quantifies whether dimensions are repeating the same semantic features. The mutual information $I(D_i; D_j)$ between two distinct dimensions must approach zero:

$$I(D_i; D_j) = \sum_{s_i \in \Sigma} \sum_{s_j \in \Sigma} P(D_i = s_i, D_j = s_j) \log_2 \frac{P(D_i = s_i, D_j = s_j)}{P(D_i = s_i) P(D_j = s_j)}$$

* **Total Correlation (Redundancy):**

$$\text{TC}(\mathbf{D}) = \sum_{i=0}^{255} H(D_i) - H(D_0, D_1, \dots, D_{255})$$

* **Target:** $I(D_i; D_j) < 0.15\text{ bits}$ for all $i \neq j$.
* **Failure Mode:** If $I(D_i; D_j) \approx H(D_i)$, dimensions $i$ and $j$ are co-linear (e.g., `TYPE_HUMAN=1` always co-occurring with `TYPE_ANIMATE=1` without independent utility). They must be factored into an ontological parent-child rule in `s(CASP)` instead of consuming two separate vector slots.

---

#### Metric C: Semantic Resolvability & Collision Rate ($R_{\text{collision}}$)

Measures whether two distinctly different lexical concepts $c_a, c_b \in \mathcal{C}$ (e.g., *Dog* vs. *Wolf*, or *Promise* vs. *Order*) collapse to the identical 256-dimension vector:

$$R_{\text{collision}} = \frac{\vert{}\{(c_a, c_b) \mid c_a \neq c_b \land \mathbf{v}(c_a) = \mathbf{v}(c_b)\}\vert{}}{\binom{\vert{}\mathcal{C}\vert{}}{2}}$$

* **Target:** $R_{\text{collision}} = 0.0$ for fundamental ontological classes.
* **Permissible Boundary:** Collision is only acceptable if $c_a$ and $c_b$ are fine-grained sub-species that are intended to be differentiated solely by the literal WordNet anchor leaf (e.g., *Golden Retriever* vs. *Labrador*).

---

### 4.2. Data-Driven Dimension Selection: The mRMR Algorithm

Instead of guessing the 256 slots by hand, start with an over-complete candidate pool of $K = 512\text{ to }1024$ candidate dimensions extracted from:

* 65 NSM Primes
* 120 Top-level WordNet Base Synsets (BBN entity types)
* 200 Core FrameNet Frame Elements / Thematic Roles
* AMR (Abstract Meaning Representation) core relation types
* Formal logic operators and modal calculus

Use **Minimal Redundancy Maximal Relevance (mRMR)** to filter down to the optimal 256 dimensions:

$$\max_{S \subset \mathcal{F}, \vert{}S\vert{}=256} \left[ \frac{1}{\vert{}S\vert{}} \sum_{f_i \in S} I(f_i; Y_{\text{semantics}}) - \frac{1}{\vert{}S\vert{}^2} \sum_{f_i, f_j \in S} I(f_i; f_j) \right]$$

Where $Y_{\text{semantics}}$ is the target conceptual category, $I(f_i; Y)$ maximizes relevance/coverage, and $I(f_i; f_j)$ penalizes redundant dimensions.

---

### 4.3. Empirical Dimension Testing Harness (Python)

Use this test harness to load a batch of encoded nodes, compute their entropy, identify redundant co-linear dimensions, and flag dead slots:

```python
import numpy as np
from typing import Dict, List, Tuple

class QuantaInformationProfiler:
    def __init__(self, data_matrix: np.ndarray, dimension_labels: List[str]):
        """
        data_matrix: (N, 256) array with values in {0, 1, 2, 3}
        dimension_labels: List of 256 human-readable dimension names
        """
        self.X = data_matrix
        self.N, self.D = data_matrix.shape
        self.labels = dimension_labels
        assert self.D == 256, "Matrix must contain exactly 256 dimensions."

    def compute_entropies(self) -> np.ndarray:
        """Calculates Shannon entropy H(D_i) for each dimension in bits."""
        entropies = np.zeros(self.D)
        for i in range(self.D):
            col = self.X[:, i]
            counts = np.bincount(col, minlength=4)
            probs = counts / self.N
            probs = probs[probs > 0]
            entropies[i] = -np.sum(probs * np.log2(probs))
        return entropies

    def compute_pairwise_mutual_information(self, top_k_pairs: int = 10) -> List[Tuple[str, str, float]]:
        """Finds the most redundant pairs of dimensions using Mutual Information I(D_i; D_j)."""
        redundant_pairs = []
        
        for i in range(self.D):
            for j in range(i + 1, self.D):
                col_i = self.X[:, i]
                col_j = self.X[:, j]
                
                # Joint probability distribution P(D_i, D_j)
                joint_counts = np.zeros((4, 4))
                for v_i, v_j in zip(col_i, col_j):
                    joint_counts[v_i, v_j] += 1
                p_ij = joint_counts / self.N
                
                p_i = np.sum(p_ij, axis=1, keepdims=True)
                p_j = np.sum(p_ij, axis=0, keepdims=True)
                
                # I(D_i; D_j) = sum(p_ij * log2(p_ij / (p_i * p_j)))
                non_zero = p_ij > 0
                mi = np.sum(p_ij[non_zero] * np.log2(p_ij[non_zero] / (p_i @ p_j)[non_zero]))
                
                redundant_pairs.append((self.labels[i], self.labels[j], float(mi)))

        # Sort by highest mutual information
        redundant_pairs.sort(key=lambda x: x[2], reverse=True)
        return redundant_pairs[:top_k_pairs]

    def run_diagnostic_suite(self, dead_threshold: float = 0.1, redundancy_threshold: float = 0.5):
        """Runs automated evaluation and outputs actionable dimension refactoring advice."""
        entropies = self.compute_entropies()
        redundancies = self.compute_pairwise_mutual_information(top_k_pairs=10)

        dead_slots = [
            (idx, self.labels[idx], entropies[idx]) 
            for idx in range(self.D) if entropies[idx] < dead_threshold
        ]

        print("=== QUANTA INFORMATION PROFILER REPORT ===")
        print(f"Total Samples Evaluated: {self.N}")
        print(f"Average Dimension Entropy: {np.mean(entropies):.3f} / 2.000 bits\n")

        print(f"--- 1. Under-Utilized / Dead Dimensions (< {dead_threshold} bits) ---")
        if not dead_slots:
            print("None! All dimensions meet minimum entropy requirements.")
        else:
            for idx, label, h in dead_slots:
                print(f"  Slot {idx:03d} [{label}]: Entropy = {h:.4f} bits (Candidate for pruning/replacement)")

        print(f"\n--- 2. High Redundancy Pairs (Mutual Information > {redundancy_threshold} bits) ---")
        for dim_a, dim_b, mi in redundancies:
            if mi >= redundancy_threshold:
                print(f"  [{dim_a}] <---> [{dim_b}]: MI = {mi:.4f} bits (High co-linearity; consider collapsing)")
```

---

### 4.4. Corpus-Based Testing Workflow

To run this testing pipeline before training:

1. **Build a Validation Extraction Corpus:**
   * Extract 5,000 diverse propositions across **FOLIO**, **ProofWriter**, **bAbI**, **CLUTRR**, and standard Python/Java ASTs.
2. **Forward Map to Candidate Tensors:**
   * Parse the corpus into quaternary arrays using your rule/WordNet forward parser.
3. **Run the Information Profiler:**
   * Identify every slot with $H(D_i) < 0.1\text{ bits}$ and every pair with $I(D_i; D_j) > 0.5\text{ bits}$.
4. **Refactor & Lock:**
   * Replace dead slots with high-value discriminating concepts (e.g., domain-independent temporal or mereological relations) until the average entropy across all 256 slots is maximized.

---

## 5. The Two-Way Translation & Execution Pipeline

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

## 6. Examples

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

## 7. Directory Structure

```bash
quanta-core/
├── data/
│   ├── wordnet_offline.db        # O(1) Anchor lookup indices
│   ├── framenet_valency.json     # Role template matrices
│   ├── validation_corpus/        # 5k proposition validation set (FOLIO, bAbI, CLUTRR)
│   └── virtual_page_table/       # Merkle-folded library schemas
├── src/
│   ├── parser/
│   │   ├── nlp_forward.py        # spaCy -> ASG parser
│   │   └── lexical_grounder.py   # Hypernym to Band 2 resolver
│   ├── solver/
│   │   ├── scasp_rules.pl        # Prolog / s(CASP) logical invariants
│   │   └── validator_gate.py     # Subprocess MUC extractor
│   ├── profiler/
│   │   └── info_profiler.py      # Discrete Information Bottleneck & entropy test suite
│   ├── models/
│   │   ├── fast_dllm.py          # Discrete Diffusion v2 backbone
│   │   └── ltn_loss.py           # Logic Tensor Network differentiable loss
│   └── realizer/
│       ├── english_nlg.py        # ASG -> SVO English mapping
│       ├── hungarian_morph.py    # ASG -> Agglutinative mapping
│       └── code_emitter.py       # ASG -> Python/Java compiler
├── tests/
│   ├── test_dimension_entropy.py # Information bottleneck & redundancy validation
│   ├── test_round_trip.py        # English -> QUANTA -> English integrity
│   ├── test_merkle_hashes.py     # BLAKE3 topological mismatch testing
│   └── test_type_constraints.py  # s(CASP) contradiction catching
└── README.md

```

---

## 8. Implementation Roadmap & Milestones

* **Phase 1: Formalization, Dimension Optimization & Core Symbolic Engine**
  * Implement $\Sigma = \{0, 1, 2, 3\}^{256}$ tensor layouts in PyTorch.
  * Execute mRMR dimension selection across candidate pool ($K = 512\text{--}1024$) from NSM, WordNet, and FrameNet.
  * Run Information Profiler on 5k validation corpus to eliminate dead slots ($H(D_i) < 0.1\text{ bits}$) and collapse redundant pairs ($I(D_i; D_j) > 0.5\text{ bits}$).
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


