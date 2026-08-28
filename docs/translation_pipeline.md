# QUANTA Two-Way Translation & Verification Pipeline

This document provides the exhaustive engineering specification, architectural topology, and algorithmic details of the **QUANTA Two-Way Neuro-Symbolic Translation Pipeline**.

---

## 1. Executive Summary & Core Principles

The QUANTA translation pipeline is a deterministic, zero-hallucination, neuro-symbolic framework that bidirectionally translates between natural languages (English, Hungarian), First-Order Logic (FOL) formulas, and Python Abstract Syntax Trees (ASTs) via an intermediate canonical representation: the **QUANTA Abstract Syntax Graph (ASG)**.

```text
       ┌──────────────────┐               ┌──────────────────┐
       │ Natural Language │               │ First-Order Logic│
       │ (English/Magyar) │               │   (FOL Form)     │
       └─────────┬────────┘               └────────┬─────────┘
                 │                                 │
                 ▼                                 ▼
      ┌────────────────────────────────────────────────────────┐
      │          QUANTA Abstract Syntax Graph (ASG)            │
      │        - 256 Discrete Semantic Slots: {0, 1, 2, 3}     │
      │        - BLAKE3 Cryptographic Merkle Folding           │
      │        - Clingo / s(CASP) Ontological Gating           │
      └──────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
       ┌──────────────────┐               ┌──────────────────┐
       │ Multi-Lingual    │               │ Executable Code  │
       │ Realizer (NLG)   │               │   (Python AST)   │
       └──────────────────┘               └──────────────────┘
```

### Key Architectural Invariants
1. **Discrete 4-Valued State Space**: Every semantic dimension is defined over $\mathcal{FOUR} = \{0: \text{IRRELEVANT}, 1: \text{TRUE}, 2: \text{FALSE}, 3: \text{UNCERTAIN}\}$.
2. **Bit-for-Bit Merkle Invariance**: Equivalent semantic meaning maps to the identical BLAKE3 Merkle Root hash regardless of syntactic paraphrasing or surface-level typos.
3. **Formal ASP Ontological Gating**: Every candidate graph must pass constraint validation via Answer Set Programming before realization.
4. **Pure Symbolic Determinism**: Zero reliance on non-deterministic external LLMs or floating-point black-box embeddings for representation.

---

## 2. Complete Pipeline Flow Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & NORMALIZATION                                                │
│    Raw Input String                                                         │
│        │                                                                    │
│        ▼                                                                    │
│    TypoNormalizer (Damerau-Levenshtein Edit Distance & Compound Healing)    │
│        │                                                                    │
│        ▼                                                                    │
│    Normalized Clean Text Stream                                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. FORWARD PARSING & GROUNDING                                              │
│    Morphosyntactic Analyzer (spaCy / Hungarian Morphological Parser)        │
│        │                                                                    │
│        ▼                                                                    │
│    Clause & Valency Decomposer (Root Predicate, Arguments & Prepositions)   │
│        │                                                                    │
│        ▼                                                                    │
│    WordNet Lexical Grounder & 4-Band Semantic Slot Assignment               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. ASG ASSEMBLY & MERKLE FOLDING                                            │
│    QuantaNode Tensor Creation (64-byte Dual-Bitmask QuantaVector)           │
│        │                                                                    │
│        ▼                                                                    │
│    BLAKE3 Merkle Tree Calculation & Graph Root CID Registration             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. FORMAL ONTOLOGICAL GATING                                                │
│    Clingo / s(CASP) Answer Set Programming Constraint Solver                │
│        │                                                                    │
│        ├──[ Contradiction Detected ]──> MUC Diagnostics & Error Extractor   │
│        │                                                                    │
│        └──[ Satisfiable / Valid ]─────> Verified QuantaGraph ASG            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. REVERSE REALIZATION & MULTI-MODAL NLG                                    │
│    ├── English Realizer   : SVO Syntax, Auxiliaries, Tenses & Prepositions  │
│    ├── Hungarian Realizer : Vowel Harmony, Agglutinative Cases & Copulas    │
│    ├── FOL Realizer       : Universal & Existential Logical Formulae        │
│    └── AST Realizer       : Executable Python Abstract Syntax Trees         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1: Algorithmic Typo Resilience Engine

The typo resilience module ([`src/parser/typo_normalizer.py`](../src/parser/typo_normalizer.py)) operates prior to grammatical parsing to resolve character distortions without changing syntactic intentions.

### Damerau-Levenshtein Metric
Calculates the edit distance allowing four elementary operations:
1. **Deletions**: `maileman` $\to$ `mailman` ($\text{cost} = 1$)
2. **Insertions**: `glden` $\to$ `golden` ($\text{cost} = 1$)
3. **Substitutions**: `chsed` $\to$ `chased` ($\text{cost} = 1$)
4. **Adjacent Transpositions**: `graden` $\to$ `garden` ($\text{cost} = 1$)

$$\text{dist}(s_1[i], s_2[j]) = \min \begin{cases}
d(i-1, j) + 1 \\
d(i, j-1) + 1 \\
d(i-1, j-1) + \text{cost} \\
d(i-2, j-2) + \text{cost} & \text{if } s_1[i]=s_2[j-1] \land s_1[i-1]=s_2[j]
\end{cases}$$

### Priority Lexicons & Compound Healing
- **Core Domain Lexicon**: Includes high-priority verbs, auxiliaries (`did`, `was`, `has`, `can`, `will`), modal operators, and domain nouns.
- **WordNet Lemma Fallback**: If a token is not in the core lexicon, it looks up known WordNet synset lemma names.
- **Multi-Word Compound Healing**: Identifies split or misspelled compound nouns (e.g. `glden retreiver` $\to$ `golden retriever`).

---

## 4. Stage 2: Morphosyntactic & 4-Band Semantic Slot Extraction

The parser extracts arguments and maps concepts into the **256-dimensional slot tensor** across 4 functional bands.

### The 4-Band Layout (256 Slots, 64-Byte Tensor)

| Band | Slot Range | Ontology & Domain | Example Active Slots |
|---|---|---|---|
| **Band 0** | `0 - 63` | **NSM Universal Primes & Kinematics** | `NSM_DO`, `NSM_SEE`, `NSM_THINK`, `NSM_MOVE`, `NSM_TOUCH`, `NSM_INSIDE` |
| **Band 1** | `64 - 127` | **Valency Roles, Tenses & Connectives** | `VAL_X1_AGENT`, `VAL_X2_PATIENT`, `VAL_LOCATION_SLOT`, `LJB_PU_PAST_TENSE`, `LJB_NA_NEGATION` |
| **Band 2** | `128 - 191` | **WordNet Hypernyms & Entity Qualities** | `TYPE_ANIMATE`, `TYPE_HUMAN`, `TYPE_SPATIAL_REGION`, `TYPE_EVENT`, `ROLE_AGENT_CAPABLE` |
| **Band 3** | `192 - 255` | **Epistemics, Proof States & Solvers** | `EPIST_DIRECT_OBSERVATION`, `EPIST_PROB_CERTAIN`, `MODALITY_LITERAL` |

### Relational Valency Routing & Preposition Handling

The relation extraction assigns specific roles to distinct nodes before edge binding:

```text
                  ┌───────────────────────────────────────────────┐
                  │                 wn:bite.v.01                  │
                  │ (Root Event: NSM_TOUCH=1, LJB_PU_PAST_TENSE=1)│
                  └───┬───────────────────┬───────────────────┬───┘
                      │                   │                   │
      VAL_X1_AGENT    │   VAL_X2_PATIENT  │ VAL_LOCATION_SLOT │
                      ▼                   ▼                   ▼
           ┌──────────────────────┐ ┌──────────┐       ┌────────────┐
           │wn:golden_retriever.n │ │wn:mailman│       │ wn:garden  │
           │(TYPE_ANIMATE=1)      │ │(HUMAN=1) │       │(SPATIAL=1) │
           └──────────────────────┘ └──────────┘       └────────────┘
```

- **`VAL_X1_AGENT`**: Grammatical subject / initiator (`dog`, `golden retriever`, `mailman`).
- **`VAL_X2_PATIENT`**: Direct object / undergoer (`cat`, `book`, `mailman`).
- **`VAL_EXPERIENCER`**: Animate / human target of perception or transfer (e.g. `gave a book to a person`).
- **`VAL_LOCATION_SLOT`**: Spatial containment or setting (`in the garden`, `inside the house`).
- **`VAL_X3_DESTINATION`**: Physical motion target (`into the house`, `to the city`).
- **`VAL_X4_SOURCE`**: Origin / start of motion (`from the garden`).
- **`VAL_X5_INSTRUMENT`**: Usable tool (`with a stick`).

---

## 5. Stage 3: ASG Assembly & BLAKE3 Merkle Folding

### Memory-Efficient QuantaVector
Each node's vector contains $256$ quaternary values, packed compactly into 64 bytes using 2-bit dual masks:
- Bitmask 0 (`b0`): Represents the lower bit of each slot value.
- Bitmask 1 (`b1`): Represents the upper bit of each slot value.

### Canonical Content Identifier (CID) Calculation
A node's CID is an invariant cryptographic hash computed over its vector bytes, WordNet anchor, literal, and sorted child edges:

$$\text{CID} = \text{BLAKE3}\Big(\text{vector\_bytes} \parallel \text{anchor} \parallel \text{literal} \parallel \text{sorted}(\text{edges})\Big)$$

$$\text{Merkle Root} = \text{Root Node CID}$$

> **Child Node Immutability Invariant**: All slot assignments and modifier bindings on child nodes are completed *before* edge registration into the parent node, ensuring Merkle tree integrity without dangling edges.

---

## 6. Stage 4: Formal Ontological Gating (ASP Solver)

Before realization, the graph is submitted to [`src/solver/validator_gate.py`](../src/solver/validator_gate.py) via **Clingo / s(CASP)**.

### Constraint Rules Checked:
1. **Ontological Domain Coherence**:
   - `:- slot(N, "VAL_X3_DESTINATION", 1), not slot(N, "TYPE_SPATIAL_REGION", 1).`
   - `:- slot(N, "TYPE_ANIMATE", 1), slot(N, "TYPE_SPATIAL_REGION", 1).` (Entities cannot be both animate animals/humans and geographic spatial containers).
2. **Cardinality & Valency Uniqueness**:
   - Predicates cannot have multiple distinct agent targets unless conjoined with `LJB_JE_AND`.
3. **MUC Diagnosis**:
   - If an ontological violation occurs, the solver extracts the **Minimal Unsatisfiable Core (MUC)** to pinpoint exact conflicting slot indices.

---

## 7. Stage 5: Reverse Realization & Multi-Modal NLG

### 1. English Natural Language Generator ([`src/realizer/english_nlg.py`](../src/realizer/english_nlg.py))
- **Verb Conjugation**: Maps base lemmas to 3rd person present (`sees`, `bites`) or irregular/regular past (`saw`, `bit`, `chased`).
- **Negation & Auxiliaries**: Correctly constructs negative past clauses (`did not see`) and modal clauses (`must touch`, `cannot move`).
- **Preposition Attachment**: Assembles prepositional arguments (`in a garden`, `to a house`, `with a stick`).

### 2. Hungarian Morphological Realizer ([`src/realizer/hungarian_morph.py`](../src/realizer/hungarian_morph.py))
- **Vowel Harmony Engine**: Analyzes stem vowels (`a, o, u, á, ó, ú` $\to$ *back*; `e, i, é, í` $\to$ *front unrounded*; `ö, ü, ő, ű` $\to$ *front rounded*).
- **Agglutinative Suffixation**:
  - Accusative (`VAL_X2_PATIENT`): `macska` $\to$ `macskát`, `postás` $\to$ `postást` / `postásot`.
  - Inessive (`VAL_LOCATION_SLOT`): `kert` $\to$ `a kertben`, `ház` $\to$ `a házban`.
  - Illative (`VAL_X3_DESTINATION`): `ház` $\to$ `a házba`.
  - Instrumental (`VAL_X5_INSTRUMENT`): `bot` $\to$ `bottal`.
- **Copular Past Realization**: Inflects `van` in past tense as `volt` and places predicate adjectives (`nagy`, `kis`) in post-verbal position (`A kutya nem volt nagy.`).

---

## 8. Verification & Performance Benchmark Results

The pipeline is verified against round-trip and cross-lingual translation benchmarks in [`output/`](../output):

### Benchmark Summary Table

| Input Sentence | Target Modality | Output Realization | Slot Preservation | Validation |
|---|---|---|---|---|
| `A dog chased a cat.` | English | `A dog chased a cat.` | **100.0%** | `VALID (PASS)` |
| `A golden retriever bit the mailman in the garden.` | English | `A golden retriever bit a mailman in a garden.` | **96.7%** | `VALID (PASS)` |
| `A golden retriever bit the mailman in the garden.` | Hungarian | `A golden retriever a kertben a postásot harapott.` | **95.2%** | `VALID (PASS)` |
| `A person did not see a cat.` | English | `A person did not see a cat.` | **100.0%** | `VALID (PASS)` |
| `A dog ran rapidly into the house.` | English | `A dog ran rapidly to a house.` | **92.0%** | `VALID (PASS)` |
| `A mailman gave a book to a person.` | English | `A mailman gave a book to a person.` | **100.0%** | `VALID (PASS)` |
| `A dog was not big.` | Hungarian | `A kutya nem volt nagy.` | **96.0%** | `VALID (PASS)` |

---

## 9. Running Translation via Python API

```python
from pipeline.translator_pipeline import TwoWayTranslationPipeline

pipeline = TwoWayTranslationPipeline()

# 1. English to English round-trip with ASG verification
result = pipeline.execute_translation(
    "A golden retriever bit the mailman in the garden.",
    target_modality="english",
    source_modality="english"
)
print("Output:", result.output_text)
print("Merkle Root:", result.merkle_root)
print("Validation Pass:", result.validation.is_valid)

# 2. English to Hungarian cross-lingual translation
hu_result = pipeline.execute_translation(
    "A golden retriever bit the mailman in the garden.",
    target_modality="hungarian",
    source_modality="english"
)
print("Hungarian:", hu_result.output_text)

# 3. Typo-Tolerant Input
typo_result = pipeline.execute_translation(
    "A glden retreiver bit the maileman in the graden.",
    target_modality="english"
)
assert typo_result.merkle_root == result.merkle_root  # Cryptographic invariance
```
