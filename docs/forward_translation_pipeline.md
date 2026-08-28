# QUANTA Forward Translation & Real Dataset Ingestion Pipeline

This document defines the formal engineering specification for translating natural language sentences, first-order logic formulas, and code syntax into **QUANTA Abstract Syntax Graphs (ASGs)** over the discrete quaternary state space $\Sigma^{256} = \{0, 1, 2, 3\}^{256}$.

---

## 1. Architectural Overview

```
                      [ Input: Natural Language / Logic / AST ]
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  Stage 1: Syntactic & Dependency Pass │
                      │   (spaCy: POS, Lemmas, Dep Arcs)      │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  Stage 2: Clause & Argument Partition │
                      │  (Root Predicate, Noun Chunks, Mods)  │
                      └───────────────────┬───────────────────┘
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         ▼                                ▼                                ▼
┌──────────────────┐            ┌──────────────────┐             ┌──────────────────┐
│  Stage 3A: NSM   │            │ Stage 3B: Valency│             │ Stage 3C: WordNet│
│   Prime Mapping  │            │  & Graph Routing │             │ Hypernym Grounder│
│     (Band 0)     │            │     (Band 1)     │             │     (Band 2)     │
└────────┬─────────┘            └────────┬─────────┘             └────────┬─────────┘
         │                               │                                │
         └────────────────────────────────┼────────────────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  Stage 4: Epistemic & Modal Calibration│
                      │ (Tense, Quantifiers, Negation, Band 3)│
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  Stage 5: ASG Assembly & ASP Gating   │
                      │ (BLAKE3 Merkle Tree + Clingo Validate)│
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                            [ Verified QuantaGraph ASG ]
```

---

## 2. The 5 Translation Stages

### Stage 1: Syntactic & Dependency Extraction (`spaCy`)
- **POS & Morphological Analysis**: Identifies root predicates (`VERB`, `AUX`), noun heads (`NOUN`, `PROPN`), modifiers (`ADJ`, `ADV`), and determiners (`DET`).
- **Dependency Arcs**:
  - Subject relations: `nsubj`, `nsubjpass`, `csubj`
  - Object relations: `dobj`, `pobj`, `dative`, `attr`
  - Prepositional phrases: `prep` + `pobj`
  - Modifiers: `amod`, `advmod`, `neg`
  - Clause linkage: `advcl`, `relcl`, `cc`, `conj`, `mark`

### Stage 2: Clause & Argument Decomposition
A natural language sentence is decomposed into atomic `QuantaNode` units:
1. **Root Event / Predicate Node**: The central action or copular state.
2. **Entity Nodes**: Noun phrase heads grounded to WordNet anchors.
3. **Modifier / Sub-Expression Nodes**: Nested clauses, prepositional frames, or temporal locations.

### Stage 3: The 4-Band Slot Mapping Matrix

#### Band 0: Universal NSM Primes & Kinematics (0–63)
- **Verbs of Motion** (*run, walk, move, fly, enter*) $\implies$ `NSM_MOVE=1`, `TYPE_EVENT=1`
- **Verbs of Speech** (*say, tell, declare, claim*) $\implies$ `NSM_SAY=1`, `NSM_WORDS=1`, `TYPE_COMMUNICATION_MSG=1`
- **Verbs of Mental Cognition** (*think, believe, know*) $\implies$ `NSM_THINK=1` / `NSM_KNOW=1`
- **Verbs of Volition / Emotion** (*want, desire, fear*) $\implies$ `NSM_WANT=1` / `NSM_FEEL=1`
- **Physical Contact / Impact** (*touch, hit, bite, grab*) $\implies$ `NSM_DO=1`, `NSM_TOUCH=1`
- **Spatial Relations**:
  - *inside, in, within* $\implies$ `NSM_INSIDE=1`
  - *above, over, on* $\implies$ `NSM_ABOVE=1`
  - *below, under* $\implies$ `NSM_BELOW=1`
  - *near, by, close to* $\implies$ `NSM_NEAR=1`
  - *far, away* $\implies$ `NSM_FAR=1`

#### Band 1: Valencies, Connectives & Graph Topology (64–127)
- **Valency Edges**:
  - `nsubj` $\implies$ `VAL_X1_AGENT`
  - `dobj` / `attr` $\implies$ `VAL_X2_PATIENT`
  - `prep_to`, `prep_into` $\implies$ `VAL_X3_DESTINATION`
  - `prep_from` $\implies$ `VAL_X4_SOURCE`
  - `prep_with`, `prep_by` $\implies$ `VAL_X5_INSTRUMENT`
  - Sensation target $\implies$ `VAL_EXPERIENCER`
- **Tense Markers**:
  - Past tense (`VBD`, `VBN`) $\implies$ `LJB_PU_PAST_TENSE=1`
  - Present tense (`VBP`, `VBZ`) $\implies$ `LJB_CA_PRESENT_TENSE=1`
  - Future tense (`will`, `shall`) $\implies$ `LJB_BA_FUTURE_TENSE=1`
- **Logical Connectives**:
  - *if...then* $\implies$ `LJB_GANAI_IF_THEN=1`
  - *and* $\implies$ `LJB_JE_AND=1`
  - *or* $\implies$ `LJB_JA_OR=1`
  - *not, never, no* $\implies$ `LJB_NA_NEGATION=2`

#### Band 2: Lexical Anchoring & Ontological Hierarchy (128–191)
For entity nodes, `WordNetLexicalGrounder` resolves the noun lemma into a WordNet synset and traverses hypernym chains:
- Ancestor `person.n.01` $\implies$ `TYPE_HUMAN=1`, `TYPE_ANIMATE=1`, `ROLE_AGENT_CAPABLE=1`, `ROLE_SENTIENT=1`, `ROLE_COMMUNICATOR=1`, `WN_PERSON_HUMAN=1`
- Ancestor `animal.n.01` $\implies$ `TYPE_ANIMATE=1`, `ROLE_AGENT_CAPABLE=1`, `ROLE_SENTIENT=1`, `WN_ANIMAL_FAUNA=1`
- Ancestor `artifact.n.01` $\implies$ `TYPE_ARTIFACT=1`, `TYPE_INANIMATE_PHYSICAL=1`, `ROLE_INSTRUMENT_USABLE=1`
- Ancestor `abstraction.n.06` $\implies$ `TYPE_ABSTRACT_CONCEPT=1`
- Ancestor `location.n.01` $\implies$ `TYPE_SPATIAL_REGION=1`, `WN_LOCATION_PLACE=1`

#### Band 3: Epistemic Context, Probability & Metarules (192–255)
- **Quantifiers**:
  - *all, every, each* $\implies$ `NSM_ALL=1`, `LJB_RO_ALL_QUANT=1`
  - *some, exists, a* $\implies$ `NSM_SOME=1`, `LJB_SUO_AT_LEAST_ONE=1`
  - *no, none* $\implies$ `LJB_NO_NONE_QUANT=1`, `LJB_NA_NEGATION=2`
- **Modal Auxiliaries**:
  - *must, obliged* $\implies$ `EPIST_DEONTIC_OBLIGATION=1`
  - *may, permitted* $\implies$ `EPIST_DEONTIC_PERMISSION=1`
  - *might, maybe* $\implies$ `NSM_MAYBE=1`, `EPIST_PROB_MARGINAL=1`
- **Dataset Provenance & Solver Invariants**:
  - Deductive FOLIO logic $\implies$ `EPIST_DEDUCTIVE_INFERENCE=1`, `SOLVER_PROOF_VALIDATED=1`
  - ProofWriter theories $\implies$ `EPIST_AXIOMATIC_PREMISE=1`, `SOLVER_CWA_CLOSED_WORLD=1`
  - bAbI state changes $\implies$ `EPIST_DIRECT_OBSERVATION=1`

---

## 3. Real Benchmark Ingestion Specification

The real data ingestion pipeline streams authentic samples across 5 core benchmarks:

1. **FOLIO (First-Order Logic & Natural Syllogisms)**:
   - Direct parsing of both Natural Language premises and formal FOL strings ($\forall x, \exists x, \rightarrow, \land, \neg$).
2. **ProofWriter (Multi-step Rule Logic)**:
   - Ingestion of premise theories, facts, and Horn rules (`if A then B`).
3. **bAbI (Spatial & World State Tracking)**:
   - Ingestion of 20 physical reasoning tasks (movements, object transfers, path finding).
4. **CLUTRR (Inductive Kinship Graphs)**:
   - Stories of multi-hop family relationships mapping to transitivity and reciprocal roles.
5. **Code ASTs (Python Standard Library)**:
   - Direct recursive mapping from Python `ast.AST` nodes into `GRAPH_*` AST primitives and Merkle-folded execution branches.
