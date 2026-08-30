# Generalized Multilingual Realizer Architecture

## 1. Architectural Motivation & Theoretical Foundation

In existing continuous token-based LLMs, multilingual reasoning suffers from the **cross-lingual representational drift problem**: prompt semantics shift when translating queries across languages due to tokenization imbalances (fertility disparities), syntactic alignment divergence, and continuous embedding drift.

QUANTA resolves this by treating the discrete quaternary vector space ($\Sigma^{1024} = \{0, 1, 2, 3\}^{1024}$) and Content-Addressed Abstract Syntax Graphs (ASGs) as the universal, invariant semantic pivot (*Mentalese*):

$$\text{Source Surface Language } \mathcal{L}_1 \xrightarrow{\text{Forward Parser}} \mathcal{G}_{\text{ASG}} \in \Sigma^{1024} \xrightarrow{\text{Reverse Realizer}} \text{Target Surface Language } \mathcal{L}_2$$

Because $\mathcal{G}_{\text{ASG}}$ is grounded in Natural Semantic Metalanguage (NSM) primes (Band 0), structural valency routing (Band 1), register scoping (Band 2), and ConceptNet 5.7.0 256-D data-driven ontological/affordance dimensions (Bands 3 & 4) governed by Polymorphic 2-Bit Typing per Band, reverse realizers unroll pure abstract meaning into arbitrary human natural languages without relying on intermediate pairwise translation models.

---

## 2. Typological Classification & Morphosyntactic Pipeline

Natural languages vary fundamentally in how grammatical relationships (tense, aspect, modality, case, agreement) are encoded. The QUANTA Multilingual Realizer organizes natural language generation into three primary typological families:

```
                          ┌────────────────────────────┐
                          │   Mentalese ASG (Σ^1024)   │
                          └─────────────┬──────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
      ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
      │     ISOLATING      │ │   AGGLUTINATIVE    │ │      FUSIONAL      │
      │ (Analytic Grammar) │ │ (Morpheme Chaining)│ │(Portmanteau Fusing)│
      └──────────┬─────────┘ └──────────┬─────────┘ └──────────┬─────────┘
                 │                      │                      │
                 ▼                      ▼                      ▼
          Free Particles &      Prefix/Suffix Chains     Fused Case/Gender/
         Strict Word Order        & Case Suffixes         Tense Inflection
```

### 2.1 Isolating / Analytic Family
- **Characteristics:** Minimal to zero bound inflectional morphemes; grammatical relationships and aspect/tense are expressed through free lexical particles and strict constituent word order (typically SVO).
- **Archetypes:** Mandarin Chinese, Vietnamese, Classical Chinese, analytic English expressions.
- **Mapping:**
  - `NSM_ONE=1` $\to$ singular numerical particle / classifier.
  - `LJB_PU_PAST_TENSE=1` $\to$ aspect particle (`did`, `le 了`, `da 已`).
  - `LJB_NA_NEGATION=2` $\to$ pre-verbal negative particle (`not`, `bu 不`, `mei 没`).
  - `GRAPH_QUERY_TARGET=3` $\to$ modal interrogative particle (`maybe`, `ma 吗`).

### 2.2 Agglutinative Family
- **Characteristics:** Monomorphemic, highly regular affix chains concatenated onto invariable roots; explicit case suffixes for semantic roles; flexible constituent ordering (often SOV / Free Word Order).
- **Archetypes:** Turkish, Hungarian, Finnish, Swahili, Japanese, Korean, Basque.
- **Mapping:**
  - Agent (`VAL_X1_AGENT`) $\to$ Nominative ($\emptyset$).
  - Patient (`VAL_X2_PATIENT`) $\to$ Accusative (`-t`, `-i`, `-ni`, `-yı`).
  - Location (`VAL_LOCATION_SLOT`) $\to$ Inessive/Locative (`-ban/-ben`, `-da/-de`, `-de`).
  - Destination (`VAL_X3_DESTINATION`) $\to$ Allative/Illative (`-ba/-be`, `-a/-e`, `-ye`).
  - Instrument (`VAL_X5_INSTRUMENT`) $\to$ Instrumental (`-val/-vel`, `-le/-la`).
  - Tense & Polarity $\to$ Verb stem suffix concatenation: $\text{Root} + \text{TenseSuffix} + \text{PersonSuffix}$.

### 2.3 Fusional / Inflectional Family
- **Characteristics:** Portmanteau inflections combining case, number, gender, and definiteness into single indivisible morphemes; stem alternations (ablaut, umlaut); complex agreement systems.
- **Archetypes:** German, Latin, Spanish, Russian, Sanskrit, Polish, Greek.
- **Mapping:**
  - Case agreement computed across Determiner + Adjective + Noun heads.
  - Fused 3rd-person singular past/present verbal suffixes (e.g. German `-te`, Latin `-avit`).

---

## 3. Polymorphic Adapter Interface

Every language adapter implements the abstract `LanguageAdapter` contract:

```python
class LanguageAdapter(abc.ABC):
    @abc.abstractmethod
    def realize_noun_phrase(self, node: QuantaNode, graph: QuantaGraph, case: Optional[str] = None) -> str:
        ...

    @abc.abstractmethod
    def inflect_verb(self, verb_node: QuantaNode, graph: QuantaGraph, is_negated: bool = False, is_uncertain: bool = False) -> str:
        ...

    @abc.abstractmethod
    def format_clause(self, subject: Optional[str], verb: str, patient: Optional[str], modifiers: List[str]) -> str:
        ...

    def realize_graph(self, graph: QuantaGraph) -> str:
        ...
```

### 3.1 Dynamic Language Registration & Dispatch

The `MultilingualRealizerRegistry` provides centralized discovery and runtime polymorphic dispatch:

```python
from realizer.multilingual import MultilingualRealizerRegistry

# Register custom language adapters
MultilingualRealizerRegistry.register_adapter("tr", TurkishLanguageAdapter(...))

# Realize ASG into any registered language
output_text = MultilingualRealizerRegistry.realize_graph(asg_graph, lang_code="agglutinative_ref")
```

---

## 4. Zero Semantic Drift Verification

Under the Mentalese pivot, a proposition $\mathcal{P}$ round-tripped across languages maintains:

$$d_H\big(\mathbf{v}(\mathcal{P}_{\mathcal{L}_1}), \mathbf{v}(\mathcal{P}_{\mathcal{L}_2})\big) = 0$$

for all canonical semantic slots in Bands 0 through 7.
