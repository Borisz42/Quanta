% QUANTA Neuro-Symbolic Invariants & Integrity Constraints
% s(CASP) / Prolog encoding for semantic gate validation and MUC extraction

% ----------------------------------------------------------------------
% 1. Ontological Domain / Range Constraints
% ----------------------------------------------------------------------

% Rule 1: Abstract concepts cannot possess physical agent capability unless figurative
false :-
    slot(N, 'TYPE_ABSTRACT_CONCEPT', 1),
    slot(N, 'ROLE_AGENT_CAPABLE', 1),
    not slot(N, 'MODALITY_FIGURATIVE', 1).

% Rule 2: Inanimate physical entities cannot be sentient unless figurative
false :-
    slot(N, 'TYPE_INANIMATE_PHYSICAL', 1),
    slot(N, 'ROLE_SENTIENT', 1),
    not slot(N, 'MODALITY_FIGURATIVE', 1).

% Rule 3: Mutual exclusivity of fundamental entity types
false :-
    slot(N, 'TYPE_ANIMATE', 1),
    slot(N, 'TYPE_INANIMATE_PHYSICAL', 1).

false :-
    slot(N, 'TYPE_ABSTRACT_CONCEPT', 1),
    slot(N, 'TYPE_INANIMATE_PHYSICAL', 1).

false :-
    slot(N, 'TYPE_TEMPORAL_INTERVAL', 1),
    slot(N, 'TYPE_ANIMATE', 1).

false :-
    slot(N, 'TYPE_SPATIAL_REGION', 1),
    slot(N, 'TYPE_ANIMATE', 1).

false :-
    slot(N, 'TYPE_NUMERIC_VALUE', 1),
    slot(N, 'TYPE_ANIMATE', 1).

% Rule 4: Consumable entities must be physical substances or objects
false :-
    slot(N, 'ROLE_CONSUMABLE', 1),
    slot(N, 'TYPE_ABSTRACT_CONCEPT', 1),
    not slot(N, 'MODALITY_FIGURATIVE', 1).

% ----------------------------------------------------------------------
% 2. Graph Valency & Thematic Role Constraints
% ----------------------------------------------------------------------

% Rule 5: An entity acting as an Agent (VAL_X1_AGENT) in a literal event
% must be agent-capable (animate, human, organization, or explicitly agent-capable)
false :-
    edge(EventNode, 'VAL_X1_AGENT', AgentNode),
    slot(EventNode, 'MODALITY_LITERAL', 1),
    not slot(AgentNode, 'ROLE_AGENT_CAPABLE', 1),
    not slot(AgentNode, 'TYPE_HUMAN', 1),
    not slot(AgentNode, 'TYPE_ANIMATE', 1),
    not slot(AgentNode, 'TYPE_ORGANIZATION', 1).

% Rule 6: Experiencer role requires sentient entity
false :-
    edge(EventNode, 'VAL_EXPERIENCER', ExpNode),
    slot(EventNode, 'MODALITY_LITERAL', 1),
    not slot(ExpNode, 'ROLE_SENTIENT', 1),
    not slot(ExpNode, 'TYPE_HUMAN', 1),
    not slot(ExpNode, 'TYPE_ANIMATE', 1).

% Rule 7: Instrument role requires usable physical artifact
false :-
    edge(EventNode, 'VAL_X5_INSTRUMENT', InstNode),
    slot(EventNode, 'MODALITY_LITERAL', 1),
    slot(InstNode, 'TYPE_ABSTRACT_CONCEPT', 1),
    not slot(InstNode, 'MODALITY_FIGURATIVE', 1).

% ----------------------------------------------------------------------
% 3. Allen's Interval Temporal Calculus Consistency
% ----------------------------------------------------------------------

% Rule 8: Mutual exclusivity among incompatible Allen temporal relations
false :-
    slot(N, 'TEMP_ALLEN_BEFORE', 1),
    slot(N, 'TEMP_ALLEN_DURING', 1).

false :-
    slot(N, 'TEMP_ALLEN_BEFORE', 1),
    slot(N, 'TEMP_ALLEN_EQUALS', 1).

false :-
    slot(N, 'TEMP_ALLEN_MEETS', 1),
    slot(N, 'TEMP_ALLEN_DURING', 1).

% ----------------------------------------------------------------------
% 4. Spatial Mereotopology (RCC-8) Consistency
% ----------------------------------------------------------------------

% Rule 9: Disconnected regions cannot simultaneously overlap or be congruent
false :-
    slot(N, 'SPATIAL_RCC_DISCONNECTED', 1),
    slot(N, 'SPATIAL_RCC_CONGRUENT_EQ', 1).

false :-
    slot(N, 'SPATIAL_RCC_DISCONNECTED', 1),
    slot(N, 'SPATIAL_RCC_PARTIAL_OVERLAP', 1).

false :-
    slot(N, 'SPATIAL_RCC_DISCONNECTED', 1),
    slot(N, 'SPATIAL_RCC_NON_TANG_PART', 1).

% ----------------------------------------------------------------------
% 5. Epistemic & Deontic Modal Consistency
% ----------------------------------------------------------------------

% Rule 10: Deontic contradiction: Simultaneous Obligation and Prohibition
false :-
    slot(N, 'EPIST_DEONTIC_OBLIGATION', 1),
    slot(N, 'EPIST_DEONTIC_PROHIBITION', 1).

% Rule 11: Direct observation cannot be marked as hearsay
false :-
    slot(N, 'EPIST_DIRECT_OBSERVATION', 1),
    slot(N, 'EPIST_HEARSAY_TESTIMONY', 1).

% Rule 12: Unhedged direct affirmation cannot be simultaneously negated
false :-
    slot(N, 'NSM_TRUE', 1),
    slot(N, 'LJB_NA_NEGATION', 1),
    not slot(N, 'NSM_MAYBE', 1),
    not slot(N, 'MODALITY_FIGURATIVE', 1).
