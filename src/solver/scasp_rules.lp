% QUANTA Neuro-Symbolic Invariants & Integrity Constraints
% Clingo / ASP encoding for semantic gate validation and MUC extraction

% ----------------------------------------------------------------------
% 1. Ontological Domain / Range Constraints
% ----------------------------------------------------------------------

% Rule 1: Abstract concepts cannot possess physical agent capability unless figurative
:- slot(N, "TYPE_ABSTRACT_CONCEPT", 1),
   slot(N, "ROLE_AGENT_CAPABLE", 1),
   not slot(N, "MODALITY_FIGURATIVE", 1).

% Rule 2: Inanimate physical entities cannot be sentient unless figurative
:- slot(N, "TYPE_INANIMATE_PHYSICAL", 1),
   slot(N, "ROLE_SENTIENT", 1),
   not slot(N, "MODALITY_FIGURATIVE", 1).

% Rule 3: Natural objects cannot be manufactured artifacts unless figurative
:- slot(N, "TYPE_NATURAL_OBJECT", 1),
   slot(N, "TYPE_ARTIFACT", 1),
   not slot(N, "MODALITY_FIGURATIVE", 1).

% Rule 4: Mutual exclusivity of fundamental entity types
:- slot(N, "TYPE_ANIMATE", 1),
   slot(N, "TYPE_INANIMATE_PHYSICAL", 1).

:- slot(N, "TYPE_ABSTRACT_CONCEPT", 1),
   slot(N, "TYPE_INANIMATE_PHYSICAL", 1).

:- slot(N, "TYPE_TEMPORAL_INTERVAL", 1),
   slot(N, "TYPE_ANIMATE", 1).

:- slot(N, "TYPE_SPATIAL_REGION", 1),
   slot(N, "TYPE_ANIMATE", 1).

:- slot(N, "TYPE_NUMERIC_VALUE", 1),
   slot(N, "TYPE_ANIMATE", 1).

:- slot(N, "TYPE_TEMPORAL_INTERVAL", 1),
   slot(N, "TYPE_SPATIAL_REGION", 1).

:- slot(N, "TYPE_PROPOSITION", 1),
   slot(N, "TYPE_ANIMATE", 1).

:- slot(N, "TYPE_SUBSTANCE_MASS", 1),
   slot(N, "TYPE_NUMERIC_VALUE", 1).

% Rule 5: Consumable entities must be physical substances or objects
:- slot(N, "ROLE_CONSUMABLE", 1),
   slot(N, "TYPE_ABSTRACT_CONCEPT", 1),
   not slot(N, "MODALITY_FIGURATIVE", 1).

% ----------------------------------------------------------------------
% 2. Graph Valency & Thematic Role Constraints
% ----------------------------------------------------------------------

% Rule 6: An entity acting as an Agent (VAL_X1_AGENT) in a literal event
% must be agent-capable (animate, human, organization, variable binding, or explicitly agent-capable)
:- edge(EventNode, "VAL_X1_AGENT", AgentNode),
   slot(EventNode, "MODALITY_LITERAL", 1),
   not slot(AgentNode, "ROLE_AGENT_CAPABLE", 1),
   not slot(AgentNode, "TYPE_HUMAN", 1),
   not slot(AgentNode, "TYPE_ANIMATE", 1),
   not slot(AgentNode, "TYPE_ORGANIZATION", 1),
   not slot(AgentNode, "WN_PERSON_HUMAN", 1),
   not slot(AgentNode, "WN_ANIMAL_FAUNA", 1),
   not slot(AgentNode, "GRAPH_VARIABLE_BIND", 1).

% Rule 7: Experiencer role requires sentient entity
:- edge(EventNode, "VAL_EXPERIENCER", ExpNode),
   slot(EventNode, "MODALITY_LITERAL", 1),
   not slot(ExpNode, "ROLE_SENTIENT", 1),
   not slot(ExpNode, "TYPE_HUMAN", 1),
   not slot(ExpNode, "TYPE_ANIMATE", 1),
   not slot(ExpNode, "WN_PERSON_HUMAN", 1),
   not slot(ExpNode, "WN_ANIMAL_FAUNA", 1),
   not slot(ExpNode, "GRAPH_VARIABLE_BIND", 1).

% Rule 8: Instrument role requires usable physical artifact
:- edge(EventNode, "VAL_X5_INSTRUMENT", InstNode),
   slot(EventNode, "MODALITY_LITERAL", 1),
   slot(InstNode, "TYPE_ABSTRACT_CONCEPT", 1),
   not slot(InstNode, "MODALITY_FIGURATIVE", 1).

% ----------------------------------------------------------------------
% 3. Allen's Interval Temporal Calculus Consistency (7B.2)
% ----------------------------------------------------------------------

% Rule 9: Pairwise mutual exclusivity among incompatible Allen temporal relations
:- slot(N, "TEMP_ALLEN_BEFORE", 1),
   slot(N, "TEMP_ALLEN_DURING", 1).

:- slot(N, "TEMP_ALLEN_BEFORE", 1),
   slot(N, "TEMP_ALLEN_EQUALS", 1).

:- slot(N, "TEMP_ALLEN_BEFORE", 1),
   slot(N, "TEMP_ALLEN_OVERLAPS", 1).

:- slot(N, "TEMP_ALLEN_BEFORE", 1),
   slot(N, "TEMP_ALLEN_STARTS", 1).

:- slot(N, "TEMP_ALLEN_BEFORE", 1),
   slot(N, "TEMP_ALLEN_FINISHES", 1).

:- slot(N, "TEMP_ALLEN_MEETS", 1),
   slot(N, "TEMP_ALLEN_DURING", 1).

:- slot(N, "TEMP_ALLEN_MEETS", 1),
   slot(N, "TEMP_ALLEN_EQUALS", 1).

:- slot(N, "TEMP_ALLEN_MEETS", 1),
   slot(N, "TEMP_ALLEN_OVERLAPS", 1).

:- slot(N, "TEMP_ALLEN_DURING", 1),
   slot(N, "TEMP_ALLEN_EQUALS", 1).

:- slot(N, "TEMP_ALLEN_STARTS", 1),
   slot(N, "TEMP_ALLEN_FINISHES", 1).

% ----------------------------------------------------------------------
% 4. Spatial Mereotopology (RCC-8) Consistency (7B.1)
% ----------------------------------------------------------------------

% Rule 10: Pairwise mutual exclusivity among RCC-8 spatial relations
:- slot(N, "SPATIAL_RCC_DISCONNECTED", 1),
   slot(N, "SPATIAL_RCC_CONGRUENT_EQ", 1).

:- slot(N, "SPATIAL_RCC_DISCONNECTED", 1),
   slot(N, "SPATIAL_RCC_PARTIAL_OVERLAP", 1).

:- slot(N, "SPATIAL_RCC_DISCONNECTED", 1),
   slot(N, "SPATIAL_RCC_NON_TANG_PART", 1).

:- slot(N, "SPATIAL_RCC_DISCONNECTED", 1),
   slot(N, "SPATIAL_RCC_TANGENTIAL_PART", 1).

:- slot(N, "SPATIAL_RCC_DISCONNECTED", 1),
   slot(N, "SPATIAL_RCC_EXT_CONNECTED", 1).

:- slot(N, "SPATIAL_RCC_EXT_CONNECTED", 1),
   slot(N, "SPATIAL_RCC_CONGRUENT_EQ", 1).

:- slot(N, "SPATIAL_RCC_EXT_CONNECTED", 1),
   slot(N, "SPATIAL_RCC_NON_TANG_PART", 1).

:- slot(N, "SPATIAL_RCC_EXT_CONNECTED", 1),
   slot(N, "SPATIAL_RCC_PARTIAL_OVERLAP", 1).

:- slot(N, "SPATIAL_RCC_NON_TANG_PART", 1),
   slot(N, "SPATIAL_RCC_CONGRUENT_EQ", 1).

:- slot(N, "SPATIAL_RCC_TANGENTIAL_PART", 1),
   slot(N, "SPATIAL_RCC_CONGRUENT_EQ", 1).

:- slot(N, "SPATIAL_RCC_PARTIAL_OVERLAP", 1),
   slot(N, "SPATIAL_RCC_CONGRUENT_EQ", 1).

:- slot(N, "MEREOLOGY_HOLONYM_WHOLE", 1),
   slot(N, "MEREOLOGY_MERONYM_PART", 1).

% ----------------------------------------------------------------------
% 5. Pearl's Causal Hierarchy Exclusivity Rules (7B.3)
% ----------------------------------------------------------------------

% Rule 11: Direct mechanism vs Preventive blocker exclusivity
:- slot(N, "CAUSAL_DIRECT_MECHANISM", 1),
   slot(N, "CAUSAL_PREVENTIVE_BLOCK", 1).

% Rule 12: Enabling condition vs Preventive blocker exclusivity
:- slot(N, "CAUSAL_ENABLING_CONDITION", 1),
   slot(N, "CAUSAL_PREVENTIVE_BLOCK", 1).

% Rule 13: Common Confounder (fork) vs Collider Effect (sink) exclusivity
:- slot(N, "CAUSAL_COMMON_CONFOUNDER", 1),
   slot(N, "CAUSAL_COLLIDER_EFFECT", 1).

% Rule 14: Counterfactual necessity vs Preventive blocker exclusivity
:- slot(N, "CAUSAL_COUNTERFACTUAL_NEC", 1),
   slot(N, "CAUSAL_PREVENTIVE_BLOCK", 1).

% ----------------------------------------------------------------------
% 6. Band 1 Structural Topology Invariants (7B.4)
% ----------------------------------------------------------------------

% Rule 15: Root proposition and terminal leaf exclusivity
:- slot(N, "GRAPH_ROOT_NODE", 1),
   slot(N, "GRAPH_LEAF", 1).

% Rule 16: Branch Head cannot be simultaneously THEN and ELSE branch
:- slot(N, "GRAPH_BRANCH_THEN", 1),
   slot(N, "GRAPH_BRANCH_ELSE", 1).

% Rule 17: AST Function Definition cannot be a pure Return Value sink
:- slot(N, "GRAPH_FUNCTION_DEF", 1),
   slot(N, "GRAPH_RETURN_VALUE", 1).

% Rule 18: Inclusive AND vs Exclusive XOR connective exclusivity
:- slot(N, "LJB_JE_AND", 1),
   slot(N, "LJB_JON_XOR", 1).

% ----------------------------------------------------------------------
% 7. Epistemic & Deontic Modal Consistency
% ----------------------------------------------------------------------

% Rule 19: Deontic contradiction: Simultaneous Obligation and Prohibition
:- slot(N, "EPIST_DEONTIC_OBLIGATION", 1),
   slot(N, "EPIST_DEONTIC_PROHIBITION", 1).

% Rule 20: Direct observation cannot be marked as hearsay
:- slot(N, "EPIST_DIRECT_OBSERVATION", 1),
   slot(N, "EPIST_HEARSAY_TESTIMONY", 1).

% Rule 21: Unhedged direct affirmation cannot be simultaneously negated
:- slot(N, "NSM_TRUE", 1),
   slot(N, "LJB_NA_NEGATION", 1),
   not slot(N, "NSM_MAYBE", 1),
   not slot(N, "MODALITY_FIGURATIVE", 1).

% ----------------------------------------------------------------------
% 8. Profiler-Derived Dimension Refactoring & Deduction Rules (Phase 11B.4)
% ----------------------------------------------------------------------

% Deduction 1: Animate entities automatically possess sentience and motility
slot(N, "ROLE_SENTIENT", 1) :- slot(N, "TYPE_ANIMATE", 1).
slot(N, "ROLE_MOVEABLE", 1) :- slot(N, "TYPE_ANIMATE", 1).

% Deduction 2: Ontological Attribute Property equivalence
slot(N, "WN_ATTRIBUTE_PROP", 1) :- slot(N, "TYPE_ATTRIBUTE_PROPERTY", 1).

% Deduction 3: Deductive inference implies proof validation
slot(N, "SOLVER_PROOF_VALIDATED", 1) :- slot(N, "EPIST_DEDUCTIVE_INFERENCE", 1).

% Deduction 4: Axiomatic premises operate under Closed-World Assumption
slot(N, "SOLVER_CWA_CLOSED_WORLD", 1) :- slot(N, "EPIST_AXIOMATIC_PREMISE", 1).
