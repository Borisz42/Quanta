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

% Rule 3: Mutual exclusivity of fundamental entity types
:- slot(N, "TYPE_ANIMATE", 1),
   slot(N, "TYPE_INANIMATE_PHYSICAL", 1).

:- slot(N, "TYPE_ABSTRACT_CONCEPT", 1),
   slot(N, "TYPE_INANIMATE_PHYSICAL", 1).

% ----------------------------------------------------------------------
% 2. Graph Valency & Thematic Role Constraints
% ----------------------------------------------------------------------

% Rule 4: An entity acting as an Agent (VAL_X1_AGENT) in a literal event
% must be agent-capable (animate, human, organization, or explicitly agent-capable)
:- edge(EventNode, "VAL_X1_AGENT", AgentNode),
   slot(EventNode, "MODALITY_LITERAL", 1),
   not slot(AgentNode, "ROLE_AGENT_CAPABLE", 1),
   not slot(AgentNode, "TYPE_HUMAN", 1),
   not slot(AgentNode, "TYPE_ANIMATE", 1),
   not slot(AgentNode, "TYPE_ORGANIZATION", 1).

% Rule 5: Experiencer role requires sentient entity
:- edge(EventNode, "VAL_EXPERIENCER", ExpNode),
   slot(EventNode, "MODALITY_LITERAL", 1),
   not slot(ExpNode, "ROLE_SENTIENT", 1),
   not slot(ExpNode, "TYPE_HUMAN", 1),
   not slot(ExpNode, "TYPE_ANIMATE", 1).

% ----------------------------------------------------------------------
% 3. Epistemic & Deontic Modal Consistency
% ----------------------------------------------------------------------

% Rule 6: Deontic contradiction: Simultaneous Obligation and Prohibition
:- slot(N, "EPIST_DEONTIC_OBLIGATION", 1),
   slot(N, "EPIST_DEONTIC_PROHIBITION", 1).

% Rule 7: Direct observation cannot be marked as hearsay
:- slot(N, "EPIST_DIRECT_OBSERVATION", 1),
   slot(N, "EPIST_HEARSAY_TESTIMONY", 1).

% Rule 8: Unhedged direct affirmation cannot be simultaneously negated
:- slot(N, "NSM_TRUE", 1),
   slot(N, "LJB_NA_NEGATION", 1),
   not slot(N, "NSM_MAYBE", 1),
   not slot(N, "MODALITY_FIGURATIVE", 1).
