# QUANTA Context Expansion Action Plan

This document points to the master engineering roadmap and todo-style action plan:

👉 **[`CONTEXT_EXPANSION_ROADMAP.md`](file:///c:/Users/PC/Documents/GitHub/Quanta/CONTEXT_EXPANSION_ROADMAP.md)**

---

## Action Plan Sections Summary

| Section | Title | Primary Focus | Key Deliverable |
|:---:|:---|:---|:---|
| **1** | [Canonical Node Interning & Global Hash-Consing](CONTEXT_EXPANSION_ROADMAP.md#section-1-canonical-node-interning--global-hash-consing-memory-efficiency--node-reuse) | Memory Efficiency & Node Reuse | `src/memory/node_interner.py`, register-masked BLAKE3 CID |
| **2** | [Memory-Mapped Lexical Grounding & High-Throughput Ingestion](CONTEXT_EXPANSION_ROADMAP.md#section-2-memory-mapped-lexical-grounding--high-throughput-ingestion-speed) | Ingestion & Transduction Speed | `data/concept_codebook.bin`, zero-copy SIMD grounder |
| **3** | [Closed-Loop Round-Trip Lattice Meet Gate & Deep NSM Explication](CONTEXT_EXPANSION_ROADMAP.md#section-3-closed-loop-round-trip-lattice-meet-gate--deep-nsm-explication-translation-accuracy) | Translation Accuracy & Cycle-Consistency | `src/verification/lattice_gate.py`, NSM schema expansion |
| **4** | [Query-Driven Spreading-Activation Sub-Graph Attention](CONTEXT_EXPANSION_ROADMAP.md#section-4-query-driven-spreading-activation-sub-graph-attention-context-retrieval) | Context Retrieval & Prompt Injection | `src/memory/spreading_activation.py`, $k$-hop traversal |
| **5** | [Dynamic World-State Tracking & Non-Monotonic Belief Revision](CONTEXT_EXPANSION_ROADMAP.md#section-5-dynamic-world-state-tracking--non-monotonic-belief-revision-world-model) | Temporal Invalidation & State Updates | `src/memory/world_state.py`, fluent state records |
| **6** | [OpenAI-Compatible Reverse Proxy & MCP Server](CONTEXT_EXPANSION_ROADMAP.md#section-6-openai-compatible-reverse-proxy--model-context-protocol-mcp-server-llm-integration) | Host LLM & Agent Middleware | `src/server/proxy.py`, `src/server/mcp_server.py`, `:8000` |
| **7** | [Phase 10 Global Knowledge Base Mount (Wikipedia/Wikidata)](CONTEXT_EXPANSION_ROADMAP.md#section-7-phase-10-global-knowledge-base-mount-wikipedia--wikidata-pre-compilation-world-knowledge) | Encyclopedic Grounding | `src/data/wikidata_ingester.py`, `data/wikipedia_quanta.db` |
| **8** | [Cross-Lingual Multilingual Forward Transduction Adapters](CONTEXT_EXPANSION_ROADMAP.md#section-8-cross-lingual-multilingual-forward-transduction-adapters-universal-pivot) | Universal Non-English Ingestion | `src/parser/multilingual_transducer.py` |

---

## Autonomous Research Protocol (`orx.ps1`)

Every experimental line is managed under OpenResearch (`orx`) following [`AGENTS.md`](file:///c:/Users/PC/Documents/GitHub/Quanta/AGENTS.md) and [`EVAL.md`](file:///c:/Users/PC/Documents/GitHub/Quanta/EVAL.md).
See [`CONTEXT_EXPANSION_ROADMAP.md`](file:///c:/Users/PC/Documents/GitHub/Quanta/CONTEXT_EXPANSION_ROADMAP.md) for full technical specifications, experiment hypotheses, and unit tests.
