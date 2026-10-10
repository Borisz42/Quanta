"""Model Context Protocol (MCP) Server for QUANTA Context Expansion (Section 6).

Implements Task 6.2 & Feature Parity with OpenAI-Compatible Reverse Proxy:
- Standard JSON-RPC 2.0 stdio server supporting the MCP specification (2024-11-05).
- Exposes tools:
  1. quanta_ingest_document: Ingests long document/narrative into PageTable Merkle memory,
     with support for session isolation, validation gates, skeleton formats, and bipartite provenance.
  2. quanta_query_memory: Retrieves minimal verified ASG sub-graph context via spreading activation,
     with support for formats ('english', 'sexpr', 'dual_stream', 'svm'), session isolation,
     global knowledge base toggles, and dynamic context budgeting.
  3. quanta_get_entity_details: Returns full 1024-D vector band analysis, canonical CID, semantic anchor,
     and active graph edges for an entity.
  4. quanta_reset_memory: Flushes working memory, ActiveCanvas, episodic registry, PageTable,
     and session hashes for a given session.
  5. quanta_get_memory_stats: Returns current QUANTA memory metrics including interned node count,
     canvas utilization, Merkle episodes, plus complete reverse proxy health & request telemetry.
  6. quanta_expand_context: Performs token footprint evaluation, bulky discourse decomposition,
     ASG memory ingestion, spreading activation retrieval, system prompt formatting, and bulky turn pruning.
  7. quanta_chat_completion: Full host LLM dialogue execution with neuro-symbolic context expansion,
     forwarding to downstream backend (Unsloth :8888 or cloud provider), GPU guard enforcement,
     or local neuro-symbolic fallback.
  8. quanta_answer_query: Direct sub-10ms neuro-symbolic question answering using verified graph topology
     and honest realizer.
  9. quanta_list_models: Discovers models available for QUANTA context expansion.
- Exposes MCP Resources:
  - quanta://models: List of available models.
  - quanta://memory/stats: Live memory and proxy telemetry stats.
  - quanta://health: Overall health status.
- Exposes MCP Prompts:
  - quanta_context_expansion: Prompt template for context expansion.
  - quanta_knowledge_retrieval: Prompt template for factual knowledge retrieval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.asg import QuantaNode
from core.slots import (
    SLOT_INDEX_TO_NAME,
    SlotBand,
    get_slot_by_index,
)
from core.types import QuantaVector
from pipeline.cognitive_pipeline import CognitivePipeline
from pipeline.tracer import PipelineExecutionTracer
from server.proxy import (
    ChatMessage,
    ChatCompletionRequest,
    ModelCard,
    ModelListResponse,
    PipelinePool,
    QuantaProxyConfig,
    _generate_local_answer,
    estimate_messages_tokens,
    estimate_tokens,
    execute_chat_completion,
    expand_context_dialogue,
)

logger = logging.getLogger("quanta.server.mcp")

# MCP Specification Constants
MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "quanta-mcp"
SERVER_VERSION = "0.1.0"


class MCPServer:
    """Model Context Protocol JSON-RPC 2.0 Server for QUANTA Memory with Reverse Proxy Parity."""

    def __init__(
        self,
        pipeline: Optional[CognitivePipeline] = None,
        page_table_path: Optional[Union[str, Path]] = None,
        transducer_backend: str = "mock",
        global_kb: Optional[Any] = None,
        config: Optional[QuantaProxyConfig] = None,
        pool: Optional[PipelinePool] = None,
    ):
        """Initializes the MCP Server with configuration and session-isolated pipeline pool."""
        self.config = config or QuantaProxyConfig(
            transducer_backend=transducer_backend,
            page_table_path=page_table_path,
            pipeline=pipeline,
        )
        self.pool = pool or PipelinePool(self.config)
        if pipeline is not None:
            self.pool.default_pipeline = pipeline

        self.pipeline = self.pool.default_pipeline
        self.global_kb = global_kb
        if self.global_kb is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            kb_path = repo_root / "data" / "wikipedia_quanta.db"
            if not kb_path.exists():
                kb_path = Path("data/wikipedia_quanta.db")
            if kb_path.exists():
                try:
                    from memory.global_kb import GlobalKnowledgeBase
                    self.global_kb = GlobalKnowledgeBase(kb_path)
                    logger.info("MCPServer automatically mounted GlobalKnowledgeBase from %s", kb_path)
                except Exception as e:
                    logger.debug("Could not auto-mount GlobalKnowledgeBase: %s", e)

        # Telemetry state matching reverse proxy
        self.stats = {
            "start_time": time.time(),
            "total_requests": 0,
            "compressed_requests": 0,
            "tokens_saved": 0,
            "nodes_ingested": 0,
        }
        self.tracer = self.config.tracer or PipelineExecutionTracer.get_instance()

        self._tools = {
            "quanta_ingest_document": self._tool_ingest_document,
            "quanta_query_memory": self._tool_query_memory,
            "quanta_get_entity_details": self._tool_get_entity_details,
            "quanta_reset_memory": self._tool_reset_memory,
            "quanta_get_memory_stats": self._tool_get_memory_stats,
            "quanta_expand_context": self._tool_expand_context,
            "quanta_chat_completion": self._tool_chat_completion,
            "quanta_answer_query": self._tool_answer_query,
            "quanta_list_models": self._tool_list_models,
        }

    # -------------------------------------------------------------------------
    # Async Runner Helper
    # -------------------------------------------------------------------------

    def _run_async(self, coro):
        """Safely executes an async coroutine from synchronous tool methods."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    def _get_session_pipeline(self, session_id: Optional[str] = None) -> CognitivePipeline:
        """Retrieves session-isolated CognitivePipeline instance."""
        return self._run_async(self.pool.get_pipeline(session_id))

    def _get_session_hashes(self, session_id: Optional[str] = None) -> Set[str]:
        """Retrieves session ingested hashes set."""
        return self.pool.get_ingested_hashes(session_id)

    # -------------------------------------------------------------------------
    # JSON-RPC 2.0 Message Dispatcher
    # -------------------------------------------------------------------------

    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes a single JSON-RPC 2.0 request or notification.

        Args:
            message: Parsed JSON dictionary.

        Returns:
            JSON-RPC response dictionary, or None for notifications.
        """
        msg_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})

        is_notification = msg_id is None

        if method == "notifications/initialized":
            logger.info("Client completed MCP initialization handshake")
            return None

        if method == "initialize":
            return self._make_response(msg_id, {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {
                        "listChanged": False,
                    },
                    "resources": {
                        "subscribe": False,
                        "listChanged": False,
                    },
                    "prompts": {
                        "listChanged": False,
                    },
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            })

        if method == "ping":
            return self._make_response(msg_id, {})

        if method == "tools/list":
            return self._make_response(msg_id, {
                "tools": self.get_tool_definitions()
            })

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name not in self._tools:
                return self._make_error(
                    msg_id,
                    code=-32601,
                    message=f"Unknown tool: '{tool_name}'",
                )

            try:
                result_content = self._tools[tool_name](arguments)
                return self._make_response(msg_id, {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result_content),
                        }
                    ],
                    "isError": False,
                })
            except Exception as e:
                logger.exception("Error executing tool '%s': %s", tool_name, e)
                return self._make_response(msg_id, {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error executing tool '{tool_name}': {str(e)}",
                        }
                    ],
                    "isError": True,
                })

        # Resources Support
        if method == "resources/list":
            return self._make_response(msg_id, {
                "resources": [
                    {
                        "uri": "quanta://models",
                        "name": "QUANTA Model Catalog",
                        "mimeType": "application/json",
                        "description": "List of available models supported by QUANTA reverse proxy and MCP",
                    },
                    {
                        "uri": "quanta://memory/stats",
                        "name": "QUANTA Memory & Proxy Statistics",
                        "mimeType": "application/json",
                        "description": "Real-time PageTable node counts, token savings, and proxy health metrics",
                    },
                    {
                        "uri": "quanta://health",
                        "name": "QUANTA Health Status",
                        "mimeType": "application/json",
                        "description": "Service health, uptime, and backend configuration",
                    },
                ]
            })

        if method == "resources/read":
            uri = params.get("uri")
            if uri == "quanta://models":
                text = self._tool_list_models()
            elif uri in ("quanta://memory/stats", "quanta://health"):
                text = self._tool_get_memory_stats()
            else:
                return self._make_error(msg_id, code=-32602, message=f"Resource not found: '{uri}'")

            return self._make_response(msg_id, {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": text,
                    }
                ]
            })

        # Prompts Support
        if method == "prompts/list":
            return self._make_response(msg_id, {
                "prompts": [
                    {
                        "name": "quanta_context_expansion",
                        "description": "Compress long dialogue into minimal verified neuro-symbolic context",
                        "arguments": [
                            {"name": "history", "description": "Prior dialogue turns", "required": True},
                            {"name": "query", "description": "Active user question", "required": True},
                        ],
                    },
                    {
                        "name": "quanta_knowledge_retrieval",
                        "description": "Query QUANTA Merkle memory for grounded factual subgraphs",
                        "arguments": [
                            {"name": "query", "description": "Question or entity name", "required": True},
                        ],
                    },
                ]
            })

        if method == "prompts/get":
            prompt_name = params.get("name")
            p_args = params.get("arguments", {})
            if prompt_name == "quanta_context_expansion":
                hist = p_args.get("history", "")
                q = p_args.get("query", "")
                messages = [
                    {"role": "user", "content": f"Context:\n{hist}\n\nQuestion:\n{q}"}
                ]
                return self._make_response(msg_id, {"messages": messages})
            elif prompt_name == "quanta_knowledge_retrieval":
                q = p_args.get("query", "")
                messages = [
                    {"role": "user", "content": f"Retrieve verified facts for: {q}"}
                ]
                return self._make_response(msg_id, {"messages": messages})
            else:
                return self._make_error(msg_id, code=-32602, message=f"Prompt not found: '{prompt_name}'")

        # Unknown method
        if is_notification:
            return None
        return self._make_error(
            msg_id,
            code=-32601,
            message=f"Method not found: '{method}'",
        )

    # -------------------------------------------------------------------------
    # Tool Definitions
    # -------------------------------------------------------------------------

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns the list of available MCP tool definitions and JSON schemas."""
        return [
            {
                "name": "quanta_ingest_document",
                "description": (
                    "Ingest a document or narrative text into QUANTA's neuro-symbolic "
                    "Virtual Page-Table Merkle memory, compiling it into canonical 1024-D ASGs. "
                    "Supports session isolation, validation gates, and skeleton format overrides."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The raw text content or narrative document to ingest and verify.",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Optional unique document or chapter identifier.",
                            "default": "doc_01",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID for isolated memory partition.",
                            "default": "default",
                        },
                        "validate": {
                            "type": "boolean",
                            "description": "Whether to trigger formal ASP/Belnap validation gate.",
                            "default": False,
                        },
                        "skeleton_format": {
                            "type": "string",
                            "enum": ["sexpr_compact", "sexpr_positional", "json"],
                            "description": "S-expression transduction format override.",
                        },
                        "co_decoded_kev": {
                            "type": "boolean",
                            "description": "Whether to enable co-decoded single-letter Kev decisions.",
                        },
                        "background_ingest": {
                            "type": "boolean",
                            "description": "Whether background ingestion is enabled.",
                        },
                        "fast_path_mode": {
                            "type": "string",
                            "description": "Fast-path scoring mode (e.g. 'coverage_adaptive', 'passthrough').",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "quanta_query_memory",
                "description": (
                    "Retrieve minimal verified ASG sub-graph context for an external LLM "
                    "using query-driven spreading activation along valency and causal links. "
                    "Supports multiple output formats ('english', 'sexpr', 'dual_stream', 'svm') "
                    "and dynamic context budgeting."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language question or thematic query pattern.",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum token budget for retrieved context.",
                            "default": 500,
                        },
                        "format": {
                            "type": "string",
                            "enum": ["english", "sexpr", "dual_stream", "svm"],
                            "description": "Output format: 'english', 'sexpr', 'dual_stream', or 'svm'.",
                            "default": "english",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID for memory context.",
                            "default": "default",
                        },
                        "global_kb": {
                            "type": "boolean",
                            "description": "Mount and query global encyclopedic Wikidata knowledge base.",
                        },
                        "no_global_kb": {
                            "type": "boolean",
                            "description": "Explicitly disable global encyclopedic lookup.",
                        },
                        "dynamic_budget": {
                            "type": "boolean",
                            "description": "Enable adaptive context budgeting based on query hop depth.",
                            "default": False,
                        },
                        "fast_path_mode": {
                            "type": "string",
                            "description": "Fast-path retrieval override.",
                        },
                        "validate": {
                            "type": "boolean",
                            "description": "Run logic validator gate during query evaluation.",
                            "default": False,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "quanta_get_entity_details",
                "description": (
                    "Inspect full 1024-dimension quaternary vector analysis, canonical BLAKE3 CID, "
                    "semantic anchor, and active graph valencies for an entity or concept."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name_or_cid": {
                            "type": "string",
                            "description": "Entity canonical CID, anchor name, or literal text (e.g. 'Eleanor Vance').",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID.",
                            "default": "default",
                        },
                    },
                    "required": ["name_or_cid"],
                },
            },
            {
                "name": "quanta_reset_memory",
                "description": (
                    "Flushes ephemeral working memory, resetting ActiveCanvas, PageTable session nodes, "
                    "and ingested turn hashes for a specified session."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID to reset (default: 'default').",
                            "default": "default",
                        },
                    },
                },
            },
            {
                "name": "quanta_get_memory_stats",
                "description": (
                    "Returns current QUANTA memory metrics including interned node count, canvas utilization, "
                    "Merkle episodes, plus reverse proxy request telemetry and token savings."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID.",
                            "default": "default",
                        },
                    },
                },
            },
            {
                "name": "quanta_expand_context",
                "description": (
                    "Evaluates a list of dialogue messages or a long context prompt, ingests bulky prior discourse "
                    "into ASG PageTable memory, retrieves minimal grounded context, and returns compressed messages "
                    "with enriched system prompt and detailed token compression telemetry."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                            "description": "List of chat messages in standard OpenAI format.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Alternative single-turn user prompt containing bulky context.",
                        },
                        "threshold": {
                            "type": "integer",
                            "description": "Token count threshold above which compression is triggered.",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["english", "sexpr", "dual_stream", "svm"],
                            "description": "Context injection format.",
                            "default": "english",
                        },
                        "force_enrich": {
                            "type": "boolean",
                            "description": "Force context enrichment even if under token threshold.",
                            "default": False,
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID.",
                            "default": "default",
                        },
                        "global_kb": {
                            "type": "boolean",
                            "description": "Enable global encyclopedic Wikidata knowledge base.",
                        },
                        "no_global_kb": {
                            "type": "boolean",
                            "description": "Explicitly disable global encyclopedic Wikidata lookup.",
                        },
                        "validate": {
                            "type": "boolean",
                            "description": "Run formal validation gates during ASG ingestion.",
                            "default": False,
                        },
                        "max_context_tokens": {
                            "type": "integer",
                            "description": "Maximum token budget for injected context.",
                        },
                        "fast_path_mode": {
                            "type": "string",
                            "description": "Fast-path scoring mode override.",
                        },
                        "background_ingest": {
                            "type": "boolean",
                            "description": "Enable background ingestion.",
                        },
                        "profile": {
                            "type": "string",
                            "description": "Execution profile: 'passthrough' or 'calibrated'.",
                        },
                        "skeleton_format": {
                            "type": "string",
                            "description": "S-expression transduction format.",
                        },
                        "co_decoded": {
                            "type": "boolean",
                            "description": "Co-decoded single-letter Kev decisions.",
                        },
                    },
                },
            },
            {
                "name": "quanta_chat_completion",
                "description": (
                    "Full OpenAI-compatible chat completion through QUANTA. Intercepts dialogue, "
                    "ingests bulky history into PageTable memory, retrieves compact verified context, "
                    "forwards compressed messages to downstream LLM backend (or provides local neuro-symbolic fallback), "
                    "and returns the assistant completion with token usage and quanta_metadata."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                            "description": "List of chat messages.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Alternative single-turn user prompt.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Target model ID.",
                            "default": "quanta-context-expander",
                        },
                        "threshold": {
                            "type": "integer",
                            "description": "Token compression threshold.",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["english", "sexpr", "dual_stream", "svm"],
                            "default": "english",
                        },
                        "force_enrich": {
                            "type": "boolean",
                            "default": False,
                        },
                        "session_id": {
                            "type": "string",
                            "default": "default",
                        },
                        "global_kb": {"type": "boolean"},
                        "no_global_kb": {"type": "boolean"},
                        "validate": {"type": "boolean", "default": False},
                        "max_context_tokens": {"type": "integer"},
                        "temperature": {"type": "number", "default": 1.0},
                        "max_tokens": {"type": "integer"},
                        "fast_path_mode": {"type": "string"},
                        "background_ingest": {"type": "boolean"},
                        "profile": {"type": "string"},
                        "skeleton_format": {"type": "string"},
                        "co_decoded": {"type": "boolean"},
                        "timeout": {"type": "number"},
                        "raw_json": {
                            "type": "boolean",
                            "description": "Return full OpenAI completion JSON (True) or only text (False).",
                            "default": True,
                        },
                    },
                },
            },
            {
                "name": "quanta_answer_query",
                "description": (
                    "Direct sub-10ms neuro-symbolic question answering over the verified knowledge graph "
                    "topology without calling external host LLMs."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question to answer from the verified knowledge graph.",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID.",
                            "default": "default",
                        },
                        "format": {
                            "type": "string",
                            "default": "english",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "default": 500,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "quanta_list_models",
                "description": "Lists models supported by QUANTA reverse proxy and MCP server.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def _tool_ingest_document(self, arguments: Dict[str, Any]) -> str:
        """Executes quanta_ingest_document tool with dual-node SVM bipartite reporting."""
        content = arguments.get("content", "")
        doc_id = arguments.get("doc_id", "doc_01")
        session_id = arguments.get("session_id", "default")
        validate = bool(arguments.get("validate", False))
        skeleton_format = arguments.get("skeleton_format")
        co_decoded_kev = arguments.get("co_decoded_kev")
        background_ingest = arguments.get("background_ingest")
        fast_path_mode = arguments.get("fast_path_mode")

        if not content or not content.strip():
            return "Error: Document content cannot be empty."

        pipeline = self._get_session_pipeline(session_id)
        if skeleton_format is not None or co_decoded_kev is not None:
            tgt_fmt = skeleton_format or pipeline.skeleton_format
            tgt_co = co_decoded_kev if co_decoded_kev is not None else pipeline.co_decoded_kev
            pipeline.set_skeleton_format(tgt_fmt, co_decoded=tgt_co)

        if background_ingest is not None:
            bg_flag = str(background_ingest).lower().strip() in ("1", "true", "yes", "on")
            pipeline.background_enabled = bg_flag
            if hasattr(pipeline, "multi_scale_config") and pipeline.multi_scale_config is not None:
                pipeline.multi_scale_config = pipeline.multi_scale_config.with_overrides(
                    {"background.enabled": bg_flag}
                )

        t0 = time.perf_counter()
        try:
            # Ingest through cognitive pipeline with SVM dual-node storage
            if hasattr(pipeline, "ingest_document"):
                graph = pipeline.ingest_document(content, doc_id=doc_id)
            else:
                graph = pipeline.process(content, chapter_id=doc_id, validate=validate)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            import hashlib
            doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            self._get_session_hashes(session_id).add(doc_hash)

            node_count = pipeline.page_table.count_nodes() if hasattr(pipeline.page_table, "count_nodes") else len(pipeline.page_table)
            root_cid = graph.root.compute_cid() if (graph and graph.root) else (getattr(graph, "root_cid", "none") or "none")

            self.stats["nodes_ingested"] = node_count

            # Bipartite graph statistics
            ps = getattr(pipeline, "passage_store", None)
            total_passages = len(ps) if ps is not None else 0
            bipartite_anchors = len(graph.node_to_passage) if hasattr(graph, "node_to_passage") else 0
            bt = getattr(pipeline, "binary_table", None)
            binary_nodes = len(bt) if bt is not None else 0

            # Collect grounded passage span mappings
            span_mappings = []
            if hasattr(graph, "node_to_passage") and graph.node_to_passage:
                for n_cid, (p_id, s_start, s_end) in list(graph.node_to_passage.items())[:6]:
                    node = graph.get_node(n_cid)
                    anchor_txt = node.anchor if node else n_cid[:8]
                    span_mappings.append(f"  * Node [{anchor_txt}]: {p_id} @ [{s_start}:{s_end}]")

            span_report = ("\n- Grounded Passage Spans:\n" + "\n".join(span_mappings)) if span_mappings else ""

            return (
                f"Successfully ingested document '{doc_id}' into QUANTA Semantic Virtual Memory in {elapsed_ms:.2f} ms (Session: '{session_id}').\n"
                f"- Graph Nodes Created: {len(graph.nodes) if graph else 0}\n"
                f"- Graph Merkle Root CID: {root_cid}\n"
                f"- Bipartite Provenance Links (E_ground): {bipartite_anchors}\n"
                f"- Total Passage Records (V_passage): {total_passages}\n"
                f"- 128-Byte Binary Node Table Entries: {binary_nodes}\n"
                f"- Total PageTable Interned Nodes: {node_count}"
                f"{span_report}"
            )
        except Exception as e:
            logger.exception("Error ingesting document '%s' in MCP: %s", doc_id, e)
            return f"Error ingesting document '{doc_id}': {str(e)}"

    def _tool_reset_memory(self, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Executes quanta_reset_memory tool with session-isolated reset."""
        session_id = arguments.get("session_id", "default") if arguments else "default"
        self.pool.reset_session(session_id)
        if session_id == "default" and hasattr(self, "pipeline") and self.pipeline is not self.pool.default_pipeline:
            if hasattr(self.pipeline, "active_canvas"):
                self.pipeline.active_canvas.clear()
            if hasattr(self.pipeline, "passage_store"):
                self.pipeline.passage_store.clear()
        return f"Successfully reset QUANTA memory state and session hashes for session '{session_id}'."

    def _tool_get_memory_stats(self, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Executes quanta_get_memory_stats tool returning combined memory & proxy telemetry."""
        session_id = arguments.get("session_id", "default") if arguments else "default"
        pipeline = self._get_session_pipeline(session_id)

        uptime = time.time() - self.stats["start_time"]
        pt = pipeline.page_table
        node_count = pt.count_nodes() if hasattr(pt, "count_nodes") else len(pt)
        canvas_len = len(pipeline.active_canvas) if hasattr(pipeline, "active_canvas") else 0
        canvas_cap = pipeline.active_canvas.capacity if hasattr(pipeline, "active_canvas") else 512
        vector_count = len(pt.vector_index) if hasattr(pt, "vector_index") else 0
        ps = getattr(pipeline, "passage_store", None)
        bt = getattr(pipeline, "binary_table", None)

        stats = {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "total_requests": self.stats["total_requests"],
            "compressed_requests": self.stats["compressed_requests"],
            "tokens_saved_estimate": self.stats["tokens_saved"],
            "total_page_table_nodes": node_count,
            "vector_index_size": vector_count,
            "active_canvas_usage": f"{canvas_len}/{canvas_cap} nodes",
            "passage_store_count": len(ps) if ps is not None else 0,
            "binary_table_count": len(bt) if bt is not None else 0,
            "backend_url": self.config.backend_url,
            "compression_threshold": self.config.compression_threshold,
            "global_kb_mounted": (getattr(pipeline.page_table, "global_kb", None) is not None) or (self.global_kb is not None),
            "has_merkle_book": hasattr(pipeline, "merkle_book"),
            "transducer_backend": self.config.transducer_backend,
            "skeleton_format": getattr(pipeline, "skeleton_format", self.config.skeleton_format),
            "co_decoded_kev": getattr(pipeline, "co_decoded_kev", self.config.co_decoded_kev),
            "session_id": session_id,
        }
        return json.dumps(stats, indent=2)

    def _tool_query_memory(self, arguments: Dict[str, Any]) -> str:
        """Executes quanta_query_memory tool with dual-stream context and passage span reporting."""
        query = arguments.get("query", "")
        max_tokens = int(arguments.get("max_tokens", 500))
        format_type = str(arguments.get("format", "english"))
        session_id = arguments.get("session_id", "default")
        global_kb_opt = arguments.get("global_kb")
        no_global_kb_opt = arguments.get("no_global_kb")
        dynamic_budget = bool(arguments.get("dynamic_budget", False))
        fast_path_mode = arguments.get("fast_path_mode")

        if not query or not query.strip():
            return "Error: Query string cannot be empty."

        pipeline = self._get_session_pipeline(session_id)

        # Domain-aware gating
        is_code_task = bool(re.search(r"(?:def\s+\w+\s*\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import)", query))

        # Dynamic context budgeting if requested or hop depth >= 3
        effective_max_tokens = max_tokens
        if dynamic_budget:
            query_hop_depth = pipeline.retriever.detect_query_hop_depth(query) if hasattr(pipeline, "retriever") else 2
            if query_hop_depth >= 3:
                effective_max_tokens = max(max_tokens, 1200)

        # Mount/unmount global KB on pipeline if specified
        if global_kb_opt and getattr(pipeline.page_table, "global_kb", None) is None:
            if self.global_kb is not None:
                pipeline.page_table.mount_global_kb(self.global_kb)
        elif no_global_kb_opt and hasattr(pipeline.page_table, "global_kb"):
            pipeline.page_table.global_kb = None

        # Query via SVM query_memory or retrieve_context
        if format_type in ("dual_stream", "svm") and hasattr(pipeline, "query_memory"):
            raw_ctx = pipeline.query_memory(query, max_tokens=effective_max_tokens, format="dual_stream")
            if hasattr(raw_ctx, "full_context"):
                context = raw_ctx.full_context
                # Report bipartite passage mappings if available
                if raw_ctx.passages:
                    passage_info = []
                    for p in raw_ctx.passages[:4]:
                        passage_info.append(f"  * Passage {p.passage_id} (Doc: {p.doc_id}, Score: {p.score:.2f}) @ span {p.char_span}")
                    context = f"{context}\n\n[Bipartite Retrieved Passages]\n" + "\n".join(passage_info)
            else:
                context = str(raw_ctx)
        else:
            context = pipeline.retrieve_context(
                query=query,
                format=format_type,
                max_tokens=effective_max_tokens,
            )

        if not context or not context.strip():
            kb_to_use = getattr(pipeline.page_table, "global_kb", None) or self.global_kb
            if kb_to_use is not None and not is_code_task and not no_global_kb_opt:
                # Query global encyclopedic knowledge base
                kb_nodes = kb_to_use.lookup_entity(query, limit=3)
                if kb_nodes:
                    facts = []
                    for kn in kb_nodes:
                        qid = kn.literal.get("qid") if kn.literal else None
                        lbl = kn.literal.get("label", kn.anchor) if kn.literal else kn.anchor
                        desc = kn.literal.get("description", "") if kn.literal else ""
                        triples = kb_to_use.get_triples(subject_qid=qid, limit=8) if qid else []
                        triple_strs = [f"{t['property_name']}: {t['object_qid']}" for t in triples]
                        fact_desc = f"- {lbl} ({qid}): {desc}" + (f" | Relations: {', '.join(triple_strs)}" if triple_strs else "")
                        facts.append(fact_desc)
                    return "QUANTA Encyclopedic Knowledge Graph:\n" + "\n".join(facts)

            # Fallback to direct answerer check
            ans = pipeline.answer_query(query)
            if ans and ans.strip() and "I do not have sufficient" not in ans and not ans.startswith("The hypothesis"):
                return f"Direct Graph Fact: {ans.strip()}"
            return f"No active sub-graph memories found matching query: '{query}'."

        return context.strip()

    def _tool_expand_context(self, arguments: Dict[str, Any]) -> str:
        """Executes quanta_expand_context tool to compress dialogue and retrieve verified context."""
        raw_msgs = arguments.get("messages")
        prompt = arguments.get("prompt")
        if not raw_msgs and prompt:
            raw_msgs = [{"role": "user", "content": prompt}]
        elif not raw_msgs:
            return json.dumps({"error": "Either 'messages' list or 'prompt' string must be provided."})

        session_id = arguments.get("session_id", "default")
        threshold = arguments.get("threshold")
        format_type = arguments.get("format", "english")
        force_enrich = bool(arguments.get("force_enrich", False))
        global_kb = arguments.get("global_kb")
        no_global_kb = arguments.get("no_global_kb")
        validate = bool(arguments.get("validate", False))
        max_context_tokens = arguments.get("max_context_tokens")
        fast_path_mode = arguments.get("fast_path_mode")
        background_ingest = arguments.get("background_ingest")
        profile = arguments.get("profile")
        skeleton_format = arguments.get("skeleton_format")
        co_decoded = arguments.get("co_decoded")

        pipeline = self._get_session_pipeline(session_id)
        hashes = self._get_session_hashes(session_id)

        res = self._run_async(
            expand_context_dialogue(
                messages=raw_msgs,
                pipeline=pipeline,
                ingested_turn_hashes=hashes,
                config=self.config,
                threshold=threshold,
                format_type=format_type,
                force_enrich=force_enrich,
                global_kb=global_kb,
                no_global_kb=no_global_kb,
                validate=validate,
                max_context_tokens=max_context_tokens,
                fast_path_mode=fast_path_mode,
                background_ingest=background_ingest,
                profile=profile,
                skeleton_format=skeleton_format,
                co_decoded=co_decoded,
                stats=self.stats,
                tracer=self.tracer,
            )
        )

        raw_tokens = res["raw_tokens"]
        comp_tokens = res["compressed_tokens"]
        saved = res["tokens_saved"]
        ratio = round(comp_tokens / max(1, raw_tokens), 4)

        output = {
            "compressed_messages": res["compressed_messages"],
            "retrieved_context": res["retrieved_context"],
            "user_query": res["user_query"],
            "raw_tokens": raw_tokens,
            "compressed_tokens": comp_tokens,
            "tokens_saved": saved,
            "compression_ratio": ratio,
            "quanta_metadata": res["quanta_meta"],
            "headers": res["quanta_headers"],
        }
        return json.dumps(output, indent=2)

    def _tool_chat_completion(self, arguments: Dict[str, Any]) -> str:
        """Executes full reverse-proxy chat completion through MCP."""
        raw_msgs = arguments.get("messages")
        prompt = arguments.get("prompt")
        if not raw_msgs and prompt:
            raw_msgs = [{"role": "user", "content": prompt}]
        elif not raw_msgs:
            return json.dumps({"error": "Either 'messages' list or 'prompt' string must be provided."})

        messages = [
            ChatMessage(**m) if isinstance(m, dict) else m
            for m in raw_msgs
        ]

        model = arguments.get("model", "quanta-context-expander")
        temperature = float(arguments.get("temperature", 1.0))
        max_tokens = arguments.get("max_tokens")
        session_id = arguments.get("session_id", "default")
        threshold = arguments.get("threshold")
        format_type = arguments.get("format", "english")
        force_enrich = bool(arguments.get("force_enrich", False))
        global_kb = arguments.get("global_kb")
        no_global_kb = arguments.get("no_global_kb")
        validate = bool(arguments.get("validate", False))
        max_context_tokens = arguments.get("max_context_tokens")
        timeout = arguments.get("timeout")
        fast_path_mode = arguments.get("fast_path_mode")
        background_ingest = arguments.get("background_ingest")
        profile = arguments.get("profile")
        skeleton_format = arguments.get("skeleton_format")
        co_decoded = arguments.get("co_decoded")
        raw_json = bool(arguments.get("raw_json", True))

        req = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        res = self._run_async(
            execute_chat_completion(
                request=req,
                pool=self.pool,
                config=self.config,
                stats=self.stats,
                session_id=session_id,
                threshold=threshold,
                format_type=format_type,
                force_enrich=force_enrich,
                global_kb=global_kb,
                no_global_kb=no_global_kb,
                validate=validate,
                max_context_tokens=max_context_tokens,
                timeout=timeout,
                fast_path_mode=fast_path_mode,
                background_ingest=background_ingest,
                profile=profile,
                skeleton_format=skeleton_format,
                co_decoded=co_decoded,
                as_dict=True,
            )
        )

        if not raw_json and isinstance(res, dict):
            choices = res.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")

        return json.dumps(res, indent=2)

    def _tool_answer_query(self, arguments: Dict[str, Any]) -> str:
        """Executes direct neuro-symbolic question answering using verified graph topology."""
        query = arguments.get("query", "").strip()
        session_id = arguments.get("session_id", "default")
        format_type = arguments.get("format", "english")
        max_tokens = int(arguments.get("max_tokens", 500))

        if not query:
            return "Error: query parameter is required."

        pipeline = self._get_session_pipeline(session_id)
        # Try direct graph query lookup
        ans = pipeline.answer_query(query)
        if ans and ans.strip() and "I do not have sufficient" not in ans and not ans.startswith("The hypothesis"):
            return ans.strip()

        # Retrieve verified context
        ctx = pipeline.retrieve_context(query=query, format=format_type, max_tokens=max_tokens)
        return _generate_local_answer(pipeline, query, ctx)

    def _tool_list_models(self, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Returns JSON list of supported models."""
        models = [
            {"id": "quanta-context-expander", "object": "model", "owned_by": "quanta", "root": "quanta-context-expander"},
            {"id": "gpt-4o", "object": "model", "owned_by": "quanta", "root": "quanta-context-expander"},
            {"id": "claude-3-5-sonnet", "object": "model", "owned_by": "quanta", "root": "quanta-context-expander"},
            {"id": "qwen-2.5-7b", "object": "model", "owned_by": "quanta", "root": "quanta-context-expander"},
        ]
        return json.dumps({"object": "list", "data": models}, indent=2)

    def _tool_get_entity_details(self, arguments: Dict[str, Any]) -> str:
        """Executes quanta_get_entity_details tool."""
        target = str(arguments.get("name_or_cid", "")).strip()
        session_id = arguments.get("session_id", "default")
        if not target:
            return "Error: name_or_cid parameter is required."

        pipeline = self._get_session_pipeline(session_id)

        # 1. Look up node
        node = self._find_node(target, pipeline=pipeline)
        if node is None:
            return f"Entity '{target}' not found in QUANTA memory or interner."

        # 2. Extract vector analysis across all 8 bands
        cid = node.compute_cid()
        vec = node.vector
        edges = node.edges

        active_slots = []
        band_distribution: Dict[str, int] = {f"Band {b}": 0 for b in range(8)}

        for slot_idx in range(1024):
            val = int(vec[slot_idx])
            if val != 0:
                band_num = slot_idx // 128
                band_name = SlotBand(band_num).name if band_num < 8 else f"BAND_{band_num}"
                band_distribution[f"Band {band_num}"] += 1
                slot_def = get_slot_by_index(slot_idx)
                slot_name = slot_def.name if slot_def else SLOT_INDEX_TO_NAME.get(slot_idx, f"SLOT_{slot_idx}")
                cat = slot_def.category if slot_def else "General"
                active_slots.append({
                    "index": slot_idx,
                    "slot_name": slot_name,
                    "value": val,
                    "value_name": {1: "TRUE/POS", 2: "FALSE/NEG", 3: "UNCERTAIN/QUERY"}.get(val, str(val)),
                    "band": band_name,
                    "category": cat,
                })

        details = {
            "canonical_cid": cid,
            "anchor": node.anchor,
            "literal": node.literal,
            "active_edges": edges,
            "total_active_slots": len(active_slots),
            "band_distribution": band_distribution,
            "sample_active_slots": active_slots[:20],
        }

        return json.dumps(details, indent=2)

    def _find_node(self, target: str, pipeline: Optional[CognitivePipeline] = None) -> Optional[QuantaNode]:
        """Finds a QuantaNode by CID, anchor, or literal in PageTable / Interner."""
        pipe = pipeline or self.pipeline
        pt = pipe.page_table

        # 1. Direct CID fetch
        if hasattr(pt, "fetch_node"):
            node = pt.fetch_node(target)
            if node is not None:
                return node

        # 2. Inter-process interner lookup
        if hasattr(pipe, "compiler") and hasattr(pipe.compiler, "interner"):
            interner = pipe.compiler.interner
            if interner is not None:
                node = interner.lookup(canonical_cid=target)
                if node is not None:
                    return node
                node = interner.lookup(anchor=target)
                if node is not None:
                    return node
                node = interner.lookup(literal=target)
                if node is not None:
                    return node

        # 3. Scan SQLite nodes table if accessible
        if hasattr(pt, "_conn"):
            with pt._lock:
                cur = pt._conn.cursor()
                cur.execute("SELECT cid FROM nodes WHERE literal LIKE ? LIMIT 1", (f"%{target}%",))
                row = cur.fetchone()
                if row:
                    return pt.fetch_node(row[0])
                cur.execute(
                    """
                    SELECT n.cid FROM nodes n
                    JOIN interned_strings s ON n.anchor_id = s.id
                    WHERE s.text LIKE ? LIMIT 1
                    """,
                    (f"%{target}%",),
                )
                row = cur.fetchone()
                if row:
                    return pt.fetch_node(row[0])

        # 4. Fallback to GlobalKnowledgeBase if available
        kb_to_use = getattr(pt, "global_kb", None) or self.global_kb
        if kb_to_use is not None:
            if target.upper().startswith("Q") and target[1:].isdigit():
                kn = kb_to_use.get_entity_by_qid(target.upper())
                if kn is not None:
                    return kn
            kn_list = kb_to_use.lookup_entity(target, limit=1)
            if kn_list:
                return kn_list[0]

        return None

    # -------------------------------------------------------------------------
    # Response Formatting Helpers
    # -------------------------------------------------------------------------

    def _make_response(self, msg_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result,
        }

    def _make_error(self, msg_id: Any, code: int, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
        err = {
            "code": code,
            "message": message,
        }
        if data is not None:
            err["data"] = data
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": err,
        }

    # -------------------------------------------------------------------------
    # Stdio Event Loop Runner
    # -------------------------------------------------------------------------

    def run_stdio(self):
        """Runs the blocking stdio JSON-RPC 2.0 loop reading from stdin and writing to stdout."""
        logger.info("Starting QUANTA MCP stdio server loop...")
        for line in sys.stdin:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                msg = json.loads(line_str)
                resp = self.handle_message(msg)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                logger.exception("Malformed JSON-RPC message: %s", e)
                err_resp = self._make_error(
                    msg_id=None,
                    code=-32700,
                    message=f"Parse error: {str(e)}",
                )
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def run_mcp_stdio(page_table_path: Optional[Union[str, Path]] = None, config: Optional[QuantaProxyConfig] = None):
    """Entrypoint function to run the MCP server over stdio."""
    server = MCPServer(page_table_path=page_table_path, config=config)
    server.run_stdio()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="QUANTA Model Context Protocol (MCP) Server")
    parser.add_argument("--test", action="store_true", help="Run automated MCP tool self-test")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive MCP test shell")
    parser.add_argument("--db-path", type=str, default=None, help="Path to PageTable database")
    args = parser.parse_args()

    if args.test or args.interactive:
        from server.service_manager import get_service_manager
        mgr = get_service_manager()
        if args.interactive:
            mgr.run_mcp_interactive()
        else:
            mgr.run_mcp_test()
    else:
        run_mcp_stdio(page_table_path=args.db_path)


if __name__ == "__main__":
    main()
