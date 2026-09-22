"""Model Context Protocol (MCP) Server for QUANTA Context Expansion (Section 6).

Implements Task 6.2:
- Standard JSON-RPC 2.0 stdio server supporting the MCP specification (2024-11-05).
- Exposes tools:
  1. quanta_ingest_document(doc_id: str, content: str)
     - Ingests long document/narrative into PageTable Merkle memory.
  2. quanta_query_memory(query: str, max_tokens: int = 500, format: str = "english")
     - Retrieves minimal verified ASG sub-graph context via spreading activation.
  3. quanta_get_entity_details(name_or_cid: str)
     - Returns full 1024-D vector band analysis, canonical CID, semantic anchor,
       and active graph edges for an entity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.asg import QuantaNode
from core.slots import (
    SLOT_INDEX_TO_NAME,
    SlotBand,
    get_slot_by_index,
)
from core.types import QuantaVector
from pipeline.cognitive_pipeline import CognitivePipeline

logger = logging.getLogger("quanta.server.mcp")

# MCP Specification Constants
MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "quanta-mcp"
SERVER_VERSION = "0.1.0"


class MCPServer:
    """Model Context Protocol JSON-RPC 2.0 Server for QUANTA Memory."""

    def __init__(
        self,
        pipeline: Optional[CognitivePipeline] = None,
        page_table_path: Optional[Union[str, Path]] = None,
        transducer_backend: str = "mock",
    ):
        """Initializes the MCP Server with an underlying CognitivePipeline."""
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = CognitivePipeline(
                transducer_backend=transducer_backend,
                page_table_path=page_table_path,
            )

        self._tools = {
            "quanta_ingest_document": self._tool_ingest_document,
            "quanta_query_memory": self._tool_query_memory,
            "quanta_get_entity_details": self._tool_get_entity_details,
        }

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

        # Notifications do not expect a response
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
                    }
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
                            "text": result_content,
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
                    "Virtual Page-Table Merkle memory, compiling it into canonical 1024-D ASGs."
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
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "quanta_query_memory",
                "description": (
                    "Retrieve minimal verified ASG sub-graph context for an external LLM "
                    "using query-driven spreading activation along valency and causal links."
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
                            "enum": ["english", "sexpr"],
                            "description": "Output format: 'english' (compositional prose) or 'sexpr' (GBNF S-expression).",
                            "default": "english",
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
                    },
                    "required": ["name_or_cid"],
                },
            },
        ]

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def _tool_ingest_document(self, arguments: Dict[str, Any]) -> str:
        """Executes quanta_ingest_document tool."""
        content = arguments.get("content", "")
        doc_id = arguments.get("doc_id", "doc_01")

        if not content or not content.strip():
            return "Error: Document content cannot be empty."

        t0 = time.perf_counter()
        # Ingest through cognitive pipeline with Merkle folding
        graph = self.pipeline.process(content, chapter_id=doc_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        node_count = self.pipeline.page_table.count_nodes() if hasattr(self.pipeline.page_table, "count_nodes") else len(self.pipeline.page_table)
        root_cid = graph.root.compute_cid() if graph.root else "none"

        return (
            f"Successfully ingested document '{doc_id}' into QUANTA memory in {elapsed_ms:.2f} ms.\n"
            f"- Graph Nodes Created: {len(graph.nodes)}\n"
            f"- Graph Merkle Root CID: {root_cid}\n"
            f"- Total PageTable Interned Nodes: {node_count}"
        )

    def _tool_query_memory(self, arguments: Dict[str, Any]) -> str:
        """Executes quanta_query_memory tool."""
        query = arguments.get("query", "")
        max_tokens = int(arguments.get("max_tokens", 500))
        format_type = str(arguments.get("format", "english"))

        if not query or not query.strip():
            return "Error: Query string cannot be empty."

        context = self.pipeline.retrieve_context(
            query=query,
            format=format_type,
            max_tokens=max_tokens,
        )

        if not context or not context.strip():
            # Fallback to direct answerer check
            ans = self.pipeline.answer_query(query)
            if ans and ans.strip():
                return f"Direct Graph Fact: {ans.strip()}"
            return f"No active sub-graph memories found matching query: '{query}'."

        return context.strip()

    def _tool_get_entity_details(self, arguments: Dict[str, Any]) -> str:
        """Executes quanta_get_entity_details tool."""
        target = str(arguments.get("name_or_cid", "")).strip()
        if not target:
            return "Error: name_or_cid parameter is required."

        # 1. Look up node
        node = self._find_node(target)
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
            "sample_active_slots": active_slots[:20],  # Sample first 20 for readability
        }

        return json.dumps(details, indent=2)

    def _find_node(self, target: str) -> Optional[QuantaNode]:
        """Finds a QuantaNode by CID, anchor, or literal in PageTable / Interner."""
        pt = self.pipeline.page_table

        # 1. Direct CID fetch
        if hasattr(pt, "fetch_node"):
            node = pt.fetch_node(target)
            if node is not None:
                return node

        # 2. Inter-process interner lookup
        if hasattr(self.pipeline, "compiler") and hasattr(self.pipeline.compiler, "interner"):
            interner = self.pipeline.compiler.interner
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
                # Check literal match
                cur.execute("SELECT cid FROM nodes WHERE literal LIKE ? LIMIT 1", (f"%{target}%",))
                row = cur.fetchone()
                if row:
                    return pt.fetch_node(row[0])
                # Check interned string anchor match
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


def run_mcp_stdio(page_table_path: Optional[Union[str, Path]] = None):
    """Entrypoint function to run the MCP server over stdio."""
    server = MCPServer(page_table_path=page_table_path)
    server.run_stdio()


if __name__ == "__main__":
    run_mcp_stdio()
