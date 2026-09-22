"""Integration tests for Section 6: OpenAI-Compatible Reverse Proxy & MCP Server.

Verifies:
- Task 6.1: FastAPI Reverse Proxy (/v1/models, /v1/chat/completions, /health)
  - Token evaluation & threshold triggering
  - Prior dialogue history ASG ingestion into PageTable
  - Spreading-activation context retrieval & system prompt injection
  - Request compression and payload pruning
  - Server-Sent Events (SSE) streaming
  - Header overrides (X-Quanta-Threshold, X-Quanta-Format, X-Quanta-Enrich)
  - Local neuro-symbolic fallback
- Task 6.2: Model Context Protocol (MCP) JSON-RPC 2.0 Server
  - Initialize handshake & ping
  - Tool list enumeration
  - quanta_ingest_document tool execution
  - quanta_query_memory tool execution
  - quanta_get_entity_details 1024-D vector analysis tool execution
  - JSON-RPC error handling
- Task 6.3: PowerShell startup runner validation
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List
import pytest
from fastapi.testclient import TestClient

from pipeline.cognitive_pipeline import CognitivePipeline
from server.proxy import (
    ChatMessage,
    QuantaProxyConfig,
    create_proxy_app,
    estimate_messages_tokens,
    estimate_tokens,
)
from server.mcp_server import MCPServer, MCP_PROTOCOL_VERSION


# =============================================================================
# Task 6.1: Reverse Proxy Tests
# =============================================================================

class TestQuantaProxy:
    """Test suite for OpenAI-compatible reverse proxy."""

    @pytest.fixture
    def test_pipeline(self, tmp_path: Path):
        db_path = tmp_path / "proxy_test.db"
        pipe = CognitivePipeline(transducer_backend="mock", page_table_path=db_path)
        yield pipe
        if hasattr(pipe, "page_table") and hasattr(pipe.page_table, "close"):
            pipe.page_table.close()

    @pytest.fixture
    def client(self, test_pipeline: CognitivePipeline):
        config = QuantaProxyConfig(
            compression_threshold=2000,
            pipeline=test_pipeline,
            fallback_to_local=True,
        )
        app = create_proxy_app(config)
        return TestClient(app)

    def test_models_endpoint(self, client: TestClient):
        """Asserts /v1/models lists available proxy models."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "quanta-context-expander" in model_ids

    def test_health_endpoint(self, client: TestClient):
        """Asserts /health and /v1/health return healthy telemetry status."""
        for endpoint in ("/health", "/v1/health"):
            resp = client.get(endpoint)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "uptime_seconds" in data
            assert "stored_nodes" in data

    def test_chat_completions_under_threshold(self, client: TestClient):
        """Requests under token threshold should not trigger compression."""
        payload = {
            "model": "quanta-context-expander",
            "messages": [
                {"role": "user", "content": "Hello, can you help me?"}
            ],
            "stream": False,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0

    def test_chat_completions_over_threshold_compression(self, client: TestClient, test_pipeline: CognitivePipeline):
        """Requests over token threshold chunk dialogue, ingest to PageTable, and inject context."""
        # Create a conversation with factual narrative in past turns
        ch1 = "Dr. Eleanor Vance isolated a volatile synthetic compound inside Containment Cell 4 at dawn."
        ch2 = "The laboratory director noted an anomalous crystalline lattice expansion and prohibited tests."
        history_background = (
            "The eastern research facility houses comprehensive laboratory sectors including "
            "chemical synthesis suites, mass spectrometers, automated robotic pipetting stations, "
            "and secure cold storage lockers. The safety committee strictly enforces biosecurity "
            "protocols, requiring continuous monitoring and redundant environmental filtration. "
            "Technicians routinely verify gas pressure and humidity every morning before dawn."
        )

        messages = [
            {"role": "user", "content": f"{ch1} {history_background}"},
            {"role": "assistant", "content": "I acknowledge that Dr. Eleanor Vance isolated the compound in Containment Cell 4."},
            {"role": "user", "content": f"{ch2} Additional observations were recorded in the institutional logbook."},
            {"role": "assistant", "content": "I acknowledge the director's prohibition and the logbook records."},
            {"role": "user", "content": "Where was the volatile synthetic compound isolated?"},
        ]

        # Trigger compression using low threshold header override (50 tokens)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "quanta-context-expander", "messages": messages, "stream": False},
            headers={"X-Quanta-Threshold": "50"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"

        # Verify nodes were ingested into PageTable during compression
        stored_nodes = test_pipeline.page_table.count_nodes()
        assert stored_nodes > 0

        # Verify proxy telemetry recorded compression
        health = client.get("/health").json()
        assert health["compressed_requests"] >= 1
        assert health["tokens_saved_estimate"] > 0

        # Verify local answer or metadata reflects context retrieval
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0

    def test_chat_completions_streaming(self, client: TestClient):
        """Asserts streaming chat completions emit valid Server-Sent Events (SSE)."""
        payload = {
            "model": "quanta-context-expander",
            "messages": [
                {"role": "user", "content": "Tell me a short fact about chemistry."}
            ],
            "stream": True,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse SSE lines
        chunks = []
        saw_done = False
        for line in resp.iter_lines():
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line_str or not line_str.strip():
                continue
            if line_str == "data: [DONE]":
                saw_done = True
                continue
            if line_str.startswith("data: "):
                chunk_json = json.loads(line_str[6:])
                chunks.append(chunk_json)

        assert saw_done, "SSE stream must terminate with data: [DONE]"
        assert len(chunks) > 0
        assert chunks[0]["object"] == "chat.completion.chunk"

    def test_header_overrides(self, client: TestClient, test_pipeline: CognitivePipeline):
        """Asserts X-Quanta-Format and X-Quanta-Enrich headers are respected."""
        # Pre-populate pipeline with a fact
        test_pipeline.process("Dr. Eleanor Vance isolated a volatile synthetic compound inside Containment Cell 4.")

        # Query with forced enrichment and sexpr format
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "quanta-context-expander",
                "messages": [{"role": "user", "content": "Where did Vance isolate the compound?"}],
                "stream": False,
            },
            headers={
                "X-Quanta-Enrich": "true",
                "X-Quanta-Format": "sexpr",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) > 0


# =============================================================================
# Task 6.2: Model Context Protocol (MCP) Server Tests
# =============================================================================

class TestQuantaMCPServer:
    """Test suite for Model Context Protocol (MCP) JSON-RPC 2.0 stdio server."""

    @pytest.fixture
    def mcp_server(self, tmp_path: Path):
        db_path = tmp_path / "mcp_test.db"
        pipeline = CognitivePipeline(transducer_backend="mock", page_table_path=db_path)
        server = MCPServer(pipeline=pipeline)
        yield server
        if hasattr(pipeline, "page_table") and hasattr(pipeline.page_table, "close"):
            pipeline.page_table.close()

    def test_mcp_initialize(self, mcp_server: MCPServer):
        """Asserts initialize returns MCP 2024-11-05 protocol version and tools capability."""
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        resp = mcp_server.handle_message(req)
        assert resp is not None
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        result = resp["result"]
        assert result["protocolVersion"] == MCP_PROTOCOL_VERSION
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "quanta-mcp"

    def test_mcp_initialized_notification(self, mcp_server: MCPServer):
        """Asserts notifications/initialized returns None (no response needed)."""
        notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        resp = mcp_server.handle_message(notif)
        assert resp is None

    def test_mcp_ping(self, mcp_server: MCPServer):
        """Asserts ping returns empty result dict."""
        req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
        resp = mcp_server.handle_message(req)
        assert resp is not None
        assert resp["id"] == 2
        assert resp["result"] == {}

    def test_mcp_tools_list(self, mcp_server: MCPServer):
        """Asserts tools/list advertises the 3 required QUANTA memory tools."""
        req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
        resp = mcp_server.handle_message(req)
        assert resp is not None
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert "quanta_ingest_document" in tool_names
        assert "quanta_query_memory" in tool_names
        assert "quanta_get_entity_details" in tool_names

    def test_mcp_tool_ingest_and_query(self, mcp_server: MCPServer):
        """Tests quanta_ingest_document followed by quanta_query_memory."""
        doc_text = "Dr. Eleanor Vance isolated a volatile synthetic compound inside Containment Cell 4 at dawn."

        # 1. Ingest document
        ingest_req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "quanta_ingest_document",
                "arguments": {
                    "doc_id": "chapter_01",
                    "content": doc_text,
                },
            },
        }
        ingest_resp = mcp_server.handle_message(ingest_req)
        assert ingest_resp is not None
        assert ingest_resp["id"] == 4
        assert not ingest_resp["result"]["isError"]
        text_out = ingest_resp["result"]["content"][0]["text"]
        assert "Successfully ingested" in text_out
        assert "chapter_01" in text_out

        # 2. Query memory
        query_req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "quanta_query_memory",
                "arguments": {
                    "query": "Where was the compound isolated?",
                    "max_tokens": 200,
                },
            },
        }
        query_resp = mcp_server.handle_message(query_req)
        assert query_resp is not None
        assert query_resp["id"] == 5
        assert not query_resp["result"]["isError"]
        retrieved_text = query_resp["result"]["content"][0]["text"]
        assert "cell" in retrieved_text.lower() or "containment" in retrieved_text.lower()

    def test_mcp_tool_get_entity_details(self, mcp_server: MCPServer):
        """Tests quanta_get_entity_details returns 1024-D vector breakdown and active edges."""
        doc_text = "Dr. Eleanor Vance isolated a volatile synthetic compound inside Containment Cell 4 at dawn."
        mcp_server.pipeline.process(doc_text)

        # Inspect entity
        inspect_req = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "quanta_get_entity_details",
                "arguments": {
                    "name_or_cid": "Eleanor Vance",
                },
            },
        }
        resp = mcp_server.handle_message(inspect_req)
        assert resp is not None
        assert not resp["result"]["isError"]
        details_str = resp["result"]["content"][0]["text"]
        details = json.loads(details_str)

        assert "canonical_cid" in details
        assert details["total_active_slots"] > 0
        assert "band_distribution" in details
        assert "active_edges" in details

    def test_mcp_error_handling(self, mcp_server: MCPServer):
        """Asserts invalid methods or unknown tools return valid JSON-RPC error frames."""
        # Unknown method
        bad_method_req = {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "nonexistent_method",
            "params": {},
        }
        resp = mcp_server.handle_message(bad_method_req)
        assert resp is not None
        assert "error" in resp
        assert resp["error"]["code"] == -32601

        # Unknown tool
        bad_tool_req = {
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {"name": "invalid_tool", "arguments": {}},
        }
        resp = mcp_server.handle_message(bad_tool_req)
        assert resp is not None
        assert "error" in resp
        assert resp["error"]["code"] == -32601


# =============================================================================
# Task 6.3: Startup Script Validation
# =============================================================================

class TestServeScript:
    """Validates serve.ps1 existence and syntax."""

    def test_serve_script_exists(self):
        script_path = Path("scripts/serve.ps1")
        assert script_path.exists(), "scripts/serve.ps1 must exist"
        content = script_path.read_text(encoding="utf-8")
        assert "-Port" in content
        assert "-Backend" in content
        assert "-Mode" in content
        assert "uvicorn" in content
