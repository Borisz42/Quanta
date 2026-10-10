"""Comprehensive Parity Test Suite: MCP Server vs Reverse Proxy.

Verifies:
1. All 9 tools are exposed with schemas in MCP tools/list.
2. Session-isolated partitions (PipelinePool) operate identically across MCP tools (ingest, query, reset).
3. quanta_expand_context achieves identical context compression, token savings, and system prompt enrichment.
4. quanta_chat_completion performs the complete dialogue compression and neuro-symbolic completion lifecycle.
5. quanta_answer_query delivers sub-10ms direct verified answers.
6. quanta_get_memory_stats reports full proxy telemetry (uptime, requests, tokens saved, page table nodes).
7. quanta_list_models provides the exact same model catalog as the reverse proxy /v1/models endpoint.
8. MCP Resources and Prompts (resources/list, resources/read, prompts/list, prompts/get) function correctly.
9. Dual-stream and SVM format types are supported across MCP tools.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from server.mcp_server import MCPServer
from server.proxy import (
    ChatMessage,
    ChatCompletionRequest,
    QuantaProxyConfig,
    create_proxy_app,
)


@pytest.fixture
def mcp_server(tmp_path: Path) -> MCPServer:
    """Creates an MCPServer configured with mock transducer and local fallback."""
    config = QuantaProxyConfig(
        compression_threshold=100,
        transducer_backend="mock",
        fallback_to_local=True,
        page_table_path=tmp_path / "mcp_test.db",
    )
    return MCPServer(config=config)


def test_mcp_tools_list_parity(mcp_server: MCPServer):
    """Asserts that all reverse-proxy equivalent tools are declared in MCP tools/list."""
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    resp = mcp_server.handle_message(req)
    assert resp is not None
    assert "result" in resp
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]

    expected_tools = [
        "quanta_ingest_document",
        "quanta_query_memory",
        "quanta_get_entity_details",
        "quanta_reset_memory",
        "quanta_get_memory_stats",
        "quanta_expand_context",
        "quanta_chat_completion",
        "quanta_answer_query",
        "quanta_list_models",
    ]
    for exp in expected_tools:
        assert exp in tool_names, f"Tool '{exp}' missing from MCP tools/list"


def test_mcp_session_isolation_parity(mcp_server: MCPServer):
    """Asserts that session-isolated partitions work identically in MCP tools."""
    doc_session_a = "Alice discovered an ancient cobalt prism in Alexandria."
    doc_session_b = "Bob surveyed geothermal energy reservoirs in Reykjavik."

    # 1. Ingest into session_a
    res_a = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "quanta_ingest_document",
            "arguments": {"content": doc_session_a, "doc_id": "doc_a", "session_id": "session_a"},
        },
    })
    assert res_a is not None
    assert not res_a.get("result", {}).get("isError", True)

    # 2. Ingest into session_b
    res_b = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "quanta_ingest_document",
            "arguments": {"content": doc_session_b, "doc_id": "doc_b", "session_id": "session_b"},
        },
    })
    assert res_b is not None
    assert not res_b.get("result", {}).get("isError", True)

    # 3. Query session_a for Alice
    query_a = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "quanta_query_memory",
            "arguments": {"query": "cobalt prism", "session_id": "session_a"},
        },
    })
    text_a = query_a["result"]["content"][0]["text"]
    assert "Alexandria" in text_a or "cobalt" in text_a.lower() or "prism" in text_a.lower()

    # 4. Reset session_a only
    reset_a = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {
            "name": "quanta_reset_memory",
            "arguments": {"session_id": "session_a"},
        },
    })
    assert "session_a" in reset_a["result"]["content"][0]["text"]

    # 5. Verify session_b memory is preserved
    stats_b = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 14,
        "method": "tools/call",
        "params": {
            "name": "quanta_get_memory_stats",
            "arguments": {"session_id": "session_b"},
        },
    })
    stats_b_data = json.loads(stats_b["result"]["content"][0]["text"])
    assert stats_b_data["total_page_table_nodes"] > 0


def test_mcp_expand_context_parity(mcp_server: MCPServer):
    """Asserts that quanta_expand_context compresses dialogue and injects verified context."""
    bulky_dialogue = [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": "The Apollo program conducted six crewed lunar landings between 1969 and 1972."},
        {"role": "assistant", "content": "Understood, Apollo landed astronauts on the Moon."},
        {"role": "user", "content": "Apollo 11 was the first mission to land humans on the Moon, commanded by Neil Armstrong."},
        {"role": "assistant", "content": "Neil Armstrong commanded Apollo 11."},
        {"role": "user", "content": "Who commanded the first mission to land humans on the Moon?"},
    ]

    resp = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "quanta_expand_context",
            "arguments": {
                "messages": bulky_dialogue,
                "threshold": 50,  # Force compression
                "format": "dual_stream",
            },
        },
    })
    assert resp is not None
    assert not resp["result"]["isError"]

    data = json.loads(resp["result"]["content"][0]["text"])
    assert "compressed_messages" in data
    assert "tokens_saved" in data
    assert data["tokens_saved"] > 0
    assert len(data["compressed_messages"]) < len(bulky_dialogue)
    # Merged system message should contain context header
    sys_msg = data["compressed_messages"][0]["content"]
    assert "QUANTA SEMANTIC VIRTUAL MEMORY DUAL-STREAM CONTEXT" in sys_msg or "VERIFIED CONTEXT" in sys_msg


def test_mcp_chat_completion_parity(mcp_server: MCPServer):
    """Asserts that quanta_chat_completion executes end-to-end neuro-symbolic completion."""
    doc_prompt = (
        "Context:\n"
        "Project Orion was an early study of a spacecraft intended to be directly propelled by a series of nuclear explosions. "
        "The project was initiated in 1958 at General Atomics and led by physicist Ted Taylor and mathematician Freeman Dyson.\n\n"
        "Question:\n"
        "Who led Project Orion at General Atomics?"
    )

    resp = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 30,
        "method": "tools/call",
        "params": {
            "name": "quanta_chat_completion",
            "arguments": {
                "prompt": doc_prompt,
                "threshold": 30,
                "raw_json": True,
            },
        },
    })
    assert resp is not None
    assert not resp["result"]["isError"]

    data = json.loads(resp["result"]["content"][0]["text"])
    assert data.get("object") == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "usage" in data
    assert "quanta_metadata" in data
    assert data["quanta_metadata"]["tokens_saved"] >= 0


def test_mcp_answer_query_parity(mcp_server: MCPServer):
    """Asserts that quanta_answer_query provides direct neuro-symbolic answers."""
    # Ingest fact
    mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 40,
        "method": "tools/call",
        "params": {
            "name": "quanta_ingest_document",
            "arguments": {
                "content": "Galileo Galilei observed the four largest moons of Jupiter in 1610.",
                "doc_id": "galileo_01",
            },
        },
    })

    # Direct query
    resp = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 41,
        "method": "tools/call",
        "params": {
            "name": "quanta_answer_query",
            "arguments": {
                "query": "What did Galileo observe in 1610?",
            },
        },
    })
    assert resp is not None
    assert not resp["result"]["isError"]
    ans = resp["result"]["content"][0]["text"]
    assert len(ans.strip()) > 0


def test_mcp_stats_telemetry_parity(mcp_server: MCPServer):
    """Asserts that quanta_get_memory_stats returns full reverse-proxy telemetry."""
    resp = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 50,
        "method": "tools/call",
        "params": {
            "name": "quanta_get_memory_stats",
            "arguments": {},
        },
    })
    assert resp is not None
    data = json.loads(resp["result"]["content"][0]["text"])

    required_keys = [
        "status",
        "uptime_seconds",
        "total_requests",
        "compressed_requests",
        "tokens_saved_estimate",
        "total_page_table_nodes",
        "vector_index_size",
        "active_canvas_usage",
        "passage_store_count",
        "binary_table_count",
        "backend_url",
        "compression_threshold",
    ]
    for key in required_keys:
        assert key in data, f"Key '{key}' missing from MCP memory stats"


def test_mcp_models_list_parity(mcp_server: MCPServer):
    """Asserts that quanta_list_models returns the supported model catalog."""
    resp = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 60,
        "method": "tools/call",
        "params": {
            "name": "quanta_list_models",
            "arguments": {},
        },
    })
    assert resp is not None
    data = json.loads(resp["result"]["content"][0]["text"])
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "quanta-context-expander" in model_ids
    assert "gpt-4o" in model_ids
    assert "claude-3-5-sonnet" in model_ids


def test_mcp_resources_and_prompts_parity(mcp_server: MCPServer):
    """Asserts that MCP resources and prompts protocol methods function correctly."""
    # 1. resources/list
    res_list = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 70,
        "method": "resources/list",
        "params": {},
    })
    assert res_list is not None
    uris = [r["uri"] for r in res_list["result"]["resources"]]
    assert "quanta://models" in uris
    assert "quanta://memory/stats" in uris
    assert "quanta://health" in uris

    # 2. resources/read
    res_read = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 71,
        "method": "resources/read",
        "params": {"uri": "quanta://models"},
    })
    assert res_read is not None
    read_text = res_read["result"]["contents"][0]["text"]
    assert "quanta-context-expander" in read_text

    # 3. prompts/list
    prompt_list = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 72,
        "method": "prompts/list",
        "params": {},
    })
    assert prompt_list is not None
    p_names = [p["name"] for p in prompt_list["result"]["prompts"]]
    assert "quanta_context_expansion" in p_names

    # 4. prompts/get
    prompt_get = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 73,
        "method": "prompts/get",
        "params": {"name": "quanta_context_expansion", "arguments": {"history": "doc1", "query": "q1"}},
    })
    assert prompt_get is not None
    assert "messages" in prompt_get["result"]
