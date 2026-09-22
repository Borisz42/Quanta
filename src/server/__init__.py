"""QUANTA Context Expansion Server & LLM Middleware Package (Section 6).

Provides:
- OpenAI-compatible reverse proxy (/v1/chat/completions, /v1/models) with dynamic
  context compression and spreading-activation context injection.
- Model Context Protocol (MCP) JSON-RPC 2.0 stdio server exposing memory ingestion,
  sub-graph retrieval, and 1024-D vector inspection tools.
"""

from __future__ import annotations

from server.proxy import QuantaProxyConfig, create_proxy_app
from server.mcp_server import MCPServer, run_mcp_stdio

__all__ = [
    "QuantaProxyConfig",
    "create_proxy_app",
    "MCPServer",
    "run_mcp_stdio",
]
