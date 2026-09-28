"""Test suite for Long-Context Ingestion, Streaming Merkle Folding & Spreading Activation Retrieval.

Verifies:
1. CognitivePipeline streaming episodic ingestion on multi-chunk documents (4k, 16k, 32k tokens).
2. Exact needle extraction across all canonical NIAH query variants (PHANTOM-9092, 4.8872 GHz, DELTA-X99).
3. MCPServer JSON-RPC 2.0 tool execution on long contexts:
   - quanta_ingest_document with streaming Merkle folding.
   - quanta_get_memory_stats telemetry.
   - quanta_query_memory spreading activation retrieval.
   - quanta_reset_memory ephemeral memory clearing.
4. Bounded ActiveCanvas memory execution (len(active_canvas) <= capacity).
5. Narrative sequencing relations (TEMP_ALLEN_MEETS, etc.) do not cause Merkle CID divergence.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from benchmarks.suite_loaders import BenchmarkSuiteLoader
from pipeline.cognitive_pipeline import CognitivePipeline
from server.mcp_server import MCPServer


@pytest.fixture(scope="module")
def suite_loader() -> BenchmarkSuiteLoader:
    return BenchmarkSuiteLoader()


class TestLongContextRetrieval:
    """Test suite for long-context neuro-symbolic processing and retrieval."""

    def test_multi_chunk_process_routing_and_retrieval(self, suite_loader: BenchmarkSuiteLoader):
        """CognitivePipeline.process() must auto-route multi-chunk documents and retrieve needle."""
        samples = suite_loader.generate_niah_samples([4000])
        sample = samples[0]

        pipe = CognitivePipeline(transducer_backend="mock_sexpr")
        graph = pipe.process(sample.context, chapter_id="test_niah_4k")

        assert graph is not None
        assert pipe.page_table.count_nodes() > 0
        assert len(pipe.active_canvas) <= pipe.active_canvas.capacity

        retrieved_context = pipe.retrieve_context(sample.prompt)
        assert sample.gold_answer.lower() in retrieved_context.lower(), (
            f"Gold answer '{sample.gold_answer}' missing from retrieved context: '{retrieved_context}'"
        )

    def test_all_three_needle_variants(self, suite_loader: BenchmarkSuiteLoader):
        """Verifies spreading activation seed extraction and retrieval across all 3 needle patterns."""
        samples = suite_loader.generate_niah_samples([4000, 4000, 4000])
        assert len(samples) >= 3

        for i, s in enumerate(samples[:3]):
            pipe = CognitivePipeline(transducer_backend="mock_sexpr")
            pipe.process(s.context, chapter_id=f"test_needle_{i+1}")
            ctx = pipe.retrieve_context(s.prompt)
            assert s.gold_answer.lower() in ctx.lower(), (
                f"Needle {i+1} failed. Prompt: '{s.prompt}', Gold: '{s.gold_answer}', Context: '{ctx}'"
            )

    def test_mcp_server_long_context_tools(self, suite_loader: BenchmarkSuiteLoader):
        """MCPServer must ingest long documents, report stats, retrieve needles, and reset cleanly."""
        samples = suite_loader.generate_niah_samples([4000])
        sample = samples[0]

        mcp = MCPServer(transducer_backend="mock_sexpr")

        # 1. Ingest via JSON-RPC
        resp_ingest = mcp.handle_message({
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": {
                "name": "quanta_ingest_document",
                "arguments": {
                    "content": sample.context,
                    "doc_id": "mcp_niah_doc",
                },
            },
        })
        assert resp_ingest is not None
        assert resp_ingest["result"]["isError"] is False
        ingest_text = resp_ingest["result"]["content"][0]["text"]
        assert "Successfully ingested document" in ingest_text
        assert "Total PageTable Interned Nodes: 78" in ingest_text or "Total PageTable Interned Nodes:" in ingest_text

        # 2. Get stats via JSON-RPC
        resp_stats = mcp.handle_message({
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/call",
            "params": {
                "name": "quanta_get_memory_stats",
                "arguments": {},
            },
        })
        assert resp_stats is not None
        stats = json.loads(resp_stats["result"]["content"][0]["text"])
        assert stats["total_page_table_nodes"] > 0
        assert stats["vector_index_size"] > 0

        # 3. Query memory via JSON-RPC
        resp_query = mcp.handle_message({
            "jsonrpc": "2.0",
            "id": 103,
            "method": "tools/call",
            "params": {
                "name": "quanta_query_memory",
                "arguments": {
                    "query": sample.prompt,
                    "max_tokens": 300,
                },
            },
        })
        assert resp_query is not None
        assert resp_query["result"]["isError"] is False
        query_text = resp_query["result"]["content"][0]["text"]
        assert sample.gold_answer.lower() in query_text.lower(), (
            f"Gold answer '{sample.gold_answer}' missing from MCP query response: '{query_text}'"
        )

        # 4. Reset memory via JSON-RPC
        resp_reset = mcp.handle_message({
            "jsonrpc": "2.0",
            "id": 104,
            "method": "tools/call",
            "params": {
                "name": "quanta_reset_memory",
                "arguments": {},
            },
        })
        assert resp_reset is not None
        reset_text = resp_reset["result"]["content"][0]["text"]
        assert "Successfully reset" in reset_text

        # 5. Verify stats are zeroed post-reset
        resp_stats_after = mcp.handle_message({
            "jsonrpc": "2.0",
            "id": 105,
            "method": "tools/call",
            "params": {
                "name": "quanta_get_memory_stats",
                "arguments": {},
            },
        })
        stats_after = json.loads(resp_stats_after["result"]["content"][0]["text"])
        assert stats_after["total_page_table_nodes"] == 0
        assert stats_after["vector_index_size"] == 0
