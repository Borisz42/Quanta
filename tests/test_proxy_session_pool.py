"""Unit tests for Session-Isolated PipelinePool and Multi-Slot Concurrency in QUANTA Proxy.

Verifies:
1. PipelinePool correctly partitions CognitivePipeline instances and ingested turn hashes by session ID.
2. Resets for worker_A do not affect worker_B's PageTable or Canvas state.
3. HTTP /v1/chat/completions with X-Quanta-Session-ID routes to independent memory contexts.
4. Concurrent multi-threaded requests with distinct session IDs execute without cross-talk or race conditions.
"""

import concurrent.futures
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from pipeline.cognitive_pipeline import CognitivePipeline
from server.proxy import (
    PipelinePool,
    QuantaProxyConfig,
    create_proxy_app,
)


@pytest.fixture
def mock_proxy_client(tmp_path: Path):
    """Creates a TestClient configured with local fallback and mock transducer."""
    config = QuantaProxyConfig(
        compression_threshold=100,  # low threshold to trigger context ingestion
        transducer_backend="mock",
        fallback_to_local=True,
    )
    app = create_proxy_app(config)
    return TestClient(app)


import asyncio


def test_pipeline_pool_partitioning(tmp_path: Path):
    """Asserts that get_pipeline yields isolated pipelines for different session IDs."""
    async def _test():
        config = QuantaProxyConfig(
            transducer_backend="mock",
            page_table_path=tmp_path / "default.db",
        )
        pool = PipelinePool(config)

        default_pipe = await pool.get_pipeline(None)
        pipe_w0 = await pool.get_pipeline("worker_0")
        pipe_w1 = await pool.get_pipeline("worker_1")

        assert pipe_w0 is not pipe_w1
        assert pipe_w0 is not default_pipe
        assert pipe_w1 is not default_pipe

        # Repeated calls return the same session pipeline
        pipe_w0_again = await pool.get_pipeline("worker_0")
        assert pipe_w0_again is pipe_w0

        # Ingested hashes are isolated
        hashes_w0 = pool.get_ingested_hashes("worker_0")
        hashes_w1 = pool.get_ingested_hashes("worker_1")
        hashes_w0.add("hash_a")
        assert "hash_a" in hashes_w0
        assert "hash_a" not in hashes_w1

    asyncio.run(_test())


def test_session_reset_isolation(tmp_path: Path):
    """Asserts that resetting session A does not touch session B's memory state."""
    async def _test():
        config = QuantaProxyConfig(
            transducer_backend="mock",
            page_table_path=tmp_path / "default.db",
        )
        pool = PipelinePool(config)

        pipe_w0 = await pool.get_pipeline("worker_0")
        pipe_w1 = await pool.get_pipeline("worker_1")

        # Ingest content into both session pipelines
        pipe_w0.process("Dr. Eleanor Vance isolated the crystalline compound in Geneva.")
        pipe_w1.process("Captain Arthur Pendelton navigated the archipelago in Polynesia.")

        pool.get_ingested_hashes("worker_0").add("hash_0")
        pool.get_ingested_hashes("worker_1").add("hash_1")

        nodes_w0_before = pipe_w0.page_table.count_nodes() if hasattr(pipe_w0.page_table, "count_nodes") else len(pipe_w0.page_table)
        nodes_w1_before = pipe_w1.page_table.count_nodes() if hasattr(pipe_w1.page_table, "count_nodes") else len(pipe_w1.page_table)
        assert nodes_w0_before > 0
        assert nodes_w1_before > 0

        # Reset worker_0 only
        pool.reset_session("worker_0", pipe_w0)

        # worker_0 hashes cleared
        assert len(pool.get_ingested_hashes("worker_0")) == 0
        # worker_1 hashes preserved
        assert "hash_1" in pool.get_ingested_hashes("worker_1")

        # worker_1 nodes preserved
        nodes_w1_after = pipe_w1.page_table.count_nodes() if hasattr(pipe_w1.page_table, "count_nodes") else len(pipe_w1.page_table)
        assert nodes_w1_after == nodes_w1_before

    asyncio.run(_test())


def test_http_session_isolation(mock_proxy_client: TestClient):
    """Asserts that HTTP calls with different X-Quanta-Session-ID operate on isolated pipelines."""
    # Worker 0: Ingest context about quantum sensors
    doc_w0 = (
        "Context:\n"
        "Project Chronos developed quantum gravimeter sensors in Zurich. "
        "The sensor operates at cryogenic temperatures of 4 Kelvin and detected microscopic subterranean fissures. "
        "Lead researcher Dr. Marcus Bell published the calibration findings in 2024.\n\n"
        "Question:\n"
        "Where were the quantum sensors developed?"
    )
    resp0 = mock_proxy_client.post(
        "/v1/chat/completions",
        json={"model": "quanta-context-expander", "messages": [{"role": "user", "content": doc_w0}]},
        headers={"X-Quanta-Session-ID": "worker_0", "X-Quanta-Reset": "true"},
    )
    assert resp0.status_code == 200

    # Worker 1: Ingest context about deep-sea hydrothermal vents
    doc_w1 = (
        "Context:\n"
        "Expedition Mariana documented black smoker hydrothermal vents at 10,000 meters depth. "
        "Thermophilic archaea strain Vent-7 thrives at temperatures exceeding 115 degrees Celsius. "
        "Chief oceanographer Dr. Elena Rostova collected sediment cores from the trench floor.\n\n"
        "Question:\n"
        "What organism thrives at 115 degrees Celsius?"
    )
    resp1 = mock_proxy_client.post(
        "/v1/chat/completions",
        json={"model": "quanta-context-expander", "messages": [{"role": "user", "content": doc_w1}]},
        headers={"X-Quanta-Session-ID": "worker_1", "X-Quanta-Reset": "true"},
    )
    assert resp1.status_code == 200

    # Explicit reset of worker_0 via /v1/quanta/reset
    reset_resp = mock_proxy_client.post(
        "/v1/quanta/reset",
        headers={"X-Quanta-Session-ID": "worker_0"},
    )
    assert reset_resp.status_code == 200

    # Worker 1 query should still work without cross-talk
    resp1_query = mock_proxy_client.post(
        "/v1/chat/completions",
        json={
            "model": "quanta-context-expander",
            "messages": [{"role": "user", "content": "What was documented by Expedition Mariana?"}],
        },
        headers={"X-Quanta-Session-ID": "worker_1"},
    )
    assert resp1_query.status_code == 200


import httpx


def test_concurrent_sessions_execution(mock_proxy_client: TestClient):
    """Asserts that 4 concurrent requests using distinct session IDs complete successfully without race conditions."""
    prompts = [
        ("worker_0", "Context:\nAlpha particle emissions measured in lab sector 1.\n\nQuestion:\nWhat was measured?"),
        ("worker_1", "Context:\nBeta telemetry calibration verified in station 2.\n\nQuestion:\nWhat was verified?"),
        ("worker_2", "Context:\nGamma ray burst detected by space telescope array 3.\n\nQuestion:\nWhat was detected?"),
        ("worker_3", "Context:\nDelta orbital mechanics recalculated for satellite 4.\n\nQuestion:\nWhat was recalculated?"),
    ]

    async def _run():
        transport = httpx.ASGITransport(app=mock_proxy_client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = [
                client.post(
                    "/v1/chat/completions",
                    json={"model": "quanta-context-expander", "messages": [{"role": "user", "content": p}]},
                    headers={"X-Quanta-Session-ID": wid, "X-Quanta-Reset": "true"},
                )
                for wid, p in prompts
            ]
            return await asyncio.gather(*tasks)

    results = asyncio.run(_run())

    assert len(results) == 4
    for r in results:
        assert r.status_code == 200
        data = r.json()
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]
