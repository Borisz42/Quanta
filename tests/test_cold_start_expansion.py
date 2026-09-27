"""Tests for QUANTA Cold-Start Single-Query Context Expansion.

Verifies:
1. decompose_query_context on single bulky prompts (English and Hungarian).
2. CognitivePipeline.process_and_answer_single_query starting from zero with allow_fixtures=False.
3. Proxy /v1/chat/completions dynamic decomposition and in-flight ingestion.
4. PipelineExecutionTracer cold-start recording and break-even calculations.
"""

import pytest
import time
from pathlib import Path
from fastapi.testclient import TestClient

from core.asg import QuantaGraph
from memory.page_table import VirtualPageTable
from pipeline.cognitive_pipeline import CognitivePipeline
from pipeline.tracer import PipelineExecutionTracer
from server.proxy import (
    ChatMessage,
    QuantaProxyConfig,
    create_proxy_app,
    decompose_query_context,
    estimate_tokens,
)


def test_decompose_query_context_explicit_english():
    text = (
        "Context:\n"
        "The James Webb Space Telescope operates at Sun-Earth L2 Lagrange point. "
        "The primary mirror consists of eighteen hexagonal beryllium segments coated with vapor-deposited gold. "
        "Astronomers observed exoplanet WASP-96b with NIRCam.\n\n"
        "Question:\n"
        "What did Near-Infrared Camera observe on exoplanet WASP-96b?"
    )
    doc, query = decompose_query_context(text)
    assert doc is not None
    assert "James Webb Space Telescope" in doc
    assert query == "What did Near-Infrared Camera observe on exoplanet WASP-96b?"


def test_decompose_query_context_hungarian():
    text = (
        "Eredeti forrásdokumentumok:\n"
        "Dr. Kovács János vegyészmérnök fluoropolimer mátrixot szintetizált a budapesti központi laboratóriumban. "
        "A szintetizált polimert a szegedi lézeres kutatóközpontba szállították, ahol száz gigawattos impulzuslézerrel vizsgálták meg.\n\n"
        "Kérdés:\n"
        "Hová szállították a polimert és milyen lézerrel vizsgálták meg?"
    )
    doc, query = decompose_query_context(text)
    assert doc is not None
    assert "Dr. Kovács János" in doc
    assert "Hová szállították" in query


def test_decompose_query_context_paragraph_split():
    text = (
        "The enterprise order processing microservice utilizes Spring Boot and Kafka event streaming. "
        "When customer Benjamin placed checkout for Order 1043 with invalid token tok_declined, "
        "the OrderFulfillmentService received CardDeclinedException from payment gateway. "
        "The service immediately rolled back inventory reservations and marked Order 1043 as status CANCELLED.\n\n"
        "Why was Order 1043 cancelled by the OrderFulfillmentService?"
    )
    doc, query = decompose_query_context(text)
    assert doc is not None
    assert "Order 1043" in doc
    assert "Why was Order 1043 cancelled" in query


def test_pipeline_process_and_answer_single_query(tmp_path):
    db_file = tmp_path / "test_cold_start.db"
    pipeline = CognitivePipeline(
        transducer_backend="mock",
        page_table_path=db_file,
        canvas_capacity=128,
    )

    combined_text = (
        "Context:\n"
        "At 10:15 AM, the OrderFulfillmentService initiated payment authorization for Order 1042 with token tok_visa_4242. "
        "The external payment gateway authorized the monetary charge and returned transaction reference txn_9941. "
        "The service updated Order 1042 to status FULFILLED.\n\n"
        "Question:\n"
        "What was the authorization transaction reference for Order 1042?"
    )

    # Execute cold start from 0 with allow_fixtures=False
    res = pipeline.process_and_answer_single_query(
        query_text=combined_text,
        cold_start=True,
        allow_fixtures=False,
    )

    assert res["cold_start"] is True
    assert res["allow_fixtures"] is False
    assert res["ingest_latency_ms"] > 0
    assert res["retrieval_latency_ms"] > 0
    assert "txn_9941" in res["retrieved_context"] or "Order 1042" in res["retrieved_context"]

    # Warm start follow-up query on existing memory with zero hints
    t0 = time.perf_counter()
    warm_ctx = pipeline.retrieve_context(
        "What is the status of Order 1042?",
        format="english",
        max_tokens=100,
    )
    t_warm_ms = (time.perf_counter() - t0) * 1000.0
    assert t_warm_ms < 50.0
    assert "Order 1042" in warm_ctx or "FULFILLED" in warm_ctx

    pipeline.close()


def test_proxy_single_turn_cold_start_decompression(tmp_path):
    db_file = tmp_path / "proxy_cold_test.db"
    pipeline = CognitivePipeline(
        transducer_backend="mock",
        page_table_path=db_file,
        canvas_capacity=128,
    )

    cfg = QuantaProxyConfig(
        compression_threshold=100,
        max_context_tokens=300,
        pipeline=pipeline,
        fallback_to_local=True,
    )
    app = create_proxy_app(cfg)
    client = TestClient(app)

    long_context = " ".join([
        "The James Webb Space Telescope carries four scientific instruments including NIRCam and MIRI.",
        "MIRI is cooled to six Kelvin using a closed-cycle helium loop cryocooler.",
        "Astronomers observed galaxy GLASS-z12 and exoplanet WASP-96b.",
    ] * 10)

    combined_prompt = f"Context:\n{long_context}\n\nQuestion:\nWhat cooling system is used on MIRI?"
    raw_tokens = estimate_tokens(combined_prompt)

    payload = {
        "model": "quanta-context-expander",
        "messages": [{"role": "user", "content": combined_prompt}],
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "quanta_metadata" in data
    q_meta = data["quanta_metadata"]
    assert q_meta["cold_start"] is True
    assert q_meta["ingest_latency_ms"] >= 0
    assert data["usage"]["prompt_tokens"] < raw_tokens

    pipeline.close()


def test_tracer_cold_start_recording(tmp_path):
    tracer = PipelineExecutionTracer.get_instance()
    tracer.reset()

    tracer.record_cold_start_eval(
        task="JWST Exoplanet Cold Start",
        query="What did NIRCam observe on WASP-96b?",
        baseline_prompt_tokens=1800,
        quanta_prompt_tokens=150,
        baseline_latency_s=0.65,
        ingest_latency_s=0.15,
        retrieval_latency_s=0.002,
        quanta_llm_latency_s=0.20,
        total_cold_quanta_latency_s=0.352,
        warm_start_latency_s=0.202,
        baseline_answer="NIRCam observed water vapor absorption on WASP-96b.",
        quanta_cold_answer="Water vapor detected on WASP-96b.",
        quanta_warm_answer="MIRI cooled by cryocooler.",
        factual_token="water vapor",
        is_baseline_correct=True,
        is_cold_correct=True,
        is_warm_correct=True,
        break_even_queries=0.15 / (0.65 - 0.202),
    )

    assert len(tracer.cold_start_results) == 1
    csr = tracer.cold_start_results[0]
    assert csr["token_savings_pct"] > 90.0
    assert csr["break_even_queries"] < 1.0

    out_md = tmp_path / "test_trace.md"
    md_content = tracer.export_markdown(out_md)
    assert "Cold-Start vs. Warm-Start Single-Query Evaluation" in md_content
    assert "Multi-Query Amortization Curve" in md_content
    assert "JWST Exoplanet Cold Start" in md_content
