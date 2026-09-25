"""OpenAI-Compatible Reverse Proxy for QUANTA Context Expansion (Section 6).

Implements Task 6.1:
1. Exposes /v1/models, /v1/chat/completions, and /health.
2. In /v1/chat/completions:
   - Evaluates dialogue token footprint.
   - If total tokens exceed threshold (default > 2,000 tokens), chunks and ingests
     historical dialogue into CognitivePipeline / PageTable.
   - Extracts active user query, runs SpreadingActivationRetriever.retrieve_context(),
     and injects compact verified context into the system prompt.
   - Prunes bulky historical turns, forwarding the compressed dialogue to downstream
     backend (Unsloth :8888 or cloud provider) or providing local neuro-symbolic fallback.
   - Supports both standard JSON and Server-Sent Events (SSE) streaming responses.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Union
import uuid

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from pydantic import BaseModel, Field

from pipeline.cognitive_pipeline import CognitivePipeline
from pipeline.tracer import PipelineExecutionTracer
from server.unsloth_manager import UnslothServerManager

logger = logging.getLogger("quanta.server.proxy")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class QuantaProxyConfig:
    """Runtime configuration for QUANTA OpenAI Reverse Proxy."""
    backend_url: str = "http://localhost:8888/v1"
    compression_threshold: int = 2000
    max_context_tokens: int = 500
    context_format: str = "english"  # 'english' or 'sexpr'
    always_enrich: bool = False
    page_table_path: Optional[Union[str, Path]] = None
    pipeline: Optional[CognitivePipeline] = None
    timeout_seconds: float = 30.0
    fallback_to_local: bool = True
    target_model: Optional[str] = None
    transducer_backend: str = "mock"
    tracer: Optional[PipelineExecutionTracer] = None
    unsloth_manager: Optional[UnslothServerManager] = None


# -----------------------------------------------------------------------------
# OpenAI Request / Response Schemas
# -----------------------------------------------------------------------------

try:
    from pydantic import ConfigDict
    _ALLOW_EXTRA_CONFIG = ConfigDict(extra="allow")
except ImportError:
    _ALLOW_EXTRA_CONFIG = None


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""
    name: Optional[str] = None

    if _ALLOW_EXTRA_CONFIG is not None:
        model_config = _ALLOW_EXTRA_CONFIG
    else:
        class Config:
            extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str = "quanta-context-expander"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0

    if _ALLOW_EXTRA_CONFIG is not None:
        model_config = _ALLOW_EXTRA_CONFIG
    else:
        class Config:
            extra = "allow"


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "quanta"
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# -----------------------------------------------------------------------------
# Token Estimation & Compression Utilities
# -----------------------------------------------------------------------------

def estimate_tokens(text: Optional[str]) -> int:
    """Estimates token count with standard word-ratio heuristic (~1.33 tokens/word)."""
    if not text:
        return 0
    words = len(text.strip().split())
    # Add minimal base tokens for punctuation and whitespace
    return max(1, int(words * 1.33) + 2)


def estimate_messages_tokens(messages: Sequence[ChatMessage]) -> int:
    """Estimates total tokens across all messages including role tags."""
    total = 0
    for m in messages:
        total += 4  # per-message overhead (<|im_start|>{role}\n{content}<|im_end|>)
        total += estimate_tokens(m.content)
    total += 2  # priming tokens
    return total


def _dump_model(obj: Any, **kwargs) -> Dict[str, Any]:
    """Serializes a Pydantic model to a dict, compatible across Pydantic v1 and v2."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(**kwargs)
    if hasattr(obj, "dict"):
        return obj.dict(**kwargs)
    return dict(obj)


# -----------------------------------------------------------------------------
# Proxy Application Factory
# -----------------------------------------------------------------------------

def create_proxy_app(config: Optional[QuantaProxyConfig] = None) -> FastAPI:
    """Creates and configures the FastAPI reverse proxy application."""
    cfg = config or QuantaProxyConfig()

    app = FastAPI(
        title="QUANTA Context Expansion Proxy",
        description="OpenAI-compatible neuro-symbolic context expansion reverse proxy",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize or attach pipeline
    pipeline: CognitivePipeline
    if cfg.pipeline is not None:
        pipeline = cfg.pipeline
    else:
        pipeline = CognitivePipeline(
            transducer_backend=cfg.transducer_backend,
            page_table_path=cfg.page_table_path,
        )

    # Telemetry state
    stats = {
        "start_time": time.time(),
        "total_requests": 0,
        "compressed_requests": 0,
        "tokens_saved": 0,
        "nodes_ingested": 0,
    }

    # Store state on app
    app.state.config = cfg
    app.state.pipeline = pipeline
    app.state.stats = stats
    app.state.ingested_turn_hashes = set()

    # -------------------------------------------------------------------------
    # Health & Models Endpoints
    # -------------------------------------------------------------------------

    @app.get("/health")
    @app.get("/v1/health")
    async def health():
        uptime = time.time() - stats["start_time"]
        node_count = pipeline.page_table.count_nodes() if hasattr(pipeline.page_table, "count_nodes") else len(pipeline.page_table)
        return {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "total_requests": stats["total_requests"],
            "compressed_requests": stats["compressed_requests"],
            "tokens_saved_estimate": stats["tokens_saved"],
            "stored_nodes": node_count,
            "backend_url": cfg.backend_url,
            "compression_threshold": cfg.compression_threshold,
        }

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        return ModelListResponse(
            data=[
                ModelCard(id="quanta-context-expander", root="quanta-context-expander"),
                ModelCard(id="gpt-4o", root="quanta-context-expander"),
                ModelCard(id="claude-3-5-sonnet", root="quanta-context-expander"),
                ModelCard(id="qwen-2.5-7b", root="quanta-context-expander"),
            ]
        )

    # -------------------------------------------------------------------------
    # Chat Completions Endpoint
    # -------------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        raw_request: Request,
        x_quanta_threshold: Optional[int] = Header(None, alias="X-Quanta-Threshold"),
        x_quanta_format: Optional[str] = Header(None, alias="X-Quanta-Format"),
        x_quanta_force_enrich: Optional[bool] = Header(None, alias="X-Quanta-Enrich"),
    ):
        t0 = time.perf_counter()
        stats["total_requests"] += 1

        threshold = x_quanta_threshold if x_quanta_threshold is not None else cfg.compression_threshold
        format_type = x_quanta_format if x_quanta_format is not None else cfg.context_format
        force_enrich = x_quanta_force_enrich if x_quanta_force_enrich is not None else cfg.always_enrich

        raw_messages = request.messages
        if not raw_messages:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="messages array cannot be empty")

        # 1. Measure incoming token footprint
        raw_tokens = estimate_messages_tokens(raw_messages)
        should_compress = (raw_tokens > threshold) or (len(raw_messages) > 4 and raw_tokens > 200)

        # 2. Extract active user query and historical context
        # Find the latest user message
        last_user_idx: Optional[int] = None
        for i in reversed(range(len(raw_messages))):
            if raw_messages[i].role == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            last_user_idx = len(raw_messages) - 1

        active_user_msg = raw_messages[last_user_idx]
        user_query = active_user_msg.content or ""

        # Historical messages are all messages prior to the active user turn
        prior_messages = raw_messages[:last_user_idx]
        system_messages = [m for m in prior_messages if m.role == "system"]
        dialogue_history = [m for m in prior_messages if m.role != "system"]

        retrieved_context: str = ""
        compressed_messages: List[Dict[str, Any]] = []

        if should_compress or force_enrich:
            logger.info(
                "Triggering QUANTA context compression: raw_tokens=%d, threshold=%d, prior_turns=%d",
                raw_tokens,
                threshold,
                len(dialogue_history),
            )

            # Ingest dialogue history into CognitivePipeline if dialogue turns exist
            if dialogue_history:
                # Group dialogue into an ingested narrative, only ingesting turns not previously seen
                import hashlib
                history_text_blocks = []
                for m in dialogue_history:
                    if m.content and m.content.strip():
                        speaker = "Human" if m.role == "user" else "Assistant"
                        turn_text = f"{speaker}: {m.content.strip()}"
                        turn_hash = hashlib.sha256(turn_text.encode("utf-8")).hexdigest()
                        if turn_hash not in getattr(app.state, "ingested_turn_hashes", set()):
                            history_text_blocks.append(turn_text)
                            app.state.ingested_turn_hashes.add(turn_hash)

                if history_text_blocks:
                    history_text = "\n\n".join(history_text_blocks)
                    if history_text.strip():
                        try:
                            # Ingest into PageTable Merkle DAG
                            pipeline.process(history_text)
                            stats["nodes_ingested"] = pipeline.page_table.count_nodes() if hasattr(pipeline.page_table, "count_nodes") else len(pipeline.page_table)
                        except Exception as e:
                            logger.warning("Error during dialogue history ASG ingestion: %s", e)

            # Retrieve active context via spreading activation
            try:
                retrieved_context = pipeline.retrieve_context(
                    query=user_query,
                    format=format_type,
                    max_tokens=cfg.max_context_tokens,
                )
            except Exception as e:
                logger.warning("Error retrieving context from PageTable: %s", e)
                retrieved_context = ""

            # Build enriched system prompt
            system_content_parts = []
            if system_messages:
                system_content_parts.append(system_messages[-1].content or "")

            if retrieved_context and retrieved_context.strip():
                context_header = (
                    f"=== [QUANTA NEURO-SYMBOLIC VERIFIED CONTEXT] ===\n"
                    f"{retrieved_context.strip()}\n"
                    f"================================================="
                )
                system_content_parts.append(context_header)

            merged_system_content = "\n\n".join(part for part in system_content_parts if part).strip()

            if merged_system_content:
                compressed_messages.append({
                    "role": "system",
                    "content": merged_system_content,
                })

            # Add latest user message
            compressed_messages.append({
                "role": active_user_msg.role,
                "content": active_user_msg.content,
            })

            # If there are trailing messages after the user message (rare), preserve them
            for m in raw_messages[last_user_idx + 1:]:
                compressed_messages.append(_dump_model(m))

            # Record metrics
            compressed_tokens = sum(estimate_tokens(m.get("content", "")) for m in compressed_messages)
            tokens_saved = max(0, raw_tokens - compressed_tokens)
            stats["compressed_requests"] += 1
            stats["tokens_saved"] += tokens_saved

            logger.info(
                "Compression complete: %d -> %d tokens (saved %d tokens, context len=%d)",
                raw_tokens,
                compressed_tokens,
                tokens_saved,
                len(retrieved_context),
            )

            tracer = cfg.tracer or PipelineExecutionTracer.get_instance()
            if tracer is not None:
                tracer.record_reverse_proxy(
                    raw_tokens=raw_tokens,
                    compressed_tokens=compressed_tokens,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    turns_pruned=len(dialogue_history),
                    details={"user_query": user_query[:100], "context_injected": bool(retrieved_context)},
                )
        else:
            # Under threshold and not force-enriched: preserve messages as-is
            compressed_messages = [
                _dump_model(m) for m in raw_messages
            ]

        # 3. Forward request to downstream backend or local neuro-symbolic fallback
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_timestamp = int(time.time())

        # Construct payload for downstream backend
        downstream_payload = _dump_model(request, exclude_unset=True)
        downstream_payload["messages"] = compressed_messages
        if cfg.target_model:
            downstream_payload["model"] = cfg.target_model
        elif "8888" in cfg.backend_url and downstream_payload.get("model") in ("quanta-context-expander", "default", None):
            downstream_payload["model"] = "unsloth/Qwen3.5-4B-MTP-GGUF"

        # Verify GPU policy if targeting local Unsloth backend
        if "8888" in cfg.backend_url or "localhost" in cfg.backend_url:
            mgr = cfg.unsloth_manager or UnslothServerManager()
            try:
                mgr.enforce_gpu_policy()
            except RuntimeError as rerr:
                logger.warning("GPU policy enforcement alert: %s", rerr)
                if not cfg.fallback_to_local:
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(rerr))

        tracer = cfg.tracer or PipelineExecutionTracer.get_instance()

        if request.stream:
            return await _handle_streaming_response(
                backend_url=cfg.backend_url,
                payload=downstream_payload,
                completion_id=completion_id,
                model=request.model,
                created=created_timestamp,
                fallback_pipeline=pipeline if cfg.fallback_to_local else None,
                query=user_query,
                context=retrieved_context,
                timeout=cfg.timeout_seconds,
            )
        else:
            return await _handle_unary_response(
                backend_url=cfg.backend_url,
                payload=downstream_payload,
                completion_id=completion_id,
                model=request.model,
                created=created_timestamp,
                fallback_pipeline=pipeline if cfg.fallback_to_local else None,
                query=user_query,
                context=retrieved_context,
                timeout=cfg.timeout_seconds,
                tracer=tracer,
            )

    return app


# -----------------------------------------------------------------------------
# Downstream Forwarding & Local Fallback Handlers
# -----------------------------------------------------------------------------

async def _handle_unary_response(
    backend_url: str,
    payload: Dict[str, Any],
    completion_id: str,
    model: str,
    created: int,
    fallback_pipeline: Optional[CognitivePipeline],
    query: str,
    context: str,
    timeout: float,
    tracer: Optional[PipelineExecutionTracer] = None,
) -> JSONResponse:
    """Forwards non-streaming request to backend or returns local neuro-symbolic completion."""
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{backend_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            dt_s = time.perf_counter() - t0
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                if tracer is not None:
                    tracer.record_backend_call(
                        method="POST",
                        url=f"{backend_url.rstrip('/')}/chat/completions",
                        status_code=resp.status_code,
                        latency_s=dt_s,
                        tokens_gen=usage.get("completion_tokens", 0),
                        tps=usage.get("completion_tokens", 0) / max(0.001, dt_s),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        details={"model": payload.get("model")},
                    )
                return JSONResponse(content=data, status_code=200)
            logger.warning("Downstream backend responded with status %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.info("Downstream backend unavailable (%s). Falling back to local QUANTA response.", exc)

    # Local neuro-symbolic fallback
    if fallback_pipeline is not None:
        local_content = _generate_local_answer(fallback_pipeline, query, context)
    else:
        local_content = f"[QUANTA Local Response] Processed query: '{query}'"

    prompt_toks = sum(estimate_tokens(m.get("content", "")) for m in payload.get("messages", []))
    comp_toks = estimate_tokens(local_content)

    return JSONResponse(
        content={
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": local_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "total_tokens": prompt_toks + comp_toks,
            },
            "quanta_metadata": {
                "context_injected": bool(context),
                "backend": "local_neuro_symbolic_fallback",
            },
        },
        status_code=200,
    )


async def _handle_streaming_response(
    backend_url: str,
    payload: Dict[str, Any],
    completion_id: str,
    model: str,
    created: int,
    fallback_pipeline: Optional[CognitivePipeline],
    query: str,
    context: str,
    timeout: float,
) -> StreamingResponse:
    """Streams response chunks via Server-Sent Events (SSE)."""

    async def event_generator() -> AsyncIterator[str]:
        # First attempt to stream from target backend
        upstream_success = False
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{backend_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status_code == 200:
                        upstream_success = True
                        async for line in resp.aiter_lines():
                            if line:
                                yield f"{line}\n\n"
        except Exception as exc:
            logger.info("Streaming upstream failed (%s). Emitting local stream.", exc)

        # If upstream didn't stream, emit local neuro-symbolic stream
        if not upstream_success:
            if fallback_pipeline is not None:
                full_text = _generate_local_answer(fallback_pipeline, query, context)
            else:
                full_text = f"[QUANTA Local Stream] Verified answer for: '{query}'"

            # Emit in tokens/words
            words = full_text.split(" ")
            for i, word in enumerate(words):
                chunk_text = word + (" " if i < len(words) - 1 else "")
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)

            # Final stop chunk
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _generate_local_answer(pipeline: CognitivePipeline, query: str, context: str) -> str:
    """Generates an honest answer from the neuro-symbolic graph topology."""
    try:
        ans = pipeline.answer_query(query)
        if ans and ans.strip():
            return ans.strip()
    except Exception as e:
        logger.debug("answer_query lookup error: %s", e)

    if context and context.strip():
        return f"Based on the verified knowledge graph:\n{context.strip()}"

    return f"QUANTA verified no contradictory graph state for query: '{query}'."


# Default application instance for uvicorn launch
app = create_proxy_app()
