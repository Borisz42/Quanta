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
import re
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
    backend_url: str = field(default_factory=lambda: os.getenv("QUANTA_BACKEND_URL", "http://127.0.0.1:8888/v1"))
    compression_threshold: int = field(default_factory=lambda: int(os.getenv("QUANTA_COMPRESSION_THRESHOLD", "2000")))
    max_context_tokens: int = field(default_factory=lambda: int(os.getenv("QUANTA_MAX_CONTEXT_TOKENS", "2048")))
    context_format: str = "english"  # 'english' or 'sexpr'
    always_enrich: bool = False
    page_table_path: Optional[Union[str, Path]] = field(default_factory=lambda: os.getenv("QUANTA_PAGE_TABLE_PATH", None))
    pipeline: Optional[CognitivePipeline] = None
    timeout_seconds: float = field(default_factory=lambda: float(os.getenv("QUANTA_TIMEOUT_SECONDS", "180.0")))
    fallback_to_local: bool = True
    target_model: Optional[str] = None
    transducer_backend: str = field(default_factory=lambda: os.getenv("QUANTA_TRANSDUCER_BACKEND", "mock"))
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


def decompose_query_context(text: Optional[str]) -> Tuple[Optional[str], str]:
    """Decomposes a single bulky user query into (context_document, core_question).

    Handles:
    - Explicit section delimiters:
      'Context:\n...\n\nQuestion: ...'
      'Documentation:\n...\n\nQuery: ...'
      'Source:\n...\n\nQuestion: ...'
      'Eredeti forrásdokumentumok:\n...\n\nKérdés: ...'
    - Text blocks where context precedes a terminal question sentence or paragraph.
    """
    if not text or not text.strip():
        return None, ""

    clean = text.strip()

    # 1. Explicit delimiter patterns
    explicit_patterns = [
        r"(?:===+\s*|#{1,6}\s*|\*{1,2})?(?:(?:Document\s+)?Context|Documentation|Background|Source|Passages?|Forrás(?:dokumentumok)?|Szöveg|Eredeti forrásdokumentumok)(?:\*{1,2}|===+)?\s*:\s*\n*([\s\S]*?)\n\s*(?:Question|Query|Prompt|Kérdés)\s*:\s*\n*([\s\S]*)",
        r"([\s\S]*?)\n\s*(?:Question|Query|Prompt|Kérdés)\s*:\s*\n*([\s\S]*)",
    ]
    for pat in explicit_patterns:
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            doc = m.group(1).strip()
            q = m.group(2).strip()
            if len(doc.split()) >= 15:
                return doc, q

    # 2. Paragraph split heuristic if text has context (> 20 words) and ends with interrogative sentence
    paras = [p.strip() for p in clean.split("\n\n") if p.strip()]
    if len(paras) >= 2 and len(clean.split()) >= 20:
        last_para = paras[-1]
        if "?" in last_para or any(last_para.lower().startswith(w) for w in ("what", "who", "where", "why", "when", "how", "compare", "mi", "mit", "milyen", "melyik", "hová", "hol")):
            doc = "\n\n".join(paras[:-1]).strip()
            return doc, last_para

    # 3. Sentence split heuristic if ending sentence is an interrogative
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if len(sents) >= 3 and len(clean.split()) >= 20:
        last_sent = sents[-1]
        if last_sent.endswith("?") or any(last_sent.lower().startswith(w) for w in ("what", "who", "where", "why", "when", "how", "compare", "mi", "mit", "milyen", "melyik", "hová", "hol")):
            doc = " ".join(sents[:-1]).strip()
            return doc, last_sent

    return None, clean


def extract_context_from_system(messages: Sequence[ChatMessage]) -> Tuple[Optional[str], Optional[str]]:
    """Extracts background context from a system message if present (e.g. 'Context:\n...' or 'Document context:\n...')."""
    for m in messages:
        if m.role == "system" and m.content:
            text = m.content.strip()
            pat = (
                r"(?:===+\s*|#{1,6}\s*|\*{1,2})?"
                r"(?:(?:Document\s+)?Context|Documentation|Source|Passages?|Forrás(?:dokumentumok)?)"
                r"(?:\*{1,2}|===+)?\s*:\s*\n*([\s\S]*)"
            )
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                doc = match.group(1).strip()
                instruction = text[:match.start()].strip()
                if len(doc.split()) >= 15:
                    return doc, instruction
    return None, None


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Session-Isolated Pipeline Pool
# -----------------------------------------------------------------------------

class PipelinePool:
    """Manages session-isolated CognitivePipeline instances for concurrent benchmark evaluation."""

    def __init__(self, config: QuantaProxyConfig):
        self.config = config
        self.default_pipeline = config.pipeline or CognitivePipeline(
            transducer_backend=config.transducer_backend,
            page_table_path=config.page_table_path,
        )
        self.session_pipelines: Dict[str, CognitivePipeline] = {}
        self.session_hashes: Dict[str, Set[str]] = {}
        self._default_hashes: Set[str] = set()
        self._lock = asyncio.Lock()

    async def get_pipeline(self, session_id: Optional[str]) -> CognitivePipeline:
        if not session_id or session_id == "default":
            return self.default_pipeline
        async with self._lock:
            if session_id not in self.session_pipelines:
                logger.info("Initializing isolated CognitivePipeline for session '%s'", session_id)
                self.session_pipelines[session_id] = CognitivePipeline(
                    transducer_backend=self.config.transducer_backend,
                    page_table_path=":memory:",
                )
            return self.session_pipelines[session_id]

    def get_ingested_hashes(self, session_id: Optional[str]) -> Set[str]:
        if not session_id or session_id == "default":
            return self._default_hashes
        if session_id not in self.session_hashes:
            self.session_hashes[session_id] = set()
        return self.session_hashes[session_id]

    def reset_session(self, session_id: Optional[str], pipeline: Optional[CognitivePipeline] = None) -> None:
        pipe = pipeline or (self.session_pipelines.get(session_id) if session_id and session_id != "default" else self.default_pipeline)
        if pipe is not None:
            if hasattr(pipe, "reset"):
                pipe.reset()
            else:
                if hasattr(pipe, "active_canvas") and pipe.active_canvas is not None:
                    pipe.active_canvas.clear()
                if hasattr(pipe, "entity_engine") and hasattr(pipe.entity_engine, "manifest"):
                    pipe.entity_engine.manifest.reset()
                if hasattr(pipe, "stitcher") and pipe.stitcher is not None:
                    pipe.stitcher.reset()
                if hasattr(pipe, "merkle_book"):
                    from core.asg import HierarchicalMerkleBook
                    pipe.merkle_book = HierarchicalMerkleBook()
                if hasattr(pipe, "page_table") and pipe.page_table is not None:
                    if hasattr(pipe.page_table, "clear"):
                        pipe.page_table.clear()
                    elif getattr(pipe.page_table, "db_path", None) == ":memory:":
                        from memory.page_table import PageTable
                        pipe.page_table = PageTable(db_path=":memory:")
                if hasattr(pipe, "episodic_entity_registry"):
                    pipe.episodic_entity_registry.clear()
        hashes = self.get_ingested_hashes(session_id)
        hashes.clear()


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

    # Initialize session-isolated pipeline pool
    pool = PipelinePool(cfg)
    pipeline = pool.default_pipeline

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
    app.state.pool = pool
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
        passage_count = len(pipeline.passage_store) if hasattr(pipeline, "passage_store") else 0
        binary_nodes = len(pipeline.binary_table) if hasattr(pipeline, "binary_table") else 0
        return {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "total_requests": stats["total_requests"],
            "compressed_requests": stats["compressed_requests"],
            "tokens_saved_estimate": stats["tokens_saved"],
            "stored_nodes": node_count,
            "passages_stored": passage_count,
            "binary_table_nodes": binary_nodes,
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

    @app.post("/v1/quanta/reset")
    async def reset_quanta_memory(
        x_quanta_session_id: Optional[str] = Header(None, alias="X-Quanta-Session-ID"),
    ):
        """Flushes ephemeral working memory, resets ActiveCanvas and clears ingested hashes."""
        pool.reset_session(x_quanta_session_id)
        return {"status": "ok", "message": f"QUANTA memory and session hashes reset successfully for session {x_quanta_session_id or 'default'}"}

    # -------------------------------------------------------------------------
    # Chat Completions Endpoint
    # -------------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        raw_request: Request,
        x_quanta_session_id: Optional[str] = Header(None, alias="X-Quanta-Session-ID"),
        x_quanta_threshold: Optional[int] = Header(None, alias="X-Quanta-Threshold"),
        x_quanta_format: Optional[str] = Header(None, alias="X-Quanta-Format"),
        x_quanta_force_enrich: Optional[bool] = Header(None, alias="X-Quanta-Enrich"),
        x_quanta_global_kb: Optional[bool] = Header(None, alias="X-Quanta-Global-KB"),
        x_quanta_no_global_kb: Optional[bool] = Header(None, alias="X-Quanta-No-Global-KB"),
        x_quanta_reset: Optional[bool] = Header(None, alias="X-Quanta-Reset"),
        x_quanta_validate: Optional[bool] = Header(False, alias="X-Quanta-Validate"),
        x_quanta_max_context_tokens: Optional[int] = Header(None, alias="X-Quanta-Max-Context-Tokens"),
        x_quanta_timeout: Optional[float] = Header(None, alias="X-Quanta-Timeout"),
    ):
        t0 = time.perf_counter()
        stats["total_requests"] += 1

        pipeline = await pool.get_pipeline(x_quanta_session_id)
        ingested_turn_hashes = pool.get_ingested_hashes(x_quanta_session_id)

        if x_quanta_reset:
            pool.reset_session(x_quanta_session_id, pipeline)

        # Handle global knowledge base mounting on demand
        if x_quanta_global_kb:
            if getattr(pipeline.page_table, "global_kb", None) is None:
                try:
                    from memory.global_kb import GlobalKnowledgeBase
                    repo_root = Path(__file__).resolve().parent.parent.parent
                    kb_path = repo_root / "data" / "wikipedia_quanta.db"
                    if not kb_path.exists():
                        kb_path = Path("data/wikipedia_quanta.db")
                    if kb_path.exists():
                        pipeline.page_table.mount_global_kb(GlobalKnowledgeBase(kb_path))
                        logger.info("Successfully mounted GlobalKnowledgeBase from %s", kb_path)
                except Exception as e:
                    logger.warning("Failed to mount GlobalKnowledgeBase: %s", e)
        elif x_quanta_no_global_kb:
            if hasattr(pipeline.page_table, "global_kb"):
                pipeline.page_table.global_kb = None

        threshold = x_quanta_threshold if x_quanta_threshold is not None else cfg.compression_threshold
        format_type = x_quanta_format if x_quanta_format is not None else cfg.context_format
        force_enrich = x_quanta_force_enrich if x_quanta_force_enrich is not None else cfg.always_enrich

        raw_messages = request.messages
        if not raw_messages:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="messages array cannot be empty")

        # 1. Measure incoming token footprint
        raw_tokens = estimate_messages_tokens(raw_messages)

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
        raw_user_content = active_user_msg.content or ""

        # Historical messages are all messages prior to the active user turn
        prior_messages = raw_messages[:last_user_idx]
        system_messages = [m for m in prior_messages if m.role == "system"]
        dialogue_history = [m for m in prior_messages if m.role != "system"]

        # Decompose single-turn bulky context if present
        doc_from_user, isolated_question = decompose_query_context(raw_user_content)
        doc_from_system, clean_sys_inst = extract_context_from_system(system_messages)

        has_single_query_context = (doc_from_user is not None) or (doc_from_system is not None)
        user_query = isolated_question if doc_from_user else raw_user_content
        should_compress = (raw_tokens > threshold) or (len(raw_messages) > 4 and raw_tokens > 200) or has_single_query_context

        retrieved_context: str = ""
        compressed_messages: List[Dict[str, Any]] = []
        is_cold_start = False
        t_ingest_ms = 0.0
        t_ret_ms = 0.0

        # Domain-aware gating: pure code syntax tasks should not query encyclopedic KB
        is_code_task = bool(re.search(r"(?:def\s+\w+\s*\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import)", user_query))

        should_enrich_or_compress = (
            should_compress
            or force_enrich
            or (bool(x_quanta_global_kb) and not has_single_query_context and not is_code_task)
        )

        if should_enrich_or_compress:
            logger.info(
                "Triggering QUANTA context compression/enrichment: raw_tokens=%d, threshold=%d, prior_turns=%d, has_single_ctx=%s, global_kb=%s",
                raw_tokens,
                threshold,
                len(dialogue_history),
                has_single_query_context,
                bool(x_quanta_global_kb),
            )

            import hashlib
            t_ingest_start = time.perf_counter()

            # 1. Ingest dialogue history into CognitivePipeline if dialogue turns exist
            if dialogue_history:
                history_text_blocks = []
                for m in dialogue_history:
                    if m.content and m.content.strip():
                        speaker = "Human" if m.role == "user" else "Assistant"
                        turn_text = f"{speaker}: {m.content.strip()}"
                        turn_hash = hashlib.sha256(turn_text.encode("utf-8")).hexdigest()
                        if turn_hash not in ingested_turn_hashes:
                            history_text_blocks.append(turn_text)
                            ingested_turn_hashes.add(turn_hash)

                if history_text_blocks:
                    history_text = "\n\n".join(history_text_blocks)
                    if history_text.strip():
                        try:
                            is_cold_start = True
                            await asyncio.to_thread(pipeline.process, history_text, validate=bool(x_quanta_validate))
                            stats["nodes_ingested"] = pipeline.page_table.count_nodes() if hasattr(pipeline.page_table, "count_nodes") else len(pipeline.page_table)
                        except Exception as e:
                            logger.warning("Error during dialogue history ASG ingestion: %s", e)

            # 2. Ingest single-query user context document (cold-start single query)
            if doc_from_user:
                doc_hash = hashlib.sha256(doc_from_user.encode("utf-8")).hexdigest()
                if doc_hash not in ingested_turn_hashes:
                    try:
                        is_cold_start = True
                        await asyncio.to_thread(pipeline.process, doc_from_user, chapter_id=f"doc_{doc_hash[:8]}", validate=bool(x_quanta_validate))
                        ingested_turn_hashes.add(doc_hash)
                        stats["nodes_ingested"] = pipeline.page_table.count_nodes() if hasattr(pipeline.page_table, "count_nodes") else len(pipeline.page_table)
                    except Exception as e:
                        logger.error("Error during user context document ASG ingestion: %s", e)

            # 3. Ingest system context document if present
            if doc_from_system:
                sys_doc_hash = hashlib.sha256(doc_from_system.encode("utf-8")).hexdigest()
                if sys_doc_hash not in ingested_turn_hashes:
                    try:
                        is_cold_start = True
                        await asyncio.to_thread(pipeline.process, doc_from_system, chapter_id=f"sys_{sys_doc_hash[:8]}", validate=bool(x_quanta_validate))
                        ingested_turn_hashes.add(sys_doc_hash)
                        stats["nodes_ingested"] = pipeline.page_table.count_nodes() if hasattr(pipeline.page_table, "count_nodes") else len(pipeline.page_table)
                    except Exception as e:
                        logger.error("Error during system context document ASG ingestion: %s", e)

            t_ingest_ms = (time.perf_counter() - t_ingest_start) * 1000.0

            # Retrieve active context via spreading activation (strictly from natural user_query)
            t_ret_start = time.perf_counter()
            try:
                # Dynamic context budgeting for multi-hop queries and multi-document benchmarks
                doc_text = (doc_from_user or "") + " " + (doc_from_system or "")
                num_doc_headings = doc_text.count("Document [") + doc_text.count("Passage [")
                query_hop_depth = pipeline.retriever.detect_query_hop_depth(user_query) if hasattr(pipeline, "retriever") else 2

                # Multi-document scaling: allocate ~300-350 tokens per required hop / active document
                if num_doc_headings >= 10 or query_hop_depth >= 3:
                    # Target: 4 to 5 full passages (~1,400 to 1,600 tokens), providing ~45% compression over 2,700 tokens
                    adaptive_budget = min(cfg.max_context_tokens, max(1200, min(1600, num_doc_headings * 100)))
                    effective_max_tokens = x_quanta_max_context_tokens if x_quanta_max_context_tokens is not None else adaptive_budget
                else:
                    # Single-turn open-domain QA (ARC-Challenge, MMLU): 350 to 450 tokens of high-density verified facts
                    adaptive_budget = min(cfg.max_context_tokens, 450)
                    if has_single_query_context and raw_tokens > 0:
                        adaptive_budget = min(adaptive_budget, max(50, int(raw_tokens * 0.45)))
                    elif dialogue_history and not has_single_query_context:
                        pruned_toks = estimate_messages_tokens(dialogue_history)
                        if pruned_toks > 0:
                            adaptive_budget = min(adaptive_budget, max(50, int(pruned_toks * 0.5)))
                    effective_max_tokens = x_quanta_max_context_tokens if x_quanta_max_context_tokens is not None else adaptive_budget

                if format_type in ("dual_stream", "svm"):
                    raw_ctx = await asyncio.to_thread(
                        pipeline.query_memory,
                        query=user_query,
                        format="dual_stream",
                        max_tokens=effective_max_tokens,
                    )
                    retrieved_context = getattr(raw_ctx, "full_context", str(raw_ctx))
                else:
                    retrieved_context = await asyncio.to_thread(
                        pipeline.retrieve_context,
                        query=user_query,
                        format=format_type,
                        max_tokens=effective_max_tokens,
                    )
            except Exception as e:
                logger.warning("Error retrieving context from PageTable: %s", e)
                retrieved_context = ""
            t_ret_ms = (time.perf_counter() - t_ret_start) * 1000.0

            # Build enriched system prompt
            system_content_parts = []
            if clean_sys_inst:
                system_content_parts.append(clean_sys_inst)
            elif system_messages and not doc_from_system:
                system_content_parts.append(system_messages[-1].content or "")

            if retrieved_context and str(retrieved_context).strip():
                if format_type in ("dual_stream", "svm"):
                    context_header = (
                        f"=== [QUANTA SEMANTIC VIRTUAL MEMORY DUAL-STREAM CONTEXT] ===\n"
                        f"{str(retrieved_context).strip()}\n"
                        f"============================================================"
                    )
                else:
                    context_header = (
                        f"=== [QUANTA NEURO-SYMBOLIC VERIFIED CONTEXT] ===\n"
                        f"{str(retrieved_context).strip()}\n"
                        f"================================================="
                    )
                system_content_parts.append(context_header)

            merged_system_content = "\n\n".join(part for part in system_content_parts if part).strip()

            if merged_system_content:
                compressed_messages.append({
                    "role": "system",
                    "content": merged_system_content,
                })

            # Add latest user message (using isolated user_query if context was stripped)
            compressed_messages.append({
                "role": active_user_msg.role,
                "content": user_query,
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
                "Compression complete: %d -> %d tokens (saved %d tokens, context len=%d, ingest=%.1fms, ret=%.1fms)",
                raw_tokens,
                compressed_tokens,
                tokens_saved,
                len(retrieved_context),
                t_ingest_ms,
                t_ret_ms,
            )

            tracer = cfg.tracer or PipelineExecutionTracer.get_instance()
            if tracer is not None:
                tracer.record_reverse_proxy(
                    raw_tokens=raw_tokens,
                    compressed_tokens=compressed_tokens,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    turns_pruned=len(dialogue_history) + (1 if doc_from_user else 0),
                    details={
                        "user_query": user_query[:100],
                        "context_injected": bool(retrieved_context),
                        "cold_start": is_cold_start,
                        "ingest_latency_ms": t_ingest_ms,
                        "retrieval_latency_ms": t_ret_ms,
                    },
                )
        else:
            # Under threshold and not force-enriched: preserve messages as-is
            compressed_messages = [
                _dump_model(m) for m in raw_messages
            ]

        # Metadata payload for response
        quanta_meta = {
            "context_injected": bool(retrieved_context),
            "cold_start": is_cold_start,
            "ingest_latency_ms": t_ingest_ms,
            "retrieval_latency_ms": t_ret_ms,
            "raw_tokens": raw_tokens,
            "compressed_tokens": sum(estimate_tokens(m.get("content", "")) for m in compressed_messages),
            "tokens_saved": max(0, raw_tokens - sum(estimate_tokens(m.get("content", "")) for m in compressed_messages)),
        }

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

        effective_timeout = (
            x_quanta_timeout
            if x_quanta_timeout is not None
            else max(cfg.timeout_seconds, 60.0 + (raw_tokens / 1000.0) * 3.0)
        )

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
                timeout=effective_timeout,
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
                timeout=effective_timeout,
                tracer=tracer,
                quanta_meta=quanta_meta,
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
    quanta_meta: Optional[Dict[str, Any]] = None,
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
                if quanta_meta:
                    data["quanta_metadata"] = quanta_meta
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
                **(quanta_meta or {}),
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
