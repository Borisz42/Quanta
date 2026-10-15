"""Auxiliary Neural Vector Coprocessor (ANVC) Embedding Manager for QUANTA.

Provides dedicated serving, client management, task-prefix formatting, and
Dynamic Matryoshka Representation Learning (MRL) for EmbeddingGemma 2.

Supported Backends:
1. Dedicated llama-server (:8889): Non-blocking HTTP bi-encoder endpoint.
2. In-Process (sentence_transformers / transformers): Local GPU/CPU PyTorch execution.
3. MockEmbeddingManager: Deterministic, pseudo-semantic offline backend for CI.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import httpx

from config.multi_scale_config import MultiScaleConfig

logger = logging.getLogger("quanta.server.embedding_manager")

DEFAULT_EMBEDDING_HOST = "127.0.0.1"
DEFAULT_EMBEDDING_PORT = 8889
DEFAULT_MODEL_ID = "google/embeddinggemma-2"
DEFAULT_MATRYOSHKA_DIM = 768
DEFAULT_BATCH_SIZE = 16
NATIVE_DIMENSION = 768

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# -----------------------------------------------------------------------------
# Task Prefix Formatters
# -----------------------------------------------------------------------------

def format_query_prefix(query: str, task: str = "question answering") -> str:
    """Formats an asymmetric query prefix for EmbeddingGemma 2."""
    clean_task = task.strip() if task else "question answering"
    clean_query = query.strip()
    return f"task: {clean_task} | query: {clean_query}"


def format_document_prefix(text: str, title: Optional[str] = None) -> str:
    """Formats an asymmetric document chunk prefix for EmbeddingGemma 2."""
    clean_title = (title.strip() if title and title.strip() else "none")
    clean_text = text.strip()
    return f"title: {clean_title} | text: {clean_text}"


def format_code_prefix(code_or_query: str) -> str:
    """Formats a code retrieval prefix for EmbeddingGemma 2."""
    clean_content = code_or_query.strip()
    return f"task: code retrieval | query: {clean_content}"


# -----------------------------------------------------------------------------
# Dynamic Matryoshka Truncation & L2 Normalization
# -----------------------------------------------------------------------------

def truncate_and_normalize(vectors: np.ndarray, matryoshka_dim: int) -> np.ndarray:
    """Truncates raw dense embeddings to dimension D and applies L2 normalization.

    Given raw native 768-D vectors v in R^(N x 768) (or R^768):
    1. Truncate dynamically: v_D = v[..., :matryoshka_dim]
    2. Normalize: v_hat_D = v_D / ||v_D||_2
    """
    matryoshka_dim = int(matryoshka_dim)
    if matryoshka_dim <= 0:
        raise ValueError(f"matryoshka_dim must be positive, got {matryoshka_dim}")

    is_1d = vectors.ndim == 1
    if is_1d:
        v = vectors[np.newaxis, :]
    else:
        v = vectors

    orig_dim = v.shape[-1]
    dim = min(matryoshka_dim, orig_dim)
    truncated = v[:, :dim].astype(np.float32, copy=True)

    # Compute L2 norms along last axis
    norms = np.linalg.norm(truncated, axis=-1, keepdims=True)
    # Avoid division by zero
    nonzero = norms > 1e-12
    normalized = np.zeros_like(truncated)
    np.divide(truncated, norms, out=normalized, where=nonzero)

    # If any norm was 0, fill with uniform unit vector
    zero_rows = ~nonzero.squeeze(axis=-1)
    if np.any(zero_rows):
        unit_val = 1.0 / np.sqrt(dim)
        normalized[zero_rows] = unit_val

    if is_1d:
        return normalized[0]
    return normalized


# -----------------------------------------------------------------------------
# Backends
# -----------------------------------------------------------------------------

class BaseEmbeddingBackend:
    """Abstract interface for embedding generation backends."""

    def embed_raw(self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


class MockEmbeddingBackend(BaseEmbeddingBackend):
    """Deterministic, offline pseudo-embedding generator for testing and CI.

    Produces high-fidelity synthetic vectors where:
    - Same text yields identical vectors.
    - Texts with shared tokens exhibit correlated similarity.
    - Disjoint texts exhibit near-orthogonal vectors.
    """

    def __init__(self, native_dim: int = NATIVE_DIMENSION):
        self.native_dim = native_dim

    def _hash_vector(self, token: str) -> np.ndarray:
        """Generates a pseudo-random unit vector deterministically from token string."""
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(h)
        vec = rng.randn(self.native_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 1e-12 else 1.0)

    def embed_raw(self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        results: List[np.ndarray] = []
        for text in texts:
            clean = text.strip()
            # Whole text hash for global semantic seed
            whole_vec = self._hash_vector(clean)

            # Token-level hash for lexical / bag-of-words similarity correlation
            import re
            words = [w.lower() for w in re.findall(r"[a-zA-Z0-9_]+", clean) if len(w) > 2]
            if words:
                token_sum = np.zeros(self.native_dim, dtype=np.float32)
                for w in words[:40]:  # limit for speed
                    token_sum += self._hash_vector(w)
                token_norm = np.linalg.norm(token_sum)
                if token_norm > 1e-12:
                    token_sum /= token_norm
                # Blend global text identity (20%) with token overlap (80%)
                combined = 0.2 * whole_vec + 0.8 * token_sum
            else:
                combined = whole_vec

            results.append(combined)

        return np.array(results, dtype=np.float32)

    def is_available(self) -> bool:
        return True


class LlamaServerEmbeddingBackend(BaseEmbeddingBackend):
    """HTTP client communicating with dedicated llama-server (:8889)."""

    def __init__(
        self,
        host: str = DEFAULT_EMBEDDING_HOST,
        port: int = DEFAULT_EMBEDDING_PORT,
        model_id: str = DEFAULT_MODEL_ID,
        timeout: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.model_id = model_id
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=0.25) as client:
                # 1. Probe /v1/embeddings directly
                try:
                    r = client.post(
                        f"{self.base_url}/v1/embeddings",
                        json={"input": ["probe"], "model": self.model_id},
                    )
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                # 2. Probe native llama.cpp /embedding
                try:
                    r2 = client.post(f"{self.base_url}/embedding", json={"content": "probe"})
                    if r2.status_code == 200:
                        return True
                except Exception:
                    pass
                # 3. Dedicated embedding server /health
                try:
                    r3 = client.get(f"{self.base_url}/health")
                    if r3.status_code == 200:
                        # Only accept if port is dedicated embedding port
                        if self.port == DEFAULT_EMBEDDING_PORT:
                            return True
                except Exception:
                    pass
        except Exception:
            pass
        return False

    def embed_raw(self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        all_embeddings: List[List[float]] = []

        with httpx.Client(timeout=self.timeout) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                # Try OpenAI format POST /v1/embeddings
                payload = {
                    "input": batch,
                    "model": self.model_id,
                }
                resp = None
                try:
                    resp = client.post(f"{self.base_url}/v1/embeddings", json=payload)
                except Exception as exc:
                    logger.warning("Error hitting /v1/embeddings: %s", exc)

                if resp is not None and resp.status_code == 200:
                    data = resp.json()
                    # Expect {"data": [{"embedding": [...]}, ...]}
                    items = data.get("data", [])
                    # Sort items by index if provided
                    sorted_items = sorted(items, key=lambda x: x.get("index", 0)) if len(items) > 1 else items
                    for item in sorted_items:
                        all_embeddings.append(item["embedding"])
                else:
                    # Fallback: llama.cpp native POST /embedding
                    for item_text in batch:
                        r_native = client.post(f"{self.base_url}/embedding", json={"content": item_text})
                        if r_native.status_code == 200:
                            emb = r_native.json().get("embedding", [])
                            all_embeddings.append(emb)
                        else:
                            raise RuntimeError(
                                f"Llama-server embedding endpoint failed with HTTP {r_native.status_code}: {r_native.text}"
                            )

        return np.array(all_embeddings, dtype=np.float32)


class InProcessEmbeddingBackend(BaseEmbeddingBackend):
    """In-process execution using sentence_transformers or Hugging Face transformers.

    Implements a strict process-level model cache (_shared_models) and synchronized inference
    lock (_inference_lock) to guarantee EmbeddingGemma 2 is loaded at most once in memory/VRAM
    and prevent concurrent activation memory spikes on CUDA.
    """

    _shared_models: Dict[Tuple[str, str], Tuple[Any, str, str]] = {}
    _model_lock: threading.Lock = threading.Lock()
    _inference_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
        self._backend_type = None
        self._init_model()

    def _init_model(self):
        target_device = self.device
        if target_device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.info("CUDA requested but not available; falling back to CPU for %s", self.model_id)
                    target_device = "cpu"
            except ImportError:
                target_device = "cpu"

        key = (self.model_id, target_device)

        with self._model_lock:
            # Check if this model or any variant is ALREADY loaded in this process
            for (m_id, dev), (mod, b_type, actual_dev) in self._shared_models.items():
                if mod is not None and (
                    m_id == self.model_id
                    or (m_id.endswith("embeddinggemma-2") and self.model_id.endswith("embeddinggemma-2"))
                    or (m_id.split("/")[-1] == self.model_id.split("/")[-1])
                ):
                    self._model = mod
                    self._backend_type = b_type
                    self.device = actual_dev
                    logger.info(
                        "EmbeddingGemma model already loaded; reusing shared %s instance for '%s' on device '%s'",
                        self._backend_type,
                        self.model_id,
                        self.device,
                    )
                    return

            cpu_key = (self.model_id, "cpu")

            # Prefer sentence_transformers if available
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                actual_device = target_device
                if actual_device == "cuda":
                    torch.cuda.empty_cache()

                logger.info("Loading in-process SentenceTransformer model: %s on %s", self.model_id, actual_device)
                try:
                    model_kwargs = {}
                    if actual_device == "cuda":
                        model_kwargs["torch_dtype"] = torch.float16
                    loaded_model = SentenceTransformer(
                        self.model_id,
                        device=actual_device,
                        model_kwargs=model_kwargs if model_kwargs else None,
                    )
                except Exception as load_err:
                    if actual_device == "cuda" and ("out of memory" in str(load_err).lower() or "cuda" in str(load_err).lower() or "allocation" in str(load_err).lower()):
                        logger.warning("Failed to load SentenceTransformer on CUDA (%s); falling back to CPU", load_err)
                        torch.cuda.empty_cache()
                        actual_device = "cpu"
                        if cpu_key in self._shared_models:
                            self._model, self._backend_type, self.device = self._shared_models[cpu_key]
                            return
                        loaded_model = SentenceTransformer(self.model_id, device="cpu")
                    else:
                        raise

                self._model = loaded_model
                self.device = actual_device
                self._backend_type = "sentence_transformers"
                self._shared_models[(self.model_id, actual_device)] = (self._model, self._backend_type, actual_device)
                logger.info("Successfully registered shared SentenceTransformer model for %s on %s", self.model_id, actual_device)
                return
            except Exception as st_err:
                logger.debug("Could not initialize SentenceTransformer (%s), trying transformers...", st_err)

            # Fallback to Hugging Face transformers
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                actual_device = target_device
                if actual_device == "cuda":
                    torch.cuda.empty_cache()

                logger.info("Loading in-process HF AutoModel: %s on %s", self.model_id, actual_device)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                try:
                    loaded_model = AutoModel.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if actual_device == "cuda" else torch.float32,
                    ).to(actual_device)
                except Exception as hf_load_err:
                    if actual_device == "cuda":
                        logger.warning("Failed to load HF model on CUDA (%s); falling back to CPU", hf_load_err)
                        torch.cuda.empty_cache()
                        actual_device = "cpu"
                        if cpu_key in self._shared_models:
                            self._model, self._backend_type, self.device = self._shared_models[cpu_key]
                            return
                        loaded_model = AutoModel.from_pretrained(self.model_id).to("cpu")
                    else:
                        raise
                self._model = loaded_model
                self.device = actual_device
                self._model.eval()
                self._backend_type = "transformers"
                self._shared_models[(self.model_id, actual_device)] = (self._model, self._backend_type, actual_device)
                logger.info("Successfully registered shared HF model for %s on %s", self.model_id, actual_device)
                return
            except Exception as hf_err:
                logger.info("In-process embedding model '%s' unavailable (%s).", self.model_id, hf_err)
                raise

    def is_available(self) -> bool:
        return self._model is not None

    def embed_raw(self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        with self._inference_lock:
            if self._backend_type == "sentence_transformers":
                # Do NOT normalize here; Matryoshka truncation requires raw vectors before normalization!
                embs = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                )
                return np.array(embs, dtype=np.float32)

            elif self._backend_type == "transformers":
                import torch

                all_embs: List[np.ndarray] = []
                device = next(self._model.parameters()).device
                with torch.no_grad():
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i : i + batch_size]
                        inputs = self._tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                        outputs = self._model(**inputs)
                        # Mean pooling over attention mask
                        token_embeddings = outputs.last_hidden_state
                        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                        mean_pooled = (sum_embeddings / sum_mask).cpu().float().numpy()
                        all_embs.append(mean_pooled)

                return np.vstack(all_embs)

        raise RuntimeError("InProcessEmbeddingBackend has no loaded model.")


# -----------------------------------------------------------------------------
# Embedding Manager (ANVC Primary Orchestrator)
# -----------------------------------------------------------------------------

class EmbeddingManager:
    """Primary lifecycle and inference manager for EmbeddingGemma 2 (ANVC).

    Handles:
    - Task prefix formatting for asymmetric retrieval.
    - Dynamic Matryoshka truncation D in [64..768] with L2 normalization.
    - Multi-backend dispatch (llama-server :8889, in-process PyTorch, or mock).
    - Subprocess lifecycle management for standalone llama-server.
    """

    _default_instance: Optional[EmbeddingManager] = None
    _instance_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_default(
        cls,
        allow_mock: bool = False,
        config: Optional[MultiScaleConfig] = None,
        backend: Optional[str] = None,
        device: Optional[str] = None,
    ) -> EmbeddingManager:
        """Obtain or initialize the global singleton EmbeddingManager instance in a thread-safe manner."""
        with cls._instance_lock:
            if cls._default_instance is None:
                cls._default_instance = cls(
                    config=config,
                    backend=backend,
                    device=device,
                    allow_mock=allow_mock,
                )
            return cls._default_instance

    def __init__(
        self,
        config: Optional[MultiScaleConfig] = None,
        backend: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        model_id: Optional[str] = None,
        matryoshka_dim: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        allow_mock: bool = False,
    ):
        self.config = config or MultiScaleConfig()

        self.backend_choice = (backend or self.config.embedding_backend or "auto").lower()
        self.host = host or self.config.get("embedding.host", DEFAULT_EMBEDDING_HOST)
        self.port = port if port is not None else self.config.embedding_port
        self.model_id = model_id or self.config.embedding_model_id
        self.matryoshka_dim = (
            matryoshka_dim if matryoshka_dim is not None else self.config.embedding_matryoshka_dim
        )
        self.device = device or self.config.embedding_device
        self.batch_size = batch_size if batch_size is not None else self.config.embedding_batch_size
        self.allow_mock = allow_mock

        self._backend: Optional[BaseEmbeddingBackend] = None
        self._server_process: Optional[subprocess.Popen] = None
        self._active_backend_name: str = "uninitialized"
        self._query_cache: Dict[Tuple[str, str, int], np.ndarray] = {}

        self._init_backend()

    def _init_backend(self):
        """Resolves and instantiates the chosen backend with strict live guard."""
        choice = self.backend_choice

        if choice == "mock":
            self._backend = MockEmbeddingBackend(native_dim=NATIVE_DIMENSION)
            self._active_backend_name = "mock"
            return

        if choice == "llama-server":
            self._backend = LlamaServerEmbeddingBackend(
                host=self.host,
                port=self.port,
                model_id=self.model_id,
            )
            self._active_backend_name = "llama-server"
            return

        if choice in ("sentence_transformers", "transformers", "in_process", "in-process"):
            try:
                self._backend = InProcessEmbeddingBackend(
                    model_id=self.model_id,
                    device=self.device,
                )
                self._active_backend_name = self._backend._backend_type or "in-process"
                return
            except Exception as e:
                if self.allow_mock:
                    logger.warning("Failed to initialize in-process backend: %s. Falling back to mock (allow_mock=True).", e)
                    self._backend = MockEmbeddingBackend(native_dim=NATIVE_DIMENSION)
                    self._active_backend_name = "mock"
                    return
                raise RuntimeError(
                    f"ANVC Strict Live Backend Failure: In-process EmbeddingGemma 2 ('{self.model_id}') "
                    f"failed to initialize on device '{self.device}': {e}. Mock fallback is disallowed."
                )

        # Auto dispatch: Probe llama-server (:8889), then proxy (:8000), then in-process, then fail if not allowed
        if choice == "auto":
            # 1. Probe if dedicated llama-server is already alive on self.port (:8889)
            probe = LlamaServerEmbeddingBackend(
                host=self.host,
                port=self.port,
                model_id=self.model_id,
            )
            if probe.is_available():
                self._backend = probe
                self._active_backend_name = "llama-server"
                logger.info("Auto-dispatch connected to live dedicated llama-server on %s:%d", self.host, self.port)
                return

            # 2. Probe if QUANTA Reverse Proxy is alive serving /v1/embeddings
            proxy_url_env = os.environ.get("QUANTA_PROXY_URL", "")
            target_host = self.host
            proxy_port = int(os.environ.get("QUANTA_PROXY_PORT", "8000"))
            if proxy_url_env:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(proxy_url_env if "://" in proxy_url_env else f"http://{proxy_url_env}")
                    if parsed.hostname:
                        target_host = parsed.hostname
                    if parsed.port:
                        proxy_port = parsed.port
                except Exception:
                    pass

            if proxy_port != self.port:
                proxy_probe = LlamaServerEmbeddingBackend(
                    host=target_host,
                    port=proxy_port,
                    model_id=self.model_id,
                )
                if proxy_probe.is_available():
                    self._backend = proxy_probe
                    self._active_backend_name = "quanta-proxy-embeddings"
                    logger.info("Auto-dispatch connected to live QUANTA proxy embedding service on %s:%d", target_host, proxy_port)
                    return

            # 3. Try in-process live EmbeddingGemma 2
            last_err = None
            try:
                self._backend = InProcessEmbeddingBackend(
                    model_id=self.model_id,
                    device=self.device,
                )
                self._active_backend_name = self._backend._backend_type or "in-process"
                logger.info("Auto-dispatch initialized live in-process embedding engine (%s on %s)", self._active_backend_name, self.device)
                return
            except Exception as exc:
                last_err = exc
                logger.warning("In-process EmbeddingGemma 2 initialization error: %s", exc)

            # 3. Fallback to mock ONLY if explicitly permitted
            if self.allow_mock:
                self._backend = MockEmbeddingBackend(native_dim=NATIVE_DIMENSION)
                self._active_backend_name = "mock"
                logger.warning("Auto-dispatch fell back to MockEmbeddingBackend (allow_mock=True).")
                return

            raise RuntimeError(
                f"ANVC Strict Live Backend Failure: Dedicated llama-server on port {self.port} is offline, "
                f"and in-process EmbeddingGemma 2 ('{self.model_id}') failed to load on device '{self.device}': {last_err}. "
                "ANVC strictly mandates a real, live EmbeddingGemma 2 backend. Mock fallback is disabled."
            )

    @property
    def active_backend_name(self) -> str:
        """Returns the active backend descriptor string."""
        return self._active_backend_name

    @property
    def is_mock(self) -> bool:
        """Returns True if the active backend is a mock pseudo-generator."""
        return self._active_backend_name == "mock"

    @property
    def is_live(self) -> bool:
        """Returns True if a real, live EmbeddingGemma 2 model is responsive."""
        return not self.is_mock and self.is_service_responsive()

    def assert_live_backend(self):
        """Strict hardware and model guard enforcing live EmbeddingGemma 2 execution."""
        if self.is_mock:
            raise RuntimeError(
                "ANVC Live Backend Guard Violation: MockEmbeddingBackend is currently active! "
                "ANVC strictly mandates live execution with 'google/embeddinggemma-2' on CPU/GPU or llama-server :8889."
            )
        if not self.is_service_responsive():
            raise RuntimeError(
                f"ANVC Live Backend Guard Violation: Embedding backend '{self._active_backend_name}' is not responsive!"
            )

    # -------------------------------------------------------------------------
    # Inference Methods
    # -------------------------------------------------------------------------

    def embed_raw(
        self,
        texts: Union[str, List[str]],
        matryoshka_dim: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embeds raw texts without prefix injection, truncated to matryoshka_dim."""
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)
        if not text_list:
            target_dim = matryoshka_dim or self.matryoshka_dim
            return np.empty((0, target_dim), dtype=np.float32)

        bs = batch_size or self.batch_size
        target_dim = matryoshka_dim or self.matryoshka_dim

        raw_vectors = self._backend.embed_raw(text_list, batch_size=bs)
        normalized = truncate_and_normalize(raw_vectors, target_dim)

        if is_single:
            return normalized[0]
        return normalized

    def embed_query(
        self,
        query: str,
        task: str = "question answering",
        matryoshka_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Embeds a user query with task prefix formatting.

        Returns 1D array of shape (D,) with L2-normalized values.
        """
        dim = matryoshka_dim or self.matryoshka_dim
        cache_key = (query, task, dim)
        if hasattr(self, "_query_cache") and cache_key in self._query_cache:
            return self._query_cache[cache_key].copy()

        formatted = format_query_prefix(query, task=task)
        vec = self.embed_raw(formatted, matryoshka_dim=dim)
        if hasattr(self, "_query_cache") and len(self._query_cache) < 4096:
            self._query_cache[cache_key] = vec
        return vec

    def clear_cache(self) -> None:
        """Clears in-memory query embedding cache."""
        if hasattr(self, "_query_cache"):
            self._query_cache.clear()

    def embed_document(
        self,
        text: str,
        title: Optional[str] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Embeds a single document chunk with title prefix formatting.

        Returns 1D array of shape (D,) with L2-normalized values.
        """
        formatted = format_document_prefix(text, title=title)
        dim = matryoshka_dim or self.matryoshka_dim
        return self.embed_raw(formatted, matryoshka_dim=dim)

    def embed_documents(
        self,
        texts: List[str],
        titles: Optional[List[Optional[str]]] = None,
        matryoshka_dim: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embeds a list of document chunks with prefix formatting.

        Returns 2D array of shape (N, D) with L2-normalized rows.
        """
        if not texts:
            dim = matryoshka_dim or self.matryoshka_dim
            return np.empty((0, dim), dtype=np.float32)

        if titles is None:
            formatted = [format_document_prefix(t, title=None) for t in texts]
        else:
            formatted = [
                format_document_prefix(t, title=titles[i] if i < len(titles) else None)
                for i, t in enumerate(texts)
            ]

        dim = matryoshka_dim or self.matryoshka_dim
        bs = batch_size or self.batch_size
        return self.embed_raw(formatted, matryoshka_dim=dim, batch_size=bs)

    def embed_code(
        self,
        code_or_query: str,
        matryoshka_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Embeds a code snippet or code retrieval query with code prefix.

        Returns 1D array of shape (D,) with L2-normalized values.
        """
        formatted = format_code_prefix(code_or_query)
        dim = matryoshka_dim or self.matryoshka_dim
        return self.embed_raw(formatted, matryoshka_dim=dim)

    # -------------------------------------------------------------------------
    # Similarity Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine similarity between two L2-normalized vectors."""
        v1 = np.asarray(vec1, dtype=np.float32)
        v2 = np.asarray(vec2, dtype=np.float32)
        return float(np.dot(v1, v2))

    @staticmethod
    def batch_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Computes cosine similarities between a query vector and a matrix of doc vectors."""
        q = np.asarray(query_vec, dtype=np.float32)
        docs = np.asarray(doc_vecs, dtype=np.float32)
        if docs.ndim == 1:
            return np.array([float(np.dot(docs, q))], dtype=np.float32)
        return np.dot(docs, q)

    # -------------------------------------------------------------------------
    # Server Lifecycle Management (llama-server)
    # -------------------------------------------------------------------------

    def is_service_responsive(self, timeout: float = 2.0) -> bool:
        """Checks if the embedding service is currently responsive."""
        if isinstance(self._backend, LlamaServerEmbeddingBackend):
            return self._backend.is_available()
        return self._backend is not None and self._backend.is_available()

    def ensure_service_running(self, timeout: float = 30.0) -> bool:
        """Ensures the dedicated embedding llama-server is online, spawning if necessary."""
        if self.is_service_responsive():
            return True

        # Re-probe auto-dispatch in case proxy or service just became available
        try:
            self._init_backend()
            if self.is_service_responsive():
                return True
        except Exception:
            pass

        # Find executable candidate for llama-server
        candidates = [
            Path(r"C:\Users\PC\.docker\bin\inference\llama-server.exe"),
            Path(r"C:\Users\PC\.unsloth\llama.cpp\build\bin\Release\llama-server.exe"),
            Path(shutil.which("llama-server.exe")) if shutil.which("llama-server.exe") else None,
            Path(shutil.which("llama-server")) if shutil.which("llama-server") else None,
            Path(os.path.expanduser("~")) / ".lmstudio" / "extensions" / "backends" / "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.54.0" / "llama-server.exe",
            Path(os.path.expanduser("~")) / ".unsloth" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
            Path("bin/llama-server.exe"),
            Path("llama.cpp/llama-server.exe"),
        ]

        # Model candidates
        model_candidates = [
            Path(f"models/{self.model_id}.gguf"),
            Path("models/embeddinggemma-270m.gguf"),
            Path("models/embeddinggemma-270m-q8_0.gguf"),
            Path("models/embeddinggemma-2-270m-q8_0.gguf"),
        ]
        target_model_file = None
        for mc in model_candidates:
            if mc and mc.exists() and mc.is_file():
                target_model_file = mc
                break

        server_bin = None
        for cand in candidates:
            if cand and str(cand) not in ("", ".") and cand.is_file():
                server_bin = cand
                break

        if not server_bin or not target_model_file:
            logger.debug(
                "Cannot spawn dedicated llama-server: bin=%s, model=%s.",
                server_bin,
                target_model_file,
            )
            if self.is_service_responsive():
                return True
            if self.allow_mock and self._active_backend_name != "mock":
                self._backend = MockEmbeddingBackend(native_dim=NATIVE_DIMENSION)
                self._active_backend_name = "mock"
            return self.is_service_responsive()

        cmd = [
            str(server_bin),
            "-m",
            str(target_model_file),
            "--port",
            str(self.port),
            "--host",
            str(self.host),
            "--embedding",
            "-ngl",
            "99",
            "-c",
            "2048",
        ]

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        try:
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            logger.info("Spawned dedicated embedding llama-server (PID: %d)", self._server_process.pid)
        except Exception as exc:
            logger.error("Failed to spawn embedding llama-server: %s", exc)
            return False

        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.5)
            if self.is_service_responsive(timeout=1.0):
                self._backend = LlamaServerEmbeddingBackend(
                    host=self.host,
                    port=self.port,
                    model_id=self.model_id,
                )
                self._active_backend_name = "llama-server"
                return True

        return False

    def stop_service(self):
        """Cleanly terminates any spawned embedding server process."""
        if self._server_process:
            try:
                if self._server_process.poll() is None:
                    self._server_process.terminate()
                    self._server_process.wait(timeout=2.0)
            except Exception:
                try:
                    self._server_process.kill()
                except Exception:
                    pass
            self._server_process = None


class MockEmbeddingManager(EmbeddingManager):
    """Convenience subclass enforcing the mock backend for deterministic unit tests."""

    def __init__(
        self,
        config: Optional[MultiScaleConfig] = None,
        matryoshka_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            backend="mock",
            matryoshka_dim=matryoshka_dim,
            allow_mock=True,
            **kwargs,
        )
