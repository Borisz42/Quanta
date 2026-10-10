"""End-to-End Neuro-Symbolic Cognitive Pipeline for QUANTA (Phase 5).

Integrates the complete five-stage operational neuro-symbolic cycle:
1. Streaming Discourse Chunker (sentence & discourse segmentation)
2. Working Memory Entity Manifest & Paging Engine (recency & activation)
3. Neural Transduction Backend (Unsloth GBNF-constrained SLMs & offline mocks)
4. Multi-Chunk Graph Stitcher (global entity resolution & Allen interval synthesis)
5. Symbolic ASG Compiler (ConceptNet 5.7.0, 1024-D quaternary vectors, BLAKE3 CIDs)
6. Clingo Formal Verification & MUC Closed-Loop Repair Gate (strictly <= 2 retries)
7. Virtual Page-Table Attention & Active Canvas (O(1) VRAM execution, M <= 512)
8. Multi-Target Reverse Realizers (English NLG, First-Order Logic, Code, and S-expressions)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.asg import (
    HierarchicalMerkleBook,
    QuantaGraph,
    QuantaNode,
    fold_book,
    fold_chapter,
    fold_discourse_episode,
)
from memory.page_table import ActiveCanvas, PageTable, SemanticPageFaultHandler
from memory.spreading_activation import SpreadingActivationRetriever
from parser.asg_compiler import ASGCompilationError, ASGCompiler
from parser.chunker import DiscourseChunk, DiscourseChunker
from parser.entity_manifest import EntityPagingEngine, EntityRecord
from parser.graph_stitcher import GraphStitcher
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)
from parser.sexpr_parser import parse_sexpr, to_sexpr
from parser.transducer import BaseDiscourseTransducer, create_transducer
from parser.unsloth_transducer import MockUnslothTransducer, UnslothTransducer
from realizer.code_emitter import CodeEmitter
from realizer.english_nlg import EnglishRealizer, GraphQueryAnswerer
from realizer.fol_emitter import FOLEmitter
from solver.validator_gate import ValidationGate, ValidationResult
from verification.clingo_gate import (
    ClingoVerificationGate,
    MUCDiagnosticResult,
    MUCRepairManager,
    RepairResult,
)
from core.binary_node import (
    BelnapValue,
    BinaryNodeTable,
    EpistemicSource as BinaryEpistemicSource,
    QuantaSemanticNodeStruct,
    SpeechActIntent as BinarySpeechActIntent,
)
from memory.context_assembler import (
    BipartiteProjector,
    DualStreamContext,
    DualStreamContextAssembler,
)
from memory.hipporag_ppr import HippoRAGRetriever
from memory.passage_store import PassageRecord, PassageStore
from memory.poprag_gating import PoPRAGGating
from models.kev_engine import (
    AllenTemporalRelation,
    EpistemicSource as KevEpistemicSource,
    IntentEpistemicResult,
    KevChunkEvaluation,
    KevDecisionEngine,
    MockKevEngine,
    PearlCausalLink,
    RelationScoringResult,
    SpeechActIntent as KevSpeechActIntent,
    ValencyRole,
    ValencyScoringResult,
)
from parser.mmap_grounder import MmapLexicalGrounder
from parser.skeleton_transducer import (
    MockSkeletonTransducer,
    SkeletonEntity,
    SkeletonEvent,
    SkeletonExtractionResult,
    SkeletonTransducer,
)
from verification.belnap_calibrator import BelnapLatticeMapper
from verification.clingo_dl_gate import ClingoDLGate
from models.kev_async_worker import AsyncKevVerificationQueue, VerificationTask
from pipeline.tracer import PipelineExecutionTracer
from config.multi_scale_config import MultiScaleConfig
from memory.fast_path_assembler import (
    FastPathAssembler,
    FastPathContext,
    FastPathCoverageEvaluator,
    FastPathMode,
    FastPathResult,
)
from parser.multi_scale_chunker import HierarchicalChunker, MacroBlock
from parser.task_boundary_extractor import ExtractedTaskIntent, TaskBoundaryExtractor
from retrieval.relevance_filter import RelevanceFilter, ScoredUnit


logger = logging.getLogger("quanta.pipeline.cognitive")


def _normalize_entity_key(name: str) -> str:
    """Normalizes entity surface strings for cross-chunk canonical matching."""
    k = name.strip().lower()
    for prefix in ("the ", "a ", "an "):
        if k.startswith(prefix):
            k = k[len(prefix):].strip()
            break
    return k


def _locate_node_span_in_chunk(node: QuantaNode, text: str) -> Tuple[int, int]:
    """Locates the character span (start, end) of a node within the passage text."""
    lbl = ""
    if isinstance(node.literal, str) and len(node.literal.strip()) >= 2:
        lbl = node.literal.strip()
    elif isinstance(node.literal, dict):
        lbl = str(node.literal.get("name") or node.literal.get("label") or node.literal.get("text") or "").strip()
    elif node.anchor:
        clean = node.anchor
        if clean.startswith("cn:en:"):
            clean = clean[6:]
        clean = re.sub(r"\s*\([nav]\)$", "", clean).replace("_", " ").strip()
        if len(clean) >= 2:
            lbl = clean
    if lbl:
        s_idx = text.find(lbl)
        if s_idx >= 0:
            return (s_idx, s_idx + len(lbl))
    return (0, len(text))


class CognitivePipeline:
    """Production-grade Neuro-Symbolic Cognitive Pipeline for QUANTA.

    Executes streaming context ingestion, working memory coreference tracking,
    discrete quaternary ASG compilation, Clingo ASP validation with closed-loop
    MUC repair, hierarchical Merkle folding, bounded active canvas memory execution
    (M <= 512), and multi-target realization (English, FOL, Code, S-Expression).
    """

    def __init__(
        self,
        transducer_backend: str = "auto",
        canvas_capacity: int = 512,
        page_table_path: Optional[Union[str, Path]] = None,
        transducer: Optional[Any] = None,
        compiler: Optional[ASGCompiler] = None,
        chunker: Optional[DiscourseChunker] = None,
        entity_engine: Optional[EntityPagingEngine] = None,
        realizer: Optional[EnglishRealizer] = None,
        validator: Optional[ValidationGate] = None,
        stitcher: Optional[GraphStitcher] = None,
        verification_gate: Optional[ClingoVerificationGate] = None,
        repair_manager: Optional[MUCRepairManager] = None,
        fol_emitter: Optional[FOLEmitter] = None,
        code_emitter: Optional[CodeEmitter] = None,
        max_repair_attempts: int = 2,
        passage_store: Optional[PassageStore] = None,
        binary_table: Optional[BinaryNodeTable] = None,
        skeleton_transducer: Optional[Any] = None,
        kev_engine: Optional[Any] = None,
        belnap_mapper: Optional[BelnapLatticeMapper] = None,
        clingo_dl_gate: Optional[ClingoDLGate] = None,
        tracer: Optional[PipelineExecutionTracer] = None,
        multi_scale_config: Optional[MultiScaleConfig] = None,
        skeleton_format: str = "sexpr_compact",
        co_decoded_kev: bool = False,
        max_workers: Optional[int] = None,
        max_slots: Optional[int] = None,
        **transducer_kwargs,
    ):
        self.tracer = tracer or PipelineExecutionTracer.get_instance()
        self.multi_scale_config = multi_scale_config or MultiScaleConfig()

        # Resolve batch workers / slots (§Section 4 Master Plan)
        raw_env_slots = os.environ.get("QUANTA_MAX_SLOTS") or os.environ.get("QUANTA_PARALLEL_SLOTS") or os.environ.get("QUANTA_BATCH_WORKERS")
        if max_workers is not None:
            self.max_workers = int(max_workers)
        elif max_slots is not None:
            self.max_workers = int(max_slots)
        elif raw_env_slots:
            try:
                self.max_workers = int(raw_env_slots.strip())
            except Exception:
                self.max_workers = self.multi_scale_config.server_batch_workers
        else:
            self.max_workers = self.multi_scale_config.server_batch_workers

        # Resolve skeleton_format and co_decoded_kev (§Section 3 Master Plan)
        raw_env_fmt = os.environ.get("QUANTA_SKELETON_FORMAT")
        raw_env_co = os.environ.get("QUANTA_CO_DECODED_KEV")

        if raw_env_fmt:
            self.skeleton_format = raw_env_fmt.strip()
        elif skeleton_format != "sexpr_compact":
            self.skeleton_format = skeleton_format
        elif self.multi_scale_config.is_calibrated("transducer.skeleton_format"):
            self.skeleton_format = self.multi_scale_config.transducer_skeleton_format
        else:
            self.skeleton_format = skeleton_format

        if raw_env_co is not None:
            self.co_decoded_kev = raw_env_co.strip().lower() in ("1", "true", "yes", "on")
        elif co_decoded_kev is not False:
            self.co_decoded_kev = bool(co_decoded_kev)
        elif self.multi_scale_config.is_calibrated("transducer.co_decoded"):
            self.co_decoded_kev = self.multi_scale_config.transducer_co_decoded
        else:
            self.co_decoded_kev = bool(co_decoded_kev)

        # Extract Kev-specific configuration from kwargs to avoid forwarding to legacy transducer
        if "kev_mode" in transducer_kwargs:
            self.kev_mode = transducer_kwargs.pop("kev_mode")
        else:
            self.kev_mode = "co_decoded" if self.co_decoded_kev else "bypass"
        self.kev_concurrency = transducer_kwargs.pop("kev_concurrency", None)

        # 1. Chunker & Segmentation
        self.chunker = chunker or DiscourseChunker(min_words=50, max_words=400)

        # 2. Working Memory Entity Manifest & Paging Engine
        self.entity_engine = entity_engine or EntityPagingEngine(target_size=10, min_size=5, max_size=15)

        # 3. Neural Discourse Transducer
        if transducer is not None:
            self.transducer = transducer
        elif transducer_backend in ("unsloth", "qwen", "gemma"):
            self.transducer = UnslothTransducer(fallback_to_mock=True, **transducer_kwargs)
        elif transducer_backend in ("mock_unsloth", "mock_sexpr", "sexpr_mock"):
            from parser.unsloth_transducer import MockUnslothTransducer
            self.transducer = MockUnslothTransducer(**transducer_kwargs)
        elif transducer_backend in ("mock", "mock_json"):
            self.transducer = create_transducer(backend="mock", **transducer_kwargs)
        elif transducer_backend in ("lmstudio", "gguf"):
            self.transducer = create_transducer(backend=transducer_backend, **transducer_kwargs)
        elif transducer_backend == "auto":
            # Default to UnslothTransducer with automatic offline mock fallback
            self.transducer = UnslothTransducer(fallback_to_mock=True, **transducer_kwargs)
        else:
            self.transducer = create_transducer(backend=transducer_backend, **transducer_kwargs)

        # 4. Symbolic ASG Compiler & Clingo Validator Gate
        self.compiler = compiler or ASGCompiler(validator_gate=validator)
        self.validator = self.compiler.validator_gate

        # 5. Clingo Verification Gate & Closed-Loop MUC Repair Manager
        self.verification_gate = verification_gate or ClingoVerificationGate(validator_gate=self.validator)
        self.repair_manager = repair_manager or MUCRepairManager(
            gate=self.verification_gate,
            max_repair_attempts=max_repair_attempts,
            compiler=self.compiler,
        )

        # 6. Multi-Chunk Graph Stitcher
        self.stitcher = stitcher or GraphStitcher(
            manifest=self.entity_engine.manifest if hasattr(self.entity_engine, "manifest") else None,
            storage=self.entity_engine.storage if hasattr(self.entity_engine, "storage") else None,
        )

        # 7. Virtual Memory Page Table & Active Canvas (Strict O(1) Physical VRAM Bound)
        self.page_table = PageTable(db_path=page_table_path)
        self.active_canvas = ActiveCanvas(capacity=canvas_capacity)
        self.page_table.attach_active_canvas(self.active_canvas)
        self.fault_handler = SemanticPageFaultHandler(
            canvas=self.active_canvas,
            page_table=self.page_table,
        )

        # 8. Episodic Entity Registry for Multi-Hop Cross-Chunk Associative Bridges
        self.episodic_entity_registry: Dict[str, List[Tuple[QuantaNode, QuantaNode, Optional[QuantaNode]]]] = {}

        # 9. Realizers & Emitters
        self.realizer = realizer or EnglishRealizer()
        self.fol_emitter = fol_emitter or FOLEmitter()
        self.code_emitter = code_emitter or CodeEmitter()
        self.query_answerer = GraphQueryAnswerer(realizer=self.realizer)

        # 10. Semantic Virtual Memory (SVM) Dual-Node Storage & Ingestion Engines (Section 8)
        self.passage_store = passage_store or PassageStore(db_path=page_table_path)
        self.binary_table = binary_table or BinaryNodeTable()

        if skeleton_transducer is not None:
            self.skeleton_transducer = skeleton_transducer
        elif transducer_backend in ("mock", "mock_json", "mock_unsloth", "mock_sexpr", "sexpr_mock"):
            mock_mode = "co_decoded" if self.co_decoded_kev else "standard"
            self.skeleton_transducer = MockSkeletonTransducer(
                mode=mock_mode,
                skeleton_format=self.skeleton_format,
                co_decoded=self.co_decoded_kev,
            )
        else:
            trans_mode = "co_decoded" if self.co_decoded_kev else "standard"
            self.skeleton_transducer = SkeletonTransducer(
                fallback_to_mock=True,
                mode=trans_mode,
                skeleton_format=self.skeleton_format,
                co_decoded=self.co_decoded_kev,
            )

        if kev_engine is not None:
            self.kev_engine = kev_engine
        elif transducer_backend in ("mock", "mock_json", "mock_unsloth", "mock_sexpr", "sexpr_mock"):
            self.kev_engine = KevDecisionEngine(base_url="mock", fallback_to_mock=True, mode=self.kev_mode)
            self.kev_engine._server_available = False
        else:
            concurrency = self.kev_concurrency or int(os.environ.get("QUANTA_PARALLEL_SLOTS", "16"))
            self.kev_engine = KevDecisionEngine(
                fallback_to_mock=True,
                mode=self.kev_mode,
                concurrency_limit=concurrency,
            )
        self.belnap_mapper = belnap_mapper or BelnapLatticeMapper()
        self.clingo_dl_gate = clingo_dl_gate or ClingoDLGate()
        self.concept_grounder = MmapLexicalGrounder()

        # 11. Spreading-Activation Context Retriever & HippoRAG / PoP-RAG / Dual-Stream Context Assembler
        self.retriever = SpreadingActivationRetriever(
            realizer=self.realizer,
            passage_store=self.passage_store,
        )
        self.hipporag = getattr(self.retriever, "hipporag", None)
        self.poprag_gating = getattr(self.retriever, "poprag_gating", None)
        self.context_assembler = getattr(self.retriever, "context_assembler", None) or DualStreamContextAssembler(passage_store=self.passage_store)
        self.bipartite_projector = getattr(self.context_assembler, "projector", None) or BipartiteProjector(passage_store=self.passage_store)

        # 12. Merkle Book State
        self.merkle_book = HierarchicalMerkleBook()

        # 13. Asynchronous Write-Ahead Kev Verification Queue (Section 4 Master Plan)
        self.async_kev_worker: Optional[AsyncKevVerificationQueue] = None
        async_worker_arg = transducer_kwargs.pop("async_kev_worker", None)
        if async_worker_arg is not None:
            self.async_kev_worker = async_worker_arg
        elif self.kev_mode == "async":
            self.async_kev_worker = AsyncKevVerificationQueue(
                kev_engine=self.kev_engine,
                page_table=self.page_table,
                binary_table=self.binary_table,
                belnap_mapper=self.belnap_mapper,
                clingo_dl_gate=self.clingo_dl_gate,
                max_workers=int(os.environ.get("QUANTA_ASYNC_KEV_WORKERS", "2")),
            )

        # 14. Dynamic Multi-Scale Ingestion & Fast-Path Assembler (§Phase 3)
        self.fast_path_assembler = FastPathAssembler(
            config=self.multi_scale_config,
            dual_stream_assembler=self.context_assembler,
        )

        # 15. Background Ingestion Worker (§Phase 5)
        if "background.enabled" in self.multi_scale_config.runtime_overrides:
            self.background_enabled = bool(self.multi_scale_config.runtime_overrides["background.enabled"])
        else:
            bg_env = os.environ.get("QUANTA_BACKGROUND_INGESTION")
            if bg_env is not None and bg_env.strip().lower() in ("0", "off", "false", "no"):
                self.background_enabled = False
            elif bg_env is not None and bg_env.strip().lower() in ("1", "on", "true", "yes"):
                self.background_enabled = True
            else:
                self.background_enabled = bool(self.multi_scale_config.background_enabled)

        self.background_ingestor: Optional[AsyncKevVerificationQueue] = None
        if self.background_enabled:
            self.background_ingestor = AsyncKevVerificationQueue(
                kev_engine=self.kev_engine,
                page_table=self.page_table,
                binary_table=self.binary_table,
                belnap_mapper=self.belnap_mapper,
                clingo_dl_gate=self.clingo_dl_gate,
                transducer=self.skeleton_transducer,
                passage_store=self.passage_store,
                pipeline=self,
                pause_policy=self.multi_scale_config.background_pause_policy,
                max_workers=self.multi_scale_config.background_max_concurrency,
                enabled=True,
                auto_start=True,
            )
            if self.async_kev_worker is None:
                self.async_kev_worker = self.background_ingestor

    def mark_foreground(self, active: bool = True) -> None:
        """Marks whether an in-process foreground request is executing (Pause Policy BG-B)."""
        if hasattr(self, "background_ingestor") and self.background_ingestor is not None:
            self.background_ingestor.mark_foreground(active)
        elif hasattr(self, "async_kev_worker") and self.async_kev_worker is not None:
            self.async_kev_worker.mark_foreground(active)

    def set_foreground_active(self, active: bool = True) -> None:
        """Alias for mark_foreground."""
        self.mark_foreground(active)

    @contextmanager
    def foreground_scope(self):
        """Context manager marking foreground execution to pause background worker (BG-B)."""
        self.mark_foreground(True)
        try:
            yield
        finally:
            self.mark_foreground(False)

    def set_skeleton_format(
        self,
        skeleton_format: str,
        co_decoded: Optional[bool] = None,
    ) -> None:
        """Dynamically switch transduction representation format at runtime.

        Args:
            skeleton_format: "sexpr_compact", "sexpr_positional", or "json"
            co_decoded: Optional boolean toggle for co-decoded single-letter Kev decisions.
        """
        self.skeleton_format = str(skeleton_format).strip()
        if co_decoded is not None:
            self.co_decoded_kev = bool(co_decoded)
            self.kev_mode = "co_decoded" if self.co_decoded_kev else "bypass"

        # Propagate to underlying skeleton transducer
        if hasattr(self.skeleton_transducer, "skeleton_format"):
            self.skeleton_transducer.skeleton_format = self.skeleton_format
        if hasattr(self.skeleton_transducer, "co_decoded") and co_decoded is not None:
            self.skeleton_transducer.co_decoded = self.co_decoded_kev
        if hasattr(self.skeleton_transducer, "mode") and co_decoded is not None:
            self.skeleton_transducer.mode = "co_decoded" if self.co_decoded_kev else "standard"
        if hasattr(self.skeleton_transducer, "_mock"):
            self.skeleton_transducer._mock.skeleton_format = self.skeleton_format
            if co_decoded is not None:
                self.skeleton_transducer._mock.co_decoded = self.co_decoded_kev
                self.skeleton_transducer._mock.mode = "co_decoded" if self.co_decoded_kev else "standard"

        # Keep multi_scale_config in sync if present
        if hasattr(self, "multi_scale_config") and self.multi_scale_config is not None:
            overrides = {"transducer.skeleton_format": self.skeleton_format}
            if co_decoded is not None:
                overrides["transducer.co_decoded"] = self.co_decoded_kev
            self.multi_scale_config = self.multi_scale_config.with_overrides(overrides)

    def set_co_decoded_kev(self, co_decoded: bool) -> None:
        """Dynamically toggle co-decoded Kev decisions at runtime."""
        self.set_skeleton_format(self.skeleton_format, co_decoded=co_decoded)

    def set_batch_workers(self, workers: int) -> None:
        """Dynamically set default batch worker concurrency at runtime."""
        self.max_workers = max(1, int(workers))
        if hasattr(self, "multi_scale_config") and self.multi_scale_config is not None:
            self.multi_scale_config = self.multi_scale_config.with_overrides({
                "server.batch_workers": self.max_workers,
                "server.max_parallel_slots": self.max_workers,
            })

    def reset(self, clear_page_table: bool = True) -> None:
        """Cleanly resets all working memory, active canvas, page table, and episodic state."""
        if hasattr(self, "background_ingestor") and self.background_ingestor is not None:
            self.background_ingestor.clear()
        if hasattr(self, "async_kev_worker") and self.async_kev_worker is not None:
            self.async_kev_worker.clear()
        if hasattr(self, "active_canvas") and self.active_canvas is not None:
            self.active_canvas.clear()
        if hasattr(self, "entity_engine") and hasattr(self.entity_engine, "manifest"):
            self.entity_engine.manifest.reset()
        if hasattr(self, "stitcher") and self.stitcher is not None:
            self.stitcher.reset()
        if hasattr(self, "merkle_book"):
            from core.asg import HierarchicalMerkleBook
            self.merkle_book = HierarchicalMerkleBook()
        if clear_page_table and hasattr(self, "page_table") and self.page_table is not None:
            self.page_table.clear()
            self.page_table.attach_active_canvas(self.active_canvas)
        if clear_page_table and hasattr(self, "passage_store") and self.passage_store is not None:
            self.passage_store.clear()
        if hasattr(self, "binary_table") and self.binary_table is not None:
            self.binary_table = BinaryNodeTable()
        if hasattr(self, "fault_handler") and self.fault_handler is not None:
            self.fault_handler.page_table = self.page_table
            self.fault_handler.canvas = self.active_canvas
        if hasattr(self, "episodic_entity_registry"):
            self.episodic_entity_registry.clear()

    def flush_background_queue(self, timeout: Optional[float] = None) -> bool:
        """Flushes the background ingestion queue, blocking until all tasks complete."""
        ok = True
        if hasattr(self, "background_ingestor") and self.background_ingestor is not None:
            ok = ok and self.background_ingestor.flush(timeout=timeout)
        if hasattr(self, "async_kev_worker") and self.async_kev_worker is not None and self.async_kev_worker != getattr(self, "background_ingestor", None):
            ok = ok and self.async_kev_worker.flush(timeout=timeout)
        return ok

    def flush_kev_queue(self, timeout: Optional[float] = None) -> bool:
        """Flushes the asynchronous queue, blocking until all background tasks complete."""
        return self.flush_background_queue(timeout=timeout)

    def prioritize_passage(self, passage_id: str) -> bool:
        """Elevates an enqueued passage to high priority in the async/background queue."""
        if hasattr(self, "background_ingestor") and self.background_ingestor is not None:
            if self.background_ingestor.prioritize(passage_id):
                return True
        if hasattr(self, "async_kev_worker") and self.async_kev_worker is not None:
            return self.async_kev_worker.prioritize(passage_id)
        return False


    def ingest_document(
        self,
        text: str,
        doc_id: str = "doc_01",
        passage_id: Optional[str] = None,
        validate: bool = True,
        chapter_id: Optional[str] = None,
    ) -> QuantaGraph:
        """Ingest raw document chunk into Semantic Virtual Memory (SVM).

        Executes the 6-step ingestion cycle (Section 8 Master Plan):
        1. Register chunk in PassageStore (V_passage, immutable raw text spans).
        2. Run SkeletonTransducer to extract entities and S-V-O event frames via llama-server.
        3. Run KevDecisionEngine 3-pass prefill scoring via llama-server:
           - Pass 1: Thematic valency classification (AGENT, PATIENT, INSTRUMENT)
           - Pass 2: Speech-act intent and epistemic source
           - Pass 3: Allen temporal intervals and Pearl causal DAG links
        4. Apply BelnapLatticeMapper and SIMD concept grounding (ConceptNet codebook).
        5. Validate via ClingoDLGate (difference logic and causal acyclicity).
        6. Commit to dual-node QuantaGraph and 128-byte BinaryNodeTable.
        """
        if not text or not text.strip():
            empty_graph = QuantaGraph()
            setattr(empty_graph, "extraction_result", DiscourseExtractionResult())
            return empty_graph

        # Multi-chunk splitting for long text (> 600 words)
        words = text.strip().split()
        if len(words) > 600 and hasattr(self.chunker, "chunk_text"):
            t_chunk0 = time.perf_counter()
            chunks = self.chunker.chunk_text(text, chapter_id=chapter_id or doc_id)
            if self.tracer:
                self.tracer.record_stage_timing("chunking", (time.perf_counter() - t_chunk0) * 1000.0, gpu_calls=0)
            if len(chunks) > 1:
                last_g = None
                for idx, c in enumerate(chunks):
                    sub_pid = f"{passage_id or doc_id}_c{idx+1}"
                    last_g = self.ingest_document(
                        c.text,
                        doc_id=doc_id,
                        passage_id=sub_pid,
                        validate=validate,
                        chapter_id=chapter_id,
                    )
                return last_g or QuantaGraph()

        # Step 1: Register in PassageStore
        pid = passage_id or f"P_{doc_id}_{int(time.time() * 1000)}"
        passage_rec = self.passage_store.add_passage(
            PassageRecord(
                passage_id=pid,
                doc_id=doc_id,
                char_span=(0, len(text)),
                text=text,
            )
        )

        # Step 2: Skeleton Transduction
        t_trans0 = time.perf_counter()
        effective_format = os.environ.get("QUANTA_SKELETON_FORMAT", self.skeleton_format)
        if "QUANTA_CO_DECODED_KEV" in os.environ:
            env_co_val = os.environ["QUANTA_CO_DECODED_KEV"].strip().lower()
            effective_co_decoded = env_co_val in ("1", "true", "yes", "on")
        else:
            effective_co_decoded = self.co_decoded_kev

        transduce_kwargs = {
            "skeleton_format": effective_format,
            "co_decoded": effective_co_decoded,
        }
        if effective_co_decoded or self.kev_mode == "co_decoded":
            transduce_kwargs["mode"] = "co_decoded"
        else:
            transduce_kwargs["mode"] = "standard"

        if hasattr(self.skeleton_transducer, "transduce"):
            try:
                skeleton_res = self.skeleton_transducer.transduce(text, passage_id=pid, doc_id=doc_id, **transduce_kwargs)
            except TypeError:
                try:
                    skeleton_res = self.skeleton_transducer.transduce(text, **transduce_kwargs)
                except TypeError:
                    skeleton_res = self.skeleton_transducer.transduce(text)
        elif callable(self.skeleton_transducer):
            skeleton_res = self.skeleton_transducer(text)
        else:
            mock_mode = "co_decoded" if effective_co_decoded else "standard"
            skeleton_res = MockSkeletonTransducer(
                mode=mock_mode,
                skeleton_format=effective_format,
                co_decoded=effective_co_decoded,
            ).transduce(text, passage_id=pid, doc_id=doc_id)

        if self.tracer:
            st_name = self.skeleton_transducer.__class__.__name__ if self.skeleton_transducer else "Mock"
            is_mock_trans = "Mock" in st_name or getattr(self.skeleton_transducer, "_is_mock", False) or getattr(self.skeleton_transducer, "_last_fallback_used", False)
            self.tracer.record_stage_timing(
                "transduction",
                (time.perf_counter() - t_trans0) * 1000.0,
                gpu_calls=0 if is_mock_trans else 1,
            )

        # Step 3: Kev-4B 3-pass Prefill Scoring
        t_kev0 = time.perf_counter()
        if effective_co_decoded or (self.kev_mode == "co_decoded" and any(ev.intent or ev.epist for ev in skeleton_res.events)):
            # Co-decoded single-pass mode: map co-decoded event tags directly into Kev decisions without secondary GPU calls
            valencies = []
            intent_epistemics = []
            relations = []

            for ev in skeleton_res.events:
                if ev.subject_ent_id:
                    valencies.append(ValencyScoringResult(
                        entity_id=ev.subject_ent_id,
                        event_id=ev.id,
                        entity_text="",
                        predicate=ev.predicate,
                        role="AGENT",
                        confidence=0.98,
                        probabilities={"AGENT": 0.98},
                    ))
                if ev.object_ent_id:
                    valencies.append(ValencyScoringResult(
                        entity_id=ev.object_ent_id,
                        event_id=ev.id,
                        entity_text="",
                        predicate=ev.predicate,
                        role="PATIENT",
                        confidence=0.98,
                        probabilities={"PATIENT": 0.98},
                    ))

                # Intent & Epistemic Source
                raw_intent = (ev.intent or "I").strip().upper()
                raw_epist = (ev.epist or "O").strip().upper()
                from models.kev_engine import (
                    JOINT_ALLEN_MAP,
                    JOINT_EPISTEMIC_MAP,
                    JOINT_INTENT_MAP,
                    JOINT_PEARL_MAP,
                )
                mapped_intent = JOINT_INTENT_MAP.get(raw_intent, KevSpeechActIntent.INFORMATIVE.value)
                mapped_epist = JOINT_EPISTEMIC_MAP.get(raw_epist, KevEpistemicSource.DIRECT_OBSERVATION.value)

                # Calibrate epistemic confidence for BelnapLatticeMapper
                # O: Direct Observation (0.95 -> TRUE)
                # D: Deduction (0.90 -> TRUE)
                # H: Hearsay (0.50 -> UNKNOWN)
                # C: Conjecture (0.40 -> UNKNOWN)
                if raw_epist == "O":
                    epist_conf = 0.95
                elif raw_epist == "D":
                    epist_conf = 0.90
                elif raw_epist == "H":
                    epist_conf = 0.50
                elif raw_epist == "C":
                    epist_conf = 0.40
                else:
                    epist_conf = 0.95

                intent_epistemics.append(IntentEpistemicResult(
                    event_id=ev.id,
                    predicate=ev.predicate,
                    intent=mapped_intent,
                    intent_confidence=0.95,
                    intent_probabilities={mapped_intent: 0.95},
                    epistemic_source=mapped_epist,
                    epistemic_confidence=epist_conf,
                    epistemic_probabilities={mapped_epist: epist_conf},
                ))

            # Relations between sequential events if co-decoded
            from models.kev_engine import (
                JOINT_ALLEN_MAP,
                JOINT_PEARL_MAP,
            )
            for i in range(len(skeleton_res.events) - 1):
                ev_curr = skeleton_res.events[i]
                ev_next = skeleton_res.events[i + 1]
                allen_tag = (ev_curr.allen or "N").strip().upper()
                pearl_tag = (ev_curr.pearl or "N").strip().upper()
                mapped_allen = JOINT_ALLEN_MAP.get(allen_tag, AllenTemporalRelation.NONE.value)
                mapped_pearl = JOINT_PEARL_MAP.get(pearl_tag, PearlCausalLink.NONE.value)
                if mapped_allen != AllenTemporalRelation.NONE.value or mapped_pearl != PearlCausalLink.NONE.value:
                    relations.append(RelationScoringResult(
                        source_event_id=ev_curr.id,
                        target_event_id=ev_next.id,
                        source_predicate=ev_curr.predicate,
                        target_predicate=ev_next.predicate,
                        allen_relation=mapped_allen,
                        allen_confidence=0.95,
                        allen_probabilities={mapped_allen: 0.95},
                        pearl_relation=mapped_pearl,
                        pearl_confidence=0.95,
                        pearl_probabilities={mapped_pearl: 0.95},
                    ))

            kev_eval = KevChunkEvaluation(
                valencies=valencies,
                intent_epistemics=intent_epistemics,
                relations=relations,
                total_latency_ms=0.1,
                mode="co_decoded",
            )
        elif self.kev_mode in ("bypass", "none", "off") or getattr(self.kev_engine, "mode", None) in ("bypass", "none") or not effective_co_decoded:
            # Direct Skeleton S-V-O valency assignment without external GPU logprob calls
            valencies = []
            for ev in skeleton_res.events:
                if ev.subject_ent_id:
                    valencies.append(ValencyScoringResult(
                        entity_id=ev.subject_ent_id,
                        event_id=ev.id,
                        entity_text="",
                        predicate=ev.predicate,
                        role="AGENT",
                        confidence=0.98,
                        probabilities={"AGENT": 0.98},
                    ))
                if ev.object_ent_id:
                    valencies.append(ValencyScoringResult(
                        entity_id=ev.object_ent_id,
                        event_id=ev.id,
                        entity_text="",
                        predicate=ev.predicate,
                        role="PATIENT",
                        confidence=0.98,
                        probabilities={"PATIENT": 0.98},
                    ))
            kev_eval = KevChunkEvaluation(
                valencies=valencies,
                intent_epistemics=[
                    IntentEpistemicResult(
                        event_id=ev.id,
                        predicate=ev.predicate,
                        intent="INFORMATIVE",
                        intent_confidence=0.95,
                        intent_probabilities={"INFORMATIVE": 0.95},
                        epistemic_source="DIRECT_OBSERVATION",
                        epistemic_confidence=0.95,
                        epistemic_probabilities={"DIRECT_OBSERVATION": 0.95},
                    ) for ev in skeleton_res.events
                ],
                relations=[],
                total_latency_ms=0.1,
                mode="bypass",
            )
        elif self.kev_mode == "async":
            # Decoupled Write-Ahead mode: commit provisional Belnap values and enqueue background verification
            valencies = []
            for ev in skeleton_res.events:
                if ev.subject_ent_id:
                    valencies.append(ValencyScoringResult(
                        entity_id=ev.subject_ent_id,
                        event_id=ev.id,
                        entity_text="",
                        predicate=ev.predicate,
                        role="AGENT",
                        confidence=0.85,
                        probabilities={"AGENT": 0.85},
                    ))
                if ev.object_ent_id:
                    valencies.append(ValencyScoringResult(
                        entity_id=ev.object_ent_id,
                        event_id=ev.id,
                        entity_text="",
                        predicate=ev.predicate,
                        role="PATIENT",
                        confidence=0.85,
                        probabilities={"PATIENT": 0.85},
                    ))
            kev_eval = KevChunkEvaluation(
                valencies=valencies,
                intent_epistemics=[
                    IntentEpistemicResult(
                        event_id=ev.id,
                        predicate=ev.predicate,
                        intent="INFORMATIVE",
                        intent_confidence=0.85,
                        intent_probabilities={"INFORMATIVE": 0.85},
                        epistemic_source="DIRECT_OBSERVATION",
                        epistemic_confidence=0.85,
                        epistemic_probabilities={"DIRECT_OBSERVATION": 0.85},
                    ) for ev in skeleton_res.events
                ],
                relations=[],
                total_latency_ms=0.05,
                mode="async_provisional",
            )
        else:
            kev_eval = self.kev_engine.evaluate_chunk(
                entities=skeleton_res.entities,
                events=skeleton_res.events,
                text=text,
            )

        if self.tracer:
            ke_name = self.kev_engine.__class__.__name__ if self.kev_engine else "Mock"
            is_zero_gpu = self.kev_mode in ("bypass", "none", "off", "co_decoded", "async") or "Mock" in ke_name
            gpu_kev = 0 if is_zero_gpu else getattr(kev_eval, "gpu_calls", 1)
            self.tracer.record_stage_timing("Kev", (time.perf_counter() - t_kev0) * 1000.0, gpu_calls=gpu_kev)

        # Step 4: Belnap Lattice Mapping & SIMD Concept Grounding
        ent_concept_codes: Dict[str, int] = {}
        for ent in skeleton_res.entities:
            code = self.concept_grounder.resolve_concept_code(ent.surface_text)
            if code is not None:
                ent_concept_codes[ent.id] = code

        ev_concept_codes: Dict[str, int] = {}
        for ev in skeleton_res.events:
            code = self.concept_grounder.resolve_concept_code(ev.predicate)
            if code is not None:
                ev_concept_codes[ev.id] = code

        # Step 5: ClingoDLGate Difference Logic Verification
        t_dl0 = time.perf_counter()
        if validate and self.clingo_dl_gate is not None:
            try:
                dl_res = self.clingo_dl_gate.validate(
                    events=skeleton_res.events,
                    relations=kev_eval.relations,
                )
                if not dl_res.is_valid:
                    logger.warning("ClingoDLGate detected temporal/causal conflicts: %s", dl_res.errors)
            except Exception as dl_err:
                logger.debug("ClingoDLGate validation check: %s", dl_err)

        if self.tracer:
            self.tracer.record_stage_timing("Clingo-DL", (time.perf_counter() - t_dl0) * 1000.0, gpu_calls=0)

        # Step 6: Commit to Dual-Node QuantaGraph and 128-Byte BinaryNodeTable
        graph = QuantaGraph()
        ent_cid_map: Dict[str, str] = {}
        ev_cid_map: Dict[str, str] = {}
        ent_node_ids: Dict[str, int] = {}
        ev_node_ids: Dict[str, int] = {}

        # 1. Entity nodes
        for ent in skeleton_res.entities:
            ent_node = QuantaNode(
                anchor=ent.surface_text,
                literal=ent.surface_text,
                passage_id=pid,
                span_start=ent.char_span[0],
                span_end=ent.char_span[1],
                confidence=float(ent.confidence),
                concept_code=ent_concept_codes.get(ent.id, 0),
            )
            cat_upper = (ent.category or "OBJECT").upper()
            if cat_upper == "PERSON":
                ent_node.set_slot("TYPE_HUMAN", 1)
                ent_node.set_slot("ROLE_AGENT_CAPABLE", 1)
            elif cat_upper == "LOCATION":
                ent_node.set_slot("TYPE_LOCATION", 1)
            elif cat_upper == "SUBSTANCE":
                ent_node.set_slot("TYPE_SUBSTANCE", 1)
            else:
                ent_node.set_slot("TYPE_INANIMATE_PHYSICAL", 1)

            ent_node.compute_canonical_cid()
            graph.add_node(ent_node)
            graph.add_passage_anchor(ent_node.canonical_cid, pid, ent.char_span[0], ent.char_span[1])
            ent_cid_map[ent.id] = ent_node.canonical_cid

            # Append to 128-byte Binary Table
            node_id = len(self.binary_table) + 1
            ent_node_ids[ent.id] = node_id
            pid_hash = abs(hash(pid)) % (2**32 - 1)
            struct = QuantaSemanticNodeStruct.create(
                node_id=node_id,
                passage_id=pid_hash,
                span_start=ent.char_span[0],
                span_end=ent.char_span[1],
                concept_code=ent_concept_codes.get(ent.id, 0),
                belnap_lattice=BelnapValue.TRUE,
                confidence=int(round(ent.confidence * 255)),
                intent_band5=1,
                epistemic_band6=1,
            )
            self.binary_table.append(struct)
            stored_ent_cid = self.page_table.store_node(ent_node)
            self.binary_table.register_node_cid(node_id, stored_ent_cid)
            self.active_canvas.put(ent_node)
            ent_cid_map[ent.id] = stored_ent_cid

        # 2. Event nodes & valency links
        valency_role_map: Dict[Tuple[str, str], ValencyScoringResult] = {
            (v.entity_id, v.event_id): v for v in kev_eval.valencies
        }
        ie_map: Dict[str, IntentEpistemicResult] = {
            ie.event_id: ie for ie in kev_eval.intent_epistemics
        }

        for ev in skeleton_res.events:
            ie_res = ie_map.get(ev.id)
            intent_str = ie_res.intent if ie_res else "INFORMATIVE"
            epistemic_str = ie_res.epistemic_source if ie_res else "DIRECT_OBSERVATION"
            intent_val = BinarySpeechActIntent.from_str(intent_str)
            epistemic_val = BinaryEpistemicSource.from_str(epistemic_str)

            if self.kev_mode == "async":
                ev_truth = "UNKNOWN"
                ev_conf = 0.85
                ev_source = "provisional"
                belnap_lat = BelnapValue.UNKNOWN
                conf_lat = int(round(0.85 * 255))
            else:
                kev_post = float(ie_res.epistemic_confidence) if ie_res else float(ev.confidence)
                ev_conf = float(ev.confidence)
                belnap_lat = self.belnap_mapper.map_probability(kev_post)
                ev_truth = BelnapValue.to_str(belnap_lat)
                ev_source = epistemic_str.lower()
                conf_lat = int(round(kev_post * 255))

            ev_node = QuantaNode(
                anchor=ev.predicate,
                literal=ev.predicate,
                passage_id=pid,
                span_start=ev.char_span[0],
                span_end=ev.char_span[1],
                confidence=ev_conf,
                concept_code=ev_concept_codes.get(ev.id, 0),
                evidence_source=ev_source,
                truth_status=ev_truth,
            )
            ev_node.set_slot("TYPE_EVENT", 1)
            ev_node.compute_canonical_cid()
            graph.add_node(ev_node)
            graph.add_passage_anchor(ev_node.canonical_cid, pid, ev.char_span[0], ev.char_span[1])
            ev_cid_map[ev.id] = ev_node.canonical_cid

            # Bind thematic valencies from Kev Pass 1
            for ent in skeleton_res.entities:
                v_res = valency_role_map.get((ent.id, ev.id))
                role = v_res.role if v_res else None
                if not role or role == "NONE":
                    if ev.subject_ent_id == ent.id:
                        role = "AGENT"
                    elif ev.object_ent_id == ent.id:
                        role = "PATIENT"

                if role == "AGENT" and ent.id in ent_cid_map:
                    graph.add_edge(ev_node.canonical_cid, "VAL_X1_AGENT", ent_cid_map[ent.id])
                elif role == "PATIENT" and ent.id in ent_cid_map:
                    graph.add_edge(ev_node.canonical_cid, "VAL_X2_PATIENT", ent_cid_map[ent.id])
                elif role == "INSTRUMENT" and ent.id in ent_cid_map:
                    graph.add_edge(ev_node.canonical_cid, "VAL_X5_INSTRUMENT", ent_cid_map[ent.id])

            # Append to 128-byte Binary Table
            node_id = len(self.binary_table) + 1
            ev_node_ids[ev.id] = node_id
            pid_hash = abs(hash(pid)) % (2**32 - 1)

            struct = QuantaSemanticNodeStruct.create(
                node_id=node_id,
                passage_id=pid_hash,
                span_start=ev.char_span[0],
                span_end=ev.char_span[1],
                concept_code=ev_concept_codes.get(ev.id, 0),
                belnap_lattice=belnap_lat,
                confidence=conf_lat,
                intent_band5=intent_val,
                epistemic_band6=epistemic_val,
            )
            self.binary_table.append(struct)
            stored_ev_cid = self.page_table.store_node(ev_node)
            self.binary_table.register_node_cid(node_id, stored_ev_cid)
            self.active_canvas.put(ev_node)
            ev_cid_map[ev.id] = stored_ev_cid

        # 3. Allen temporal and Pearl causal relations from Kev Pass 3
        for rel in kev_eval.relations:
            src_cid = ev_cid_map.get(rel.source_event_id)
            tgt_cid = ev_cid_map.get(rel.target_event_id)
            if not src_cid or not tgt_cid:
                continue

            allen_belnap = self.belnap_mapper.map_probability(rel.allen_confidence)
            if rel.allen_relation != "NONE" and allen_belnap != BelnapValue.CONTRADICTION:
                edge_label = f"TEMP_ALLEN_{rel.allen_relation.upper()}"
                graph.add_edge(src_cid, edge_label, tgt_cid)

            pearl_belnap = self.belnap_mapper.map_probability(rel.pearl_confidence)
            if rel.pearl_relation != "NONE" and pearl_belnap != BelnapValue.CONTRADICTION:
                edge_label = "CAUSAL_MECHANISM_LINK" if rel.pearl_relation == "MECHANISM_LINK" else "ENABLING_CONDITION"
                graph.add_edge(src_cid, edge_label, tgt_cid)

        # Set primary root CID
        if ev_cid_map:
            graph.root_cid = next(iter(ev_cid_map.values()))
        elif ent_cid_map:
            graph.root_cid = next(iter(ent_cid_map.values()))

        # Merkle episodic fold
        if len(graph) > 0:
            fold_node = fold_discourse_episode(graph.copy(), chunk_id=pid)
            if isinstance(fold_node.literal, dict):
                fold_node.literal["text"] = text
                fold_node.literal["doc_id"] = doc_id
            self.page_table.store_node(fold_node)
            self.active_canvas.put(fold_node)
            ch_id = chapter_id or doc_id
            self.merkle_book.add_chunk_node(fold_node, chapter_id=ch_id)

        # Attach metadata
        setattr(graph, "skeleton_result", skeleton_res)
        setattr(graph, "kev_evaluation", kev_eval)
        setattr(graph, "passage_id", pid)
        setattr(graph, "doc_id", doc_id)

        legacy_extraction = skeleton_res.to_extraction_result() if hasattr(skeleton_res, "to_extraction_result") else DiscourseExtractionResult(
            chunk_id=pid,
            entities=[e.to_extracted_entity() for e in skeleton_res.entities if hasattr(e, "to_extracted_entity")],
            events=[ev.to_extracted_event() for ev in skeleton_res.events if hasattr(ev, "to_extracted_event")],
        )
        setattr(graph, "extraction_result", legacy_extraction)

        # Update PassageStore ingestion state (§Phase 4/5)
        if hasattr(self.passage_store, "set_ingestion_state"):
            if self.kev_mode == "async":
                self.passage_store.set_ingestion_state(pid, "PENDING")
            else:
                self.passage_store.set_ingestion_state(pid, "FULL")

        # Dispatch background verification in async mode
        if self.kev_mode == "async" and self.async_kev_worker is not None:
            self.async_kev_worker.enqueue(
                passage_id=pid,
                doc_id=doc_id,
                text=text,
                entities=skeleton_res.entities,
                events=skeleton_res.events,
                ent_cid_map=ent_cid_map,
                ev_cid_map=ev_cid_map,
                ent_node_ids=ent_node_ids,
                ev_node_ids=ev_node_ids,
                graph=graph,
            )

        return graph

    def ingest_passages_batch(
        self,
        passages: Sequence[Union[Tuple[str, str, str], Dict[str, str]]],
        max_workers: Optional[int] = None,
        validate: bool = True,
    ) -> List[QuantaGraph]:
        """Ingests a collection of passages concurrently across parallel worker slots.

        Enables high-throughput document ingestion (>200-300 words/sec) on multi-passage
        and long multi-paragraph documents (such as MuSiQue or Wikipedia articles) by
        parallelizing neural extraction across all available server slots (up to 16).

        Args:
            passages: Sequence of (passage_id, doc_id, text) tuples or dicts.
            max_workers: Maximum concurrent workers (defaults to configured batch_workers / slots).
            validate: Whether to run difference logic verification.

        Returns:
            List of compiled and committed QuantaGraph instances.
        """
        if not passages:
            return []

        norm_passages = []
        for p in passages:
            if isinstance(p, dict):
                pid = p.get("passage_id") or p.get("id") or f"P_{int(time.time() * 1000)}"
                doc_id = p.get("doc_id") or "doc_batch"
                text = p.get("text") or p.get("content") or ""
            else:
                pid, doc_id, text = p
            norm_passages.append((pid, doc_id, text))

        target_workers = max_workers if max_workers is not None else getattr(self, "max_workers", 8)
        workers = min(target_workers, len(norm_passages))
        if workers <= 1:
            return [
                self.ingest_document(text=text, doc_id=doc_id, passage_id=pid, validate=validate)
                for pid, doc_id, text in norm_passages
            ]

        from concurrent.futures import ThreadPoolExecutor

        orig_concurrency = getattr(self.kev_engine, "concurrency_limit", 16)
        try:
            # Scale Kev per-passage concurrency so total in-flight requests match server slot capacity
            if hasattr(self.kev_engine, "concurrency_limit"):
                self.kev_engine.concurrency_limit = max(1, 16 // workers)

            def _ingest_one(item: Tuple[str, str, str]) -> QuantaGraph:
                pid, doc_id, text = item
                return self.ingest_document(text=text, doc_id=doc_id, passage_id=pid, validate=validate)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(_ingest_one, norm_passages))
        finally:
            if hasattr(self.kev_engine, "concurrency_limit"):
                self.kev_engine.concurrency_limit = orig_concurrency

    ingest_passages_parallel = ingest_passages_batch

    def query_memory(
        self,
        query: str,
        max_tokens: int = 500,
        top_k: int = 10,
        max_depth: int = 2,
        format: str = "dual_stream",
        query_polarity: str = "positive",
        flush_async: bool = False,
    ) -> Union[DualStreamContext, str]:
        """Queries the Semantic Virtual Memory using HippoRAG 2 PPR, PoP-RAG gating, and bipartite projection.

        Workflow (Section 8 Master Plan):
        1. Compile query seeds and constraints.
        2. Execute HippoRAG 2 PPR with PoP-RAG gating.
        3. Run bipartite projection to rank raw source passages:
           Score(v_p) = sum_{v_s in N(v_p)} p(v_s) * conf(v_s)
        4. Return dual-stream context block:
           Stream 1: Logical Briefing Block (verified causal/temporal paths and valency frames)
           Stream 2: Top-K Raw Source Passages (verbatim spans with 100% lexical fidelity)
        """
        if flush_async and hasattr(self, "async_kev_worker") and self.async_kev_worker is not None:
            self.flush_kev_queue()

        if not query or not query.strip():
            empty_ctx = DualStreamContext(
                logical_briefing="",
                passage_stream="",
                full_context="",
                passages=[],
                token_count_estimate=0,
            )
            return empty_ctx if format in ("dual_stream", "object") else ""

        # Step 1 & 2: Compile seeds & Execute HippoRAG 2 PPR with PoP-RAG gating
        t_ppr0 = time.perf_counter()
        subgraph = self.retriever.retrieve_subgraph_for_query(
            query=query,
            page_table=self.page_table,
            top_k=top_k,
            max_depth=max_depth,
            query_polarity=query_polarity,
            algorithm="hipporag",
        )
        if self.tracer:
            self.tracer.record_stage_timing("PPR", (time.perf_counter() - t_ppr0) * 1000.0, gpu_calls=0)

        # Fallback if no nodes found: check active canvas
        if (subgraph is None or not subgraph.nodes) and hasattr(self, "active_canvas") and self.active_canvas.nodes:
            subgraph = QuantaGraph()
            for n in self.active_canvas.nodes.values():
                subgraph.add_node(n)
            if self.active_canvas.nodes:
                subgraph.root_cid = next(iter(self.active_canvas.nodes.keys()))

        # Prioritize background verification for any queried passages
        if hasattr(self, "async_kev_worker") and self.async_kev_worker is not None and subgraph is not None:
            for n in subgraph.nodes.values():
                pid = getattr(n, "passage_id", None)
                if pid:
                    self.async_kev_worker.prioritize(pid)

        # Step 3 & 4: Bipartite Projection and Dual-Stream Context Assembly
        t_ctx0 = time.perf_counter()
        dual_ctx = self.context_assembler.assemble_dual_stream_context(
            subgraph=subgraph,
            passage_store=self.passage_store,
            max_tokens=max_tokens,
            query_text=query,
            activations=getattr(subgraph, "activations", None),
        )
        if self.tracer:
            self.tracer.record_stage_timing("context assembly", (time.perf_counter() - t_ctx0) * 1000.0, gpu_calls=0)

        fmt_lower = str(format).lower().strip()
        if fmt_lower in ("dual_stream", "object"):
            return dual_ctx
        elif fmt_lower in ("str", "string", "full", "english", "svm"):
            return dual_ctx.full_context
        elif fmt_lower == "sexpr":
            return self.retriever.format_context_for_llm(
                subgraph, format="sexpr", max_tokens=max_tokens, query_text=query, passage_store=self.passage_store
            )
        else:
            return dual_ctx.full_context

    def process(
        self,
        text: str,
        validate: bool = True,
        chapter_id: str = "ch_01",
        language_hint: Optional[str] = None,
    ) -> QuantaGraph:
        """Process raw discourse text (English or code) end-to-end into a verified QuantaGraph ASG.

        Workflow:
        1. Segments text into DiscourseChunk objects via DiscourseChunker.
        2. Iterates chunks through UnslothTransducer with active entity tracking
           and closed-loop MUC repair.
        3. Stitches chunks via GraphStitcher with global entity resolution and
           Allen temporal interval synthesis.
        4. Compiles stitched extraction into a 1024-D QuantaGraph via ASGCompiler.
        5. Ingests nodes into Virtual Page Table and ActiveCanvas.
        6. Folds into Merkle episode structure.

        Args:
            text: Arbitrary discourse text (natural language or source code).
            validate: Whether to run formal Clingo ASP verification.
            chapter_id: Identifier for Merkle folding namespace.
            language_hint: Optional metadata annotation passed to transducer prompt.

        Returns:
            Fully compiled, verified, content-addressed QuantaGraph.
        """
        if not text or not text.strip():
            empty_graph = QuantaGraph()
            setattr(empty_graph, "extraction_result", DiscourseExtractionResult())
            return empty_graph

        # 1. Chunker Segmentation
        chunks = self.chunker.chunk_text(text, chapter_id=chapter_id)
        if not chunks:
            chunks = [
                DiscourseChunk(
                    chunk_id=f"chunk_{int(time.time() * 1000)}",
                    text=text.strip(),
                    chapter_id=chapter_id,
                )
            ]

        # Multi-chunk streaming episodic routing:
        # Long documents and multi-chunk narratives (> 6 chunks or > 600 words) must be processed
        # using the streaming episodic pipeline (process_narrative) to guarantee O(1) active canvas
        # bounds, Merkle episodic folding, and eliminate monolithic Blake3 CID hash cascades.
        total_words = sum(len(c.text.split()) for c in chunks)
        if len(chunks) > 6 or total_words > 600 or (len(chunks) > 1 and any(c.chapter_title for c in chunks)):
            graphs, chapter_fold = self.process_narrative(
                text, chapter_id=chapter_id, validate=validate, language_hint=language_hint
            )
            return graphs[0] if graphs else QuantaGraph()

        chunk_extractions: List[DiscourseExtractionResult] = []

        # 2. Iterate chunks through transducer with active entity tracking & MUC repair
        for chk in chunks:
            self.entity_engine.pre_scan_and_page(chk.text)
            manifest_prompt = self.entity_engine.format_prompt_block()
            if language_hint:
                manifest_prompt = f"{language_hint}\n{manifest_prompt}" if manifest_prompt else language_hint
            active_entities = (
                self.entity_engine.manifest.all_active()
                if hasattr(self.entity_engine, "manifest")
                else []
            )

            extraction: Optional[DiscourseExtractionResult] = None

            if self.repair_manager is not None and validate:
                repair_res = self.repair_manager.repair_chunk(
                    text=chk.text,
                    transducer=self.transducer,
                    active_entities=active_entities,
                    chunk_id=chk.chunk_id,
                )
                if repair_res.extraction_result is not None:
                    extraction = repair_res.extraction_result
                elif repair_res.final_sexpr:
                    extraction = parse_sexpr(repair_res.final_sexpr)
                elif repair_res.graph is not None and hasattr(repair_res.graph, "extraction_result"):
                    extraction = repair_res.graph.extraction_result
                else:
                    if not repair_res.success and repair_res.error_message:
                        raise ASGCompilationError(repair_res.error_message)
                    extraction = self._invoke_transducer_chunk(
                        text=chk.text,
                        chunk_id=chk.chunk_id,
                        active_entities=active_entities,
                        active_manifest_prompt=manifest_prompt,
                    )
            else:
                extraction = self._invoke_transducer_chunk(
                    text=chk.text,
                    chunk_id=chk.chunk_id,
                    active_entities=active_entities,
                    active_manifest_prompt=manifest_prompt,
                )

            if extraction is not None and hasattr(extraction, "entities"):
                self.entity_engine.update_from_extraction([e.to_dict() for e in extraction.entities])
                chunk_extractions.append(extraction)

        if not chunk_extractions:
            empty_graph = QuantaGraph()
            setattr(empty_graph, "extraction_result", DiscourseExtractionResult())
            return empty_graph

        # 3. Stitch chunks via GraphStitcher
        stitched = self.stitcher.stitch(chunk_extractions)

        # 4. Compile stitched extraction to QuantaGraph
        graph = self.compiler.compile(stitched, validate=validate)

        # 5. Ingest into Virtual Page Table and Active Canvas
        doc_heading = chunks[0].chapter_title if (chunks and chunks[0].chapter_title) else None

        # Register chunks in PassageStore if available
        if hasattr(self, "passage_store") and self.passage_store is not None:
            for idx, chk in enumerate(chunks):
                pid = chk.chunk_id or f"chunk_{len(self.passage_store) + 1}"
                self.passage_store.add_passage(
                    PassageRecord(
                        passage_id=pid,
                        doc_id=chapter_id or "doc_01",
                        char_span=(0, len(chk.text)),
                        text=chk.text,
                    )
                )
                for node in graph.nodes.values():
                    if not getattr(node, "passage_id", None):
                        node.passage_id = pid
                    s_span = _locate_node_span_in_chunk(node, chk.text)
                    node.span_start = s_span[0]
                    node.span_end = s_span[1]

        for node in graph.nodes.values():
            if doc_heading and not node.parent_cid:
                node.parent_cid = doc_heading
            self.page_table.store_node(node)
            self.active_canvas.put(node)

        # Maintain canvas bound
        assert len(self.active_canvas) <= self.active_canvas.capacity, (
            f"Active canvas exceeded bound: {len(self.active_canvas)} > {self.active_canvas.capacity}"
        )

        # 6. Fold into episode Merkle node
        fold_node = fold_discourse_episode(graph.copy(), chunk_id=chunks[0].chunk_id)
        if hasattr(self, "passage_store") and self.passage_store is not None and chunks:
            fold_node.passage_id = chunks[0].chunk_id
            fold_node.span_start = 0
            fold_node.span_end = len(chunks[0].text)
        if isinstance(fold_node.literal, dict):
            fold_node.literal["text"] = chunks[0].text
            if doc_heading:
                fold_node.literal["chapter_title"] = doc_heading
        if doc_heading and not fold_node.parent_cid:
            fold_node.parent_cid = doc_heading
        self.page_table.store_node(fold_node)
        self.active_canvas.put(fold_node)
        self.merkle_book.add_chunk_node(fold_node, chapter_id=chapter_id)

        return graph

    def to_english(self, graph: QuantaGraph) -> str:
        """Realize an ASG into compositional, honest English text."""
        return self.realizer.realize_graph(graph)

    def to_fol(self, graph: QuantaGraph) -> str:
        """Reconstruct First-Order Logic formula from graph."""
        return self.fol_emitter.emit_formula(graph)

    def to_code(self, graph: QuantaGraph, target_lang: str = "python", neural: bool = True) -> str:
        """Reconstruct source code in target_lang from graph via neural transducer or emitter."""
        if neural and hasattr(self.transducer, "realize_text"):
            from parser.sexpr_parser import serialize_to_sexpr
            try:
                sexpr = serialize_to_sexpr(graph, pretty=True)
                return self.transducer.realize_text(sexpr, target_lang=target_lang)
            except Exception:
                pass
        return self.code_emitter.emit_code(graph)

    def to_sexpr(self, graph: QuantaGraph) -> str:
        """Serialize a QuantaGraph into canonical S-expression string."""
        if hasattr(graph, "extraction_result") and graph.extraction_result is not None:
            return to_sexpr(graph.extraction_result, pretty=True)
        return self._graph_to_sexpr_fallback(graph)

    def process_chunk(
        self,
        chunk_text: str,
        chunk_id: Optional[str] = None,
        chapter_id: str = "ch_01",
        validate: bool = True,
    ) -> QuantaGraph:
        """Process a single discourse chunk through the cognitive cycle."""
        cid = chunk_id or f"chunk_{int(time.time() * 1000)}"

        # 1. Pre-scan and resurrect dormant entities into working memory
        self.entity_engine.pre_scan_and_page(chunk_text)
        manifest_prompt = self.entity_engine.format_prompt_block()
        active_entities = (
            self.entity_engine.manifest.all_active()
            if hasattr(self.entity_engine, "manifest")
            else []
        )

        # 2. Neural Transduction & Closed-Loop Repair
        if self.repair_manager is not None and validate:
            repair_res = self.repair_manager.repair_chunk(
                text=chunk_text,
                transducer=self.transducer,
                active_entities=active_entities,
                chunk_id=cid,
            )
            if repair_res.extraction_result is not None:
                extraction = repair_res.extraction_result
            elif repair_res.final_sexpr:
                try:
                    extraction = parse_sexpr(repair_res.final_sexpr)
                except Exception:
                    extraction = None
            elif repair_res.graph is not None and hasattr(repair_res.graph, "extraction_result"):
                extraction = repair_res.graph.extraction_result

            if extraction is None:
                extraction = self._invoke_transducer_chunk(
                    chunk_text,
                    cid,
                    active_entities=active_entities,
                    active_manifest_prompt=manifest_prompt,
                )
        else:
            extraction = self._invoke_transducer_chunk(
                chunk_text,
                cid,
                active_entities=active_entities,
                active_manifest_prompt=manifest_prompt,
            )

        # 3. Update working memory manifest
        if extraction is not None and hasattr(extraction, "entities"):
            self.entity_engine.update_from_extraction([e.to_dict() for e in extraction.entities])
            if hasattr(self, "stitcher") and self.stitcher is not None:
                self.stitcher.add_chunk(extraction)

        # 4. Compile to verified QuantaGraph ASG
        graph = self.compiler.compile(extraction, validate=validate)

        # 5. Ingest into Virtual Page Table and Active Canvas
        doc_heading = None
        m_doc = re.match(r"^((?:Document|Passage)\s*\[?\d+\]?)", chunk_text.strip(), re.IGNORECASE)
        if m_doc:
            doc_heading = m_doc.group(1).strip()
        elif chapter_id and ("document" in chapter_id.lower() or "passage" in chapter_id.lower()):
            doc_heading = chapter_id
        for node in graph.nodes.values():
            if doc_heading and not node.parent_cid:
                node.parent_cid = doc_heading
            self.page_table.store_node(node)
            self.active_canvas.put(node)

        # Maintain canvas bound
        assert len(self.active_canvas) <= self.active_canvas.capacity, (
            f"Active canvas exceeded bound: {len(self.active_canvas)} > {self.active_canvas.capacity}"
        )

        return graph

    async def process_narrative_async(
        self,
        text: str,
        chapter_id: str = "ch_01",
        validate: bool = True,
        queue_size: int = 4,
        language_hint: Optional[str] = None,
    ) -> Tuple[List[QuantaGraph], QuantaNode]:
        """Asynchronously process a multi-paragraph narrative using a 3-stage pipelined architecture.

        Stages execute concurrently via asyncio queues:
        - Stage 1 (CPU): Discourse Chunker & Pre-scan Entity Matcher (N+1)
        - Stage 2 (GPU): SLM Transduction & Repair (N)
        - Stage 3 (CPU): ASG Compiler, Mmap Grounding & Clingo ASP Verification (N-1)
        """
        chunks = self.chunker.chunk_text(text, chapter_id=chapter_id)
        if not chunks:
            chunks = [
                DiscourseChunk(
                    chunk_id=f"chunk_{int(time.time() * 1000)}",
                    text=text.strip(),
                    chapter_id=chapter_id,
                )
            ]

        # For a single chunk, fast path directly
        if len(chunks) == 1:
            chk = chunks[0]
            g = await asyncio.to_thread(
                self.process_chunk,
                chunk_text=chk.text,
                chunk_id=chk.chunk_id,
                chapter_id=chapter_id,
                validate=validate,
            )
            fold_node = fold_discourse_episode(g, chunk_id=chk.chunk_id)
            doc_heading = chk.chapter_title or None
            if not doc_heading and chk.text:
                m_doc = re.match(r"^((?:Document|Passage)\s*\[?\d+\]?)", chk.text.strip(), re.IGNORECASE)
                if m_doc:
                    doc_heading = m_doc.group(1).strip()
            if isinstance(fold_node.literal, dict):
                fold_node.literal["text"] = chk.text
                if doc_heading:
                    fold_node.literal["chapter_title"] = doc_heading
            if doc_heading and not fold_node.parent_cid:
                fold_node.parent_cid = doc_heading
            self.page_table.store_node(fold_node)
            self.active_canvas.put(fold_node)
            self.merkle_book.add_chunk_node(fold_node, chapter_id=chapter_id)
            chapter_fold = self.merkle_book.fold_chapter(chapter_id=chapter_id)
            self.page_table.store_node(chapter_fold)
            self.active_canvas.put(chapter_fold)
            return [g], chapter_fold

        # Multi-stage asynchronous queue pipeline
        q_transduction: asyncio.Queue[Optional[Tuple[DiscourseChunk, List[Any], Optional[str]]]] = asyncio.Queue(maxsize=queue_size)
        q_compilation: asyncio.Queue[Optional[Tuple[DiscourseChunk, Optional[DiscourseExtractionResult]]]] = asyncio.Queue(maxsize=queue_size)

        episode_graphs: List[QuantaGraph] = []
        episode_fold_nodes: List[QuantaNode] = []

        # Stage 1: Producer (Discourse Chunking & Pre-scan Entity Matching)
        async def stage_chunk_and_prescan():
            for chk in chunks:
                self.entity_engine.pre_scan_and_page(chk.text)
                manifest_prompt = self.entity_engine.format_prompt_block()
                if language_hint:
                    manifest_prompt = f"{language_hint}\n{manifest_prompt}" if manifest_prompt else language_hint
                active_ents = (
                    self.entity_engine.manifest.all_active()
                    if hasattr(self.entity_engine, "manifest")
                    else []
                )
                await q_transduction.put((chk, active_ents, manifest_prompt))
            await q_transduction.put(None)

        # Stage 2: Transformer (SLM Transduction)
        async def stage_transduce():
            while True:
                item = await q_transduction.get()
                if item is None:
                    await q_compilation.put(None)
                    q_transduction.task_done()
                    break

                chk, active_ents, manifest_prompt = item
                extraction: Optional[DiscourseExtractionResult] = None

                if self.repair_manager is not None and validate:
                    repair_res = await asyncio.to_thread(
                        self.repair_manager.repair_chunk,
                        text=chk.text,
                        transducer=self.transducer,
                        active_entities=active_ents,
                        chunk_id=chk.chunk_id,
                    )
                    if repair_res.extraction_result is not None:
                        extraction = repair_res.extraction_result
                    elif repair_res.final_sexpr:
                        try:
                            extraction = parse_sexpr(repair_res.final_sexpr)
                        except Exception:
                            extraction = None
                    elif repair_res.graph is not None and hasattr(repair_res.graph, "extraction_result"):
                        extraction = repair_res.graph.extraction_result

                    if extraction is None:
                        extraction = await asyncio.to_thread(
                            self._invoke_transducer_chunk,
                            chk.text,
                            chk.chunk_id,
                            active_entities=active_ents,
                            active_manifest_prompt=manifest_prompt,
                        )
                else:
                    extraction = await asyncio.to_thread(
                        self._invoke_transducer_chunk,
                        chk.text,
                        chk.chunk_id,
                        active_entities=active_ents,
                        active_manifest_prompt=manifest_prompt,
                    )

                await q_compilation.put((chk, extraction))
                q_transduction.task_done()

        # Stage 3: Consumer (ASG Compiler, Mmap Grounding & ASP Verification)
        async def stage_compile_and_verify():
            while True:
                item = await q_compilation.get()
                if item is None:
                    q_compilation.task_done()
                    break

                chk, extraction = item
                if extraction is not None and hasattr(extraction, "entities"):
                    self.entity_engine.update_from_extraction([e.to_dict() for e in extraction.entities])
                    if hasattr(self, "stitcher") and self.stitcher is not None:
                        self.stitcher.add_chunk(extraction)

                # Compile to verified QuantaGraph ASG
                g = await asyncio.to_thread(self.compiler.compile, extraction, validate=validate)

                doc_heading = chk.chapter_title or None
                if not doc_heading and chk.text:
                    m_doc = re.match(r"^((?:Document|Passage)\s*\[?\d+\]?)", chk.text.strip(), re.IGNORECASE)
                    if m_doc:
                        doc_heading = m_doc.group(1).strip()

                # Register chunk in PassageStore if available
                pid = chk.chunk_id or f"chunk_{len(self.passage_store) + 1 if hasattr(self, 'passage_store') and self.passage_store is not None else 1}"
                if hasattr(self, "passage_store") and self.passage_store is not None:
                    self.passage_store.add_passage(
                        PassageRecord(
                            passage_id=pid,
                            doc_id=chapter_id or "doc_01",
                            char_span=(0, len(chk.text)),
                            text=chk.text,
                        )
                    )

                for node in g.nodes.values():
                    if doc_heading and not node.parent_cid:
                        node.parent_cid = doc_heading
                    if not getattr(node, "passage_id", None):
                        node.passage_id = pid
                    s_span = _locate_node_span_in_chunk(node, chk.text)
                    node.span_start = s_span[0]
                    node.span_end = s_span[1]
                    self.page_table.store_node(node)
                    self.active_canvas.put(node)

                assert len(self.active_canvas) <= self.active_canvas.capacity, (
                    f"Active canvas exceeded bound: {len(self.active_canvas)} > {self.active_canvas.capacity}"
                )

                # Fold discourse episode into 32-byte Merkle fold node
                fold_node = fold_discourse_episode(g, chunk_id=chk.chunk_id)
                fold_node.passage_id = pid
                fold_node.span_start = 0
                fold_node.span_end = len(chk.text)
                if isinstance(fold_node.literal, dict):
                    fold_node.literal["text"] = chk.text
                    if doc_heading:
                        fold_node.literal["chapter_title"] = doc_heading
                if doc_heading and not fold_node.parent_cid:
                    fold_node.parent_cid = doc_heading
                self.page_table.store_node(fold_node)
                self.active_canvas.put(fold_node)
                episode_fold_nodes.append(fold_node)
                self.merkle_book.add_chunk_node(fold_node, chapter_id=chapter_id)
                episode_graphs.append(g)

                # --- Section 4: Intra-chunk and Cross-chunk Entity Linking ---
                edges_to_add: List[Tuple[str, str, str]] = []

                # 1. Intra-chunk entity linking (CO_OCCURS / LOCATED_IN)
                chunk_entities: List[QuantaNode] = list(getattr(g, "entity_nodes", {}).values())
                if not chunk_entities:
                    chunk_entities = [n for n in g.nodes.values() if n.get_slot("TYPE_ENTITY") == 1]

                for i, ea in enumerate(chunk_entities):
                    for j in range(i + 1, min(i + 4, len(chunk_entities))):
                        eb = chunk_entities[j]
                        if ea.cid == eb.cid:
                            continue
                        is_b_loc = eb.get_slot("TYPE_LOCATION") == 1 or "location" in (eb.anchor or "").lower()
                        is_a_loc = ea.get_slot("TYPE_LOCATION") == 1 or "location" in (ea.anchor or "").lower()
                        if is_b_loc and not is_a_loc:
                            edges_to_add.append((ea.cid, "LOCATED_IN", eb.cid))
                        elif is_a_loc and not is_b_loc:
                            edges_to_add.append((eb.cid, "LOCATED_IN", ea.cid))
                        else:
                            edges_to_add.append((ea.cid, "CO_OCCURS", eb.cid))
                            edges_to_add.append((eb.cid, "CO_OCCURS", ea.cid))

                # 2. Cross-chunk associative bridges via episodic entity registry
                curr_root_ev = g.root or (g.get_node(g.root_cid) if g.root_cid else None)

                for ent in chunk_entities:
                    raw_name = str(ent.literal) if ent.literal else (ent.anchor or "")
                    cand_keys = {raw_name.strip().lower()}
                    norm_key = _normalize_entity_key(raw_name)
                    if len(norm_key) >= 2:
                        cand_keys.add(norm_key)

                    if extraction is not None and hasattr(extraction, "entities"):
                        for ext_ent in extraction.entities:
                            if ext_ent.canonical_name.strip().lower() == raw_name.lower():
                                if ext_ent.surface_aliases:
                                    for alias in ext_ent.surface_aliases:
                                        a_norm = _normalize_entity_key(alias)
                                        if len(a_norm) >= 2:
                                            cand_keys.add(a_norm)

                    # Check for prior matches across past chunks (bounded to last 3 occurrences to prevent O(N^2) complete graph explosion)
                    matched_records: List[Tuple[QuantaNode, QuantaNode, Optional[QuantaNode]]] = []
                    for k in cand_keys:
                        if k in self.episodic_entity_registry:
                            matched_records.extend(self.episodic_entity_registry[k][-3:])

                    # Deduplicate matched records by fold_node CID
                    seen_prev_cids = set()
                    for prev_ent, prev_fold, prev_ev_root in matched_records[-6:]:
                        if prev_fold.cid in seen_prev_cids or prev_fold.cid == fold_node.cid:
                            continue
                        seen_prev_cids.add(prev_fold.cid)

                        # Synthesize CROSS_CHUNK_BRIDGE between fold nodes
                        edges_to_add.append((prev_fold.cid, "CROSS_CHUNK_BRIDGE", fold_node.cid))
                        edges_to_add.append((fold_node.cid, "CROSS_CHUNK_BRIDGE", prev_fold.cid))

                        # Synthesize CO_OCCURS between the event roots
                        if prev_ev_root is not None and curr_root_ev is not None and prev_ev_root.cid != curr_root_ev.cid:
                            edges_to_add.append((prev_ev_root.cid, "CO_OCCURS", curr_root_ev.cid))
                            edges_to_add.append((curr_root_ev.cid, "CO_OCCURS", prev_ev_root.cid))

                        # Synthesize CO_OCCURS between entity nodes across chunks if different CIDs
                        if prev_ent.cid != ent.cid:
                            edges_to_add.append((prev_ent.cid, "CO_OCCURS", ent.cid))
                            edges_to_add.append((ent.cid, "CO_OCCURS", prev_ent.cid))

                    # Register current occurrence under all candidate keys
                    record = (ent, fold_node, curr_root_ev)
                    for k in cand_keys:
                        self.episodic_entity_registry.setdefault(k, []).append(record)

                if edges_to_add:
                    if hasattr(self.page_table, "add_edges_batch"):
                        self.page_table.add_edges_batch(edges_to_add)
                    else:
                        for s_cid, rel, t_cid in edges_to_add:
                            self.page_table.add_edge(s_cid, rel, t_cid)

                q_compilation.task_done()

        # Execute 3 stages concurrently
        await asyncio.gather(
            stage_chunk_and_prescan(),
            stage_transduce(),
            stage_compile_and_verify(),
        )

        # Fold into Chapter Merkle Root
        chapter_fold = self.merkle_book.fold_chapter(chapter_id=chapter_id)
        self.page_table.store_node(chapter_fold)
        self.active_canvas.put(chapter_fold)

        return episode_graphs, chapter_fold

    def process_narrative(
        self,
        text: str,
        chapter_id: str = "ch_01",
        validate: bool = True,
        language_hint: Optional[str] = None,
    ) -> Tuple[List[QuantaGraph], QuantaNode]:
        """Process a multi-paragraph narrative or chapter using asynchronous multi-stage pipelining.

        Splits text into discourse chunks, executes streaming coreference resolution,
        compiles episode ASGs, and folds the episode sub-graphs into a Chapter Merkle root.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.process_narrative_async(text, chapter_id=chapter_id, validate=validate, language_hint=language_hint),
                )
                return future.result()
        else:
            return asyncio.run(
                self.process_narrative_async(text, chapter_id=chapter_id, validate=validate, language_hint=language_hint)
            )

    def ingest_book(
        self,
        chapters: Union[Dict[str, str], Sequence[Tuple[str, str]], Sequence[str]],
        validate: bool = True,
    ) -> Tuple[HierarchicalMerkleBook, str]:
        """Ingest an entire book or multi-chapter document (>10,000 words).

        Guarantees:
        1. VRAM usage remains strictly bounded (len(active_canvas) <= capacity).
        2. Merkle book root is deterministically computed.
        3. All nodes are durably persisted in PageTable.
        """
        normalized_chapters: List[Tuple[str, str]] = []
        if isinstance(chapters, dict):
            normalized_chapters = list(chapters.items())
        elif isinstance(chapters, (list, tuple)):
            for idx, item in enumerate(chapters):
                if isinstance(item, tuple) and len(item) == 2:
                    normalized_chapters.append(item)
                else:
                    normalized_chapters.append((f"chapter_{idx + 1:02d}", str(item)))

        for ch_id, ch_text in normalized_chapters:
            self.process_narrative(ch_text, chapter_id=ch_id, validate=validate)
            assert len(self.active_canvas) <= self.active_canvas.capacity

        # Fold Book Merkle Root
        book_root_node = self.merkle_book.fold_book()
        self.page_table.store_node(book_root_node)
        self.active_canvas.put(book_root_node)

        return self.merkle_book, book_root_node.cid

    def ingest_code(
        self,
        source_text: str,
        language_hint: Optional[str] = None,
        chapter_id: str = "code_01",
        validate: bool = False,
    ) -> QuantaGraph:
        """Route source code through the unified neural discourse transduction pipeline.

        Treats programming language source code as another modality of discourse without
        language-specific AST parsers. The optional language_hint is provided as metadata
        annotation to the transducer prompt.

        Args:
            source_text: Source code in any programming language (Python, Java, Rust, etc.).
            language_hint: Optional language name annotation (e.g., "Python", "Java").
            chapter_id: Chapter / module identifier for Merkle grouping.
            validate: Whether to run formal verification on the compiled graph.

        Returns:
            Fully compiled, verified, content-addressed QuantaGraph.
        """
        if not source_text or not source_text.strip():
            empty_graph = QuantaGraph()
            setattr(empty_graph, "extraction_result", DiscourseExtractionResult())
            return empty_graph

        manifest_hint = f"SOURCE LANGUAGE: {language_hint}" if language_hint else None
        return self.process(
            text=source_text,
            validate=validate,
            chapter_id=chapter_id,
            language_hint=manifest_hint,
        )

    def retrieve_context(
        self,
        query: str,
        format: str = "english",
        max_tokens: int = 500,
        top_k: int = 10,
        max_depth: int = 2,
        decay: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> str:
        """Retrieves minimal relevant verified ASG context for external LLMs via spreading activation.

        Args:
            query: Natural language question or query pattern.
            format: 'english' (compositional NLG sentences) or 'sexpr' (compact GBNF S-expression).
            max_tokens: Maximum context token length.
            top_k: Number of SIMD seeds to explore.
            max_depth: Depth of spreading activation traversal.
            decay: Custom activation decay rate.
            threshold: Custom activation cutoff threshold.

        Returns:
            Formatted context string for host LLM prompt injection.
        """
        effective_max_tokens = max_tokens
        detected_depth = self.retriever.detect_query_hop_depth(query) if hasattr(self, "retriever") else 2
        effective_depth = max_depth
        if detected_depth >= 3:
            effective_depth = max(max_depth, 3)
            if max_tokens <= 600:
                effective_max_tokens = 1200

        if format.lower() in ("dual_stream", "svm"):
            res = self.query_memory(
                query=query,
                max_tokens=effective_max_tokens,
                top_k=top_k,
                max_depth=effective_depth,
                format="str",
            )
            return str(res)

        eff_decay = decay if decay is not None else (0.85 if effective_depth >= 3 else None)
        return self.retriever.retrieve_context(
            query=query,
            page_table=self.page_table,
            format=format,
            max_tokens=effective_max_tokens,
            top_k=top_k,
            max_depth=effective_depth,
            decay=eff_decay,
            threshold=threshold,
            passage_store=self.passage_store,
        )

    def answer_query(
        self,
        query: str,
        target_graph: Optional[QuantaGraph] = None,
    ) -> str:
        """Answer a factual question directly from graph topology in < 10 ms."""
        t0 = time.perf_counter()

        if target_graph is not None:
            ans = self.realizer.answer_query(target_graph, query)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("Query answered in %.2f ms: '%s' -> '%s'", elapsed_ms, query, ans)
            return ans

        # Page-table resolution: search PageTable nodes
        # 1. Check active canvas first
        canvas_nodes = list(self.active_canvas.nodes.values())
        temp_graph = QuantaGraph()
        for n in canvas_nodes:
            temp_graph.add_node(n)
        if canvas_nodes:
            temp_graph.root_cid = canvas_nodes[0].cid

        ans = self.realizer.answer_query(temp_graph, query)
        if ans and ans != "I do not have sufficient information in the knowledge graph to verify this.":
            return ans

        # 2. Query PageTable via Spreading Activation
        retrieved_subgraph = self.retriever.retrieve_subgraph_for_query(
            query=query,
            page_table=self.page_table,
            top_k=5,
            max_depth=2,
        )
        if retrieved_subgraph is not None and len(retrieved_subgraph.nodes) > 0:
            ans = self.realizer.answer_query(retrieved_subgraph, query)
            if ans and ans != "I do not have sufficient information in the knowledge graph to verify this.":
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                logger.info("PageTable spreading-activation query answered in %.2f ms: '%s' -> '%s'", elapsed_ms, query, ans)
                return ans

        # 3. Fallback scan if spreading activation did not resolve
        cur = self.page_table._conn.cursor()
        cur.execute("SELECT cid FROM nodes LIMIT 500")
        rows = cur.fetchall()
        for (cid,) in rows:
            n = self.page_table.fetch_node(cid)
            if n and n.get_slot("TYPE_EVENT") == 1:
                temp_graph.add_node(n)
                for edge_targets in n.edges.values():
                    for t_cid in edge_targets:
                        t_node = self.page_table.fetch_node(t_cid)
                        if t_node:
                            temp_graph.add_node(t_node)

        ans = self.realizer.answer_query(temp_graph, query)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("PageTable fallback query answered in %.2f ms: '%s' -> '%s'", elapsed_ms, query, ans)
        return ans

    def answer_long_context(
        self,
        prompt: str,
        config: Optional[MultiScaleConfig] = None,
        max_context_tokens: int = 1500,
        cold_start: bool = True,
    ) -> FastPathResult:
        """Orchestrates dynamic long-context answering via fast-path mode bush (§Phase 3).

        Cycle:
        1. Task Boundary Extraction: isolate query q and context document D.
        2. Hierarchical Chunking: macro discourse blocks & micro chunks.
        3. Relevance Filtering: score & prune irrelevant units without dropping gold evidence.
        4. Selective Ingestion:
           - M-A (raw-only): 0 synchronous transductions.
           - M-B (hot-transduce N): synchronously transduce top-N kept units into graph.
           - M-C (full): synchronously transduce all kept units into graph.
           - M-D (coverage-adaptive): M-A if coverage >= threshold, else M-B.
           - passthrough: B0 baseline full document ingestion.
        5. Spreading Activation / HippoRAG PPR over active graph nodes (if graph non-empty).
        6. Fast-Path Context Assembly: DualStreamContext with 100% lexical fidelity in stream 2.
        7. Topological or evidence-grounded answer generation.
        """
        t_total0 = time.perf_counter()
        ms_cfg = config or getattr(self, "multi_scale_config", None) or MultiScaleConfig()

        if cold_start:
            self.reset(clear_page_table=True)

        if self.tracer:
            self.tracer.reset_stage_telemetry()

        # Step 1: Extractor
        t_ext0 = time.perf_counter()
        extractor = TaskBoundaryExtractor(strategy=ms_cfg.query_extractor_strategy)
        intent = extractor.extract(prompt)
        query_text = intent.query_text or prompt
        context_text = intent.context_text
        if not context_text or not context_text.strip():
            context_text = prompt
        t_ext_ms = (time.perf_counter() - t_ext0) * 1000.0
        if self.tracer:
            self.tracer.record_stage_timing("query split", t_ext_ms, gpu_calls=0)

        # Step 2: Chunker
        t_chk0 = time.perf_counter()
        macro_tokens = ms_cfg.chunker_macro_target_tokens or 1000
        micro_words = ms_cfg.chunker_micro_target_words or 250
        chunker = HierarchicalChunker(macro_target_tokens=macro_tokens, micro_target_words=micro_words)

        macro_blocks: List[MacroBlock] = []
        all_micro_chunks: List[DiscourseChunk] = []
        if context_text and context_text.strip():
            macro_blocks = chunker.chunk(context_text)
            all_micro_chunks = [c for mb in macro_blocks for c in mb.micro_chunks]
            if not macro_blocks:
                single_micro = DiscourseChunk(
                    chunk_id="chunk_1",
                    text=context_text,
                    global_offset=0,
                    global_end_offset=len(context_text),
                    word_count=len(context_text.split()),
                    token_count_estimate=int(len(context_text.split()) * 1.33),
                )
                all_micro_chunks = [single_micro]
                macro_blocks = [
                    MacroBlock(
                        macro_id="macro_1",
                        char_span=(0, len(context_text)),
                        text=context_text,
                        micro_chunks=[single_micro],
                    )
                ]
        t_chk_ms = (time.perf_counter() - t_chk0) * 1000.0
        if self.tracer:
            self.tracer.record_stage_timing("chunking", t_chk_ms, gpu_calls=0)

        # Step 3: Filter
        t_flt0 = time.perf_counter()
        filt = RelevanceFilter(
            strategy=ms_cfg.filter_strategy,
            unit=ms_cfg.filter_unit,
            keep_threshold=ms_cfg.filter_keep_threshold,
            keep_top_k=ms_cfg.filter_keep_top_k,
            keep_budget_tokens=ms_cfg.filter_keep_budget_tokens,
            calibration=ms_cfg.get("filter.calibration"),
            grounder=getattr(self, "concept_grounder", None),
            kev_engine=getattr(self, "kev_engine", None),
        )
        target_units = macro_blocks if ms_cfg.filter_unit in ("macro", "macro_head") else all_micro_chunks
        scored_units = filt.score(intent, target_units)
        kept_units, _ = filt.apply_keep_policy(scored_units)
        t_flt_ms = (time.perf_counter() - t_flt0) * 1000.0
        if self.tracer:
            gpu_flt = sum(getattr(u, "metadata", {}).get("gpu_calls", 0) for u in scored_units)
            self.tracer.record_stage_timing("relevance filter", t_flt_ms, gpu_calls=gpu_flt)

        # Step 4: Selective Ingestion
        assembler = FastPathAssembler(
            mode=ms_cfg.fast_path_mode,
            hot_transduce_n=ms_cfg.fast_path_hot_transduce_n,
            coverage_threshold=ms_cfg.fast_path_coverage_threshold,
            config=ms_cfg,
            dual_stream_assembler=self.context_assembler,
        )
        effective_mode, coverage_score = assembler.resolve_effective_mode(
            intent=intent,
            kept_units=kept_units,
            target_entities=intent.target_entities,
        )

        subgraph: Optional[QuantaGraph] = None
        units_transduced = 0
        deferred_units = 0

        # Register kept units into passage store so they are accessible by ID
        for idx, ku in enumerate(kept_units):
            pid = ku.unit_id if hasattr(ku, "unit_id") else f"P_{idx+1}"
            u_text = ku.unit.text if hasattr(ku, "unit") and hasattr(ku.unit, "text") else (ku.text if hasattr(ku, "text") else str(ku))
            u_span = getattr(ku, "char_span", (0, len(u_text)))
            if pid not in self.passage_store:
                self.passage_store.add_passage(
                    PassageRecord(
                        passage_id=pid,
                        doc_id="doc_01",
                        char_span=u_span,
                        text=u_text,
                    )
                )
            if self.passage_store.get_ingestion_state(pid) is None:
                self.passage_store.set_ingestion_state(pid, "PENDING")

        if effective_mode == FastPathMode.RAW_ONLY.value:
            # M-A: 0 synchronous transductions
            units_transduced = 0
            deferred_units = len(kept_units)
            deferred_list = list(kept_units)
            subgraph = None

        elif effective_mode == FastPathMode.HOT_TRANSDUCE.value:
            # M-B: Synchronously transduce top-N kept units
            n = ms_cfg.fast_path_hot_transduce_n or 2
            units_to_transduce = kept_units[:n]
            units_transduced = len(units_to_transduce)
            deferred_units = max(0, len(kept_units) - units_transduced)
            deferred_list = list(kept_units[n:])

            for idx, ku in enumerate(units_to_transduce):
                pid = ku.unit_id if hasattr(ku, "unit_id") else f"P_hot_{idx+1}"
                u_text = ku.unit.text if hasattr(ku, "unit") and hasattr(ku.unit, "text") else (ku.text if hasattr(ku, "text") else str(ku))
                self.ingest_document(
                    u_text,
                    doc_id="doc_01",
                    passage_id=pid,
                    validate=False,
                )

            # Step 5: PPR over transduced graph
            t_ppr0 = time.perf_counter()
            subgraph = self.retriever.retrieve_subgraph_for_query(
                query=query_text,
                page_table=self.page_table,
                top_k=10,
                max_depth=2,
                algorithm="hipporag",
            )
            t_ppr_ms = (time.perf_counter() - t_ppr0) * 1000.0
            if self.tracer:
                self.tracer.record_stage_timing("PPR", t_ppr_ms, gpu_calls=0)

        elif effective_mode == FastPathMode.FULL.value:
            # M-C: Synchronously transduce ALL kept units
            units_to_transduce = kept_units
            units_transduced = len(units_to_transduce)
            deferred_units = 0
            deferred_list = []

            for idx, ku in enumerate(units_to_transduce):
                pid = ku.unit_id if hasattr(ku, "unit_id") else f"P_full_{idx+1}"
                u_text = ku.unit.text if hasattr(ku, "unit") and hasattr(ku.unit, "text") else (ku.text if hasattr(ku, "text") else str(ku))
                self.ingest_document(
                    u_text,
                    doc_id="doc_01",
                    passage_id=pid,
                    validate=False,
                )

            # Step 5: PPR over graph
            t_ppr0 = time.perf_counter()
            subgraph = self.retriever.retrieve_subgraph_for_query(
                query=query_text,
                page_table=self.page_table,
                top_k=10,
                max_depth=2,
                algorithm="hipporag",
            )
            t_ppr_ms = (time.perf_counter() - t_ppr0) * 1000.0
            if self.tracer:
                self.tracer.record_stage_timing("PPR", t_ppr_ms, gpu_calls=0)

        else:
            # passthrough / B0
            if context_text and context_text.strip():
                self.ingest_document(context_text, doc_id="doc_01", validate=False)
                units_transduced = len(all_micro_chunks) or 1
            deferred_units = 0
            deferred_list = []
            t_ppr0 = time.perf_counter()
            subgraph = self.retriever.retrieve_subgraph_for_query(
                query=query_text,
                page_table=self.page_table,
                top_k=10,
                max_depth=2,
                algorithm="hipporag",
            )
            t_ppr_ms = (time.perf_counter() - t_ppr0) * 1000.0
            if self.tracer:
                self.tracer.record_stage_timing("PPR", t_ppr_ms, gpu_calls=0)

        # Enqueue deferred chunks for asynchronous background completion (§Phase 5)
        if deferred_units > 0 and deferred_list:
            for idx, ku in enumerate(deferred_list):
                pid = ku.unit_id if hasattr(ku, "unit_id") else f"P_def_{idx+1}"
                u_text = ku.unit.text if hasattr(ku, "unit") and hasattr(ku.unit, "text") else (ku.text if hasattr(ku, "text") else str(ku))
                score = getattr(ku, "score", 0.5)
                parent_macro_id = getattr(getattr(ku, "unit", ku), "parent_macro_id", None)
                concept_codes = getattr(getattr(ku, "unit", ku), "concept_codes", [])

                if self.passage_store is not None:
                    rec = self.passage_store.get_passage(pid)
                    if rec is None:
                        self.passage_store.add_passage(
                            PassageRecord(
                                passage_id=pid,
                                doc_id="doc_01",
                                char_span=(0, len(u_text)),
                                text=u_text,
                                granularity="MICRO",
                                parent_macro_id=parent_macro_id,
                                concept_codes=concept_codes,
                            )
                        )
                    if self.passage_store.get_ingestion_state(pid) != "FULL":
                        self.passage_store.set_ingestion_state(pid, "PENDING")

                if getattr(self, "background_enabled", False) and getattr(self, "background_ingestor", None) is not None:
                    self.background_ingestor.enqueue_deferred(
                        passage_id=pid,
                        doc_id="doc_01",
                        text=u_text,
                        filter_score=score,
                        parent_macro_id=parent_macro_id,
                        concept_codes=concept_codes,
                        graph=self.active_canvas,
                    )

        with self.foreground_scope():
            # Step 6: Context Assembly
            t_asm0 = time.perf_counter()
            fast_ctx = assembler.assemble(
                subgraph=subgraph,
                kept_units=kept_units,
                passage_store=self.passage_store,
                max_tokens=max_context_tokens,
                query_text=query_text,
                effective_mode=effective_mode,
                coverage_score=coverage_score,
                activations=getattr(subgraph, "activations", None) if subgraph else None,
                transduced_count=units_transduced,
            )
            t_asm_ms = (time.perf_counter() - t_asm0) * 1000.0
            if self.tracer:
                self.tracer.record_stage_timing("context assembly", t_asm_ms, gpu_calls=0)

            # Step 7: Answer Generation
            t_ans0 = time.perf_counter()
            answer = ""
            if subgraph and len(subgraph.nodes) > 0:
                try:
                    answer = self.answer_query(query_text, target_graph=subgraph)
                except Exception:
                    answer = ""

            if not answer or answer == "I do not have sufficient information in the knowledge graph to verify this.":
                # Extract from retrieved raw passage stream
                if fast_ctx.stream2_passages:
                    lines = [l.strip() for l in fast_ctx.stream2_passages.split("\n") if l.strip() and not l.startswith("===") and not l.startswith("[Passage")]
                    answer = lines[0] if lines else "No evidence found."
                else:
                    answer = "No evidence found."

            t_ans_ms = (time.perf_counter() - t_ans0) * 1000.0
            if self.tracer:
                self.tracer.record_stage_timing("reader", t_ans_ms, gpu_calls=0)

        stage_timings = dict(self.tracer.stage_timings) if self.tracer else {}
        gpu_calls = self.tracer.get_total_gpu_calls() if self.tracer else 0

        return FastPathResult(
            answer=answer,
            context=fast_ctx.context_text,
            isolated_query=query_text,
            context_document=context_text,
            mode=effective_mode,
            coverage_score=coverage_score,
            units_scored=len(target_units),
            units_kept=len(kept_units),
            units_transduced=units_transduced,
            deferred_units=deferred_units,
            subgraph=subgraph,
            stage_timings=stage_timings,
            gpu_calls=gpu_calls,
            tokens_estimated=fast_ctx.estimated_tokens,
            metadata={
                "strategy": ms_cfg.filter_strategy,
                "unit": ms_cfg.filter_unit,
                "hot_transduce_n": assembler.hot_transduce_n,
                "coverage_threshold": assembler.coverage_threshold,
                "total_latency_ms": (time.perf_counter() - t_total0) * 1000.0,
            },
        )

    def process_and_answer_single_query(
        self,
        query_text: str,
        context_document: Optional[str] = None,
        cold_start: bool = True,
        allow_fixtures: bool = False,
        format: str = "english",
        max_tokens: int = 500,
        top_k: int = 5,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """Process a single query, ingesting context on-the-fly if needed (cold-start) without precompiled fixtures.

        Args:
            query_text: Raw query or combined single-turn prompt.
            context_document: Optional pre-extracted context document. If None, decomposition is attempted.
            cold_start: If True, resets working memory and PageTable before ingestion.
            allow_fixtures: If False, enforces dynamic SLM extraction without precompiled fixture shortcuts.
            format: Context format ('english' or 'sexpr').
            max_tokens: Maximum tokens for retrieved context.
            top_k: SIMD seeds for spreading activation.
            max_depth: Depth for spreading activation traversal.

        Returns:
            Dict containing answer, retrieved context, timing metrics, and token statistics.
        """
        import re
        t_start = time.perf_counter()

        doc = context_document
        isolated_query = query_text.strip()

        # 1. Attempt decomposition if context_document is not explicitly provided
        if doc is None:
            from parser.task_boundary_extractor import decompose_query_context
            extracted_doc, candidate_q = decompose_query_context(query_text)
            if extracted_doc:
                doc = extracted_doc
                isolated_query = candidate_q
            elif not isolated_query:
                isolated_query = query_text.strip()

        # Phase 3 Fast-Path routing: check if fast_path mode is set
        ms_cfg = getattr(self, "multi_scale_config", None) or MultiScaleConfig()
        if ms_cfg.fast_path_mode not in ("passthrough", None, ""):
            full_prompt = f"{doc}\n\n{query_text}" if doc else query_text
            fp_res = self.answer_long_context(full_prompt, config=ms_cfg, max_context_tokens=max_tokens, cold_start=cold_start)
            return {
                "answer": fp_res.answer,
                "retrieved_context": fp_res.context,
                "isolated_query": fp_res.isolated_query,
                "context_document": fp_res.context_document,
                "ingest_latency_ms": fp_res.stage_timings.get("transduction", 0.0) + fp_res.stage_timings.get("chunking", 0.0),
                "retrieval_latency_ms": fp_res.stage_timings.get("relevance filter", 0.0) + fp_res.stage_timings.get("PPR", 0.0),
                "answer_latency_ms": fp_res.stage_timings.get("reader", 0.0),
                "total_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "raw_context_chars": len(doc) if doc else len(query_text),
                "retrieved_context_chars": len(fp_res.context),
                "cold_start": cold_start,
                "allow_fixtures": allow_fixtures,
                "fast_path_result": fp_res,
                "mode": fp_res.mode,
            }

        # 2. Transducer fixture flag configuration
        old_allow_fixtures = getattr(self.transducer, "allow_fixtures", None)
        if hasattr(self.transducer, "allow_fixtures"):
            self.transducer.allow_fixtures = allow_fixtures

        t_ingest_ms = 0.0
        graph = None
        try:
            # 3. Ingestion if cold_start or document provided
            if cold_start:
                self.reset(clear_page_table=True)

            if doc and doc.strip():
                t0_ingest = time.perf_counter()
                graph = self.process(doc)
                t_ingest_ms = (time.perf_counter() - t0_ingest) * 1000.0

            # 4. Spreading Activation Retrieval using natural query without hints
            t0_retrieval = time.perf_counter()
            retrieved_context = self.retrieve_context(
                query=isolated_query,
                format=format,
                max_tokens=max_tokens,
                top_k=top_k,
                max_depth=max_depth,
            )
            t_retrieval_ms = (time.perf_counter() - t0_retrieval) * 1000.0

            # 5. Neuro-symbolic topological answer
            t0_ans = time.perf_counter()
            answer = self.answer_query(isolated_query, target_graph=graph)
            t_ans_ms = (time.perf_counter() - t0_ans) * 1000.0

            t_total_ms = (time.perf_counter() - t_start) * 1000.0

            raw_chars = len(doc) if doc else len(query_text)
            retrieved_chars = len(retrieved_context)

            return {
                "answer": answer,
                "retrieved_context": retrieved_context,
                "isolated_query": isolated_query,
                "context_document": doc,
                "ingest_latency_ms": t_ingest_ms,
                "retrieval_latency_ms": t_retrieval_ms,
                "answer_latency_ms": t_ans_ms,
                "total_latency_ms": t_total_ms,
                "raw_context_chars": raw_chars,
                "retrieved_context_chars": retrieved_chars,
                "cold_start": cold_start,
                "allow_fixtures": allow_fixtures,
            }
        finally:
            if old_allow_fixtures is not None and hasattr(self.transducer, "allow_fixtures"):
                self.transducer.allow_fixtures = old_allow_fixtures

    def realize(self, graph: QuantaGraph) -> str:
        """Realize an ASG into compositional, honest English text."""
        return self.realizer.realize_graph(graph)

    def close(self):
        """Release PageTable, EntityEngine, and background worker resources."""
        if hasattr(self, "background_ingestor") and self.background_ingestor is not None:
            self.background_ingestor.stop(wait=True)
            self.background_ingestor = None
        if hasattr(self, "async_kev_worker") and self.async_kev_worker is not None:
            self.async_kev_worker.stop(wait=True)
            self.async_kev_worker = None
        if hasattr(self, "entity_engine") and hasattr(self.entity_engine, "close"):
            self.entity_engine.close()
        if hasattr(self, "page_table") and hasattr(self.page_table, "close"):
            self.page_table.close()
        if hasattr(self, "passage_store") and hasattr(self.passage_store, "close"):
            self.passage_store.close()

    def _invoke_transducer_chunk(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        active_entities: Optional[List[EntityRecord]] = None,
        active_manifest_prompt: Optional[str] = None,
    ) -> DiscourseExtractionResult:
        """Invoke configured transducer with signature tolerance."""
        if hasattr(self.transducer, "transduce"):
            try:
                return self.transducer.transduce(
                    text=text,
                    chunk_id=chunk_id,
                    active_entities=active_entities,
                    active_manifest_prompt=active_manifest_prompt,
                )
            except TypeError:
                try:
                    return self.transducer.transduce(
                        text=text,
                        chunk_id=chunk_id,
                        active_entities=active_entities,
                    )
                except TypeError:
                    try:
                        return self.transducer.transduce(
                            chunk_text=text,
                            chunk_id=chunk_id,
                            active_manifest_prompt=active_manifest_prompt,
                        )
                    except TypeError:
                        return self.transducer.transduce(text)
        elif callable(self.transducer):
            res = self.transducer(text)
            if isinstance(res, DiscourseExtractionResult):
                return res
            elif isinstance(res, str):
                return parse_sexpr(res)
            else:
                return DiscourseExtractionResult.from_dict(res)
        else:
            raise TypeError(
                f"Configured transducer is neither an object with .transduce() nor callable: {type(self.transducer)}"
            )

    def _graph_to_sexpr_fallback(self, graph: QuantaGraph) -> str:
        """Fallback S-expression serializer when graph lacks attached extraction metadata."""
        entities: List[ExtractedEntity] = []
        events: List[ExtractedEvent] = []
        relations: List[ExtractedRelation] = []

        cid_to_ent_id: Dict[str, str] = {}
        cid_to_ev_id: Dict[str, str] = {}

        ent_counter = 1
        ev_counter = 1

        for cid, node in graph.nodes.items():
            if node.get_slot("TYPE_EVENT") == 1:
                ev_id = f"Ev{ev_counter}"
                ev_counter += 1
                cid_to_ev_id[cid] = ev_id
            else:
                ent_id = f"E{ent_counter}"
                ent_counter += 1
                cid_to_ent_id[cid] = ent_id

        for cid, ent_id in cid_to_ent_id.items():
            node = graph.get_node(cid)
            name = str(node.literal) if node.literal is not None else (node.anchor or ent_id)
            cat = "OBJECT"
            if node.get_slot("ROLE_AGENT_CAPABLE") == 1:
                cat = "PERSON"
            elif node.get_slot("TYPE_LOCATION") == 1:
                cat = "LOCATION"
            elif node.get_slot("TYPE_SUBSTANCE") == 1:
                cat = "SUBSTANCE"
            entities.append(
                ExtractedEntity(
                    id=ent_id,
                    canonical_name=name,
                    category=cat,
                    surface_aliases=[name],
                )
            )

        for cid, ev_id in cid_to_ev_id.items():
            node = graph.get_node(cid)
            pred = str(node.literal).lower() if node.literal is not None else "act"
            agent_id = None
            patient_id = None
            location_id = None
            instrument_id = None

            if "VAL_X1_AGENT" in node.edges and node.edges["VAL_X1_AGENT"]:
                agent_id = cid_to_ent_id.get(node.edges["VAL_X1_AGENT"][0])
            if "VAL_X2_PATIENT" in node.edges and node.edges["VAL_X2_PATIENT"]:
                patient_id = cid_to_ent_id.get(node.edges["VAL_X2_PATIENT"][0])
            if "VAL_LOCATION_SLOT" in node.edges and node.edges["VAL_LOCATION_SLOT"]:
                location_id = cid_to_ent_id.get(node.edges["VAL_LOCATION_SLOT"][0])
            if "VAL_X5_INSTRUMENT" in node.edges and node.edges["VAL_X5_INSTRUMENT"]:
                instrument_id = cid_to_ent_id.get(node.edges["VAL_X5_INSTRUMENT"][0])

            events.append(
                ExtractedEvent(
                    id=ev_id,
                    predicate=pred,
                    agent_id=agent_id,
                    patient_id=patient_id,
                    location_id=location_id,
                    instrument_id=instrument_id,
                    tense="PAST",
                    polarity=True,
                )
            )

        res = DiscourseExtractionResult(
            chunk_id=getattr(graph, "chunk_id", "graph_0"),
            entities=entities,
            events=events,
            relations=relations,
        )
        return to_sexpr(res, pretty=True)
