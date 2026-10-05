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
import logging
from pathlib import Path
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

logger = logging.getLogger("quanta.pipeline.cognitive")


def _normalize_entity_key(name: str) -> str:
    """Normalizes entity surface strings for cross-chunk canonical matching."""
    k = name.strip().lower()
    for prefix in ("the ", "a ", "an "):
        if k.startswith(prefix):
            k = k[len(prefix):].strip()
            break
    return k


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
        **transducer_kwargs,
    ):
        """Initialize the unified Neuro-Symbolic Cognitive Pipeline."""
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

        # 9. Spreading-Activation Context Retriever
        self.retriever = SpreadingActivationRetriever(realizer=self.realizer)

        # 10. Merkle Book State
        self.merkle_book = HierarchicalMerkleBook()

    def reset(self, clear_page_table: bool = True) -> None:
        """Cleanly resets all working memory, active canvas, page table, and episodic state."""
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
        if hasattr(self, "fault_handler") and self.fault_handler is not None:
            self.fault_handler.page_table = self.page_table
            self.fault_handler.canvas = self.active_canvas
        if hasattr(self, "episodic_entity_registry"):
            self.episodic_entity_registry.clear()

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
        if len(chunks) > 6 or total_words > 600:
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
        for node in graph.nodes.values():
            self.page_table.store_node(node)
            self.active_canvas.put(node)

        # Maintain canvas bound
        assert len(self.active_canvas) <= self.active_canvas.capacity, (
            f"Active canvas exceeded bound: {len(self.active_canvas)} > {self.active_canvas.capacity}"
        )

        # 6. Fold into episode Merkle node
        fold_node = fold_discourse_episode(graph.copy(), chunk_id=chunks[0].chunk_id)
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
        for node in graph.nodes.values():
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

                for node in g.nodes.values():
                    self.page_table.store_node(node)
                    self.active_canvas.put(node)

                assert len(self.active_canvas) <= self.active_canvas.capacity, (
                    f"Active canvas exceeded bound: {len(self.active_canvas)} > {self.active_canvas.capacity}"
                )

                # Fold discourse episode into 32-byte Merkle fold node
                fold_node = fold_discourse_episode(g, chunk_id=chk.chunk_id)
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
        if max_tokens <= 600:
            detected_depth = self.retriever.detect_query_hop_depth(query)
            if detected_depth >= 3:
                effective_max_tokens = 1200

        return self.retriever.retrieve_context(
            query=query,
            page_table=self.page_table,
            format=format,
            max_tokens=effective_max_tokens,
            top_k=top_k,
            max_depth=max_depth,
            decay=decay,
            threshold=threshold,
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
            from server.proxy import decompose_query_context
            extracted_doc, candidate_q = decompose_query_context(query_text)
            if extracted_doc:
                doc = extracted_doc
                isolated_query = candidate_q
            elif not isolated_query:
                isolated_query = query_text.strip()

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
        """Release PageTable and EntityEngine resources."""
        self.entity_engine.close()
        self.page_table.close()

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
