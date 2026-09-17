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
        elif transducer_backend in ("unsloth", "mock_unsloth", "qwen", "gemma"):
            self.transducer = UnslothTransducer(fallback_to_mock=True, **transducer_kwargs)
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
        self.fault_handler = SemanticPageFaultHandler(
            canvas=self.active_canvas,
            page_table=self.page_table,
        )

        # 8. Realizers & Emitters
        self.realizer = realizer or EnglishRealizer()
        self.fol_emitter = fol_emitter or FOLEmitter()
        self.code_emitter = code_emitter or CodeEmitter()
        self.query_answerer = GraphQueryAnswerer(realizer=self.realizer)

        # 9. Merkle Book State
        self.merkle_book = HierarchicalMerkleBook()

    def process(
        self,
        text: str,
        validate: bool = True,
        chapter_id: str = "ch_01",
    ) -> QuantaGraph:
        """Process raw English text end-to-end into a verified QuantaGraph ASG.

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
            text: Arbitrary English text (single sentence, paragraph, or document).
            validate: Whether to run formal Clingo ASP verification.
            chapter_id: Identifier for Merkle folding namespace.

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

        chunk_extractions: List[DiscourseExtractionResult] = []

        # 2. Iterate chunks through transducer with active entity tracking & MUC repair
        for chk in chunks:
            self.entity_engine.pre_scan_and_page(chk.text)
            manifest_prompt = self.entity_engine.format_prompt_block()
            active_entities = (
                self.entity_engine.manifest.all_active()
                if hasattr(self.entity_engine, "manifest")
                else []
            )

            extraction: Optional[DiscourseExtractionResult] = None

            if self.repair_manager is not None:
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

    def to_code(self, graph: QuantaGraph) -> str:
        """Reconstruct executable Python code from graph."""
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
        if self.repair_manager is not None:
            repair_res = self.repair_manager.repair_chunk(
                text=chunk_text,
                transducer=self.transducer,
                active_entities=active_entities,
                chunk_id=cid,
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

    def process_narrative(
        self,
        text: str,
        chapter_id: str = "ch_01",
        validate: bool = True,
    ) -> Tuple[List[QuantaGraph], QuantaNode]:
        """Process a multi-paragraph narrative or chapter.

        Splits text into discourse chunks, executes streaming coreference resolution,
        compiles episode ASGs, and folds the episode sub-graphs into a Chapter Merkle root.
        """
        chunks = self.chunker.chunk_text(text, chapter_id=chapter_id)
        episode_graphs: List[QuantaGraph] = []
        episode_fold_nodes: List[QuantaNode] = []

        for chk in chunks:
            g = self.process_chunk(
                chunk_text=chk.text,
                chunk_id=chk.chunk_id,
                chapter_id=chapter_id,
                validate=validate,
            )
            episode_graphs.append(g)

            # Fold discourse episode into 32-byte Merkle fold node
            fold_node = fold_discourse_episode(g, chunk_id=chk.chunk_id)
            self.page_table.store_node(fold_node)
            self.active_canvas.put(fold_node)
            episode_fold_nodes.append(fold_node)
            self.merkle_book.add_chunk_node(fold_node, chapter_id=chapter_id)

        # Fold into Chapter Merkle Root
        chapter_fold = self.merkle_book.fold_chapter(chapter_id=chapter_id)
        self.page_table.store_node(chapter_fold)
        self.active_canvas.put(chapter_fold)

        return episode_graphs, chapter_fold

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

        # 2. Query PageTable if not in active canvas
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
        logger.info("PageTable query answered in %.2f ms: '%s' -> '%s'", elapsed_ms, query, ans)
        return ans

    def realize(self, graph: QuantaGraph) -> str:
        """Realize an ASG into compositional, honest English text."""
        return self.realizer.realize_graph(graph)

    def reset(self):
        """Reset working memory entity manifest, active canvas, stitcher, and Merkle book."""
        self.entity_engine.reset()
        self.active_canvas.clear()
        if hasattr(self, "stitcher") and self.stitcher is not None:
            self.stitcher.reset()
        self.merkle_book = HierarchicalMerkleBook()

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
