"""End-to-End Neuro-Symbolic Cognitive Pipeline for QUANTA (Phase 8).

Integrates the full cognitive cycle across all subsystems:
1. Streaming Discourse Chunker (Phase 1)
2. Active Entity Manifest & Paging Engine (Phase 2)
3. Neural Discourse Transducer (Phase 3: Qwen-4B / LM Studio / Mock)
4. Symbolic ASG Compiler (Phase 4: ConceptNet 5.7.0, 1024-d quaternary vectors, BLAKE3 CIDs)
5. Clingo ASP Validation Gate (Phase 4.5: MUC extraction & structural verification)
6. Hierarchical Merkle Sub-Graph Folding (Phase 5: Book-scale cryptographic folding)
7. Virtual Page-Table Attention (Phase 6: O(1) VRAM execution, M <= 512 active canvas)
8. Honest Semantic Realizer & Zero-Attention Query Answering (Phase 7: Compositional NLG)
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
from parser.asg_compiler import ASGCompiler
from parser.chunker import DiscourseChunk, DiscourseChunker
from parser.entity_manifest import EntityPagingEngine
from parser.schema import DiscourseExtractionResult
from parser.transducer import BaseDiscourseTransducer, create_transducer
from realizer.english_nlg import EnglishRealizer, GraphQueryAnswerer
from solver.validator_gate import ValidationGate, ValidationResult

logger = logging.getLogger("quanta.pipeline.cognitive")


class CognitivePipeline:
    """Production-grade Neuro-Symbolic Cognitive Pipeline for QUANTA.

    Executes streaming context ingestion, working memory coreference tracking,
    discrete quaternary ASG compilation, Clingo ASP validation, hierarchical
    Merkle folding, bounded active canvas memory execution (M <= 512), and honest NLG.
    """

    def __init__(
        self,
        transducer_backend: str = "auto",
        canvas_capacity: int = 512,
        page_table_path: Optional[Union[str, Path]] = None,
        transducer: Optional[BaseDiscourseTransducer] = None,
        compiler: Optional[ASGCompiler] = None,
        chunker: Optional[DiscourseChunker] = None,
        entity_engine: Optional[EntityPagingEngine] = None,
        realizer: Optional[EnglishRealizer] = None,
        validator: Optional[ValidationGate] = None,
        **transducer_kwargs,
    ):
        """Initialize the complete Neuro-Symbolic Cognitive Pipeline."""
        # 1. Chunker & Segmentation
        self.chunker = chunker or DiscourseChunker(min_words=50, max_words=400)

        # 2. Working Memory Entity Manifest & Paging Engine
        self.entity_engine = entity_engine or EntityPagingEngine(target_size=10, min_size=5, max_size=15)

        # 3. Neural Discourse Transducer
        if transducer is not None:
            self.transducer = transducer
        else:
            self.transducer = create_transducer(backend=transducer_backend, **transducer_kwargs)

        # 4. Symbolic ASG Compiler
        self.compiler = compiler or ASGCompiler(validator_gate=validator)
        self.validator = self.compiler.validator_gate

        # 5. Virtual Memory Page Table & Active Canvas (Strict O(1) Physical VRAM Bound)
        self.page_table = PageTable(db_path=page_table_path)
        self.active_canvas = ActiveCanvas(capacity=canvas_capacity)
        self.fault_handler = SemanticPageFaultHandler(
            canvas=self.active_canvas,
            page_table=self.page_table,
        )

        # 6. Honest Semantic Realizer & Query Answerer
        self.realizer = realizer or EnglishRealizer()
        self.query_answerer = GraphQueryAnswerer(realizer=self.realizer)

        # 7. Merkle Book State
        self.merkle_book = HierarchicalMerkleBook()

    def process_chunk(
        self,
        chunk_text: str,
        chunk_id: Optional[str] = None,
        chapter_id: str = "ch_01",
        validate: bool = True,
    ) -> QuantaGraph:
        """Process a single discourse chunk through the cognitive cycle.

        Steps:
        1. Fast regex pre-scan pages dormant entities into active manifest (< 1 ms).
        2. Injects active manifest prompt block into Neural Transducer.
        3. Transducer extracts normalized intermediate JSON.
        4. Entity manifest updates recency and registers newly discovered entities.
        5. ASG Compiler converts JSON into 1024-d quaternary nodes with BLAKE3 CIDs.
        6. Clingo ASP verifies ontological and temporal consistency.
        7. Nodes are ingested into PageTable and loaded into ActiveCanvas (O(1) bound).
        """
        cid = chunk_id or f"chunk_{int(time.time() * 1000)}"

        # 1. Pre-scan and resurrect dormant entities into working memory
        self.entity_engine.pre_scan_and_page(chunk_text)
        manifest_prompt = self.entity_engine.format_prompt_block()

        # 2. Neural Transduction
        extraction = self.transducer.transduce(
            chunk_text=chunk_text,
            chunk_id=cid,
            active_manifest_prompt=manifest_prompt,
        )

        # 3. Update working memory manifest
        self.entity_engine.update_from_extraction([e.to_dict() for e in extraction.entities])

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
            # Verify canvas stays bounded throughout book ingestion
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
        """Answer a factual question directly from graph topology in < 10 ms.

        If target_graph is not provided, uses PageTable and ActiveCanvas to resolve
        the relevant sub-graph via semantic page faulting.
        """
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
        """Reset working memory entity manifest, active canvas, and Merkle book between independent documents."""
        self.entity_engine.reset()
        self.active_canvas.clear()
        self.merkle_book = HierarchicalMerkleBook()

    def close(self):
        """Release PageTable and EntityEngine resources."""
        self.entity_engine.close()
        self.page_table.close()
