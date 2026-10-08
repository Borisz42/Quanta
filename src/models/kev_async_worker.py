"""Decoupled Write-Ahead Semantic Virtual Memory (SVM) Asynchronous Verification Worker.

Implements Session 4 of the QUANTA Tiered Kev Master Plan:
- AsyncKevVerificationQueue: Thread pool daemon executing asynchronous Belnap truth
  lattice refinement and relational scoring on idle slots without blocking the user.
- Write-Ahead Architecture: Documents are committed provisionally and accessible
  immediately (>= 200 words/sec, return latency < 1.3s per chunk), while background
  workers evaluate the multi-slot/logprob decisions and update SQLite PageTable and
  the 128-byte BinaryNodeTable in-place.
- Priority Queueing: Chunks referenced by active queries are prioritized ahead of
  unqueried background chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core.asg import QuantaGraph, QuantaNode
from core.binary_node import (
    BelnapValue,
    BinaryNodeTable,
    EpistemicSource as BinaryEpistemicSource,
    QuantaSemanticNodeStruct,
    SpeechActIntent as BinarySpeechActIntent,
)
from memory.page_table import PageTable
from models.kev_engine import (
    AllenTemporalRelation,
    EpistemicSource as KevEpistemicSource,
    IntentEpistemicResult,
    KevChunkEvaluation,
    KevDecisionEngine,
    PearlCausalLink,
    RelationScoringResult,
    SpeechActIntent as KevSpeechActIntent,
    ValencyScoringResult,
)
from parser.skeleton_transducer import SkeletonEntity, SkeletonEvent
from verification.belnap_calibrator import BelnapLatticeMapper
from verification.clingo_dl_gate import ClingoDLGate

logger = logging.getLogger("quanta.models.kev_async_worker")


@dataclass(order=True)
class VerificationTask:
    """Represents a chunk enqueued for asynchronous Kev verification."""

    priority: int  # 0 = prioritized (query-active), 10 = standard background
    seq_id: int  # monotonic counter to break ties and preserve FIFO order
    passage_id: str = field(compare=False)
    doc_id: str = field(compare=False)
    text: str = field(compare=False)
    entities: List[SkeletonEntity] = field(compare=False)
    events: List[SkeletonEvent] = field(compare=False)
    ent_cid_map: Dict[str, str] = field(compare=False, default_factory=dict)
    ev_cid_map: Dict[str, str] = field(compare=False, default_factory=dict)
    ent_node_ids: Dict[str, int] = field(compare=False, default_factory=dict)
    ev_node_ids: Dict[str, int] = field(compare=False, default_factory=dict)
    graph_ref: Optional[Any] = field(compare=False, default=None)
    status: str = field(compare=False, default="pending")  # pending, processing, completed, failed
    created_at: float = field(compare=False, default_factory=time.time)
    completed_at: Optional[float] = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)


class AsyncKevVerificationQueue:
    """Asynchronous Write-Ahead verification queue and background thread pool.
    
    Decouples document ingestion acknowledgment from epistemic lattice refinement.
    Evaluates tiered / logprob Kev decisions in the background and commits in-place
    updates to PageTable and BinaryNodeTable.
    """

    def __init__(
        self,
        kev_engine: Any,
        page_table: PageTable,
        binary_table: Optional[BinaryNodeTable] = None,
        belnap_mapper: Optional[BelnapLatticeMapper] = None,
        clingo_dl_gate: Optional[ClingoDLGate] = None,
        max_workers: int = 2,
        auto_start: bool = True,
    ):
        self.kev_engine = kev_engine
        self.page_table = page_table
        self.binary_table = binary_table
        self.belnap_mapper = belnap_mapper or BelnapLatticeMapper()
        self.clingo_dl_gate = clingo_dl_gate

        self._heap: List[VerificationTask] = []
        self._tasks: Dict[str, VerificationTask] = {}
        self._completed_evals: Dict[str, KevChunkEvaluation] = {}
        self._seq_counter: int = 0

        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._all_done = threading.Condition(self._lock)
        self._in_flight: int = 0
        self._stop_event = threading.Event()
        self._workers: List[threading.Thread] = []
        self._max_workers = max(1, int(max_workers))

        if auto_start:
            self.start()

    def start(self) -> None:
        """Starts worker threads if not already running."""
        with self._lock:
            if self._workers:
                return
            self._stop_event.clear()
            for i in range(self._max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"AsyncKevWorker-{i+1}",
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()

    def stop(self, wait: bool = True, timeout: Optional[float] = 5.0) -> None:
        """Signals background workers to terminate."""
        with self._lock:
            self._stop_event.set()
            self._not_empty.notify_all()
            workers = list(self._workers)
            self._workers.clear()

        if wait:
            for w in workers:
                w.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        """Returns True if the background queue is active."""
        with self._lock:
            return not self._stop_event.is_set() and len(self._workers) > 0

    @property
    def pending_count(self) -> int:
        """Number of tasks waiting in queue."""
        with self._lock:
            return len(self._heap)

    @property
    def in_flight_count(self) -> int:
        """Number of tasks currently being evaluated by workers."""
        with self._lock:
            return self._in_flight

    @property
    def completed_count(self) -> int:
        """Number of completed evaluations."""
        with self._lock:
            return len(self._completed_evals)

    def enqueue(
        self,
        passage_id: str,
        doc_id: str,
        text: str,
        entities: List[SkeletonEntity],
        events: List[SkeletonEvent],
        ent_cid_map: Optional[Dict[str, str]] = None,
        ev_cid_map: Optional[Dict[str, str]] = None,
        ent_node_ids: Optional[Dict[str, int]] = None,
        ev_node_ids: Optional[Dict[str, int]] = None,
        graph: Optional[QuantaGraph] = None,
        priority: int = 10,
    ) -> VerificationTask:
        """Enqueues a chunk for background Kev verification.
        
        Args:
            passage_id: Unique passage identifier.
            doc_id: Source document ID.
            text: Raw chunk text.
            entities: Extracted skeleton entities.
            events: Extracted skeleton events.
            ent_cid_map: Map of entity id -> canonical CID.
            ev_cid_map: Map of event id -> canonical CID.
            ent_node_ids: Map of entity id -> 128-byte binary table node_id.
            ev_node_ids: Map of event id -> 128-byte binary table node_id.
            graph: Optional in-memory QuantaGraph to update in-place upon verification.
            priority: Priority rank (0=urgent, 10=normal).

        Returns:
            The created VerificationTask.
        """
        with self._lock:
            # Ensure workers are active
            if not self._workers and not self._stop_event.is_set():
                self.start()

            self._seq_counter += 1
            task = VerificationTask(
                priority=priority,
                seq_id=self._seq_counter,
                passage_id=passage_id,
                doc_id=doc_id,
                text=text,
                entities=list(entities),
                events=list(events),
                ent_cid_map=dict(ent_cid_map or {}),
                ev_cid_map=dict(ev_cid_map or {}),
                ent_node_ids=dict(ent_node_ids or {}),
                ev_node_ids=dict(ev_node_ids or {}),
                graph_ref=graph,
                status="pending",
            )
            self._tasks[passage_id] = task
            heapq.heappush(self._heap, task)
            self._not_empty.notify()
            return task

    def prioritize(self, passage_id: str) -> bool:
        """Elevates an enqueued chunk to top priority (0) for immediate query resolution."""
        with self._lock:
            task = self._tasks.get(passage_id)
            if task is None or task.status != "pending":
                return False
            task.priority = 0
            heapq.heapify(self._heap)
            self._not_empty.notify()
            return True

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Blocks until all enqueued and in-flight tasks have finished processing.
        
        Returns:
            True if all tasks finished within timeout, False if timed out.
        """
        end_time = time.time() + timeout if timeout is not None else None
        with self._all_done:
            while self._heap or self._in_flight > 0:
                if self._stop_event.is_set() and not self._heap and self._in_flight == 0:
                    break
                remaining = end_time - time.time() if end_time is not None else None
                if remaining is not None and remaining <= 0:
                    return False
                wait_time = min(0.1, remaining) if remaining is not None else 0.1
                self._all_done.wait(timeout=wait_time)
            return True

    def is_verified(self, passage_id: str) -> bool:
        """Returns True if the specified passage has completed Kev verification."""
        with self._lock:
            return passage_id in self._completed_evals

    def get_status(self, passage_id: str) -> Optional[str]:
        """Returns status of passage ('pending', 'processing', 'completed', 'failed', or None)."""
        with self._lock:
            if passage_id in self._completed_evals:
                return "completed"
            task = self._tasks.get(passage_id)
            return task.status if task is not None else None

    def get_evaluation(self, passage_id: str) -> Optional[KevChunkEvaluation]:
        """Retrieves the refined Kev evaluation for a passage if completed."""
        with self._lock:
            return self._completed_evals.get(passage_id)

    def clear(self) -> None:
        """Clears pending tasks and completed evaluations."""
        with self._lock:
            self._heap.clear()
            self._tasks.clear()
            self._completed_evals.clear()

    def _worker_loop(self) -> None:
        """Daemon worker loop popping tasks from the priority queue."""
        while not self._stop_event.is_set():
            with self._not_empty:
                while not self._heap and not self._stop_event.is_set():
                    self._not_empty.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                task = heapq.heappop(self._heap)
                if task.status != "pending":
                    continue
                task.status = "processing"
                self._in_flight += 1

            try:
                self._execute_verification(task)
                with self._lock:
                    task.status = "completed"
                    task.completed_at = time.time()
            except Exception as e:
                logger.error("Error verifying passage %s: %s", task.passage_id, e, exc_info=True)
                with self._lock:
                    task.status = "failed"
                    task.error = str(e)
            finally:
                with self._all_done:
                    self._in_flight -= 1
                    if not self._heap and self._in_flight == 0:
                        self._all_done.notify_all()

    def _execute_verification(self, task: VerificationTask) -> None:
        """Executes Kev evaluation, Clingo difference logic, and in-place table updates."""
        # 1. Run Kev Evaluation
        kev_eval = self.kev_engine.evaluate_chunk(
            entities=task.entities,
            events=task.events,
            text=task.text,
        )

        # 2. Clingo Difference Logic Validation
        if self.clingo_dl_gate is not None:
            try:
                dl_res = self.clingo_dl_gate.validate(
                    events=task.events,
                    relations=kev_eval.relations,
                )
                if not dl_res.is_valid:
                    logger.warning("ClingoDLGate detected conflicts during async verification for %s: %s", task.passage_id, dl_res.errors)
            except Exception as dl_err:
                logger.debug("ClingoDLGate validation check in async worker: %s", dl_err)

        # 3. Build Mapping Lookup Tables
        valency_role_map: Dict[Tuple[str, str], ValencyScoringResult] = {
            (v.entity_id, v.event_id): v for v in kev_eval.valencies
        }
        ie_map: Dict[str, IntentEpistemicResult] = {
            ie.event_id: ie for ie in kev_eval.intent_epistemics
        }

        # 4. In-Place Updates: PageTable and Active Canvas
        page_updates: List[Dict[str, Any]] = []
        node_edges_map: Dict[str, Dict[str, List[str]]] = {}

        for ev in task.events:
            ev_cid = task.ev_cid_map.get(ev.id)
            if not ev_cid:
                continue

            ie_res = ie_map.get(ev.id)
            intent_str = ie_res.intent if ie_res else "INFORMATIVE"
            epistemic_str = ie_res.epistemic_source if ie_res else "DIRECT_OBSERVATION"

            conf = float(ev.confidence)
            belnap_val = self.belnap_mapper.map_probability(conf)
            truth_status = BelnapValue.to_str(belnap_val)

            # Assemble edges
            ev_edges: Dict[str, List[str]] = {}

            # Thematic valencies
            for ent in task.entities:
                v_res = valency_role_map.get((ent.id, ev.id))
                role = v_res.role if v_res else None
                if not role or role == "NONE":
                    if ev.subject_ent_id == ent.id:
                        role = "AGENT"
                    elif ev.object_ent_id == ent.id:
                        role = "PATIENT"

                ent_cid = task.ent_cid_map.get(ent.id)
                if ent_cid:
                    if role == "AGENT":
                        ev_edges.setdefault("VAL_X1_AGENT", []).append(ent_cid)
                    elif role == "PATIENT":
                        ev_edges.setdefault("VAL_X2_PATIENT", []).append(ent_cid)
                    elif role == "INSTRUMENT":
                        ev_edges.setdefault("VAL_X5_INSTRUMENT", []).append(ent_cid)

            # Temporal and causal relations
            for rel in kev_eval.relations:
                if rel.source_event_id == ev.id:
                    tgt_cid = task.ev_cid_map.get(rel.target_event_id)
                    if tgt_cid:
                        allen_belnap = self.belnap_mapper.map_probability(rel.allen_confidence)
                        if rel.allen_relation != "NONE" and allen_belnap != BelnapValue.CONTRADICTION:
                            label = f"TEMP_ALLEN_{rel.allen_relation.upper()}"
                            ev_edges.setdefault(label, []).append(tgt_cid)

                        pearl_belnap = self.belnap_mapper.map_probability(rel.pearl_confidence)
                        if rel.pearl_relation != "NONE" and pearl_belnap != BelnapValue.CONTRADICTION:
                            label = "CAUSAL_MECHANISM_LINK" if rel.pearl_relation == "MECHANISM_LINK" else "ENABLING_CONDITION"
                            ev_edges.setdefault(label, []).append(tgt_cid)

            node_edges_map[ev_cid] = ev_edges
            page_updates.append({
                "cid": ev_cid,
                "truth_status": truth_status,
                "confidence": conf,
                "evidence_source": epistemic_str.lower(),
                "edges": ev_edges,
            })

        # Update PageTable in single synchronized batch
        if hasattr(self.page_table, "update_nodes_batch"):
            self.page_table.update_nodes_batch(page_updates)

        # 5. In-Place Updates: 128-byte BinaryNodeTable
        if self.binary_table is not None:
            # Update entities
            for ent in task.entities:
                nid = task.ent_node_ids.get(ent.id)
                if nid is not None:
                    self.binary_table.update_node(
                        node_id=nid,
                        belnap_lattice=BelnapValue.TRUE,
                        confidence=float(ent.confidence),
                    )

            # Update events
            for ev in task.events:
                nid = task.ev_node_ids.get(ev.id)
                ev_cid = task.ev_cid_map.get(ev.id)
                if nid is not None and ev_cid is not None:
                    ie_res = ie_map.get(ev.id)
                    intent_str = ie_res.intent if ie_res else "INFORMATIVE"
                    epistemic_str = ie_res.epistemic_source if ie_res else "DIRECT_OBSERVATION"
                    conf = float(ev.confidence)
                    belnap_val = self.belnap_mapper.map_probability(conf)

                    ev_edges = node_edges_map.get(ev_cid, {})
                    target_nids = []
                    for rel_targets in ev_edges.values():
                        for t_cid in rel_targets:
                            t_nid = self.binary_table.get_node_id(t_cid)
                            if t_nid is not None and t_nid not in target_nids:
                                target_nids.append(t_nid)

                    self.binary_table.update_node(
                        node_id=nid,
                        belnap_lattice=belnap_val,
                        confidence=conf,
                        intent_band5=BinarySpeechActIntent.from_str(intent_str),
                        epistemic_band6=BinaryEpistemicSource.from_str(epistemic_str),
                        add_edges=target_nids[:4],
                    )

        # 6. In-Place Updates: In-Memory QuantaGraph Reference
        if task.graph_ref is not None:
            for ev in task.events:
                ev_cid = task.ev_cid_map.get(ev.id)
                if ev_cid and ev_cid in task.graph_ref.nodes:
                    node = task.graph_ref.nodes[ev_cid]
                    ie_res = ie_map.get(ev.id)
                    epistemic_str = ie_res.epistemic_source if ie_res else "DIRECT_OBSERVATION"
                    conf = float(ev.confidence)
                    belnap_val = self.belnap_mapper.map_probability(conf)
                    node.truth_status = BelnapValue.to_str(belnap_val)
                    node.confidence = conf
                    node.evidence_source = epistemic_str.lower()

                    ev_edges = node_edges_map.get(ev_cid, {})
                    for rel, tgts in ev_edges.items():
                        for t_cid in tgts:
                            task.graph_ref.add_edge(ev_cid, rel, t_cid)

            setattr(task.graph_ref, "kev_evaluation", kev_eval)

        # 7. Record Completion in Cache
        with self._lock:
            self._completed_evals[task.passage_id] = kev_eval
