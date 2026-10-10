"""Decoupled Write-Ahead Semantic Virtual Memory (SVM) Asynchronous Verification & Background Ingestion Worker.

Implements Session 4 of the QUANTA Tiered Kev Master Plan and Phase 5 (§multi_scale_plan.md):
- AsyncKevVerificationQueue / BackgroundIngestor: Thread pool daemon executing:
  1. Asynchronous Belnap truth lattice refinement and relational scoring on idle slots without blocking the user.
  2. Full background completion (transduction + Kev + Clingo-DL + graph upgrade) for deferred chunks.
- Priority Queueing: Items prioritized by calibrated filter score from Phase 2, with immediate top-priority elevation
  for query-active chunks.
- Pause Policies: BG-A (none), BG-B (in-process foreground lock), BG-C (slot polling on llama-server).
- Controlled Lifecycle: QUANTA_BACKGROUND_INGESTION env flag and clean thread teardown on reset/close.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import heapq
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import urllib.request

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
    JOINT_ALLEN_MAP,
    JOINT_EPISTEMIC_MAP,
    JOINT_INTENT_MAP,
    JOINT_PEARL_MAP,
)
from parser.skeleton_transducer import MockSkeletonTransducer, SkeletonEntity, SkeletonEvent
from verification.belnap_calibrator import BelnapLatticeMapper
from verification.clingo_dl_gate import ClingoDLGate

logger = logging.getLogger("quanta.models.kev_async_worker")


@dataclass(order=True)
class VerificationTask:
    """Represents a chunk enqueued for asynchronous Kev verification or deferred background ingestion."""

    priority: float  # Lower value = higher priority in min-heap (e.g. -1e9 for prioritized)
    seq_id: int  # Monotonic counter to break ties and preserve FIFO order
    passage_id: str = field(compare=False)
    doc_id: str = field(compare=False)
    text: str = field(compare=False)
    entities: List[SkeletonEntity] = field(compare=False, default_factory=list)
    events: List[SkeletonEvent] = field(compare=False, default_factory=list)
    ent_cid_map: Dict[str, str] = field(compare=False, default_factory=dict)
    ev_cid_map: Dict[str, str] = field(compare=False, default_factory=dict)
    ent_node_ids: Dict[str, int] = field(compare=False, default_factory=dict)
    ev_node_ids: Dict[str, int] = field(compare=False, default_factory=dict)
    graph_ref: Optional[Any] = field(compare=False, default=None)
    status: str = field(compare=False, default="pending")  # pending, processing, completed, failed
    task_type: str = field(compare=False, default="verify")  # "verify" or "full_ingest"
    filter_score: Optional[float] = field(compare=False, default=None)
    parent_macro_id: Optional[str] = field(compare=False, default=None)
    concept_codes: List[int] = field(compare=False, default_factory=list)
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
        kev_engine: Any = None,
        page_table: Optional[PageTable] = None,
        binary_table: Optional[BinaryNodeTable] = None,
        belnap_mapper: Optional[BelnapLatticeMapper] = None,
        clingo_dl_gate: Optional[ClingoDLGate] = None,
        max_workers: int = 2,
        auto_start: bool = True,
        transducer: Optional[Any] = None,
        passage_store: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        pause_policy: str = "none",  # "none", "foreground_lock", "slot_polling"
        busy_slot_threshold: int = 1,
        llama_server_url: Optional[str] = None,
        slot_checker: Optional[Callable[[], int]] = None,
        enabled: Optional[bool] = None,
    ):
        self.kev_engine = kev_engine
        self.page_table = page_table
        self.binary_table = binary_table
        self.belnap_mapper = belnap_mapper or BelnapLatticeMapper()
        self.clingo_dl_gate = clingo_dl_gate
        self.transducer = transducer
        self.passage_store = passage_store
        self.pipeline = pipeline
        self.pause_policy = str(pause_policy or "none").strip().lower()
        self.busy_slot_threshold = max(1, int(busy_slot_threshold))
        self.llama_server_url = llama_server_url or os.environ.get("QUANTA_LLAMA_SERVER_URL", "http://127.0.0.1:8888")
        self._slot_checker = slot_checker

        # Enabled flag handling
        if enabled is not None:
            self._enabled = bool(enabled)
        else:
            self._enabled = True

        self._heap: List[VerificationTask] = []
        self._tasks: Dict[str, VerificationTask] = {}
        self._completed_evals: Dict[str, KevChunkEvaluation] = {}
        self._completed_subgraphs: Dict[str, QuantaGraph] = {}
        self._seq_counter: int = 0

        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._all_done = threading.Condition(self._lock)
        self._in_flight: int = 0
        self._stop_event = threading.Event()
        self._workers: List[threading.Thread] = []
        self._max_workers = max(1, int(max_workers))

        # Pause policy BG-B: in-process foreground lock state
        self._foreground_active = False
        self._foreground_lock = threading.Lock()
        self._foreground_cond = threading.Condition(self._foreground_lock)

        if auto_start and self._enabled:
            self.start()

    def start(self) -> None:
        """Starts worker threads if not already running."""
        with self._lock:
            if not self._enabled:
                return
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
        """Signals background workers to terminate and cleans up threads."""
        with self._lock:
            self._stop_event.set()
            with self._foreground_lock:
                self._foreground_cond.notify_all()
            self._not_empty.notify_all()
            workers = list(self._workers)
            self._workers.clear()

        if wait:
            for w in workers:
                w.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        """Returns True if the background queue is active and has live worker threads."""
        with self._lock:
            return not self._stop_event.is_set() and len(self._workers) > 0 and any(w.is_alive() for w in self._workers)

    @property
    def enabled(self) -> bool:
        """Returns True if background worker is enabled."""
        return self._enabled

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

    # -------------------------------------------------------------------------
    # Pause Policy Management (§Phase 5)
    # -------------------------------------------------------------------------

    def set_foreground_active(self, active: bool = True) -> None:
        """Sets in-process foreground execution state for pause policy BG-B."""
        with self._foreground_lock:
            self._foreground_active = bool(active)
            if not active:
                self._foreground_cond.notify_all()

    def mark_foreground(self, active: bool = True) -> None:
        """Alias for set_foreground_active."""
        self.set_foreground_active(active)

    def is_foreground_active(self) -> bool:
        """Returns True if foreground execution is currently in progress."""
        with self._foreground_lock:
            return self._foreground_active

    @contextmanager
    def foreground_scope(self):
        """Context manager marking in-process foreground execution."""
        self.set_foreground_active(True)
        try:
            yield
        finally:
            self.set_foreground_active(False)

    def _get_busy_slots(self) -> int:
        """Queries llama-server /slots or slot_checker to count active slots (BG-C)."""
        if self._slot_checker is not None:
            try:
                return int(self._slot_checker())
            except Exception:
                return 0
        if not self.llama_server_url:
            return 0
        try:
            url = f"{self.llama_server_url.rstrip('/')}/slots"
            req = urllib.request.Request(url, headers={"User-Agent": "QUANTA-BackgroundIngestor"})
            with urllib.request.urlopen(req, timeout=0.2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list):
                    busy = 0
                    for s in data:
                        if isinstance(s, dict):
                            if s.get("is_processing", False) or s.get("state", 0) != 0:
                                busy += 1
                    return busy
        except Exception:
            pass
        return 0

    def _check_pause_policy(self) -> None:
        """Checks and waits if active pause policy requires pausing background work."""
        # BG-B: In-process foreground lock
        if self.pause_policy in ("foreground_lock", "bg-b"):
            with self._foreground_lock:
                while self._foreground_active and not self._stop_event.is_set():
                    self._foreground_cond.wait(timeout=0.05)

        # BG-C: Poll llama-server slots
        elif self.pause_policy in ("slot_polling", "bg-c", "slots"):
            while not self._stop_event.is_set():
                busy = self._get_busy_slots()
                if busy < self.busy_slot_threshold:
                    break
                time.sleep(0.05)

    # -------------------------------------------------------------------------
    # Enqueue & Priority API
    # -------------------------------------------------------------------------

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
        priority: Union[int, float] = 10,
    ) -> VerificationTask:
        """Enqueues a chunk for background Kev verification (Session 4 API)."""
        with self._lock:
            if not self._workers and not self._stop_event.is_set() and self._enabled:
                self.start()

            self._seq_counter += 1
            task = VerificationTask(
                priority=float(priority),
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
                task_type="verify",
            )
            self._tasks[passage_id] = task
            heapq.heappush(self._heap, task)
            self._not_empty.notify()
            return task

    def enqueue_deferred(
        self,
        passage_id: str,
        doc_id: str,
        text: str,
        filter_score: Optional[float] = None,
        parent_macro_id: Optional[str] = None,
        concept_codes: Optional[List[int]] = None,
        graph: Optional[QuantaGraph] = None,
        priority: Optional[float] = None,
    ) -> Optional[VerificationTask]:
        """Enqueues a deferred chunk for background transduction, Kev scoring, and graph upgrade (§Phase 5).

        Priority calculation:
        - If priority is explicitly given, uses it directly.
        - If filter_score is given, maps [0.0, 1.0] to (1.0 - score) * 1000.0 so highest relevance
          gets the lowest min-heap key (popped first).
        - Otherwise defaults to 500.0.
        """
        with self._lock:
            if not self._enabled:
                return None
            if not self._workers and not self._stop_event.is_set():
                self.start()

            self._seq_counter += 1
            if priority is not None:
                task_prio = float(priority)
            elif filter_score is not None:
                clamped_score = max(0.0, min(1.0, float(filter_score)))
                task_prio = (1.0 - clamped_score) * 1000.0
            else:
                task_prio = 500.0

            task = VerificationTask(
                priority=task_prio,
                seq_id=self._seq_counter,
                passage_id=passage_id,
                doc_id=doc_id,
                text=text,
                graph_ref=graph,
                status="pending",
                task_type="full_ingest",
                filter_score=filter_score,
                parent_macro_id=parent_macro_id,
                concept_codes=list(concept_codes or []),
            )
            self._tasks[passage_id] = task
            heapq.heappush(self._heap, task)
            self._not_empty.notify()
            return task

    def prioritize(self, passage_id: str) -> bool:
        """Elevates an enqueued chunk to top priority for immediate query resolution."""
        with self._lock:
            task = self._tasks.get(passage_id)
            if task is None or task.status != "pending":
                return False
            task.priority = -1e9  # Root of min-heap
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

    def get_subgraph(self, passage_id: str) -> Optional[QuantaGraph]:
        """Retrieves completed fine-grained subgraph for an upgraded passage (§Phase 5)."""
        with self._lock:
            return self._completed_subgraphs.get(passage_id)

    def clear(self) -> None:
        """Clears pending tasks, completed evaluations, and subgraphs."""
        with self._lock:
            self._heap.clear()
            self._tasks.clear()
            self._completed_evals.clear()
            self._completed_subgraphs.clear()

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
                # Pause policy check (BG-B / BG-C)
                self._check_pause_policy()

                if task.task_type == "full_ingest":
                    self._execute_full_ingestion(task)
                else:
                    self._execute_verification(task)

                with self._lock:
                    task.status = "completed"
                    task.completed_at = time.time()
            except Exception as e:
                logger.error("Error processing passage %s: %s", task.passage_id, e, exc_info=True)
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

    def _execute_full_ingestion(self, task: VerificationTask) -> None:
        """Executes full deferred ingestion cycle: Transduction + Kev + Clingo-DL + Graph Upgrade (§Phase 5)."""
        # 1. Transduce skeleton
        transducer = self.transducer
        if transducer is None:
            transducer = MockSkeletonTransducer(mode="co_decoded")

        if hasattr(transducer, "transduce"):
            try:
                skeleton_res = transducer.transduce(task.text, passage_id=task.passage_id, doc_id=task.doc_id, mode="co_decoded")
            except TypeError:
                try:
                    skeleton_res = transducer.transduce(task.text, passage_id=task.passage_id, doc_id=task.doc_id)
                except TypeError:
                    skeleton_res = transducer.transduce(task.text)
        elif callable(transducer):
            skeleton_res = transducer(task.text)
        else:
            skeleton_res = MockSkeletonTransducer(mode="co_decoded").transduce(task.text, passage_id=task.passage_id, doc_id=task.doc_id)

        if not hasattr(skeleton_res, "events") or not hasattr(skeleton_res, "entities"):
            if hasattr(skeleton_res, "to_skeleton_result"):
                skeleton_res = skeleton_res.to_skeleton_result()

        task.entities = list(getattr(skeleton_res, "entities", []))
        task.events = list(getattr(skeleton_res, "events", []))

        # 2. Kev evaluation
        has_codec = any(getattr(ev, "intent", None) or getattr(ev, "epist", None) for ev in task.events)
        if has_codec:
            valencies = []
            intent_epistemics = []
            relations = []

            for ev in task.events:
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

                raw_intent = (ev.intent or "I").strip().upper()
                raw_epist = (ev.epist or "O").strip().upper()
                mapped_intent = JOINT_INTENT_MAP.get(raw_intent, KevSpeechActIntent.INFORMATIVE.value)
                mapped_epist = JOINT_EPISTEMIC_MAP.get(raw_epist, KevEpistemicSource.DIRECT_OBSERVATION.value)

                intent_epistemics.append(IntentEpistemicResult(
                    event_id=ev.id,
                    predicate=ev.predicate,
                    intent=mapped_intent,
                    intent_confidence=0.95,
                    intent_probabilities={mapped_intent: 0.95},
                    epistemic_source=mapped_epist,
                    epistemic_confidence=0.95,
                    epistemic_probabilities={mapped_epist: 0.95},
                ))

            for i in range(len(task.events) - 1):
                ev_curr = task.events[i]
                ev_next = task.events[i + 1]
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
        elif self.kev_engine is not None and hasattr(self.kev_engine, "evaluate_chunk"):
            kev_eval = self.kev_engine.evaluate_chunk(
                entities=task.entities,
                events=task.events,
                text=task.text,
            )
        else:
            valencies = []
            for ev in task.events:
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
                    ) for ev in task.events
                ],
                relations=[],
                total_latency_ms=0.1,
                mode="bypass",
            )

        # 3. Clingo difference logic validation
        if self.clingo_dl_gate is not None:
            try:
                dl_res = self.clingo_dl_gate.validate(
                    events=task.events,
                    relations=kev_eval.relations,
                )
                if not dl_res.is_valid:
                    logger.warning("ClingoDLGate validation issue during background ingestion for %s: %s", task.passage_id, dl_res.errors)
            except Exception as dl_err:
                logger.debug("ClingoDLGate check error: %s", dl_err)

        # 4. Construct fine-grained QuantaGraph subgraph
        subgraph = QuantaGraph()
        subgraph.chunk_id = task.passage_id

        valency_role_map = {(v.entity_id, v.event_id): v for v in kev_eval.valencies}
        ie_map = {ie.event_id: ie for ie in kev_eval.intent_epistemics}

        # Entities
        for ent in task.entities:
            node = QuantaNode(
                anchor=ent.surface_text,
                literal=ent.surface_text,
                truth_status="TRUE",
                confidence=float(ent.confidence),
                passage_id=task.passage_id,
                node_type=ent.category,
            )
            node.compute_canonical_cid()
            subgraph.add_node(node)
            task.ent_cid_map[ent.id] = node.canonical_cid

            if self.page_table is not None:
                self.page_table.store_node(node)
            if self.binary_table is not None:
                c_u8 = int(float(ent.confidence) * 255) if float(ent.confidence) <= 1.0 else int(ent.confidence)
                nid = self.binary_table.append(QuantaSemanticNodeStruct.create(
                    node_id=len(self.binary_table) + 1,
                    belnap_lattice=BelnapValue.TRUE,
                    confidence=c_u8,
                ))
                task.ent_node_ids[ent.id] = nid + 1

        # Events
        for ev in task.events:
            ie_res = ie_map.get(ev.id)
            intent_str = ie_res.intent if ie_res else "INFORMATIVE"
            epistemic_str = ie_res.epistemic_source if ie_res else "DIRECT_OBSERVATION"
            conf = float(ev.confidence)
            belnap_val = self.belnap_mapper.map_probability(conf)
            truth_status = BelnapValue.to_str(belnap_val)

            node = QuantaNode(
                anchor=ev.predicate,
                literal=ev.predicate,
                truth_status=truth_status,
                confidence=conf,
                evidence_source=epistemic_str.lower(),
                passage_id=task.passage_id,
            )
            node.set_slot("TYPE_EVENT", 1)
            node.compute_canonical_cid()
            subgraph.add_node(node)
            task.ev_cid_map[ev.id] = node.canonical_cid

            if self.page_table is not None:
                self.page_table.store_node(node)
            if self.binary_table is not None:
                ev_c_u8 = int(conf * 255) if conf <= 1.0 else int(conf)
                nid = self.binary_table.append(QuantaSemanticNodeStruct.create(
                    node_id=len(self.binary_table) + 1,
                    belnap_lattice=belnap_val,
                    confidence=ev_c_u8,
                    intent_band5=BinarySpeechActIntent.from_str(intent_str),
                    epistemic_band6=BinaryEpistemicSource.from_str(epistemic_str),
                ))
                task.ev_node_ids[ev.id] = nid + 1

        # Valency edges
        for ent in task.entities:
            for ev in task.events:
                v_res = valency_role_map.get((ent.id, ev.id))
                role = v_res.role if v_res else None
                if not role or role == "NONE":
                    if ev.subject_ent_id == ent.id:
                        role = "AGENT"
                    elif ev.object_ent_id == ent.id:
                        role = "PATIENT"

                ent_cid = task.ent_cid_map.get(ent.id)
                ev_cid = task.ev_cid_map.get(ev.id)
                if ent_cid and ev_cid and role and role != "NONE":
                    if role == "AGENT":
                        subgraph.add_edge(ev_cid, "VAL_X1_AGENT", ent_cid)
                    elif role == "PATIENT":
                        subgraph.add_edge(ev_cid, "VAL_X2_PATIENT", ent_cid)
                    elif role == "INSTRUMENT":
                        subgraph.add_edge(ev_cid, "VAL_X5_INSTRUMENT", ent_cid)

        # Relation edges
        for rel in kev_eval.relations:
            src_cid = task.ev_cid_map.get(rel.source_event_id)
            tgt_cid = task.ev_cid_map.get(rel.target_event_id)
            if src_cid and tgt_cid:
                allen_belnap = self.belnap_mapper.map_probability(rel.allen_confidence)
                if rel.allen_relation != "NONE" and allen_belnap != BelnapValue.CONTRADICTION:
                    label = f"TEMP_ALLEN_{rel.allen_relation.upper()}"
                    subgraph.add_edge(src_cid, label, tgt_cid)
                pearl_belnap = self.belnap_mapper.map_probability(rel.pearl_confidence)
                if rel.pearl_relation != "NONE" and pearl_belnap != BelnapValue.CONTRADICTION:
                    label = "CAUSAL_MECHANISM_LINK" if rel.pearl_relation == "MECHANISM_LINK" else "ENABLING_CONDITION"
                    subgraph.add_edge(src_cid, label, tgt_cid)

        # 5. In-place upgrade of graph
        target_graph = task.graph_ref
        if target_graph is None and self.pipeline is not None:
            target_graph = getattr(self.pipeline, "active_canvas", None)

        if target_graph is not None:
            if hasattr(target_graph, "upgrade_passage"):
                target_graph.upgrade_passage(task.passage_id, subgraph)
            elif hasattr(target_graph, "add_node"):
                for n in subgraph.nodes.values():
                    target_graph.add_node(n)
                for s_cid, edges in subgraph.edges.items():
                    for lbl, tgts in edges.items():
                        for t_cid in tgts:
                            target_graph.add_edge(s_cid, lbl, t_cid)
            elif hasattr(target_graph, "put"):
                for n in subgraph.nodes.values():
                    target_graph.put(n)

            if task.concept_codes and hasattr(target_graph, "add_concept_anchor"):
                for code in task.concept_codes:
                    target_graph.add_concept_anchor(task.passage_id, code)

        # 6. Update PassageStore status to FULL
        pstore = self.passage_store
        if pstore is None and self.pipeline is not None:
            pstore = getattr(self.pipeline, "passage_store", None)
        if pstore is not None and hasattr(pstore, "set_ingestion_state"):
            pstore.set_ingestion_state(task.passage_id, "FULL")

        # 7. Record Completion in Cache
        with self._lock:
            self._completed_evals[task.passage_id] = kev_eval
            self._completed_subgraphs[task.passage_id] = subgraph


# Alias BackgroundIngestor to AsyncKevVerificationQueue (§Phase 5)
BackgroundIngestor = AsyncKevVerificationQueue

