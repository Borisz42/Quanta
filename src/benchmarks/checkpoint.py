"""Incremental Checkpointing & Resumption Manager for QUANTA Paired Benchmarks.

Provides atomic checkpoint writing, sample-level resumption, and telemetry persistence
to ensure long-running academic evaluation runs can recover seamlessly from interruptions.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Set, Union

from .metrics import ComparativeScorecard
from .paired_evaluator import PairedResult

logger = logging.getLogger("quanta.benchmarks.checkpoint")


class BenchmarkCheckpointManager:
    """Manages incremental checkpointing and resumption of paired benchmark runs."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        active: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path or (Path("output") / ".paired_benchmark_checkpoint.json"))
        self.active = active
        self._state: Dict[str, Any] = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "in_progress",
            "execution_mode": None,
            "ablation_mode": None,
            "suite_config": {},
            "completed_suites": [],
            "results": {},  # suite_name -> list of serialized PairedResult
            "scorecards": {},  # suite_name -> serialized ComparativeScorecard
            "latency_datapoints": [],  # list of LatencyProfilePoint dicts
        }

    def has_checkpoint(self) -> bool:
        """Checks if a valid non-empty checkpoint file exists on disk."""
        if not self.checkpoint_path.exists():
            return False
        try:
            return self.checkpoint_path.stat().st_size > 0
        except OSError:
            return False

    def load(self) -> bool:
        """Loads state from checkpoint file. Returns True if successful."""
        if not self.has_checkpoint():
            return False

        try:
            text = self.checkpoint_path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict) and "results" in data:
                self._state = data
                logger.info(
                    "Loaded benchmark checkpoint from %s (%d suites recorded)",
                    self.checkpoint_path,
                    len(data.get("results", {})),
                )
                return True
        except Exception as e:
            logger.warning("Failed to load checkpoint from %s: %s", self.checkpoint_path, e)

        return False

    def initialize_run(
        self,
        execution_mode: str,
        ablation_mode: str,
        suite_config: Dict[str, int],
        resume: bool = False,
    ):
        """Initializes run state. If resume is False, resets checkpoint."""
        if resume and self.has_checkpoint():
            loaded = self.load()
            if loaded:
                self._state["updated_at"] = datetime.now().isoformat()
                self._state["status"] = "resumed"
                self._state["suite_config"] = suite_config
                self._save_atomic()
                return

        # Fresh start
        self._state = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "in_progress",
            "execution_mode": execution_mode,
            "ablation_mode": ablation_mode,
            "suite_config": suite_config,
            "completed_suites": [],
            "results": {},
            "scorecards": {},
            "latency_datapoints": [],
        }
        if self.active:
            self._save_atomic()

    def _save_atomic(self):
        """Atomically writes checkpoint JSON to disk using temporary file rename."""
        if not self.active:
            return

        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._state["updated_at"] = datetime.now().isoformat()
            payload = json.dumps(self._state, indent=2, ensure_ascii=False)

            temp_file = self.checkpoint_path.with_suffix(f".tmp_{os.getpid()}")
            temp_file.write_text(payload, encoding="utf-8")
            temp_file.replace(self.checkpoint_path)
        except Exception as e:
            logger.warning("Error writing checkpoint to %s: %s", self.checkpoint_path, e)

    def save_sample(
        self,
        suite: str,
        sample_id: str,
        paired_result: PairedResult,
        latency_datapoint: Optional[Dict[str, Any]] = None,
    ):
        """Records an evaluated sample incrementally and persists to disk."""
        if not self.active:
            return

        results_by_suite = self._state.setdefault("results", {})
        suite_list = results_by_suite.setdefault(suite, [])

        # Check if already present to update or append
        existing_idx = next((i for i, r in enumerate(suite_list) if r.get("id") == sample_id), None)
        serialized_pr = paired_result.to_dict()

        if existing_idx is not None:
            suite_list[existing_idx] = serialized_pr
        else:
            suite_list.append(serialized_pr)

        if latency_datapoint:
            self._state.setdefault("latency_datapoints", []).append(latency_datapoint)

        self._save_atomic()

    def save_suite_scorecard(self, suite: str, scorecard: ComparativeScorecard):
        """Records a completed suite scorecard and marks suite as completed."""
        if not self.active:
            return

        self._state.setdefault("scorecards", {})[suite] = asdict(scorecard)
        completed = set(self._state.get("completed_suites", []))
        completed.add(suite)
        self._state["completed_suites"] = sorted(list(completed))
        self._save_atomic()

    def mark_completed(self):
        """Marks the overall benchmark run as successfully finished."""
        if not self.active:
            return
        self._state["status"] = "completed"
        self._state["completed_at"] = datetime.now().isoformat()
        self._save_atomic()

    def get_completed_sample_ids(self, suite: str, skip_errors: bool = True) -> Set[str]:
        """Returns set of sample IDs already evaluated for a given suite.
        
        Args:
            suite: Benchmark suite name.
            skip_errors: If True, samples whose answers ended in fatal timeout/communication errors
                         are excluded so that subsequent runs can automatically retry them.
        """
        suite_results = self._state.get("results", {}).get(suite, [])
        completed = set()
        for r in suite_results:
            if "id" not in r:
                continue
            if skip_errors:
                has_fatal_err = False
                for cond_key in ("quanta_local", "quanta_global", "base"):
                    cond = r.get(cond_key)
                    if cond and isinstance(cond, dict):
                        ans = str(cond.get("answer", ""))
                        if ans.startswith("ERROR: timed out") or ans.startswith("ERROR: [Errno") or ans.startswith("HTTP 500"):
                            has_fatal_err = True
                            break
                if has_fatal_err:
                    continue
            completed.add(r["id"])
        return completed

    def get_sample_result(self, suite: str, sample_id: str) -> Optional[PairedResult]:
        """Retrieves and deserializes a single cached sample result if available."""
        suite_results = self._state.get("results", {}).get(suite, [])
        for r in suite_results:
            if r.get("id") == sample_id:
                try:
                    return PairedResult.from_dict(r)
                except Exception as e:
                    logger.warning("Failed to deserialize cached sample %s: %s", sample_id, e)
                    return None
        return None

    def get_suite_results(self, suite: str) -> List[PairedResult]:
        """Retrieves all deserialized PairedResult instances for a suite."""
        raw_list = self._state.get("results", {}).get(suite, [])
        deserialized = []
        for r in raw_list:
            try:
                deserialized.append(PairedResult.from_dict(r))
            except Exception as e:
                logger.warning("Error parsing cached result for %s: %s", r.get("id"), e)
        return deserialized

    def get_suite_scorecard(self, suite: str) -> Optional[ComparativeScorecard]:
        """Retrieves cached ComparativeScorecard for a suite if available."""
        sc_dict = self._state.get("scorecards", {}).get(suite)
        if not sc_dict:
            return None
        try:
            return ComparativeScorecard(**sc_dict)
        except Exception as e:
            logger.warning("Error loading cached scorecard for %s: %s", suite, e)
            return None

    def get_all_scorecards(self) -> List[ComparativeScorecard]:
        """Retrieves all cached ComparativeScorecards."""
        scorecards = []
        for suite, sc_dict in self._state.get("scorecards", {}).items():
            try:
                scorecards.append(ComparativeScorecard(**sc_dict))
            except Exception:
                pass
        return scorecards

    def get_all_raw_predictions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Returns raw predictions dict formatted for exporter and JSON dumping."""
        return self._state.get("results", {})

    def get_latency_datapoints(self) -> List[Dict[str, Any]]:
        """Returns all recorded latency profile datapoints."""
        return self._state.get("latency_datapoints", [])

    def record_latency_datapoint(self, point: Dict[str, Any]):
        """Records a latency profile point."""
        self._state.setdefault("latency_datapoints", []).append(point)
        self._save_atomic()

    def clear(self):
        """Removes the checkpoint file and resets state."""
        self._state = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "cleared",
            "results": {},
            "scorecards": {},
            "latency_datapoints": [],
        }
        if self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
            except OSError as e:
                logger.warning("Failed to remove checkpoint file %s: %s", self.checkpoint_path, e)
