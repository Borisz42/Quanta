"""Calibrated Parameter Registry for QUANTA Dynamic Multi-Scale Ingestion (§1).

Every threshold, weight, block size, budget, and SLA is a calibrated parameter.
When uncalibrated, its default is pass-through (feature is off, B0 behaviour preserved).
Calibration scripts persist values to config/multi_scale_profile.json with explicit
source_exp and source_commit metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("quanta.config.multi_scale")

DEFAULT_PROFILE_PATH = Path("config/multi_scale_profile.json")

# Default registry specifications: (name, default_passthrough, notes)
REGISTERED_PARAMS: Dict[str, Tuple[Any, str]] = {
    # Phase 1: Query Extraction
    "query_extractor.strategy": ("passthrough", "Strategy for isolating query from context (passthrough/QE-A control)"),
    # Phase 2: Hierarchical Chunker & Relevance Filter
    "chunker.macro_target_tokens": (None, "Target token length for macro discourse blocks"),
    "chunker.micro_target_words": (None, "Target word length for micro discourse chunks"),
    "filter.strategy": ("passthrough", "Scorer variant: lexical, concept, kev_micro, kev_macro, kev_head, or passthrough"),
    "filter.unit": ("micro", "Scoring granularity unit: micro, macro, or macro_head"),
    "filter.keep_threshold": (None, "Confidence/probability threshold to keep a unit"),
    "filter.keep_top_k": (None, "Maximum number of units to keep"),
    "filter.keep_budget_tokens": (None, "Token budget cap for kept units"),
    "filter.calibration": (None, "Calibration mapping (temperature or Platt coefficients)"),
    # Phase 3: Fast-Path Mode Bush
    "fast_path.mode": ("passthrough", "Fast-path mode: raw_only, hot_transduce, full, or passthrough"),
    "fast_path.hot_transduce_n": (None, "Number of top-ranked units synchronously transduced"),
    "fast_path.coverage_threshold": (None, "Decision threshold for coverage check"),
    # Phase 4: Multi-Scale Graph & PPR
    "ppr.inter_scale_weight": (None, "Edge weight connecting micro chunks to coarse macro nodes"),
    "poprag.coarse_node_weight": (None, "PPR damping weight for coarse macro nodes"),
    # Phase 5: Background Completion
    "background.enabled": (False, "Whether asynchronous background ingestion is enabled"),
    "background.pause_policy": ("none", "Background worker pause policy: none, foreground_lock, slot_polling"),
    "background.max_concurrency": (1, "Maximum concurrent background worker threads"),
}


@dataclass
class CalibratedParam:
    """Represents a single parameter entry in the Calibrated Parameter Registry."""
    name: str
    value: Any
    default_passthrough: Any
    source_exp: Optional[str] = None
    source_commit: Optional[str] = None
    calibrated_on: Optional[str] = None  # dataset / split, e.g. "musique_long/dev"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "default_passthrough": self.default_passthrough,
            "source_exp": self.source_exp,
            "source_commit": self.source_commit,
            "calibrated_on": self.calibrated_on,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalibratedParam:
        return cls(
            name=data["name"],
            value=data.get("value"),
            default_passthrough=data.get("default_passthrough"),
            source_exp=data.get("source_exp"),
            source_commit=data.get("source_commit"),
            calibrated_on=data.get("calibrated_on"),
            notes=data.get("notes", ""),
        )


class MultiScaleConfig:
    """Unified configuration resolver for Dynamic Multi-Scale Ingestion.

    Precedence order:
    1. Built-in pass-through defaults
    2. config/multi_scale_profile.json (written strictly by calibration scripts)
    3. Environment variables (QUANTA_MS_<PARAM>)
    4. Per-request proxy headers / runtime overrides
    """

    def __init__(
        self,
        profile_path: Optional[Union[str, Path]] = None,
        runtime_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.profile_path = Path(profile_path) if profile_path else DEFAULT_PROFILE_PATH
        self.runtime_overrides = dict(runtime_overrides or {})
        self.params: Dict[str, CalibratedParam] = {}
        self._load()

    def _load(self):
        """Loads parameters according to the strict 4-tier precedence order."""
        self.params.clear()

        # Tier 1: Built-in pass-through defaults
        for name, (default_val, notes) in REGISTERED_PARAMS.items():
            self.params[name] = CalibratedParam(
                name=name,
                value=default_val,
                default_passthrough=default_val,
                source_exp=None,
                source_commit=None,
                calibrated_on=None,
                notes=notes,
            )

        # Tier 2: config/multi_scale_profile.json
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
                if not isinstance(profile_data, dict):
                    raise ValueError(f"Profile {self.profile_path} must be a JSON dictionary")

                for key, entry in profile_data.items():
                    if isinstance(entry, dict):
                        source_exp = entry.get("source_exp")
                        if not source_exp or not str(source_exp).strip():
                            raise ValueError(
                                f"Profile entry '{key}' in {self.profile_path} is missing required 'source_exp'"
                            )
                        default_val = REGISTERED_PARAMS.get(key, (None, ""))[0]
                        self.params[key] = CalibratedParam(
                            name=key,
                            value=entry.get("value"),
                            default_passthrough=default_val,
                            source_exp=source_exp,
                            source_commit=entry.get("source_commit"),
                            calibrated_on=entry.get("calibrated_on"),
                            notes=entry.get("notes", ""),
                        )
                    else:
                        raise ValueError(
                            f"Profile entry '{key}' must be a dictionary with 'value' and 'source_exp'"
                        )
            except Exception as e:
                logger.error("Failed to load profile %s: %s", self.profile_path, e)
                raise

        # Tier 3: Environment variables (QUANTA_MS_<PARAM>)
        prefix = "QUANTA_MS_"
        for env_k, env_v in os.environ.items():
            if env_k.startswith(prefix):
                param_slug = env_k[len(prefix):].lower()
                # Find matching registered param or direct replace '_' with '.'
                matched_name = None
                for reg_name in self.params.keys():
                    if reg_name.lower().replace(".", "_") == param_slug:
                        matched_name = reg_name
                        break
                if not matched_name:
                    matched_name = param_slug.replace("_", ".")

                parsed_v = self._parse_env_value(env_v)
                curr = self.params.get(matched_name)
                default_val = curr.default_passthrough if curr else None
                source_exp = curr.source_exp if curr else None
                source_commit = curr.source_commit if curr else None
                calibrated_on = curr.calibrated_on if curr else None

                self.params[matched_name] = CalibratedParam(
                    name=matched_name,
                    value=parsed_v,
                    default_passthrough=default_val,
                    source_exp=f"env:{env_k}" if not source_exp else source_exp,
                    source_commit=source_commit,
                    calibrated_on=calibrated_on,
                    notes=f"Overridden via environment variable {env_k}",
                )

        # Tier 4: Runtime / per-request overrides
        for ov_k, ov_v in self.runtime_overrides.items():
            # Normalize key
            k_norm = ov_k.lower().strip()
            matched_name = None
            for reg_name in self.params.keys():
                if reg_name.lower() == k_norm or reg_name.lower().replace(".", "_") == k_norm.replace(".", "_"):
                    matched_name = reg_name
                    break
            if not matched_name:
                matched_name = ov_k

            curr = self.params.get(matched_name)
            default_val = curr.default_passthrough if curr else None
            self.params[matched_name] = CalibratedParam(
                name=matched_name,
                value=ov_v,
                default_passthrough=default_val,
                source_exp="runtime_override",
                source_commit=curr.source_commit if curr else None,
                calibrated_on=curr.calibrated_on if curr else None,
                notes=f"Overridden at runtime: {ov_v}",
            )

    @staticmethod
    def _parse_env_value(val_str: str) -> Any:
        """Parses an environment variable string into Python primitive."""
        clean = val_str.strip()
        if clean.lower() in ("true", "1", "yes"):
            return True
        if clean.lower() in ("false", "0", "no"):
            return False
        if clean.lower() in ("none", "null"):
            return None
        try:
            return int(clean)
        except ValueError:
            pass
        try:
            return float(clean)
        except ValueError:
            pass
        try:
            return json.loads(clean)
        except Exception:
            pass
        return clean

    def get(self, name: str, default: Any = None) -> Any:
        """Returns the active resolved value for parameter name."""
        if name in self.params:
            return self.params[name].value
        # Check targets.* prefix
        if name.startswith("targets."):
            return default
        return default

    def get_param(self, name: str) -> Optional[CalibratedParam]:
        """Returns the full CalibratedParam metadata object."""
        return self.params.get(name)

    def is_calibrated(self, name: str) -> bool:
        """Returns True if the parameter has an empirically calibrated value."""
        p = self.params.get(name)
        if not p:
            return False
        return p.source_exp is not None and not p.source_exp.startswith("env:") and p.source_exp != "runtime_override"

    def with_overrides(self, overrides: Dict[str, Any]) -> MultiScaleConfig:
        """Returns a new MultiScaleConfig instance combining existing settings with new overrides."""
        merged_overrides = dict(self.runtime_overrides)
        merged_overrides.update(overrides)
        return MultiScaleConfig(
            profile_path=self.profile_path,
            runtime_overrides=merged_overrides,
        )

    # -------------------------------------------------------------------------
    # Convenience properties
    # -------------------------------------------------------------------------
    @property
    def query_extractor_strategy(self) -> str:
        return str(self.get("query_extractor.strategy", "passthrough"))

    @property
    def chunker_macro_target_tokens(self) -> Optional[int]:
        return self.get("chunker.macro_target_tokens")

    @property
    def chunker_micro_target_words(self) -> Optional[int]:
        return self.get("chunker.micro_target_words")

    @property
    def filter_strategy(self) -> str:
        return str(self.get("filter.strategy", "passthrough"))

    @property
    def filter_unit(self) -> str:
        return str(self.get("filter.unit", "micro"))

    @property
    def filter_keep_threshold(self) -> Optional[float]:
        return self.get("filter.keep_threshold")

    @property
    def filter_keep_top_k(self) -> Optional[int]:
        return self.get("filter.keep_top_k")

    @property
    def filter_keep_budget_tokens(self) -> Optional[int]:
        return self.get("filter.keep_budget_tokens")

    @property
    def fast_path_mode(self) -> str:
        return str(self.get("fast_path.mode", "passthrough"))

    @property
    def fast_path_hot_transduce_n(self) -> Optional[int]:
        return self.get("fast_path.hot_transduce_n")

    @property
    def fast_path_coverage_threshold(self) -> Optional[float]:
        return self.get("fast_path.coverage_threshold")

    @property
    def ppr_inter_scale_weight(self) -> Optional[float]:
        val = self.get("ppr.inter_scale_weight")
        return float(val) if val is not None else None

    @property
    def poprag_coarse_node_weight(self) -> Optional[float]:
        val = self.get("poprag.coarse_node_weight")
        return float(val) if val is not None else None

    @property
    def background_enabled(self) -> bool:
        return bool(self.get("background.enabled", False))

    @property
    def background_pause_policy(self) -> str:
        return str(self.get("background.pause_policy", "none"))

    @property
    def background_max_concurrency(self) -> int:
        return int(self.get("background.max_concurrency", 1))

    # -------------------------------------------------------------------------
    # Provenance Report & Profile Writing (§1)
    # -------------------------------------------------------------------------

    def provenance_report(self) -> str:
        """Outputs a clean Markdown table of every parameter and where its value came from.

        This table is pasted directly into every EVAL.md entry.
        """
        lines = [
            "| Parameter | Active Value | Default (Pass-through) | Source Experiment | Source Commit | Calibrated On | Notes |",
            "|---|---|---|---|---|---|---|",
        ]
        for name in sorted(self.params.keys()):
            p = self.params[name]
            val_str = f"`{p.value}`" if p.value is not None else "*None*"
            def_str = f"`{p.default_passthrough}`" if p.default_passthrough is not None else "*None*"
            src_exp = f"`{p.source_exp}`" if p.source_exp else "*Uncalibrated (B0)*"
            src_sha = f"`{p.source_commit[:7]}`" if (p.source_commit and len(p.source_commit) > 10) else (f"`{p.source_commit}`" if p.source_commit else "-")
            cal_on = p.calibrated_on or "-"
            notes = p.notes or "-"
            lines.append(f"| `{name}` | {val_str} | {def_str} | {src_exp} | {src_sha} | {cal_on} | {notes} |")
        return "\n".join(lines)

    @classmethod
    def write_calibrated(
        cls,
        name: str,
        value: Any,
        source_exp: str,
        source_commit: Optional[str] = None,
        calibrated_on: Optional[str] = None,
        notes: str = "",
        profile_path: Optional[Union[str, Path]] = None,
    ):
        """Persists a calibrated parameter to config/multi_scale_profile.json.

        Enforces strict calibration protocol:
        - source_exp is REQUIRED. Empty or None raises ValueError.
        - Humans do not write JSON by hand; calibration scripts invoke this method.
        """
        if not source_exp or not str(source_exp).strip():
            raise ValueError(
                f"Cannot calibrate parameter '{name}' without a valid source_exp! "
                "Every calibrated parameter must reference its producing experiment node (e.g. 'exp-028a')."
            )

        path = Path(profile_path) if profile_path else DEFAULT_PROFILE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        existing[name] = {
            "value": value,
            "source_exp": str(source_exp).strip(),
            "source_commit": str(source_commit).strip() if source_commit else None,
            "calibrated_on": str(calibrated_on).strip() if calibrated_on else None,
            "notes": str(notes).strip(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        logger.info(
            "Calibrated parameter '%s' = %s written to %s (source_exp=%s)",
            name, value, path, source_exp
        )
