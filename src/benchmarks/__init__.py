"""QUANTA Paired & Ablated Benchmarking Suite.

Provides publication-grade, reproducible evaluation harnesses comparing:
1. Base LLM Standalone (Unsloth Studio :8888)
2. Base LLM + QUANTA Local (Ephemeral ASG)
3. Base LLM + QUANTA + Mounted 14GB Wikidata KB (wikipedia_quanta.db)
"""

from __future__ import annotations

from .code_evaluator import CodeEvaluator, CodeEvalResult
from .latency_profiler import LatencyProfiler, ExtrapolationForecast
from .metrics import BenchmarkMetrics, ComparativeScorecard
from .paired_evaluator import PairedEvaluator, PairedResult, EvaluationCondition
from .publication_exporter import PublicationExporter
from .suite_loaders import BenchmarkSuiteLoader, BenchmarkSample

__all__ = [
    "CodeEvaluator",
    "CodeEvalResult",
    "LatencyProfiler",
    "ExtrapolationForecast",
    "BenchmarkMetrics",
    "ComparativeScorecard",
    "PairedEvaluator",
    "PairedResult",
    "EvaluationCondition",
    "PublicationExporter",
    "BenchmarkSuiteLoader",
    "BenchmarkSample",
]
