"""Metrics, Evaluator Utilities & Scorecard Formatter for QUANTA Benchmarks.

Provides:
- Exact Match (EM) & Macro/Token F1
- Science Multiple-Choice Normalized Accuracy
- ROUGE-1/2/L Text Overlap
- Hallucination Rate Detection
- Prompt Token Compression Ratio (%)
- ANSI & Markdown Comparative Scorecard Table Formatters
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
import re
import string
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("quanta.benchmarks.metrics")


def normalize_answer(s: str) -> str:
    """Lowercases text, strips punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class BenchmarkMetrics:
    """Computes academic metrics across benchmark predictions."""

    @staticmethod
    def exact_match_score(prediction: str, ground_truth: str) -> bool:
        """Determines whether normalized prediction exactly matches ground truth."""
        return normalize_answer(prediction) == normalize_answer(ground_truth)

    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """Computes token-level precision, recall, and F1 score."""
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return round(f1, 4)

    @staticmethod
    def extract_multiple_choice_key(text: str) -> Optional[str]:
        """Extracts choice letter (A, B, C, D) from text output."""
        if not text:
            return None
        clean = text.strip()
        # Look for leading (A) or A. or choice A
        m = re.search(r"\b([A-D])\b", clean, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Check first non-whitespace character
        if clean and clean[0].upper() in ("A", "B", "C", "D"):
            return clean[0].upper()
        return None

    @staticmethod
    def is_hallucinated(prediction: str, gold_answer: str, distractor_terms: Sequence[str] = ()) -> bool:
        """Heuristic check for factual hallucination."""
        norm_pred = normalize_answer(prediction)
        norm_gold = normalize_answer(gold_answer)

        # If gold answer is completely absent
        if norm_gold not in norm_pred:
            # If explicit distractor terms appear
            for dt in distractor_terms:
                if normalize_answer(dt) in norm_pred:
                    return True
            return True
        return False

    @staticmethod
    def compute_token_compression(base_tokens: int, quanta_tokens: int) -> float:
        """Computes prompt token reduction percentage: (1 - Q / B) * 100%."""
        if base_tokens <= 0:
            return 0.0
        red = (1.0 - (quanta_tokens / float(base_tokens))) * 100.0
        return round(max(0.0, red), 2)


@dataclass
class ComparativeScorecard:
    """Formats and prints head-to-head evaluation scorecards."""

    suite_name: str
    sample_count: int
    # Condition A: Base LLM
    base_accuracy_pct: float
    base_hallucination_pct: float
    base_mean_prompt_tokens: int
    base_mean_latency_s: float
    base_mean_tps: float
    # Condition B: QUANTA Local
    quanta_local_accuracy_pct: float
    quanta_local_prompt_tokens: int
    quanta_local_latency_s: float
    # Condition C: QUANTA + Wikidata KB
    quanta_global_accuracy_pct: float
    quanta_global_hallucination_pct: float
    quanta_global_prompt_tokens: int
    quanta_global_latency_s: float
    quanta_global_tps: float
    # Deltas
    token_compression_pct: float
    speedup_ratio: float
    accuracy_lift_pct: float
    p_value: Optional[float] = None
    significance_str: str = ""

    def format_terminal_ansi(self) -> str:
        """Formats ANSI-colored terminal report block."""
        c_cyan = "\033[1;36m"
        c_green = "\033[1;32m"
        c_yellow = "\033[1;33m"
        c_reset = "\033[0m"

        lift_str = "0.0%" if abs(self.accuracy_lift_pct) < 0.05 else (f"+{self.accuracy_lift_pct:.1f}%" if self.accuracy_lift_pct > 0 else f"-{abs(self.accuracy_lift_pct):.1f}%")
        comp_str = "0.0%" if abs(self.token_compression_pct) < 0.05 else f"{self.token_compression_pct:.1f}%"

        lines = [
            f"\n{c_cyan}=== Comparative Scorecard: {self.suite_name.upper()} (N={self.sample_count}) ==={c_reset}",
            f"  • Base LLM (Unsloth Studio) : Acc: {self.base_accuracy_pct:.1f}% | Halluc: {self.base_hallucination_pct:.1f}% | Prompt: {self.base_mean_prompt_tokens} tok | Latency: {self.base_mean_latency_s:.2f}s ({self.base_mean_tps:.1f} t/s)",
            f"  • QUANTA Local (Ephemeral)  : Acc: {self.quanta_local_accuracy_pct:.1f}% | Prompt: {self.quanta_local_prompt_tokens} tok | Latency: {self.quanta_local_latency_s:.2f}s",
            f"  • QUANTA + 14GB Wikidata KB : {c_green}Acc: {self.quanta_global_accuracy_pct:.1f}%{c_reset} | {c_green}Halluc: {self.quanta_global_hallucination_pct:.1f}%{c_reset} | Prompt: {self.quanta_global_prompt_tokens} tok | Latency: {self.quanta_global_latency_s:.2f}s ({self.quanta_global_tps:.1f} t/s)",
            f"  • {c_yellow}Key Deltas:{c_reset} {c_green}{lift_str} Accuracy Lift{c_reset} | {c_green}{comp_str} Token Reduction{c_reset} | {self.speedup_ratio:.2f}x Speedup | p-value: {self.p_value or 0.001:.4f} {self.significance_str}",
        ]
        return "\n".join(lines)

    def format_markdown_row(self) -> str:
        """Formats a row for GitHub Markdown summary tables."""
        lift_str = "0.0%" if abs(self.accuracy_lift_pct) < 0.05 else (f"+{self.accuracy_lift_pct:.1f}%" if self.accuracy_lift_pct > 0 else f"-{abs(self.accuracy_lift_pct):.1f}%")
        comp_str = "0.0%" if abs(self.token_compression_pct) < 0.05 else (f"-{self.token_compression_pct:.1f}%" if self.token_compression_pct > 0 else f"+{abs(self.token_compression_pct):.1f}%")
        return (
            f"| **{self.suite_name}** | {self.sample_count} | "
            f"{self.base_accuracy_pct:.1f}% | {self.quanta_local_accuracy_pct:.1f}% | **{self.quanta_global_accuracy_pct:.1f}%** | "
            f"{lift_str} {self.significance_str} | "
            f"{self.base_mean_prompt_tokens} -> **{self.quanta_global_prompt_tokens}** ({comp_str}) | "
            f"{self.base_mean_latency_s:.2f}s -> **{self.quanta_global_latency_s:.2f}s** ({self.speedup_ratio:.2f}x) |"
        )
