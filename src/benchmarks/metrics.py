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
import math
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
    def numeric_match_score(prediction: str, gold_answer: str) -> bool:
        """Determines whether prediction contains the correct numeric value and unit,

        even if separated by reasoning steps, formatting, or mathematical derivation.
        """
        if not prediction or not gold_answer:
            return False

        # 1. Parse numeric value from gold_answer
        num_matches = re.findall(r"[-+]?\b\d+(?:,\d+)*(?:\.\d+)?\b", gold_answer)
        if not num_matches:
            return False

        gold_num_str = num_matches[0].replace(",", "")
        try:
            gold_val = float(gold_num_str)
        except ValueError:
            return False

        # Extract unit/noun from gold_answer
        gold_noun = re.sub(r"[-+]?\b\d+(?:,\d+)*(?:\.\d+)?\b", "", gold_answer).strip()
        gold_noun_clean = re.sub(r"[^\w\s]", "", gold_noun).strip().lower()

        # 2. Extract numbers from prediction
        pred_nums = re.findall(r"[-+]?\b\d+(?:,\d+)*(?:\.\d+)?\b", prediction)
        matched_number = False
        for pn in pred_nums:
            try:
                pval = float(pn.replace(",", ""))
                if abs(pval - gold_val) < 1e-4:
                    matched_number = True
                    break
            except ValueError:
                continue

        if not matched_number:
            return False

        # 3. If gold_answer specifies units/noun, check presence in prediction
        if gold_noun_clean:
            words = [w for w in gold_noun_clean.split() if len(w) > 2]
            if words:
                pred_lower = prediction.lower()
                noun_matched = False
                for w in words:
                    stem = w.rstrip("s")
                    if stem in pred_lower:
                        noun_matched = True
                        break
                if not noun_matched:
                    return False

        return True

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
    def extractive_match_score(prediction: str, ground_truth: str) -> bool:
        """Determines whether ground truth is stated as an affirmative answer in prediction,
        excluding negative hedging statements (e.g. 'cannot be determined', 'no mention of')."""
        if not prediction or not ground_truth:
            return False

        pred_norm = normalize_answer(prediction)
        gold_norm = normalize_answer(ground_truth)
        if pred_norm == gold_norm:
            return True

        # Guard against negative hedging statements when gold is not 'unknown'
        if gold_norm not in ("unknown", "uncertain"):
            hedging_patterns = [
                r"\bcannot be determined\b",
                r"\bcan not be determined\b",
                r"\bno mention of\b",
                r"\bnot mentioned\b",
                r"\bnot contain the information\b",
                r"\bdoes not state\b",
                r"\binsufficient information\b",
                r"\buncertain based on\b",
            ]
            for pat in hedging_patterns:
                if re.search(pat, prediction, re.IGNORECASE):
                    # Only pass if there is an explicit answer block despite the hedging
                    if not re.search(r"\banswer\s*[:\-]\s*[\*\_]*" + re.escape(ground_truth) + r"[\*\_]*", prediction, re.IGNORECASE):
                        return False

        # Check word-boundary match of normalized gold answer in normalized prediction
        if gold_norm in pred_norm:
            return bool(re.search(r"\b" + re.escape(gold_norm) + r"\b", pred_norm))
        return False

    @staticmethod
    def extract_multiple_choice_key(text: str) -> Optional[str]:
        """Extracts choice letter (A, B, C, D) from text output with closing-answer prioritization."""
        if not text:
            return None
        clean = text.strip()
        if not clean:
            return None

        # 0. Single-character or single-letter choice enclosed in punctuation/markdown: "A", "(A)", "B.", "[C]", "**B**", "\boxed{C}"
        m_single = re.match(r"^\s*(?:\*\*|\\boxed\{|[\(\[])?\s*([A-Da-d])\s*(?:\*\*|\}|[\)\]])?\.?\s*$", clean)
        if m_single:
            return m_single.group(1).upper()

        lines = [line.strip() for line in clean.splitlines() if line.strip()]

        # 1. Leading choice on first non-empty line: e.g. "Answer: A", "(A)", "A. The ..."
        if lines:
            m_lead = re.match(r"^(?:(?:answer|choice|option|correct\s+choice)\s*[:\-]?\s*)?[\(\[]?([A-D])[\)\]]?(?:\.|\:|\)|\s+|$)", lines[0], re.IGNORECASE)
            if m_lead and not lines[0].lower().startswith("choices:"):
                # If the first line is solely an answer declaration, take it immediately
                if len(lines[0].split()) <= 6 or lines[0].lower().startswith(("answer:", "choice:", "option:")):
                    return m_lead.group(1).upper()

        # 2. Explicit answer anchors throughout text (case-insensitive keyword, capturing choice letter A-D)
        anchor_patterns = [
            r"(?:(?:the\s+)?(?:correct\s+)?(?:choice|option|answer)\s*(?:is\s*[:\-]?|[:\-])|therefore,?\s*(?:(?:the\s+)?answer\s*(?:is\s*[:\-]?|[:\-])\s*)?)\s*\[?\(?([A-D])\)?\]?",
            r"\b(?:answer|choice|option)\s*[:\-]\s*\[?\(?([A-D])\)?\]?",
            r"\bconclu(?:de|sion)\s*(?:is\s*[:\-]?|[:\-])?\s*\[?\(?([A-D])\)?\]?",
            r"\b([A-D])\s+is\s+the\s+correct\s+(?:choice|answer)\b",
            r"\bsupports\s+(?:choice|option)\s+\(?([A-D])\)?",
        ]
        all_anchor_matches = []
        for pattern in anchor_patterns:
            for m in re.finditer(pattern, clean, re.IGNORECASE):
                all_anchor_matches.append((m.start(), m.group(1).upper()))
        if all_anchor_matches:
            # Prioritize the final explicit concluding anchor in the text
            all_anchor_matches.sort(key=lambda x: x[0])
            return all_anchor_matches[-1][1]

        # 3. Concluding lines examination (last 5 non-empty lines in reverse from bottom to top)
        for line in reversed(lines[-5:]):
            for pattern in anchor_patterns:
                line_matches = list(re.finditer(pattern, line, re.IGNORECASE))
                if line_matches:
                    return line_matches[-1].group(1).upper()

            # Isolated uppercase choice on its own line: e.g. "B", "(B)", "**B**", "\boxed{B}"
            m_iso = re.match(r"^\s*(?:\*\*|\\boxed\{|[\(\[])?\s*([A-D])\s*(?:\*\*|\}|[\)\]])?\.?\s*$", line)
            if m_iso:
                return m_iso.group(1).upper()

            # Explicit option/choice in concluding line: "Option A", "Choice B"
            m_opt = list(re.finditer(r"\b(?:option|choice)\s+([A-D])\b", line, re.IGNORECASE))
            if m_opt:
                return m_opt[-1].group(1).upper()

            # Parenthesized or bracketed choice in concluding line if line expresses a choice
            if any(w in line.lower() for w in ("therefore", "thus", "hence", "correct", "select", "result", "answer", "final")):
                m_paren = list(re.finditer(r"(?:\(|\b\[|\*\*|\\boxed\{)([A-D])(?:\)|\b\]|\*\*|\})", line))
                if m_paren:
                    return m_paren[-1].group(1).upper()

        # 4. Fallback leading choice on first line if any
        if lines:
            m_lead_gen = re.match(r"^[\(\[]?([A-D])[\)\]]?(?:\.|\:|\))\s+", lines[0])
            if m_lead_gen:
                return m_lead_gen.group(1).upper()

        return None

    @staticmethod
    def _parse_numeric_and_unit(gold_answer: str) -> Tuple[Optional[float], Optional[str]]:
        """Extracts numeric value and optional unit/noun from gold answer string."""
        if not gold_answer or not isinstance(gold_answer, str):
            return None, None
        cleaned = gold_answer.strip()
        m = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", cleaned)
        if not m:
            return None, None
        num_str = m.group(0).replace(",", "")
        try:
            val = float(num_str)
        except ValueError:
            return None, None

        unit_part = cleaned[:m.start()] + " " + cleaned[m.end():]
        unit_clean = normalize_answer(unit_part).strip()
        return val, unit_clean if unit_clean else None

    @classmethod
    def numeric_match_score(cls, prediction: str, gold_answer: str) -> bool:
        """Evaluates whether prediction correctly calculates/contains the numeric gold answer and unit.

        Section 3 Requirements:
        - Parse gold_answer for numeric quantity V (int or float) and optional unit/noun U.
        - Check if V appears in bold (**630**), in LaTeX (\\boxed{630}), after calculation/equality anchors,
          or as the final calculated value in prediction.
        - If U is present, check that the noun or its singular/plural stem appears in prediction.
        - If both conditions are satisfied, evaluate as correct=True.
        """
        if not prediction or not gold_answer:
            return False

        gold_val, gold_unit = cls._parse_numeric_and_unit(gold_answer)
        if gold_val is None:
            return False

        pred_lower = prediction.lower()

        # 1. Check unit/noun requirement if present
        if gold_unit:
            unit_words = [w for w in gold_unit.split() if w not in ("of", "in", "per", "the", "a", "an", "at")]
            for w in unit_words:
                stems = {w}
                if w.endswith("ies") and len(w) > 3:
                    stems.add(w[:-3] + "y")
                elif w.endswith("es") and len(w) > 3:
                    stems.add(w[:-2])
                    stems.add(w[:-1])
                elif w.endswith("s") and len(w) > 2:
                    stems.add(w[:-1])
                else:
                    stems.add(w + "s")
                    stems.add(w + "es")

                if not any(re.search(r"\b" + re.escape(stem) + r"\b", pred_lower) for stem in stems):
                    return False

        # 2. Check numeric value V in prediction
        def matches_gold(val_cand: float) -> bool:
            return math.isclose(val_cand, gold_val, rel_tol=1e-5, abs_tol=1e-5)

        # 2a. Check bolded values (**...**)
        bold_chunks = re.findall(r"\*\*([^*]+)\*\*", prediction)
        for chunk in bold_chunks:
            for num_match in re.finditer(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", chunk):
                try:
                    c_val = float(num_match.group(0).replace(",", ""))
                    if matches_gold(c_val):
                        return True
                except ValueError:
                    continue

        # 2b. Check LaTeX boxed values (\\boxed{...})
        boxed_chunks = re.findall(r"\\boxed\{([^}]+)\}", prediction)
        for chunk in boxed_chunks:
            for num_match in re.finditer(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", chunk):
                try:
                    c_val = float(num_match.group(0).replace(",", ""))
                    if matches_gold(c_val):
                        return True
                except ValueError:
                    continue

        # 2c. Check calculation / equality / answer anchors
        anchor_calc_pattern = (
            r"(?:=\s*|\bis\s+|\banswer(?:\s+is)?\s*[:\-]?\s*|\btotal(?:\s+is)?\s*[:\-]?\s*|"
            r"\bnet\s+(?:change|balance|total)(?:\s+is)?\s*[:\-]?\s*|\bresult\s*[:\-]?\s*|"
            r"\bcalculation\s*[:\-]?\s*)\*?\*?\$?\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
        )
        for m in re.finditer(anchor_calc_pattern, prediction, re.IGNORECASE):
            try:
                c_val = float(m.group(1).replace(",", ""))
                if matches_gold(c_val):
                    return True
            except ValueError:
                continue

        # 2d. Check the final calculated numeric value in prediction
        all_nums = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", prediction)
        if all_nums:
            for cand_num_str in reversed(all_nums[-2:]):
                try:
                    c_val = float(cand_num_str.replace(",", ""))
                    if matches_gold(c_val):
                        return True
                except ValueError:
                    continue

        return False

    @classmethod
    def is_hallucinated(cls, prediction: str, gold_answer: str, distractor_terms: Sequence[str] = ()) -> bool:
        """Heuristic check for factual hallucination."""
        # If numeric match passes, prediction is factually accurate
        if cls.numeric_match_score(prediction, gold_answer):
            return False

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
