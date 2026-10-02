"""Sandboxed Python code execution harness for OpenAI HumanEval pass@1.

Provides safe, isolated evaluation of generated Python code in temporary subprocesses
with a strict timeout, capturing stdout, stderr, and assertion status.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("quanta.benchmarks.code_evaluator")


def clean_pre_code(pre_text: str) -> str:
    """Extracts valid Python code declarations before def entry_point, dropping conversational prose."""
    if not pre_text.strip():
        return ""
    lines = pre_text.split("\n")
    # Try finding the start of code by trimming prose lines from the top
    for start_i in range(len(lines)):
        candidate = "\n".join(lines[start_i:])
        if not candidate.strip():
            break
        try:
            ast.parse(candidate)
            return candidate.strip()
        except SyntaxError:
            continue
    # Fallback: retain lines that look like Python code statements
    filtered = [
        ln for ln in lines
        if ln.strip().startswith(("import ", "from ", "def ", "class ", "@", "#"))
    ]
    return "\n".join(filtered).strip()


def clean_code_body(code_str: str) -> str:
    """Trims trailing conversational prose from code if syntax error occurs."""
    if not code_str.strip():
        return ""
    try:
        ast.parse(code_str)
        return code_str.strip()
    except SyntaxError:
        pass

    lines = code_str.rstrip().split("\n")
    for end_i in range(len(lines) - 1, 0, -1):
        candidate = "\n".join(lines[:end_i]).rstrip()
        if not candidate:
            break
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return code_str.strip()


def extract_top_level_names(code: str) -> Set[str]:
    """Extracts top-level function, class, and variable names from code."""
    names: Set[str] = set()
    if not code.strip():
        return names
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
    except SyntaxError:
        for m in re.finditer(r"^\s*(?:def|class)\s+([a-zA-Z0-9_]+)", code, re.MULTILINE):
            names.add(m.group(1))
    return names


def extract_import_lines(code: str) -> Set[str]:
    """Extracts normalized import statements from code."""
    imports: Set[str] = set()
    for line in code.splitlines():
        trimmed = line.strip()
        if trimmed.startswith(("import ", "from ")):
            imports.add(trimmed)
    return imports


def deduplicate_prompt_helpers(prompt_helpers: str, extracted_code: str) -> str:
    """Filters prompt_helpers to remove functions/classes/imports already defined in extracted_code."""
    if not prompt_helpers.strip():
        return ""

    extracted_names = extract_top_level_names(extracted_code)
    extracted_imports = extract_import_lines(extracted_code)

    try:
        tree = ast.parse(prompt_helpers)
    except SyntaxError:
        return prompt_helpers.strip()

    kept_segments = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in extracted_names:
                continue
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in extracted_names for t in targets):
                continue
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(prompt_helpers, node)
            if seg and seg.strip() in extracted_imports:
                continue

        seg = ast.get_source_segment(prompt_helpers, node)
        if seg:
            kept_segments.append(seg)

    return "\n\n".join(kept_segments).strip()



@dataclass
class CodeEvalResult:
    """Diagnostic result of sandboxed code execution."""
    task_id: str
    passed: bool
    execution_time_s: float
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    is_harness_bug: bool = False
    harness_bug_detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "execution_time_s": round(self.execution_time_s, 4),
            "error_message": self.error_message,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "is_harness_bug": self.is_harness_bug,
            "harness_bug_detail": self.harness_bug_detail,
        }


class CodeEvaluator:
    """Executes HumanEval Python solutions against unit test assertions in an isolated sandbox."""

    def __init__(self, timeout_seconds: float = 5.0, sandbox_dir: Optional[Path] = None):
        self.timeout_seconds = timeout_seconds
        self.sandbox_dir = sandbox_dir or (Path("scratch") / "eval_sandbox")
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_python_code(completion: str, entry_point: Optional[str] = None) -> str:
        """Extracts executable Python code from LLM output, handling markdown blocks and indentation."""
        if not completion:
            return ""

        # Check for ```python ... ``` markdown block
        pattern = r"```(?:python)?\s*\n?([\s\S]*?)```"
        matches = re.findall(pattern, completion, re.IGNORECASE)
        if matches:
            chosen = matches[0]
            if entry_point:
                for block in matches:
                    if f"def {entry_point}" in block:
                        chosen = block
                        break
            raw_code = chosen
        else:
            raw_code = completion

        # If entry_point is not explicitly provided, attempt to infer from a top-level function definition
        effective_entry_point = entry_point
        if not effective_entry_point:
            m = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", raw_code)
            if m:
                effective_entry_point = m.group(1)

        # If it defines the function itself, extract while preserving preceding helpers/imports
        if effective_entry_point and f"def {effective_entry_point}" in raw_code:
            idx = raw_code.find(f"def {effective_entry_point}")
            pre_code = clean_pre_code(raw_code[:idx])
            body = clean_code_body(raw_code[idx:])
            return f"{pre_code}\n\n{body}".strip() if pre_code else body

        # Otherwise it's the function body to append to prompt
        lines = raw_code.rstrip().split("\n")
        processed_lines = []
        for line in lines:
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                processed_lines.append("    " + line)
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def evaluate_solution(
        self,
        task_id: str,
        prompt: str,
        completion: str,
        test_code: str,
        entry_point: str,
    ) -> CodeEvalResult:
        """Evaluates a code completion against test_code in a sandboxed subprocess."""
        extracted_code = self.extract_python_code(completion, entry_point)

        # Standard typing and mathematical library preamble
        preamble = "from typing import *\nimport math\nimport re\nimport collections\n\n"

        # Full test script construction:
        if entry_point and f"def {entry_point}" in extracted_code:
            # Extract prompt helper declarations preceding def {entry_point}
            if f"def {entry_point}" in prompt:
                idx = prompt.find(f"def {entry_point}")
                raw_helpers = prompt[:idx].rstrip() if idx != -1 else ""
            else:
                raw_helpers = ""

            prompt_helpers = deduplicate_prompt_helpers(raw_helpers, extracted_code)
            helpers_block = f"{prompt_helpers}\n\n" if prompt_helpers else ""
            full_code = preamble + helpers_block + extracted_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
        else:
            full_code = preamble + prompt.rstrip() + "\n" + extracted_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

        # Write to temporary script in sandbox dir
        safe_task_id = re.sub(r"[^a-zA-Z0-9_]", "_", task_id)
        temp_file = (self.sandbox_dir / f"test_{safe_task_id}_{int(time.time() * 1000)}.py").resolve()

        try:
            temp_file.write_text(full_code, encoding="utf-8")

            t0 = time.perf_counter()
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": os.path.abspath("src"),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            }

            process = subprocess.run(
                [sys.executable, str(temp_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(self.sandbox_dir.resolve()),
                env=env,
            )
            elapsed = time.perf_counter() - t0

            passed = (process.returncode == 0)
            err = None if passed else (process.stderr.strip() or f"Process exited with code {process.returncode}")

            # Real-time forensic diagnostic: check if failure was caused by harness stripping prompt helpers
            is_harness_bug = False
            harness_bug_detail = None
            if not passed and err and "NameError" in err and entry_point and f"def {entry_point}" in extracted_code:
                m = re.search(r"NameError:\s+name\s+'([^']+)'\s+is\s+not\s+defined", err)
                if m:
                    missing_name = m.group(1)
                    if missing_name in prompt:
                        # Re-run with prompt helpers preceding entry_point to verify whether code passes
                        idx = prompt.find(f"def {entry_point}")
                        prompt_helpers = prompt[:idx].rstrip() if idx != -1 else prompt.rstrip()
                        full_code_with_prompt = (
                            preamble + (prompt_helpers + "\n\n" if prompt_helpers else "") + extracted_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
                        )
                        temp_file_retry = (self.sandbox_dir / f"test_{safe_task_id}_retry_{int(time.time() * 1000)}.py").resolve()
                        try:
                            temp_file_retry.write_text(full_code_with_prompt, encoding="utf-8")
                            proc_retry = subprocess.run(
                                [sys.executable, str(temp_file_retry)],
                                capture_output=True,
                                text=True,
                                timeout=self.timeout_seconds,
                                cwd=str(self.sandbox_dir.resolve()),
                                env=env,
                            )
                            if proc_retry.returncode == 0:
                                is_harness_bug = True
                                harness_bug_detail = (
                                    f"Harness stripped prompt preamble containing '{missing_name}'; "
                                    f"solution verified passing when prompt preamble is preserved."
                                )
                        except Exception:
                            pass
                        finally:
                            if temp_file_retry.exists():
                                try:
                                    temp_file_retry.unlink()
                                except OSError:
                                    pass

            return CodeEvalResult(
                task_id=task_id,
                passed=passed,
                execution_time_s=elapsed,
                error_message=err,
                stdout=process.stdout,
                stderr=process.stderr,
                is_harness_bug=is_harness_bug,
                harness_bug_detail=harness_bug_detail,
            )

        except subprocess.TimeoutExpired:
            return CodeEvalResult(
                task_id=task_id,
                passed=False,
                execution_time_s=self.timeout_seconds,
                error_message=f"Timeout expired after {self.timeout_seconds} seconds (possible infinite loop)",
            )
        except Exception as e:
            return CodeEvalResult(
                task_id=task_id,
                passed=False,
                execution_time_s=0.0,
                error_message=f"Execution error: {str(e)}",
            )
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    @staticmethod
    def compute_pass_at_k(results: List[CodeEvalResult]) -> float:
        """Computes pass@1 metric as the fraction of successfully passing test cases."""
        if not results:
            return 0.0
        passed = sum(1 for r in results if r.passed)
        return round((passed / len(results)) * 100.0, 2)
