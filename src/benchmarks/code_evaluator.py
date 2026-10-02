"""Sandboxed Python code execution harness for OpenAI HumanEval pass@1.

Provides safe, isolated evaluation of generated Python code in temporary subprocesses
with a strict timeout, capturing stdout, stderr, and assertion status.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("quanta.benchmarks.code_evaluator")


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
        raw_code = matches[0] if matches else completion

        # If it defines the function itself, extract from 'def entry_point' while retaining imports
        if entry_point and f"def {entry_point}" in raw_code:
            idx = raw_code.find(f"def {entry_point}")
            pre_lines = [ln for ln in raw_code[:idx].split("\n") if ln.strip().startswith(("import ", "from "))]
            pre_imports = "\n".join(pre_lines)
            body = raw_code[idx:].rstrip()
            return f"{pre_imports}\n{body}".strip() if pre_imports else body

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
            full_code = preamble + extracted_code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
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
