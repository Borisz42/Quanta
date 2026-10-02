"""Unit tests for CodeEvaluator: Prompt helper preservation, sandboxed scope isolation, and deduping."""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from benchmarks.code_evaluator import (
    CodeEvaluator,
    CodeEvalResult,
    clean_pre_code,
    clean_code_body,
    deduplicate_prompt_helpers,
    extract_top_level_names,
    extract_import_lines,
)


@pytest.fixture
def evaluator():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield CodeEvaluator(timeout_seconds=4.0, sandbox_dir=Path(tmp_dir))


def test_humaneval_10_quanta_completion(evaluator: CodeEvaluator):
    """Verifies HumanEval/10 passes when model generates self-contained def make_palindrome

    without redundantly defining the prompt helper is_palindrome (QUANTA failure mode resolved).
    """
    prompt = """

def is_palindrome(string: str) -> bool:
    \"\"\" Test if given string is a palindrome \"\"\"
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    \"\"\" Find the shortest palindrome that begins with a supplied string.
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
    'catac'
    >>> make_palindrome('cata')
    'catac'
    \"\"\"
"""
    test_code = """
def check(candidate):
    assert candidate('') == ''
    assert candidate('x') == 'x'
    assert candidate('xyz') == 'xyzyx'
    assert candidate('xyx') == 'xyx'
    assert candidate('jerry') == 'jerryrrej'
"""
    completion_quanta = """```python
def make_palindrome(string: str) -> str:
    if not string:
        return ''
    beginning_of_suffix = 0
    while not is_palindrome(string[beginning_of_suffix:]):
        beginning_of_suffix += 1
    return string + string[:beginning_of_suffix][::-1]
```"""
    res = evaluator.evaluate_solution(
        "HumanEval/10", prompt, completion_quanta, test_code, "make_palindrome"
    )
    assert res.passed is True
    assert res.error_message is None


def test_humaneval_10_base_llm_completion(evaluator: CodeEvaluator):
    """Verifies HumanEval/10 passes when model appends is_palindrome at the bottom (Base LLM pattern)."""
    prompt = """
def is_palindrome(string: str) -> bool:
    return string == string[::-1]

def make_palindrome(string: str) -> str:
    \"\"\" Docstring \"\"\"
"""
    test_code = """
def check(candidate):
    assert candidate('xyz') == 'xyzyx'
"""
    completion_base_llm = """```python
def make_palindrome(string: str) -> str:
    if not string:
        return ''
    beginning_of_suffix = 0
    while not is_palindrome(string[beginning_of_suffix:]):
        beginning_of_suffix += 1
    return string + string[:beginning_of_suffix][::-1]

def is_palindrome(string: str) -> bool:
    return string == string[::-1]
```"""
    res = evaluator.evaluate_solution(
        "HumanEval/10", prompt, completion_base_llm, test_code, "make_palindrome"
    )
    assert res.passed is True
    assert res.error_message is None


def test_humaneval_10_helper_at_top_completion(evaluator: CodeEvaluator):
    """Verifies HumanEval/10 passes when model defines helper before entry_point."""
    prompt = """
def is_palindrome(string: str) -> bool:
    return string == string[::-1]

def make_palindrome(string: str) -> str:
    \"\"\" Docstring \"\"\"
"""
    test_code = """
def check(candidate):
    assert candidate('cat') == 'catac'
"""
    completion_top = """```python
def is_palindrome(string: str) -> bool:
    return string == string[::-1]

def make_palindrome(string: str) -> str:
    if not string:
        return ''
    beginning_of_suffix = 0
    while not is_palindrome(string[beginning_of_suffix:]):
        beginning_of_suffix += 1
    return string + string[:beginning_of_suffix][::-1]
```"""
    res = evaluator.evaluate_solution(
        "HumanEval/10", prompt, completion_top, test_code, "make_palindrome"
    )
    assert res.passed is True
    assert res.error_message is None


def test_humaneval_10_body_only_completion(evaluator: CodeEvaluator):
    """Verifies HumanEval/10 passes when model supplies only indented function body."""
    prompt = """
def is_palindrome(string: str) -> bool:
    return string == string[::-1]

def make_palindrome(string: str) -> str:
    \"\"\" Docstring \"\"\"
"""
    test_code = """
def check(candidate):
    assert candidate('cat') == 'catac'
"""
    completion_body = """
    if not string:
        return ''
    beginning_of_suffix = 0
    while not is_palindrome(string[beginning_of_suffix:]):
        beginning_of_suffix += 1
    return string + string[:beginning_of_suffix][::-1]
"""
    res = evaluator.evaluate_solution(
        "HumanEval/10", prompt, completion_body, test_code, "make_palindrome"
    )
    assert res.passed is True
    assert res.error_message is None


def test_humaneval_32_poly_helper(evaluator: CodeEvaluator):
    """Verifies HumanEval/32 find_zero preserves poly helper from prompt."""
    prompt = """import math


def poly(xs: list, x: float):
    \"\"\" Evaluates polynomial \"\"\"
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])


def find_zero(xs: list):
    \"\"\" Find zero point \"\"\"
"""
    test_code = """
def check(candidate):
    assert round(candidate([1, 2]), 2) == -0.5
"""
    # Self-contained completion that uses poly helper
    completion = """```python
def find_zero(xs: list):
    import math
    begin, end = -1.0, 1.0
    while poly(xs, begin) * poly(xs, end) > 0:
        begin *= 2.0
        end *= 2.0
    while end - begin > 1e-10:
        center = (begin + end) / 2.0
        if poly(xs, center) * poly(xs, begin) > 0:
            begin = center
        else:
            end = center
    return begin
```"""
    res = evaluator.evaluate_solution(
        "HumanEval/32", prompt, completion, test_code, "find_zero"
    )
    assert res.passed is True
    assert res.error_message is None


def test_humaneval_38_encode_cyclic_helper(evaluator: CodeEvaluator):
    """Verifies HumanEval/38 decode_cyclic preserves encode_cyclic helper from prompt."""
    prompt = """
def encode_cyclic(s: str):
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)


def decode_cyclic(s: str):
    \"\"\" Decodes \"\"\"
"""
    test_code = """
def check(candidate):
    s = "abcdef"
    assert candidate(encode_cyclic(s)) == s
"""
    completion = """```python
def decode_cyclic(s: str):
    return encode_cyclic(encode_cyclic(s))
```"""
    res = evaluator.evaluate_solution(
        "HumanEval/38", prompt, completion, test_code, "decode_cyclic"
    )
    assert res.passed is True
    assert res.error_message is None


def test_humaneval_50_encode_shift_helper(evaluator: CodeEvaluator):
    """Verifies HumanEval/50 decode_shift preserves encode_shift helper from prompt."""
    prompt = """
def encode_shift(s: str):
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])


def decode_shift(s: str):
    \"\"\" Decodes \"\"\"
"""
    test_code = """
def check(candidate):
    s = "helloworld"
    assert candidate(encode_shift(s)) == s
"""
    completion = """```python
def decode_shift(s: str):
    return "".join([chr(((ord(ch) - 5 - ord("a")) % 26) + ord("a")) for ch in s])
```"""
    res = evaluator.evaluate_solution(
        "HumanEval/50", prompt, completion, test_code, "decode_shift"
    )
    assert res.passed is True
    assert res.error_message is None


def test_humaneval_64_fix_constant(evaluator: CodeEvaluator):
    """Verifies HumanEval/64 vowels_count preserves FIX constant preamble from prompt."""
    prompt = """
FIX = \"\"\"
Add more test cases.
\"\"\"

def vowels_count(s):
    \"\"\" Counts vowels \"\"\"
"""
    test_code = """
def check(candidate):
    assert candidate("abcde") == 2
    assert candidate("Alone") == 3
"""
    completion = """```python
def vowels_count(s):
    vowels = "aeiouAEIOU"
    n = sum(c in vowels for c in s)
    if s and (s[-1] == 'y' or s[-1] == 'Y'):
        n += 1
    return n
```"""
    res = evaluator.evaluate_solution(
        "HumanEval/64", prompt, completion, test_code, "vowels_count"
    )
    assert res.passed is True
    assert res.error_message is None


def test_clean_pre_code():
    """Verifies that clean_pre_code drops conversational prose while keeping code declarations."""
    prose_and_code = """Here is the helper function you asked for:
Sure, let's write it down!

def helper(x: int) -> int:
    return x * 2
"""
    cleaned = clean_pre_code(prose_and_code)
    assert "Here is the helper" not in cleaned
    assert "def helper" in cleaned
    assert "return x * 2" in cleaned


def test_clean_code_body_trailing_prose():
    """Verifies that clean_code_body removes trailing prose that would cause SyntaxError."""
    code_with_prose = """def solve(a: int) -> int:
    return a + 1

I hope this helps! Let me know if you need more details.
"""
    cleaned = clean_code_body(code_with_prose)
    assert "I hope this helps" not in cleaned
    assert "def solve" in cleaned


def test_deduplicate_prompt_helpers():
    """Verifies that deduplicate_prompt_helpers omits items already defined in completion."""
    prompt_helpers = """
import math
from typing import List

def is_palindrome(s: str) -> bool:
    return s == s[::-1]

CONST_VAL = 100
"""
    completion = """
import math

def is_palindrome(s: str) -> bool:
    return s == s[::-1]

def make_palindrome(s: str) -> str:
    return s
"""
    deduped = deduplicate_prompt_helpers(prompt_helpers, completion)
    # is_palindrome and import math should be omitted since they exist in completion
    assert "def is_palindrome" not in deduped
    assert "CONST_VAL = 100" in deduped
    assert "from typing import List" in deduped


def test_infinite_loop_timeout(evaluator: CodeEvaluator):
    """Verifies that code with infinite loop hits timeout cleanly."""
    prompt = "def loop_forever():\n"
    completion = """```python
def loop_forever():
    while True:
        pass
```"""
    test_code = "def check(candidate):\n    candidate()\n"
    res = evaluator.evaluate_solution("test/timeout", prompt, completion, test_code, "loop_forever")
    assert res.passed is False
    assert "Timeout expired" in (res.error_message or "")


def test_syntax_error_handling(evaluator: CodeEvaluator):
    """Verifies that broken syntax is flagged with error message."""
    prompt = "def broken():\n"
    completion = "```python\ndef broken():\n    return ;;;invalid;;\n```"
    test_code = "def check(candidate):\n    pass\n"
    res = evaluator.evaluate_solution("test/syntax", prompt, completion, test_code, "broken")
    assert res.passed is False
    assert "SyntaxError" in (res.error_message or "")


def test_compute_pass_at_k():
    """Verifies pass@k calculation accuracy."""
    res1 = CodeEvalResult("t1", True, 0.1)
    res2 = CodeEvalResult("t2", False, 0.1)
    assert CodeEvaluator.compute_pass_at_k([res1, res2]) == 50.0
    assert CodeEvaluator.compute_pass_at_k([res1]) == 100.0
    assert CodeEvaluator.compute_pass_at_k([]) == 0.0
