# QUANTA Evaluation & Experiment Lineage (EVAL.md)

This document tracks baseline benchmarks, experimental hypotheses, and empirical results across OpenResearch iterations. Every autonomous research cycle must record its findings here.

## Evaluation Protocol

- **Run Command**: `pytest` or `.\scripts\dev.ps1 test`
- **Primary Metrics**:
  - Test Suite Pass Rate (%)
  - Vector Representation Fidelity / Orthogonality
  - Binding & Unbinding Accuracy
  - Inference Latency (ms)

---

## Baseline (Root)

- **Date**: 2026-09-17
- **Commit**: `main`
- **Configuration**: Standard 1024-dimension quaternary vector architecture with default parameters.
- **Baseline Results**:
  - Tests: Passing (100%)
  - Architecture: Quaternary Vector Symbolic Architecture (1024-D)

---

## Experiment Lineage Table

| Run ID | Branch / Worktree | Parent | Commit SHA | Hypothesis & Modification | Metric (Before -> After) | Verdict | Next Action |
|---|---|---|---|---|---|---|---|
| `exp-000` | `main` | - | `main` | Initial baseline root | Baseline established | Baseline | Ready for Round 1 |
| `exp-001` | `main` | `exp-000` | `3e534dc` | Phase 1: Representation & Parsing Core (GBNF grammar, S-expression lexer/parser, AST converter, QuantaGraph bridge) | 268 passed -> 286 passed (100%) | Success | Promote to Phase 2 |

---

## Round Guidelines

1. **Before Launching**: Record the proposed row with `Verdict: Pending`.
2. **After Completion**: Update the row with exact metric diffs (`Before -> After`), commit SHA, and verdict (`Success`, `Regression`, `Inconclusive`, `Crash`).
3. **Decide Next Move**:
   - **Promote**: Best-performing variant becomes parent for the next round.
   - **Refill**: If inconclusive or small regression, try an alternative sibling hypothesis under the same parent.
   - **Repair**: If run crashed due to syntax/runtime error without producing metrics (max 2 repairs).
   - **Stop**: If 3 consecutive rounds fail to produce gains.
