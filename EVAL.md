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
| `exp-002` | `test_unsloth_example_generation` | `exp-001` | `50c7f21` | Fix ASG multi-band slot starvation & backtranslation breakdown on complex corpora; ground 1024-D conceptual/epistemic vectors (Bands 0, 3..7); enforce Merkle DAG acyclicity on temporal clauses; restore honest NLG determiner preservation on modified nouns and honorific sentence segmentation | Complex: 0/10 -> 10/10 passed (100%), Round-trip canonical Hamming distance: drift -> 0 (11/11 passed), Full test suite: 376/376 passed (100%) | Success | Promote & Merge to `main` |
| `exp-003b` | `exp/canonical-node-interning` | `exp-002` | `8df1733` | Decouple ephemeral Band 2 registers from node CIDs and apply global hash-consing interner to maximize node reuse across chunks and translations | Cross-chunk node reuse: < 5% -> 73.3% (narrative) / 90.0% (repetition), Node allocation reduction: 0% -> 73.3%, Tests: 380/380 -> 387/387 passed (100%), Canonical Hamming drift: d_H = 0 | Success | Ready to merge / Proceed to Section 2 |

---

## Round Guidelines

1. **Before Launching**: Record the proposed row with `Verdict: Pending`.
2. **After Completion**: Update the row with exact metric diffs (`Before -> After`), commit SHA, and verdict (`Success`, `Regression`, `Inconclusive`, `Crash`).
3. **Decide Next Move**:
   - **Promote**: Best-performing variant becomes parent for the next round.
   - **Refill**: If inconclusive or small regression, try an alternative sibling hypothesis under the same parent.
   - **Repair**: If run crashed due to syntax/runtime error without producing metrics (max 2 repairs).
   - **Stop**: If 3 consecutive rounds fail to produce gains.
