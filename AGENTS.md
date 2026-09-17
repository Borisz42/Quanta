# QUANTA Autonomous Research Protocol (OpenResearch + Google Antigravity)

This repository integrates **OpenResearch (`orx`)** with **Google Antigravity** to run structured, reproducible, autonomous ML research using your Google AI Pro subscription quota.

## The Autonomous Research Loop

When acting as an autonomous ML research agent, follow this 5-step cycle for every experiment round:

1. **Formulate Hypothesis**:
   - Inspect [EVAL.md](file:///c:/Users/PC/Documents/GitHub/Quanta/EVAL.md) to review baseline metrics and past experiment runs.
   - Propose a single, specific, falsifiable hypothesis (e.g., hyperparameter adjustment, vector dimensionality change, bundling/binding heuristic, loss modification).
2. **Worktree / Branch Isolation**:
   - Do not collide on `main`. Create an isolated Git branch or worktree:
     ```powershell
     git worktree add ../exp-<name> -b exp/<name>
     ```
   - Or create an OpenResearch experiment node:
     ```powershell
     .\orx.ps1 create-experiment <projectId> --parent <parentExpId> --title "<Hypothesis Title>" --description "<Concrete change & measurement>"
     ```
3. **Apply Code Modifications**:
   - Implement only the minimal code changes necessary to test the hypothesis.
   - Verify code compiles and passes syntax/lint checks before launching.
4. **Execute Run**:
   - Execute the training, benchmark, or evaluation run using `.\orx.ps1 exp run` or the project runner:
     ```powershell
     .\orx.ps1 exp run <expId>
     # Or native test / benchmark runner:
     pytest tests/
     .\scripts\dev.ps1 test
     ```
5. **Inspect & Record in EVAL.md**:
   - Inspect stdout and log outputs with `.\orx.ps1 logs <runId>`.
   - Record the baseline vs. variant metrics, commit SHA, and outcome verdict in [EVAL.md](file:///c:/Users/PC/Documents/GitHub/Quanta/EVAL.md).
   - Determine the next move: **repair** (if crashed/unanswered), **refill** (try an alternate sibling), **promote** (winner descends into next round), or **stop**.

---

## Experiment Tree Governance

To prevent chaotic exploration, enforce the following tree structure:

- **Stacked Bushes Shape**:
  - Run a small fan of co-equal options per round (2–4 variants).
  - Compare the results, pick the winning variant, promote it as the new parent, and branch the next generation from it.
  - Avoid a flat fan (dozens of siblings without decisions) and avoid a noodle (endless single-node chains without comparison).
- **Frozen Nodes**:
  - Once a run produces a valid numerical/evaluation answer (good, bad, or neutral), the node is **permanently frozen**.
  - Never edit code on an answered node; branch a new child node to explore modifications.
- **Provisional Nodes & Repair Cap**:
  - A run that crashes with an unexpected bug, syntax error, or environment issue has *not* answered the question.
  - Fix the node's branch in place and re-run.
  - **Hard limit**: Maximum 2 repair runs on the same node. If it still crashes after 2 repairs, stop and consult the user.

---

## Tooling & Command Reference (Windows PowerShell)

OpenResearch CLI is available via the root PowerShell wrapper `.\orx.ps1`:

| Task | Command |
|---|---|
| Dashboard & local server | `.\orx.ps1 up` (starts UI at `http://127.0.0.1:4791`) |
| List projects | `.\orx.ps1 projects` |
| View experiment tree | `.\orx.ps1 project view <projectId>` |
| Run experiment | `.\orx.ps1 exp run <expId>` |
| View run logs | `.\orx.ps1 logs <runId>` |
| Load OpenResearch guide | `.\orx.ps1 skill` / `.\orx.ps1 skill <module-name>` |
| Literature search | `.\orx.ps1 discover keyword "<query>"` / `.\orx.ps1 paper <id>` |
| QUANTA test runner | `.\scripts\dev.ps1 test` / `pytest` |

---

## Utilizing Google AI Pro Quota

- Your interactive Antigravity CLI and Antigravity IDE sessions are powered by your Google AI Pro subscription quota across frontier models (e.g., Gemini Pro and Flash).
- All literature search, hypothesis formulation, code modifications, run supervision, and metric logging can be driven entirely through Antigravity agents without external API credit billing.
- Use the `/boost` command on complex theoretical, mathematical, or debugging hurdles to invoke deep multi-step reasoning.
