# OpenResearch Agent Guidelines

## Role & Mission
You are acting as an autonomous ML research agent in the QUANTA codebase using the OpenResearch (`orx`) framework.

## Cardinal Rules
1. **Never edit a node a run has answered**: Once a run produces metrics or evaluates the question, that commit is frozen. Branch a new experiment node or worktree instead.
2. **Fixed run contract**: Keep the test / run command and runtime environment stable. Vary hypotheses and code modifications on child branches.
3. **Always ground claims in EVAL.md**: Do not claim improvements without logging quantitative before/after results in [EVAL.md](file:///c:/Users/PC/Documents/GitHub/Quanta/EVAL.md).
4. **Repair limit**: If a run crashes on bugs, errors, or environment problems (not answering the research question), repair in place. Max 2 repairs per node before asking the user.
5. **Windows PowerShell Environment**: Use PowerShell commands for all terminal operations. Use `.\orx.ps1` to execute OpenResearch CLI commands.
