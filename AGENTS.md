# Tooling
- Use uv for Python project dependencies, environments, and commands.

# Style
- Keep one authoritative representation of each piece of state; avoid duplicate or drifting copies.
- Pass only required context and derive values locally where practical.
- Return only consumed data; avoid speculative metadata, wrappers, and pass-through helpers.
- Keep coordinator control flow straightforward; let lower-level operations own their state transitions.

## Python
- Use modern Python language features supported by the project's minimum version in `pyproject.toml`.
    - Prefer built-in generic types (e.g. `list[str]`, `dict[str, int]`) over legacy aliases (e.g. `List`, `Dict`).
    - Prefer inline type parameters (e.g. `class Box[T]:`) over `TypeVar`.
- Do not add `from __future__ import annotations`; the project targets Python 3.14+.

# SPAR
## Persistence

- Avoid duplication; keep a single source of truth for each type of data:
  - SQLite: structured state
  - Artifacts: file-oriented data such as target-repository worktrees and profiling evidence
- Derive reports and views from authoritative data; do not use them to drive control flow.
