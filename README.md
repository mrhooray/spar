# SPAR

**Structured Parallel Automated Research**

SPAR is a command-line tool for structured, parallel automated research, combining Monte Carlo tree search guidance with judgement from delegated research agents.

- **Structured**: explicit objective, candidate hypotheses, lineage, measurements, decisions, and artifacts
- **Parallel**: isolated candidate work proceeds concurrently

## Getting Started

Inside the target Git repository:

```sh
spar init SESSION_NAME
```

```text
Initialized session SESSION_NAME.

Edit:
  objective: /path/to/repo/.spar/SESSION_NAME/objective.md
  config:    /path/to/repo/.spar/SESSION_NAME/config.toml

Then run:
  spar start SESSION_NAME
```

```sh
spar start SESSION_NAME
```

Check progress or inspect candidates with:

```sh
spar status SESSION_NAME
spar top SESSION_NAME
spar inspect SESSION_NAME CANDIDATE_ID
```

Use `spar stop SESSION_NAME` or press `Ctrl-C` to request a stop. A repeated `spar start SESSION_NAME` continues a stopped or interrupted session.
