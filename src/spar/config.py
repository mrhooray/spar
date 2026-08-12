import math
import tomllib
from pathlib import Path
from typing import Any

from .error import SparError
from .mcts import DEFAULT_EXPLORATION_CONSTANT

DEFAULT_MAX_CANDIDATES = 64
DEFAULT_MAX_PARALLEL = 4
DEFAULT_AGENT_TIMEOUT_SECONDS = 3600
DEFAULT_COMMAND_TIMEOUT_SECONDS = 3600


DEFAULT_OBJECTIVE = """# Objective

## Goal

Describe what should improve and why.

## Success criterion

Describe how success is measured. The evaluator must return a finite numeric
`score`; higher scores are better.

## Constraints

List requirements, invariants, or limits that must be preserved.
"""


DEFAULT_CONFIG = f"""# Replace the required placeholders before starting.
# Other settings have usable defaults; leave them or change them as needed.

# Defaults. Change or omit these as needed.
max_candidates = {DEFAULT_MAX_CANDIDATES}
# Maximum candidate subagents reserved at once.
max_parallel = {DEFAULT_MAX_PARALLEL}

[mcts]
# UCT exploration constant; sqrt(2) is the standard default.
exploration_constant = {DEFAULT_EXPLORATION_CONSTANT}

[agent]
# REQUIRED: replace all three values.
# Supported CLIs: claude-code, codex, opencode, pi.
cli = "REPLACE_WITH_CLI" # "pi"
model = "REPLACE_WITH_MODEL" # "openai/gpt-5.6-luna"
effort = "REPLACE_WITH_EFFORT" # "high"
timeout_seconds = {DEFAULT_AGENT_TIMEOUT_SECONDS}

[evaluation]
# REQUIRED: replace with the canonical evaluation command.
# Stdout must be a JSON object with a finite numeric `score` field, where higher
# is better. Stderr is captured; additional JSON fields are preserved.
# Keep non-ignored, untracked files out of the candidate worktree.
command = ["REPLACE_WITH_EVALUATION_COMMAND"] # ["/path/to/eval.sh"]
timeout_seconds = {DEFAULT_COMMAND_TIMEOUT_SECONDS}

# Optional. Uncomment to make a profiling command available during research.
# Stdout must be a JSON object. Stderr is captured.
# Write additional evidence under SPAR_PROFILING_DIR rather than the worktree.
# [profiling]
# command = ["./profiling.sh"]
# timeout_seconds = {DEFAULT_COMMAND_TIMEOUT_SECONDS}
"""


def load_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.toml"
    try:
        with config_path.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise SparError(f"configuration does not exist: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SparError(f"configuration is invalid TOML: {config_path}: {exc}") from exc
    agent = raw.get("agent")
    if not isinstance(agent, dict):
        raise SparError("configuration is missing required table: agent")
    for name in ("cli", "model", "effort"):
        if name not in agent:
            raise SparError(f"configuration is missing required field: agent.{name}")
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, dict) or "command" not in evaluation:
        raise SparError("configuration is missing required field: evaluation.command")
    profiling = raw.get("profiling", {})
    if not isinstance(profiling, dict):
        raise SparError("profiling must be a table")
    mcts = raw.get("mcts", {})
    if not isinstance(mcts, dict):
        raise SparError("mcts must be a table")
    config = {
        "max_candidates": raw.get("max_candidates", DEFAULT_MAX_CANDIDATES),
        "max_parallel": raw.get("max_parallel", DEFAULT_MAX_PARALLEL),
        "mcts": {
            "exploration_constant": mcts.get("exploration_constant", DEFAULT_EXPLORATION_CONSTANT),
        },
        "agent": {
            "cli": agent["cli"],
            "model": agent["model"],
            "effort": agent["effort"],
            "timeout_seconds": agent.get("timeout_seconds", DEFAULT_AGENT_TIMEOUT_SECONDS),
        },
        "evaluation": {
            "command": evaluation["command"],
            "timeout_seconds": evaluation.get("timeout_seconds", DEFAULT_COMMAND_TIMEOUT_SECONDS),
        },
        "profiling": {
            "command": profiling.get("command"),
            "timeout_seconds": profiling.get("timeout_seconds", DEFAULT_COMMAND_TIMEOUT_SECONDS),
        },
    }
    for name in ("max_candidates", "max_parallel"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise SparError(f"{name} must be a positive integer")
    exploration_constant = config["mcts"]["exploration_constant"]
    if (
        isinstance(exploration_constant, bool)
        or not isinstance(exploration_constant, (int, float))
        or not math.isfinite(exploration_constant)
        or exploration_constant <= 0
    ):
        raise SparError("mcts.exploration_constant must be a positive finite number")
    for name in ("cli", "model", "effort"):
        value = config["agent"][name]
        if not isinstance(value, str) or not value.strip():
            raise SparError(f"agent.{name} must be a non-empty string")
    for section in ("agent", "evaluation", "profiling"):
        timeout = config[section]["timeout_seconds"]
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
            raise SparError(f"{section}.timeout_seconds must be a positive integer")
    if not _valid_command(config["evaluation"]["command"]):
        raise SparError("evaluation.command must be a non-empty string array")
    if config["profiling"]["command"] is not None and not _valid_command(config["profiling"]["command"]):
        raise SparError("profiling.command must be a non-empty string array when provided")
    return config


def _valid_command(command: Any) -> bool:
    return isinstance(command, list) and bool(command) and all(isinstance(part, str) and part for part in command)
