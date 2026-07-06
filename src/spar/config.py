from pathlib import Path
from typing import Any
import tomllib

from .errors import SparError


DEFAULT_MAX_CANDIDATES = 64
DEFAULT_MAX_PARALLEL = 4


DEFAULT_OBJECTIVE = """# Objective

Describe the optimization target, success criterion, and constraints. The
evaluator's finite numeric `score` must be higher for better results.
"""


DEFAULT_CONFIG = f"""# Defaulted session settings. Change as needed.
max_candidates = {DEFAULT_MAX_CANDIDATES}
# Maximum candidate subagents reserved at once.
max_parallel = {DEFAULT_MAX_PARALLEL}

[evaluation]
# Canonical evaluation command. Confirm or replace before starting research.
# Stdout must be a JSON object with a finite numeric `score` field, where higher
# is better. Any additional fields are preserved in the evaluation artifact.
# Keep non-ignored, untracked files out of the candidate worktree.
command = ["/path/to/eval.sh"]

# Optional. Uncomment to make a profiling command available during research.
# Profiling stdout and stderr are captured. Write profiling artifacts under
# the SPAR_PROFILING_DIR environment path rather than the candidate worktree.
# [profiling]
# command = ["./profiling.sh"]
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
    for name in ("max_candidates", "max_parallel"):
        if name not in raw:
            raise SparError(f"configuration is missing required field: {name}")
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, dict) or "command" not in evaluation:
        raise SparError("configuration is missing required field: evaluation.command")
    profiling = raw.get("profiling", {})
    if not isinstance(profiling, dict):
        raise SparError("profiling must be a table")
    config = {
        "max_candidates": raw["max_candidates"],
        "max_parallel": raw["max_parallel"],
        "evaluation": {"command": evaluation["command"]},
        "profiling": {"command": profiling.get("command")},
    }
    for name in ("max_candidates", "max_parallel"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise SparError(f"{name} must be a positive integer")
    if not _valid_command(config["evaluation"]["command"]):
        raise SparError("evaluation.command must be a non-empty string array")
    if config["profiling"]["command"] is not None and not _valid_command(config["profiling"]["command"]):
        raise SparError("profiling.command must be a non-empty string array when provided")
    return config


def _valid_command(command: Any) -> bool:
    return (
        isinstance(command, list)
        and bool(command)
        and all(isinstance(part, str) and part for part in command)
    )
