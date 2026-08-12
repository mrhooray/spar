import hashlib
import time
from pathlib import Path
from typing import Any

from .. import mcts
from ..config import DEFAULT_CONFIG, DEFAULT_OBJECTIVE, load_config
from ..error import SparError
from ..lifecycle import (
    ROOT_CANDIDATE_ID,
    SessionStatus,
)
from ..storage import git
from ..storage.db import DB
from ..storage.file import require_session_dir, session_dir

DEFAULT_TOP_LIMIT = 3


def init(session_name: str) -> dict[str, Any]:
    repo = git.repo_root()
    _require_clean_worktree(repo)
    path = session_dir(repo, session_name)
    ref = f"refs/spar/{session_name}/{ROOT_CANDIDATE_ID}"
    if not git.valid_ref(repo, ref):
        raise SparError(f"session name is not valid in a Git ref: {session_name}")
    if path.exists():
        raise SparError(f"session already exists: {path}")

    (path / "artifacts" / "candidates" / ROOT_CANDIDATE_ID).mkdir(parents=True)
    (path / "worktrees").mkdir()
    (path / "objective.md").write_text(DEFAULT_OBJECTIVE, encoding="utf-8")
    (path / "config.toml").write_text(DEFAULT_CONFIG, encoding="utf-8")
    git.ensure_info_exclude(repo)

    with DB(path) as db:
        db.initialize(
            repo_path=str(repo),
            root_commit=git.head(repo),
        )
    return {
        "session_name": session_name,
        "session_dir": str(path),
        "objective_path": str(path / "objective.md"),
        "config_path": str(path / "config.toml"),
    }


def status(session_name: str) -> dict[str, Any]:
    repo, path, config = _session_context(session_name)
    with DB(path) as db:
        session = _session_status(db)
        snapshot = _status_snapshot(db, config["max_candidates"])
    candidates = snapshot["candidates"]
    return {
        "session_name": session_name,
        "session_dir": str(path),
        "repository": str(repo),
        "objective_path": str(path / "objective.md"),
        "config_path": str(path / "config.toml"),
        "objective_sha256": hashlib.sha256((path / "objective.md").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((path / "config.toml").read_bytes()).hexdigest(),
        "workspace_root": str(path / "worktrees"),
        "target_starting_head": candidates[0]["commit_sha"] if candidates else None,
        "max_parallel": config["max_parallel"],
        "evaluation": config["evaluation"],
        "profiling": config["profiling"],
        "session": session,
        **snapshot,
    }


def start(session_name: str) -> None:
    _, path, _ = _session_context(session_name)
    with DB(path) as db, db.transaction():
        session = db.session()
        if session["status"] == SessionStatus.COMPLETED:
            return
        db.update_session(
            {
                "status": SessionStatus.RUNNING,
                "stop_requested": 0,
                "started_at": session["started_at"] or _now_ms(),
                "completed_at": None,
                "stop_reason": None,
            }
        )


def request_stop(session_name: str) -> dict[str, Any]:
    _, path, _ = _session_context(session_name)
    with DB(path) as db, db.transaction():
        if db.session()["status"] == SessionStatus.RUNNING:
            db.update_session({"stop_requested": 1})
    return status(session_name)


def finish(session_name: str, status_value: str, reason: str) -> dict[str, Any]:
    if status_value not in {
        SessionStatus.STOPPED,
        SessionStatus.COMPLETED,
        SessionStatus.BLOCKED,
        SessionStatus.FAILED,
    }:
        raise SparError(f"invalid session status: {status_value}")
    _, path, _ = _session_context(session_name)
    with DB(path) as db, db.transaction():
        db.update_session(
            {
                "status": status_value,
                "stop_requested": 0,
                "completed_at": _now_ms(),
                "stop_reason": reason.strip(),
            }
        )
    return status(session_name)


def top(session_name: str, *, k: int = DEFAULT_TOP_LIMIT) -> dict[str, Any]:
    if k < 1:
        raise SparError("top --k must be a positive integer")
    _, path, config = _session_context(session_name)
    with DB(path) as db:
        candidates = mcts.top_candidates(
            db,
            k,
            exploration_constant=config["mcts"]["exploration_constant"],
        )
    return {"session_name": session_name, "k": k, "candidates": candidates}


def _session_context(session_name: str) -> tuple[Path, Path, dict[str, Any]]:
    repo = git.repo_root()
    path = require_session_dir(repo, session_name)
    config = load_config(path)
    with DB(path) as db:
        db.require_current_schema()
    return repo, path, config


def _require_clean_worktree(repo: Path) -> None:
    if git.status(repo, include_untracked=False).strip():
        raise SparError("tracked working tree must be clean before initializing a session")
    if git.status(repo, include_untracked=True).strip():
        raise SparError("working tree must have no untracked files before initializing a session")


def _session_status(db: DB) -> dict[str, Any]:
    session = db.session()
    session["stop_requested"] = bool(session["stop_requested"])
    started_at = session["started_at"]
    completed_at = session["completed_at"]
    session["elapsed_ms"] = completed_at - started_at if started_at is not None and completed_at is not None else None
    return session


def _status_snapshot(db: DB, max_candidates: int) -> dict[str, Any]:
    candidates = db.candidates()
    candidates_used = len(candidates)
    return {
        "max_candidates": max_candidates,
        "candidates_used": candidates_used,
        "counts": {
            status: sum(candidate["status"] == status for candidate in candidates)
            for status in sorted({candidate["status"] for candidate in candidates})
        },
        "candidates": [_candidate_summary(candidate) for candidate in candidates],
    }


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "id",
            "parent_id",
            "commit_sha",
            "hypothesis",
            "status",
            "eval_score",
            "learnings",
            "decision",
            "decision_reason",
            "error",
            "started_at",
            "completed_at",
        )
    }


def _now_ms() -> int:
    return time.time_ns() // 1_000_000
