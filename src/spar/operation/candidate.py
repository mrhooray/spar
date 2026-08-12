import json
import math
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from .. import mcts
from ..error import SparError
from ..lifecycle import (
    NONTERMINAL_CANDIDATE_STATUSES,
    ROOT_CANDIDATE_ID,
    CandidateStatus,
    Decision,
)
from ..storage import git
from ..storage.db import DB, Span
from ..storage.file import (
    artifact_manifest,
    candidate_artifact_dir,
)
from .session import _session_context


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def start(
    session_name: str,
    *,
    parent_id: str,
    hypothesis: str,
    instructions: str,
    rationale: str,
    profiling_question: str | None = None,
) -> dict[str, Any]:
    repo, path, config = _session_context(session_name)
    for name, value in (
        ("hypothesis", hypothesis),
        ("instructions", instructions),
        ("rationale", rationale),
    ):
        if not value.strip():
            raise SparError(f"candidate {name} must not be empty")
    if profiling_question is not None and not profiling_question.strip():
        raise SparError("candidate profiling_question must not be empty")
    candidate_id = uuid.uuid4().hex[:12]
    timestamp = _now_ms()
    with DB(path) as db:
        with db.transaction():
            parent = db.candidate(parent_id)
            if not mcts.is_expandable(parent):
                raise SparError(f"parent candidate is not expandable: {parent_id}")
            if not db.insert_candidate(
                {
                    "id": candidate_id,
                    "parent_id": parent_id,
                    "worktree_path": str(path / "worktrees" / candidate_id),
                    "hypothesis": hypothesis.strip(),
                    "instructions": instructions.strip(),
                    "rationale": rationale.strip(),
                    "profiling_question": (profiling_question.strip() if profiling_question is not None else None),
                    "status": CandidateStatus.IMPLEMENTING,
                    "started_at": timestamp,
                },
                maximum=config["max_candidates"],
            ):
                raise SparError(f"maximum candidates reached: {config['max_candidates']}")
        candidate = db.candidate(candidate_id)

    workspace = Path(candidate["worktree_path"])
    workspace.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Span(path, candidate_id=candidate["id"], kind="worktree") as span:
            git.add_worktree(repo, workspace, parent["commit_sha"])
            if _workspace_commit(workspace) != parent["commit_sha"]:
                raise SparError("created candidate worktree does not match its parent commit")
    except Exception as exc:  # noqa: BLE001
        with DB(path) as db:
            candidate = _fail_candidate(
                db,
                candidate["id"],
                f"could not create candidate worktree: {exc}",
                interrupted=False,
            )
        return {
            "candidate": candidate,
            "parent_commit": parent["commit_sha"],
            "span": span.record,
        }

    artifact_dir = candidate_artifact_dir(path, candidate["id"])
    artifact_dir.mkdir(parents=True)
    return {
        "candidate": candidate,
        "parent_commit": parent["commit_sha"],
        "span": span.record,
    }


def evaluate(session_name: str, candidate_id: str) -> dict[str, Any]:
    repo, path, config = _session_context(session_name)
    candidate, parent = _candidate_and_parent(path, candidate_id)
    if candidate["status"] not in {
        CandidateStatus.IMPLEMENTING,
        CandidateStatus.EVALUATING,
    }:
        raise SparError(f"candidate is not ready for evaluation: {candidate_id}")

    workspace = Path(candidate["worktree_path"])
    commit_sha = _workspace_commit(workspace)
    if candidate["status"] == CandidateStatus.EVALUATING and commit_sha != candidate["commit_sha"]:
        raise SparError("candidate worktree HEAD does not match its evaluation commit")
    _validate_candidate_commit(repo, candidate, parent, commit_sha, workspace)
    if candidate["status"] == CandidateStatus.IMPLEMENTING:
        with DB(path) as db, db.transaction():
            if not db.update_candidate(
                candidate_id,
                CandidateStatus.IMPLEMENTING,
                {
                    "commit_sha": commit_sha,
                    "status": CandidateStatus.EVALUATING,
                },
            ):
                raise SparError(f"candidate is not implementing: {candidate_id}")
    artifact_dir = candidate_artifact_dir(path, candidate_id)
    with Span(path, candidate_id=candidate_id, kind="evaluation") as span:
        result = _execute_command(
            workspace,
            config["evaluation"]["command"],
            extra_env=_candidate_environment(path, candidate, parent["commit_sha"] if parent else None),
            timeout=config["evaluation"]["timeout_seconds"],
        )
        (artifact_dir / "evaluation.stdout").write_text(result.stdout, encoding="utf-8")
        (artifact_dir / "evaluation.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise SparError(f"evaluation command failed with exit code {result.returncode}")
        try:
            evaluation = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise SparError("evaluation command stdout must be a JSON object") from exc
        if not isinstance(evaluation, dict):
            raise SparError("evaluation command stdout must be a JSON object")
        evaluation_score(evaluation)
        with span.complete() as db:
            if not db.update_candidate(
                candidate_id,
                CandidateStatus.EVALUATING,
                {
                    "status": CandidateStatus.REFLECTING,
                    "evaluation": evaluation,
                },
            ):
                raise SparError(f"candidate is not evaluating: {candidate_id}")
    return {
        "candidate_id": candidate_id,
        "commit_sha": commit_sha,
        "evaluation": evaluation,
        "span": span.record,
        "artifacts": artifact_manifest(artifact_dir),
    }


def profile(session_name: str, candidate_id: str) -> dict[str, Any]:
    repo, path, config = _session_context(session_name)
    candidate, parent = _candidate_and_parent(path, candidate_id)
    if candidate["status"] not in {
        CandidateStatus.REFLECTING,
        CandidateStatus.COMPLETED,
    }:
        raise SparError(f"candidate cannot be profiled with status {candidate['status']}: {candidate_id}")

    command = config["profiling"]["command"]
    artifact_dir = candidate_artifact_dir(path, candidate_id)
    if command is None:
        return {
            "candidate_id": candidate_id,
            "commit_sha": candidate["commit_sha"],
            "profile": None,
            "artifacts": artifact_manifest(artifact_dir),
        }

    if candidate["commit_sha"] is None:
        raise SparError("candidate must be evaluated before profiling")
    workspace = Path(candidate["worktree_path"])
    commit_sha = _workspace_commit(workspace)
    if commit_sha != candidate["commit_sha"]:
        raise SparError("candidate worktree HEAD does not match its recorded commit")
    _validate_candidate_commit(repo, candidate, parent, commit_sha, workspace)
    with Span(path, candidate_id=candidate_id, kind="profiling") as span:
        (artifact_dir / "profiling").mkdir(parents=True, exist_ok=True)
        result = _execute_command(
            workspace,
            command,
            extra_env=_candidate_environment(path, candidate, parent["commit_sha"] if parent else None),
            timeout=config["profiling"]["timeout_seconds"],
        )
        (artifact_dir / "profiling.stdout").write_text(result.stdout, encoding="utf-8")
        (artifact_dir / "profiling.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise SparError(f"profiling command failed with exit code {result.returncode}")
        try:
            profiling = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise SparError("profiling command stdout must be a JSON object") from exc
        if not isinstance(profiling, dict):
            raise SparError("profiling command stdout must be a JSON object")
        with span.complete() as db:
            if not db.update_candidate(
                candidate_id,
                candidate["status"],
                {"profiling": profiling},
            ):
                raise SparError(f"candidate status changed while profiling: {candidate_id}")
    return {
        "candidate_id": candidate_id,
        "commit_sha": commit_sha,
        "profile": profiling,
        "span": span.record,
        "artifacts": artifact_manifest(artifact_dir),
    }


def complete(
    session_name: str,
    candidate_id: str,
    *,
    learnings: str,
    decision: str,
    decision_reason: str,
) -> dict[str, Any]:
    repo, path, _ = _session_context(session_name)
    candidate, parent = _candidate_and_parent(path, candidate_id)
    if candidate["status"] != CandidateStatus.REFLECTING:
        raise SparError(f"candidate is not ready for completion: {candidate_id}")
    workspace = Path(candidate["worktree_path"])
    commit_sha = candidate["commit_sha"]
    if commit_sha is None:
        raise SparError(f"candidate has not been evaluated: {candidate_id}")
    if _workspace_commit(workspace) != commit_sha:
        raise SparError("candidate worktree HEAD does not match its recorded commit")
    _validate_candidate_commit(repo, candidate, parent, commit_sha, workspace)

    artifact_dir = candidate_artifact_dir(path, candidate_id)
    if candidate["eval_score"] is None:
        raise SparError(f"candidate has no evaluation score: {candidate_id}")
    if decision not in (Decision.KEEP, Decision.DISCARD):
        raise SparError("candidate decision must be keep or discard")
    if not learnings.strip() or not decision_reason.strip():
        raise SparError("candidate learnings and decision reason must not be empty")
    with Span(path, candidate_id=candidate_id, kind="finalization") as span:
        git.update_ref(repo, f"refs/spar/{session_name}/{candidate_id}", commit_sha)
        with span.complete() as db:
            candidate = db.candidate(candidate_id)
            timestamp = _now_ms()
            if not db.update_candidate(
                candidate_id,
                CandidateStatus.REFLECTING,
                {
                    "status": CandidateStatus.COMPLETED,
                    "learnings": learnings.strip(),
                    "decision": decision,
                    "decision_reason": decision_reason.strip(),
                    "completed_at": timestamp,
                },
            ):
                raise SparError(f"candidate is not reflecting: {candidate_id}")
            mcts.backpropagate(db, candidate_id, float(candidate["eval_score"]))
            completed = db.candidate(candidate_id)
    return {
        "candidate": completed,
        "span": span.record,
        "artifacts": artifact_manifest(artifact_dir),
    }


def fail(session_name: str, candidate_id: str, error: str, *, interrupted: bool) -> dict[str, Any]:
    _, path, _ = _session_context(session_name)
    with DB(path) as db:
        candidate = _fail_candidate(db, candidate_id, error, interrupted=interrupted)
    return {"candidate": candidate}


def inspect(session_name: str, candidate_id: str) -> dict[str, Any]:
    _, path, config = _session_context(session_name)
    with DB(path) as db:
        candidate = db.candidate(candidate_id)
        candidates = db.candidates()
        by_id = {row["id"]: row for row in candidates}
        ancestors: list[dict[str, Any]] = []
        parent_id = candidate["parent_id"]
        while parent_id is not None:
            parent = by_id.get(parent_id)
            if parent is None:
                break
            ancestors.append(parent)
            parent_id = parent["parent_id"]
        suggestions = {
            item["candidate_id"]: item
            for item in mcts.top_candidates(
                db,
                exploration_constant=config["mcts"]["exploration_constant"],
            )
        }
        snapshot = {
            "candidate": candidate,
            "ancestors": ancestors,
            "children": [row for row in candidates if row["parent_id"] == candidate_id],
            "mcts": suggestions.get(candidate_id),
            "spans": db.spans(candidate_id),
        }
    artifact_dir = candidate_artifact_dir(path, candidate_id)
    candidate = snapshot["candidate"]
    parent_sha = snapshot["ancestors"][0]["commit_sha"] if snapshot["ancestors"] else None
    return {
        **snapshot,
        "artifact_dir": str(artifact_dir),
        "artifacts": artifact_manifest(artifact_dir),
        "objective_path": str(path / "objective.md"),
        "evaluation": config["evaluation"],
        "profiling": config["profiling"],
        "environment": _candidate_environment(path, candidate, parent_sha),
    }


def _candidate_and_parent(path: Path, candidate_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    with DB(path) as db:
        candidate = db.candidate(candidate_id)
        parent = None if candidate["parent_id"] is None else db.candidate(candidate["parent_id"])
    return candidate, parent


def _fail_candidate(db: DB, candidate_id: str, error: str, *, interrupted: bool) -> dict[str, Any]:
    if not error.strip():
        raise SparError("candidate error must not be empty")
    if candidate_id == ROOT_CANDIDATE_ID:
        raise SparError("root candidate cannot be failed; retry baseline evaluation")
    candidate = db.candidate(candidate_id)
    if candidate["status"] not in NONTERMINAL_CANDIDATE_STATUSES:
        raise SparError(f"candidate is not unfinished: {candidate_id}")
    status = CandidateStatus.INTERRUPTED if interrupted else CandidateStatus.FAILED
    timestamp = _now_ms()
    with db.transaction():
        if not db.update_candidate(
            candidate_id,
            candidate["status"],
            {
                "status": status,
                "error": error.strip(),
                "completed_at": timestamp,
            },
        ):
            raise SparError(f"candidate is not unfinished: {candidate_id}")
        return db.candidate(candidate_id)


def _candidate_environment(path: Path, candidate: dict[str, Any], parent_sha: str | None) -> dict[str, str]:
    artifact_dir = candidate_artifact_dir(path, candidate["id"])
    return {
        "SPAR_CANDIDATE_ID": candidate["id"],
        "SPAR_PARENT_SHA": parent_sha or "",
        "SPAR_PROFILING_DIR": str(artifact_dir / "profiling"),
    }


def _workspace_commit(workspace: Path) -> str:
    if not workspace.exists():
        raise SparError(f"candidate worktree does not exist: {workspace}")
    return git.head(workspace)


def _validate_candidate_commit(
    repo: Path,
    candidate: dict[str, Any],
    parent: dict[str, Any] | None,
    commit_sha: str,
    workspace: Path,
) -> None:
    if not git.commit_exists(repo, commit_sha):
        raise SparError(f"candidate commit does not exist: {commit_sha}")
    if parent is None:
        if commit_sha != candidate["commit_sha"]:
            raise SparError("root candidate commit must match the initialized baseline")
    elif not git.is_ancestor(repo, parent["commit_sha"], commit_sha):
        raise SparError("candidate commit must descend from its parent commit")
    if workspace.resolve() != Path(candidate["worktree_path"]).resolve():
        raise SparError("candidate worktree does not match its recorded workspace")
    if workspace.exists() and git.status(workspace, include_untracked=False).strip():
        raise SparError("candidate worktree has uncommitted tracked changes")
    if git.status(workspace, include_untracked=True).strip():
        raise SparError("candidate worktree has untracked files")


def _execute_command(
    cwd: Path,
    command: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SparError(f"command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SparError(f"command timed out after {timeout}s: {command}") from exc


def evaluation_score(payload: dict[str, Any]) -> float:
    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(score):
        raise SparError("evaluation result must include a finite numeric score")
    return float(score)
