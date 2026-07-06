from pathlib import Path
from typing import Any
import json
import time

from .config import DEFAULT_CONFIG, DEFAULT_OBJECTIVE, load_config
from .errors import SparError
from .process import run_command, run_git
from .repo import (
    artifact_manifest,
    candidate_artifact_dir,
    ensure_info_exclude,
    existing_session_dir,
    read_json_object,
    repo_root,
    session_dir,
    write_json,
)
from .state import (
    ROOT_CANDIDATE_ID,
    SessionState,
    STATUS_COMPLETED,
    STATUS_EVALUATING,
    STATUS_FINALIZING,
    STATUS_IMPLEMENTING,
    STATUS_REFLECTING,
    evaluation_score,
)


DEFAULT_PARENT_LIMIT = 3


class _OperationTimer:
    def __init__(self, path: Path, candidate_id: str | None, operation: str) -> None:
        self.path = path
        self.candidate_id = candidate_id
        self.operation = operation
        self.started_at = _now_ms()
        self.started_monotonic = time.monotonic_ns()
        self.success = True
        self.error: str | None = None
        self.record: dict[str, Any] | None = None

    def __enter__(self) -> _OperationTimer:
        return self

    def fail(self, error: str) -> None:
        self.success = False
        self.error = error

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            error = self.error
            if exc is not None:
                error = str(exc) or getattr(exc_type, "__name__", "operation interrupted")
            with SessionState(self.path) as state:
                self.record = state.record_operation(
                    candidate_id=self.candidate_id,
                    operation=self.operation,
                    started_at=self.started_at,
                    completed_at=_now_ms(),
                    elapsed_ms=(time.monotonic_ns() - self.started_monotonic) // 1_000_000,
                    success=exc_type is None and self.success,
                    error=error,
                )
        except Exception:
            self.record = None


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def init(session_name: str) -> dict[str, Any]:
    repo = repo_root()
    _require_clean_worktree(repo)
    path = session_dir(repo, session_name)
    ref = f"refs/spar/{session_name}/{ROOT_CANDIDATE_ID}"
    if run_git(repo, ["check-ref-format", ref], check=False).returncode:
        raise SparError(f"session name is not valid in a Git ref: {session_name}")
    if path.exists():
        raise SparError(f"session already exists: {path}")

    (path / "artifacts" / "candidates" / ROOT_CANDIDATE_ID).mkdir(parents=True)
    (path / "worktrees").mkdir()
    (path / "objective.md").write_text(DEFAULT_OBJECTIVE, encoding="utf-8")
    (path / "config.toml").write_text(DEFAULT_CONFIG, encoding="utf-8")
    ensure_info_exclude(repo)

    with SessionState(path) as state:
        state.initialize(repo_path=str(repo), root_commit=run_git(repo, ["rev-parse", "HEAD"]).stdout.strip())
        root = state.candidate(ROOT_CANDIDATE_ID)
    return {
        "session_name": session_name,
        "session_dir": str(path),
        "objective_path": str(path / "objective.md"),
        "config_path": str(path / "config.toml"),
        "root_candidate": root,
    }


def status(session_name: str) -> dict[str, Any]:
    repo, path, config = _session_context(session_name)
    with SessionState(path) as state:
        snapshot = state.status_snapshot(config["max_candidates"])
    return {
        "session_name": session_name,
        "session_dir": str(path),
        "repository": str(repo),
        "objective_path": str(path / "objective.md"),
        "config_path": str(path / "config.toml"),
        "max_parallel": config["max_parallel"],
        "evaluation": config["evaluation"],
        "profiling": config["profiling"],
        **snapshot,
    }


def parents(session_name: str, *, k: int = DEFAULT_PARENT_LIMIT) -> dict[str, Any]:
    if k < 1:
        raise SparError("parents --k must be a positive integer")
    _, path, _ = _session_context(session_name)
    with SessionState(path) as state:
        parents = state.parent_candidates(k)
    return {"session_name": session_name, "k": k, "parents": parents}


def candidate_inspect(session_name: str, candidate_id: str) -> dict[str, Any]:
    _, path, config = _session_context(session_name)
    with SessionState(path) as state:
        snapshot = state.inspect_candidate(candidate_id)
    artifact_dir = candidate_artifact_dir(path, candidate_id)
    candidate = snapshot["candidate"]
    parent_sha = snapshot["ancestors"][0]["commit_sha"] if snapshot["ancestors"] else None
    return {
        **snapshot,
        "artifacts": artifact_manifest(artifact_dir),
        "objective_path": str(path / "objective.md"),
        "evaluation": config["evaluation"],
        "profiling": config["profiling"],
        "result_paths": _result_paths(artifact_dir),
        "environment": _candidate_environment(path, candidate, parent_sha),
    }


def candidate_start(
    session_name: str,
    *,
    parent_id: str,
    hypothesis: str,
    instructions: str,
    rationale: str,
) -> dict[str, Any]:
    repo, path, config = _session_context(session_name)
    with SessionState(path) as state:
        candidate = state.start_candidate(
            parent_id=parent_id,
            hypothesis=hypothesis,
            instructions=instructions,
            rationale=rationale,
            max_candidates=config["max_candidates"],
            max_parallel=config["max_parallel"],
        )
        parent = state.candidate(parent_id)

    workspace = Path(candidate["workspace_path"])
    try:
        with _OperationTimer(path, candidate["id"], "worktree") as operation:
            run_git(repo, ["worktree", "add", "--detach", str(workspace), parent["commit_sha"]])
            if _workspace_commit(workspace) != parent["commit_sha"]:
                raise SparError("created candidate worktree does not match its parent commit")
    except Exception as exc:
        with SessionState(path) as state:
            state.fail_candidate(candidate["id"], f"could not create candidate worktree: {exc}", interrupted=True)
        raise

    artifact_dir = candidate_artifact_dir(path, candidate["id"])
    artifact_dir.mkdir(parents=True)
    return {
        "candidate": candidate,
        "parent_commit": parent["commit_sha"],
        "timing": operation.record,
        "objective_path": str(path / "objective.md"),
        "evaluation": config["evaluation"],
        "profiling": config["profiling"],
        "result_paths": _result_paths(artifact_dir),
        "environment": _candidate_environment(path, candidate, parent["commit_sha"]),
    }


def candidate_evaluate(session_name: str, candidate_id: str) -> dict[str, Any]:
    repo, path, config = _session_context(session_name)
    candidate, parent = _candidate_and_parent(path, candidate_id)
    if candidate["status"] not in {STATUS_IMPLEMENTING, STATUS_EVALUATING}:
        raise SparError(f"candidate is not ready for evaluation: {candidate_id}")

    workspace = Path(candidate["workspace_path"])
    commit_sha = _workspace_commit(workspace)
    if candidate["status"] == STATUS_EVALUATING and commit_sha != candidate["commit_sha"]:
        raise SparError("candidate worktree HEAD does not match its evaluation commit")
    _validate_candidate_commit(repo, candidate, parent, commit_sha, workspace)
    if candidate["status"] == STATUS_IMPLEMENTING:
        with SessionState(path) as state:
            state.begin_evaluation(candidate_id, commit_sha)
    artifact_dir = candidate_artifact_dir(path, candidate_id)
    with _OperationTimer(path, candidate_id, "evaluation") as operation:
        result = run_command(
            workspace,
            config["evaluation"]["command"],
            extra_env=_candidate_environment(path, candidate, parent["commit_sha"] if parent else None),
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

        write_json(artifact_dir / "evaluation-result.json", evaluation)
        with SessionState(path) as state:
            state.record_evaluation(candidate_id, evaluation)
    return {
        "candidate_id": candidate_id,
        "commit_sha": commit_sha,
        "evaluation": evaluation,
        "status": STATUS_REFLECTING,
        "timing": operation.record,
        "artifacts": artifact_manifest(artifact_dir),
    }


def candidate_profile(session_name: str, candidate_id: str) -> dict[str, Any]:
    repo, path, config = _session_context(session_name)
    candidate, parent = _candidate_and_parent(path, candidate_id)
    if candidate["status"] not in {STATUS_REFLECTING, STATUS_COMPLETED}:
        raise SparError(f"candidate cannot be profiled with status {candidate['status']}: {candidate_id}")

    command = config["profiling"]["command"]
    artifact_dir = candidate_artifact_dir(path, candidate_id)
    if command is None:
        return {
            "candidate_id": candidate_id,
            "profiling_status": "not_configured",
            "artifacts": artifact_manifest(artifact_dir),
        }

    if candidate["commit_sha"] is None:
        raise SparError("candidate must be evaluated before profiling")
    workspace = Path(candidate["workspace_path"])
    commit_sha = _workspace_commit(workspace)
    if commit_sha != candidate["commit_sha"]:
        raise SparError("candidate worktree HEAD does not match its recorded commit")
    _validate_candidate_commit(repo, candidate, parent, commit_sha, workspace)
    with _OperationTimer(path, candidate_id, "profiling") as operation:
        (artifact_dir / "profiling").mkdir(parents=True, exist_ok=True)
        result = run_command(
            workspace,
            command,
            extra_env=_candidate_environment(path, candidate, parent["commit_sha"] if parent else None),
        )
        (artifact_dir / "profiling.stdout").write_text(result.stdout, encoding="utf-8")
        (artifact_dir / "profiling.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            operation.fail(f"profiling command failed with exit code {result.returncode}")
    return {
        "candidate_id": candidate_id,
        "profiling_status": "completed" if result.returncode == 0 else "failed",
        "timing": operation.record,
        "artifacts": artifact_manifest(artifact_dir),
    }


def candidate_complete(
    session_name: str,
    candidate_id: str,
    *,
    summary: str,
    decision: str,
    decision_reason: str,
) -> dict[str, Any]:
    repo, path, _ = _session_context(session_name)
    candidate, parent = _candidate_and_parent(path, candidate_id)
    if candidate["status"] not in {STATUS_REFLECTING, STATUS_FINALIZING}:
        raise SparError(f"candidate is not ready for completion: {candidate_id}")
    workspace = Path(candidate["workspace_path"])
    commit_sha = candidate["commit_sha"]
    if commit_sha is None:
        raise SparError(f"candidate has not been evaluated: {candidate_id}")
    if _workspace_commit(workspace) != commit_sha:
        raise SparError("candidate worktree HEAD does not match its recorded commit")
    _validate_candidate_commit(repo, candidate, parent, commit_sha, workspace)

    artifact_dir = candidate_artifact_dir(path, candidate_id)
    evaluation = read_json_object(artifact_dir / "evaluation-result.json", "evaluation result")
    if candidate["eval_score"] != evaluation_score(evaluation):
        raise SparError("evaluation result does not match the recorded candidate score")
    with _OperationTimer(path, candidate_id, "completion") as operation:
        with SessionState(path) as state:
            finalizing = state.begin_finalization(
                candidate_id, summary=summary, decision=decision, decision_reason=decision_reason
            )

        run_git(repo, ["update-ref", f"refs/spar/{session_name}/{candidate_id}", commit_sha])
        write_json(
            artifact_dir / "reflection-result.json",
            {
                "summary": finalizing["eval_summary"],
                "decision": finalizing["decision"],
                "decision_reason": finalizing["decision_reason"],
            },
        )
        with SessionState(path) as state:
            completed = state.complete_candidate(candidate_id)
    return {"candidate": completed, "timing": operation.record, "artifacts": artifact_manifest(artifact_dir)}


def candidate_fail(session_name: str, candidate_id: str, error: str, *, interrupted: bool) -> dict[str, Any]:
    _, path, _ = _session_context(session_name)
    with _OperationTimer(path, candidate_id, "failure") as operation:
        with SessionState(path) as state:
            candidate = state.fail_candidate(candidate_id, error, interrupted=interrupted)
        artifact_dir = candidate_artifact_dir(path, candidate_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "error.txt").write_text(error.strip() + "\n", encoding="utf-8")
    return {"candidate": candidate, "timing": operation.record}


def _session_context(session_name: str) -> tuple[Path, Path, dict[str, Any]]:
    repo = repo_root()
    path = existing_session_dir(repo, session_name)
    config = load_config(path)
    with SessionState(path) as state:
        state.require_current_schema()
    return repo, path, config


def _candidate_and_parent(path: Path, candidate_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    with SessionState(path) as state:
        candidate = state.candidate(candidate_id)
        parent = None if candidate["parent_id"] is None else state.candidate(candidate["parent_id"])
    return candidate, parent


def _result_paths(artifact_dir: Path) -> dict[str, str]:
    return {
        "evaluation_result": str(artifact_dir / "evaluation-result.json"),
        "evaluation_stdout": str(artifact_dir / "evaluation.stdout"),
        "evaluation_stderr": str(artifact_dir / "evaluation.stderr"),
        "profiling_result": str(artifact_dir / "profiling-result.json"),
        "profiling_dir": str(artifact_dir / "profiling"),
        "reflection_result": str(artifact_dir / "reflection-result.json"),
    }


def _candidate_environment(path: Path, candidate: dict[str, Any], parent_sha: str | None) -> dict[str, str]:
    artifact_dir = candidate_artifact_dir(path, candidate["id"])
    return {
        "SPAR_CANDIDATE_ID": candidate["id"],
        "SPAR_SESSION_DIR": str(path),
        "SPAR_PARENT_SHA": parent_sha or "",
        "SPAR_PROFILING_DIR": str(artifact_dir / "profiling"),
        "SPAR_PROFILING_RESULT": str(artifact_dir / "profiling-result.json"),
    }


def _workspace_commit(workspace: Path) -> str:
    if not workspace.exists():
        raise SparError(f"candidate worktree does not exist: {workspace}")
    return run_git(workspace, ["rev-parse", "HEAD"]).stdout.strip()


def _validate_candidate_commit(
    repo: Path,
    candidate: dict[str, Any],
    parent: dict[str, Any] | None,
    commit_sha: str,
    workspace: Path,
) -> None:
    if run_git(repo, ["cat-file", "-e", f"{commit_sha}^{{commit}}"], check=False).returncode != 0:
        raise SparError(f"candidate commit does not exist: {commit_sha}")
    if parent is None:
        if commit_sha != candidate["commit_sha"]:
            raise SparError("root candidate commit must match the initialized baseline")
    elif run_git(repo, ["merge-base", "--is-ancestor", parent["commit_sha"], commit_sha], check=False).returncode != 0:
        raise SparError("candidate commit must descend from its parent commit")
    if workspace.resolve() != Path(candidate["workspace_path"]).resolve():
        raise SparError("candidate worktree does not match its recorded workspace")
    if workspace.exists() and run_git(workspace, ["status", "--short", "--untracked-files=no"]).stdout.strip():
        raise SparError("candidate worktree has uncommitted tracked changes")
    if run_git(workspace, ["status", "--short", "--untracked-files=all"]).stdout.strip():
        raise SparError("candidate worktree has untracked files")


def _require_clean_worktree(repo: Path) -> None:
    if run_git(repo, ["status", "--short", "--untracked-files=no"]).stdout.strip():
        raise SparError("tracked working tree must be clean before initializing a session")
    if run_git(repo, ["status", "--short", "--untracked-files=all"]).stdout.strip():
        raise SparError("working tree must have no untracked files before initializing a session")
