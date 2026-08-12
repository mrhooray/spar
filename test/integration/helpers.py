import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from spar.lifecycle import CandidateStatus, Decision
from spar.operation import candidate as candidate_ops
from spar.operation import session as session_ops
from spar.storage.db import DB, Span


class chdir:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.previous = Path.cwd()

    def __enter__(self) -> None:
        os.chdir(self.path)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        os.chdir(self.previous)


class temp_git_repo:
    def __enter__(self) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        run(["git", "init", "--initial-branch=master"], self.path)
        run(["git", "config", "user.name", "Test"], self.path)
        run(["git", "config", "user.email", "test@example.invalid"], self.path)
        return self.path

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.tmp.cleanup()


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)


def run_spar(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(Path(sys.executable).with_name("spar")), *command],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def run_spar_result(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(Path(sys.executable).with_name("spar")), *command],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def write_session_config(
    session: Path,
    *,
    max_candidates: int,
    max_parallel: int,
    evaluation_command: list[str],
    profiling_command: list[str] | None = None,
) -> None:
    profiling = ""
    if profiling_command is not None:
        profiling = textwrap.dedent(
            f"""

            [profiling]
            command = {json.dumps(profiling_command)}
            """
        )
    (session / "config.toml").write_text(
        textwrap.dedent(
            f"""
            max_candidates = {max_candidates}
            max_parallel = {max_parallel}

            [agent]
            cli = "codex"
            model = "test-model"
            effort = "high"

            [evaluation]
            command = {json.dumps(evaluation_command)}
            """
        ).lstrip()
        + profiling,
        encoding="utf-8",
    )


def initialize_session(repo: Path, *, max_candidates: int = 10, profiling: bool = False) -> Path:
    with chdir(repo):
        session_ops.init("demo")
    session = repo / ".spar" / "demo"
    write_session_config(
        session,
        max_candidates=max_candidates,
        max_parallel=2,
        evaluation_command=[str((repo / "eval.sh").resolve())],
        profiling_command=["./profile.sh"] if profiling else None,
    )
    return session


def complete_root(
    repo: Path,
    session: Path,
    *,
    score: float,
    profiling: dict[str, Any] | None = None,
) -> None:
    evaluation = {"score": score, "value": score}
    _record_evaluation(repo, session, "root", evaluation)
    if profiling is not None:
        with Span(session, candidate_id="root", kind="profiling") as span, span.complete() as transaction:
            assert transaction.update_candidate(
                "root",
                CandidateStatus.REFLECTING,
                {"profiling": profiling},
            )
    with chdir(repo):
        candidate_ops.complete(
            "demo",
            "root",
            learnings="baseline measured",
            decision=Decision.KEEP,
            decision_reason="valid baseline",
        )


def create_completed_child(
    repo: Path,
    session: Path,
    *,
    score: float,
    decision: str,
    suffix: str,
    parent_id: str = "root",
) -> str:
    with chdir(repo):
        started = candidate_ops.start(
            "demo",
            parent_id=parent_id,
            hypothesis=f"candidate {suffix}",
            instructions=f"write {suffix}",
            rationale="test MCTS",
        )
    candidate_id = started["candidate"]["id"]
    worktree = Path(started["candidate"]["worktree_path"])
    (worktree / "value.txt").write_text(f"{suffix}\n", encoding="utf-8")
    run(["git", "add", "value.txt"], worktree)
    run(["git", "commit", "-m", suffix], worktree)
    _record_evaluation(repo, session, candidate_id, {"score": score})
    with chdir(repo):
        candidate_ops.complete(
            "demo",
            candidate_id,
            learnings=f"{suffix} measured",
            decision=decision,
            decision_reason=f"{decision} by researcher",
        )
    return candidate_id


def start_same_candidate() -> dict[str, Any]:
    return candidate_ops.start(
        "demo",
        parent_id="root",
        hypothesis="same idea",
        instructions="try an implementation",
        rationale="compare alternate implementations",
    )


def _record_evaluation(repo: Path, session: Path, candidate_id: str, evaluation: dict[str, Any]) -> None:
    with DB(session) as db:
        candidate = db.candidate(candidate_id)
        commit_sha = run(["git", "rev-parse", "HEAD"], Path(candidate["worktree_path"])).stdout.strip()
        with db.transaction():
            assert db.update_candidate(
                candidate_id,
                CandidateStatus.IMPLEMENTING,
                {
                    "commit_sha": commit_sha,
                    "status": CandidateStatus.EVALUATING,
                },
            )
        candidate_ops.evaluation_score(evaluation)
    with Span(session, candidate_id=candidate_id, kind="evaluation") as span, span.complete() as transaction:
        assert transaction.update_candidate(
            candidate_id,
            CandidateStatus.EVALUATING,
            {
                "status": CandidateStatus.REFLECTING,
                "evaluation": evaluation,
            },
        )


def commit_file(repo: Path, name: str, text: str, message: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    run(["git", "add", name], repo)
    run(["git", "commit", "-m", message], repo)
