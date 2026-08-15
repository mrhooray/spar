import fcntl
import json
import signal
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .agent import invocation as agent_invocation
from .agent.adapter import AgentAdapter, create_agent
from .config import load_config
from .error import SparError
from .lifecycle import (
    NONTERMINAL_CANDIDATE_STATUSES,
    ROOT_CANDIDATE_ID,
    CandidateStatus,
    Decision,
    SessionStatus,
)
from .operation import candidate as candidate_ops
from .operation import session as session_ops
from .storage import git
from .storage.db import DB
from .storage.file import candidate_artifact_dir, require_session_dir

_MAX_CONSECUTIVE_PROPOSAL_FAILURES = 3
_PARENT_CANDIDATE_LIMIT = 1
_RECENT_INTERVENTION_LIMIT = 10


def format_progress(scope: str, identifier: str, phase: str, message: str = "") -> str:
    message = " ".join(message.splitlines())
    line = f"[{scope}:{identifier}] [{phase}]"
    return f"{line} {message}" if message else line


class ResearchLoop:
    def __init__(
        self,
        session_name: str,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.session_name = session_name
        self.repo = git.repo_root()
        self.session_dir = require_session_dir(self.repo, session_name)
        self.progress = progress

    def start(self) -> dict[str, Any]:
        with _session_lock(self.session_dir):
            with DB(self.session_dir) as db:
                db.require_current_schema()
                if db.session()["status"] == SessionStatus.COMPLETED:
                    return session_ops.status(self.session_name)
            self._initialize()
            session_ops.start(self.session_name)
            self._log("session", self.session_name, "started")
            with _graceful_stop(self.session_name):
                try:
                    status, reason = self._research()
                except Exception as exc:  # noqa: BLE001
                    status = SessionStatus.FAILED
                    reason = f"research loop failed: {_error_text(exc)}"
                result = session_ops.finish(self.session_name, status, reason)
                self._log("session", self.session_name, status, reason)
                return result

    def _initialize(self) -> None:
        config = load_config(self.session_dir)
        self.agent_config = config["agent"]
        with DB(self.session_dir) as db:
            self.agent = create_agent(self.agent_config, session_id=db.session()["agent_session_id"])
        self.max_candidates = config["max_candidates"]
        self.max_parallel = config["max_parallel"]
        self.profiler_configured = config["profiling"]["command"] is not None

        self.objective_path = self.session_dir / "objective.md"
        self.config_path = self.session_dir / "config.toml"
        self.objective_bytes = self.objective_path.read_bytes()
        self.config_bytes = self.config_path.read_bytes()
        self.objective = self.objective_bytes.decode("utf-8").strip()
        self.repo_head = git.head(self.repo)

        artifacts_dir = self.session_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "proposals").mkdir(exist_ok=True)

    def _research(self) -> tuple[SessionStatus, str]:
        self._require_inputs_unchanged()
        self._reconcile_unfinished_candidates()
        self._ensure_root_baseline()
        self._log("session", self.session_name, "baseline-ready")
        stop_reason: str | None = None
        proposal_failures = 0
        futures: set[Future[None]] = set()

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            while True:
                can_admit = (
                    stop_reason is None
                    and len(futures) < self.max_parallel
                    and self._candidates_used() < self.max_candidates
                )
                if not can_admit:
                    if not futures:
                        break
                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    futures.difference_update(completed)
                    for future in completed:
                        future.result()
                    continue

                if self._stop_requested():
                    stop_reason = "stop requested"
                    continue
                self._require_inputs_unchanged()

                top_candidates = session_ops.top(self.session_name, k=_PARENT_CANDIDATE_LIMIT)["candidates"]
                if not top_candidates:
                    stop_reason = "no completed kept parent is available"
                    continue
                parent_id = top_candidates[0]["candidate_id"]
                try:
                    proposal = self._propose(parent_id)
                except Exception as exc:  # noqa: BLE001
                    proposal_failures += 1
                    if proposal_failures >= _MAX_CONSECUTIVE_PROPOSAL_FAILURES:
                        stop_reason = f"proposal phase failed {proposal_failures} consecutive times: {_error_text(exc)}"
                    continue
                proposal_failures = 0

                started = candidate_ops.start(
                    self.session_name,
                    parent_id=parent_id,
                    hypothesis=proposal["hypothesis"],
                    instructions=proposal["instructions"],
                    rationale=proposal["rationale"],
                    profiling_question=proposal["profiling_question"],
                )
                candidate = started["candidate"]
                if candidate["status"] == CandidateStatus.FAILED:
                    self._log(
                        "candidate",
                        candidate["id"],
                        "failed",
                        candidate["error"] or "candidate could not be started",
                    )
                    continue

                candidate_id = candidate["id"]
                self._log(
                    "candidate",
                    candidate_id,
                    "started",
                    f"parent={candidate['parent_id']}",
                )
                try:
                    future = executor.submit(
                        self._process_candidate,
                        candidate,
                    )
                except Exception as exc:  # noqa: BLE001
                    candidate_ops.fail(
                        self.session_name,
                        candidate_id,
                        f"candidate launch failed: {_error_text(exc)}",
                        interrupted=True,
                    )
                    continue
                futures.add(future)

        if self._stop_requested():
            return SessionStatus.STOPPED, "stop requested"
        if stop_reason:
            return SessionStatus.BLOCKED, stop_reason
        return SessionStatus.COMPLETED, "maximum candidates reached"

    def _require_inputs_unchanged(self) -> None:
        if self.objective_path.read_bytes() != self.objective_bytes:
            raise SparError("session objective changed while research was active")
        if self.config_path.read_bytes() != self.config_bytes:
            raise SparError("session configuration changed while research was active")
        if git.head(self.repo) != self.repo_head:
            raise SparError("repository HEAD changed while research was active")
        if git.status(self.repo, include_untracked=True):
            raise SparError("repository working tree changed while research was active")

    def _reconcile_unfinished_candidates(self) -> None:
        status = session_ops.status(self.session_name)
        for candidate in status["candidates"]:
            if candidate["id"] == ROOT_CANDIDATE_ID or candidate["status"] not in NONTERMINAL_CANDIDATE_STATUSES:
                continue
            candidate_ops.fail(
                self.session_name,
                candidate["id"],
                "previous SPAR process ended before candidate completion",
                interrupted=True,
            )

    def _ensure_root_baseline(self) -> None:
        root = candidate_ops.inspect(self.session_name, ROOT_CANDIDATE_ID)["candidate"]
        if root["status"] in {CandidateStatus.IMPLEMENTING, CandidateStatus.EVALUATING}:
            candidate_ops.evaluate(self.session_name, ROOT_CANDIDATE_ID)
            root = candidate_ops.inspect(self.session_name, ROOT_CANDIDATE_ID)["candidate"]
        if root["status"] == CandidateStatus.REFLECTING:
            candidate_ops.complete(
                self.session_name,
                ROOT_CANDIDATE_ID,
                learnings="Canonical baseline measured.",
                decision=Decision.KEEP,
                decision_reason="The measured root is the baseline for candidate comparison.",
            )
            return
        if root["status"] != CandidateStatus.COMPLETED or root["decision"] == Decision.DISCARD:
            raise SparError(f"root candidate is not an expandable baseline: {root['status']}")

    def _propose(self, parent_id: str) -> dict[str, Any]:
        with DB(self.session_dir) as db:
            parent = db.candidate(parent_id)
            candidates = db.candidates()
            proposal_number = 1 + db.count_spans("proposal")
        proposal_id = f"{proposal_number:04d}"
        self._log("proposal", proposal_id, "started", f"parent={parent_id}")
        active_siblings = [
            candidate
            for candidate in candidates
            if candidate["parent_id"] == parent_id and candidate["status"] in NONTERMINAL_CANDIDATE_STATUSES
        ]
        active_ids = {candidate["id"] for candidate in active_siblings}
        excluded_ids = {ROOT_CANDIDATE_ID, parent_id} | active_ids
        recent_interventions = [
            candidate
            for candidate in candidates
            if candidate["id"] not in excluded_ids and candidate["status"] not in NONTERMINAL_CANDIDATE_STATUSES
        ][-_RECENT_INTERVENTION_LIMIT:]
        progress = {
            "selected_parent": _candidate_fields(
                parent,
                "id",
                "hypothesis",
                "evaluation_json",
                "profiling_json",
                "learnings",
            ),
            "active_sibling_interventions": [
                _candidate_fields(candidate, "id", "hypothesis", "instructions") for candidate in active_siblings
            ],
            "recent_interventions": [
                _candidate_fields(
                    candidate,
                    "id",
                    "hypothesis",
                    "instructions",
                    "eval_score",
                    "decision",
                    "decision_reason",
                    "learnings",
                    "error",
                    "profiling_json",
                )
                for candidate in recent_interventions
            ],
            "profiler_configured": self.profiler_configured,
        }
        event_path = self.session_dir / "artifacts" / "proposals" / f"{proposal_number:04d}.events.jsonl"
        try:
            previous_session_id = self.agent.session_id
            proposal = agent_invocation.propose(
                self.agent,
                objective=self.objective,
                progress=progress,
                session_name=self.session_name,
                cwd=Path(parent["worktree_path"]),
                event_path=event_path,
                state_path=self.session_dir,
                parent_id=parent_id,
            )
            if self.agent.session_id != previous_session_id:
                self._save_agent_session()
        except Exception as exc:
            self._log("proposal", proposal_id, "failed", _error_text(exc))
            raise
        self._log("proposal", proposal_id, "completed")
        return proposal

    def _process_candidate(self, candidate: dict[str, Any]) -> None:
        candidate_id = candidate["id"]
        worker = create_agent(self.agent_config)
        try:
            parent = candidate_ops.inspect(self.session_name, candidate["parent_id"])["candidate"]
            parent_commit = parent["commit_sha"]
            implementation_report = self._implement_candidate(candidate, parent_commit, worker)
            self._require_inputs_unchanged()
            evaluation = candidate_ops.evaluate(self.session_name, candidate_id)
            profiles = {
                "candidate": None,
                "parent": (json.loads(parent["profiling_json"]) if parent["profiling_json"] is not None else None),
            }
            if candidate["profiling_question"] is not None:
                profiling_result = candidate_ops.profile(self.session_name, candidate_id)
                profiles["candidate"] = profiling_result["profile"]
                if profiles["candidate"] is None:
                    raise SparError("requested profiling produced no result")
            candidate = candidate_ops.inspect(self.session_name, candidate_id)["candidate"]
            intervention = _candidate_fields(
                candidate,
                "id",
                "hypothesis",
                "instructions",
                "rationale",
                "profiling_question",
            )
            reflection = agent_invocation.reflect(
                worker,
                objective=self.objective,
                intervention=intervention,
                implementation_report=implementation_report,
                evaluation=evaluation["evaluation"],
                profiles=profiles,
                parent=compact_candidate(parent),
                session_name=self.session_name,
                cwd=Path(candidate["worktree_path"]),
                event_path=(candidate_artifact_dir(self.session_dir, candidate_id) / "reflection.events.jsonl"),
                state_path=self.session_dir,
                candidate_id=candidate_id,
            )
            completed = candidate_ops.complete(
                self.session_name,
                candidate_id,
                learnings=reflection["learnings"],
                decision=reflection["decision"],
                decision_reason=reflection["decision_reason"],
            )
            completed_candidate = completed["candidate"]
            self._log(
                "candidate",
                candidate_id,
                "completed",
                f"decision={completed_candidate['decision']} score={completed_candidate['eval_score']}",
            )
        except Exception as exc:  # noqa: BLE001
            error = _error_text(exc)
            candidate_ops.fail(
                self.session_name,
                candidate_id,
                error,
                interrupted=False,
            )
            self._log("candidate", candidate_id, "failed", error)

    def _implement_candidate(
        self,
        candidate: dict[str, Any],
        parent_commit: str,
        worker: AgentAdapter,
    ) -> dict[str, Any]:
        workspace = Path(candidate["worktree_path"])
        artifact_dir = candidate_artifact_dir(self.session_dir, candidate["id"])
        implementation = agent_invocation.implement(
            worker,
            objective=self.objective,
            candidate=candidate,
            cwd=workspace,
            event_path=artifact_dir / "implementation.events.jsonl",
            state_path=self.session_dir,
            candidate_id=candidate["id"],
        )
        git.commit_candidate_workspace(workspace, candidate["id"], parent_commit)
        return implementation

    def _save_agent_session(self) -> None:
        session_id = self.agent.session_id
        if session_id is None:
            return
        with DB(self.session_dir) as db, db.transaction():
            db.update_session({"agent_session_id": session_id})

    def _candidates_used(self) -> int:
        with DB(self.session_dir) as db:
            return db.candidates_used()

    def _stop_requested(self) -> bool:
        with DB(self.session_dir) as db:
            return bool(db.session()["stop_requested"])

    def _log(self, scope: str, identifier: str, phase: str, message: str = "") -> None:
        if self.progress is not None:
            self.progress(format_progress(scope, identifier, phase, message))


def compact_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    fields = (
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
    )
    return {field: candidate.get(field) for field in fields}


def _candidate_fields(candidate: dict[str, Any], *fields: str) -> dict[str, Any]:
    return {field: candidate[field] for field in fields}


def _error_text(error: Exception) -> str:
    return str(error).strip() or type(error).__name__


@contextmanager
def _session_lock(session: Path) -> Iterator[None]:
    with (session / "session.lock").open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SparError(f"session is already active: {session.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def _graceful_stop(session_name: str) -> Iterator[None]:
    previous = signal.getsignal(signal.SIGINT)

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        try:
            session_ops.request_stop(session_name)
        except SparError:
            pass

    try:
        signal.signal(signal.SIGINT, request_stop)
    except ValueError:
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
