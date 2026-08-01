import fcntl
import hashlib
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import commands
from .agent import (
    IMPLEMENTATION_SCHEMA,
    PROPOSAL_SCHEMA,
    REFLECTION_SCHEMA,
    PhaseAgent,
    compact_candidate,
    implementation_prompt,
    proposal_prompt,
    reflection_prompt,
    run_phase,
    truncate_diff,
)
from .errors import SparError
from .process import run_git
from .repo import (
    candidate_artifact_dir,
    commit_candidate_workspace,
    read_json_object,
    write_json,
)
from .state import (
    DECISION_DISCARD,
    DECISION_KEEP,
    NONTERMINAL_STATUSES,
    ROOT_CANDIDATE_ID,
    STATUS_COMPLETED,
    STATUS_EVALUATING,
    STATUS_FINALIZING,
    STATUS_IMPLEMENTING,
    STATUS_REFLECTING,
    STATUS_REJECTED,
)

RUN_SCHEMA_VERSION = 1
DEFAULT_AGENT_TIMEOUT_SECONDS = 3600
_MAX_PROPOSAL_FAILURES = 3


def run_session(
    session_name: str,
    agent: PhaseAgent,
    *,
    proposal_agent: PhaseAgent | None = None,
    spine_agent: PhaseAgent | None = None,
    agent_metadata: dict[str, Any] | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    session = Path(commands.status(session_name)["session_dir"])
    with _session_lock(session):
        return ResearchRunner(
            session_name,
            agent,
            proposal_agent=proposal_agent,
            spine_agent=spine_agent,
            agent_metadata=agent_metadata or {"adapter": type(agent).__name__},
            workspace_root=workspace_root,
        ).run()


class ResearchRunner:
    def __init__(
        self,
        session_name: str,
        agent: PhaseAgent,
        *,
        proposal_agent: PhaseAgent | None,
        spine_agent: PhaseAgent | None,
        agent_metadata: dict[str, Any],
        workspace_root: Path | None,
    ) -> None:
        self.session_name = session_name
        self.agent = agent
        self.proposal_agent = proposal_agent or agent
        self.spine_agent = spine_agent
        self.started_at = _now_ms()
        self.started_monotonic = time.monotonic_ns()
        initial = commands.status(session_name)
        self.repo = Path(initial["repository"])
        self.session = Path(initial["session_dir"])
        self.objective_path = Path(initial["objective_path"])
        self.config_path = Path(initial["config_path"])
        self.objective_bytes = self.objective_path.read_bytes()
        self.config_bytes = self.config_path.read_bytes()
        self.objective = self.objective_bytes.decode("utf-8").strip()
        self.repo_head = run_git(self.repo, ["rev-parse", "HEAD"]).stdout.strip()
        self.session_started_at = commands.candidate_inspect(
            session_name, ROOT_CANDIDATE_ID
        )["candidate"]["created_at"]
        self.max_parallel = initial["max_parallel"]
        if self.spine_agent is not None and self.max_parallel != 3:
            raise SparError("hybrid-spine workflow requires max_parallel = 3")
        self.run_id = f"{self.started_at}-{uuid.uuid4().hex[:8]}"
        self.run_dir = self.session / "artifacts" / "runs" / self.run_id
        self.workspace_root = workspace_root or (
            self.repo.parent
            / f".{self.repo.name}-spar-worktrees"
            / self.session_name
            / self.run_id
        )
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "objective.md").write_bytes(self.objective_bytes)
        (self.run_dir / "config.toml").write_bytes(self.config_bytes)
        self.recorded_candidates: set[str] = set()
        self.candidate_roles: dict[str, str] = {}
        self.best_candidate_id: str | None = None
        self.best_score: float | None = None
        self.result: dict[str, Any] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "session_name": session_name,
            "session_dir": str(self.session),
            "workspace_root": str(self.workspace_root),
            "status": "running",
            "stop_reason": None,
            "started_at": self.started_at,
            "session_started_at": self.session_started_at,
            "target_starting_head": self.repo_head,
            "completed_at": None,
            "elapsed_ms": None,
            "max_parallel": self.max_parallel,
            "candidate_lanes": (
                {"spine": 1, "explorer": 2}
                if self.spine_agent is not None
                else {"standard": self.max_parallel}
            ),
            "agent": agent_metadata,
            "objective_sha256": hashlib.sha256(self.objective_bytes).hexdigest(),
            "config_sha256": hashlib.sha256(self.config_bytes).hexdigest(),
            "objective_snapshot": str(self.run_dir / "objective.md"),
            "config_snapshot": str(self.run_dir / "config.toml"),
            "proposal_attempts": [],
            "admission_failures": [],
            "rejected_attempts": [],
            "blocked_candidates": [],
            "timeline": [],
        }
        self._sync_terminal_candidates(initial)
        self._persist()

    def run(self) -> dict[str, Any]:
        try:
            return self._run()
        except Exception as exc:  # noqa: BLE001
            self._sync_terminal_candidates()
            return self._finish("failed", f"runner failed: {_error_text(exc)}")

    def _run(self) -> dict[str, Any]:
        self._require_inputs_unchanged()
        self._reconcile_unfinished_candidates()
        self._ensure_root_baseline()
        admissions_closed = False
        proposal_failures = 0
        futures: dict[Future[dict[str, Any]], str] = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            while futures or self._remaining_budget() > 0:
                while (
                    not admissions_closed
                    and len(futures) < self.max_parallel
                    and self._remaining_budget() > 0
                ):
                    self._require_inputs_unchanged()
                    status = commands.status(self.session_name)
                    parents = commands.parents(self.session_name, k=3)["parents"]
                    if not parents:
                        admissions_closed = True
                        self.result["stop_reason"] = "no completed kept parent is available"
                        break
                    role = "standard"
                    proposal_agent = self.proposal_agent
                    candidate_agent = self.agent
                    parent_id = parents[0]["candidate_id"]
                    if self.spine_agent is not None:
                        role = "explorer"
                        if "spine" not in futures.values():
                            role = "spine"
                            proposal_agent = self.spine_agent
                            candidate_agent = self.spine_agent
                            parent_id = status["best_candidate"]["id"]
                    try:
                        proposal = self._propose(
                            parents,
                            parent_id=parent_id,
                            role=role,
                            agent=proposal_agent,
                        )
                    except Exception as exc:  # noqa: BLE001
                        proposal_failures += 1
                        if proposal_failures >= _MAX_PROPOSAL_FAILURES:
                            admissions_closed = True
                            self.result["stop_reason"] = (
                                "proposal phase failed "
                                f"{proposal_failures} consecutive times: {_error_text(exc)}"
                            )
                        continue
                    proposal_failures = 0
                    budget_used = status["candidate_budget"]["used"]
                    try:
                        started = commands.candidate_start(
                            self.session_name,
                            parent_id=parent_id,
                            hypothesis=proposal["hypothesis"],
                            instructions=proposal["instructions"],
                            rationale=proposal["rationale"],
                            workspace_root=self.workspace_root,
                        )
                    except Exception as exc:
                        self._sync_terminal_candidates()
                        if commands.status(self.session_name)["candidate_budget"]["used"] == budget_used:
                            raise
                        self.result["admission_failures"].append(
                            {
                                "parent_id": parent_id,
                                "role": role,
                                "error": _error_text(exc),
                            }
                        )
                        self._persist()
                        continue
                    candidate = started["candidate"]
                    candidate_id = candidate["id"]
                    self.candidate_roles[candidate_id] = role
                    try:
                        future = executor.submit(
                            self._run_candidate,
                            candidate,
                            started["parent_commit"],
                            proposal,
                            candidate_agent,
                        )
                    except Exception as exc:  # noqa: BLE001
                        failed = commands.candidate_fail(
                            self.session_name,
                            candidate_id,
                            f"candidate launch failed: {_error_text(exc)}",
                            interrupted=True,
                        )["candidate"]
                        self._record_candidate(failed, source="runner")
                        self._persist()
                        continue
                    futures[future] = role
                    self._persist()

                if not futures:
                    break
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    del futures[future]
                outcomes = [future.result() for future in completed]
                for outcome in sorted(
                    outcomes,
                    key=lambda item: (
                        item["candidate"].get("completed_at")
                        or item["candidate"]["updated_at"],
                        item["candidate"]["id"],
                    ),
                ):
                    if blocked_reason := outcome.get("blocked_reason"):
                        admissions_closed = True
                        self.result["stop_reason"] = (
                            self.result["stop_reason"] or blocked_reason
                        )
                        self.result["blocked_candidates"].append(
                            {
                                "candidate_id": outcome["candidate"]["id"],
                                "role": self.candidate_roles.get(
                                    outcome["candidate"]["id"]
                                ),
                                "status": outcome["candidate"]["status"],
                                "reason": blocked_reason,
                            }
                        )
                    else:
                        self._record_candidate(outcome["candidate"], source="runner")
                    self._persist()

        if self.result["blocked_candidates"]:
            return self._finish(
                "blocked",
                self.result["stop_reason"] or "candidate implementation blocked",
            )
        if self._remaining_budget() == 0:
            return self._finish("completed", "candidate budget exhausted")
        return self._finish("blocked", self.result["stop_reason"] or "runner stopped before exhausting budget")

    def _ensure_root_baseline(self) -> None:
        root = commands.candidate_inspect(self.session_name, ROOT_CANDIDATE_ID)["candidate"]
        if root["status"] in {STATUS_IMPLEMENTING, STATUS_EVALUATING}:
            commands.candidate_evaluate(self.session_name, ROOT_CANDIDATE_ID)
            root = commands.candidate_inspect(self.session_name, ROOT_CANDIDATE_ID)["candidate"]
        if root["status"] == STATUS_REFLECTING:
            completed = commands.candidate_complete(
                self.session_name,
                ROOT_CANDIDATE_ID,
                summary="Canonical baseline measured.",
                decision=DECISION_KEEP,
                decision_reason="The measured root is the baseline for candidate comparison.",
            )["candidate"]
            self._record_candidate(completed, source="runner")
            self._persist()
            return
        if root["status"] == STATUS_FINALIZING:
            if root["decision"] != DECISION_KEEP:
                raise SparError("root baseline finalization must keep the root candidate")
            completed = commands.candidate_complete(
                self.session_name,
                ROOT_CANDIDATE_ID,
                summary=root["eval_summary"],
                decision=root["decision"],
                decision_reason=root["decision_reason"],
            )["candidate"]
            self._record_candidate(completed, source="runner")
            self._persist()
            return
        if root["status"] != STATUS_COMPLETED or root["decision"] == DECISION_DISCARD:
            raise SparError(f"root candidate is not an expandable baseline: {root['status']}")

    def _reconcile_unfinished_candidates(self) -> None:
        status = commands.status(self.session_name)
        for candidate in status["candidates"]:
            if candidate["id"] == ROOT_CANDIDATE_ID or candidate["status"] not in NONTERMINAL_STATUSES:
                continue
            failed = commands.candidate_fail(
                self.session_name,
                candidate["id"],
                "runner resumed without a live external-agent phase",
                interrupted=True,
            )["candidate"]
            self._record_candidate(failed, source="reconciliation")
        self._persist()

    def _propose(
        self,
        parents: list[dict[str, Any]],
        *,
        parent_id: str,
        role: str,
        agent: PhaseAgent,
    ) -> dict[str, Any]:
        status = commands.status(self.session_name)
        parent = commands.candidate_inspect(self.session_name, parent_id)["candidate"]
        active_siblings = [
            commands.candidate_inspect(self.session_name, candidate["id"])["candidate"]
            for candidate in status["candidates"]
            if candidate["parent_id"] == parent_id
            and candidate["status"] in NONTERMINAL_STATUSES
        ]
        evidence = {
            "budget": status["candidate_budget"],
            "best_candidate": compact_candidate(status["best_candidate"]),
            "search_role": role,
            "selected_parent": compact_candidate(parent),
            "mcts_parents": parents,
            "active_sibling_interventions": [
                {
                    "candidate_id": candidate["id"],
                    "hypothesis": candidate["hypothesis"],
                    "instructions": candidate["instructions"],
                }
                for candidate in active_siblings
            ],
            "recent_candidates": [
                compact_candidate(candidate) for candidate in status["candidates"][-10:]
            ],
            "profiling_available": status["profiling"]["command"] is not None,
        }
        prompt = proposal_prompt(self.objective, evidence)
        path = self.run_dir / "proposals" / f"{len(self.result['proposal_attempts']) + 1:04d}.json"
        attempt = {
            "path": str(path.relative_to(self.run_dir)),
            "parent_id": parent_id,
            "role": role,
        }
        try:
            proposal, _ = run_phase(
                agent,
                phase="propose",
                prompt=prompt,
                cwd=Path(parent["workspace_path"]),
                schema=PROPOSAL_SCHEMA,
                artifact_path=path,
            )
        except Exception:
            self.result["proposal_attempts"].append({**attempt, "success": False})
            self._persist()
            raise
        self.result["proposal_attempts"].append({**attempt, "success": True})
        self._persist()
        return proposal

    def _run_candidate(
        self,
        candidate: dict[str, Any],
        parent_commit: str,
        proposal: dict[str, Any],
        agent: PhaseAgent,
    ) -> dict[str, Any]:
        candidate_id = candidate["id"]
        workspace = Path(candidate["workspace_path"])
        artifact_dir = candidate_artifact_dir(self.session, candidate_id)
        try:
            implementation, implementation_commit = self._implement_candidate(
                workspace,
                candidate_id,
                parent_commit,
                proposal,
                artifact_dir,
                agent,
            )
            commit = implementation_commit["commit_sha"]
            self._require_inputs_unchanged()
            evaluation = commands.candidate_evaluate(self.session_name, candidate_id)
            profiling = {
                "requested": proposal["request_profiling"],
                "reason": proposal["profiling_reason"],
                "result": None,
                "report": None,
                "parent_report": None,
            }
            parent_report_path = (
                candidate_artifact_dir(self.session, candidate["parent_id"])
                / "profiling-result.json"
            )
            if parent_report_path.is_file():
                profiling["parent_report"] = read_json_object(
                    parent_report_path, "parent profiling result"
                )
            if profiling["requested"]:
                profiling["result"] = commands.candidate_profile(self.session_name, candidate_id)
                report_path = artifact_dir / "profiling-result.json"
                if report_path.is_file():
                    profiling["report"] = read_json_object(report_path, "profiling result")
            diff = run_git(
                workspace,
                ["diff", "--no-ext-diff", "--unified=3", parent_commit, commit],
            ).stdout
            status = commands.status(self.session_name)
            comparison = {
                "parent": compact_candidate(
                    commands.candidate_inspect(
                        self.session_name, candidate["parent_id"]
                    )["candidate"]
                ),
                "candidate": compact_candidate(
                    commands.candidate_inspect(self.session_name, candidate_id)["candidate"]
                ),
                "current_best": compact_candidate(status["best_candidate"]),
            }
            reflection, _ = run_phase(
                agent,
                phase="reflect",
                prompt=reflection_prompt(
                    self.objective,
                    proposal,
                    implementation,
                    evaluation["evaluation"],
                    profiling,
                    comparison,
                    truncate_diff(diff),
                ),
                cwd=workspace,
                schema=REFLECTION_SCHEMA,
                artifact_path=artifact_dir / "reflection-agent.json",
            )
            completed = commands.candidate_complete(
                self.session_name,
                candidate_id,
                summary=reflection["summary"],
                decision=reflection["decision"],
                decision_reason=reflection["decision_reason"],
            )["candidate"]
            return {"candidate": completed}
        except Exception as exc:  # noqa: BLE001
            failed = commands.candidate_fail(
                self.session_name,
                candidate_id,
                _error_text(exc),
                interrupted=False,
            )["candidate"]
            return {"candidate": failed}

    def _implement_candidate(
        self,
        workspace: Path,
        candidate_id: str,
        parent_commit: str,
        proposal: dict[str, Any],
        artifact_dir: Path,
        agent: PhaseAgent,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        implementation, _ = run_phase(
            agent,
            phase="implement",
            prompt=implementation_prompt(self.objective, proposal),
            cwd=workspace,
            schema=IMPLEMENTATION_SCHEMA,
            artifact_path=artifact_dir / "implementation-attempt-001.json",
        )
        implementation_commit = commit_candidate_workspace(
            workspace, candidate_id, parent_commit
        )
        write_json(
            artifact_dir / "implementation-commit.json", implementation_commit
        )
        return implementation, implementation_commit

    def _record_candidate(self, candidate: dict[str, Any], *, source: str) -> None:
        candidate_id = candidate["id"]
        if candidate_id in self.recorded_candidates:
            return
        if (
            candidate["status"] == STATUS_COMPLETED
            and candidate["eval_score"] is not None
            and (self.best_score is None or candidate["eval_score"] > self.best_score)
        ):
            self.best_candidate_id = candidate_id
            self.best_score = candidate["eval_score"]
        self.recorded_candidates.add(candidate_id)
        self.result["timeline"].append(
            {
                "terminal_sequence": len(self.result["timeline"]) + 1,
                "candidate_number": len(self.result["timeline"]) + 1,
                "candidate_id": candidate_id,
                "parent_id": candidate["parent_id"],
                "role": (
                    "baseline"
                    if candidate_id == ROOT_CANDIDATE_ID
                    else self.candidate_roles.get(candidate_id)
                ),
                "terminal_at": candidate.get("completed_at") or candidate["updated_at"],
                "run_elapsed_ms": max(
                    0,
                    (candidate.get("completed_at") or candidate["updated_at"])
                    - self.started_at,
                ),
                "status": candidate["status"],
                "score": candidate["eval_score"],
                "decision": candidate["decision"],
                "running_best_candidate_id": self.best_candidate_id,
                "running_best_score": self.best_score,
                "source": source,
            }
        )

    def _record_rejected_candidate(
        self, candidate: dict[str, Any], *, source: str
    ) -> None:
        candidate_id = candidate["id"]
        if candidate_id in self.recorded_candidates:
            return
        self.recorded_candidates.add(candidate_id)
        self.result["rejected_attempts"].append(
            {
                "candidate_id": candidate_id,
                "parent_id": candidate["parent_id"],
                "role": self.candidate_roles.get(candidate_id),
                "rejected_at": candidate.get("completed_at")
                or candidate["updated_at"],
                "reason": candidate["error"],
                "source": source,
            }
        )

    def _sync_terminal_candidates(self, status: dict[str, Any] | None = None) -> None:
        status = status or commands.status(self.session_name)
        for candidate in status["candidates"]:
            source = (
                "existing" if candidate["updated_at"] < self.started_at else "runner"
            )
            if candidate["status"] == STATUS_REJECTED:
                self._record_rejected_candidate(candidate, source=source)
            elif candidate["status"] not in NONTERMINAL_STATUSES:
                self._record_candidate(
                    candidate,
                    source=source,
                )

    def _remaining_budget(self) -> int:
        return commands.status(self.session_name)["candidate_budget"]["remaining"]

    def _require_inputs_unchanged(self) -> None:
        if self.objective_path.read_bytes() != self.objective_bytes:
            raise SparError("session objective changed during the run")
        if self.config_path.read_bytes() != self.config_bytes:
            raise SparError("session configuration changed during the run")
        if run_git(self.repo, ["rev-parse", "HEAD"]).stdout.strip() != self.repo_head:
            raise SparError("repository HEAD changed during the run")
        if run_git(
            self.repo,
            ["status", "--short", "--untracked-files=all"],
        ).stdout:
            raise SparError("repository working tree changed during the run")

    def _persist(self) -> None:
        status = commands.status(self.session_name)
        self.result["budget"] = status["candidate_budget"]
        self.result["counts"] = status["counts"]
        self.result["elapsed_ms"] = (time.monotonic_ns() - self.started_monotonic) // 1_000_000
        temporary = self.run_dir / "run.json.tmp"
        write_json(temporary, self.result)
        temporary.replace(self.run_dir / "run.json")

    def _finish(self, status: str, reason: str) -> dict[str, Any]:
        self.result["status"] = status
        self.result["stop_reason"] = reason
        self.result["completed_at"] = _now_ms()
        self._persist()
        return {**self.result, "artifact_path": str(self.run_dir / "run.json")}


def _error_text(error: Exception) -> str:
    return str(error).strip() or type(error).__name__


@contextmanager
def _session_lock(session: Path) -> Iterator[None]:
    with (session / "runner.lock").open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SparError(f"another runner is active for session: {session.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000
