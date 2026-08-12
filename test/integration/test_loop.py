import fcntl
import json
import re
import signal
import sys
import time
from pathlib import Path
from threading import Barrier, Lock
from typing import Any
from unittest.mock import patch

import pytest

import spar.storage.git as spar_git
from spar import loop
from spar.agent.invocation import (
    IMPLEMENTATION_SCHEMA,
    PROPOSAL_SCHEMA,
    REFLECTION_SCHEMA,
)
from spar.error import SparError
from spar.operation import candidate as candidate_ops
from spar.operation import session as session_ops
from spar.storage.db import DB, Span

from .helpers import (
    chdir,
    run,
    temp_git_repo,
    write_session_config,
)


def test_research_loop_reaches_candidate_limit_with_lineage_decisions_and_failure() -> None:
    with temp_git_repo() as repo:
        session = initialize_research_session(repo, max_candidates=4, max_parallel=1)
        agent = FakeAgent(
            [1, 2, 3],
            discard_values={2},
            implementation_failure_values={3},
        )
        progress: list[str] = []

        with chdir(repo):
            result = start_loop(agent, progress)
            status = session_ops.status("demo")

        assert progress[0] == "[session:demo] [started]"
        assert "[session:demo] [baseline-ready]" in progress
        assert any(line == "[proposal:0001] [started] parent=root" for line in progress)
        assert any(line.startswith("[candidate:") and "[started] parent=root" in line for line in progress)
        assert any(
            line.startswith("[candidate:") and "[completed] decision=keep score=1.0" in line for line in progress
        )
        assert any(line.startswith("[candidate:") and "[failed]" in line for line in progress)
        assert progress[-1] == "[session:demo] [completed] maximum candidates reached"
        assert result == status
        assert result["session"]["status"] == "completed"
        assert result["session"]["stop_reason"] == "maximum candidates reached"
        assert (
            not {
                "status",
                "stop_reason",
                "started_at",
                "completed_at",
                "elapsed_ms",
                "workflow",
                "proposal_attempts",
                "admission_failures",
            }
            & result.keys()
        )
        assert "agent" not in result
        assert len(result["objective_sha256"]) == len(result["config_sha256"]) == 64
        assert Path(result["objective_path"]).is_file()
        assert Path(result["config_path"]).is_file()
        assert result["target_starting_head"] == run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
        assert result["max_candidates"] == 4
        assert result["candidates_used"] == 4

        by_hypothesis = {candidate["hypothesis"]: candidate for candidate in status["candidates"]}
        first = by_hypothesis["value 1 should improve the score"]
        discarded = by_hypothesis["value 2 should improve the score"]
        failed = by_hypothesis["value 3 should improve the score"]
        assert first["parent_id"] == "root"
        assert first["status"] == "completed"
        assert first["decision"] == "keep"
        assert discarded["parent_id"] == first["id"]
        assert discarded["status"] == "completed"
        assert discarded["decision"] == "discard"
        assert failed["parent_id"] == first["id"]
        assert failed["status"] == "failed"
        assert "simulated implementation failure after concrete change" in failed["error"]
        with chdir(repo):
            first_workspace = Path(candidate_ops.inspect("demo", first["id"])["candidate"]["workspace_path"])
        assert [cwd.resolve() for cwd in agent.cwds["proposal"]] == [
            repo.resolve(),
            first_workspace.resolve(),
            first_workspace.resolve(),
        ]
        proposal_progress = [
            json.loads(prompt.split("Research progress:\n", 1)[1]) for prompt in agent.prompts["proposal"]
        ]
        assert all(
            set(progress)
            == {
                "selected_parent",
                "active_sibling_interventions",
                "recent_interventions",
                "profiler_configured",
            }
            for progress in proposal_progress
        )
        assert [progress["selected_parent"]["id"] for progress in proposal_progress] == [
            "root",
            first["id"],
            first["id"],
        ]
        assert proposal_progress[0]["recent_interventions"] == []
        assert proposal_progress[1]["recent_interventions"] == []
        assert proposal_progress[2]["recent_interventions"] == [
            {
                "id": discarded["id"],
                "hypothesis": discarded["hypothesis"],
                "instructions": "set value to 2",
                "eval_score": 2.0,
                "decision": "discard",
                "decision_reason": "deterministic fake chose discard",
                "learnings": "candidate measured 2.0",
                "error": None,
                "profiling_json": None,
            }
        ]

        reflection_evidence = [
            json.loads(prompt.split("Intervention evidence:\n", 1)[1]) for prompt in agent.prompts["reflection"]
        ]
        assert [
            {
                "parent": (evidence["parent"]["id"], evidence["parent"]["eval_score"]),
                "candidate": (
                    evidence["intervention"]["id"],
                    evidence["canonical_evaluation"]["score"],
                ),
            }
            for evidence in reflection_evidence
        ] == [
            {
                "parent": ("root", 0.0),
                "candidate": (first["id"], 1.0),
            },
            {
                "parent": (first["id"], 1.0),
                "candidate": (discarded["id"], 2.0),
            },
        ]

        first_artifacts = session / "artifacts" / "candidates" / first["id"]
        failed_artifacts = session / "artifacts" / "candidates" / failed["id"]
        spans = recorded_spans(session)
        proposal_span = next(item for item in spans if item["kind"] == "proposal")
        implementation_span = next(
            item for item in spans if item["kind"] == "implementation" and item["candidate_id"] == first["id"]
        )
        reflection_span = next(
            item for item in spans if item["kind"] == "reflection" and item["candidate_id"] == first["id"]
        )
        failed_implementation = next(
            item for item in spans if item["kind"] == "implementation" and item["candidate_id"] == failed["id"]
        )
        assert proposal_span["response"] == {
            "hypothesis": "value 1 should improve the score",
            "instructions": "set value to 1",
            "rationale": "measure one deterministic change",
            "profiling_question": None,
        }
        assert proposal_span["prompt"] == agent.prompts["proposal"][0]
        assert proposal_span["schema"] == PROPOSAL_SCHEMA
        assert implementation_span["prompt"] == agent.prompts["implementation"][0]
        assert implementation_span["schema"] == IMPLEMENTATION_SCHEMA
        assert reflection_span["prompt"] == agent.prompts["reflection"][0]
        assert reflection_span["schema"] == REFLECTION_SCHEMA
        assert failed_implementation["success"] is False
        assert (first_artifacts / "implementation.events.jsonl").is_file()
        assert (first_artifacts / "reflection.events.jsonl").is_file()
        assert (failed_artifacts / "implementation.events.jsonl").is_file()
        assert all(
            "Propose the next intervention" in prompt
            and "spar inspect demo <candidate-id>" in prompt
            and "spar inspect --help" in prompt
            for prompt in agent.prompts["proposal"]
        )
        assert all("canonical_evaluation" not in prompt for prompt in agent.prompts["implementation"])
        assert all(str(repo / "evaluate.py") not in prompt for prompt in agent.prompts["implementation"])
        assert all(
            "Maximize the numeric value while testing one change per candidate." in prompt
            for prompt in agent.prompts["implementation"]
        )


def test_research_loop_obeys_parallel_capacity_and_records_timing() -> None:
    observed: dict[int, int] = {}
    parallel_proposal_progress: list[dict[str, Any]] = []
    for max_parallel in (1, 3):
        with temp_git_repo() as repo:
            initialize_research_session(
                repo,
                max_candidates=4 if max_parallel == 3 else 2,
                max_parallel=max_parallel,
            )
            agent = FakeAgent(
                [1, 2, 3],
                implementation_barrier=Barrier(3) if max_parallel == 3 else None,
            )

            with chdir(repo):
                result = start_loop(agent)

            assert result["session"]["status"] == "completed"
            assert result["max_parallel"] == max_parallel
            assert agent.max_active <= max_parallel
            assert result["session"]["elapsed_ms"] >= 0
            observed[max_parallel] = agent.max_active
            if max_parallel == 3:
                parallel_proposal_progress = [
                    json.loads(prompt.split("Research progress:\n", 1)[1]) for prompt in agent.prompts["proposal"]
                ]

    assert observed == {1: 1, 3: 3}
    assert [
        [
            (intervention["hypothesis"], intervention["instructions"])
            for intervention in progress["active_sibling_interventions"]
        ]
        for progress in parallel_proposal_progress
    ] == [
        [],
        [("value 1 should improve the score", "set value to 1")],
        [
            ("value 1 should improve the score", "set value to 1"),
            ("value 2 should improve the score", "set value to 2"),
        ],
    ]


def test_research_loop_profiles_only_on_structured_request() -> None:
    with temp_git_repo() as repo:
        session = initialize_research_session(
            repo,
            max_candidates=3,
            max_parallel=1,
            profiling=True,
        )
        agent = FakeAgent([1, 2], profile_values={2})
        with DB(session) as db:
            root = db.candidate("root")
        with Span(session, candidate_id="root", kind="profiling") as span, span.complete() as transaction:
            assert transaction.update_candidate(
                "root",
                root["status"],
                {"profiling": {"observed_value": "baseline"}},
            )

        with chdir(repo):
            start_loop(agent)
        spans = recorded_spans(session)

        profiling = [span for span in spans if span["kind"] == "profiling" and span["candidate_id"] != "root"]
        assert len(profiling) == 1
        profiled_spans = [span["kind"] for span in spans if span["candidate_id"] == profiling[0]["candidate_id"]]
        assert profiled_spans == [
            "worktree",
            "implementation",
            "evaluation",
            "profiling",
            "reflection",
            "finalization",
        ]
        proposal_progress = [
            json.loads(prompt.split("Research progress:\n", 1)[1]) for prompt in agent.prompts["proposal"]
        ]
        assert json.loads(proposal_progress[0]["selected_parent"]["profiling_json"]) == {"observed_value": "baseline"}
        assert all(
            "The profiler runs after implementation and informs reflection" in prompt
            for prompt in agent.prompts["proposal"]
        )
        profiled_candidate_id = profiling[0]["candidate_id"]
        with DB(session) as db:
            assert json.loads(db.candidate(profiled_candidate_id)["profiling_json"]) == {"observed_value": "2"}
        assert any('"observed_value": "2"' in prompt for prompt in agent.prompts["reflection"])
        reflection_evidence = [
            json.loads(prompt.split("Intervention evidence:\n", 1)[1]) for prompt in agent.prompts["reflection"]
        ]
        assert reflection_evidence[0]["profiles"]["parent"] == {"observed_value": "baseline"}


def test_proposal_failures_are_recorded_and_capped() -> None:
    with temp_git_repo() as repo:
        session = initialize_research_session(repo, max_candidates=2, max_parallel=1)
        agent = FakeAgent([], proposal_failures=10)

        with chdir(repo):
            result = start_loop(agent)

        assert result["session"]["status"] == "blocked"
        assert result["max_candidates"] == 2
        assert result["candidates_used"] == 1
        assert len(proposal_spans(result)) == 3
        assert all(not attempt["success"] for attempt in proposal_spans(result))
        assert "3 consecutive times" in result["session"]["stop_reason"]
        assert Path(result["workspace_root"]).resolve().is_relative_to(session.resolve())
        for attempt in proposal_spans(result):
            assert (session / attempt["events_path"]).is_file()
        with DB(session) as db:
            assert all(not span["success"] for span in db.spans() if span["kind"] == "proposal")


def test_research_loop_counts_an_unchanged_program_from_the_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with temp_git_repo() as repo:
        session = initialize_research_session(repo, max_candidates=2, max_parallel=1)
        agent = FakeAgent([1])
        original_commit = spar_git.commit_candidate_workspace
        commit_attempts = 0

        def reject_first_commit(workspace: Path, candidate_id: str, parent_commit: str) -> dict[str, Any]:
            nonlocal commit_attempts
            commit_attempts += 1
            if commit_attempts == 1:
                raise SparError("candidate program is unchanged")
            return original_commit(workspace, candidate_id, parent_commit)

        monkeypatch.setattr(spar_git, "commit_candidate_workspace", reject_first_commit)
        with chdir(repo):
            result = start_loop(agent)
            status = session_ops.status("demo")

        candidate = next(item for item in status["candidates"] if item["id"] != "root")
        artifacts = session / "artifacts" / "candidates" / candidate["id"]
        assert result["session"]["status"] == "completed"
        assert result["max_candidates"] == 2
        assert result["candidates_used"] == 2
        assert candidate["status"] == "failed"
        assert candidate["error"] == "candidate program is unchanged"
        assert commit_attempts == 1
        assert len(agent.prompts["implementation"]) == 1
        assert (artifacts / "implementation.events.jsonl").is_file()


def test_research_loop_counts_an_implementation_with_no_workspace_changes() -> None:
    with temp_git_repo() as repo:
        session = initialize_research_session(repo, max_candidates=2, max_parallel=1)
        agent = FakeAgent([1], no_change_attempts={1: 1})

        with chdir(repo):
            result = start_loop(agent)
            status = session_ops.status("demo")

        candidate = next(item for item in status["candidates"] if item["id"] != "root")
        artifacts = session / "artifacts" / "candidates" / candidate["id"]
        assert result["session"]["status"] == "completed"
        assert result["max_candidates"] == 2
        assert result["candidates_used"] == 2
        assert candidate["status"] == "failed"
        assert candidate["error"] == "candidate workspace is unchanged"
        assert len(agent.prompts["implementation"]) == 1
        assert (artifacts / "implementation.events.jsonl").is_file()


def test_research_loop_counts_unchanged_attempts_as_candidates() -> None:
    with temp_git_repo() as repo:
        initialize_research_session(repo, max_candidates=3, max_parallel=1)
        agent = FakeAgent([1, 2], no_change_values={1, 2})

        with chdir(repo):
            result = start_loop(agent)
            status = session_ops.status("demo")

        failed = [candidate for candidate in status["candidates"] if candidate["status"] == "failed"]
        assert result["session"]["status"] == "completed"
        assert result["max_candidates"] == 3
        assert result["candidates_used"] == 3
        assert len(failed) == 2
        assert len(proposal_spans(result)) == 2
        assert_proposal_artifacts_survive(result)


def test_parallel_unchanged_attempts_count_as_candidates() -> None:
    with temp_git_repo() as repo:
        initialize_research_session(repo, max_candidates=4, max_parallel=3)
        agent = FakeAgent(
            [1, 2, 3],
            no_change_values={2},
            implementation_delay=0.05,
        )

        with chdir(repo):
            result = start_loop(agent)
            status = session_ops.status("demo")

        failed = [candidate for candidate in status["candidates"] if candidate["status"] == "failed"]
        assert result["session"]["status"] == "completed"
        assert result["max_candidates"] == 4
        assert result["candidates_used"] == 4
        assert len(status["candidates"]) == 4
        assert len(failed) == 1
        assert len(proposal_spans(result)) == 3
        assert_proposal_artifacts_survive(result)


def test_worktree_creation_failure_counts_as_a_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with temp_git_repo() as repo:
        initialize_research_session(repo, max_candidates=2, max_parallel=1)
        agent = FakeAgent([1])

        def fail_worktree(repo: Path, workspace: Path, commit: str) -> None:
            del repo, workspace, commit
            raise SparError("simulated worktree failure")

        monkeypatch.setattr(spar_git, "add_worktree", fail_worktree)
        with chdir(repo):
            result = start_loop(agent)
            status = session_ops.status("demo")

        assert result["session"]["status"] == "completed"
        assert result["max_candidates"] == 2
        assert result["candidates_used"] == 2
        failed = next(candidate for candidate in status["candidates"] if candidate["status"] == "failed")
        assert failed["error"] == "could not create candidate worktree: simulated worktree failure"
        spans = recorded_spans(Path(result["session_dir"]))
        assert len([span for span in spans if span["kind"] == "worktree" and span["error"]]) == 1
        assert status["counts"] == {"completed": 1, "failed": 1}


def test_restart_marks_abandoned_candidates_interrupted() -> None:
    with temp_git_repo() as repo:
        initialize_research_session(repo, max_candidates=2, max_parallel=1)
        with chdir(repo):
            candidate_ops.evaluate("demo", "root")
            candidate_ops.complete(
                "demo",
                "root",
                learnings="baseline measured",
                decision="keep",
                decision_reason="valid baseline",
            )
            started = candidate_ops.start(
                "demo",
                parent_id="root",
                hypothesis="retry this intervention",
                instructions="change the value",
                rationale="simulate a lost worker",
            )
            session_ops.start("demo")
            result = start_loop(FakeAgent([]))

        candidate = next(item for item in result["candidates"] if item["id"] == started["candidate"]["id"])
        assert candidate["status"] == "interrupted"
        assert candidate["error"] == ("previous SPAR process ended before candidate completion")


def test_research_loop_rejects_a_second_process_for_the_same_session() -> None:
    with temp_git_repo() as repo:
        session = initialize_research_session(repo, max_candidates=1, max_parallel=1)
        with (session / "session.lock").open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with chdir(repo), pytest.raises(SparError, match="session is already active"):
                start_loop(FakeAgent([]))


def test_completed_session_start_is_a_noop_with_the_existing_status() -> None:
    with temp_git_repo() as repo:
        initialize_research_session(repo, max_candidates=1, max_parallel=1)
        first_agent = FakeAgent([])
        second_agent = FakeAgent([1])

        with chdir(repo):
            first = start_loop(first_agent)
            second = start_loop(second_agent)

        assert first["session"]["status"] == second["session"]["status"] == "completed"
        assert second == first
        assert not second_agent.prompts["proposal"]


def test_graceful_stop_handler_requests_stop() -> None:
    with temp_git_repo() as repo:
        initialize_research_session(repo, max_candidates=2, max_parallel=1)
        with chdir(repo):
            session_ops.start("demo")
            with loop._graceful_stop("demo"):
                handler = signal.getsignal(signal.SIGINT)
                assert callable(handler)
                handler(signal.SIGINT, None)
            status = session_ops.status("demo")

        assert status["session"]["stop_requested"] is True


def start_loop(agent: FakeAgent, progress: list[str] | None = None) -> dict[str, Any]:
    callback = None if progress is None else progress.append
    with patch.object(loop, "create_agent", return_value=agent):
        return loop.ResearchLoop("demo", progress=callback).start()


class FakeAgent:
    def __init__(
        self,
        values: list[int],
        *,
        discard_values: set[int] | None = None,
        no_change_values: set[int] | None = None,
        no_change_attempts: dict[int, int] | None = None,
        implementation_failure_values: set[int] | None = None,
        profile_values: set[int] | None = None,
        proposal_failures: int = 0,
        implementation_delay: float = 0,
        implementation_delays: dict[int, float] | None = None,
        implementation_barrier: Barrier | None = None,
    ) -> None:
        self.values = values
        self.discard_values = discard_values or set()
        self.no_change_values = no_change_values or set()
        self.no_change_attempts = dict(no_change_attempts or {})
        self.implementation_failure_values = implementation_failure_values or set()
        self.profile_values = profile_values or set()
        self.proposal_failures = proposal_failures
        self.implementation_delay = implementation_delay
        self.implementation_delays = implementation_delays or {}
        self.implementation_barrier = implementation_barrier
        self.next_proposal = 0
        self.active = 0
        self.max_active = 0
        self.lock = Lock()
        self.prompts: dict[str, list[str]] = {kind: [] for kind in ("proposal", "implementation", "reflection")}
        self.cwds: dict[str, list[Path]] = {kind: [] for kind in self.prompts}

    def invoke(
        self,
        *,
        kind: str,
        prompt: str,
        cwd: Path,
        schema: dict[str, Any],
        event_path: Path | None = None,
    ) -> dict[str, Any]:
        del schema
        if event_path is not None:
            event_path.parent.mkdir(parents=True, exist_ok=True)
            event_path.write_text(json.dumps({"kind": kind}) + "\n", encoding="utf-8")
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.prompts[kind].append(prompt)
            self.cwds[kind].append(cwd)
        try:
            if kind == "proposal":
                return self._propose()
            value = int(re.search(r"set value to (\d+)", prompt).group(1))
            if kind == "implementation":
                return self._implement(cwd, value)
            score = float(re.search(r'"score": (-?\d+(?:\.\d+)?)', prompt).group(1))
            decision = "discard" if int(score) in self.discard_values else "keep"
            return {
                "decision": decision,
                "learnings": f"candidate measured {score}",
                "decision_reason": f"deterministic fake chose {decision}",
            }
        finally:
            with self.lock:
                self.active -= 1

    def _propose(self) -> dict[str, Any]:
        with self.lock:
            if self.proposal_failures:
                self.proposal_failures -= 1
                raise RuntimeError("simulated proposal failure")
            value = self.values[self.next_proposal]
            self.next_proposal += 1
        return {
            "hypothesis": f"value {value} should improve the score",
            "instructions": f"set value to {value}",
            "rationale": "measure one deterministic change",
            "profiling_question": ("confirm the observed value" if value in self.profile_values else None),
        }

    def _implement(self, cwd: Path, value: int) -> dict[str, Any]:
        if self.implementation_barrier is not None:
            self.implementation_barrier.wait(timeout=30)
        delay = self.implementation_delays.get(value, self.implementation_delay)
        if delay:
            time.sleep(delay)
        with self.lock:
            no_change_attempts = self.no_change_attempts.get(value, 0)
            if no_change_attempts:
                self.no_change_attempts[value] = no_change_attempts - 1
        if value not in self.no_change_values and not no_change_attempts:
            (cwd / "value.txt").write_text(f"{value}\n", encoding="utf-8")
        if value in self.implementation_failure_values:
            raise RuntimeError("simulated implementation failure after concrete change")
        return {
            "summary": f"set value to {value}",
            "limitations": [],
        }


def initialize_research_session(
    repo: Path,
    *,
    max_candidates: int,
    max_parallel: int,
    profiling: bool = False,
) -> Path:
    (repo / "value.txt").write_text("0\n", encoding="utf-8")
    (repo / "evaluate.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "value = float(Path('value.txt').read_text(encoding='utf-8'))\n"
        "print(json.dumps({'score': value, 'value': value}))\n",
        encoding="utf-8",
    )
    paths = ["value.txt", "evaluate.py"]
    if profiling:
        (repo / "profile.py").write_text(
            "import json\n"
            "from pathlib import Path\n"
            "value = Path('value.txt').read_text(encoding='utf-8').strip()\n"
            "print(json.dumps({'observed_value': value}))\n",
            encoding="utf-8",
        )
        paths.append("profile.py")
    run(["git", "add", *paths], repo)
    run(["git", "commit", "-m", "initial"], repo)
    with chdir(repo):
        session_ops.init("demo")
    session = repo / ".spar" / "demo"
    write_session_config(
        session,
        max_candidates=max_candidates,
        max_parallel=max_parallel,
        evaluation_command=[sys.executable, str(repo / "evaluate.py")],
        profiling_command=([sys.executable, str(repo / "profile.py")] if profiling else None),
    )
    (session / "objective.md").write_text(
        "# Objective\n\nMaximize the numeric value while testing one change per candidate.\n",
        encoding="utf-8",
    )
    return session


def assert_proposal_artifacts_survive(result: dict[str, Any]) -> None:
    assert all((Path(result["session_dir"]) / attempt["events_path"]).is_file() for attempt in proposal_spans(result))


def proposal_spans(status: dict[str, Any]) -> list[dict[str, Any]]:
    return [span for span in recorded_spans(Path(status["session_dir"])) if span["kind"] == "proposal"]


def recorded_spans(session: Path) -> list[dict[str, Any]]:
    with DB(session) as db:
        spans = db.spans()
    return [{**span, **(span["invocation"] or {})} for span in spans]
