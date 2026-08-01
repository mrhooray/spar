import fcntl
import json
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from threading import Barrier, Lock
from typing import Any

import pytest

import spar.repo as spar_repo
from spar import commands
from spar.agent import (
    IMPLEMENTATION_SCHEMA,
    PROPOSAL_SCHEMA,
    REFLECTION_SCHEMA,
    CodexAgent,
)
from spar.errors import SparError
from spar.run_cli import main as run_cli_main
from spar.runner import run_session
from spar.tests.test_cli import chdir, complete_root, run, temp_git_repo


def test_runner_consumes_budget_with_lineage_decisions_and_failure() -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=4, max_parallel=1)
        agent = FakeAgent(
            [1, 2, 3],
            discard_values={2},
            implementation_failure_values={3},
        )

        with chdir(repo):
            result = run_session("demo", agent, workspace_root=session / "runner-worktrees")
            status = commands.status("demo")

        assert result["status"] == "completed"
        assert result["stop_reason"] == "candidate budget exhausted"
        assert result["agent"] == {"adapter": "FakeAgent"}
        assert len(result["objective_sha256"]) == len(result["config_sha256"]) == 64
        assert Path(result["objective_snapshot"]).is_file()
        assert Path(result["config_snapshot"]).is_file()
        assert result["target_starting_head"] == run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
        assert result["budget"] == {"maximum": 4, "used": 4, "remaining": 0}
        assert [item["running_best_score"] for item in result["timeline"]] == [0.0, 1.0, 2.0, 2.0]
        assert all("session_elapsed_ms" not in item for item in result["timeline"])

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
        assert (
            "simulated implementation failure after concrete change" in failed["error"]
        )
        with chdir(repo):
            first_workspace = Path(
                commands.candidate_inspect("demo", first["id"])["candidate"]["workspace_path"]
            )
        assert [cwd.resolve() for cwd in agent.cwds["propose"]] == [
            repo.resolve(),
            first_workspace.resolve(),
            first_workspace.resolve(),
        ]

        reflection_evidence = [
            json.loads(prompt.split("Canonical result, diff, and reports:\n", 1)[1])
            for prompt in agent.prompts["reflect"]
        ]
        assert [
            {
                name: (
                    evidence["comparison"][name]["id"],
                    evidence["comparison"][name]["eval_score"],
                )
                for name in ("parent", "candidate", "current_best")
            }
            for evidence in reflection_evidence
        ] == [
            {
                "parent": ("root", 0.0),
                "candidate": (first["id"], 1.0),
                "current_best": ("root", 0.0),
            },
            {
                "parent": (first["id"], 1.0),
                "candidate": (discarded["id"], 2.0),
                "current_best": (first["id"], 1.0),
            },
        ]

        first_artifacts = session / "artifacts" / "candidates" / first["id"]
        failed_artifacts = session / "artifacts" / "candidates" / failed["id"]
        proposal_artifact = (
            Path(result["artifact_path"]).parent / result["proposal_attempts"][0]["path"]
        )
        assert json_load(proposal_artifact)["phase"] == "propose"
        first_implementation = first_artifacts / "implementation-attempt-001.json"
        failed_implementation = failed_artifacts / "implementation-attempt-001.json"
        assert json_load(first_implementation)["success"] is True
        assert json_load(first_artifacts / "implementation-commit.json")[
            "changed_paths"
        ] == ["value.txt"]
        assert json_load(first_artifacts / "reflection-agent.json")["success"] is True
        assert json_load(proposal_artifact)["prompt"] == agent.prompts["propose"][0]
        assert json_load(proposal_artifact)["schema"] == PROPOSAL_SCHEMA
        assert not (first_artifacts / "proposal.json").exists()
        assert json_load(first_implementation)["prompt"] == agent.prompts["implement"][0]
        assert json_load(first_implementation)["schema"] == IMPLEMENTATION_SCHEMA
        assert json_load(first_artifacts / "reflection-agent.json")["prompt"] == agent.prompts["reflect"][0]
        assert json_load(first_artifacts / "reflection-agent.json")["schema"] == REFLECTION_SCHEMA
        assert json_load(failed_implementation)["success"] is False
        assert Path(result["artifact_path"]).is_file()
        assert all(
            "inspect the selected parent's task-local source" in prompt
            and "concrete existing code location and mechanism" in prompt
            for prompt in agent.prompts["propose"]
        )
        assert all("canonical_evaluation" not in prompt for prompt in agent.prompts["implement"])
        assert all(str(repo / "evaluate.py") not in prompt for prompt in agent.prompts["implement"])
        assert all(
            "Maximize the numeric value while testing one change per candidate." in prompt
            for prompt in agent.prompts["implement"]
        )


def test_runner_can_preserve_proposal_context_without_sharing_worker_context() -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=2, max_parallel=1)
        proposal_agent = FakeAgent([1])
        worker_agent = FakeAgent([])

        with chdir(repo):
            result = run_session(
                "demo",
                worker_agent,
                proposal_agent=proposal_agent,
                workspace_root=session / "runner-worktrees",
            )

        assert result["status"] == "completed"
        assert len(proposal_agent.prompts["propose"]) == 1
        assert not proposal_agent.prompts["implement"]
        assert not proposal_agent.prompts["reflect"]
        assert not worker_agent.prompts["propose"]
        assert len(worker_agent.prompts["implement"]) == 1
        assert len(worker_agent.prompts["reflect"]) == 1


def test_runner_preserves_one_hybrid_spine_with_two_mcts_explorers() -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=7, max_parallel=3)
        spine_agent = FakeAgent([1, 2], implementation_delay=0.4)
        proposal_agent = FakeAgent([10, 20, 30, 40])
        worker_agent = FakeAgent(
            [],
            implementation_delays={10: 0.2, 20: 0.2, 30: 0.8, 40: 0.8},
        )

        with chdir(repo):
            result = run_session(
                "demo",
                worker_agent,
                proposal_agent=proposal_agent,
                spine_agent=spine_agent,
                workspace_root=session / "runner-worktrees",
            )

        assert result["status"] == "completed"
        assert result["candidate_lanes"] == {"spine": 1, "explorer": 2}
        assert [attempt["role"] for attempt in result["proposal_attempts"]].count(
            "spine"
        ) == 2
        assert [item["role"] for item in result["timeline"]].count("spine") == 2
        assert [item["role"] for item in result["timeline"]].count("explorer") == 4

        assert {phase: len(prompts) for phase, prompts in spine_agent.prompts.items()} == {
            "propose": 2,
            "implement": 2,
            "reflect": 2,
        }
        assert len(proposal_agent.prompts["propose"]) == 4
        assert not proposal_agent.prompts["implement"]
        assert not proposal_agent.prompts["reflect"]
        assert not worker_agent.prompts["propose"]
        assert len(worker_agent.prompts["implement"]) == 4
        assert len(worker_agent.prompts["reflect"]) == 4
        assert spine_agent.max_active == 1
        assert worker_agent.max_active == 2

        spine_attempts = [
            attempt
            for attempt in result["proposal_attempts"]
            if attempt["role"] == "spine"
        ]
        assert spine_attempts[1]["parent_id"] != "root"

        run_dir = Path(result["artifact_path"]).parent
        for attempt in result["proposal_attempts"]:
            artifact = json_load(run_dir / attempt["path"])
            evidence = json.loads(
                artifact["prompt"].split(
                    "Compact state, MCTS ranking, and prior evidence:\n", 1
                )[1]
            )
            assert evidence["search_role"] == attempt["role"]
            assert evidence["selected_parent"]["id"] == attempt["parent_id"]
            if attempt["role"] == "spine":
                assert evidence["selected_parent"]["id"] == evidence["best_candidate"][
                    "id"
                ]


def test_runner_obeys_parallel_capacity_and_exposes_wall_clock_timing() -> None:
    elapsed: dict[int, int] = {}
    observed: dict[int, int] = {}
    parallel_proposal_evidence: list[dict[str, Any]] = []
    for max_parallel in (1, 3):
        with temp_git_repo() as repo:
            session = initialize_runner_session(repo, max_candidates=4, max_parallel=max_parallel)
            agent = FakeAgent(
                [1, 2, 3],
                implementation_delay=0.5,
                implementation_barrier=Barrier(3) if max_parallel == 3 else None,
            )

            with chdir(repo):
                result = run_session("demo", agent, workspace_root=session / "runner-worktrees")

            assert result["status"] == "completed"
            assert result["max_parallel"] == max_parallel
            assert agent.max_active <= max_parallel
            elapsed[max_parallel] = result["elapsed_ms"]
            observed[max_parallel] = agent.max_active
            if max_parallel == 3:
                parallel_proposal_evidence = [
                    json.loads(
                        prompt.split(
                            "Compact state, MCTS ranking, and prior evidence:\n", 1
                        )[1]
                    )
                    for prompt in agent.prompts["propose"]
                ]

    assert observed == {1: 1, 3: 3}
    assert elapsed[3] < elapsed[1]
    assert [
        [
            (intervention["hypothesis"], intervention["instructions"])
            for intervention in evidence["active_sibling_interventions"]
        ]
        for evidence in parallel_proposal_evidence
    ] == [
        [],
        [("value 1 should improve the score", "set value to 1")],
        [
            ("value 1 should improve the score", "set value to 1"),
            ("value 2 should improve the score", "set value to 2"),
        ],
    ]


def test_runner_profiles_only_on_structured_request() -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(
            repo,
            max_candidates=3,
            max_parallel=1,
            profiling=True,
        )
        agent = FakeAgent([1, 2], profile_values={2})
        root_report = session / "artifacts" / "candidates" / "root" / "profiling-result.json"
        root_report.parent.mkdir(parents=True, exist_ok=True)
        root_report.write_text(
            json.dumps({"observed_value": "baseline"}) + "\n",
            encoding="utf-8",
        )

        with chdir(repo):
            result = run_session("demo", agent, workspace_root=session / "runner-worktrees")
            operations = commands.status("demo")["operations"]

        profiling = [operation for operation in operations if operation["operation"] == "profiling"]
        assert len(profiling) == 1
        profiled_operations = [
            operation["operation"]
            for operation in operations
            if operation["candidate_id"] == profiling[0]["candidate_id"]
        ]
        assert profiled_operations == ["worktree", "evaluation", "profiling", "completion"]
        assert all(
            "Requested profiling runs only after the" in prompt
            and "canonically evaluated, solely to inform reflection" in prompt
            for prompt in agent.prompts["propose"]
        )
        profiled = next(
            item for item in result["timeline"] if item["candidate_id"] == profiling[0]["candidate_id"]
        )
        report = session / "artifacts" / "candidates" / profiled["candidate_id"] / "profiling-result.json"
        assert json_load(report) == {"observed_value": "2"}
        assert any('"observed_value": "2"' in prompt for prompt in agent.prompts["reflect"])
        reflection_evidence = [
            json.loads(prompt.split("Canonical result, diff, and reports:\n", 1)[1])
            for prompt in agent.prompts["reflect"]
        ]
        assert reflection_evidence[0]["profiling"]["parent_report"] == {
            "observed_value": "baseline"
        }


def test_proposal_failures_are_recorded_and_capped() -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=2, max_parallel=1)
        agent = FakeAgent([], proposal_failures=10)

        with chdir(repo):
            result = run_session("demo", agent)

        assert result["status"] == "blocked"
        assert result["budget"] == {"maximum": 2, "used": 1, "remaining": 1}
        assert len(result["proposal_attempts"]) == 3
        assert all(not attempt["success"] for attempt in result["proposal_attempts"])
        assert "3 consecutive times" in result["stop_reason"]
        assert not Path(result["workspace_root"]).is_relative_to(session)
        for attempt in result["proposal_attempts"]:
            artifact = Path(result["artifact_path"]).parent / attempt["path"]
            assert json_load(artifact)["success"] is False


def test_runner_counts_a_program_rejected_by_the_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=2, max_parallel=1)
        agent = FakeAgent([1])
        original_run_git = spar_repo.run_git
        commit_attempts = 0

        def reject_first_commit(
            cwd: Path, args: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            nonlocal commit_attempts
            if args[:2] == ["commit", "-m"]:
                commit_attempts += 1
                if commit_attempts == 1:
                    return subprocess.CompletedProcess(
                        ["git", *args], 1, "", "SPAR_PROGRAM_UNCHANGED\n"
                    )
            return original_run_git(cwd, args, **kwargs)

        monkeypatch.setattr(spar_repo, "run_git", reject_first_commit)
        with chdir(repo):
            result = run_session(
                "demo", agent, workspace_root=session / "runner-worktrees"
            )
            status = commands.status("demo")

        candidate = next(item for item in status["candidates"] if item["id"] != "root")
        artifacts = session / "artifacts" / "candidates" / candidate["id"]
        assert result["status"] == "completed"
        assert result["budget"] == {"maximum": 2, "used": 2, "remaining": 0}
        assert candidate["status"] == "failed"
        assert candidate["error"] == "candidate program is unchanged"
        assert commit_attempts == 1
        assert len(agent.prompts["implement"]) == 1
        assert (artifacts / "implementation-attempt-001.json").is_file()
        assert not (artifacts / "implementation-commit.json").exists()


def test_runner_counts_an_implementation_with_no_workspace_changes() -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=2, max_parallel=1)
        agent = FakeAgent([1], no_change_attempts={1: 1})

        with chdir(repo):
            result = run_session(
                "demo", agent, workspace_root=session / "runner-worktrees"
            )
            status = commands.status("demo")

        candidate = next(item for item in status["candidates"] if item["id"] != "root")
        artifacts = session / "artifacts" / "candidates" / candidate["id"]
        assert result["status"] == "completed"
        assert result["budget"] == {"maximum": 2, "used": 2, "remaining": 0}
        assert candidate["status"] == "failed"
        assert candidate["error"] == "candidate workspace is unchanged"
        assert len(agent.prompts["implement"]) == 1
        assert (artifacts / "implementation-attempt-001.json").is_file()
        assert not (artifacts / "implementation-commit.json").exists()


def test_runner_counts_unchanged_attempts_toward_the_budget() -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=3, max_parallel=1)
        agent = FakeAgent([1, 2], no_change_values={1, 2})

        with chdir(repo):
            result = run_session(
                "demo", agent, workspace_root=session / "runner-worktrees"
            )
            status = commands.status("demo")

        failed = [
            candidate
            for candidate in status["candidates"]
            if candidate["status"] == "failed"
        ]
        assert result["status"] == "completed"
        assert result["budget"] == {"maximum": 3, "used": 3, "remaining": 0}
        assert len(failed) == 2
        assert not result["rejected_attempts"]
        assert len(result["proposal_attempts"]) == 2
        assert [item["candidate_number"] for item in result["timeline"]] == [1, 2, 3]
        assert_proposal_artifacts_survive(result)


def test_unadmitted_rejection_is_invalid_after_evaluation() -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=2, max_parallel=1)
        complete_root(repo, session, score=0.0)
        with chdir(repo):
            started = commands.candidate_start(
                "demo",
                parent_id="root",
                hypothesis="increase the value",
                instructions="set value to 1",
                rationale="measure a concrete change",
            )
        candidate_id = started["candidate"]["id"]
        workspace = Path(started["candidate"]["workspace_path"])
        (workspace / "value.txt").write_text("1\n", encoding="utf-8")
        run(["git", "add", "value.txt"], workspace)
        run(["git", "commit", "-m", "candidate"], workspace)

        with chdir(repo):
            commands.candidate_evaluate("demo", candidate_id)
            with pytest.raises(SparError, match="not an unadmitted implementation"):
                commands.reject_unadmitted_candidate("demo", candidate_id, "too late")

        with chdir(repo):
            budget = commands.status("demo")["candidate_budget"]

        assert budget == {
            "maximum": 2,
            "used": 2,
            "remaining": 0,
        }


def test_parallel_unchanged_attempt_counts_with_terminal_numbering() -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=4, max_parallel=3)
        agent = FakeAgent(
            [1, 2, 3],
            no_change_values={2},
            implementation_delay=0.05,
        )

        with chdir(repo):
            result = run_session(
                "demo", agent, workspace_root=session / "runner-worktrees"
            )
            status = commands.status("demo")

        failed = [
            candidate
            for candidate in status["candidates"]
            if candidate["status"] == "failed"
        ]
        assert result["status"] == "completed"
        assert result["budget"] == {"maximum": 4, "used": 4, "remaining": 0}
        assert len(status["candidates"]) == 4
        assert len(failed) == 1
        assert len(result["proposal_attempts"]) == 3
        assert [item["terminal_sequence"] for item in result["timeline"]] == [1, 2, 3, 4]
        assert [item["candidate_number"] for item in result["timeline"]] == [1, 2, 3, 4]
        assert failed[0]["id"] in {
            item["candidate_id"] for item in result["timeline"]
        }
        assert_proposal_artifacts_survive(result)


def test_worktree_creation_failure_consumes_budget_and_finishes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=2, max_parallel=1)
        agent = FakeAgent([1])
        original_run_git = commands.run_git

        def fail_worktree(
            cwd: Path, args: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            if args[:2] == ["worktree", "add"]:
                raise SparError("simulated worktree failure")
            return original_run_git(cwd, args, **kwargs)

        monkeypatch.setattr(commands, "run_git", fail_worktree)
        with chdir(repo):
            result = run_session("demo", agent, workspace_root=session / "runner-worktrees")
            status = commands.status("demo")

        assert result["status"] == "completed"
        assert result["budget"] == {"maximum": 2, "used": 2, "remaining": 0}
        assert len(result["admission_failures"]) == 1
        assert [item["status"] for item in result["timeline"]] == ["completed", "interrupted"]
        assert status["counts"] == {"completed": 1, "interrupted": 1}
        assert json_load(Path(result["artifact_path"]))["status"] == "completed"


def test_runner_rejects_a_second_process_for_the_same_session() -> None:
    with temp_git_repo() as repo:
        session = initialize_runner_session(repo, max_candidates=1, max_parallel=1)
        with (session / "runner.lock").open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with chdir(repo), pytest.raises(SparError, match="another runner is active"):
                run_session("demo", FakeAgent([]))


def test_codex_adapter_uses_explicit_model_effort_and_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["input"] = kwargs["input"]
        response_path = Path(command[command.index("--output-last-message") + 1])
        response_path.write_text(
            json.dumps(
                {
                    "hypothesis": "test",
                    "instructions": "change one thing",
                    "rationale": "measure it",
                    "request_profiling": False,
                    "profiling_reason": "",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, '{"type":"audit-event"}\n', "")

    monkeypatch.setattr("spar.agent.subprocess.run", fake_run)
    adapter = CodexAgent(model="gpt-test", effort="high", timeout_seconds=30)

    response = adapter.run(
        phase="propose",
        prompt="bounded prompt",
        cwd=tmp_path,
        schema=PROPOSAL_SCHEMA,
        event_path=tmp_path / "propose.codex.jsonl",
    )

    command = captured["command"]
    single_agent_configs = {
        "agents.enabled=false",
        "features.multi_agent=false",
        "features.multi_agent_v2=false",
    }

    def assert_phase_command(command: list[str], sandbox: str) -> None:
        configs = {
            command[index + 1]
            for index, part in enumerate(command[:-1])
            if part == "--config"
        }
        assert command[command.index("--sandbox") + 1] == sandbox
        assert single_agent_configs <= configs

    assert response["hypothesis"] == "test"
    assert command[command.index("--model") + 1] == "gpt-test"
    assert command[command.index("--config") + 1] == "model_reasoning_effort=high"
    assert_phase_command(command, "read-only")
    assert "--output-schema" in command
    assert "--json" in command
    assert captured["input"] == "bounded prompt"
    assert (tmp_path / "propose.codex.jsonl").read_text(encoding="utf-8") == (
        '{"type":"audit-event"}\n'
    )

    adapter.run(
        phase="implement",
        prompt="task-local prompt",
        cwd=tmp_path,
        schema=PROPOSAL_SCHEMA,
        event_path=tmp_path / "implement.codex.jsonl",
    )
    assert_phase_command(captured["command"], "workspace-write")

    adapter.run(
        phase="reflect",
        prompt="evidence prompt",
        cwd=tmp_path,
        schema=PROPOSAL_SCHEMA,
        event_path=tmp_path / "reflect.codex.jsonl",
    )
    assert_phase_command(captured["command"], "read-only")


def test_codex_adapter_preserves_one_researcher_thread_across_phase_worktrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    thread_ids = ["thread-123", "thread-123", "thread-other"]

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        response_path = Path(command[command.index("--output-last-message") + 1])
        response_path.write_text(
            json.dumps(
                {
                    "hypothesis": "test",
                    "instructions": "change one thing",
                    "rationale": "measure it",
                    "request_profiling": False,
                    "profiling_reason": "",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {"type": "thread.started", "thread_id": thread_ids[len(commands) - 1]}
            )
            + "\n",
            "",
        )

    monkeypatch.setattr("spar.agent.subprocess.run", fake_run)
    adapter = CodexAgent(
        model="gpt-test",
        effort="medium",
        timeout_seconds=30,
        preserve_context=True,
    )
    parent = tmp_path / "parent"
    candidate = tmp_path / "candidate"
    parent.mkdir()
    candidate.mkdir()

    adapter.run(
        phase="propose",
        prompt="choose an intervention",
        cwd=parent,
        schema=PROPOSAL_SCHEMA,
        event_path=tmp_path / "propose.codex.jsonl",
    )
    adapter.run(
        phase="implement",
        prompt="implement the intervention",
        cwd=candidate,
        schema=PROPOSAL_SCHEMA,
        event_path=tmp_path / "implement.codex.jsonl",
    )

    initial, resumed = commands
    assert "--ephemeral" not in initial
    assert "resume" not in initial
    assert initial[initial.index("--sandbox") + 1] == "read-only"
    assert initial[initial.index("--cd") + 1] == str(parent)
    assert resumed[resumed.index("--sandbox") + 1] == "workspace-write"
    assert resumed[resumed.index("--cd") + 1] == str(candidate)
    assert resumed[-2:] == ["thread-123", "-"]
    assert adapter.thread_id == "thread-123"
    assert "thread-123" in (tmp_path / "implement.codex.jsonl").read_text(
        encoding="utf-8"
    )

    with pytest.raises(SparError, match="did not resume the sequential researcher thread"):
        adapter.run(
            phase="reflect",
            prompt="reflect on the result",
            cwd=candidate,
            schema=PROPOSAL_SCHEMA,
        )


def test_sequential_mcts_cli_requires_serial_session(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with temp_git_repo() as repo:
        initialize_runner_session(repo, max_candidates=2, max_parallel=3)

        exit_code = run_cli_main(
            [
                "demo",
                "--repo",
                str(repo),
                "--model",
                "gpt-test",
                "--effort",
                "medium",
                "--workflow",
                "sequential-mcts",
            ]
        )

    assert exit_code == 1
    assert "sequential-mcts workflow requires max_parallel = 1" in capsys.readouterr().err


def test_hybrid_spine_cli_requires_three_parallel_slots(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with temp_git_repo() as repo:
        initialize_runner_session(repo, max_candidates=2, max_parallel=2)

        exit_code = run_cli_main(
            [
                "demo",
                "--repo",
                str(repo),
                "--model",
                "gpt-test",
                "--effort",
                "medium",
                "--workflow",
                "hybrid-spine",
            ]
        )

    assert exit_code == 1
    assert "hybrid-spine workflow requires max_parallel = 3" in capsys.readouterr().err


def test_hybrid_spine_cli_constructs_context_topology(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    agents: list[Any] = []
    captured: dict[str, Any] = {}

    class StubAgent:
        def __init__(self, **kwargs: Any) -> None:
            self.preserve_context = kwargs["preserve_context"]
            agents.append(self)

    def fake_run_session(
        session_name: str, agent: Any, **kwargs: Any
    ) -> dict[str, Any]:
        captured.update(session_name=session_name, agent=agent, **kwargs)
        return {"status": "completed"}

    monkeypatch.setattr("spar.run_cli.CodexAgent", StubAgent)
    monkeypatch.setattr("spar.run_cli.run_session", fake_run_session)
    with temp_git_repo() as repo:
        initialize_runner_session(repo, max_candidates=2, max_parallel=3)
        exit_code = run_cli_main(
            [
                "demo",
                "--repo",
                str(repo),
                "--model",
                "gpt-test",
                "--effort",
                "medium",
                "--workflow",
                "hybrid-spine",
            ]
        )

    assert exit_code == 0
    assert [agent.preserve_context for agent in agents] == [False, True, True]
    assert captured["agent"] is agents[0]
    assert captured["proposal_agent"] is agents[1]
    assert captured["spine_agent"] is agents[2]
    assert captured["agent_metadata"]["workflow"] == "hybrid-spine"
    assert (
        captured["agent_metadata"]["researcher_context"]
        == "one persistent propose/implement/reflect spine thread; one persistent explorer "
        "proposal thread; fresh explorer implementation and reflection threads"
    )
    capsys.readouterr()


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
        self.prompts: dict[str, list[str]] = {phase: [] for phase in ("propose", "implement", "reflect")}
        self.cwds: dict[str, list[Path]] = {phase: [] for phase in self.prompts}

    def run(
        self,
        *,
        phase: str,
        prompt: str,
        cwd: Path,
        schema: dict[str, Any],
        event_path: Path | None = None,
    ) -> dict[str, Any]:
        del schema, event_path
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.prompts[phase].append(prompt)
            self.cwds[phase].append(cwd)
        try:
            if phase == "propose":
                return self._propose()
            value = int(re.search(r"set value to (\d+)", prompt).group(1))
            if phase == "implement":
                return self._implement(cwd, value)
            score = float(re.search(r'"score": (-?\d+(?:\.\d+)?)', prompt).group(1))
            decision = "discard" if int(score) in self.discard_values else "keep"
            return {
                "decision": decision,
                "summary": f"candidate measured {score}",
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
        request_profiling = value in self.profile_values
        return {
            "hypothesis": f"value {value} should improve the score",
            "instructions": f"set value to {value}",
            "rationale": "measure one deterministic change",
            "request_profiling": request_profiling,
            "profiling_reason": "confirm the observed value" if request_profiling else "",
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
            "observations": ["committed one file"],
            "limitations": [],
        }


def initialize_runner_session(
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
            "import os\n"
            "from pathlib import Path\n"
            "result = Path(os.environ['SPAR_PROFILING_RESULT'])\n"
            "value = Path('value.txt').read_text(encoding='utf-8').strip()\n"
            "result.write_text(json.dumps({'observed_value': value}) + '\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )
        paths.append("profile.py")
    run(["git", "add", *paths], repo)
    run(["git", "commit", "-m", "initial"], repo)
    with chdir(repo):
        commands.init("demo")
    session = repo / ".spar" / "demo"
    profile_config = ""
    if profiling:
        profile_config = textwrap.dedent(
            f"""

            [profiling]
            command = [{json.dumps(sys.executable)}, {json.dumps(str(repo / 'profile.py'))}]
            """
        )
    (session / "config.toml").write_text(
        textwrap.dedent(
            f"""
            max_candidates = {max_candidates}
            max_parallel = {max_parallel}

            [evaluation]
            command = [{json.dumps(sys.executable)}, {json.dumps(str(repo / 'evaluate.py'))}]
            """
        ).lstrip()
        + profile_config,
        encoding="utf-8",
    )
    (session / "objective.md").write_text(
        "# Objective\n\nMaximize the numeric value while testing one change per candidate.\n",
        encoding="utf-8",
    )
    return session


def assert_rejected_artifacts_survive(
    session: Path, workspace_root: Path, candidates: list[dict[str, Any]]
) -> None:
    for candidate in candidates:
        artifact_dir = session / "artifacts" / "candidates" / candidate["id"]
        assert [
            path.name
            for path in sorted(artifact_dir.glob("implementation-attempt-*.json"))
        ] == [
            "implementation-attempt-001.json",
            "implementation-attempt-002.json",
            "implementation-attempt-003.json",
        ]
        assert json_load(artifact_dir / "rejection.json")["reason"] == candidate["error"]
        assert not (artifact_dir / "implementation-commit.json").exists()
        assert not (artifact_dir / "evaluation-result.json").exists()
        assert (workspace_root / candidate["id"]).is_dir()


def assert_proposal_artifacts_survive(result: dict[str, Any]) -> None:
    run_dir = Path(result["artifact_path"]).parent
    assert all(
        (run_dir / attempt["path"]).is_file()
        for attempt in result["proposal_attempts"]
    )


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
