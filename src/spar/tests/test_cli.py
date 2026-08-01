from pathlib import Path
from typing import Any
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

from spar import commands
from spar.config import load_config
from spar.errors import SparError
from spar.state import (
    DECISION_DISCARD,
    DECISION_KEEP,
    STATUS_COMPLETED,
    STATUS_EVALUATING,
    STATUS_FINALIZING,
    STATUS_IMPLEMENTING,
    STATUS_INTERRUPTED,
    STATUS_REFLECTING,
    SessionState,
)


def test_init_creates_v10_state() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        with chdir(repo):
            result = commands.init("demo")
            status = commands.status("demo")
            parents = commands.parents("demo")
            inspected = commands.candidate_inspect("demo", "root")

        session = repo / ".spar" / "demo"
        import sqlite3

        with sqlite3.connect(session / "state.sqlite") as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 10
            assert db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'metadata'"
            ).fetchone() is None
            assert db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'operations'"
            ).fetchone()
        assert Path(result["session_dir"]) == session.resolve()
        assert status["config_path"] == str(Path(result["config_path"]))
        assert "higher for better results" in (session / "objective.md").read_text(encoding="utf-8")
        config_template = (session / "config.toml").read_text(encoding="utf-8")
        assert "where higher" in config_template
        assert "additional fields" in config_template
        assert 'command = ["/path/to/eval.sh"]' in config_template
        assert ".spar/" in (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8")
        assert status["candidate_budget"] == {"maximum": 64, "used": 1, "remaining": 63}
        assert status["active_candidates"][0]["id"] == "root"
        assert "suggested_parents" not in status
        assert parents == {"session_name": "demo", "k": 3, "parents": []}
        assert "instructions" not in status["candidates"][0]
        assert "workspace_path" not in status["candidates"][0]
        assert inspected["environment"]["SPAR_PARENT_SHA"] == ""
        assert Path(inspected["environment"]["SPAR_SESSION_DIR"]) == session.resolve()
        assert inspected["result_paths"]["evaluation_result"].endswith("root/evaluation-result.json")
        assert not (session / "artifacts" / "proposals").exists()
        assert not (session / "report.md").exists()


@pytest.mark.parametrize("session_name", ["a b", "foo.lock", "a..b"])
def test_init_rejects_session_names_that_are_invalid_in_git_refs(
    session_name: str,
) -> None:
    with temp_git_repo() as repo, chdir(repo):
        commit_file(repo, "value.txt", "0\n", "initial")

        with pytest.raises(SparError, match="not valid in a Git ref"):
            commands.init(session_name)

        assert not (repo / ".spar" / session_name).exists()


def test_init_allows_ignored_files_and_rejects_untracked_files() -> None:
    with temp_git_repo() as repo, chdir(repo):
        commit_file(repo, ".gitignore", ".cache/\n", "ignore cache")
        (repo / ".cache").mkdir()
        (repo / ".cache" / "artifact").write_text("ignored\n", encoding="utf-8")
        (repo / "notes.txt").write_text("untracked\n", encoding="utf-8")

        with pytest.raises(SparError, match="no untracked files"):
            commands.init("demo")

        (repo / "notes.txt").unlink()
        commands.init("demo")


@pytest.mark.parametrize(
    ("config", "missing"),
    [
        (
            'max_parallel = 2\n[evaluation]\ncommand = ["/path/to/eval.sh"]\n',
            "max_candidates",
        ),
        (
            'max_candidates = 10\n[evaluation]\ncommand = ["/path/to/eval.sh"]\n',
            "max_parallel",
        ),
        (
            "max_candidates = 10\nmax_parallel = 2\n",
            "evaluation.command",
        ),
        (
            "max_candidates = 10\nmax_parallel = 2\n[evaluation]\n",
            "evaluation.command",
        ),
    ],
)
def test_required_session_settings_have_no_fallbacks(
    tmp_path: Path, config: str, missing: str
) -> None:
    (tmp_path / "config.toml").write_text(config, encoding="utf-8")

    with pytest.raises(SparError, match=missing.replace(".", r"\.")):
        load_config(tmp_path)


@pytest.mark.parametrize("name", ["max_candidates", "max_parallel"])
def test_integer_settings_reject_booleans(tmp_path: Path, name: str) -> None:
    values = {"max_candidates": "10", "max_parallel": "2"}
    values[name] = "true"
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            f"""
            max_candidates = {values["max_candidates"]}
            max_parallel = {values["max_parallel"]}

            [evaluation]
            command = ["/path/to/eval.sh"]
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(SparError, match=f"{name} must be a positive integer"):
        load_config(tmp_path)


def test_profiling_must_be_a_table(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            """
            max_candidates = 10
            max_parallel = 2
            profiling = "bad"

            [evaluation]
            command = ["/path/to/eval.sh"]
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(SparError, match="profiling must be a table"):
        load_config(tmp_path)


def test_candidate_lifecycle_creates_driver_worktree_and_records_artifacts() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo)
        complete_root(repo, session, score=0.0)

        with chdir(repo):
            started = commands.candidate_start(
                "demo",
                parent_id="root",
                hypothesis="increase the value",
                instructions="set value.txt to 7",
                rationale="the baseline is zero",
            )
        candidate = started["candidate"]
        assert set(started) == {
            "candidate",
            "parent_commit",
            "timing",
            "objective_path",
        }
        worktree = Path(candidate["workspace_path"])
        assert worktree.exists()
        assert run(["git", "rev-parse", "HEAD"], worktree).stdout.strip() == started["parent_commit"]
        (worktree / "value.txt").write_text("7\n", encoding="utf-8")
        run(["git", "add", "value.txt"], worktree)
        run(["git", "commit", "-m", "candidate"], worktree)

        with chdir(repo):
            inspected = commands.candidate_inspect("demo", candidate["id"])
        evaluation_path = Path(inspected["result_paths"]["evaluation_result"])
        write_json(
            evaluation_path,
            {
                "score": 7,
                "cycles": 7,
                "commentary": {"result": "value increased"},
            },
        )
        _record_evaluation(repo, session, candidate["id"], json_load(evaluation_path))
        with chdir(repo):
            completed = commands.candidate_complete(
                "demo",
                candidate["id"],
                summary="value increased to seven",
                decision=DECISION_KEEP,
                decision_reason="focused improvement",
            )
            status = commands.status("demo")
            parents = commands.parents("demo")

        assert completed["candidate"]["eval_score"] == 7.0
        assert started["timing"]["operation"] == "worktree"
        assert started["timing"]["success"] is True
        assert completed["timing"]["operation"] == "completion"
        assert completed["timing"]["success"] is True
        assert status["best_candidate"]["id"] == candidate["id"]
        assert parents["parents"][0]["candidate_id"] == candidate["id"]
        assert run(["git", "branch", "--show-current"], repo).stdout.strip() == "master"
        assert (repo / "value.txt").read_text(encoding="utf-8") == "0\n"
        assert run(["git", "show-ref", "--verify", f"refs/spar/demo/{candidate['id']}"] , repo).stdout
        with chdir(worktree):
            assert commands.status("demo")["best_candidate"]["id"] == candidate["id"]
        artifact_dir = session / "artifacts" / "candidates" / candidate["id"]
        assert json_load(artifact_dir / "evaluation-result.json")["score"] == 7
        assert json_load(artifact_dir / "reflection-result.json")["summary"] == "value increased to seven"
        with chdir(repo):
            inspected = commands.candidate_inspect("demo", candidate["id"])
        assert [item["operation"] for item in inspected["operations"]] == ["worktree", "completion"]


def test_candidate_start_allows_retries_and_enforces_budget() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo, max_candidates=3)
        complete_root(repo, session, score=0.0)

        with chdir(repo):
            first = start_same_candidate()
            commands.candidate_fail("demo", first["candidate"]["id"], "test release", interrupted=True)
            second = start_same_candidate()
            with pytest.raises(SparError, match="candidate budget exhausted"):
                start_same_candidate()

        assert first["candidate"]["id"] != second["candidate"]["id"]
        assert first["candidate"]["hypothesis"] == second["candidate"]["hypothesis"]


def test_candidate_start_enforces_parallel_admission() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo, max_candidates=10)
        with chdir(repo):
            complete_root(repo, session, score=0.0)
            first = start_same_candidate()
            second = start_same_candidate()
            with pytest.raises(SparError, match="parallel candidate limit reached"):
                start_same_candidate()

        first_id = first["candidate"]["id"]
        with SessionState(session) as state:
            state.begin_evaluation(first_id, first["parent_commit"])

        with chdir(repo):
            third = start_same_candidate()
            status = commands.status("demo")
            parents = commands.parents("demo")

        assert first["candidate"]["workspace_path"] != second["candidate"]["workspace_path"]
        assert {candidate["id"] for candidate in status["active_candidates"]} == {
            second["candidate"]["id"],
            third["candidate"]["id"],
        }
        assert next(candidate for candidate in status["candidates"] if candidate["id"] == first_id)[
            "status"
        ] == STATUS_EVALUATING
        assert parents["parents"][0]["pending_rollouts"] == 3


def test_evaluator_runs_in_recorded_candidate_worktree() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        (repo / "eval.sh").write_text(
            "#!/bin/sh\nprintf '{\"score\": %s, \"cycles\": %s, \"notes\": [\"measured\"]}\\n' \"$(cat value.txt)\" \"$(cat value.txt)\"\n",
            encoding="utf-8",
        )
        (repo / "eval.sh").chmod(0o755)
        run(["git", "add", "eval.sh"], repo)
        run(["git", "commit", "-m", "evaluator"], repo)
        session = initialize_session(repo)
        complete_root(repo, session, score=0.0)
        with chdir(repo):
            started = start_same_candidate()
        worktree = Path(started["candidate"]["workspace_path"])
        (worktree / "value.txt").write_text("7\n", encoding="utf-8")
        run(["git", "add", "value.txt"], worktree)
        run(["git", "commit", "-m", "candidate"], worktree)

        with chdir(repo):
            evaluated = commands.candidate_evaluate("demo", started["candidate"]["id"])
            completed = commands.candidate_complete(
                "demo",
                started["candidate"]["id"],
                summary="evaluated candidate worktree",
                decision=DECISION_KEEP,
                decision_reason="candidate worktree was measured",
            )

        assert evaluated["evaluation"]["cycles"] == 7
        assert evaluated["evaluation"]["notes"] == ["measured"]
        assert evaluated["status"] == STATUS_REFLECTING
        assert evaluated["timing"]["operation"] == "evaluation"
        assert evaluated["timing"]["success"] is True
        assert completed["candidate"]["eval_score"] == 7
        assert (repo / "value.txt").read_text(encoding="utf-8") == "0\n"
        assert (session / "artifacts" / "candidates" / started["candidate"]["id"] / "evaluation.stdout").exists()


def test_evaluator_rejects_missing_score_immediately() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        (repo / "eval.sh").write_text("#!/bin/sh\nprintf '{\"cycles\": 1}\\n'\n", encoding="utf-8")
        (repo / "eval.sh").chmod(0o755)
        run(["git", "add", "eval.sh"], repo)
        run(["git", "commit", "-m", "evaluator"], repo)
        session = initialize_session(repo)

        with chdir(repo), pytest.raises(SparError, match="finite numeric score"):
            commands.candidate_evaluate("demo", "root")

        assert not (session / "artifacts" / "candidates" / "root" / "evaluation-result.json").exists()


def test_root_baseline_cannot_be_closed_as_failed() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        initialize_session(repo)

        with chdir(repo), pytest.raises(SparError, match="root candidate cannot be failed"):
            commands.candidate_fail("demo", "root", "baseline evaluator failed", interrupted=True)

        with chdir(repo):
            assert commands.candidate_inspect("demo", "root")["candidate"]["status"] == STATUS_IMPLEMENTING


def test_evaluator_allows_ignored_inputs_and_rejects_untracked_inputs() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, ".gitignore", ".cache/\n", "ignore cache")
        commit_file(repo, "value.txt", "0\n", "initial")
        (repo / "eval.sh").write_text("#!/bin/sh\nprintf '{\"score\": 1}\\n'\n", encoding="utf-8")
        (repo / "eval.sh").chmod(0o755)
        run(["git", "add", "eval.sh"], repo)
        run(["git", "commit", "-m", "evaluator"], repo)
        session = initialize_session(repo)
        complete_root(repo, session, score=0.0)
        with chdir(repo):
            started = start_same_candidate()

        worktree = Path(started["candidate"]["workspace_path"])
        (worktree / ".cache").mkdir()
        (worktree / ".cache" / "artifact").write_text("ignored\n", encoding="utf-8")
        with chdir(repo):
            commands.candidate_evaluate("demo", started["candidate"]["id"])

        with chdir(repo):
            untracked = start_same_candidate()
        worktree = Path(untracked["candidate"]["workspace_path"])
        (worktree / "untracked-fixture.txt").write_text("fixture\n", encoding="utf-8")
        with chdir(repo), pytest.raises(SparError, match="untracked files"):
            commands.candidate_evaluate("demo", untracked["candidate"]["id"])

        assert not (
            session / "artifacts" / "candidates" / untracked["candidate"]["id"] / "evaluation-result.json"
        ).exists()


def test_profiling_captures_opaque_artifacts_for_completed_candidate() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        (repo / "profile.sh").write_text(
            "#!/bin/sh\nprintf 'not json\\n'\nprintf 'trace bytes\\n' > \"$SPAR_PROFILING_DIR/trace.data\"\n",
            encoding="utf-8",
        )
        (repo / "profile.sh").chmod(0o755)
        run(["git", "add", "profile.sh"], repo)
        run(["git", "commit", "-m", "profiler"], repo)
        session = initialize_session(repo, profiling=True)
        complete_root(repo, session, score=0.0)
        with chdir(repo):
            result = commands.candidate_profile("demo", "root")

        artifact_dir = session / "artifacts" / "candidates" / "root"
        assert result["profiling_status"] == "completed"
        assert (artifact_dir / "profiling" / "trace.data").read_text(encoding="utf-8") == "trace bytes\n"
        assert (artifact_dir / "profiling.stdout").read_text(encoding="utf-8") == "not json\n"
        assert not (artifact_dir / "profiling-run.json").exists()


def test_profiling_rejects_a_later_worktree_commit() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        (repo / "eval.sh").write_text("#!/bin/sh\nprintf '{\"score\": 1}\\n'\n", encoding="utf-8")
        (repo / "eval.sh").chmod(0o755)
        (repo / "profile.sh").write_text("#!/bin/sh\nprintf 'trace\\n'\n", encoding="utf-8")
        (repo / "profile.sh").chmod(0o755)
        run(["git", "add", "eval.sh", "profile.sh"], repo)
        run(["git", "commit", "-m", "hooks"], repo)
        session = initialize_session(repo, profiling=True)
        complete_root(repo, session, score=0.0)

        with chdir(repo):
            started = start_same_candidate()
        worktree = Path(started["candidate"]["workspace_path"])
        (worktree / "value.txt").write_text("1\n", encoding="utf-8")
        run(["git", "add", "value.txt"], worktree)
        run(["git", "commit", "-m", "candidate"], worktree)
        with chdir(repo):
            commands.candidate_evaluate("demo", started["candidate"]["id"])

        (worktree / "untracked-profile-input.txt").write_text("input\n", encoding="utf-8")
        with chdir(repo), pytest.raises(SparError, match="untracked files"):
            commands.candidate_profile("demo", started["candidate"]["id"])
        (worktree / "untracked-profile-input.txt").unlink()

        (worktree / "value.txt").write_text("2\n", encoding="utf-8")
        run(["git", "add", "value.txt"], worktree)
        run(["git", "commit", "-m", "later candidate"], worktree)
        with chdir(repo), pytest.raises(SparError, match="does not match its recorded commit"):
            commands.candidate_profile("demo", started["candidate"]["id"])


def test_mcts_backpropagates_on_completion_and_accounts_for_active_work() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo, max_candidates=5)
        complete_root(repo, session, score=-100.0)

        kept = create_completed_child(repo, session, score=-10.0, decision=DECISION_KEEP, suffix="kept")
        with chdir(repo):
            before_discard = commands.parents("demo", k=3)
        discarded = create_completed_child(
            repo,
            session,
            score=1000.0,
            decision=DECISION_DISCARD,
            suffix="discarded",
            parent_id=kept,
        )
        with chdir(repo):
            active = commands.candidate_start(
                "demo",
                parent_id=kept,
                hypothesis="follow up",
                instructions="try another increment",
                rationale="kept parent is strongest",
            )
            parents = commands.parents("demo", k=3)
            repeated = commands.parents("demo", k=3)
            limited = commands.parents("demo", k=1)

        before = {item["candidate_id"]: item for item in before_discard["parents"]}
        suggestions = {item["candidate_id"]: item for item in parents["parents"]}
        assert repeated == parents
        assert discarded not in suggestions
        assert suggestions["root"]["visits"] == before["root"]["visits"] + 1
        assert suggestions["root"]["value_sum"] == before["root"]["value_sum"] + 1000.0
        assert suggestions[kept]["mean_value"] == 495.0
        assert suggestions[kept]["exploitation_value"] == 1.0
        assert suggestions[kept]["pending_rollouts"] == 1
        assert suggestions[kept]["exploration_bonus"] == pytest.approx(
            math.sqrt(2 * math.log(5) / 3)
        )
        assert len(limited["parents"]) == 1
        assert active["candidate"]["parent_id"] == kept

        with SessionState(session) as state:
            root = state.candidate("root")
            kept_candidate = state.candidate(kept)
            assert (root["mcts_visits"], root["mcts_value_sum"]) == (3, 890.0)
            assert (kept_candidate["mcts_visits"], kept_candidate["mcts_value_sum"]) == (2, 990.0)
            assert (
                state.candidate(discarded)["mcts_visits"],
                state.candidate(discarded)["mcts_value_sum"],
            ) == (1, 1000.0)
            with pytest.raises(SparError, match="candidate is not finalizing"):
                state.complete_candidate(discarded)
            root = state.candidate("root")
            assert (root["mcts_visits"], root["mcts_value_sum"]) == (3, 890.0)


def test_mcts_prefers_the_higher_mean_score_when_visits_are_equal() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo, max_candidates=3)
        complete_root(repo, session, score=0.0)
        lower = create_completed_child(repo, session, score=10.0, decision=DECISION_KEEP, suffix="lower")
        higher = create_completed_child(repo, session, score=20.0, decision=DECISION_KEEP, suffix="higher")

        with chdir(repo):
            ranked = commands.parents("demo", k=3)["parents"]

        suggestions = {item["candidate_id"]: item for item in ranked}
        assert suggestions[higher]["visits"] == suggestions[lower]["visits"]
        assert suggestions[higher]["mean_value"] > suggestions[lower]["mean_value"]
        assert suggestions[higher]["exploitation_value"] > suggestions[lower]["exploitation_value"]
        assert suggestions[higher]["priority"] > suggestions[lower]["priority"]


def test_evaluator_defined_fields_are_preserved_without_interpretation() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo)
        evaluation = session / "artifacts" / "candidates" / "root" / "evaluation-result.json"
        write_json(
            evaluation,
            {
                "score": 999,
                "correct": False,
                "commentary": "the evaluator assigns this result its own score",
            },
        )
        _record_evaluation(repo, session, "root", json_load(evaluation))
        with chdir(repo):
            result = commands.candidate_complete(
                "demo",
                "root",
                summary="evaluator result recorded",
                decision=DECISION_KEEP,
                decision_reason="evaluator score is authoritative",
            )
            parents = commands.parents("demo")

        assert result["candidate"]["decision"] == DECISION_KEEP
        assert result["candidate"]["eval_score"] == 999
        assert json_load(evaluation)["correct"] is False
        assert parents["parents"][0]["candidate_id"] == "root"


def test_profiling_result_and_failure_are_durable() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo, profiling=True)
        complete_root(repo, session, score=0.0, profiling={"summary": "baseline trace"})

        with chdir(repo):
            started = start_same_candidate()
            failed = commands.candidate_fail("demo", started["candidate"]["id"], "worker timed out", interrupted=True)
            inspected = commands.candidate_inspect("demo", started["candidate"]["id"])

        assert failed["candidate"]["status"] == STATUS_INTERRUPTED
        assert failed["timing"]["operation"] == "failure"
        assert failed["timing"]["success"] is True
        assert inspected["candidate"]["error"] == "worker timed out"
        assert {item["path"] for item in inspected["artifacts"]} == {"error.txt"}
        assert json_load(session / "artifacts" / "candidates" / "root" / "profiling-result.json") == {
            "summary": "baseline trace"
        }


def test_candidate_complete_leaves_finalizing_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo)
        evaluation = session / "artifacts" / "candidates" / "root" / "evaluation-result.json"
        write_json(evaluation, {"score": 0.0})
        _record_evaluation(repo, session, "root", json_load(evaluation))

        original_run_git = commands.run_git

        def fail_ref_update(cwd: Path, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if args and args[0] == "update-ref":
                raise SparError("simulated ref update failure")
            return original_run_git(cwd, args, **kwargs)

        with chdir(repo):
            monkeypatch.setattr(commands, "run_git", fail_ref_update)
            with pytest.raises(SparError, match="simulated ref update failure"):
                commands.candidate_complete(
                    "demo",
                    "root",
                    summary="baseline measured",
                    decision=DECISION_KEEP,
                    decision_reason="valid baseline",
                )
            assert commands.candidate_inspect("demo", "root")["candidate"]["status"] == STATUS_FINALIZING

        monkeypatch.setattr(commands, "run_git", original_run_git)
        with chdir(repo):
            completed = commands.candidate_complete(
                "demo",
                "root",
                summary="baseline measured",
                decision=DECISION_KEEP,
                decision_reason="valid baseline",
            )

        assert completed["candidate"]["status"] == STATUS_COMPLETED


def test_cli_outputs_structured_state() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        run_spar(["init", "demo"], repo)
        status = json.loads(run_spar(["status", "demo"], repo).stdout)
        parents = json.loads(run_spar(["parents", "demo", "--k", "1"], repo).stdout)

        assert status["session_name"] == "demo"
        assert status["evaluation"]["command"] == ["/path/to/eval.sh"]
        assert "suggested_parents" not in status
        assert parents["k"] == 1
        assert parents["parents"] == []
        assert parents["session_name"] == "demo"
        assert parents["cli_timing"]["command"] == "parents"


def test_cli_help_describes_candidate_lifecycle_arguments() -> None:
    cli_help = run_spar(["--help"], Path.cwd()).stdout
    parents_help = run_spar(["parents", "--help"], Path.cwd()).stdout
    start_help = run_spar(["candidate-start", "--help"], Path.cwd()).stdout
    complete_help = run_spar(["candidate-complete", "--help"], Path.cwd()).stdout
    complete_help = " ".join(complete_help.split())

    assert "record a decision and update MCTS" in cli_help
    assert "--k COUNT" in parents_help
    assert "SESSION" in start_help
    assert "completed, kept parent ID" in start_help
    assert "one intervention to implement" in start_help
    assert "reflecting or finalizing candidate ID" in complete_help
    assert "eligible as a future parent" in complete_help


def test_cli_repo_option_sets_repository_context() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        run_spar(["--repo", str(repo), "init", "demo"], repo.parent)
        status = json.loads(run_spar(["--repo", str(repo), "status", "demo"], repo.parent).stdout)

        assert Path(status["repository"]) == repo.resolve()


def test_legacy_state_is_rejected_with_clear_message() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = repo / ".spar" / "legacy"
        session.mkdir(parents=True)
        (session / "config.toml").write_text(
            "max_candidates = 64\n"
            "max_parallel = 4\n"
            "[evaluation]\n"
            'command = ["/path/to/eval.sh"]\n',
            encoding="utf-8",
        )
        import sqlite3

        db = sqlite3.connect(session / "state.sqlite")
        db.execute("CREATE TABLE candidates (id TEXT PRIMARY KEY)")
        db.commit()
        db.close()

        result = run_spar_result(["status", "legacy"], repo)
        assert result.returncode == 1
        assert "session schema is incompatible" in result.stderr


def test_invalid_config_is_reported_without_a_traceback() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        run_spar(["init", "demo"], repo)
        (repo / ".spar" / "demo" / "config.toml").write_text("[evaluation\n", encoding="utf-8")

        result = run_spar_result(["status", "demo"], repo)

        assert result.returncode == 1
        assert "configuration is invalid TOML" in result.stderr
        assert "Traceback" not in result.stderr


def initialize_session(repo: Path, *, max_candidates: int = 10, profiling: bool = False) -> Path:
    with chdir(repo):
        commands.init("demo")
    session = repo / ".spar" / "demo"
    profiling_config = ""
    if profiling:
        profiling_config = textwrap.dedent(
            """

            [profiling]
            command = ["./profile.sh"]
            """
        )
    (session / "config.toml").write_text(
        textwrap.dedent(
            f"""
            max_candidates = {max_candidates}
            max_parallel = 2

            [evaluation]
            command = [{json.dumps(str((repo / "eval.sh").resolve()))}]
            """
        ).lstrip()
        + profiling_config,
        encoding="utf-8",
    )
    return session


def complete_root(
    repo: Path,
    session: Path,
    *,
    score: float,
    profiling: dict[str, Any] | None = None,
) -> None:
    artifact_dir = session / "artifacts" / "candidates" / "root"
    evaluation = artifact_dir / "evaluation-result.json"
    write_json(
        evaluation,
        {
            "score": score,
            "value": score,
        },
    )
    if profiling is not None:
        write_json(artifact_dir / "profiling-result.json", profiling)
    _record_evaluation(repo, session, "root", json_load(evaluation))
    with chdir(repo):
        commands.candidate_complete(
            "demo",
            "root",
            summary="baseline measured",
            decision=DECISION_KEEP,
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
        started = commands.candidate_start(
            "demo",
            parent_id=parent_id,
            hypothesis=f"candidate {suffix}",
            instructions=f"write {suffix}",
            rationale="test MCTS",
        )
    candidate_id = started["candidate"]["id"]
    worktree = Path(started["candidate"]["workspace_path"])
    (worktree / "value.txt").write_text(f"{suffix}\n", encoding="utf-8")
    run(["git", "add", "value.txt"], worktree)
    run(["git", "commit", "-m", suffix], worktree)
    evaluation = session / "artifacts" / "candidates" / candidate_id / "evaluation-result.json"
    write_json(
        evaluation,
        {
            "score": score,
        },
    )
    _record_evaluation(repo, session, candidate_id, json_load(evaluation))
    with chdir(repo):
        commands.candidate_complete(
            "demo",
            candidate_id,
            summary=f"{suffix} measured",
            decision=decision,
            decision_reason=f"{decision} by researcher",
        )
    return candidate_id


def start_same_candidate() -> dict[str, Any]:
    return commands.candidate_start(
        "demo",
        parent_id="root",
        hypothesis="same idea",
        instructions="try an implementation",
        rationale="compare alternate implementations",
    )


def _record_evaluation(repo: Path, session: Path, candidate_id: str, evaluation: dict[str, Any]) -> None:
    with SessionState(session) as state:
        candidate = state.candidate(candidate_id)
        commit_sha = run(["git", "rev-parse", "HEAD"], Path(candidate["workspace_path"])).stdout.strip()
        state.begin_evaluation(candidate_id, commit_sha)
        state.record_evaluation(candidate_id, evaluation)


def commit_file(repo: Path, name: str, text: str, message: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    run(["git", "add", name], repo)
    run(["git", "commit", "-m", message], repo)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        [sys.executable, "-m", "spar", *command],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def run_spar_result(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "spar", *command],
        cwd=cwd,
        text=True,
        capture_output=True,
    )
