import json
from pathlib import Path

import pytest

from spar.error import SparError
from spar.lifecycle import CandidateStatus, Decision
from spar.operation import candidate as candidate_ops
from spar.operation import session as session_ops
from spar.storage.db import DB

from .helpers import (
    _record_evaluation,
    chdir,
    commit_file,
    complete_root,
    initialize_session,
    run,
    start_same_candidate,
    temp_git_repo,
)


def test_candidate_lifecycle_creates_worktree_and_records_state() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo)
        complete_root(repo, session, score=0.0)

        with chdir(repo):
            started = candidate_ops.start(
                "demo",
                parent_id="root",
                hypothesis="increase the value",
                instructions="set value.txt to 7",
                rationale="the baseline is zero",
                profiling_question="does the value change at runtime?",
            )
        candidate = started["candidate"]
        assert candidate["profiling_question"] == "does the value change at runtime?"
        assert set(started) == {
            "candidate",
            "parent_commit",
            "span",
        }
        worktree = Path(candidate["worktree_path"])
        assert worktree.exists()
        assert run(["git", "rev-parse", "HEAD"], worktree).stdout.strip() == started["parent_commit"]
        (worktree / "value.txt").write_text("7\n", encoding="utf-8")
        run(["git", "add", "value.txt"], worktree)
        run(["git", "commit", "-m", "candidate"], worktree)

        _record_evaluation(
            repo,
            session,
            candidate["id"],
            {
                "score": 7,
                "cycles": 7,
                "commentary": {"result": "value increased"},
            },
        )
        with chdir(repo):
            completed = candidate_ops.complete(
                "demo",
                candidate["id"],
                learnings="value increased to seven",
                decision=Decision.KEEP,
                decision_reason="focused improvement",
            )
            top = session_ops.top("demo")

        assert completed["candidate"]["eval_score"] == 7.0
        assert started["span"]["kind"] == "worktree"
        assert started["span"]["success"] is True
        assert completed["span"]["kind"] == "finalization"
        assert completed["span"]["success"] is True
        assert top["candidates"][0]["candidate_id"] == candidate["id"]
        assert run(["git", "branch", "--show-current"], repo).stdout.strip() == "master"
        assert (repo / "value.txt").read_text(encoding="utf-8") == "0\n"
        assert run(["git", "show-ref", "--verify", f"refs/spar/demo/{candidate['id']}"], repo).stdout
        with chdir(worktree):
            assert any(item["id"] == candidate["id"] for item in session_ops.status("demo")["candidates"])
        artifact_dir = session / "artifacts" / "candidates" / candidate["id"]
        with DB(session) as db:
            stored = db.candidate(candidate["id"])
            assert json.loads(stored["evaluation_json"]) == {
                "commentary": {"result": "value increased"},
                "cycles": 7,
                "score": 7,
            }
            assert stored["learnings"] == "value increased to seven"
            listed = next(row for row in db.candidates() if row["id"] == candidate["id"])
            assert listed["evaluation_json"] == stored["evaluation_json"]
        assert not list(artifact_dir.iterdir())
        with chdir(repo):
            inspected = candidate_ops.inspect("demo", candidate["id"])
        assert [item["kind"] for item in inspected["spans"]] == [
            "worktree",
            "evaluation",
            "finalization",
        ]


def test_start_allows_retries_and_enforces_max_candidates() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo, max_candidates=3)
        complete_root(repo, session, score=0.0)

        with chdir(repo):
            first = start_same_candidate()
            candidate_ops.fail("demo", first["candidate"]["id"], "test release", interrupted=True)
            second = start_same_candidate()
            with pytest.raises(SparError, match="maximum candidates reached"):
                start_same_candidate()

        assert first["candidate"]["id"] != second["candidate"]["id"]
        assert first["candidate"]["hypothesis"] == second["candidate"]["hypothesis"]


def test_start_creates_independent_active_candidates() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo, max_candidates=10)
        with chdir(repo):
            complete_root(repo, session, score=0.0)
            first = start_same_candidate()
            second = start_same_candidate()
            third = start_same_candidate()
            status = session_ops.status("demo")
            top = session_ops.top("demo")

        assert first["candidate"]["worktree_path"] != second["candidate"]["worktree_path"]
        assert {
            candidate["id"] for candidate in status["candidates"] if candidate["status"] == CandidateStatus.IMPLEMENTING
        } == {
            first["candidate"]["id"],
            second["candidate"]["id"],
            third["candidate"]["id"],
        }
        assert top["candidates"][0]["pending_rollouts"] == 3


def test_evaluator_runs_in_recorded_candidate_worktree() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        (repo / "eval.sh").write_text(
            '#!/bin/sh\nprintf \'{"score": %s, "cycles": %s, "notes": ["measured"]}\\n\' "$(cat value.txt)" "$(cat value.txt)"\n',
            encoding="utf-8",
        )
        (repo / "eval.sh").chmod(0o755)
        run(["git", "add", "eval.sh"], repo)
        run(["git", "commit", "-m", "evaluator"], repo)
        session = initialize_session(repo)
        complete_root(repo, session, score=0.0)
        with chdir(repo):
            started = start_same_candidate()
        worktree = Path(started["candidate"]["worktree_path"])
        (worktree / "value.txt").write_text("7\n", encoding="utf-8")
        run(["git", "add", "value.txt"], worktree)
        run(["git", "commit", "-m", "candidate"], worktree)

        with chdir(repo):
            evaluated = candidate_ops.evaluate("demo", started["candidate"]["id"])
            completed = candidate_ops.complete(
                "demo",
                started["candidate"]["id"],
                learnings="evaluated candidate worktree",
                decision=Decision.KEEP,
                decision_reason="candidate worktree was measured",
            )

        assert evaluated["evaluation"]["cycles"] == 7
        assert evaluated["evaluation"]["notes"] == ["measured"]
        assert evaluated["span"]["kind"] == "evaluation"
        assert evaluated["span"]["success"] is True
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
            candidate_ops.evaluate("demo", "root")

        with DB(session) as db:
            assert db.candidate("root")["status"] == CandidateStatus.EVALUATING
            failed_span = db.spans("root")[-1]
        assert failed_span["kind"] == "evaluation"
        assert failed_span["success"] is False


def test_evaluator_command_failure_is_recorded() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        (repo / "eval.sh").write_text("#!/bin/sh\nprintf 'failed\\n'\nexit 7\n", encoding="utf-8")
        (repo / "eval.sh").chmod(0o755)
        run(["git", "add", "eval.sh"], repo)
        run(["git", "commit", "-m", "evaluator"], repo)
        session = initialize_session(repo)

        with chdir(repo), pytest.raises(SparError, match="exit code 7"):
            candidate_ops.evaluate("demo", "root")

        with DB(session) as db:
            candidate = db.candidate("root")
            failed_span = db.spans("root")[-1]
        assert candidate["status"] == CandidateStatus.EVALUATING
        assert failed_span["kind"] == "evaluation"
        assert failed_span["success"] is False
        assert (session / "artifacts" / "candidates" / "root" / "evaluation.stdout").read_text(
            encoding="utf-8"
        ) == "failed\n"


def test_root_baseline_cannot_be_closed_as_failed() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        initialize_session(repo)

        with chdir(repo), pytest.raises(SparError, match="root candidate cannot be failed"):
            candidate_ops.fail("demo", "root", "baseline evaluator failed", interrupted=True)

        with chdir(repo):
            assert candidate_ops.inspect("demo", "root")["candidate"]["status"] == CandidateStatus.IMPLEMENTING


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

        worktree = Path(started["candidate"]["worktree_path"])
        (worktree / ".cache").mkdir()
        (worktree / ".cache" / "artifact").write_text("ignored\n", encoding="utf-8")
        with chdir(repo):
            candidate_ops.evaluate("demo", started["candidate"]["id"])

        with chdir(repo):
            untracked = start_same_candidate()
        worktree = Path(untracked["candidate"]["worktree_path"])
        (worktree / "untracked-fixture.txt").write_text("fixture\n", encoding="utf-8")
        with chdir(repo), pytest.raises(SparError, match="untracked files"):
            candidate_ops.evaluate("demo", untracked["candidate"]["id"])

        with DB(session) as db:
            candidate = db.candidate(untracked["candidate"]["id"])
            assert candidate["status"] == CandidateStatus.IMPLEMENTING
            assert not any(span["kind"] == "evaluation" for span in db.spans(untracked["candidate"]["id"]))


def test_profiling_captures_structured_result_and_artifacts() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        (repo / "profile.sh").write_text(
            '#!/bin/sh\nprintf \'{"summary": "measured"}\\n\'\n'
            "printf 'trace bytes\\n' > \"$SPAR_PROFILING_DIR/trace.data\"\n",
            encoding="utf-8",
        )
        (repo / "profile.sh").chmod(0o755)
        run(["git", "add", "profile.sh"], repo)
        run(["git", "commit", "-m", "profiler"], repo)
        session = initialize_session(repo, profiling=True)
        complete_root(repo, session, score=0.0)
        with chdir(repo):
            result = candidate_ops.profile("demo", "root")

        artifact_dir = session / "artifacts" / "candidates" / "root"
        assert result["profile"] == {"summary": "measured"}
        assert (artifact_dir / "profiling" / "trace.data").read_text(encoding="utf-8") == "trace bytes\n"
        assert (artifact_dir / "profiling.stdout").read_text(encoding="utf-8") == ('{"summary": "measured"}\n')
        with DB(session) as db:
            assert json.loads(db.candidate("root")["profiling_json"]) == {"summary": "measured"}


def test_profile_without_a_configured_command_returns_no_profile() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo)
        complete_root(repo, session, score=0.0)

        with chdir(repo):
            result = candidate_ops.profile("demo", "root")

        assert result["profile"] is None
        assert result["artifacts"] == []


def test_profile_command_failure_is_recorded() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        (repo / "profile.sh").write_text(
            "#!/bin/sh\nprintf 'profile failed\\n'\nexit 9\n",
            encoding="utf-8",
        )
        (repo / "profile.sh").chmod(0o755)
        run(["git", "add", "profile.sh"], repo)
        run(["git", "commit", "-m", "profiler"], repo)
        session = initialize_session(repo, profiling=True)
        complete_root(repo, session, score=0.0)

        with chdir(repo), pytest.raises(SparError, match="exit code 9"):
            candidate_ops.profile("demo", "root")

        with DB(session) as db:
            failed_span = db.spans("root")[-1]
        assert failed_span["kind"] == "profiling"
        assert failed_span["success"] is False
        assert (session / "artifacts" / "candidates" / "root" / "profiling.stdout").read_text(
            encoding="utf-8"
        ) == "profile failed\n"


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
        worktree = Path(started["candidate"]["worktree_path"])
        (worktree / "value.txt").write_text("1\n", encoding="utf-8")
        run(["git", "add", "value.txt"], worktree)
        run(["git", "commit", "-m", "candidate"], worktree)
        with chdir(repo):
            candidate_ops.evaluate("demo", started["candidate"]["id"])

        (worktree / "untracked-profile-input.txt").write_text("input\n", encoding="utf-8")
        with chdir(repo), pytest.raises(SparError, match="untracked files"):
            candidate_ops.profile("demo", started["candidate"]["id"])
        (worktree / "untracked-profile-input.txt").unlink()

        (worktree / "value.txt").write_text("2\n", encoding="utf-8")
        run(["git", "add", "value.txt"], worktree)
        run(["git", "commit", "-m", "later candidate"], worktree)
        with chdir(repo), pytest.raises(SparError, match="does not match its recorded commit"):
            candidate_ops.profile("demo", started["candidate"]["id"])


def test_evaluator_defined_fields_are_preserved_without_interpretation() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo)
        evaluation = {
            "score": 999,
            "correct": False,
            "commentary": "the evaluator assigns this result its own score",
        }
        _record_evaluation(repo, session, "root", evaluation)
        with chdir(repo):
            result = candidate_ops.complete(
                "demo",
                "root",
                learnings="evaluator result recorded",
                decision=Decision.KEEP,
                decision_reason="evaluator score is authoritative",
            )
            top = session_ops.top("demo")

        assert result["candidate"]["decision"] == Decision.KEEP
        assert result["candidate"]["eval_score"] == 999
        with DB(session) as db:
            assert json.loads(db.candidate("root")["evaluation_json"]) == evaluation
        assert top["candidates"][0]["candidate_id"] == "root"


def test_interrupted_candidate_failure_is_durable() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo)
        complete_root(repo, session, score=0.0)

        with chdir(repo):
            started = start_same_candidate()
            failed = candidate_ops.fail("demo", started["candidate"]["id"], "worker timed out", interrupted=True)
            inspected = candidate_ops.inspect("demo", started["candidate"]["id"])

        assert failed["candidate"]["status"] == CandidateStatus.INTERRUPTED
        assert set(failed) == {"candidate"}
        assert inspected["candidate"]["error"] == "worker timed out"
        assert inspected["artifacts"] == []


def test_complete_leaves_candidate_reflecting_when_ref_update_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo)
        _record_evaluation(repo, session, "root", {"score": 0.0})

        original_update_ref = candidate_ops.git.update_ref

        def fail_ref_update(repo: Path, ref: str, commit: str) -> None:
            del repo, ref, commit
            raise SparError("simulated ref update failure")

        with chdir(repo):
            monkeypatch.setattr(candidate_ops.git, "update_ref", fail_ref_update)
            with pytest.raises(SparError, match="simulated ref update failure"):
                candidate_ops.complete(
                    "demo",
                    "root",
                    learnings="baseline measured",
                    decision=Decision.KEEP,
                    decision_reason="valid baseline",
                )
            assert candidate_ops.inspect("demo", "root")["candidate"]["status"] == CandidateStatus.REFLECTING

        monkeypatch.setattr(candidate_ops.git, "update_ref", original_update_ref)
        with chdir(repo):
            completed = candidate_ops.complete(
                "demo",
                "root",
                learnings="baseline measured",
                decision=Decision.KEEP,
                decision_reason="valid baseline",
            )

        assert completed["candidate"]["status"] == CandidateStatus.COMPLETED
