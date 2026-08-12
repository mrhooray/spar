import sqlite3
from pathlib import Path

import pytest

from spar.error import SparError
from spar.operation import candidate as candidate_ops
from spar.operation import session as session_ops
from spar.storage.db import SCHEMA_VERSION

from .helpers import (
    chdir,
    commit_file,
    temp_git_repo,
)


def test_init_creates_current_schema() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        with chdir(repo):
            result = session_ops.init("demo")
            status = session_ops.status("demo")
            top = session_ops.top("demo")
            inspected = candidate_ops.inspect("demo", "root")

        session = repo / ".spar" / "demo"
        with sqlite3.connect(session / "state.sqlite") as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'session'").fetchone()
            assert db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'spans'").fetchone()
            assert db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'invocations'").fetchone()
            candidate_columns = {row[1] for row in db.execute("PRAGMA table_xinfo(candidates)")}
            span_columns = {row[1] for row in db.execute("PRAGMA table_info(spans)")}
            invocation_columns = {row[1] for row in db.execute("PRAGMA table_info(invocations)")}
            assert {"evaluation_json", "profiling_json"} <= candidate_columns
            assert "eval_score" in candidate_columns
            assert "learnings" in candidate_columns
            assert "reflection_summary" not in candidate_columns
            assert "eval_summary" not in candidate_columns
            assert {"started_at", "completed_at"} <= candidate_columns
            assert not {"created_at", "updated_at"} & candidate_columns
            assert {"candidate_id", "kind", "started_at", "completed_at", "error"} <= span_columns
            assert {"span_id", "agent", "prompt", "response_json", "event_path"} <= invocation_columns
            assert "thread_id" not in invocation_columns
            assert not {"candidate_id", "kind", "started_at", "completed_at", "error"} & invocation_columns
        assert set(result) == {
            "session_name",
            "session_dir",
            "objective_path",
            "config_path",
        }
        assert Path(result["session_dir"]) == session.resolve()
        assert status["config_path"] == str(Path(result["config_path"]))
        assert "higher scores are better" in (session / "objective.md").read_text(encoding="utf-8")
        config_template = (session / "config.toml").read_text(encoding="utf-8")
        assert "where higher" in config_template
        assert "additional JSON fields" in config_template
        assert 'cli = "REPLACE_WITH_CLI"' in config_template
        assert 'model = "REPLACE_WITH_MODEL"' in config_template
        assert 'effort = "REPLACE_WITH_EFFORT"' in config_template
        assert 'command = ["REPLACE_WITH_EVALUATION_COMMAND"]' in config_template
        assert ".spar/" in (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8")
        assert status["max_candidates"] == 64
        assert status["candidates_used"] == 1
        assert status["session"]["status"] == "idle"
        assert status["candidates"][0]["id"] == "root"
        assert "best_candidate" not in status
        assert "active_candidates" not in status
        assert "top_candidates" not in status
        assert "timeline" not in status
        assert "agent" not in status
        assert top == {"session_name": "demo", "k": 3, "candidates": []}
        assert "started_at" in status["candidates"][0]
        assert "updated_at" not in status["candidates"][0]
        assert "instructions" not in status["candidates"][0]
        assert "worktree_path" not in status["candidates"][0]
        assert "spans" not in status
        assert "invocations" not in status
        assert inspected["spans"] == []
        assert Path(inspected["artifact_dir"]).resolve() == (session / "artifacts" / "candidates" / "root").resolve()
        assert inspected["environment"]["SPAR_PARENT_SHA"] == ""
        assert "SPAR_SESSION_DIR" not in inspected["environment"]
        assert "SPAR_PROFILING_RESULT" not in inspected["environment"]
        assert not (session / "artifacts" / "proposals").exists()


@pytest.mark.parametrize("session_name", ["a b", "foo.lock", "a..b"])
def test_init_rejects_session_names_that_are_invalid_in_git_refs(
    session_name: str,
) -> None:
    with temp_git_repo() as repo, chdir(repo):
        commit_file(repo, "value.txt", "0\n", "initial")

        with pytest.raises(SparError, match="not valid in a Git ref"):
            session_ops.init(session_name)

        assert not (repo / ".spar" / session_name).exists()


def test_init_allows_ignored_files_and_rejects_untracked_files() -> None:
    with temp_git_repo() as repo, chdir(repo):
        commit_file(repo, ".gitignore", ".cache/\n", "ignore cache")
        (repo / ".cache").mkdir()
        (repo / ".cache" / "artifact").write_text("ignored\n", encoding="utf-8")
        (repo / "notes.txt").write_text("untracked\n", encoding="utf-8")

        with pytest.raises(SparError, match="no untracked files"):
            session_ops.init("demo")

        (repo / "notes.txt").unlink()
        session_ops.init("demo")
