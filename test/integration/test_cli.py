import json
import sqlite3
from pathlib import Path

from spar.operation import session as session_ops

from .helpers import (
    chdir,
    commit_file,
    run_spar,
    run_spar_result,
    temp_git_repo,
)


def test_cli_outputs_structured_state() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        run_spar(["init", "demo"], repo)
        status = json.loads(run_spar(["status", "demo"], repo).stdout)
        top = json.loads(run_spar(["top", "demo", "--k", "1"], repo).stdout)

        assert status["session_name"] == "demo"
        assert status["evaluation"]["command"] == ["REPLACE_WITH_EVALUATION_COMMAND"]
        assert "top_candidates" not in status
        assert top["k"] == 1
        assert top["candidates"] == []
        assert top["session_name"] == "demo"
        assert top["cli_timing"]["command"] == "top"


def test_cli_init_prints_setup_instructions() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        output = run_spar(["init", "demo"], repo).stdout

        assert output == (
            "Initialized session demo.\n\n"
            "Edit:\n"
            f"  objective: {(repo / '.spar' / 'demo' / 'objective.md').resolve()}\n"
            f"  config:    {(repo / '.spar' / 'demo' / 'config.toml').resolve()}\n\n"
            "Then run:\n"
            "  spar start demo\n"
        )


def test_cli_stop_requests_a_running_session() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        run_spar(["init", "demo"], repo)
        with chdir(repo):
            session_ops.start("demo")

        stopped = json.loads(run_spar(["stop", "demo"], repo).stdout)

        assert stopped["session"]["stop_requested"] is True


def test_cli_help_describes_public_commands() -> None:
    cli_help = run_spar(["--help"], Path.cwd()).stdout
    inspect_help = run_spar(["inspect", "--help"], Path.cwd()).stdout
    top_help = run_spar(["top", "--help"], Path.cwd()).stdout
    start_help = run_spar(["start", "--help"], Path.cwd()).stdout

    assert "start" in cli_help
    assert "inspect" in cli_help
    assert "candidate state, lineage, recorded operations" in inspect_help
    assert "--k COUNT" in top_help
    assert "SESSION" in start_help
    assert "--agent" not in start_help
    assert "--model" not in start_help
    assert "--effort" not in start_help


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
            "[agent]\n"
            'cli = "codex"\n'
            'model = "test-model"\n'
            'effort = "high"\n'
            "[evaluation]\n"
            'command = ["/path/to/eval.sh"]\n',
            encoding="utf-8",
        )
        with sqlite3.connect(session / "state.sqlite") as db:
            db.execute("CREATE TABLE candidates (id TEXT PRIMARY KEY)")

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
