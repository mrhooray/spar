import textwrap
from pathlib import Path

import pytest

from spar.config import load_config
from spar.error import SparError


@pytest.mark.parametrize(
    ("config", "missing"),
    [
        ("", "agent"),
        ("[agent]\n", "agent.cli"),
        ('[agent]\ncli = "codex"\n', "agent.model"),
        ('[agent]\ncli = "codex"\nmodel = "test-model"\n', "agent.effort"),
    ],
)
def test_agent_settings_are_required(tmp_path: Path, config: str, missing: str) -> None:
    (tmp_path / "config.toml").write_text(config, encoding="utf-8")

    with pytest.raises(SparError, match=missing.replace(".", r"\.")):
        load_config(tmp_path)


def test_defaulted_session_settings_can_be_omitted(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[agent]\ncli = "codex"\nmodel = "test-model"\neffort = "high"\n[evaluation]\ncommand = ["./eval.sh"]\n',
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config["max_candidates"] == 64
    assert config["max_parallel"] == 4
    assert config["mcts"]["exploration_constant"] == pytest.approx(2**0.5)
    assert config["agent"] == {
        "cli": "codex",
        "model": "test-model",
        "effort": "high",
        "timeout_seconds": 3600,
    }
    assert config["evaluation"]["timeout_seconds"] == 3600
    assert config["profiling"]["timeout_seconds"] == 3600


def test_external_operation_timeouts_are_configurable(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            """
            [agent]
            cli = "codex"
            model = "test-model"
            effort = "high"
            timeout_seconds = 10

            [evaluation]
            command = ["./eval.sh"]
            timeout_seconds = 20

            [profiling]
            command = ["./profile.sh"]
            timeout_seconds = 30
            """
        ).lstrip(),
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config["agent"]["timeout_seconds"] == 10
    assert config["evaluation"]["timeout_seconds"] == 20
    assert config["profiling"]["timeout_seconds"] == 30


@pytest.mark.parametrize("name", ["max_candidates", "max_parallel"])
def test_integer_settings_reject_booleans(tmp_path: Path, name: str) -> None:
    values = {"max_candidates": "10", "max_parallel": "2"}
    values[name] = "true"
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            f"""
            max_candidates = {values["max_candidates"]}
            max_parallel = {values["max_parallel"]}

            [agent]
            cli = "codex"
            model = "test-model"
            effort = "high"

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

            [agent]
            cli = "codex"
            model = "test-model"
            effort = "high"

            [evaluation]
            command = ["/path/to/eval.sh"]
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(SparError, match="profiling must be a table"):
        load_config(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mcts.exploration_constant", "0", "positive finite number"),
        ("mcts.exploration_constant", '"bad"', "positive finite number"),
        ("agent.timeout_seconds", "0", "agent.timeout_seconds"),
        ("evaluation.timeout_seconds", "false", "evaluation.timeout_seconds"),
        ("evaluation.command", "[]", "evaluation.command"),
    ],
)
def test_invalid_runtime_settings_are_rejected(tmp_path: Path, field: str, value: str, message: str) -> None:
    agent_timeout = value if field == "agent.timeout_seconds" else "10"
    evaluation_timeout = value if field == "evaluation.timeout_seconds" else "10"
    evaluation_command = value if field == "evaluation.command" else '["/path/to/eval.sh"]'
    exploration_constant = value if field == "mcts.exploration_constant" else "1.0"
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            f"""
            [agent]
            cli = "codex"
            model = "test-model"
            effort = "high"
            timeout_seconds = {agent_timeout}

            [evaluation]
            command = {evaluation_command}
            timeout_seconds = {evaluation_timeout}

            [mcts]
            exploration_constant = {exploration_constant}
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(SparError, match=message):
        load_config(tmp_path)
