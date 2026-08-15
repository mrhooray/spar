import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spar.agent.adapter import (
    ClaudeCodeAgent,
    CodexAgent,
    OpenCodeAgent,
    PiAgent,
    _json_events,
    _opencode_response,
    _pi_response,
    create_agent,
)
from spar.agent.invocation import (
    IMPLEMENTATION_SCHEMA,
    PROPOSAL_SCHEMA,
    REFLECTION_SCHEMA,
)
from spar.error import SparError


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                "cli": "REPLACE_WITH_CLI",
                "model": "test-model",
                "effort": "high",
                "timeout_seconds": 30,
            },
            "agent.cli must be configured",
        ),
        (
            {
                "cli": "codex",
                "model": "REPLACE_WITH_MODEL",
                "effort": "high",
                "timeout_seconds": 30,
            },
            "agent.model must be configured",
        ),
        (
            {
                "cli": "codex",
                "model": "test-model",
                "effort": "REPLACE_WITH_EFFORT",
                "timeout_seconds": 30,
            },
            "agent.effort must be configured",
        ),
    ],
)
def test_agent_factory_rejects_unconfigured_placeholders(config: dict[str, Any], message: str) -> None:
    with pytest.raises(SparError, match=message.replace(".", r"\.")):
        create_agent(config)


def test_agent_factory_rejects_unknown_cli() -> None:
    with pytest.raises(SparError, match="agent.cli must be configured"):
        create_agent(
            {
                "cli": "other",
                "model": "test-model",
                "effort": "high",
                "timeout_seconds": 30,
            }
        )


@pytest.mark.parametrize(
    ("cli", "agent_type"),
    [
        ("claude-code", ClaudeCodeAgent),
        ("codex", CodexAgent),
        ("opencode", OpenCodeAgent),
        ("pi", PiAgent),
    ],
)
def test_agent_factory_supports_additional_coding_harnesses(
    cli: str, agent_type: type[PiAgent | ClaudeCodeAgent]
) -> None:
    agent = create_agent(
        {
            "cli": cli,
            "model": "test-model",
            "effort": "high",
            "timeout_seconds": 30,
        }
    )

    assert isinstance(agent, agent_type)


def test_codex_adapter_uses_explicit_model_effort_and_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["input"] = kwargs["input"]
        captured["kwargs"] = kwargs
        response_path = Path(command[command.index("--output-last-message") + 1])
        response_path.write_text(
            json.dumps(
                {
                    "hypothesis": "test",
                    "instructions": "change one thing",
                    "rationale": "measure it",
                    "profiling_question": None,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            0,
            '{"type":"thread.started","thread_id":"thread-1"}\n{"type":"audit-event","id":"event-1"}\n',
            "",
        )

    monkeypatch.setattr("spar.agent.adapter.subprocess.run", fake_run)
    adapter = CodexAgent(model="gpt-test", effort="high", timeout_seconds=30)

    response = adapter.invoke(
        kind="proposal",
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
        configs = {command[index + 1] for index, part in enumerate(command[:-1]) if part == "--config"}
        assert command[command.index("--sandbox") + 1] == sandbox
        assert single_agent_configs <= configs

    assert response["hypothesis"] == "test"
    assert command[command.index("--model") + 1] == "gpt-test"
    assert command[command.index("--config") + 1] == "model_reasoning_effort=high"
    assert_phase_command(command, "read-only")
    assert "--output-schema" in command
    assert "--json" in command
    assert "--ephemeral" not in command
    assert "--cd" not in command
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["input"] == "bounded prompt"
    assert adapter.session_id == "thread-1"

    adapter.invoke(
        kind="implementation",
        prompt="task-local prompt",
        cwd=tmp_path,
        schema=IMPLEMENTATION_SCHEMA,
        event_path=tmp_path / "implement.codex.jsonl",
    )
    assert_phase_command(captured["command"], "workspace-write")

    adapter.invoke(
        kind="reflection",
        prompt="evidence prompt",
        cwd=tmp_path,
        schema=REFLECTION_SCHEMA,
        event_path=tmp_path / "reflect.codex.jsonl",
    )
    command = captured["command"]
    assert_phase_command(command, "read-only")
    assert command[command.index("exec") : command.index("exec") + 3] == ["exec", "resume", "thread-1"]


def test_opencode_adapter_uses_json_events_and_returns_the_requested_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "type": "text",
                    "sessionID": "session-1",
                    "part": {
                        "type": "text",
                        "text": '{"hypothesis":"test","instructions":"change one thing",'
                        '"rationale":"measure it","profiling_question":null}',
                    },
                }
            )
            + "\n",
            "",
        )

    monkeypatch.setattr("spar.agent.adapter.subprocess.run", fake_run)
    adapter = OpenCodeAgent(model="openai/gpt-test", effort="high", timeout_seconds=30)
    event_path = tmp_path / "propose.events.jsonl"

    response = adapter.invoke(
        kind="proposal",
        prompt="bounded prompt",
        cwd=tmp_path,
        schema=PROPOSAL_SCHEMA,
        event_path=event_path,
    )

    command = captured["command"]
    assert response["hypothesis"] == "test"
    assert command[:2] == ["opencode", "run"]
    assert command[command.index("--format") + 1] == "json"
    assert command[command.index("--model") + 1] == "openai/gpt-test"
    assert command[command.index("--variant") + 1] == "high"
    assert command[command.index("--dir") + 1] == str(tmp_path)
    assert "--auto" in command
    assert "bounded prompt" in command[-1]
    assert json.dumps(PROPOSAL_SCHEMA, indent=2, sort_keys=True) in command[-1]
    assert captured["kwargs"]["timeout"] == 30
    assert event_path.is_file()
    assert adapter.session_id == "session-1"

    adapter.invoke(
        kind="proposal",
        prompt="next prompt",
        cwd=tmp_path,
        schema=PROPOSAL_SCHEMA,
    )
    assert captured["command"][:4] == ["opencode", "run", "--session", "session-1"]

    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    adapter.invoke(
        kind="proposal",
        prompt="another prompt",
        cwd=other_cwd,
        schema=PROPOSAL_SCHEMA,
    )
    assert "--session" not in captured["command"]

    resumed = OpenCodeAgent(model="openai/gpt-test", effort="high", timeout_seconds=30, session_id="stored")
    resumed.invoke(
        kind="proposal",
        prompt="resumed prompt",
        cwd=tmp_path,
        schema=PROPOSAL_SCHEMA,
    )
    assert "--session" not in captured["command"]


def test_opencode_response_recovers_json_after_text() -> None:
    events = json.dumps(
        {
            "type": "text",
            "part": {
                "type": "text",
                "text": 'analysis before result\n{"decision":"keep"}',
            },
        }
    )

    assert _opencode_response(_json_events(events), "reflection") == {"decision": "keep"}


def test_pi_response_extracts_final_assistant_json() -> None:
    events = "\n".join(
        [
            json.dumps({"type": "session", "id": "test"}),
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": '{"decision":"keep"}'},
                        ],
                    },
                }
            ),
        ]
    )

    assert _pi_response(_json_events(events), "reflection") == {"decision": "keep"}


def test_pi_agent_uses_json_mode_and_read_only_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            "\n".join(
                [
                    json.dumps({"type": "session", "id": "pi-session-1", "cwd": str(kwargs["cwd"])}),
                    json.dumps(
                        {
                            "type": "message_end",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": '{"hypothesis":"test","instructions":"change",'
                                        '"rationale":"reason","profiling_question":null}',
                                    }
                                ],
                            },
                        }
                    ),
                ]
            )
            + "\n",
            "",
        )

    monkeypatch.setattr("spar.agent.adapter.subprocess.run", fake_run)
    agent = PiAgent(model="anthropic/test", effort="high", timeout_seconds=30)

    response = agent.invoke(
        kind="proposal",
        prompt="bounded prompt",
        cwd=tmp_path,
        schema=PROPOSAL_SCHEMA,
        event_path=tmp_path / "proposal.pi.jsonl",
    )

    command = captured["command"]
    assert response["hypothesis"] == "test"
    assert command[:4] == ["pi", "--mode", "json", "--no-approve"]
    assert command[command.index("--model") + 1] == "anthropic/test"
    assert command[command.index("--thinking") + 1] == "high"
    assert command[command.index("--tools") + 1] == "read,grep,find,ls"
    assert captured["kwargs"]["input"].startswith("bounded prompt")
    assert captured["kwargs"]["cwd"] == tmp_path
    assert agent.session_id == "pi-session-1"

    agent.invoke(
        kind="proposal",
        prompt="next prompt",
        cwd=tmp_path,
        schema=PROPOSAL_SCHEMA,
    )
    assert captured["command"][captured["command"].index("--session") + 1] == "pi-session-1"

    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    agent.invoke(
        kind="proposal",
        prompt="another prompt",
        cwd=other_cwd,
        schema=PROPOSAL_SCHEMA,
    )
    assert captured["command"][captured["command"].index("--fork") + 1] == "pi-session-1"

    resumed = PiAgent(model="anthropic/test", effort="high", timeout_seconds=30, session_id="persisted-session")
    resumed.invoke(
        kind="proposal",
        prompt="resumed prompt",
        cwd=tmp_path,
        schema=PROPOSAL_SCHEMA,
    )
    assert captured["command"][captured["command"].index("--fork") + 1] == "persisted-session"


def test_claude_code_agent_uses_structured_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "result": "ignored envelope text",
                    "session_id": "claude-session-1",
                    "structured_output": {"summary": "changed", "limitations": []},
                }
            ),
            "",
        )

    monkeypatch.setattr("spar.agent.adapter.subprocess.run", fake_run)
    agent = ClaudeCodeAgent(model="sonnet", effort="high", timeout_seconds=30)

    response = agent.invoke(
        kind="implementation",
        prompt="implement one change",
        cwd=tmp_path,
        schema=IMPLEMENTATION_SCHEMA,
        event_path=tmp_path / "implementation.claude.json",
    )

    command = captured["command"]
    assert response == {"summary": "changed", "limitations": []}
    assert command[:4] == ["claude", "-p", "--output-format", "json"]
    assert command[command.index("--model") + 1] == "sonnet"
    assert command[command.index("--effort") + 1] == "high"
    assert command[command.index("--permission-mode") + 1] == "bypassPermissions"
    assert "--dangerously-skip-permissions" in command
    assert captured["kwargs"]["input"] == "implement one change"
    assert captured["kwargs"]["cwd"] == tmp_path
    assert agent.session_id == "claude-session-1"

    agent.invoke(
        kind="proposal",
        prompt="propose one change",
        cwd=tmp_path,
        schema=PROPOSAL_SCHEMA,
    )
    command = captured["command"]
    assert command[command.index("--resume") + 1] == "claude-session-1"
    assert command[command.index("--permission-mode") + 1] == "plan"
    assert "--dangerously-skip-permissions" not in command


def test_codex_adapter_records_partial_output_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        error = subprocess.TimeoutExpired(command, 30)
        error.stdout = b"partial output\n"
        raise error

    monkeypatch.setattr("spar.agent.adapter.subprocess.run", timeout)
    event_path = tmp_path / "timeout.events.jsonl"
    adapter = CodexAgent(model="gpt-test", effort="high", timeout_seconds=30)

    with pytest.raises(SparError, match="Codex proposal timed out"):
        adapter.invoke(
            kind="proposal",
            prompt="bounded prompt",
            cwd=tmp_path,
            schema=PROPOSAL_SCHEMA,
            event_path=event_path,
        )

    assert event_path.read_text(encoding="utf-8") == "partial output\n"
