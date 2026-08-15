import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol

from ..error import SparError

EXCLUDED_IMPLEMENTATION_ENV = (
    "SPAR_ROOT",
    "SPAR_CANDIDATE_ID",
    "SPAR_PARENT_SHA",
    "SPAR_PROFILING_DIR",
)


class AgentAdapter(Protocol):
    session_id: str | None

    def invoke(
        self,
        *,
        kind: str,
        prompt: str,
        cwd: Path,
        schema: dict[str, Any],
        event_path: Path | None = None,
    ) -> dict[str, Any]: ...


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SparError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SparError(f"{label} must contain valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SparError(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_agent(config: dict[str, Any], *, session_id: str | None = None) -> AgentAdapter:
    cli = config["cli"]
    if cli not in {"claude-code", "codex", "opencode", "pi"}:
        raise SparError("agent.cli must be configured as 'codex', 'claude-code', 'opencode', or 'pi'")
    for name in ("model", "effort"):
        if config[name].startswith("REPLACE_WITH_"):
            raise SparError(f"agent.{name} must be configured")
    agent_class = {
        "claude-code": ClaudeCodeAgent,
        "codex": CodexAgent,
        "opencode": OpenCodeAgent,
        "pi": PiAgent,
    }[cli]
    return agent_class(
        model=config["model"],
        effort=config["effort"],
        timeout_seconds=config["timeout_seconds"],
        session_id=session_id,
    )


class ClaudeCodeAgent:
    cli = "claude-code"

    def __init__(self, *, model: str, effort: str, timeout_seconds: int, session_id: str | None = None) -> None:
        if not model.strip() or not effort.strip():
            raise SparError("Claude Code model and effort must not be empty")
        if timeout_seconds < 1:
            raise SparError("Claude Code timeout must be a positive integer")
        self.model = model
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.session_id = session_id

    def invoke(
        self,
        *,
        kind: str,
        prompt: str,
        cwd: Path,
        schema: dict[str, Any],
        event_path: Path | None = None,
    ) -> dict[str, Any]:
        command = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema, separators=(",", ":")),
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--permission-mode",
            "bypassPermissions" if kind == "implementation" else "plan",
        ]
        if self.session_id is not None:
            command.extend(["--resume", self.session_id])
        if kind == "implementation":
            command.append("--dangerously-skip-permissions")
        try:
            result = subprocess.run(
                command,
                input=prompt,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SparError("Claude Code executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            _write_partial_output(event_path, exc.stdout)
            raise SparError(f"Claude Code {kind} timed out after {self.timeout_seconds}s") from exc
        if event_path is not None:
            _write_text(event_path, result.stdout)
        if result.returncode != 0:
            raise SparError(_command_failure("Claude Code", kind, result))
        try:
            envelope = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise SparError(f"Claude Code {kind} returned invalid JSON") from exc
        if isinstance(envelope, dict):
            session_id = envelope.get("session_id")
            if isinstance(session_id, str) and session_id.strip():
                self.session_id = session_id
        response = envelope.get("structured_output") if isinstance(envelope, dict) else None
        if not isinstance(response, dict):
            raise SparError(f"Claude Code {kind} returned no structured output")
        return response


class CodexAgent:
    cli = "codex"

    def __init__(
        self,
        *,
        model: str,
        effort: str,
        timeout_seconds: int,
        session_id: str | None = None,
    ) -> None:
        if not model.strip() or not effort.strip():
            raise SparError("Codex model and reasoning effort must not be empty")
        if timeout_seconds < 1:
            raise SparError("Codex timeout must be a positive integer")
        self.model = model
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.session_id = session_id

    def invoke(
        self,
        *,
        kind: str,
        prompt: str,
        cwd: Path,
        schema: dict[str, Any],
        event_path: Path | None = None,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"spar-{kind}-") as temporary:
            temporary_path = Path(temporary)
            schema_path = temporary_path / "schema.json"
            response_path = temporary_path / "response.json"
            _write_json(schema_path, schema)
            command = self._command(
                kind=kind,
                schema_path=schema_path,
                response_path=response_path,
            )
            try:
                result = subprocess.run(
                    command,
                    input=prompt,
                    cwd=cwd,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise SparError("Codex executable was not found") from exc
            except subprocess.TimeoutExpired as exc:
                if event_path is not None and exc.stdout:
                    event_path.parent.mkdir(parents=True, exist_ok=True)
                    output = (
                        exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
                    )
                    event_path.write_text(output, encoding="utf-8")
                raise SparError(f"Codex {kind} timed out after {self.timeout_seconds}s") from exc
            if event_path is not None:
                event_path.parent.mkdir(parents=True, exist_ok=True)
                event_path.write_text(result.stdout, encoding="utf-8")
            if result.returncode != 0:
                detail = result.stderr.strip().splitlines()
                suffix = f": {detail[-1]}" if detail else ""
                raise SparError(f"Codex {kind} failed with exit code {result.returncode}{suffix}")
            response = _read_json_object(response_path, f"Codex {kind} response")
            for event in _json_events(result.stdout):
                if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
                    self.session_id = event["thread_id"]
                    break
            return response

    def _command(
        self,
        *,
        kind: str,
        schema_path: Path,
        response_path: Path,
    ) -> list[str]:
        sandbox = "workspace-write" if kind == "implementation" else "read-only"
        command = [
            "codex",
            "--ask-for-approval",
            "never",
            "--sandbox",
            sandbox,
            "exec",
        ]
        if self.session_id is not None:
            command.extend(["resume", self.session_id])
        command.extend(
            [
                "--ignore-user-config",
                "--ignore-rules",
                "--model",
                self.model,
                "--config",
                f"model_reasoning_effort={self.effort}",
                "--config",
                "agents.enabled=false",
                "--config",
                "features.multi_agent=false",
                "--config",
                "features.multi_agent_v2=false",
                "--config",
                "shell_environment_policy.inherit=core",
                "--config",
                f"shell_environment_policy.exclude={json.dumps(EXCLUDED_IMPLEMENTATION_ENV)}",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(response_path),
                "--json",
            ]
        )
        command.append("-")
        return command


class OpenCodeAgent:
    cli = "opencode"

    def __init__(self, *, model: str, effort: str, timeout_seconds: int, session_id: str | None = None) -> None:
        if not model.strip() or not effort.strip():
            raise SparError("OpenCode model and reasoning effort must not be empty")
        if timeout_seconds < 1:
            raise SparError("OpenCode timeout must be a positive integer")
        self.model = model
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.session_id = session_id
        self.session_cwd: Path | None = None

    def invoke(
        self,
        *,
        kind: str,
        prompt: str,
        cwd: Path,
        schema: dict[str, Any],
        event_path: Path | None = None,
    ) -> dict[str, Any]:
        request = (
            f"{prompt}\n\nReturn a JSON object matching this schema:\n{json.dumps(schema, indent=2, sort_keys=True)}"
        )
        command = [
            "opencode",
            "run",
            "--format",
            "json",
            "--model",
            self.model,
            "--variant",
            self.effort,
            "--dir",
            str(cwd),
            "--auto",
            request,
        ]
        if self.session_id is not None and self.session_cwd == cwd.resolve():
            command[2:2] = ["--session", self.session_id]
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SparError("OpenCode executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            if event_path is not None and exc.stdout:
                event_path.parent.mkdir(parents=True, exist_ok=True)
                output = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
                event_path.write_text(output, encoding="utf-8")
            raise SparError(f"OpenCode {kind} timed out after {self.timeout_seconds}s") from exc
        if event_path is not None:
            event_path.parent.mkdir(parents=True, exist_ok=True)
            event_path.write_text(result.stdout, encoding="utf-8")
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise SparError(f"OpenCode {kind} failed with exit code {result.returncode}{suffix}")
        events = _json_events(result.stdout)
        response = _opencode_response(events, kind)
        for event in events:
            if isinstance(event.get("sessionID"), str):
                self.session_id = event["sessionID"]
                self.session_cwd = cwd.resolve()
                break
        return response


class PiAgent:
    cli = "pi"

    def __init__(self, *, model: str, effort: str, timeout_seconds: int, session_id: str | None = None) -> None:
        if not model.strip() or not effort.strip():
            raise SparError("Pi model and thinking level must not be empty")
        if timeout_seconds < 1:
            raise SparError("Pi timeout must be a positive integer")
        self.model = model
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.session_id = session_id
        self.session_cwd: Path | None = None

    def invoke(
        self,
        *,
        kind: str,
        prompt: str,
        cwd: Path,
        schema: dict[str, Any],
        event_path: Path | None = None,
    ) -> dict[str, Any]:
        request = (
            f"{prompt}\n\nReturn a JSON object matching this schema:\n{json.dumps(schema, indent=2, sort_keys=True)}"
        )
        command = [
            "pi",
            "--mode",
            "json",
            "--no-approve",
            "--model",
            self.model,
            "--thinking",
            self.effort,
        ]
        if self.session_id is not None:
            option = "--session" if self.session_cwd == cwd.resolve() else "--fork"
            command.extend([option, self.session_id])
        if kind != "implementation":
            command.extend(["--tools", "read,grep,find,ls"])
        command.append("-p")
        try:
            result = subprocess.run(
                command,
                input=request,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SparError("Pi executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            _write_partial_output(event_path, exc.stdout)
            raise SparError(f"Pi {kind} timed out after {self.timeout_seconds}s") from exc
        if event_path is not None:
            _write_text(event_path, result.stdout)
        if result.returncode != 0:
            raise SparError(_command_failure("Pi", kind, result))
        events = _json_events(result.stdout)
        response = _pi_response(events, kind)
        for event in events:
            if event.get("type") == "session" and isinstance(event.get("id"), str):
                self.session_id = event["id"]
                if isinstance(event.get("cwd"), str):
                    self.session_cwd = Path(event["cwd"]).resolve()
                break
        return response


def _json_events(output: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_partial_output(path: Path | None, output: str | bytes | None) -> None:
    if path is None or output is None:
        return
    _write_text(
        path,
        output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output,
    )


def _command_failure(cli: str, kind: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = result.stderr.strip().splitlines()
    suffix = f": {detail[-1]}" if detail else ""
    return f"{cli} {kind} failed with exit code {result.returncode}{suffix}"


def _pi_response(events: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    assistant_text: str | None = None
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        ).strip()
        if text:
            assistant_text = text
    if not assistant_text:
        raise SparError(f"Pi {kind} returned no assistant response")
    try:
        response = json.loads(assistant_text)
    except json.JSONDecodeError:
        response = _last_json_object(assistant_text)
    if not isinstance(response, dict):
        raise SparError(f"Pi {kind} did not return a JSON object")
    return response


def _opencode_response(events: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    text_parts: list[str] = []
    for event in events:
        part = event.get("part")
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        elif event.get("type") == "text" and isinstance(event.get("text"), str):
            text_parts.append(event["text"])
    text = "\n".join(text_parts).strip()
    if not text:
        raise SparError(f"OpenCode {kind} returned no text response")
    try:
        response = json.loads(text)
    except json.JSONDecodeError:
        response = _last_json_object(text)
    if not isinstance(response, dict):
        raise SparError(f"OpenCode {kind} did not return a JSON object")
    return response


def _last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index in range(len(text) - 1, -1, -1):
        if text[index] != "{":
            continue
        try:
            response, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(response, dict):
            return response
    raise SparError("OpenCode response did not contain a JSON object")
