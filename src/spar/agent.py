import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from .errors import SparError
from .repo import read_json_object, write_json
from .state import DECISION_DISCARD, DECISION_KEEP

EXCLUDED_IMPLEMENTATION_ENV = (
    "SPAR_ROOT",
    "SPAR_SESSION_DIR",
    "SPAR_CANDIDATE_ID",
    "SPAR_PARENT_SHA",
    "SPAR_PROFILING_DIR",
    "SPAR_PROFILING_RESULT",
)


class PhaseAgent(Protocol):
    def run(
        self,
        *,
        phase: str,
        prompt: str,
        cwd: Path,
        schema: dict[str, Any],
        event_path: Path | None = None,
    ) -> dict[str, Any]: ...


class CodexAgent:
    def __init__(
        self,
        *,
        model: str,
        effort: str,
        timeout_seconds: int,
        preserve_context: bool = False,
    ) -> None:
        if not model.strip() or not effort.strip():
            raise SparError("Codex model and reasoning effort must not be empty")
        if timeout_seconds < 1:
            raise SparError("Codex timeout must be a positive integer")
        self.model = model
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.preserve_context = preserve_context
        self.thread_id: str | None = None

    def run(
        self,
        *,
        phase: str,
        prompt: str,
        cwd: Path,
        schema: dict[str, Any],
        event_path: Path | None = None,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"spar-{phase}-") as temporary:
            temporary_path = Path(temporary)
            schema_path = temporary_path / "schema.json"
            response_path = temporary_path / "response.json"
            write_json(schema_path, schema)
            command = self._command(
                phase=phase,
                cwd=cwd,
                schema_path=schema_path,
                response_path=response_path,
            )
            try:
                result = subprocess.run(
                    command,
                    input=prompt,
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
                        exc.stdout.decode("utf-8", errors="replace")
                        if isinstance(exc.stdout, bytes)
                        else exc.stdout
                    )
                    event_path.write_text(output, encoding="utf-8")
                raise SparError(f"Codex {phase} phase timed out after {self.timeout_seconds}s") from exc
            if event_path is not None:
                event_path.parent.mkdir(parents=True, exist_ok=True)
                event_path.write_text(result.stdout, encoding="utf-8")
            if result.returncode != 0:
                detail = result.stderr.strip().splitlines()
                suffix = f": {detail[-1]}" if detail else ""
                raise SparError(f"Codex {phase} phase failed with exit code {result.returncode}{suffix}")
            if self.preserve_context:
                observed_thread_id = _codex_thread_id(result.stdout)
                if self.thread_id is None:
                    self.thread_id = observed_thread_id
                elif observed_thread_id != self.thread_id:
                    raise SparError("Codex phase did not resume the sequential researcher thread")
            return read_json_object(response_path, f"Codex {phase} response")

    def _command(
        self,
        *,
        phase: str,
        cwd: Path,
        schema_path: Path,
        response_path: Path,
    ) -> list[str]:
        sandbox = "workspace-write" if phase == "implement" else "read-only"
        command = ["codex", "--ask-for-approval", "never", "exec"]
        if self.preserve_context and self.thread_id is not None:
            command.extend(["--sandbox", sandbox, "--cd", str(cwd), "resume"])
        else:
            if not self.preserve_context:
                command.append("--ephemeral")
            command.extend(["--sandbox", sandbox, "--cd", str(cwd)])
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
        if self.preserve_context and self.thread_id is not None:
            command.append(self.thread_id)
        command.append("-")
        return command


def _codex_thread_id(events: str) -> str:
    thread_ids = []
    for line in events.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started":
            thread_ids.append(event.get("thread_id"))
    if (
        len(thread_ids) != 1
        or not isinstance(thread_ids[0], str)
        or not thread_ids[0].strip()
    ):
        raise SparError("Codex phase did not report exactly one researcher thread ID")
    return thread_ids[0]


def run_phase(
    agent: PhaseAgent,
    *,
    phase: str,
    prompt: str,
    cwd: Path,
    schema: dict[str, Any],
    artifact_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = time.time_ns() // 1_000_000
    started_monotonic = time.monotonic_ns()
    event_path = artifact_path.with_suffix(".codex.jsonl")
    response: dict[str, Any] | None = None
    try:
        response = agent.run(
            phase=phase,
            prompt=prompt,
            cwd=cwd,
            schema=schema,
            event_path=event_path,
        )
        validate_response(phase, response)
    except Exception as exc:
        record = _phase_record(
            phase,
            started_at,
            started_monotonic,
            success=False,
            prompt=prompt,
            schema=schema,
            response=response,
            error=str(exc).strip() or type(exc).__name__,
            event_path=event_path,
        )
        write_json(artifact_path, record)
        raise
    record = _phase_record(
        phase,
        started_at,
        started_monotonic,
        success=True,
        prompt=prompt,
        schema=schema,
        response=response,
        event_path=event_path,
    )
    write_json(artifact_path, record)
    return response, record


def truncate_diff(diff: str, limit: int = 50_000) -> str:
    if len(diff) <= limit:
        return diff
    return diff[:limit] + f"\n... diff truncated after {limit} characters ...\n"


def compact_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    fields = (
        "id",
        "parent_id",
        "hypothesis",
        "status",
        "eval_score",
        "eval_summary",
        "decision",
        "decision_reason",
        "error",
    )
    return {field: candidate.get(field) for field in fields}


def proposal_prompt(objective: str, evidence: dict[str, Any]) -> str:
    return f"""You are the proposal phase of a bounded SPAR research run.
Choose exactly one implementation-ready intervention from the selected parent. Return only the
requested JSON. The instructions must prescribe a concrete task-local implementation change.
Before responding, inspect the selected parent's task-local source and ground the intervention in a
concrete existing code location and mechanism.
Profiling, evaluation, or planning alone is not a candidate. Requested profiling runs only after the
change is implemented and canonically evaluated, solely to inform reflection; the intervention cannot
depend on its result.
Make the intervention materially different from every active sibling in the evidence. Paraphrasing
or reimplementing the same mechanism is not different. Use judgment here; the orchestrator only owns
selection and admission.

Objective and constraints:
{objective}

Compact state, MCTS ranking, and prior evidence:
{json.dumps(evidence, indent=2, sort_keys=True)}
"""


def implementation_prompt(objective: str, proposal: dict[str, Any]) -> str:
    intervention = {
        key: proposal[key] for key in ("hypothesis", "instructions", "rationale")
    }
    return f"""You are the implementation phase for one admitted SPAR candidate.
Implement exactly the assigned intervention in this worktree. Do not choose or add another
intervention. Use only task-local repository context and run narrow local checks. Do not commit;
leave only the intended working-tree changes for the orchestrator. Do not invoke SPAR or search
for its session, evaluator, profiler, trusted paths, or other candidates. Return only the requested
JSON.

Objective and constraints:
{objective}

Assigned intervention:
{json.dumps(intervention, indent=2, sort_keys=True)}
"""


def reflection_prompt(
    objective: str,
    proposal: dict[str, Any],
    implementation: dict[str, Any],
    evaluation: dict[str, Any],
    profiling: dict[str, Any],
    comparison: dict[str, Any],
    diff: str,
) -> str:
    evidence = {
        "proposal": proposal,
        "implementation_report": implementation,
        "canonical_evaluation": evaluation,
        "profiling": profiling,
        "comparison": comparison,
        "candidate_diff": diff,
    }
    return f"""You are the reflection phase for one measured SPAR candidate.
Judge this single intervention against the objective and its evidence. Return keep or discard plus
a concise learning summary and concrete reason. Return only the requested JSON.

Objective and constraints:
{objective}

Canonical result, diff, and reports:
{json.dumps(evidence, indent=2, sort_keys=True)}
"""


def validate_response(phase: str, response: dict[str, Any]) -> None:
    fields = {
        "propose": (
            {"hypothesis", "instructions", "rationale", "request_profiling", "profiling_reason"},
            {"hypothesis", "instructions", "rationale", "profiling_reason"},
        ),
        "implement": (
            {"summary", "observations", "limitations"},
            {"summary"},
        ),
        "reflect": (
            {"decision", "summary", "decision_reason"},
            {"decision", "summary", "decision_reason"},
        ),
    }
    if not isinstance(response, dict):
        raise SparError(f"{phase} response must be a JSON object")
    expected, string_fields = fields[phase]
    if response.keys() != expected:
        raise SparError(f"{phase} response fields must be: {', '.join(sorted(expected))}")
    for field in string_fields:
        if not isinstance(response[field], str):
            raise SparError(f"{phase} response field must be a string: {field}")
    required_strings = string_fields - {"profiling_reason"}
    if any(not response[field].strip() for field in required_strings):
        raise SparError(f"{phase} response string fields must not be empty")
    if phase == "propose" and not isinstance(response["request_profiling"], bool):
        raise SparError("propose response request_profiling must be a boolean")
    if phase == "propose" and response["request_profiling"] and not response["profiling_reason"].strip():
        raise SparError("proposal profiling reason must not be empty when profiling is requested")
    if phase == "implement":
        for field in ("observations", "limitations"):
            if not isinstance(response[field], list) or not all(
                isinstance(item, str) for item in response[field]
            ):
                raise SparError(f"implement response field must be a string array: {field}")
    if phase == "reflect" and response["decision"] not in {DECISION_KEEP, DECISION_DISCARD}:
        raise SparError("reflect response decision must be keep or discard")


def _phase_record(
    phase: str,
    started_at: int,
    started_monotonic: int,
    *,
    success: bool,
    prompt: str,
    schema: dict[str, Any],
    response: dict[str, Any] | None = None,
    error: str | None = None,
    event_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "started_at": started_at,
        "completed_at": time.time_ns() // 1_000_000,
        "elapsed_ms": (time.monotonic_ns() - started_monotonic) // 1_000_000,
        "success": success,
        "prompt": prompt,
        "schema": schema,
        "response": response,
        "error": error,
        "events_path": str(event_path) if event_path and event_path.exists() else None,
    }


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "minLength": 1},
        "instructions": {"type": "string", "minLength": 1},
        "rationale": {"type": "string", "minLength": 1},
        "request_profiling": {"type": "boolean"},
        "profiling_reason": {"type": "string"},
    },
    "required": [
        "hypothesis",
        "instructions",
        "rationale",
        "request_profiling",
        "profiling_reason",
    ],
    "additionalProperties": False,
}


IMPLEMENTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "observations": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "observations", "limitations"],
    "additionalProperties": False,
}


REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": [DECISION_KEEP, DECISION_DISCARD]},
        "summary": {"type": "string", "minLength": 1},
        "decision_reason": {"type": "string", "minLength": 1},
    },
    "required": ["decision", "summary", "decision_reason"],
    "additionalProperties": False,
}
