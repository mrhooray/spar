import json
from pathlib import Path
from typing import Any

from ..error import SparError
from ..lifecycle import Decision
from ..storage.db import Span
from .adapter import AgentAdapter


def propose(
    agent: AgentAdapter,
    *,
    objective: str,
    progress: dict[str, Any],
    session_name: str,
    cwd: Path,
    event_path: Path,
    state_path: Path,
    parent_id: str,
) -> dict[str, Any]:
    return _invoke(
        agent,
        kind="proposal",
        prompt=proposal_prompt(objective, progress, session_name),
        cwd=cwd,
        schema=PROPOSAL_SCHEMA,
        event_path=event_path,
        state_path=state_path,
        candidate_id=parent_id,
    )


def implement(
    agent: AgentAdapter,
    *,
    objective: str,
    candidate: dict[str, Any],
    cwd: Path,
    event_path: Path,
    state_path: Path,
    candidate_id: str,
) -> dict[str, Any]:
    return _invoke(
        agent,
        kind="implementation",
        prompt=implementation_prompt(objective, candidate),
        cwd=cwd,
        schema=IMPLEMENTATION_SCHEMA,
        event_path=event_path,
        state_path=state_path,
        candidate_id=candidate_id,
    )


def reflect(
    agent: AgentAdapter,
    *,
    objective: str,
    implementation_report: dict[str, Any],
    intervention: dict[str, Any],
    evaluation: dict[str, Any],
    profiles: dict[str, Any],
    parent: dict[str, Any],
    session_name: str,
    cwd: Path,
    event_path: Path,
    state_path: Path,
    candidate_id: str,
) -> dict[str, Any]:
    return _invoke(
        agent,
        kind="reflection",
        prompt=reflection_prompt(
            objective,
            intervention,
            implementation_report,
            evaluation,
            profiles,
            parent,
            session_name,
        ),
        cwd=cwd,
        schema=REFLECTION_SCHEMA,
        event_path=event_path,
        state_path=state_path,
        candidate_id=candidate_id,
    )


def _invoke(
    agent: AgentAdapter,
    *,
    kind: str,
    prompt: str,
    cwd: Path,
    schema: dict[str, Any],
    event_path: Path,
    state_path: Path,
    candidate_id: str,
) -> dict[str, Any]:
    try:
        event_reference = str(event_path.relative_to(state_path))
    except ValueError:
        event_reference = str(event_path)
    invocation = {
        "agent": str(getattr(agent, "cli", type(agent).__name__)),
        "model": _optional_text(getattr(agent, "model", None)),
        "effort": _optional_text(getattr(agent, "effort", None)),
        "prompt": prompt,
        "schema": schema,
        "event_path": event_reference,
    }
    response: dict[str, Any] | None = None
    with Span(
        state_path,
        candidate_id=candidate_id,
        kind=kind,
        invocation=invocation,
    ) as span:
        assert span.id is not None
        try:
            response = agent.invoke(
                kind=kind,
                prompt=prompt,
                cwd=cwd,
                schema=schema,
                event_path=event_path,
            )
            validate_response(kind, response)
        except Exception as exc:
            with span.complete(error=str(exc).strip() or type(exc).__name__) as transaction:
                transaction.update_invocation_response(
                    span.id,
                    response,
                )
            raise
        with span.complete() as transaction:
            transaction.update_invocation_response(
                span.id,
                response,
            )
    assert response is not None
    return response


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def proposal_prompt(objective: str, progress: dict[str, Any], session_name: str) -> str:
    return f"""Propose the next intervention in an automated, evidence-driven software research session.

The selected parent's repository state is your current working directory. Inspect its source and
ground the intervention in specific existing code and a concrete mechanism. The intervention must
be implementation-ready. Do not modify files.

Use the research progress below to avoid repeating completed, failed, discarded, or currently active
interventions. The selected parent is fixed; do not select another parent. Propose one coherent
intervention, not unrelated changes, evaluation work, profiling work, or further planning.

You may run `spar inspect {session_name} <candidate-id>` to find more information about a listed
candidate and locate its artifacts. Run `spar inspect --help` for usage. Use SPAR only for read-only
inspection.

Existing profiling artifacts are prior evidence; inspect them when useful. If a profiler is configured,
request profiling when a specific question about the implemented intervention would benefit from
runtime evidence. The profiler runs after implementation and informs reflection, so the intervention
must be implementable without it.

Return:
- hypothesis: the measurable effect this intervention is expected to have relative to the selected
  parent, including any runtime bottleneck it targets
- instructions: the exact implementation-ready intervention
- rationale: the source- and evidence-grounded mechanism that could produce the expected outcome
- profiling_question: the specific question that profiling artifacts should help reflection answer and
  why it matters, or null when profiling is not requested

Research objective:
{objective}

Research progress:
{json.dumps(progress, indent=2, sort_keys=True)}
"""


def implementation_prompt(objective: str, candidate: dict[str, Any]) -> str:
    intervention = {key: candidate[key] for key in ("hypothesis", "instructions", "rationale")}
    return f"""You are implementing the assigned intervention in an automated, evidence-driven
software research session.

Implement exactly the assigned intervention in the current worktree. Do not choose a different
intervention or add unrelated changes. Work only with the current worktree and its local files.
Do not commit; leave the intended changes in the current worktree.

Return an implementation report:
- summary: what changed
- limitations: what remains unknown or unverified; use an empty array when there are none
Return only the requested JSON.

Objective and constraints:
{objective}

Assigned intervention:
{json.dumps(intervention, indent=2, sort_keys=True)}
"""


def reflection_prompt(
    objective: str,
    intervention: dict[str, Any],
    implementation_report: dict[str, Any],
    evaluation: dict[str, Any],
    profiles: dict[str, Any],
    parent: dict[str, Any],
    session_name: str,
) -> str:
    evidence = {
        "intervention": intervention,
        "parent": parent,
        "implementation_report": implementation_report,
        "canonical_evaluation": evaluation,
        "profiles": profiles,
    }
    parent_commit = parent["commit_sha"]
    return f"""You are reflecting on one evaluated intervention in an automated, evidence-driven
software research session.
Judge this single intervention against the objective and its evidence. Return only the requested
JSON.

Inspect the current worktree and its local files as needed. To review the exact intervention, run
`git diff {parent_commit} HEAD`. Use `spar inspect {session_name} <candidate-id>` for additional
candidate information and artifact paths. Use these commands read-only.
The parent is the completed predecessor and baseline for this intervention.

Return:
- decision: keep only when the evidence supports retaining the intervention; otherwise discard
- decision_reason: the concrete evidence-based basis for the decision
- learnings: reusable findings from the intervention and evidence. If a profiling question was
  requested and profiling evidence is available, include its answer alongside the other learnings.

Objective and constraints:
{objective}

Intervention evidence:
{json.dumps(evidence, indent=2, sort_keys=True)}
"""


def validate_response(kind: str, response: dict[str, Any]) -> None:
    fields = {
        "proposal": (
            {"hypothesis", "instructions", "rationale", "profiling_question"},
            {"hypothesis", "instructions", "rationale"},
        ),
        "implementation": (
            {"summary", "limitations"},
            {"summary"},
        ),
        "reflection": (
            {"decision", "decision_reason", "learnings"},
            {"decision", "decision_reason", "learnings"},
        ),
    }
    if not isinstance(response, dict):
        raise SparError(f"{kind} response must be a JSON object")
    expected, string_fields = fields[kind]
    if response.keys() != expected:
        raise SparError(f"{kind} response fields must be: {', '.join(sorted(expected))}")
    for field in string_fields:
        if not isinstance(response[field], str):
            raise SparError(f"{kind} response field must be a string: {field}")
    if any(not response[field].strip() for field in string_fields):
        raise SparError(f"{kind} response string fields must not be empty")
    if (
        kind == "proposal"
        and response["profiling_question"] is not None
        and (not isinstance(response["profiling_question"], str) or not response["profiling_question"].strip())
    ):
        raise SparError("propose response profiling_question must be null or a non-empty string")
    if kind == "implementation" and (
        not isinstance(response["limitations"], list)
        or not all(isinstance(item, str) for item in response["limitations"])
    ):
        raise SparError("implementation response field must be a string array: limitations")
    if kind == "reflection" and response["decision"] not in {Decision.KEEP, Decision.DISCARD}:
        raise SparError("reflection response decision must be keep or discard")


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "minLength": 1},
        "instructions": {"type": "string", "minLength": 1},
        "rationale": {"type": "string", "minLength": 1},
        "profiling_question": {"type": ["string", "null"], "minLength": 1},
    },
    "required": [
        "hypothesis",
        "instructions",
        "rationale",
        "profiling_question",
    ],
    "additionalProperties": False,
}


IMPLEMENTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "limitations"],
    "additionalProperties": False,
}


REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": [Decision.KEEP, Decision.DISCARD]},
        "decision_reason": {"type": "string", "minLength": 1},
        "learnings": {"type": "string", "minLength": 1},
    },
    "required": ["decision", "decision_reason", "learnings"],
    "additionalProperties": False,
}
