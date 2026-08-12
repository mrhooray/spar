import pytest

from spar.agent.invocation import propose, validate_response
from spar.error import SparError
from spar.lifecycle import Decision
from spar.storage.db import DB


def test_validate_response_accepts_each_agent_contract() -> None:
    validate_response(
        "proposal",
        {
            "hypothesis": "the change improves the score",
            "instructions": "change one thing",
            "rationale": "the bottleneck is here",
            "profiling_question": None,
        },
    )
    validate_response(
        "implementation",
        {"summary": "changed one thing", "limitations": []},
    )
    validate_response(
        "reflection",
        {
            "decision": Decision.KEEP,
            "decision_reason": "the score improved",
            "learnings": "the change is useful",
        },
    )


@pytest.mark.parametrize(
    ("kind", "response", "message"),
    [
        (
            "proposal",
            {
                "hypothesis": "test",
                "instructions": "test",
                "profiling_question": None,
            },
            "proposal response fields",
        ),
        (
            "proposal",
            {
                "hypothesis": "test",
                "instructions": "test",
                "rationale": "test",
                "profiling_question": "",
            },
            "profiling_question must be null or a non-empty string",
        ),
        (
            "implementation",
            {"summary": "changed", "limitations": ["known issue", 1]},
            "limitations",
        ),
        (
            "reflection",
            {
                "decision": "maybe",
                "decision_reason": "test",
                "learnings": "test",
            },
            "decision must be keep or discard",
        ),
    ],
)
def test_validate_response_rejects_invalid_contracts(kind: str, response: dict[str, object], message: str) -> None:
    with pytest.raises(SparError, match=message):
        validate_response(kind, response)


def test_validate_response_rejects_empty_required_text() -> None:
    with pytest.raises(SparError, match="must not be empty"):
        validate_response(
            "reflection",
            {
                "decision": Decision.DISCARD,
                "decision_reason": " ",
                "learnings": "test",
            },
        )


def test_invalid_agent_response_is_recorded_on_a_failed_span(tmp_path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    with DB(session) as db:
        db.initialize(repo_path=str(tmp_path), root_commit="root-sha")

    class InvalidAgent:
        cli = "test-agent"
        model = "test-model"
        effort = "high"

        def invoke(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "hypothesis": "test",
                "instructions": "test",
                "rationale": "test",
                "profiling_question": "",
            }

    with pytest.raises(SparError, match="profiling_question"):
        propose(
            InvalidAgent(),
            objective="test objective",
            progress={},
            session_name="demo",
            cwd=tmp_path,
            event_path=session / "proposal.events.jsonl",
            state_path=session,
            parent_id="root",
        )

    with DB(session) as db:
        span = db.spans("root")[-1]
        assert span["kind"] == "proposal"
        assert span["success"] is False
        assert span["invocation"]["response"]["profiling_question"] == ""
