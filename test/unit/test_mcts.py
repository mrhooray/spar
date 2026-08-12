import pytest

from spar import mcts
from spar.lifecycle import CandidateStatus, Decision


class CandidateStore:
    def __init__(self, candidates: list[dict[str, object]]) -> None:
        self._candidates = candidates

    def candidates(self) -> list[dict[str, object]]:
        return self._candidates


def test_exploration_constant_scales_the_exploration_bonus() -> None:
    store = CandidateStore(
        [
            {
                "id": "root",
                "parent_id": None,
                "commit_sha": "root-sha",
                "status": CandidateStatus.COMPLETED,
                "decision": Decision.KEEP,
                "eval_score": 0.0,
                "mcts_visits": 2,
                "mcts_value_sum": 0.0,
            },
            {
                "id": "child",
                "parent_id": "root",
                "commit_sha": "child-sha",
                "status": CandidateStatus.COMPLETED,
                "decision": Decision.KEEP,
                "eval_score": 1.0,
                "mcts_visits": 1,
                "mcts_value_sum": 1.0,
            },
        ]
    )

    default = mcts.top_candidates(store, exploration_constant=1.0)
    doubled = mcts.top_candidates(store, exploration_constant=2.0)
    default_by_id = {item["candidate_id"]: item for item in default}
    doubled_by_id = {item["candidate_id"]: item for item in doubled}

    assert doubled_by_id["root"]["exploration_bonus"] == pytest.approx(2 * default_by_id["root"]["exploration_bonus"])
    assert doubled_by_id["child"]["exploration_bonus"] == pytest.approx(2 * default_by_id["child"]["exploration_bonus"])
