import math
from typing import Any

from .lifecycle import (
    NONTERMINAL_CANDIDATE_STATUSES,
    ROOT_CANDIDATE_ID,
    CandidateStatus,
    Decision,
)
from .storage.db import DB

DEFAULT_EXPLORATION_CONSTANT = math.sqrt(2)


def backpropagate(store: DB, candidate_id: str, reward: float) -> None:
    by_id = {candidate["id"]: candidate for candidate in store.candidates()}
    current_id: str | None = candidate_id
    while current_id is not None:
        candidate = by_id.get(current_id)
        if candidate is None:
            break
        store.increment_mcts(current_id, reward)
        current_id = candidate["parent_id"]


def top_candidates(
    store: DB,
    limit: int | None = None,
    *,
    exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT,
) -> list[dict[str, float | int | str]]:
    candidates = store.candidates()
    expandable = {
        candidate["id"]: candidate
        for candidate in candidates
        if is_expandable(candidate) and candidate["eval_score"] is not None
    }
    if not expandable:
        return []

    mean_values = {
        candidate_id: float(candidate["mcts_value_sum"]) / int(candidate["mcts_visits"])
        for candidate_id, candidate in expandable.items()
    }
    low, high = min(mean_values.values()), max(mean_values.values())

    def normalized(value: float) -> float:
        return 0.5 if high == low else (value - low) / (high - low)

    pending_rollouts: dict[str, int] = {}
    all_candidates = {candidate["id"]: candidate for candidate in candidates}
    for candidate in candidates:
        if candidate["status"] not in NONTERMINAL_CANDIDATE_STATUSES or candidate["parent_id"] is None:
            continue
        parent_id = candidate["parent_id"]
        while parent_id is not None:
            pending_rollouts[parent_id] = pending_rollouts.get(parent_id, 0) + 1
            parent = all_candidates.get(parent_id)
            parent_id = None if parent is None else parent["parent_id"]

    root_visits = int(all_candidates[ROOT_CANDIDATE_ID]["mcts_visits"])
    root_pending = pending_rollouts.get(ROOT_CANDIDATE_ID, 0)
    total_visits = max(1, root_visits + root_pending)
    ranked = []
    for candidate_id, candidate in expandable.items():
        candidate_visits = int(candidate["mcts_visits"])
        value_sum = float(candidate["mcts_value_sum"])
        pending = pending_rollouts.get(candidate_id, 0)
        mean_value = mean_values[candidate_id]
        exploitation_value = normalized(mean_value)
        exploration_bonus = exploration_constant * math.sqrt(math.log(total_visits + 1) / (candidate_visits + pending))
        ranked.append(
            {
                "candidate_id": candidate_id,
                "score": float(candidate["eval_score"]),
                "visits": candidate_visits,
                "value_sum": value_sum,
                "mean_value": mean_value,
                "exploitation_value": exploitation_value,
                "pending_rollouts": pending,
                "exploration_bonus": exploration_bonus,
                "priority": exploitation_value + exploration_bonus,
            }
        )
    ranked.sort(key=lambda item: (-item["priority"], -item["score"], item["candidate_id"]))
    return ranked[:limit]


def is_expandable(candidate: dict[str, Any]) -> bool:
    return (
        candidate["status"] == CandidateStatus.COMPLETED
        and candidate["decision"] != Decision.DISCARD
        and candidate["commit_sha"] is not None
    )
