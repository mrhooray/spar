import math
import sqlite3


def backpropagate(db: sqlite3.Connection, candidate_id: str, reward: float) -> None:
    current_id: str | None = candidate_id
    while current_id is not None:
        row = db.execute("SELECT parent_id FROM candidates WHERE id = ?", (current_id,)).fetchone()
        db.execute(
            """
            UPDATE candidates
            SET mcts_visits = mcts_visits + 1,
                mcts_value_sum = mcts_value_sum + ?
            WHERE id = ?
            """,
            (reward, current_id),
        )
        current_id = row["parent_id"]


def top_parents(rows: list[sqlite3.Row]) -> list[dict[str, float | int | str]]:
    from .state import DECISION_DISCARD, NONTERMINAL_STATUSES, ROOT_CANDIDATE_ID, STATUS_COMPLETED

    expandable = {
        row["id"]: row
        for row in rows
        if row["status"] == STATUS_COMPLETED
        and row["eval_score"] is not None
        if row["decision"] != DECISION_DISCARD
    }
    if not expandable:
        return []

    mean_values = {
        candidate_id: float(row["mcts_value_sum"]) / int(row["mcts_visits"])
        for candidate_id, row in expandable.items()
    }
    low, high = min(mean_values.values()), max(mean_values.values())

    def normalized(value: float) -> float:
        return 0.5 if high == low else (value - low) / (high - low)

    pending_rollouts: dict[str, int] = {}
    all_rows = {row["id"]: row for row in rows}
    for row in rows:
        if row["status"] not in NONTERMINAL_STATUSES or row["parent_id"] is None:
            continue
        parent_id = row["parent_id"]
        while parent_id is not None:
            pending_rollouts[parent_id] = pending_rollouts.get(parent_id, 0) + 1
            parent = all_rows.get(parent_id)
            parent_id = None if parent is None else parent["parent_id"]

    root_visits = int(all_rows[ROOT_CANDIDATE_ID]["mcts_visits"])
    # WU-UCT-style pending visits affect exploration without changing the completed-score mean.
    # The root counts each pending rollout once; summing nodes would count it once per ancestor.
    root_pending = pending_rollouts.get(ROOT_CANDIDATE_ID, 0)
    total_visits = max(1, root_visits + root_pending)
    parents = []
    for candidate_id, row in expandable.items():
        candidate_visits = int(row["mcts_visits"])
        value_sum = float(row["mcts_value_sum"])
        pending = pending_rollouts.get(candidate_id, 0)
        mean_value = mean_values[candidate_id]
        exploitation_value = normalized(mean_value)
        exploration_bonus = math.sqrt(2 * math.log(total_visits + 1) / (candidate_visits + pending))
        parents.append(
            {
                "candidate_id": candidate_id,
                "score": float(row["eval_score"]),
                "visits": candidate_visits,
                "value_sum": value_sum,
                "mean_value": mean_value,
                "exploitation_value": exploitation_value,
                "pending_rollouts": pending,
                "exploration_bonus": exploration_bonus,
                "priority": exploitation_value + exploration_bonus,
            }
        )
    return sorted(parents, key=lambda item: (-item["priority"], -item["score"], item["candidate_id"]))
