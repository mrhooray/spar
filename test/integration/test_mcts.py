import math

import pytest

from spar.error import SparError
from spar.lifecycle import Decision
from spar.operation import candidate as candidate_ops
from spar.operation import session as session_ops
from spar.storage.db import DB

from .helpers import (
    chdir,
    commit_file,
    complete_root,
    create_completed_child,
    initialize_session,
    temp_git_repo,
)


def test_backpropagates_on_completion_and_accounts_for_active_work() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo, max_candidates=5)
        complete_root(repo, session, score=-100.0)

        kept = create_completed_child(
            repo,
            session,
            score=-10.0,
            decision=Decision.KEEP,
            suffix="kept",
        )
        with chdir(repo):
            before_discard = session_ops.top("demo", k=3)
        discarded = create_completed_child(
            repo,
            session,
            score=1000.0,
            decision=Decision.DISCARD,
            suffix="discarded",
            parent_id=kept,
        )
        with chdir(repo):
            active = candidate_ops.start(
                "demo",
                parent_id=kept,
                hypothesis="follow up",
                instructions="try another increment",
                rationale="kept parent is strongest",
            )
            ranked = session_ops.top("demo", k=3)
            repeated = session_ops.top("demo", k=3)
            limited = session_ops.top("demo", k=1)

        before = {item["candidate_id"]: item for item in before_discard["candidates"]}
        suggestions = {item["candidate_id"]: item for item in ranked["candidates"]}
        assert repeated == ranked
        assert discarded not in suggestions
        assert suggestions["root"]["visits"] == before["root"]["visits"] + 1
        assert suggestions["root"]["value_sum"] == before["root"]["value_sum"] + 1000.0
        assert suggestions[kept]["mean_value"] == 495.0
        assert suggestions[kept]["exploitation_value"] == 1.0
        assert suggestions[kept]["pending_rollouts"] == 1
        assert suggestions[kept]["exploration_bonus"] == pytest.approx(math.sqrt(2 * math.log(5) / 3))
        assert len(limited["candidates"]) == 1
        assert active["candidate"]["parent_id"] == kept

        with DB(session) as db:
            root = db.candidate("root")
            kept_candidate = db.candidate(kept)
            assert (root["mcts_visits"], root["mcts_value_sum"]) == (3, 890.0)
            assert (kept_candidate["mcts_visits"], kept_candidate["mcts_value_sum"]) == (2, 990.0)
            assert (
                db.candidate(discarded)["mcts_visits"],
                db.candidate(discarded)["mcts_value_sum"],
            ) == (1, 1000.0)
        with chdir(repo), pytest.raises(SparError, match="candidate is not ready for completion"):
            candidate_ops.complete(
                "demo",
                discarded,
                learnings="already completed",
                decision=Decision.DISCARD,
                decision_reason="already completed",
            )
        with DB(session) as db:
            root = db.candidate("root")
            assert (root["mcts_visits"], root["mcts_value_sum"]) == (3, 890.0)


def test_prefers_the_higher_mean_score_when_visits_are_equal() -> None:
    with temp_git_repo() as repo:
        commit_file(repo, "value.txt", "0\n", "initial")
        session = initialize_session(repo, max_candidates=3)
        complete_root(repo, session, score=0.0)
        lower = create_completed_child(
            repo,
            session,
            score=10.0,
            decision=Decision.KEEP,
            suffix="lower",
        )
        higher = create_completed_child(
            repo,
            session,
            score=20.0,
            decision=Decision.KEEP,
            suffix="higher",
        )

        with chdir(repo):
            ranked = session_ops.top("demo", k=3)["candidates"]

        suggestions = {item["candidate_id"]: item for item in ranked}
        assert suggestions[higher]["visits"] == suggestions[lower]["visits"]
        assert suggestions[higher]["mean_value"] > suggestions[lower]["mean_value"]
        assert suggestions[higher]["exploitation_value"] > suggestions[lower]["exploitation_value"]
        assert suggestions[higher]["priority"] > suggestions[lower]["priority"]
