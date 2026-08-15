from pathlib import Path

import pytest

from spar.error import SparError
from spar.storage.db import DB, Span


def initialize_session(path: Path) -> None:
    path.mkdir()
    with DB(path) as db:
        db.initialize(repo_path="/repo", root_commit="root-sha")


def test_transaction_rolls_back_on_error(tmp_path: Path) -> None:
    initialize_session(tmp_path / "session")

    with DB(tmp_path / "session") as db:
        with pytest.raises(RuntimeError, match="abort"), db.transaction():
            db.update_session({"status": "running"})
            raise RuntimeError("abort")

        assert db.session()["status"] == "idle"


def test_agent_session_id_is_persisted(tmp_path: Path) -> None:
    initialize_session(tmp_path / "session")

    with DB(tmp_path / "session") as db, db.transaction():
        assert db.session()["agent_session_id"] is None
        db.update_session({"agent_session_id": "thread-1"})

    with DB(tmp_path / "session") as db:
        assert db.session()["agent_session_id"] == "thread-1"


def test_span_records_an_error_when_work_fails(tmp_path: Path) -> None:
    initialize_session(tmp_path / "session")

    with (
        pytest.raises(RuntimeError, match="span failed"),
        Span(tmp_path / "session", candidate_id="root", kind="unit") as span,
        span.complete(),
    ):
        raise RuntimeError("span failed")

    with DB(tmp_path / "session") as db:
        record = db.spans("root")[0]
        assert record["kind"] == "unit"
        assert record["success"] is False
        assert record["error"] == "span failed"


def test_span_cannot_be_completed_twice(tmp_path: Path) -> None:
    initialize_session(tmp_path / "session")

    with Span(tmp_path / "session", candidate_id="root", kind="unit") as span:
        with span.complete():
            pass
        with pytest.raises(SparError, match="span is not active"), span.complete():
            pass
