import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

from ..error import SparError
from ..lifecycle import CandidateStatus, SessionStatus

SCHEMA_VERSION = 21


class DB:
    def __init__(self, path: Path) -> None:
        self._db = sqlite3.connect(path / "state.sqlite", timeout=5.0)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA busy_timeout = 5000")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._db.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self._db.rollback()
            raise
        else:
            self._db.commit()

    def initialize(self, *, repo_path: str, root_commit: str) -> None:
        self._db.executescript(
            """
            CREATE TABLE candidates (
              id TEXT PRIMARY KEY,
              parent_id TEXT REFERENCES candidates(id),
              commit_sha TEXT,
              worktree_path TEXT,
              hypothesis TEXT NOT NULL,
              instructions TEXT NOT NULL,
              rationale TEXT NOT NULL,
              profiling_question TEXT,
              status TEXT NOT NULL,

              evaluation_json TEXT,
              eval_score REAL GENERATED ALWAYS AS (
                CAST(json_extract(evaluation_json, '$.score') AS REAL)
              ) STORED,
              profiling_json TEXT,
              learnings TEXT,
              decision TEXT,
              decision_reason TEXT,
              error TEXT,

              mcts_visits INTEGER NOT NULL DEFAULT 0,
              mcts_value_sum REAL NOT NULL DEFAULT 0,

              started_at INTEGER NOT NULL,
              completed_at INTEGER
            );

            CREATE TABLE spans (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              candidate_id TEXT NOT NULL REFERENCES candidates(id),
              kind TEXT NOT NULL,
              started_at INTEGER NOT NULL,
              completed_at INTEGER,
              error TEXT
            );

            CREATE INDEX spans_by_candidate
              ON spans(candidate_id, started_at, id);

            CREATE INDEX spans_by_kind
              ON spans(kind, id);

            CREATE TABLE invocations (
              span_id INTEGER PRIMARY KEY REFERENCES spans(id),
              agent TEXT NOT NULL,
              model TEXT,
              effort TEXT,
              prompt TEXT NOT NULL,
              schema_json TEXT NOT NULL,
              response_json TEXT,
              event_path TEXT NOT NULL
            );

            CREATE TABLE session (
              id INTEGER PRIMARY KEY CHECK (id = 1),
              agent_session_id TEXT,
              status TEXT NOT NULL,
              stop_requested INTEGER NOT NULL DEFAULT 0,
              started_at INTEGER,
              completed_at INTEGER,
              stop_reason TEXT
            );
            """
        )
        with self.transaction():
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            timestamp = _now_ms()
            self._db.execute(
                "INSERT INTO session (id, status) VALUES (1, ?)",
                (SessionStatus.IDLE,),
            )
            self._db.execute(
                """
                INSERT INTO candidates (
                  id, parent_id, commit_sha, worktree_path, hypothesis, instructions,
                  rationale, status, started_at
                )
                VALUES ('root', NULL, ?, ?, 'baseline', '', 'baseline candidate', ?, ?)
                """,
                (root_commit, repo_path, CandidateStatus.IMPLEMENTING, timestamp),
            )

    def require_current_schema(self) -> None:
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise SparError("session schema is incompatible; initialize a new session")

    def session(self) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM session WHERE id = 1").fetchone()
        if row is None:
            raise SparError("session row is missing")
        return dict(row)

    def update_session(self, values: dict[str, Any]) -> None:
        allowed = {"agent_session_id", "status", "stop_requested", "started_at", "completed_at", "stop_reason"}
        if not values or not set(values) <= allowed:
            raise SparError("invalid session update")
        assignments = ", ".join(f"{column} = ?" for column in values)
        self._db.execute(
            f"UPDATE session SET {assignments} WHERE id = 1",
            tuple(values.values()),
        )

    def candidate(self, candidate_id: str) -> dict[str, Any]:
        row = self._db.execute(
            "SELECT * FROM candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise SparError(f"unknown candidate: {candidate_id}")
        return dict(row)

    def candidates(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._db.execute("SELECT * FROM candidates ORDER BY started_at, id").fetchall()]

    def candidates_used(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    def insert_candidate(self, values: dict[str, Any], *, maximum: int) -> bool:
        cursor = self._db.execute(
            """
            INSERT INTO candidates (
              id, parent_id, worktree_path, hypothesis, instructions, rationale,
              profiling_question, status, started_at
            )
            SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?
            WHERE (SELECT COUNT(*) FROM candidates) < ?
            """,
            tuple(
                values[key]
                for key in (
                    "id",
                    "parent_id",
                    "worktree_path",
                    "hypothesis",
                    "instructions",
                    "rationale",
                    "profiling_question",
                    "status",
                    "started_at",
                )
            )
            + (maximum,),
        )
        return cursor.rowcount == 1

    def update_candidate(self, candidate_id: str, expected_status: str, values: dict[str, Any]) -> bool:
        allowed = {
            "commit_sha",
            "status",
            "evaluation",
            "profiling",
            "learnings",
            "decision",
            "decision_reason",
            "error",
            "completed_at",
        }
        if not values or not set(values) <= allowed:
            raise SparError("invalid candidate update")
        stored_values = {
            f"{key}_json" if key in {"evaluation", "profiling"} else key: (
                _json_text(value) if key in {"evaluation", "profiling"} else value
            )
            for key, value in values.items()
        }
        assignments = ", ".join(f"{column} = ?" for column in stored_values)
        cursor = self._db.execute(
            f"UPDATE candidates SET {assignments} WHERE id = ? AND status = ?",
            (*stored_values.values(), candidate_id, expected_status),
        )
        return cursor.rowcount == 1

    def increment_mcts(self, candidate_id: str, reward: float) -> None:
        self._db.execute(
            """
            UPDATE candidates
            SET mcts_visits = mcts_visits + 1,
                mcts_value_sum = mcts_value_sum + ?
            WHERE id = ?
            """,
            (reward, candidate_id),
        )

    def spans(self, candidate_id: str | None = None) -> list[dict[str, Any]]:
        if candidate_id is None:
            rows = self._db.execute("SELECT * FROM spans ORDER BY started_at, id").fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM spans WHERE candidate_id = ? ORDER BY started_at, id",
                (candidate_id,),
            ).fetchall()
        invocations = {invocation["span_id"]: invocation for invocation in self._invocations(candidate_id)}
        spans = [_decode_span(row) for row in rows]
        for span in spans:
            span["invocation"] = invocations.get(span["id"])
        return spans

    def _invocations(self, candidate_id: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT invocations.*
            FROM invocations
            JOIN spans ON spans.id = invocations.span_id
        """
        parameters: tuple[str, ...] = ()
        if candidate_id is not None:
            query += " WHERE spans.candidate_id = ?"
            parameters = (candidate_id,)
        query += " ORDER BY spans.started_at, spans.id"
        return [_decode_invocation(row) for row in self._db.execute(query, parameters).fetchall()]

    def count_spans(self, kind: str) -> int:
        return self._db.execute("SELECT COUNT(*) FROM spans WHERE kind = ?", (kind,)).fetchone()[0]

    def _insert_span(self, candidate_id: str, kind: str) -> int:
        cursor = self._db.execute(
            "INSERT INTO spans (candidate_id, kind, started_at) VALUES (?, ?, ?)",
            (candidate_id, kind, _now_ms()),
        )
        return int(cursor.lastrowid)

    def _finish_span(self, span_id: int, *, error: str | None) -> None:
        cursor = self._db.execute(
            """
            UPDATE spans
            SET completed_at = ?, error = ?
            WHERE id = ? AND completed_at IS NULL
            """,
            (_now_ms(), error.strip() if error else None, span_id),
        )
        if cursor.rowcount != 1:
            raise SparError(f"span is already completed or missing: {span_id}")

    def _span(self, span_id: int) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM spans WHERE id = ?", (span_id,)).fetchone()
        if row is None:
            raise SparError(f"unknown span: {span_id}")
        return _decode_span(row)

    def _insert_invocation(self, span_id: int, values: dict[str, Any]) -> None:
        self._db.execute(
            """
            INSERT INTO invocations (
              span_id, agent, model, effort, prompt, schema_json,
              response_json, event_path
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                span_id,
                values["agent"],
                values.get("model"),
                values.get("effort"),
                values["prompt"],
                _json_text(values["schema"]),
                values["event_path"],
            ),
        )

    def update_invocation_response(self, span_id: int, response: dict[str, Any] | None) -> None:
        cursor = self._db.execute(
            """
            UPDATE invocations
            SET response_json = ?
            WHERE span_id = ?
            """,
            (_json_text(response), span_id),
        )
        if cursor.rowcount != 1:
            raise SparError(f"invocation is missing for span: {span_id}")


class Span:
    def __init__(
        self,
        path: Path,
        *,
        candidate_id: str,
        kind: str,
        invocation: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.candidate_id = candidate_id
        self.kind = kind
        self.invocation = invocation
        self.id: int | None = None
        self.record: dict[str, Any] | None = None
        self._completed = False

    def __enter__(self) -> Self:
        if not self.candidate_id.strip() or not self.kind.strip():
            raise SparError("span candidate and kind must not be empty")
        with DB(self.path) as db:
            with db.transaction():
                self.id = db._insert_span(self.candidate_id, self.kind.strip())
                if self.invocation is not None:
                    db._insert_invocation(self.id, self.invocation)
            self.record = db._span(self.id)
        return self

    @contextmanager
    def complete(self, *, error: str | None = None) -> Iterator[DB]:
        if self.id is None or self._completed:
            raise SparError("span is not active")
        with DB(self.path) as db:
            with db.transaction():
                yield db
                db._finish_span(self.id, error=error)
            self.record = db._span(self.id)
        self._completed = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, traceback
        if self._completed:
            return
        error = None if exc is None else str(exc).strip() or type(exc).__name__
        try:
            with self.complete(error=error):
                pass
        except Exception:
            if exc is None:
                raise


def _decode_span(row: sqlite3.Row) -> dict[str, Any]:
    span = dict(row)
    completed_at = span["completed_at"]
    span["elapsed_ms"] = completed_at - span["started_at"] if completed_at is not None else None
    span["success"] = None if completed_at is None else span["error"] is None
    return span


def _decode_invocation(row: sqlite3.Row) -> dict[str, Any]:
    invocation = dict(row)
    invocation["schema"] = json.loads(invocation.pop("schema_json"))
    response_json = invocation.pop("response_json")
    invocation["response"] = json.loads(response_json) if response_json else None
    invocation["events_path"] = invocation.pop("event_path")
    return invocation


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _json_text(value: Any) -> str | None:
    return json.dumps(value, sort_keys=True) if value is not None else None
