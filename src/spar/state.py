from pathlib import Path
from typing import Any
import math
import sqlite3
import time
import uuid

from .errors import SparError
from .mcts import backpropagate, top_parents


SCHEMA_VERSION = 10
STATUS_IMPLEMENTING = "implementing"
STATUS_EVALUATING = "evaluating"
STATUS_REFLECTING = "reflecting"
STATUS_FINALIZING = "finalizing"
STATUS_COMPLETED = "completed"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"
STATUS_REJECTED = "rejected"
DECISION_KEEP = "keep"
DECISION_DISCARD = "discard"
WORKER_STATUS = STATUS_IMPLEMENTING
NONTERMINAL_STATUSES = (
    STATUS_IMPLEMENTING,
    STATUS_EVALUATING,
    STATUS_REFLECTING,
    STATUS_FINALIZING,
)
ROOT_CANDIDATE_ID = "root"


class SessionState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._db = sqlite3.connect(path / "state.sqlite", timeout=5.0)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout = 5000")

    def __enter__(self) -> SessionState:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._db.close()

    def initialize(self, *, repo_path: str, root_commit: str) -> None:
        self._db.executescript(
            """
            CREATE TABLE candidates (
              id TEXT PRIMARY KEY,
              parent_id TEXT REFERENCES candidates(id),
              commit_sha TEXT,
              workspace_path TEXT,
              hypothesis TEXT NOT NULL,
              instructions TEXT NOT NULL,
              rationale TEXT NOT NULL,
              status TEXT NOT NULL,

              eval_score REAL,
              eval_summary TEXT,
              decision TEXT,
              decision_reason TEXT,
              error TEXT,

              mcts_visits INTEGER NOT NULL DEFAULT 0,
              mcts_value_sum REAL NOT NULL DEFAULT 0,

              created_at INTEGER NOT NULL,
              completed_at INTEGER,
              updated_at INTEGER NOT NULL
            );

            CREATE TABLE operations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              candidate_id TEXT REFERENCES candidates(id),
              operation TEXT NOT NULL,
              started_at INTEGER NOT NULL,
              completed_at INTEGER NOT NULL,
              elapsed_ms INTEGER NOT NULL,
              success INTEGER NOT NULL,
              error TEXT
            );
            """
        )
        self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        timestamp = _now_ms()
        self._db.execute(
            """
            INSERT INTO candidates (
              id, parent_id, commit_sha, workspace_path, hypothesis, instructions,
              rationale, status, created_at, updated_at
            )
            VALUES ('root', NULL, ?, ?, 'baseline', '', 'baseline candidate', ?, ?, ?)
            """,
            (root_commit, repo_path, STATUS_IMPLEMENTING, timestamp, timestamp),
        )
        self._db.commit()

    def require_current_schema(self) -> None:
        try:
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.OperationalError as exc:
            raise SparError("session schema is incompatible; initialize a new session") from exc
        if version != SCHEMA_VERSION:
            raise SparError("session schema is incompatible; initialize a new session")

    def start_candidate(
        self,
        *,
        parent_id: str,
        hypothesis: str,
        instructions: str,
        rationale: str,
        max_candidates: int,
        max_parallel: int,
        workspace_root: Path | None = None,
    ) -> dict[str, Any]:
        for name, value in (("hypothesis", hypothesis), ("instructions", instructions), ("rationale", rationale)):
            if not value.strip():
                raise SparError(f"candidate {name} must not be empty")
        candidate_id = f"cand_{uuid.uuid4().hex[:12]}"
        workspace_path = str((workspace_root or self.path / "worktrees") / candidate_id)
        timestamp = _now_ms()
        try:
            self._db.execute("BEGIN IMMEDIATE")
            if self._counted_candidate_count() >= max_candidates:
                raise SparError(f"candidate budget exhausted: {max_candidates}")
            worker_count = self._db.execute(
                "SELECT COUNT(*) FROM candidates WHERE status = ?", (WORKER_STATUS,)
            ).fetchone()[0]
            if worker_count >= max_parallel:
                raise SparError(f"parallel candidate limit reached: {max_parallel}")
            parent = self._db.execute("SELECT * FROM candidates WHERE id = ?", (parent_id,)).fetchone()
            if parent is None:
                raise SparError(f"unknown parent candidate: {parent_id}")
            if parent["status"] != STATUS_COMPLETED or parent["decision"] == DECISION_DISCARD or not parent["commit_sha"]:
                raise SparError(f"parent candidate is not expandable: {parent_id}")
            self._db.execute(
                """
                INSERT INTO candidates (
                  id, parent_id, workspace_path, hypothesis, instructions, rationale,
                  status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    parent_id,
                    workspace_path,
                    hypothesis.strip(),
                    instructions.strip(),
                    rationale.strip(),
                    STATUS_IMPLEMENTING,
                    timestamp,
                    timestamp,
                ),
            )
            self._db.commit()
        except Exception:
            if self._db.in_transaction:
                self._db.rollback()
            raise
        return self.candidate(candidate_id)

    def begin_evaluation(self, candidate_id: str, commit_sha: str) -> None:
        cursor = self._db.execute(
            """
            UPDATE candidates
            SET commit_sha = ?, status = ?, updated_at = ?
            WHERE id = ? AND status = ?
            """,
            (commit_sha, STATUS_EVALUATING, _now_ms(), candidate_id, STATUS_IMPLEMENTING),
        )
        if cursor.rowcount != 1:
            raise SparError(f"candidate is not implementing: {candidate_id}")
        self._db.commit()

    def record_evaluation(self, candidate_id: str, evaluation: dict[str, Any]) -> None:
        score = evaluation_score(evaluation)
        cursor = self._db.execute(
            """
            UPDATE candidates
            SET status = ?, eval_score = ?, updated_at = ?
            WHERE id = ? AND status = ?
            """,
            (STATUS_REFLECTING, score, _now_ms(), candidate_id, STATUS_EVALUATING),
        )
        if cursor.rowcount != 1:
            raise SparError(f"candidate is not evaluating: {candidate_id}")
        self._db.commit()

    def begin_finalization(
        self, candidate_id: str, *, summary: str, decision: str, decision_reason: str
    ) -> dict[str, Any]:
        candidate = self._candidate_row(candidate_id)
        if decision not in (DECISION_KEEP, DECISION_DISCARD):
            raise SparError("candidate decision must be keep or discard")
        if not summary.strip() or not decision_reason.strip():
            raise SparError("candidate summary and decision reason must not be empty")
        if candidate["status"] == STATUS_FINALIZING:
            if (
                candidate["eval_summary"] != summary.strip()
                or candidate["decision"] != decision
                or candidate["decision_reason"] != decision_reason.strip()
            ):
                raise SparError("candidate finalization data does not match its recorded request")
            return candidate
        if candidate["status"] != STATUS_REFLECTING:
            raise SparError(f"candidate is not reflecting: {candidate_id}")

        cursor = self._db.execute(
            """
            UPDATE candidates
            SET status = ?, eval_summary = ?, decision = ?, decision_reason = ?, updated_at = ?
            WHERE id = ? AND status = ?
            """,
            (
                STATUS_FINALIZING,
                summary.strip(),
                decision,
                decision_reason.strip(),
                _now_ms(),
                candidate_id,
                STATUS_REFLECTING,
            ),
        )
        if cursor.rowcount != 1:
            raise SparError(f"candidate is not reflecting: {candidate_id}")
        self._db.commit()
        return self.candidate(candidate_id)

    def complete_candidate(self, candidate_id: str) -> dict[str, Any]:
        try:
            self._db.execute("BEGIN IMMEDIATE")
            candidate = self._candidate_row(candidate_id)
            if candidate["status"] != STATUS_FINALIZING:
                raise SparError(f"candidate is not finalizing: {candidate_id}")
            timestamp = _now_ms()
            cursor = self._db.execute(
                """
                UPDATE candidates
                SET status = ?, completed_at = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (STATUS_COMPLETED, timestamp, timestamp, candidate_id, STATUS_FINALIZING),
            )
            if cursor.rowcount != 1:
                raise SparError(f"candidate is not finalizing: {candidate_id}")
            backpropagate(self._db, candidate_id, float(candidate["eval_score"]))
            self._db.commit()
        except Exception:
            if self._db.in_transaction:
                self._db.rollback()
            raise
        return self.candidate(candidate_id)

    def fail_candidate(self, candidate_id: str, error: str, *, interrupted: bool) -> dict[str, Any]:
        if not error.strip():
            raise SparError("candidate error must not be empty")
        if candidate_id == ROOT_CANDIDATE_ID:
            raise SparError("root candidate cannot be failed; retry baseline evaluation")
        status = STATUS_INTERRUPTED if interrupted else STATUS_FAILED
        timestamp = _now_ms()
        cursor = self._db.execute(
            """
            UPDATE candidates
            SET status = ?, error = ?, completed_at = ?, updated_at = ?
            WHERE id = ? AND status IN (?, ?, ?, ?)
            """,
            (status, error.strip(), timestamp, timestamp, candidate_id, *NONTERMINAL_STATUSES),
        )
        if cursor.rowcount != 1:
            raise SparError(f"candidate is not unfinished: {candidate_id}")
        self._db.commit()
        return self.candidate(candidate_id)

    def reject_unadmitted_candidate(
        self, candidate_id: str, reason: str
    ) -> dict[str, Any]:
        if not reason.strip():
            raise SparError("candidate rejection reason must not be empty")
        if candidate_id == ROOT_CANDIDATE_ID:
            raise SparError("root candidate cannot be rejected")
        timestamp = _now_ms()
        cursor = self._db.execute(
            """
            UPDATE candidates
            SET status = ?, error = ?, completed_at = ?, updated_at = ?
            WHERE id = ? AND status = ? AND commit_sha IS NULL AND eval_score IS NULL
            """,
            (
                STATUS_REJECTED,
                reason.strip(),
                timestamp,
                timestamp,
                candidate_id,
                STATUS_IMPLEMENTING,
            ),
        )
        if cursor.rowcount != 1:
            raise SparError(
                f"candidate is not an unadmitted implementation: {candidate_id}"
            )
        self._db.commit()
        return self.candidate(candidate_id)

    def candidate(self, candidate_id: str) -> dict[str, Any]:
        return decode_candidate(self._candidate_row(candidate_id), self.path)

    def record_operation(
        self,
        *,
        candidate_id: str | None,
        operation: str,
        started_at: int,
        completed_at: int,
        elapsed_ms: int,
        success: bool,
        error: str | None = None,
    ) -> dict[str, Any]:
        if not operation.strip():
            raise SparError("operation name must not be empty")
        self._db.execute(
            """
            INSERT INTO operations (
              candidate_id, operation, started_at, completed_at, elapsed_ms, success, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                operation.strip(),
                started_at,
                completed_at,
                elapsed_ms,
                int(success),
                error.strip() if error else None,
            ),
        )
        self._db.commit()
        row = self._db.execute("SELECT * FROM operations WHERE id = last_insert_rowid()").fetchone()
        return decode_operation(row)

    def operations(self, candidate_id: str | None = None) -> list[dict[str, Any]]:
        if candidate_id is None:
            rows = self._db.execute("SELECT * FROM operations ORDER BY id").fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM operations WHERE candidate_id = ? ORDER BY id", (candidate_id,)
            ).fetchall()
        return [decode_operation(row) for row in rows]

    def status_snapshot(self, max_candidates: int) -> dict[str, Any]:
        rows = self._db.execute("SELECT * FROM candidates ORDER BY created_at, id").fetchall()
        candidates = [decode_candidate(row, self.path) for row in rows]
        budget_used = self._counted_candidate_count()
        return {
            "candidate_budget": {
                "maximum": max_candidates,
                "used": budget_used,
                "remaining": max(0, max_candidates - budget_used),
            },
            "counts": {
                row["status"]: row["count"]
                for row in self._db.execute(
                    "SELECT status, COUNT(*) AS count FROM candidates GROUP BY status ORDER BY status"
                ).fetchall()
            },
            "best_candidate": best_candidate(candidates),
            "active_candidates": [candidate for candidate in candidates if candidate["status"] == WORKER_STATUS],
            "candidates": [candidate_summary(candidate) for candidate in candidates],
            "operations": self.operations(),
        }

    def parent_candidates(self, k: int) -> list[dict[str, Any]]:
        rows = self._db.execute("SELECT * FROM candidates ORDER BY created_at, id").fetchall()
        return top_parents(rows)[:k]

    def inspect_candidate(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.candidate(candidate_id)
        rows = self._db.execute("SELECT * FROM candidates ORDER BY created_at, id").fetchall()
        by_id = {row["id"]: row for row in rows}
        ancestors: list[dict[str, Any]] = []
        parent_id = candidate["parent_id"]
        while parent_id is not None:
            parent = by_id.get(parent_id)
            if parent is None:
                break
            ancestors.append(decode_candidate(parent, self.path))
            parent_id = parent["parent_id"]
        suggestions = {item["candidate_id"]: item for item in top_parents(rows)}
        return {
            "candidate": candidate,
            "ancestors": ancestors,
            "children": [decode_candidate(row, self.path) for row in rows if row["parent_id"] == candidate_id],
            "mcts": suggestions.get(candidate_id),
            "operations": self.operations(candidate_id),
        }

    def _candidate_row(self, candidate_id: str) -> sqlite3.Row:
        row = self._db.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if row is None:
            raise SparError(f"unknown candidate: {candidate_id}")
        return row

    def _counted_candidate_count(self) -> int:
        return self._db.execute(
            "SELECT COUNT(*) FROM candidates WHERE status != ?", (STATUS_REJECTED,)
        ).fetchone()[0]


def evaluation_score(payload: dict[str, Any]) -> float:
    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(score):
        raise SparError("evaluation result must include a finite numeric score")
    return float(score)


def decode_candidate(row: sqlite3.Row, session_path: Path) -> dict[str, Any]:
    candidate = dict(row)
    candidate["artifact_dir"] = str(session_path / "artifacts" / "candidates" / candidate["id"])
    return candidate


def decode_operation(row: sqlite3.Row) -> dict[str, Any]:
    operation = dict(row)
    operation["success"] = bool(operation["success"])
    return operation


def candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "id",
            "parent_id",
            "commit_sha",
            "hypothesis",
            "status",
            "eval_score",
            "eval_summary",
            "decision",
            "decision_reason",
            "error",
            "updated_at",
        )
    }


def best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    kept = [
        candidate
        for candidate in candidates
        if candidate["status"] == STATUS_COMPLETED
        and candidate["decision"] != DECISION_DISCARD
        and candidate["eval_score"] is not None
    ]
    return max(kept, key=lambda candidate: candidate["eval_score"], default=None)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000
