from enum import StrEnum


class SessionStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class CandidateStatus(StrEnum):
    IMPLEMENTING = "implementing"
    EVALUATING = "evaluating"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class Decision(StrEnum):
    KEEP = "keep"
    DISCARD = "discard"


ROOT_CANDIDATE_ID = "root"
NONTERMINAL_CANDIDATE_STATUSES = frozenset(
    {
        CandidateStatus.IMPLEMENTING,
        CandidateStatus.EVALUATING,
        CandidateStatus.REFLECTING,
    }
)
