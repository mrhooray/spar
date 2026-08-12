from pathlib import Path
from typing import Any

from ..error import SparError


def session_dir(repo: Path, session_name: str) -> Path:
    if "/" in session_name or session_name in {"", ".", ".."}:
        raise SparError("session name must be a single path segment")
    return repo / ".spar" / session_name


def require_session_dir(repo: Path, session_name: str) -> Path:
    path = session_dir(repo, session_name)
    if not path.exists():
        raise SparError(f"session does not exist: {session_name}")
    return path


def candidate_artifact_dir(path: Path, candidate_id: str) -> Path:
    return path / "artifacts" / "candidates" / candidate_id


def artifact_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        {"path": str(item.relative_to(path)), "bytes": item.stat().st_size}
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    ]
