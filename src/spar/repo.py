from pathlib import Path
from typing import Any
import json

from .errors import SparError
from .process import run_git


def repo_root() -> Path:
    result = run_git(Path.cwd(), ["rev-parse", "--path-format=absolute", "--git-common-dir"], check=False)
    if result.returncode != 0:
        raise SparError("must be run from inside a Git repository")
    return Path(result.stdout.strip()).parent


def session_dir(repo: Path, session_name: str) -> Path:
    if "/" in session_name or session_name in {"", ".", ".."}:
        raise SparError("session name must be a single path segment")
    return repo / ".spar" / session_name


def existing_session_dir(repo: Path, session_name: str) -> Path:
    path = session_dir(repo, session_name)
    if not path.exists():
        raise SparError(f"session does not exist: {session_name}")
    return path


def ensure_info_exclude(repo: Path) -> None:
    exclude = repo / ".git" / "info" / "exclude"
    text = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if ".spar/" not in text.splitlines():
        with exclude.open("a", encoding="utf-8") as file:
            if text and not text.endswith("\n"):
                file.write("\n")
            file.write(".spar/\n")


def candidate_artifact_dir(path: Path, candidate_id: str) -> Path:
    return path / "artifacts" / "candidates" / candidate_id


def artifact_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        {"path": str(item.relative_to(path)), "bytes": item.stat().st_size}
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    ]


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SparError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SparError(f"{label} must contain valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SparError(f"{label} must contain a JSON object: {path}")
    return payload


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
