from pathlib import Path
from typing import Any
import json
import time

from .errors import ProgramUnchangedError, SparError
from .process import run_git

_PROGRAM_UNCHANGED_MARKER = "SPAR_PROGRAM_UNCHANGED"


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


def commit_candidate_workspace(
    workspace: Path, candidate_id: str, parent_commit: str
) -> dict[str, Any]:
    started_at = time.time_ns() // 1_000_000
    started_monotonic = time.monotonic_ns()
    if run_git(workspace, ["rev-parse", "HEAD"]).stdout.strip() != parent_commit:
        raise SparError("implementation agent must not create commits")
    porcelain = run_git(
        workspace,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    ).stdout
    paths = _porcelain_paths(porcelain)
    if not paths:
        raise ProgramUnchangedError("candidate workspace is unchanged")
    run_git(workspace, ["--literal-pathspecs", "add", "-A", "--", *paths])
    commit_args = ["commit", "-m", f"spar candidate {candidate_id}"]
    commit = run_git(workspace, commit_args, check=False)
    if commit.returncode != 0:
        if any(
            line.strip() == _PROGRAM_UNCHANGED_MARKER
            for line in commit.stderr.splitlines()
        ):
            raise ProgramUnchangedError("candidate program is unchanged")
        raise SparError(commit.stderr.strip() or f"git command failed: {commit_args}")
    commit_sha = run_git(workspace, ["rev-parse", "HEAD"]).stdout.strip()
    if run_git(
        workspace,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    ).stdout:
        raise SparError("candidate worktree is not clean after the orchestrated commit")
    return {
        "commit_sha": commit_sha,
        "changed_paths": paths,
        "started_at": started_at,
        "completed_at": time.time_ns() // 1_000_000,
        "elapsed_ms": (time.monotonic_ns() - started_monotonic) // 1_000_000,
    }


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


def _porcelain_paths(porcelain: str) -> list[str]:
    records = porcelain.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(records) and records[index]:
        record = records[index]
        if len(record) < 4:
            raise SparError("candidate worktree returned malformed Git status")
        status = record[:2]
        paths.append(record[3:])
        if "R" in status or "C" in status:
            index += 1
            if index >= len(records) or not records[index]:
                raise SparError("candidate worktree returned malformed rename status")
            paths.append(records[index])
        index += 1
    return list(dict.fromkeys(paths))
