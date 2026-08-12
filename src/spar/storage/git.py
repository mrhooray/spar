import subprocess
from pathlib import Path

from ..error import SparError

_PROGRAM_UNCHANGED_MARKER = "SPAR_PROGRAM_UNCHANGED"


def repo_root() -> Path:
    result = _run(Path.cwd(), ["rev-parse", "--path-format=absolute", "--git-common-dir"], check=False)
    if result.returncode != 0:
        raise SparError("must be run from inside a Git repository")
    return Path(result.stdout.strip()).parent


def head(path: Path) -> str:
    return _run(path, ["rev-parse", "HEAD"]).stdout.strip()


def valid_ref(repo: Path, ref: str) -> bool:
    return _run(repo, ["check-ref-format", ref], check=False).returncode == 0


def status(path: Path, *, include_untracked: bool) -> str:
    untracked = "all" if include_untracked else "no"
    return _run(path, ["status", "--short", f"--untracked-files={untracked}"]).stdout


def add_worktree(repo: Path, workspace: Path, commit: str) -> None:
    _run(repo, ["worktree", "add", "--detach", str(workspace), commit])


def update_ref(repo: Path, ref: str, commit: str) -> None:
    _run(repo, ["update-ref", ref, commit])


def commit_exists(repo: Path, commit: str) -> bool:
    return _run(repo, ["cat-file", "-e", f"{commit}^{{commit}}"], check=False).returncode == 0


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        _run(
            repo,
            ["merge-base", "--is-ancestor", ancestor, descendant],
            check=False,
        ).returncode
        == 0
    )


def ensure_info_exclude(repo: Path) -> None:
    exclude = repo / ".git" / "info" / "exclude"
    text = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if ".spar/" not in text.splitlines():
        with exclude.open("a", encoding="utf-8") as file:
            if text and not text.endswith("\n"):
                file.write("\n")
            file.write(".spar/\n")


def commit_candidate_workspace(workspace: Path, candidate_id: str, parent_commit: str) -> None:
    if head(workspace) != parent_commit:
        raise SparError("implementation agent must not create commits")
    if not status(workspace, include_untracked=True).strip():
        raise SparError("candidate workspace is unchanged")
    _run(workspace, ["add", "-A"])
    commit_args = ["commit", "-m", f"spar candidate {candidate_id}"]
    commit = _run(workspace, commit_args, check=False)
    if commit.returncode != 0:
        if any(line.strip() == _PROGRAM_UNCHANGED_MARKER for line in commit.stderr.splitlines()):
            raise SparError("candidate program is unchanged")
        raise SparError(commit.stderr.strip() or f"git command failed: {commit_args}")
    if _run(
        workspace,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    ).stdout:
        raise SparError("candidate worktree is not clean after the orchestrated commit")


def _run(
    cwd: Path,
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if check and result.returncode != 0:
        raise SparError(result.stderr.strip() or f"git command failed: {args}")
    return result
