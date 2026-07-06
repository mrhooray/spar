from pathlib import Path
import os
import subprocess

from .errors import SparError


def run_git(
    cwd: Path,
    args: list[str],
    *,
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if check and result.returncode != 0:
        raise SparError(result.stderr.strip() or f"git command failed: {args}")
    return result


def run_command(
    cwd: Path,
    command: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SparError(f"command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SparError(f"command timed out after {timeout}s: {command}") from exc
