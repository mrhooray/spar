import os
import subprocess
from pathlib import Path

UPSTREAM_REPOSITORY = "https://github.com/NVIDIA/SOL-ExecBench.git"
UPSTREAM_COMMIT = "a9fa0804c793d438e70850c33fe34426e66d53dd"

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = Path(__file__).resolve().parent
UPSTREAM_ROOT = EVAL_ROOT / ".upstream"
PATCH_PATH = EVAL_ROOT / "upstream.patch"
SERVICE_PATH = EVAL_ROOT / "service.py"


def main() -> None:
    prepare_upstream()
    subprocess.run(
        ["uv", "run", "modal", "deploy", str(SERVICE_PATH)],
        cwd=ROOT,
        env=os.environ,
        check=True,
    )


def prepare_upstream() -> None:
    if not (UPSTREAM_ROOT / ".git").is_dir():
        if UPSTREAM_ROOT.exists() and any(UPSTREAM_ROOT.iterdir()):
            raise RuntimeError(f"upstream path is not an empty Git repository: {UPSTREAM_ROOT}")
        UPSTREAM_ROOT.mkdir(parents=True, exist_ok=True)
        run_git("init")
        run_git("remote", "add", "origin", UPSTREAM_REPOSITORY)
        run_git("fetch", "--depth=1", "origin", UPSTREAM_COMMIT)
        run_git("checkout", "--detach", "FETCH_HEAD")

    actual_commit = run_git("rev-parse", "HEAD", capture_output=True).stdout.strip()
    if actual_commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream checkout is at {actual_commit}; expected {UPSTREAM_COMMIT}")

    staged = run_git("diff", "--cached", "--quiet", check=False)
    if staged.returncode:
        raise RuntimeError("upstream checkout contains staged changes")

    status = run_git("status", "--porcelain", "--untracked-files=all", capture_output=True).stdout
    current_patch = run_git("diff", "--no-ext-diff", "--binary", "HEAD", capture_output=True).stdout
    expected_patch = PATCH_PATH.read_text()

    if not status:
        run_git("apply", "--check", str(PATCH_PATH))
        run_git("apply", str(PATCH_PATH))
        current_patch = run_git("diff", "--no-ext-diff", "--binary", "HEAD", capture_output=True).stdout
    elif current_patch != expected_patch:
        raise RuntimeError("upstream checkout contains changes other than the expected patch")

    if current_patch != expected_patch:
        raise RuntimeError("applied upstream patch does not match upstream.patch")


def run_git(*args: str, capture_output: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(UPSTREAM_ROOT), *args],
        capture_output=capture_output,
        text=True,
        check=check,
    )


if __name__ == "__main__":
    main()
