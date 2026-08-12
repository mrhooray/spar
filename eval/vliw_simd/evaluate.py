import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

PINNED_COMMIT = "5452f74bd977807ac2e74f3d29432b9df6f25197"
ALLOWED_PATHS = {"perf_takehome.py"}


def main() -> None:
    repo = Path.cwd()
    changed_paths = _changed_paths(repo)
    unexpected = sorted(changed_paths - ALLOWED_PATHS)
    if unexpected:
        print(
            json.dumps(
                {
                    "score": 0.0,
                    "correct": False,
                    "changed_paths": sorted(changed_paths),
                }
            )
        )
        return

    tests_dir = repo / "tests"
    sys.path.insert(0, str(tests_dir))
    spec = importlib.util.spec_from_file_location("submission_tests", tests_dir / "submission_tests.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load submission tests")
    submission_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submission_tests)

    diagnostics = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(submission_tests.CorrectnessTests)
    with contextlib.redirect_stdout(diagnostics):
        result = unittest.TextTestRunner(stream=diagnostics).run(suite)
        submission_tests.cycles.cache_clear()
        cycles = submission_tests.cycles() if result.wasSuccessful() else None

    correct = result.wasSuccessful() and isinstance(cycles, int)
    evaluation = {
        "score": submission_tests.BASELINE / cycles if correct else 0.0,
        "correct": correct,
        "cycles": cycles,
        "changed_paths": sorted(changed_paths),
        "correctness_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }
    if not correct or result.failures or result.errors:
        evaluation["diagnostics"] = diagnostics.getvalue()
    print(json.dumps(evaluation))


def _changed_paths(repo: Path) -> set[str]:
    changed = set(
        subprocess.run(
            ["git", "diff", "--name-only", PINNED_COMMIT, "--"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    changed.update(
        subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    return changed


if __name__ == "__main__":
    main()
