from pathlib import Path
import json
import os
import re
import subprocess
import sys


def main() -> None:
    result = subprocess.run(
        [sys.executable, "perf_takehome.py", "Tests.test_kernel_trace"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    match = re.search(r"CYCLES:\s+(\d+)", result.stdout)
    profiling_dir = Path(os.environ["SPAR_PROFILING_DIR"])
    profiling_dir.mkdir(parents=True, exist_ok=True)
    trace = Path("trace.json")
    if trace.exists():
        trace.replace(profiling_dir / trace.name)
    print(
        json.dumps(
            {
                "cycles": int(match.group(1)) if match else None,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
