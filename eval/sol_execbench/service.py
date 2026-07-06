import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import modal

BENCHMARK_ROOT = Path(__file__).parent / ".upstream"
DATA_ROOT = "/sol-execbench/data/benchmark"
TRACE_ROOT = "/sol-execbench/data/flashinfer-trace"

IMAGE = modal.Image.from_dockerfile(
    BENCHMARK_ROOT / "docker/Dockerfile",
    context_dir=BENCHMARK_ROOT,
    build_args={"HOST_UID": "0", "HOST_GID": "0", "HOST_USER": "root"},
).run_commands(
    "uv pip install --python /venv fastapi",
    "cd /sol-execbench && python scripts/download_solexecbench.py",
    f"mkdir -p {TRACE_ROOT}",
)

app = modal.App("spar-sol-execbench")

TRIALS = 3
MAX_ATTEMPTS = 16
TIMING_FAILURE_PREFIX = "Timing failed: Expected kernel activity sequence not found"


def workload_key(workload: dict) -> str:
    return json.dumps(workload["axes"], sort_keys=True, separators=(",", ":"))


def has_valid_timing(trace: dict) -> bool:
    evaluation = trace["evaluation"]
    performance = evaluation.get("performance")
    return (
        evaluation["status"] == "PASSED"
        and isinstance(performance, dict)
        and isinstance(performance.get("latency_ms"), (int, float))
    )


@app.function(
    image=IMAGE,
    gpu="B200",
    timeout=1800,
    min_containers=0,
    max_containers=3,
    scaledown_window=60,
    env={
        "FLASHINFER_TRACE_DIR": TRACE_ROOT,
        "SOL_EXECBENCH_ROOT": "/sol-execbench",
    },
)
@modal.concurrent(max_inputs=1)
@modal.fastapi_endpoint(method="POST", label="spar-sol-execbench")
def submit(payload: dict) -> dict:
    problem = payload.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("payload.problem is required")
    problem = problem.strip().removeprefix("data/benchmark/")
    problem_path = Path(problem)
    if problem_path.is_absolute() or ".." in problem_path.parts:
        raise ValueError("payload.problem must be a relative benchmark problem path")
    problem_dir = Path(DATA_ROOT) / problem_path
    if not problem_dir.is_dir():
        raise ValueError(f"benchmark problem does not exist: {problem}")
    workloads = [
        json.loads(line)
        for line in (problem_dir / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    workload_count = len(workloads)
    definition_name = payload.get("definition", problem_dir.name)
    iterations = int(payload.get("iterations", 50))
    warmup_runs = int(payload.get("warmup_runs", 10))
    config = {
        "warmup_runs": warmup_runs,
        "iterations": iterations,
        # Modal workers do not permit GPU clock locking.
        "lock_clocks": False,
        "benchmark_reference": False,
        "seed": 200,
        "trials": TRIALS,
        "max_attempts": MAX_ATTEMPTS,
    }

    with tempfile.TemporaryDirectory() as tmp:
        solution_path = Path(tmp) / "solution.json"
        config_path = Path(tmp) / "bench.json"
        workload_path = Path(tmp) / "workload.jsonl"
        solution = payload.get("solution")
        if solution is None:
            solution = {
                "name": "custom_submission",
                "definition": definition_name,
                "author": "spar",
                "description": "SPAR submission",
                "spec": {
                    "languages": ["triton"],
                    "target_hardware": ["B200"],
                    "entry_point": "solution.py::run",
                    "destination_passing_style": False,
                    "dependencies": ["torch", "triton >= 2.3"],
                },
                "sources": [{"path": "solution.py", "content": payload["source"]}],
            }
        solution_path.write_text(json.dumps(solution))
        benchmark_config = {
            key: value
            for key, value in config.items()
            if key not in {"trials", "max_attempts"}
        }
        config_path.write_text(json.dumps(benchmark_config))

        valid_timings = defaultdict(int)
        terminal_workloads = set()
        traces = []
        attempts = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            pending = [
                workload
                for workload in workloads
                if valid_timings[workload_key(workload)] < TRIALS
                and workload_key(workload) not in terminal_workloads
            ]
            if not pending:
                break

            workload_path.write_text(
                "\n".join(json.dumps(workload) for workload in pending) + "\n"
            )
            result = subprocess.run(
                [
                    "sol-execbench",
                    "--definition",
                    str(problem_dir / "definition.json"),
                    "--workload",
                    str(workload_path),
                    "--solution",
                    str(solution_path),
                    "--config",
                    str(config_path),
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=1740,
                check=False,
            )
            attempt_traces = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.strip().startswith("{")
            ]
            for trace in attempt_traces:
                key = workload_key(trace["workload"])
                if has_valid_timing(trace):
                    valid_timings[key] += 1
                elif (
                    not trace["evaluation"]
                    .get("log", "")
                    .startswith(TIMING_FAILURE_PREFIX)
                ):
                    terminal_workloads.add(key)
                trace["service_attempt"] = attempt
            traces.extend(attempt_traces)
            attempts.append(
                {
                    "attempt": attempt,
                    "workloads": len(pending),
                    "returncode": result.returncode,
                    "traces": len(attempt_traces),
                    "stderr": result.stderr,
                }
            )

    completed = sum(count >= TRIALS for count in valid_timings.values())

    return {
        "problem": problem,
        "definition": definition_name,
        "workload_count": workload_count,
        "config": config,
        "run": {
            "returncode": 0 if completed == workload_count else 1,
            "traces": traces,
            "attempts": attempts,
        },
    }
