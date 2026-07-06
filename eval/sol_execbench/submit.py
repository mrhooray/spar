import json
import os
import statistics
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ENDPOINT = os.environ["SPAR_EVAL_ENDPOINT"]
PROBLEM = os.environ["SPAR_EVAL_PROBLEM"]
SOLUTION_PATH = os.environ.get("SPAR_EVAL_SOLUTION")
ITERATIONS = int(os.environ.get("SPAR_EVAL_ITERATIONS", "50"))
WARMUP_RUNS = int(os.environ.get("SPAR_EVAL_WARMUP_RUNS", "10"))


def submit(*, source: str | None = None, solution: dict | None = None) -> dict:
    payload = {
        "problem": PROBLEM,
        "iterations": ITERATIONS,
        "warmup_runs": WARMUP_RUNS,
    }
    if solution is None:
        if source is None:
            raise ValueError("source or solution must be provided")
        payload["source"] = source
    else:
        payload["solution"] = solution

    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def summarize(result: dict, started_at: str, elapsed_seconds: float) -> dict:
    latencies = defaultdict(list)
    statuses = defaultdict(list)
    axes = {}
    for trace in result["run"]["traces"]:
        workload_axes = trace["workload"]["axes"]
        workload_key = json.dumps(workload_axes, sort_keys=True, separators=(",", ":"))
        axes[workload_key] = workload_axes
        evaluation = trace["evaluation"]
        statuses[workload_key].append(evaluation["status"])
        performance = evaluation.get("performance")
        if evaluation["status"] == "PASSED" and isinstance(performance, dict):
            latency = performance.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies[workload_key].append(float(latency))

    workload_timings = []
    required_measurements = result["config"].get("trials", 1)
    for workload_key in sorted(statuses):
        values = latencies[workload_key]
        mean_latency = (
            statistics.mean(values)
            if len(values) >= required_measurements
            else None
        )
        workload_timings.append(
            {
                "axes": axes[workload_key],
                "latencies_ms": values,
                "mean_latency_ms": mean_latency,
                "median_latency_ms": statistics.median(values) if values else None,
                "valid_measurements": len(values),
                "statuses": statuses[workload_key],
            }
        )

    workload_means = [
        item["mean_latency_ms"]
        for item in workload_timings
        if item["mean_latency_ms"] is not None
    ]
    workload_count = result["workload_count"]
    suite_median = (
        statistics.median(workload_means)
        if len(workload_means) == workload_count
        else None
    )
    return {
        "score": 1000.0 / suite_median if suite_median else 0.0,
        "correct": len(workload_means) == workload_count,
        "suite_median_latency_ms": suite_median,
        "valid_workloads": len(workload_means),
        "workloads": workload_count,
        "required_measurements": required_measurements,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "workload_timings": workload_timings,
        "raw_run": result["run"],
    }


def main() -> None:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if SOLUTION_PATH is None:
        result = submit(source=Path("solution.py").read_text())
    else:
        solution = json.loads(Path(SOLUTION_PATH).read_text())
        if not isinstance(solution, dict):
            raise ValueError("SPAR_EVAL_SOLUTION must contain a JSON object")
        result = submit(solution=solution)
    print(json.dumps(summarize(result, started_at, time.perf_counter() - started)))


if __name__ == "__main__":
    main()
