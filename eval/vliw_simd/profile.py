import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

TRACE_ENGINES = {"alu", "valu", "load", "store", "flow"}


def main() -> None:
    result = subprocess.run(
        [sys.executable, "perf_takehome.py", "Tests.test_kernel_trace"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    sys.stderr.write(result.stderr)
    match = re.search(r"CYCLES:\s+(\d+)", result.stdout)
    profiling_dir = Path(os.environ["SPAR_PROFILING_DIR"])
    profiling_dir.mkdir(parents=True, exist_ok=True)
    trace = Path("trace.json")
    if not trace.exists():
        raise RuntimeError("profiling did not produce trace.json")
    saved_trace = profiling_dir / trace.name
    trace.replace(saved_trace)
    report = {
        "cycles": int(match.group(1)) if match else None,
        "returncode": result.returncode,
        "trace_summary": summarize_trace(saved_trace, int(match.group(1))) if match else None,
    }
    print(json.dumps(report))
    raise SystemExit(result.returncode)


def summarize_trace(path: Path, cycles: int) -> dict[str, Any]:
    candidate_pids: set[int] = set()
    lane_engines: dict[tuple[int, int], str] = {}
    lane_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    active_cycles: Counter[str] = Counter()
    saturated_cycles: Counter[str] = Counter()
    last_cycles: dict[str, int] = {}
    operation_multiset: Counter[tuple[str, str]] = Counter()
    last_packets: deque[dict[str, Any]] = deque(maxlen=8)
    packet_cycle: int | None = None
    packet_counts: Counter[str] = Counter()
    packet_operations: dict[str, list[str]] = {}

    def finish_packet() -> None:
        if packet_cycle is None:
            return
        for engine, count in packet_counts.items():
            active_cycles[engine] += 1
            if count == lane_counts[engine]:
                saturated_cycles[engine] += 1
        last_packets.append(
            {
                "cycle": packet_cycle,
                "ops": {engine: operations for engine, operations in packet_operations.items()},
            }
        )

    for event in trace_events(path):
        if event.get("ph") == "M" and event.get("name") == "process_name":
            process_name = event.get("args", {}).get("name", "")
            if process_name.startswith("Core ") and not process_name.endswith(" Scratch"):
                candidate_pids.add(event["pid"])
            continue
        if event.get("ph") == "M" and event.get("name") == "thread_name":
            lane_name = event.get("args", {}).get("name", "")
            engine = lane_name.rpartition("-")[0]
            if event.get("pid") in candidate_pids and engine in TRACE_ENGINES:
                lane = (event["pid"], event["tid"])
                lane_engines[lane] = engine
                lane_counts[engine] += 1
            continue

        engine = lane_engines.get((event.get("pid"), event.get("tid")))
        if engine is None or event.get("ph") != "X" or not event.get("dur"):
            continue
        cycle = event["ts"]
        if packet_cycle != cycle:
            finish_packet()
            packet_cycle = cycle
            packet_counts = Counter()
            packet_operations = {}

        name = event["name"]
        args = event.get("args", {})
        slot = args.get("slot", name)
        packet_counts[engine] += 1
        packet_operations.setdefault(engine, []).append(name)
        operation_counts[engine] += 1
        operation_multiset[(engine, slot)] += 1
        last_cycles[engine] = cycle
    finish_packet()

    engines = {}
    for engine in TRACE_ENGINES:
        if engine not in lane_counts:
            continue
        lanes = lane_counts[engine]
        operations = operation_counts[engine]
        last_cycle = last_cycles.get(engine)
        engines[engine] = {
            "lanes": lanes,
            "operations": operations,
            "utilization": round(operations / (cycles * lanes), 6),
            "active_cycles": active_cycles[engine],
            "saturated_cycles": saturated_cycles[engine],
            "last_cycle": last_cycle,
            "end_gap": None if last_cycle is None else cycles - last_cycle - 1,
        }

    multiset_payload = json.dumps(
        [[engine, operation, count] for (engine, operation), count in sorted(operation_multiset.items())],
        separators=(",", ":"),
    ).encode()
    return {
        "engines": engines,
        "last_packets": list(last_packets),
        "operation_multiset_sha256": hashlib.sha256(multiset_payload).hexdigest(),
    }


def trace_events(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as trace:
        for line in trace:
            item = line.strip()
            if item.startswith("["):
                item = item[1:].lstrip()
            if item.endswith("]"):
                item = item[:-1].rstrip()
            if item.endswith(","):
                item = item[:-1].rstrip()
            if item:
                yield json.loads(item)


if __name__ == "__main__":
    main()
