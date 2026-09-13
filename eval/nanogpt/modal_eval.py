"""Evaluate one NanoGPT implementation in an existing Modal sandbox."""

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import modal
from grpclib.exceptions import GRPCError, StreamTerminatedError


SOURCES = ("train_gpt.py", "triton_kernels.py", "dc_triton_kernels.py")
HARNESS = Path(__file__).resolve().parent
GPU = "H100!:8"
TRAINING_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate", "profile", "check"))
    parser.add_argument("--candidate", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sandbox-id")
    args = parser.parse_args()
    if args.mode == "check":
        snapshot = source_snapshot(args.candidate)
        print(json.dumps({key: value for key, value in snapshot.items() if key != "sources"}))
        return
    if args.output is None and "SPAR_PROFILING_DIR" in os.environ:
        args.output = Path(os.environ["SPAR_PROFILING_DIR"])
        if args.mode != "profile":
            args.output = args.output.parent
    if args.output is None:
        parser.error("Pass --output for evaluation or profiling")
    if args.mode == "profile":
        result = json.loads((args.output.parent / "result.json").read_text())
        print(json.dumps(result["timing"]))
        return
    if not args.sandbox_id:
        parser.error("Pass --sandbox-id from modal_setup.py create")
    print(json.dumps(evaluate(source_snapshot(args.candidate), args.seed,
                              args.output.resolve(), args.sandbox_id), allow_nan=False))


def evaluate(snapshot, seed, directory, sandbox_id):
    if not re.fullmatch(r"sb-[A-Za-z0-9]+", sandbox_id):
        raise ValueError("Invalid sandbox ID")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed, "gpu": GPU, "sandbox_id": sandbox_id,
        **{key: value for key, value in snapshot.items() if key != "sources"},
        "harness_sha256": {name: hashlib.sha256((HARNESS / name).read_bytes()).hexdigest()
                           for name in ("modal_eval.py", "modal_setup.py", "torch_cache.py", "seeded_train.py", "benchmark.json")},
    }
    with (directory / "manifest.json").open("x") as output:
        output.write(json.dumps(manifest, indent=2) + "\n")
    with open(f"/tmp/nanogpt-{sandbox_id}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sandbox = modal.Sandbox.from_id(sandbox_id)
        acquired = False
        try:
            metadata = json.loads(sandbox.filesystem.read_text("/tmp/nanogpt-sandbox.json"))
            if metadata["id"] != sandbox_id or metadata["gpu"] != GPU:
                raise ValueError("Sandbox setup metadata does not match the requested allocation")
            (directory / "sandbox.json").write_text(json.dumps(metadata, indent=2) + "\n")
            (directory / "app.json").write_text(json.dumps({
                key: metadata[key] for key in ("name", "app_id", "environment")
            }, indent=2) + "\n")
            claim = sandbox.exec("mkdir", "/tmp/nanogpt-eval.lock", timeout=30)
            claim.wait()
            if claim.returncode:
                raise RuntimeError("Sandbox already has an evaluation; inspect it before reuse")
            acquired = True
            busy = sandbox.exec("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", timeout=30)
            active = busy.stdout.read().strip()
            busy.wait()
            if busy.returncode or active:
                raise RuntimeError(f"Sandbox has active GPU processes or an unreadable GPU state: {active}")
            clean = sandbox.exec("bash", "-c", "find . -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +",
                                 workdir="/workspace", timeout=60)
            clean.wait()
            if clean.returncode:
                raise RuntimeError(f"Workspace cleanup failed: {clean.stderr.read()}")
            result = train(sandbox, metadata, snapshot["sources"], seed, directory)
        finally:
            evaluation_error = sys.exception()
            try:
                if acquired:
                    release = sandbox.exec("rmdir", "/tmp/nanogpt-eval.lock", timeout=30)
                    release.wait()
            except Exception as cleanup_error:
                if evaluation_error is None:
                    raise
                evaluation_error.add_note(f"Sandbox lock cleanup also failed: {cleanup_error}")
            finally:
                sandbox.detach()
    result.update(manifest)
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def train(sandbox, metadata, sources, seed, directory):
    started = time.monotonic()
    print(f"Sandbox {sandbox.object_id}; seed {seed}; artifacts {directory}", file=sys.stderr, flush=True)
    for name, source in sources.items():
        if name == "train_gpt.py":
            source = source.replace("val_loss:{val_loss:.4f}", "val_loss:{val_loss:.8f}")
        sandbox.filesystem.write_text(source, f"/workspace/{name}")
    sandbox.filesystem.write_text((HARNESS / "seeded_train.py").read_text(), "/workspace/seeded_train.py")
    for filename, command in (
        ("environment.txt", ("nvidia-smi",)),
        ("topology.txt", ("nvidia-smi", "topo", "-m")),
        ("cpu.txt", ("lscpu",)),
        ("cpuinfo.txt", ("cat", "/proc/cpuinfo")),
        ("python.txt", ("python", "-VV")),
        ("packages.txt", ("python", "-m", "pip", "freeze")),
    ):
        inventory = sandbox.exec(*command, timeout=60)
        (directory / filename).write_text(inventory.stdout.read() + inventory.stderr.read())
        inventory.wait()
    hardware = sandbox.exec("nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader")
    names = hardware.stdout.read().strip().splitlines()
    hardware.wait()
    if hardware.returncode or len(names) != 8 or any("H100" not in name for name in names):
        raise RuntimeError(f"Expected exactly eight H100 GPUs, got {names}")
    sandbox.filesystem.write_text("", "/workspace/train.stdout")
    process = sandbox.exec("bash", "-c", "torchrun --standalone --nproc_per_node=8 seeded_train.py > /workspace/train.stdout 2>&1",
                           env={"DATA_PATH": "/cache", "NANOGPT_SEED": str(seed), **TRAINING_ENV}, timeout=1600)
    returncode = collect_training(sandbox, process, directory / "train.stdout")
    log = (directory / "train.stdout").read_text()
    result = summarize(log, returncode)
    result["timing"] = timing_profile(log)
    actual_env = next(line.split("=", 1)[1] for line in log.splitlines()
                      if line.startswith("HARNESS_TRAINING_ENV="))
    result.update(seed=seed, sandbox_id=sandbox.object_id, image_id=metadata["image_id"], devices=names,
                  training_environment=json.loads(actual_env))
    result["job_wall_seconds"] = time.monotonic() - started
    return result


def collect_training(sandbox, process, logfile):
    failures = 0
    previous = 0
    while True:
        try:
            returncode = process.poll()
            output = sandbox.filesystem.read_text("/workspace/train.stdout")
        except (OSError, GRPCError, StreamTerminatedError) as error:
            failures += 1
            if failures == 3:
                raise
            print(f"Retrying log/status request: {error}", file=sys.stderr, flush=True)
            time.sleep(3)
            continue
        failures = 0
        logfile.write_text(output)
        for line in output[previous:].splitlines():
            if any(marker in line for marker in ("val_loss:", "Compiling", "Error", "Resetting")):
                print(line, file=sys.stderr, flush=True)
        previous = len(output)
        if returncode is not None:
            return returncode
        time.sleep(10)


def validate_candidate(directory=None):
    directory = Path.cwd() if directory is None else directory
    benchmark = json.loads((HARNESS / "benchmark.json").read_text())
    source = (directory / "train_gpt.py").read_text()
    tree = ast.parse(source)
    for name, expected in benchmark["data_definitions"].items():
        after = next(node for node in tree.body if getattr(node, "name", None) == name)
        if hashlib.sha256(ast.dump(after).encode()).hexdigest() != expected:
            raise ValueError(f"Protected data pipeline changed: {name}")
    for marker, expected in benchmark["protected_tails"].items():
        start = source.index(marker)
        if hashlib.sha256(source[start:].encode()).hexdigest() != expected:
            raise ValueError("Warmup, validation, training clock, and main loop must remain unchanged")
    cls = next(node for node in tree.body if getattr(node, "name", None) == "Hyperparameters")
    fields = {node.target.id: hashlib.sha256(ast.dump(node.value).encode()).hexdigest()
              for node in cls.body if isinstance(node, ast.AnnAssign)}
    if any(fields.get(name) != expected for name, expected in benchmark["validation_fields"].items()):
        raise ValueError("Protected validation configuration changed")
    print(f"Validated protected benchmark in {directory}", file=sys.stderr)


def source_snapshot(directory):
    directory = directory.resolve()
    validate_candidate(directory)
    sources = {name: (directory / name).read_text() for name in SOURCES}
    commit = None
    try:
        root = subprocess.run(["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True)
        if root.returncode == 0 and Path(root.stdout.strip()).resolve() == directory:
            commit = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    except FileNotFoundError:
        pass
    return {"sources": sources, "commit": commit,
            "source_sha256": {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}}


def summarize(log, returncode):
    rows = [
        dict(step=int(step), total_steps=int(total), val_loss=float(loss), training_seconds=float(ms) / 1000)
        for step, total, loss, ms in re.findall(
            r"^step:(\d+)/(\d+) val_loss:([\d.eE+-]+) train_time:([\d.]+)ms", log, re.MULTILINE
        )
    ]
    if returncode or not rows or rows[-1]["step"] != rows[-1]["total_steps"]:
        raise ValueError("Training failed or final validation is missing")
    final = rows[-1]
    if not all(math.isfinite(final[key]) for key in ("val_loss", "training_seconds")):
        raise ValueError("Non-finite final metric")
    if final["training_seconds"] <= 0:
        raise ValueError("Training time must be positive")
    reached = final["val_loss"] <= 3.28
    return {
        **final,
        "reached_target": reached,
        "beat_77_5": reached and final["training_seconds"] < 77.5,
        "score": 77.5 / final["training_seconds"] if reached else -final["val_loss"],
        "validation_history": rows,
    }


def timing_profile(log):
    rows = [(int(step), float(ms)) for step, ms in re.findall(
        r"^step:(\d+)/\d+ train_time:([\d.]+)ms", log, re.MULTILINE
    )]
    deltas = [(step, ms - previous) for (step, ms), (_, previous) in zip(rows[1:], rows)]
    return {
        "kind": "training-log timing; asynchronous step estimates, not a CUDA kernel trace",
        "steps": len(rows),
        "median_step_ms": statistics.median(ms for _, ms in deltas) if deltas else None,
        "slowest_step_estimates": sorted(deltas, key=lambda row: row[1], reverse=True)[:10],
        "step_estimates": deltas,
    }


if __name__ == "__main__":
    main()
