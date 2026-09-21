"""Evaluate one NanoGPT implementation across seeds and analyze all results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import uuid

from scipy.stats import ttest_1samp

import modal_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", required=True, help="Comma-separated distinct seeds")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sandbox-id", required=True)
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        parser.error("Use at least two distinct seeds")
    snapshot = modal_eval.source_snapshot(args.candidate)
    directory = (args.output or (
        Path(os.environ["SPAR_PROFILING_DIR"]).parent if "SPAR_PROFILING_DIR" in os.environ
        else Path.home() / "nanogpt-validation" / "results" / uuid.uuid4().hex
    )).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seeds": seeds, "gpu": modal_eval.GPU, "sandbox_id": args.sandbox_id,
        **{key: value for key, value in snapshot.items() if key != "sources"},
        "coordinator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with (directory / "manifest.json").open("x") as output:
        output.write(json.dumps(manifest, indent=2) + "\n")
    runs = [modal_eval.evaluate(snapshot, seed, directory / f"run-{seed}", args.sandbox_id) for seed in seeds]
    result = {**manifest, **analyze(runs), "runs": runs}
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    if not result["passed"]:
        sys.exit(1)


def analyze(runs):
    if len(runs) < 2 or len({row["seed"] for row in runs}) != len(runs):
        raise ValueError("Expected at least two runs with distinct seeds")
    if len({row["image_id"] for row in runs}) != 1:
        raise ValueError("All seeds must use the same training image")
    losses = [row["val_loss"] for row in runs]
    seconds = [row["training_seconds"] for row in runs]
    median_seconds = statistics.median(seconds)
    mean_loss = statistics.mean(losses)
    p_value = float(ttest_1samp(losses, popmean=3.28, alternative="less").pvalue)
    reached = mean_loss <= 3.28 and p_value < 0.01
    under_time = reached and all(duration < 77.5 for duration in seconds)
    return {
        "score": 77.5 / median_seconds if reached else -mean_loss,
        "median_training_seconds": median_seconds,
        "max_training_seconds": max(seconds),
        "mean_val_loss": mean_loss,
        "max_val_loss": max(losses),
        "loss_p_value_one_sided": p_value,
        "reached_target": reached,
        "beat_77_5": under_time,
        "passed": under_time,
    }


if __name__ == "__main__":
    main()
