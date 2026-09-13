# NanoGPT evaluation on Modal

Provision an eight-H100 sandbox once, then submit implementations or seeds to
that same allocation. Each evaluation starts a fresh training process and saves
its own results. GPU allocation, image, data, and compiler caches remain in place
until explicit cleanup or a timeout.

## Files

- `modal_setup.py`: create and prepare a sandbox; stop its sandbox and app.
- `modal_eval.py`: evaluate one target and seed in an existing sandbox; also
  check protected benchmark code and read saved timing summaries without GPUs.
- `modal_multi_seed.py`: invoke single-target evaluation for several seeds in
  the same sandbox, preserve all results, and analyze them.
- `seeded_train.py`: seed Python, NumPy, and Torch; execute the target in the
  existing `__main__` module without extra `runpy` frames.
- `torch_cache.py`: restore and save local TorchInductor/Triton caches.
- `benchmark.json`: protected data, validation, warmup, and timing definitions
  from upstream commit `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
- `tests/test_*.py`: lifecycle, isolation, scoring, seeding, and portability tests.

The local `pyproject.toml` and `uv.lock` install host dependencies. The host needs
`uv`, Python 3.14+, and Modal credentials. Evaluation needs neither a local GPU
nor Git history. All source and artifact paths come from arguments or the
scripts' location.

## Run

From this directory:

```bash
uv sync --locked
# If credentials are not already configured:
uv run modal token new
# If using this non-default Modal environment:
export MODAL_ENVIRONMENT=sandbox

uv run python modal_setup.py create --output /path/to/new-setup
# Use the id printed above or saved in new-setup/sandbox.json:
uv run python modal_eval.py evaluate --sandbox-id sb-... \
  --candidate /path/to/nanogpt --seed 42 --output /path/to/new-result

uv run python modal_multi_seed.py --sandbox-id sb-... \
  --candidate /path/to/nanogpt --seeds 1616,1717,1818,1919,2020 \
  --output /path/to/new-multi-seed-result

# Always stop the allocation after the batch, including on errors:
uv run python modal_setup.py stop --output /path/to/new-setup

# These commands need no GPU:
uv run python modal_eval.py check --candidate /path/to/nanogpt
uv run python modal_eval.py profile --output /path/to/new-result/profiling
```

For automated batches, call `modal_setup.stop(setup_directory)` in the
coordinator's `finally` block, or arrange the equivalent shell exit trap. Keep
the setup directory until cleanup succeeds: it records the app ID even if
sandbox creation fails. Each app has a random hexadecimal name. Setup cleans
up on preparation failure; explicit stop saves caches and terminates both the
sandbox and app. A 30-minute idle timeout and 24-hour sandbox lifetime provide
fallback limits. Each training process has a 1,600-second timeout.

A candidate contains `train_gpt.py`, `triton_kernels.py`, and
`dc_triton_kernels.py`. Evaluation requires `--sandbox-id` and never allocates a
replacement sandbox. Its workspace is cleared before uploading each target,
and concurrent use or leftover GPU processes are rejected. On runtime failure,
the coordinator must stop that allocation before continuing on another one.
Loss above the threshold remains a completed, scored result.

## Results and cache reuse

Single-run `result.json` contains `score`, `training_seconds`, `val_loss`,
`reached_target`, `beat_77_5`, `timing`, source/evaluator hashes, seed, sandbox ID,
GPU UUIDs, image ID, and the training thread environment. Raw logs and CPU,
Python, and package inventories are retained. CPU metadata reflects what the
container reports; it is not a controlled clock measurement. Existing result directories cannot be
overwritten.

The score is `77.5 / training_seconds` when loss is at most 3.28, otherwise
negative loss. Training time uses the original synchronized benchmark clock;
compilation, warmup, validation, and setup are excluded. The timing summary
contains asynchronous step estimates, not a CUDA kernel trace. Sandbox reuse
reduces allocation differences; it does not eliminate runtime variation.

Multi-seed output preserves `run-<seed>/` results and reports median/max time,
mean/max loss, and a one-sided mean-loss t-test. It exits nonzero unless every
run has loss <=3.28 and time <77.5 seconds, with p<0.01. Use distinct, unused
seeds for independent confirmation; repeating a tuning seed measures runtime
repeatability instead.

Setup downloads missing FineWeb shards and restores the compiler-cache archive
once. It verifies the Flash Attention publisher online, downloads the complete
kernel snapshot, and checks offline loading before accepting evaluations.
Training uses that verified cache in offline mode, avoiding Hub requests from
each distributed worker. Caches remain local between evaluations and are saved at shutdown.
Unchanged compiled graphs and kernels can be reused; changed shapes, graphs,
or compiler settings may require compilation. `--cache-slot` on setup selects
an archive (`shared`, `systems`, or an 8–32 character lowercase hexadecimal
name). Use separate slots for concurrent sandboxes that save their caches.

Optional environment variables are `MODAL_ENVIRONMENT`,
`NANOGPT_MODAL_VOLUME` (default `8df74031a65e4be38700504d426d8b4a`), and
`NANOGPT_MODAL_IMAGE_ID`. An explicit image ID preserves a verified runtime;
without one, setup builds the pinned Torch/CUDA dependency image. The seeded
worker and protected benchmark definitions must remain consistent across
comparisons.

## SPAR and tests

The session coordinator owns setup and teardown. Configure the evaluation
command with the resulting sandbox ID:

```toml
[evaluation]
command = ["uv", "run", "--project", "/path/to/eval/nanogpt", "python", "/path/to/eval/nanogpt/modal_eval.py", "evaluate", "--sandbox-id", "sb-..."]
timeout_seconds = 2400

[profiling]
command = ["uv", "run", "--project", "/path/to/eval/nanogpt", "python", "/path/to/eval/nanogpt/modal_eval.py", "profile"]
timeout_seconds = 60
```

SPAR supplies its artifact location; otherwise pass `--output` for single-run
evaluation and profiling. To invoke from elsewhere, use `uv run --project
/path/to/eval/nanogpt python /path/to/eval/nanogpt/modal_eval.py ...`.

Run `uv run python -m unittest discover -s tests -t .`. Source protection and
relocation tests use the repository's `nanogpt/` solution by default; set
`NANOGPT_TEST_CANDIDATE` to another compatible source directory when testing a
relocated evaluator. Runtime evaluation has no dependency on that layout.
