"""Create or stop a reusable eight-H100 NanoGPT sandbox."""

import argparse
import contextlib
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import modal

import torch_cache


HARNESS = Path(__file__).resolve().parent
GPU = "H100!:8"
VOLUME = os.environ.get("NANOGPT_MODAL_VOLUME", "8df74031a65e4be38700504d426d8b4a")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "stop"))
    parser.add_argument("--output", type=Path, required=True, help="Directory holding the setup manifest")
    parser.add_argument("--cache-slot", type=cache_slot, default="shared")
    parser.add_argument("--cloud", choices=("gcp", "aws", "oci"), default="gcp")
    args = parser.parse_args()
    directory = args.output.resolve()
    if args.mode == "create":
        result = create(directory, args.cache_slot, args.cloud)
    else:
        result = stop(directory)
    print(json.dumps(result))


def create(directory, cache_slot="shared", cloud="gcp"):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "app.json").open("x") as output:
        name = uuid.uuid4().hex
        environment = os.environ.get("MODAL_ENVIRONMENT")
        app = modal.App.lookup(name, environment_name=environment, create_if_missing=True)
        output.write(json.dumps({"name": name, "app_id": app.app_id, "environment": environment}, indent=2) + "\n")
    sandbox = None
    try:
        image = training_image()
        volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
        with contextlib.redirect_stdout(sys.stderr), modal.enable_output():
            sandbox = modal.Sandbox.create("sleep", "86400", app=app, image=image, gpu=GPU,
                                           cloud=cloud,
                                           cpu=32, memory=131072, volumes={"/persistent": volume},
                                           workdir="/workspace", timeout=86400, idle_timeout=1800)
        archive = f"/persistent/compiler-cache{'-' + cache_slot if cache_slot != 'shared' else ''}.tar"
        metadata = {"id": sandbox.object_id, "gpu": GPU, "image_id": image.object_id, "cloud": cloud,
                    "name": name, "app_id": app.app_id, "environment": environment,
                    "cache_slot": cache_slot, "archive": archive, "volume": VOLUME,
                    "created_at": datetime.now(timezone.utc).isoformat()}
        (directory / "sandbox.json").write_text(json.dumps(metadata, indent=2) + "\n")
        links = sandbox.exec("bash", "-c",
            "ln -s /persistent/data /cache/data && ln -s /persistent/huggingface /cache/huggingface", timeout=30)
        links.wait()
        if links.returncode:
            raise RuntimeError(f"Dataset/cache linking failed: {links.stderr.read()}")
        torch_cache.restore(sandbox, archive)
        prepare = sandbox.exec("python", "-c", inspect.getsource(download_fineweb)
                               + '\ndownload_fineweb("/persistent/data/fineweb10B")', timeout=1500)
        for line in prepare.stdout:
            print(line, end="", file=sys.stderr, flush=True)
        prepare.wait()
        if prepare.returncode:
            raise RuntimeError(f"Dataset preparation failed: {prepare.stderr.read()}")
        dependency = sandbox.exec("python", "-c",
            "from kernels import get_kernel; from huggingface_hub import snapshot_download; "
            "k = get_kernel('kernels-community/flash-attn3', version=1, backend='cuda', check_arch=False); "
            "print(k.flash_attn_interface); "
            "print(snapshot_download('kernels-community/flash-attn3', repo_type='kernel', revision='v1'))",
            env={"HF_HUB_OFFLINE": "0"}, timeout=300)
        output = dependency.stdout.read() + dependency.stderr.read()
        dependency.wait()
        (directory / "dependency.txt").write_text(output)
        if dependency.returncode:
            raise RuntimeError(f"Flash Attention dependency preflight failed: {output}")
        offline = sandbox.exec("python", "-c",
            "from kernels import get_kernel; print(get_kernel('kernels-community/flash-attn3', version=1).flash_attn_interface)",
            env={"HF_HUB_OFFLINE": "1"}, timeout=60)
        output = offline.stdout.read() + offline.stderr.read()
        offline.wait()
        (directory / "dependency-offline.txt").write_text(output)
        if offline.returncode:
            raise RuntimeError(f"Flash Attention offline preflight failed: {output}")
        sandbox.filesystem.write_text(json.dumps(metadata), "/tmp/nanogpt-sandbox.json")
        return metadata
    except BaseException:
        stop(directory)
        raise
    finally:
        if sandbox is not None:
            sandbox.detach()


def stop(directory):
    app = json.loads((directory / "app.json").read_text())
    sandbox = None
    try:
        if (directory / "sandbox.json").exists():
            metadata = json.loads((directory / "sandbox.json").read_text())
            try:
                sandbox = modal.Sandbox.from_id(metadata["id"])
                if sandbox.poll() is None:
                    torch_cache.save(sandbox, metadata["archive"])
            except Exception as error:
                print(f"Could not persist compiler cache: {error}", file=sys.stderr)
    finally:
        try:
            if sandbox is not None:
                sandbox.terminate(wait=True)
                sandbox.detach()
        finally:
            command = [sys.executable, "-m", "modal", "app", "stop", app["app_id"], "--yes"]
            if app["environment"]:
                command.extend(["--env", app["environment"]])
            stopped = subprocess.run(command, capture_output=True, text=True, timeout=120)
            print(stopped.stdout + stopped.stderr, file=sys.stderr)
            if not (stopped.returncode == 1 and "App is already stopped." in stopped.stdout + stopped.stderr):
                stopped.check_returncode()
    result = {"app_id": app["app_id"], "stopped_at": datetime.now(timezone.utc).isoformat()}
    (directory / "stopped.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def cache_slot(value):
    if value not in ("shared", "systems") and not re.fullmatch(r"[0-9a-f]{8,32}", value):
        raise argparse.ArgumentTypeError("Use shared, systems, or an 8–32 character lowercase hexadecimal slot")
    return value


def training_image():
    if image_id := os.environ.get("NANOGPT_MODAL_IMAGE_ID"):
        return modal.Image.from_id(image_id)
    return (
        modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
        .entrypoint([])
        .apt_install("build-essential", "git")
        .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
        .pip_install("numpy==2.2.6", "tqdm==4.67.1", "huggingface-hub==1.31.0",
                     "kernels==0.16.1", "setuptools==80.9.0", "typing-extensions==4.15.0",
                     "tiktoken==0.12.0", "uv==0.12.11")
        .run_commands(
            "uv python install 3.12.13 --install-dir /opt/python",
            "cp -a /opt/python/cpython-3.12.13-linux-x86_64-gnu/. /usr/local/",
            'python -c "import sys, torch, triton; assert sys.version_info[:3] == (3, 12, 13); '
            'assert torch.__version__ == \'2.10.0+cu128\'; assert triton.__version__ == \'3.6.0\'"',
            "mkdir -p /workspace /cache")
        .env({"HF_HOME": "/cache/huggingface", "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
              "TRITON_CACHE_DIR": "/cache/triton", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    )


def download_fineweb(directory):
    from pathlib import Path

    from huggingface_hub import snapshot_download

    files = ["fineweb_val_000000.bin", *(f"fineweb_train_{i:06d}.bin" for i in range(1, 10))]
    missing = [name for name in files if not (Path(directory) / name).is_file()]
    if missing:
        snapshot_download(repo_id="kjj0/fineweb10B-gpt2", repo_type="dataset",
                          allow_patterns=missing, local_dir=str(directory))


if __name__ == "__main__":
    main()
