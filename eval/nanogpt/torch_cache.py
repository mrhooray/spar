"""Restore and save a sandbox's local TorchInductor/Triton caches."""


def restore(sandbox, archive):
    process = sandbox.exec("bash", "-c",
        'set -e; archive="$1"; if ! test -f "$archive"; then archive=/persistent/compiler-cache.tar; fi; '
        'if test -f "$archive"; then tar -b 16384 -xf "$archive" -C /cache; fi; '
        'mkdir -p /cache/inductor /cache/triton', "cache-restore", archive, timeout=300)
    process.wait()
    if process.returncode:
        raise RuntimeError(f"Cache restore failed: {process.stderr.read()}")


def save(sandbox, archive):
    process = sandbox.exec("bash", "-c",
        'tar -b 16384 -cf "$1.tmp" -C /cache inductor triton && mv "$1.tmp" "$1"',
        "cache-save", archive, timeout=180)
    process.wait()
    if process.returncode:
        raise RuntimeError(f"Cache save failed: {process.stderr.read()}")
