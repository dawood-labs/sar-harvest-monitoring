"""Machine resources for this project: one place that decides workers and chunk sizes.

Why: code must adapt to the machine (laptop, small pod, big pod) so it uses the cores it has
and never runs out of memory. Every script imports this instead of guessing.
Standard library only, so it works in any environment.

Usage:
    from resources import detect, workers, chunk_rows
    n = workers()                       # processes or threads to start
    rows = chunk_rows(bytes_per_row=800)  # rows per batch that fit in memory
Run `python resources.py` to print what was detected.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Resources:
    cores: int            # cores this process may really use
    mem_total_gb: float   # memory limit (container limit if there is one)
    mem_free_gb: float    # memory available right now
    gpus: tuple[str, ...]  # e.g. ("NVIDIA GeForce RTX 4050, 6141 MiB",)


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cores() -> int:
    # A pod can show 64 cores but be limited to 8 by its cgroup; the limit is what counts.
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = _read("/sys/fs/cgroup/cpu.max")  # "<quota> <period>" or "max <period>"
    if quota and not quota.startswith("max"):
        q, period = quota.split()
        n = min(n, max(1, int(int(q) / int(period))))
    return max(1, n)


def _meminfo_gb(key: str) -> float:
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1024**2  # kB -> GB
    return 0.0


def _memory() -> tuple[float, float]:
    total, free = _meminfo_gb("MemTotal"), _meminfo_gb("MemAvailable")
    limit = _read("/sys/fs/cgroup/memory.max")  # container limit, if any
    if limit and limit != "max":
        limit_gb = int(limit) / 1024**3
        used = int(_read("/sys/fs/cgroup/memory.current") or 0) / 1024**3
        if limit_gb < total:
            total, free = limit_gb, min(free, max(0.0, limit_gb - used))
    return total, free


def _gpus() -> tuple[str, ...]:
    if not shutil.which("nvidia-smi"):
        return ()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return ()
    return tuple(line.strip() for line in out.splitlines() if line.strip())


def detect() -> Resources:
    total, free = _memory()
    return Resources(_cores(), round(total, 1), round(free, 1), _gpus())


def workers(mem_per_worker_gb: float = 0.0, keep_free_cores: int = 1) -> int:
    """Workers to start: all cores but one, fewer if each worker needs a lot of memory."""
    r = detect()
    n = max(1, r.cores - keep_free_cores)
    if mem_per_worker_gb > 0:
        n = min(n, max(1, int(r.mem_free_gb * 0.8 // mem_per_worker_gb)))
    return n


def chunk_rows(bytes_per_row: int, n_workers: int | None = None, use_fraction: float = 0.5) -> int:
    """Rows per batch so that all workers together stay within a share of the free memory."""
    r = detect()
    n = n_workers or workers()
    budget = r.mem_free_gb * 1024**3 * use_fraction / n
    return max(1_000, int(budget // max(1, bytes_per_row)))


if __name__ == "__main__":
    r = detect()
    print(f"cores usable : {r.cores}")
    print(f"memory       : {r.mem_total_gb} GB limit, {r.mem_free_gb} GB free now")
    print(f"gpu          : {', '.join(r.gpus) or 'none'}")
    print(f"workers()    : {workers()}")
    print(f"chunk_rows(800 bytes/row): {chunk_rows(800):,}")
