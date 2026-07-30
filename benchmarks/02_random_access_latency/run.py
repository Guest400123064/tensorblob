"""Benchmark 02: Random access latency.

Measures random single-row and random-batch read latency on a TensorBlob,
compared against in-memory indexing (upper bound). Also sweeps block_size
and max_cached_blocks to show their effect on random-access performance
(the tuning knobs documented in the README).

Data: float32 tensors of shape (768,). Main test uses 500,000 rows
(~1.5 GiB); the knob sweep uses 200,000 rows per configuration.

Latency is measured per access over N_QUERIES random queries (fixed seed);
median and p99 are reported.
"""

import os
import platform
import shutil
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from tensorblob import TensorBlob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

DIM = 768
N_MAIN = 500_000
N_SWEEP = 200_000
N_QUERIES = 10_000
BATCH = 512
N_BATCHES = 64


def machine_info() -> str:
    with open("/proc/meminfo") as f:
        mem = int(f.readline().split()[1]) // 1024
    return (
        f"Machine: {platform.machine()} {platform.system()}, {os.cpu_count()} cores, "
        f"{mem} MiB RAM | python {platform.python_version()}, "
        f"torch {torch.__version__}"
    )


def make_blob(path, n, block_size, src):
    shutil.rmtree(path, ignore_errors=True)
    with TensorBlob.open(path, "w", dtype="float32", shape=DIM, block_size=block_size) as blob:
        blob.write(src[:n])


def drop_page_cache(path):
    """Discard clean page-cache pages of every file under `path` (non-root
    alternative to /proc/sys/vm/drop_caches; see 01_raw_io_throughput)."""
    os.sync()
    for root, _, files in os.walk(path):
        for name in files:
            fd = os.open(os.path.join(root, name), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def latency(fn, queries):
    samples = []
    for q in queries:
        t0 = time.perf_counter()
        fn(q)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    med = samples[len(samples) // 2]
    p99 = samples[int(len(samples) * 0.99)]
    return med * 1e6, p99 * 1e6  # microseconds


def main():
    print(machine_info())
    print(f"Random single-row latency over {N_QUERIES:,} queries (fixed seed)\n")

    shutil.rmtree(DATA, ignore_errors=True)
    os.makedirs(DATA)
    torch.manual_seed(0)
    src = torch.randn(N_MAIN, DIM, dtype=torch.float32)
    g = torch.Generator().manual_seed(42)
    queries = torch.randint(N_MAIN, (N_QUERIES,), generator=g).tolist()

    make_blob(f"{DATA}/main", N_MAIN, 8192, src)

    # Disk-cold first: page-cache pages discarded, fresh blob handle, and a
    # smaller query count since each miss costs a disk seek on HDD.
    drop_page_cache(f"{DATA}/main")
    with TensorBlob.open(f"{DATA}/main", "r") as blob:
        med, p99 = latency(lambda q: blob[q], queries[:2_000])
        print(f"single-row  TensorBlob (disk-cold)                median {med:8.1f} us   p99 {p99:8.1f} us  (2,000 queries)")

    with TensorBlob.open(f"{DATA}/main", "r") as blob:
        med, p99 = latency(lambda q: blob[q], queries)
        print(f"single-row  TensorBlob (block=8192, default cache)  median {med:8.1f} us   p99 {p99:8.1f} us")

    med, p99 = latency(lambda q: src[q].clone(), queries)
    print(f"single-row  in-memory (upper bound)                median {med:8.1f} us   p99 {p99:8.1f} us")

    # Random-batch access: compare row-by-row gather (the old way) with
    # vectorized batch indexing, and in-memory fancy indexing as upper bound.
    print(f"\nRandom batch gather ({N_BATCHES} batches x {BATCH} rows)")
    batch_queries = [
        torch.randint(N_MAIN, (BATCH,), generator=g).tolist() for _ in range(N_BATCHES)
    ]

    with TensorBlob.open(f"{DATA}/main", "r") as blob:
        t0 = time.perf_counter()
        for idxs in batch_queries:
            torch.stack([blob[i] for i in idxs])
        dt = time.perf_counter() - t0
        print(f"  TensorBlob row-by-row gather     {dt / N_BATCHES * 1e3:8.2f} ms/batch")

        t0 = time.perf_counter()
        for idxs in batch_queries:
            blob[idxs]
        dt = time.perf_counter() - t0
        print(f"  TensorBlob vectorized blob[idxs] {dt / N_BATCHES * 1e3:8.2f} ms/batch")

    t0 = time.perf_counter()
    for idxs in batch_queries:
        src[torch.tensor(idxs)]
    dt = time.perf_counter() - t0
    print(f"  in-memory (fancy indexing)       {dt / N_BATCHES * 1e3:8.2f} ms/batch")

    # Sweep: block_size x max_cached_blocks
    print(f"\nKnob sweep ({N_SWEEP:,} rows per config, median single-row latency)")
    print(f"  {'block_size':>10} {'max_cached':>10} {'median':>10} {'p99':>10}")
    small = src[:N_SWEEP]
    sweep_queries = [q % N_SWEEP for q in queries]
    for block_size in [1024, 8192, 65536]:
        path = f"{DATA}/sweep_{block_size}"
        make_blob(path, N_SWEEP, block_size, small)
        for cache in [16, 256, 4096]:
            with TensorBlob.open(path, "r", max_cached_blocks=cache) as blob:
                med, p99 = latency(lambda q: blob[q], sweep_queries)
                print(f"  {block_size:>10,} {cache:>10,} {med:>8.1f}us {p99:>8.1f}us")
        shutil.rmtree(path)

    shutil.rmtree(DATA)  # cleanup


if __name__ == "__main__":
    main()
