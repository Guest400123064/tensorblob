"""Benchmark 01: Raw I/O throughput.

Compares write and sequential-read throughput of TensorBlob against two
baselines:
  (a) a single monolithic MemoryMappedTensor file  -> isolates the overhead of
      TensorBlob's block/LRU abstraction
  (b) a plain in-memory tensor                     -> upper bound ("direct
      memory access")

Data: float32 tensors of shape (768,), 500,000 rows (~1.5 GiB).

Reads are performed in batches of 8,192 rows (TensorBlob's default block
size). "cold" = first pass right after opening (pages not yet faulted into
the page cache); "warm" = second pass (page cache hot); "disk-cold" =
page-cache pages explicitly discarded beforehand via POSIX_FADV_DONTNEED
(works without root; dirty pages are flushed with os.sync() first), so the
read is served from the disk itself.

Each measurement is repeated REPS times; the median is reported.
"""

import os
import platform
import shutil
import statistics
import sys
import time

import torch
from tensordict import MemoryMappedTensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from tensorblob import TensorBlob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

N_ROWS = 500_000
DIM = 768
BATCH = 8192
REPS = 3

DTYPE = torch.float32
MB = N_ROWS * DIM * 4 / 1e6


def machine_info() -> str:
    with open("/proc/meminfo") as f:
        mem = int(f.readline().split()[1]) // 1024
    return (
        f"Machine: {platform.machine()} {platform.system()}, {os.cpu_count()} cores, "
        f"{mem} MiB RAM | python {platform.python_version()}, "
        f"torch {torch.__version__}"
    )


def reset():
    shutil.rmtree(DATA, ignore_errors=True)
    os.makedirs(DATA)


def drop_page_cache(path):
    """Discard clean page-cache pages of every file under `path`.

    Non-root alternative to writing /proc/sys/vm/drop_caches: dirty pages
    are flushed by os.sync(), then POSIX_FADV_DONTNEED asks the kernel to
    drop each file's cached pages. Best-effort (pages pinned by other
    processes may survive), which is fine for benchmarking.
    """
    os.sync()
    for root, _, files in os.walk(path):
        for name in files:
            fd = os.open(os.path.join(root, name), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def timeit(fn, reps=REPS):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def report(name, secs):
    print(f"  {name:<38} {secs:7.3f} s   {N_ROWS / secs:>12,.0f} rows/s   {MB / secs:7,.0f} MB/s")


def main():
    print(machine_info())
    print(f"Data: {N_ROWS:,} rows x {DIM} float32 = {MB:,.0f} MB, batch={BATCH:,}, reps={REPS}\n")

    reset()
    torch.manual_seed(0)
    src = torch.randn(N_ROWS, DIM, dtype=DTYPE)

    # ------------------------------ write ------------------------------
    print("WRITE (sequential)")

    def write_blob():
        shutil.rmtree(f"{DATA}/blob", ignore_errors=True)
        with TensorBlob.open(f"{DATA}/blob", "w", dtype="float32", shape=DIM) as blob:
            for i in range(0, N_ROWS, BATCH):
                blob.write(src[i : i + BATCH])

    def write_monolith():
        path = f"{DATA}/monolith.mmap"
        if os.path.exists(path):
            os.remove(path)
        mm = MemoryMappedTensor.empty(N_ROWS, DIM, dtype=DTYPE, filename=path)
        for i in range(0, N_ROWS, BATCH):
            mm[i : i + BATCH] = src[i : i + BATCH]
        del mm  # release the mapping

    report("TensorBlob.write", timeit(write_blob))
    report("monolithic MemoryMappedTensor", timeit(write_monolith))

    # --------------------------- sequential read ---------------------------
    print("\nREAD (sequential, batched)")

    def read_blob():
        with TensorBlob.open(f"{DATA}/blob", "r") as blob:
            while blob.tell() < len(blob):
                blob.read(BATCH)

    def read_monolith():
        mm = MemoryMappedTensor.from_filename(
            f"{DATA}/monolith.mmap", dtype=DTYPE, shape=(N_ROWS, DIM)
        )
        for i in range(0, N_ROWS, BATCH):
            mm[i : i + BATCH].clone()

    def read_memory():
        for i in range(0, N_ROWS, BATCH):
            src[i : i + BATCH].clone()

    # Cold: reopen fresh each rep so the first pass faults pages in.
    report("TensorBlob (cold)", timeit(read_blob, reps=1))
    report("monolithic mmap (cold)", timeit(read_monolith, reps=1))
    # Disk-cold: page-cache pages discarded, so the pass reads from disk.
    drop_page_cache(f"{DATA}/blob")
    report("TensorBlob (disk-cold)", timeit(read_blob, reps=1))
    drop_page_cache(f"{DATA}/monolith.mmap")
    report("monolithic mmap (disk-cold)", timeit(read_monolith, reps=1))
    report("TensorBlob (warm)", timeit(read_blob))
    report("monolithic mmap (warm)", timeit(read_monolith))
    report("in-memory tensor (upper bound)", timeit(read_memory))

    reset()  # cleanup


if __name__ == "__main__":
    main()
