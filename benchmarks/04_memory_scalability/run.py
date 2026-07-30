"""Benchmark 04: Memory scalability (bounded footprint via LRU cache).

Validates the "larger than RAM" claim: a blob with many blocks is accessed
randomly while the process's resident memory (VmRSS) and kernel VMA count
(number of entries in /proc/self/maps) are tracked. With a bounded
max_cached_blocks, both should stay flat regardless of how many distinct
blocks are touched.

Data: 128,000 rows x 768 float32 split into 2,000 blocks of 64 rows
(~393 MB total). Each configuration touches 5,000 random rows, which with
random spread hits most of the 2,000 blocks.

Metrics: VmRSS delta (MiB) and VMA count delta after the access loop,
for max_cached_blocks in {16, 128, 100000 (effectively unbounded)}.
"""

import os
import platform
import shutil
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from tensorblob import TensorBlob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

N_ROWS = 128_000
DIM = 768
BLOCK_SIZE = 64  # -> 2,000 blocks
N_QUERIES = 5_000


def machine_info() -> str:
    with open("/proc/meminfo") as f:
        mem = int(f.readline().split()[1]) // 1024
    return (
        f"Machine: {platform.machine()} {platform.system()}, {os.cpu_count()} cores, "
        f"{mem} MiB RAM | python {platform.python_version()}, "
        f"torch {torch.__version__}"
    )


def vmrss_mib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS not found")


def vma_count() -> int:
    with open(f"/proc/{os.getpid()}/maps") as f:
        return sum(1 for _ in f)


def main():
    print(machine_info())
    print(
        f"Data: {N_ROWS:,} rows x {DIM} float32 in {N_ROWS // BLOCK_SIZE:,} blocks "
        f"of {BLOCK_SIZE} rows | {N_QUERIES:,} random accesses per config\n"
    )

    shutil.rmtree(DATA, ignore_errors=True)
    os.makedirs(DATA)
    torch.manual_seed(0)
    with TensorBlob.open(
        f"{DATA}/blob", "w", dtype="float32", shape=DIM, block_size=BLOCK_SIZE
    ) as blob:
        for _ in range(20):  # chunked writes to bound peak RAM
            blob.write(torch.randn(N_ROWS // 20, DIM, dtype=torch.float32))

    # Reclaim write-phase mappings so they don't pollute the measurements
    # (the writer blob is still bound above and must be dropped first).
    import gc

    del blob
    gc.collect()

    g = torch.Generator().manual_seed(42)
    queries = torch.randint(N_ROWS, (N_QUERIES,), generator=g).tolist()

    print(f"  {'max_cached_blocks':>18} {'VmRSS delta':>12} {'VMA delta':>10}")
    base_rss, base_vma = vmrss_mib(), vma_count()
    print(f"  {'(baseline, no access)':>18} {base_rss:>10.1f}MiB {base_vma:>10}")

    for cache in [16, 128, 100_000]:
        rss0, vma0 = vmrss_mib(), vma_count()
        with TensorBlob.open(f"{DATA}/blob", "r", max_cached_blocks=cache) as blob:
            for q in queries:
                _ = blob[q]
            rss1, vma1 = vmrss_mib(), vma_count()
        print(f"  {cache:>18,} {rss1 - rss0:>+9.1f}MiB {vma1 - vma0:>+10}")

    shutil.rmtree(DATA)  # cleanup


if __name__ == "__main__":
    main()
