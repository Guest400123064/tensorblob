"""Benchmark 05: Column projection (TensorDB pays only for what you read).

A TensorDB stores each field as an independent TensorBlob. This benchmark
shows the columnar benefit: reading only a cheap field (a scalar label)
costs a fraction of reading full rows, because the expensive field (a
768-dim embedding) is never touched. A monolithic row format cannot avoid
reading (or at least paging in) the expensive bytes.

Note: TensorDB intentionally has no projection API — projection is done by
opening the field's column blob directly (columns are plain TensorBlobs in
field-named subdirectories).

Data: 200,000 rows, fields:
  label : float32 (1,)    ->   0.8 MB total
  embed : float32 (768,)  -> 614 MB total

Metrics: wall time for a full sequential pass and for 10,000 random
single-row lookups, label-only vs full-row.
"""

import os
import platform
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from tensorblob import TensorBlob, TensorDB  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

N = 200_000
DIM = 768
BATCH = 8192
N_QUERIES = 10_000
SCHEMA = {"label": ("float32", 1), "embed": ("float32", DIM)}


def machine_info() -> str:
    with open("/proc/meminfo") as f:
        mem = int(f.readline().split()[1]) // 1024
    return (
        f"Machine: {platform.machine()} {platform.system()}, {os.cpu_count()} cores, "
        f"{mem} MiB RAM | python {platform.python_version()}, "
        f"torch {torch.__version__}"
    )


def main():
    print(machine_info())
    print(f"Data: {N:,} rows | label (1 float) vs embed (768 floats)\n")

    shutil.rmtree(DATA, ignore_errors=True)
    os.makedirs(DATA)
    torch.manual_seed(0)
    with TensorDB.open(f"{DATA}/db", "w", schema=SCHEMA) as db:
        for _ in range(10):
            db.write(
                {
                    "label": torch.randn(N // 10, 1),
                    "embed": torch.randn(N // 10, DIM),
                }
            )

    def seq_label_only() -> float:
        t0 = time.perf_counter()
        with TensorBlob.open(f"{DATA}/db/label", "r") as col:
            while col.tell() < len(col):
                col.read(BATCH)
        return time.perf_counter() - t0

    def seq_full_row() -> float:
        t0 = time.perf_counter()
        with TensorDB.open(f"{DATA}/db", "r") as db:
            while db.tell() < len(db):
                db.read(BATCH)
        return time.perf_counter() - t0

    t_label, t_full = seq_label_only(), seq_full_row()
    print("Sequential full pass:")
    print(f"  label only   {t_label:7.3f} s")
    print(f"  full row     {t_full:7.3f} s   ({t_full / t_label:5.1f}x more expensive)")

    g = torch.Generator().manual_seed(42)
    queries = torch.randint(N, (N_QUERIES,), generator=g).tolist()

    def rand_latency(fn) -> float:
        samples = []
        for q in queries:
            t0 = time.perf_counter()
            fn(q)
            samples.append(time.perf_counter() - t0)
        samples.sort()
        return samples[len(samples) // 2] * 1e6

    with TensorBlob.open(f"{DATA}/db/label", "r") as col:
        med_label = rand_latency(lambda q: col[q])
    with TensorDB.open(f"{DATA}/db", "r") as db:
        med_full = rand_latency(lambda q: db[q])

    print(f"\nRandom single-row lookup (median over {N_QUERIES:,}):")
    print(f"  label only   {med_label:8.1f} us")
    print(f"  full row     {med_full:8.1f} us")

    shutil.rmtree(DATA)  # cleanup


if __name__ == "__main__":
    main()
