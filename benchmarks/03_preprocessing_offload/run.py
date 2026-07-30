"""Benchmark 03: Preprocessing offload ("compile once, train many").

Simulates a training loop over E epochs and compares two pipelines:

  raw       : each epoch reads raw uint8 samples from a memory-mapped file
              and re-runs the preprocessing on the fly (what a naive
              DataLoader does every epoch)
  compiled  : preprocessing runs ONCE into a TensorBlob; each epoch just
              reads ready-to-use float32 batches

Synthetic preprocessing (CPU-bound torch ops): uint8 -> float, per-channel
normalize, random horizontal flip, 3x3 average pool.

Data: 50,000 "images" of shape (3, 64, 64) uint8 (~600 MB raw).

Metrics: one-time compilation cost, per-epoch time for both pipelines,
total time over E epochs, and the break-even point (epochs needed for the
compiled pipeline to pay back its compilation cost).
"""

import os
import platform
import shutil
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from tensordict import MemoryMappedTensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from tensorblob import TensorBlob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

N = 50_000
SHAPE = (3, 64, 64)
BATCH = 256
EPOCHS = 5
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OUT_SHAPE = (3, 31, 31)  # after 3x3 avg pool with stride 2


def machine_info() -> str:
    with open("/proc/meminfo") as f:
        mem = int(f.readline().split()[1]) // 1024
    return (
        f"Machine: {platform.machine()} {platform.system()}, {os.cpu_count()} cores, "
        f"{mem} MiB RAM | python {platform.python_version()}, "
        f"torch {torch.__version__}"
    )


def preprocess(batch_u8: torch.Tensor, flip: bool) -> torch.Tensor:
    x = batch_u8.float() / 255.0
    x = (x - MEAN) / STD
    if flip:
        x = torch.flip(x, dims=[-1])
    return F.avg_pool2d(x, kernel_size=3, stride=2)


def main():
    print(machine_info())
    print(
        f"Data: {N:,} uint8 samples {SHAPE}, batch={BATCH}, epochs={EPOCHS}, "
        f"torch threads={torch.get_num_threads()}\n"
    )

    shutil.rmtree(DATA, ignore_errors=True)
    os.makedirs(DATA)
    torch.manual_seed(0)
    raw_path = f"{DATA}/raw.mmap"
    mm = MemoryMappedTensor.empty(N, *SHAPE, dtype=torch.uint8, filename=raw_path)
    for i in range(0, N, 4096):
        mm[i : i + 4096] = torch.randint(0, 256, (min(4096, N - i), *SHAPE), dtype=torch.uint8)
    del mm

    # ------------------------- raw pipeline -------------------------
    def epoch_raw(epoch: int) -> float:
        t0 = time.perf_counter()
        raw = MemoryMappedTensor.from_filename(raw_path, dtype=torch.uint8, shape=(N, *SHAPE))
        g = torch.Generator().manual_seed(epoch)
        for i in range(0, N, BATCH):
            batch = raw[i : i + BATCH].clone()
            out = preprocess(batch, flip=bool(torch.rand(1, generator=g) > 0.5))
            _ = out.mean()  # force materialization, stand-in for "model step"
        return time.perf_counter() - t0

    raw_epochs = [epoch_raw(e) for e in range(EPOCHS)]

    # ----------------------- compiled pipeline -----------------------
    t0 = time.perf_counter()
    with TensorBlob.open(f"{DATA}/compiled", "w", dtype="float32", shape=OUT_SHAPE) as blob:
        raw = MemoryMappedTensor.from_filename(raw_path, dtype=torch.uint8, shape=(N, *SHAPE))
        g = torch.Generator().manual_seed(0)
        for i in range(0, N, 4096):
            chunk = preprocess(raw[i : i + 4096].clone(), flip=bool(torch.rand(1, generator=g) > 0.5))
            blob.write(chunk)
    compile_time = time.perf_counter() - t0

    def epoch_compiled() -> float:
        t0 = time.perf_counter()
        with TensorBlob.open(f"{DATA}/compiled", "r") as blob:
            while blob.tell() < len(blob):
                out = blob.read(BATCH)
                _ = out.mean()
        return time.perf_counter() - t0

    compiled_epochs = [epoch_compiled() for _ in range(EPOCHS)]

    # --------------------------- report ---------------------------
    med_raw = statistics.median(raw_epochs)
    med_cmp = statistics.median(compiled_epochs)
    print(f"per-epoch raw pipeline      : {[f'{t:.2f}' for t in raw_epochs]} s (median {med_raw:.2f})")
    print(f"per-epoch compiled pipeline : {[f'{t:.2f}' for t in compiled_epochs]} s (median {med_cmp:.2f})")
    print(f"one-time compilation cost   : {compile_time:.2f} s")
    total_raw = med_raw * EPOCHS
    total_cmp = compile_time + med_cmp * EPOCHS
    print(f"\ntotal over {EPOCHS} epochs (median):")
    print(f"  raw       {total_raw:7.2f} s")
    print(f"  compiled  {total_cmp:7.2f} s   ({total_raw / total_cmp:.2f}x faster)")
    if med_raw > med_cmp:
        be = compile_time / (med_raw - med_cmp)
        print(f"  break-even: compiled pipeline pays off after ~{be:.1f} epochs")

    shutil.rmtree(DATA)  # cleanup


if __name__ == "__main__":
    main()
