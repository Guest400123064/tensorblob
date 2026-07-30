# Benchmarks

Synthetic-data performance benchmarks for `tensorblob`. Each subdirectory is
self-contained: `run.py` executes the benchmark, prints results, and cleans
up its temporary data; `results.txt` captures a reference run.

Run from the repository root:

```bash
uv run python -u benchmarks/<name>/run.py
```

Run them **sequentially** — they are I/O benchmarks and concurrent runs
skew each other's timings.

| Benchmark | Question it answers |
|---|---|
| `01_raw_io_throughput` | How fast is raw write/sequential-read vs a monolithic memory-mapped file and vs in-memory access? |
| `02_random_access_latency` | What do random single-row lookups cost, and how do `block_size` / `max_cached_blocks` affect it? |
| `03_preprocessing_offload` | How much does "compile once into a blob" save vs re-preprocessing every epoch, and when does it break even? |
| `04_memory_scalability` | Do RSS and kernel VMA count stay bounded when randomly accessing many-block blobs (LRU cache)? |
| `05_column_projection` | How much cheaper is reading one cheap field vs full rows in a `TensorDB` (columnar layout)? |

## Reference environment

- 12-core x86_64, 16 GiB RAM, **HDD via WSL2** (benchmark data on the WSL
  ext4 filesystem, *not* `/mnt/c` — DrvFS/9P would skew I/O numbers badly),
  kernel page cache available
- Python 3.10.12, torch 2.13.0 (CPU ops, 6 threads)
- Reference run: commit `e88caea` (`dev`, includes the strict write-dtype
  check), 2026-07-30; each `results.txt` header records the same metadata
- Data model unless noted: float32 rows of 768 dims (~3 KB/row), matching a
  typical embedding store

Absolute numbers will vary with hardware; the ratios are the interesting
part. On WSL2 in particular, occasional multi-second VM-level stalls can
corrupt individual measurements — if a number looks like an outlier,
rerun the suite before believing it (we observed one such stall inflate a
batch-gather measurement ~50x; an isolated rerun reproduced the expected
value exactly).

## 01 — Raw I/O throughput

500,000 rows (~1.5 GiB), batched at 8,192 rows, median of 3 reps.

| Operation | Time | Throughput |
|---|---|---|
| TensorBlob write | 1.9 s | 819 MB/s |
| Monolithic mmap write | 13.7 s | 112 MB/s |
| TensorBlob sequential read (disk-cold) | 55.6 s | 28 MB/s |
| Monolithic mmap sequential read (disk-cold) | 41.4 s | 37 MB/s |
| TensorBlob sequential read (warm) | 0.51 s | 3.0 GB/s |
| Monolithic mmap sequential read (warm) | 0.19 s | 8.0 GB/s |
| In-memory slice (upper bound) | 0.16 s | 9.9 GB/s |

**Interpretation.** Writes are at parity with (here, much faster than) a
single monolithic mmap file — splitting data across ~61 block files does
not hurt sequential write throughput. Warm sequential reads run at
~3 GB/s, about 40% of the monolithic mmap rate: the per-batch block
lookup, bounds handling and copy-out cost roughly 2.5x. The strict
write-dtype check (one enum comparison per `write()` call) has no
measurable cost.

**Disk-cold reads** (page cache discarded via `POSIX_FADV_DONTNEED`, a
non-root alternative to `/proc/sys/vm/drop_caches`) collapse to ~28-37
MB/s for *both* formats — 100x below warm. This is not the block
abstraction's fault (TensorBlob is within ~1.3x of the monolith): on HDD,
demand-paged mmap reads fault in small chunks with little effective
readahead, so a "sequential" scan behaves like random small I/O and pays
a seek per chunk. Practical consequence: at training time this only
affects the first pass over a cold dataset; epochs are long and the page
cache warms up quickly, so it is amortized away — but first-epoch latency
on HDD is disk-bound no matter the library.

**Caveats.** All write and warm-read figures are page-cache measurements:
1.5 GiB fits easily in 16 GiB RAM, so the write number reflects
write-back caching (and varies substantially run to run: we have observed
380-820 MB/s), not the HDD's sustained ~100-150 MB/s — visible in the
monolithic number, which pays file allocation per rep.

## 02 — Random access latency

10,000 random single-row lookups (fixed seed) on a 500k-row blob:

| Access | Median | p99 |
|---|---|---|
| TensorBlob `blob[i]` (disk-cold) | 32 µs | ~510 ms |
| TensorBlob `blob[i]` | 30 µs | 98 µs |
| In-memory `src[i].clone()` | 4.5 µs | 15 µs |

Random batch gather (64 batches × 512 rows):

| Access | Per batch |
|---|---|
| TensorBlob row-by-row (`[blob[i] for i in idxs]`) | 15.7 ms |
| TensorBlob vectorized `blob[idxs]` | 2.3 ms |
| In-memory fancy indexing | 0.26 ms |

**Interpretation.** Single-row random access costs tens of microseconds —
negligible next to any model computation. The disk-cold row (page cache
discarded via `POSIX_FADV_DONTNEED`, 2,000 queries) shows the HDD access
pattern is **bimodal**: the median barely moves (32 µs vs 30 µs warm)
because each first fault pulls in a kernel readahead window that covers
many neighboring rows, so most queries hit just-faulted pages — while the
~1% of queries that do touch a new region pay the full disk cost directly
(p99 ≈ 0.5 s). In training terms: cold-cache random access is a tail
latency problem, not an average-case one, and it disappears once the
working set is warm. Batch gather uses the vectorized
fancy-indexing API (`blob[idxs]`, accepts list/tuple/torch.Tensor): rows
are grouped by block, gathered with one torch call per block, and
scattered back to input order — ~7x faster than row-by-row. Residual gap
to in-memory is the per-block cache lookup and concatenation.

Knob sweep (200k rows per config, median single-row latency):

| block_size | max_cached_blocks | Median | p99 |
|---|---|---|---|
| 1,024 | 16 | 139 µs | 260 µs |
| 1,024 | 256 | 30 µs | 99 µs |
| 1,024 | 4,096 | 29 µs | 96 µs |
| 8,192 | 16 | 36 µs | 226 µs |
| 8,192 | 256 | 29 µs | 67 µs |
| 8,192 | 4,096 | 29 µs | 64 µs |
| 65,536 | 16 | 27 µs | 66 µs |
| 65,536 | 256 | 27 µs | 79 µs |
| 65,536 | 4,096 | 28 µs | 69 µs |

**Interpretation.** An undersized cache still costs ~5x median latency
(139 µs vs 29 µs) from constant remapping, but the catastrophic cliff is
gone: evicted blocks are unmapped by reference counting alone, so eviction
no longer triggers a full `gc.collect()` per miss. (In an earlier revision
of the library, the 1,024/16 cell measured **72.8 ms** — a ~2,400x
degradation caused by per-eviction garbage collection.) Beyond the
working-set size, bigger caches add nothing.

## 03 — Preprocessing offload

50,000 uint8 "images" (3, 64, 64) (~600 MB raw); preprocessing = float
conversion + normalize + random flip + 3x3 avg-pool; batch 256, 5 epochs.

| Pipeline | Per-epoch (median) | Total over 5 epochs |
|---|---|---|
| Raw (re-preprocess every epoch) | 2.55 s | 12.8 s |
| Compiled blob (read-only epochs) | 0.14 s | 3.6 s (incl. 2.9 s one-time compile) |

**Interpretation.** Compiled epochs are ~18x cheaper; including the
one-time compilation cost the pipeline breaks even after **~1.2 epochs**
and is ~3.5x faster over 5 epochs. The benefit grows linearly with epoch
count and with preprocessing complexity — this is the core value of the
"compile once, train many" workflow.

## 04 — Memory scalability

128,000 rows in 2,000 blocks of 64 rows; 5,000 random accesses per
configuration; VmRSS and VMA count (`/proc/self/maps`) measured after the
loop relative to a GC'd baseline.

| max_cached_blocks | VmRSS delta | VMA delta |
|---|---|---|
| 16 | +1.3 MiB | +16 |
| 128 | +7.3 MiB | +112 |
| 100,000 (unbounded) | +203 MiB | +1,704 |

**Interpretation.** Resident memory and kernel VMA usage track
`max_cached_blocks` almost exactly — the LRU bound works as designed, so
blobs far larger than RAM can be accessed with a flat footprint.

**Implementation note.** Closing a blob does not unmap its cached blocks;
they are reclaimed by CPython garbage collection. Code that measures or
limits memory should `del` blob references and `gc.collect()` (the
benchmark does this after its write phase).

## 05 — Column projection (TensorDB)

200,000 rows with fields `label` (1 float, 0.8 MB total) and `embed`
(768 floats, 614 MB total).

| Access | Time |
|---|---|
| Sequential pass, label only | 0.062 s |
| Sequential pass, full row | 0.301 s (4.9x) |
| Random lookup, label only (median) | 23 µs |
| Random lookup, full row (median) | 56 µs |

**Interpretation.** Because each field lives in its own blob, reading only
the cheap field never touches the expensive field's pages — projection is
~5x cheaper sequentially (and note `label`-only latency is even lower than
the 02 baseline since its blocks are tiny). A monolithic row format cannot
skip the expensive bytes. Projection is done by opening the field's column
blob directly (`TensorBlob.open("db/label")`); `TensorDB` deliberately has
no projection API.

## Known performance limitations (candidates for future work)

1. **Sequential read ~2.5x vs monolithic mmap** — per-batch block lookup
   and copy overhead; could be reduced with a fast path for block-aligned
   contiguous reads.

## Addressed in the current revision

1. ~~No vectorized batch indexing~~ — `blob[idxs]` now accepts a 1-D
   integer sequence (list/tuple/torch.Tensor) with torch fancy-indexing
   semantics (~7x faster batch gathers; see 02).
2. ~~`gc.collect()` on every LRU eviction~~ — removed; eviction relies on
   reference counting, collapsing the thrash cliff from ~73 ms to ~136 µs
   (see 02).
