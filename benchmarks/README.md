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

- 12-core x86_64 Linux, 16 GiB RAM, kernel page cache available
- Python 3.10.12, torch 2.13.0 (CPU ops, 6 threads)
- Data model unless noted: float32 rows of 768 dims (~3 KB/row), matching a
  typical embedding store

Absolute numbers will vary with hardware; the ratios are the interesting
part.

## 01 — Raw I/O throughput

500,000 rows (~1.5 GiB), batched at 8,192 rows, median of 3 reps.

| Operation | Time | Throughput |
|---|---|---|
| TensorBlob write | 9.3 s | 165 MB/s |
| Monolithic mmap write | 14.2 s | 108 MB/s |
| TensorBlob sequential read (warm) | 0.68 s | 2.2 GB/s |
| Monolithic mmap sequential read (warm) | 0.33 s | 4.7 GB/s |
| In-memory slice (upper bound) | 0.21 s | 7.2 GB/s |

**Interpretation.** Writes are at parity with (here, faster than) a single
monolithic mmap file — splitting data across ~61 block files does not hurt
sequential write throughput. Sequential reads run at ~2.2 GB/s, about half
the monolithic mmap rate: the per-batch block lookup, bounds handling and
copy-out cost roughly 2x. Still comfortably GB/s-scale.

**Caveats.** "Cold" reads were measured right after writing, so the page
cache was already warm — true disk-cold numbers require dropping OS caches
(root). The monolith write number includes file allocation of one 1.5 GiB
file per rep.

## 02 — Random access latency

10,000 random single-row lookups (fixed seed) on a 500k-row blob:

| Access | Median | p99 |
|---|---|---|
| TensorBlob `blob[i]` | 30 µs | 126 µs |
| In-memory `src[i].clone()` | 4.9 µs | 44 µs |

Random batch gather (64 batches × 512 rows):

| Access | Per batch |
|---|---|
| TensorBlob row-by-row (`[blob[i] for i in idxs]`) | 17.1 ms |
| TensorBlob vectorized `blob[idxs]` | 2.5 ms |
| In-memory fancy indexing | 0.65 ms |

**Interpretation.** Single-row random access costs tens of microseconds —
negligible next to any model computation. Batch gather uses the vectorized
fancy-indexing API (`blob[idxs]`, accepts list/tuple/torch.Tensor): rows
are grouped by block, gathered with one torch call per block, and
scattered back to input order — ~7x faster than row-by-row. Residual gap
to in-memory is the per-block cache lookup and concatenation.

Knob sweep (200k rows per config, median single-row latency):

| block_size | max_cached_blocks | Median | p99 |
|---|---|---|---|
| 1,024 | 16 | 136 µs | 339 µs |
| 1,024 | 256 | 29 µs | 96 µs |
| 1,024 | 4,096 | 29 µs | 100 µs |
| 8,192 | 16 | 34 µs | 218 µs |
| 8,192 | 256 | 28 µs | 76 µs |
| 65,536 | 16 | 28 µs | 98 µs |
| 65,536 | 4,096 | 29 µs | 120 µs |

**Interpretation.** An undersized cache still costs ~4-5x median latency
(136 µs vs 29 µs) from constant remapping, but the catastrophic cliff is
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
| Raw (re-preprocess every epoch) | 2.88 s | 14.4 s |
| Compiled blob (read-only epochs) | 0.18 s | 4.2 s (incl. 3.3 s one-time compile) |

**Interpretation.** Compiled epochs are ~16x cheaper; including the
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
| Sequential pass, label only | 0.063 s |
| Sequential pass, full row | 0.399 s (6.3x) |
| Random lookup, label only (median) | 24 µs |
| Random lookup, full row (median) | 58 µs |

**Interpretation.** Because each field lives in its own blob, reading only
the cheap field never touches the expensive field's pages — projection is
~6x cheaper sequentially (and note `label`-only latency is even lower than
the 02 baseline since its blocks are tiny). A monolithic row format cannot
skip the expensive bytes. Projection is done by opening the field's column
blob directly (`TensorBlob.open("db/label")`); `TensorDB` deliberately has
no projection API.

## Known performance limitations (candidates for future work)

1. **Sequential read ~2x vs monolithic mmap** — per-batch block lookup and
   copy overhead; could be reduced with a fast path for block-aligned
   contiguous reads.

## Addressed in the current revision

1. ~~No vectorized batch indexing~~ — `blob[idxs]` now accepts a 1-D
   integer sequence (list/tuple/torch.Tensor) with torch fancy-indexing
   semantics (~7x faster batch gathers; see 02).
2. ~~`gc.collect()` on every LRU eviction~~ — removed; eviction relies on
   reference counting, collapsing the thrash cliff from ~73 ms to ~136 µs
   (see 02).
