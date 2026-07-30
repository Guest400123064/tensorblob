[![Python 3.10](https://img.shields.io/badge/python-%203.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![test](https://github.com/Guest400123064/tensorblob/actions/workflows/test.yaml/badge.svg)](https://github.com/Guest400123064/tensorblob/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/Guest400123064/tensorblob/branch/main/graph/badge.svg?token=K00BM34OCO)](https://codecov.io/gh/Guest400123064/tensorblob)
[![PyPI](https://img.shields.io/pypi/v/tensorblob)](https://pypi.org/project/tensorblob/)

# tensorblob

A lightweight, dynamic-sized, memory-mapped tensor storage with file-like APIs, while also supporting integer indexing and slicing, built with `MemoryMappedTensor` from [`tensordict`](https://github.com/pytorch/tensordict).

## Features

- 🔗 **Memory-mapped storage**: Efficient storage of large collections of same-shaped tensors
- 💾 **File-like APIs**: Read, write, and seek like a file, while also supporting integer indexing and slicing
- ⚡ **Dynamic-sized**: No need to specify the total number of tensors upfront
- 🔄 **Extend and truncate**: Extend the blob with another blob or truncate the blob to a specific position
- 🚀 **LRU cache**: Automatic management of memory-mapped blocks for scalability with large blobs
- 🧩 **Multi-field databases**: `TensorDB` manages several row-aligned blobs for heterogeneous data, e.g., multivariate time series or event streams

## Installation

From PyPI:

```bash
pip install tensorblob
```

If you are interested in the experimental (i.e., unstable and undertested) version, you can install it from GitHub:

```bash
pip install git+https://github.com/Guest400123064/tensorblob.git
```

## Core Use Cases

### Quick Start

The example below shows how to create a new storage for a collection of randomly generated fake embeddings, and how to access them by index. Since the storage is memory-mapped, no need to read all tensors into memory; just access them by index.

```python
import torch
from tensorblob import TensorBlob

# Create a new storage for a collection of randomly generated fake embeddings;
# need to specify the data type and shape of each tensor for creation
with TensorBlob.open("embeddings.blob", "w", dtype="float32", shape=768) as blob:
    blob.write(torch.randn(100_000, 768))
    print(f"Wrote {len(blob)} embeddings")

# No need to specify the configurations again after creation
with TensorBlob.open("embeddings.blob", "r") as blob:
    e1 = blob[42]
    e2 = blob[-1:16384:-12345]
    print(f"Similarity: {torch.cosine_similarity(e1, e2)}")
```

### Processing Large Datasets

Store and preprocess datasets larger than RAM using memory mapping can be useful to accelerate the training process by reducing the time spent on data loading and transformation.

```python
with TensorBlob.open("data/images.blob", "w", dtype="float32", shape=(3, 224, 224)) as blob:
    for image_batch in data_loader:
        blob.write(preprocess(image_batch))

with TensorBlob.open("data/images.blob", "r") as blob:
    for image in blob:
        result = model(image)
```

### Incremental Data Collection

Append new data to existing blobs can be useful with streaming data collection.

```python
with TensorBlob.open("positions.blob", "w", dtype="float32", shape=3) as blob:
    blob.write(initial_position)

# Later: append more data by opening the blob in append mode
with TensorBlob.open("positions.blob", "a") as blob:
    for pos in trajectory_queue.get():
        blob.write(pos)
    print(f"Total trajectory recorded: {len(blob)}")
```

### Random Access and Updates with File-Like APIs

Read and modify specific tensors starting from a specific position.

```python
import io

with TensorBlob.open("data/features.blob", "r+") as blob:
    blob.seek(1000)
    print(f"Current position: {blob.tell()}")

    batch = blob.read(size=100)
    print(f"Read {batch.shape} tensors")

    # Update specific positions, whence is also supported
    blob.seek(-500, whence=io.SEEK_END)
    blob.write(updated_features)
    
    # Append new data
    blob.seek(len(blob))
    blob.write(additional_features)
```

### Extend and Truncate

Extend the blob with another blob or truncate the blob to a specific position. Extension could be useful if we want to merge two blobs into one, e.g., results from two different processes. Note that extension operation does not delete the original data.

```python
with TensorBlob.open("data/features.blob", "a") as blob:
    blob.extend(other_blob)

# Extension without maintaining the order is faster
with TensorBlob.open("data/features.blob", "r+") as blob:
    blob.extend(other_blob, maintain_order=False)

with TensorBlob.open("data/features.blob", "r+") as blob:
    blob.truncate(1000)
    print(f"Truncated to {len(blob)} tensors")
```

### Heterogeneous Data with TensorDB

For multi-modal or multi-field data (e.g., multivariate time series, event streams), `TensorDB` manages several `TensorBlob`s under the hood — one per field — with row orders always aligned. Each field has its own dtype and shape, and each field's storage gets its own independent LRU cache and block files.

```python
from tensorblob import TensorDB

# Create a database with a fixed schema mapping field names to (dtype, shape)
with TensorDB.open("events.db", "w",
                   schema={"price": ("float32", 1),
                           "embed": ("float16", 768)}) as db:
    # Rows are dense: every write must supply every field with the same row count
    db.write({"price": torch.randn(100_000, 1),
              "embed": torch.randn(100_000, 768).half()})
    print(f"Wrote {len(db)} rows")

# No need to specify the schema again after creation
with TensorDB.open("events.db", "r") as db:
    row = db[42]          # {"price": tensor of shape (1,), "embed": (768,)}
    batch = db[10:100]    # {"price": (90, 1), "embed": (90, 768)}
    print(f"Fields: {list(batch)}, price range: {batch['price'].min()}..{batch['price'].max()}")
```

`TensorDB` supports the same file-like APIs as `TensorBlob`, applied row-wise across all fields:

```python
with TensorDB.open("events.db", "r+") as db:
    db.seek(1000)
    batch = db.read(size=100)                 # dict of (100, ...) tensors

    db.seek(-500, whence=io.SEEK_END)
    db.write({"price": new_prices, "embed": new_embeds})  # overwrite in place

    db.truncate(10_000)                       # truncate all fields at once
    db.extend(other_db, maintain_order=False) # merge another db with the same schema

# Cleanup removes the whole database directory
TensorDB.unlink("events.db")
```

**Consistency guarantee**: a write commits the row count only after all fields are written. If a crash interrupts a write mid-way, the next open reports the last committed (fully written) row count, and writable opens automatically truncate the stray partial rows, so row alignment is always preserved.

## Performance and Scalability

### Memory Management

TensorBlob uses an LRU (Least Recently Used) cache to manage memory-mapped blocks efficiently. This allows you to work with blobs containing millions of tensors without loading everything into memory.

**Default behavior:**

- Automatically caches up to ~4,000 blocks (1/16 of system's VMA limit)
- Blocks loaded on-demand when accessed
- Least recently used blocks automatically evicted when cache is full

**For large-scale workloads:**

```python
# Increase cache for better random access performance
with TensorBlob.open("large.blob", "r", max_cached_blocks=10_000) as blob:
    for idx in random_indices:
        tensor = blob[idx]  # Cached blocks reused efficiently

# Decrease cache for memory-constrained environments
with TensorBlob.open("data.blob", "r", max_cached_blocks=100) as blob:
    for tensor in blob:  # Sequential access works fine with small cache
        process(tensor)
```

**Performance tips:**

- Sequential access patterns work well with any cache size
- Random access benefits from larger cache sizes — but **do not undersize the cache for random workloads**: when the random working set exceeds `max_cached_blocks`, every access evicts and remaps a block, degrading lookup latency several-fold (~30 µs → ~140 µs in our benchmarks). If in doubt, increase the cache or the block size
- Each cached block consumes ~200 bytes of kernel memory (VMA overhead)
- System limit: typically ~65,000 memory-mapped regions per process
- To avoid frequent cache evictions, one can also increase the block size to reduce the total number of blocks
- For random batches, use vectorized batch indexing `blob[idxs]` (list, tuple, or 1-D torch.Tensor of row indices) instead of gathering row by row — it is ~7x faster; contiguous slices are faster still, so pre-sorting indices helps when order is flexible

### Benchmarks

Headline numbers from the synthetic benchmark suite (500k × 768-dim float32 rows, 12-core x86_64, 16 GiB RAM, HDD via WSL2 with warm page cache; see [`benchmarks/`](benchmarks/) for full analysis and reproducible scripts):

| Measurement | Result |
|---|---|
| Sequential write throughput | ~0.4-0.8 GB/s (~123-267k rows/s), page-cache absorbed |
| Sequential read throughput (warm) | ~3.0 GB/s (~973k rows/s), vs ~9.9 GB/s in-memory upper bound |
| Sequential read throughput (disk-cold, HDD) | ~28 MB/s — demand-paged mmap is seek-bound when cold; amortized away after the first pass |
| Random single-row lookup | ~30 µs median (in-memory: ~5 µs); disk-cold is bimodal — median unchanged, ~0.5 s p99 on first touches |
| Random batch gather (512 rows) | ~2.3 ms vectorized vs ~16 ms row-by-row (~7x) |
| Preprocessing offload (5 epochs) | ~3.5x faster than re-preprocessing; breaks even after ~1.2 epochs |
| Memory footprint | bounded by `max_cached_blocks`; +16 VMAs / +1.3 MiB RSS at cache size 16 |
| TensorDB column projection | reading one cheap field only is ~5x cheaper than full-row reads |

## Contributing

Contributions welcome! Please submit a Pull Request.

## License

Apache License 2.0 - see LICENSE file for details.
