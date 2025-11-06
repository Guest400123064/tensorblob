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

## Contributing

Contributions welcome! Please submit a Pull Request.

## License

Apache License 2.0 - see LICENSE file for details.
