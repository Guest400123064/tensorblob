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

```python
import torch
from tensorblob import TensorBlob


# Create a new storage for a collection of randomly generated fake embeddings;
# need to specify the data type and shape of each tensor for creation
with TensorBlob.open("embeddings.blob", "w", dtype="float32", shape=768) as blob:
    blob.write(torch.randn(100_000, 768))
    print(f"Wrote {len(blob)} embeddings")

# No need to read all tensors into memory; just access them by index
with TensorBlob.open("embeddings.blob", "r") as blob:
    e1 = blob[42]
    e2 = blob[-1]
    print(f"Similarity: {torch.cosine_similarity(e1, e2, dim=0)}")
```

### Processing Large Datasets

Store and process datasets larger than RAM using memory mapping:

```python
# Store a large image dataset
with TensorBlob.open("data/images.blob", "w",
                      dtype=torch.float32,
                      shape=(3, 224, 224),
                      block_size=512) as blob:
    for batch in data_loader:
        # Process and write batch by batch
        processed = preprocess(batch)
        blob.write(processed)

# Iterate through dataset without loading everything into memory
with TensorBlob.open("data/images.blob", "r") as blob:
    for image in blob:
        # Process one image at a time
        result = model(image.unsqueeze(0))
```

### Incremental Data Collection

Append new data to existing blobs:

```python
# Initial data collection
with TensorBlob.open("logs/activations.blob", "w",
                      dtype="float32", shape=(1024,)) as blob:
    blob.write(initial_activations)

# Later: append more data
with TensorBlob.open("logs/activations.blob", "a") as blob:
    blob.write(new_activations)
    print(f"Total activations: {len(blob)}")
```

### Random Access and Updates

Read and modify specific tensors:

```python
with TensorBlob.open("data/features.blob", "r+") as blob:
    # Read specific range
    blob.seek(1000)
    batch = blob.read(size=100)
    
    # Update specific positions
    blob.seek(500)
    blob.write(updated_features)
    
    # Append new data
    blob.seek(len(blob))
    blob.write(additional_features)
```

### File-Like Operations

Use familiar file-like APIs:

```python
with TensorBlob.open("data/tensors.blob", "r") as blob:
    # Get current position
    pos = blob.tell()
    
    # Seek to different positions
    blob.seek(0)           # Start
    blob.seek(100)         # Absolute position
    blob.seek(10, 1)       # Relative: +10 from current
    blob.seek(-50, 2)      # From end: 50 back from end
    
    # Iterate from current position
    for tensor in blob:
        process(tensor)
        if should_stop():
            break
    
    # Index access
    first = blob[0]
    last = blob[-1]  # Not supported yet, use blob[len(blob)-1]
```

### Real-World Example: Training Data Pipeline

```python
# Setup: Create blob from raw data
def create_training_blob(raw_data_path, blob_path):
    with TensorBlob.open(blob_path, "w",
                          dtype="float32", 
                          shape=(512,),
                          block_size=4096) as blob:
        for file in sorted(Path(raw_data_path).glob("*.pt")):
            data = torch.load(file)
            blob.write(data)
    print(f"Created blob with {len(blob)} samples")

# Training: Efficient data loading
class BlobDataset(torch.utils.data.IterableDataset):
    def __init__(self, blob_path):
        self.blob_path = blob_path
    
    def __iter__(self):
        # Each worker gets its own blob instance
        with TensorBlob.open(self.blob_path, "r") as blob:
            for tensor in blob:
                yield tensor

# Use in training loop
dataset = BlobDataset("data/train.blob")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

for batch in dataloader:
    loss = train_step(batch)
```


## Contributing

Contributions welcome! Please submit a Pull Request.

## License

Apache License 2.0 - see LICENSE file for details.
