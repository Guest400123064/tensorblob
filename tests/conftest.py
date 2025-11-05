"""Common test fixtures for TensorBlob tests."""

import pytest
import torch
import shutil
from tensorblob import TensorBlob


@pytest.fixture
def temp_blob_dir(tmp_path):
    """Fixture providing a temporary directory for blob storage."""
    blob_dir = tmp_path / "test_blob"
    yield blob_dir
    if blob_dir.exists():
        shutil.rmtree(blob_dir)


@pytest.fixture
def sample_data():
    """Fixture providing sample tensor data (100x10)."""
    torch.manual_seed(42)
    return torch.randn(100, 10)


@pytest.fixture
def small_sample_data():
    """Fixture providing small sample tensor data (10x5)."""
    torch.manual_seed(123)
    return torch.randn(10, 5)


@pytest.fixture
def blob_with_data(tmp_path, sample_data):
    """Fixture providing a blob pre-populated with sample data."""
    blob_dir = tmp_path / "blob_with_data"
    with TensorBlob.open(blob_dir, "w", dtype="float32", shape=(10,)) as blob:
        blob.write(sample_data)
    yield blob_dir, sample_data
    if blob_dir.exists():
        shutil.rmtree(blob_dir)


@pytest.fixture
def multi_block_blob(tmp_path):
    """Fixture providing a blob with data spanning multiple blocks."""
    blob_dir = tmp_path / "multi_block_blob"
    block_size = 50
    data = torch.randn(150, 5)
    
    with TensorBlob.open(
        blob_dir, "w", dtype="float32", shape=(5,), block_size=block_size
    ) as blob:
        blob.write(data)
    
    yield blob_dir, data, block_size
    if blob_dir.exists():
        shutil.rmtree(blob_dir)

