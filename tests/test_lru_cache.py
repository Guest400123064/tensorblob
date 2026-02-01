"""Tests for LRU cache functionality."""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from tensorblob import TensorBlob


class TestLRUCache:
    """Test LRU cache for memory-mapped blocks."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test blobs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    def test_cache_limits_loaded_blocks(self, temp_dir):
        """Test that cache limits number of loaded blocks."""
        blob_path = temp_dir / "test_cache"

        # Create blob with many blocks (small block_size)
        with TensorBlob.open(
            blob_path,
            "w",
            dtype="float32",
            shape=(10,),
            block_size=10,  # Only 10 tensors per block
            max_cached_blocks=3,  # Only cache 3 blocks
        ) as blob:
            # Write 50 tensors (will create 5 blocks)
            tensors = torch.randn(50, 10)
            blob.write(tensors)

            # Initially, blocks are created as needed
            # After writing, we should have 5 blocks total
            assert len(blob._status.bds) == 5

            # But cache should have at most 3 blocks
            assert len(blob._memmap) <= 3

    def test_lru_eviction_on_access(self, temp_dir):
        """Test that LRU eviction works when accessing blocks."""
        blob_path = temp_dir / "test_lru"

        # Create blob with 5 blocks, cache only 2
        with TensorBlob.open(
            blob_path,
            "w",
            dtype="float32",
            shape=(5,),
            block_size=5,
            max_cached_blocks=2,
        ) as blob:
            # Write 25 tensors (5 blocks)
            tensors = torch.arange(25 * 5).reshape(25, 5).float()
            blob.write(tensors)

        # Reopen and access blocks in specific order
        with TensorBlob.open(blob_path, "r+", max_cached_blocks=2) as blob:
            # Access block 0 (tensor 0)
            _ = blob[0]
            assert len(blob._memmap) == 1

            # Access block 1 (tensor 5)
            _ = blob[5]
            assert len(blob._memmap) == 2  # Cache full

            # Access block 2 (tensor 10) - should evict block 0 (LRU)
            _ = blob[10]
            assert len(blob._memmap) == 2  # Still at limit

            # Verify block 0 was evicted by checking cache contents
            block_names = list(blob._memmap.keys())
            # Block 0 should be evicted, blocks 1 and 2 should remain
            assert blob._status.bds[0] not in block_names
            assert blob._status.bds[1] in block_names
            assert blob._status.bds[2] in block_names

    def test_lru_eviction_order(self, temp_dir):
        """Test that blocks are evicted in LRU order."""
        blob_path = temp_dir / "test_evict_order"

        # Create blob with 4 blocks, cache only 3
        with TensorBlob.open(
            blob_path,
            "w",
            dtype="float32",
            shape=(5,),
            block_size=5,
            max_cached_blocks=3,
        ) as blob:
            tensors = torch.arange(20 * 5).reshape(20, 5).float()
            blob.write(tensors)

        with TensorBlob.open(blob_path, "r", max_cached_blocks=3) as blob:
            # Access blocks 0, 1, 2 in order
            _ = blob[0]  # Load block 0
            _ = blob[5]  # Load block 1
            _ = blob[10]  # Load block 2

            # Cache should have 3 blocks
            assert len(blob._memmap) == 3
            cached = set(blob._memmap.keys())
            assert blob._status.bds[0] in cached
            assert blob._status.bds[1] in cached
            assert blob._status.bds[2] in cached

            # Access block 3 - should evict block 0 (least recently used)
            _ = blob[15]
            assert len(blob._memmap) == 3
            cached = set(blob._memmap.keys())
            assert blob._status.bds[0] not in cached  # Evicted
            assert blob._status.bds[1] in cached
            assert blob._status.bds[2] in cached
            assert blob._status.bds[3] in cached

            # Access block 1 again (move to end)
            _ = blob[5]

            # Access block 0 again - should evict block 2 (now LRU)
            _ = blob[0]
            assert len(blob._memmap) == 3
            cached = set(blob._memmap.keys())
            assert blob._status.bds[0] in cached
            assert blob._status.bds[1] in cached
            assert blob._status.bds[2] not in cached  # Evicted
            assert blob._status.bds[3] in cached

    def test_cache_clears_on_truncate(self, temp_dir):
        """Test that truncate properly handles cached blocks."""
        blob_path = temp_dir / "test_truncate"

        with TensorBlob.open(
            blob_path,
            "w",
            dtype="float32",
            shape=(5,),
            block_size=5,
            max_cached_blocks=5,
        ) as blob:
            # Write 25 tensors (5 blocks)
            tensors = torch.randn(25, 5)
            blob.write(tensors)

            # Ensure some blocks are cached
            _ = blob[0]
            _ = blob[5]
            _ = blob[10]
            cached_count = len(blob._memmap)
            assert cached_count > 0

            # Truncate to 10 tensors (2 blocks)
            blob.truncate(10)

            # Should have only 2 blocks now
            assert len(blob._status.bds) == 2

            # Blocks that were removed should not be in cache
            # (but cache might still have blocks 0 and 1)
            assert len(blob._memmap) <= 2

    def test_custom_cache_size(self, temp_dir):
        """Test that custom cache size is respected."""
        blob_path = temp_dir / "test_custom"

        custom_size = 10
        with TensorBlob.open(
            blob_path,
            "w",
            dtype="float32",
            shape=(5,),
            block_size=5,
            max_cached_blocks=custom_size,
        ) as blob:
            assert blob.max_cached_blocks == custom_size

            # Write many tensors (20 blocks worth)
            tensors = torch.randn(100, 5)
            blob.write(tensors)

            # Cache should never exceed custom size
            assert len(blob._memmap) <= custom_size

            # Should have 20 blocks total
            assert len(blob._status.bds) == 20

    def test_cache_persists_config_not_cache_size(self, temp_dir):
        """Test that cache size is runtime config, not persisted."""
        blob_path = temp_dir / "test_persist"

        # Create with custom cache size
        with TensorBlob.open(
            blob_path,
            "w",
            dtype="float32",
            shape=(5,),
            block_size=5,
            max_cached_blocks=5,
        ) as blob:
            tensors = torch.randn(10, 5)
            blob.write(tensors)

        # Reopen with different cache size (should work)
        with TensorBlob.open(blob_path, "r", max_cached_blocks=2) as blob:
            assert blob.max_cached_blocks == 2
            assert len(blob) == 10

    def test_lazy_loading(self, temp_dir):
        """Test that blocks are loaded lazily, not all at once."""
        blob_path = temp_dir / "test_lazy"

        # Create blob with many blocks
        with TensorBlob.open(
            blob_path,
            "w",
            dtype="float32",
            shape=(10,),
            block_size=10,
            max_cached_blocks=100,
        ) as blob:
            tensors = torch.randn(100, 10)
            blob.write(tensors)

            # Should have 10 blocks total
            assert len(blob._status.bds) == 10

        # Reopen - blocks should NOT all be loaded
        with TensorBlob.open(blob_path, "r", max_cached_blocks=100) as blob:
            # After opening, cache should be empty (lazy loading)
            assert len(blob._memmap) == 0

            # Access one tensor
            _ = blob[0]

            # Now only one block should be loaded
            assert len(blob._memmap) == 1

            # Access tensor from different block
            _ = blob[50]

            # Now two blocks loaded
            assert len(blob._memmap) == 2
