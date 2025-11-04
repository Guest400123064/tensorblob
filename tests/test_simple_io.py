"""Tests for basic sequential I/O operations on TensorBlob."""

import pytest
import torch
from pathlib import Path
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
    """Fixture providing sample tensor data."""
    torch.manual_seed(42)
    return torch.randn(100, 10)


class TestBasicWrite:
    """Tests for basic write operations."""
    
    def test_write_single_tensor(self, temp_blob_dir):
        """Test writing a single tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            n = blob.write(tensor)
            assert n == 1
            assert len(blob) == 1
            assert blob.tell() == 1
    
    def test_write_multiple_tensors(self, temp_blob_dir, sample_data):
        """Test writing multiple tensors at once."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            n = blob.write(sample_data)
            assert n == 100
            assert len(blob) == 100
            assert blob.tell() == 100
    
    def test_write_incremental(self, temp_blob_dir):
        """Test writing tensors incrementally."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(3,)) as blob:
            # Write first batch
            blob.write(torch.ones(5, 3))
            assert len(blob) == 5
            assert blob.tell() == 5
            
            # Write second batch
            blob.write(torch.ones(5, 3) * 2)
            assert len(blob) == 10
            assert blob.tell() == 10
    
    def test_write_updates_position(self, temp_blob_dir):
        """Test that write updates position correctly."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            assert blob.tell() == 0
            blob.write(torch.randn(3, 5))
            assert blob.tell() == 3
            blob.write(torch.randn(7, 5))
            assert blob.tell() == 10
    
    def test_write_with_different_shapes(self, temp_blob_dir):
        """Test that tensors are correctly reshaped during write."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(2, 3)) as blob:
            # Write 1D tensor (will be reshaped)
            data = torch.arange(12, dtype=torch.float32)
            blob.write(data)
            assert len(blob) == 2  # 12 elements = 2 tensors of shape (2,3)


class TestBasicRead:
    """Tests for basic read operations."""
    
    def test_read_all(self, temp_blob_dir, sample_data):
        """Test reading all tensors."""
        # Write data
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        # Read back
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            result = blob.read()
            assert result.shape == (100, 10)
            assert torch.allclose(result, sample_data)
    
    def test_read_partial(self, temp_blob_dir, sample_data):
        """Test reading specific number of tensors."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            result = blob.read(size=10)
            assert result.shape == (10, 10)
            assert torch.allclose(result, sample_data[:10])
    
    def test_read_updates_position(self, temp_blob_dir, sample_data):
        """Test that read updates position correctly."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert blob.tell() == 0
            blob.read(size=10)
            assert blob.tell() == 10
            blob.read(size=20)
            assert blob.tell() == 30
    
    def test_read_from_position(self, temp_blob_dir, sample_data):
        """Test reading from non-zero position."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            blob.seek(50)
            result = blob.read(size=10)
            assert result.shape == (10, 10)
            assert torch.allclose(result, sample_data[50:60])
    
    def test_read_empty_returns_none(self, temp_blob_dir):
        """Test reading from empty blob returns None."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            pass  # Empty blob
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            result = blob.read()
            assert result is None
    
    def test_read_at_end_returns_none(self, temp_blob_dir, sample_data):
        """Test reading at end of blob returns None."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            blob.seek(100)  # Seek to end
            result = blob.read()
            assert result is None


class TestIndexing:
    """Tests for indexing operations."""
    
    def test_index_single_tensor(self, temp_blob_dir, sample_data):
        """Test accessing single tensor by index."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            # Test various indices
            assert torch.allclose(blob[0], sample_data[0])
            assert torch.allclose(blob[50], sample_data[50])
            assert torch.allclose(blob[99], sample_data[99])
    
    def test_index_out_of_bounds(self, temp_blob_dir, sample_data):
        """Test that out-of-bounds indexing raises IndexError."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            with pytest.raises(IndexError, match="out of bound"):
                _ = blob[100]
    
    def test_indexing_returns_clone(self, temp_blob_dir):
        """Test that indexing returns a copy, not a reference."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            original = torch.ones(5)
            blob.write(original)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            retrieved = blob[0]
            retrieved[0] = 999.0  # Modify
            # Read again - should not be affected
            retrieved2 = blob[0]
            assert retrieved2[0] == 1.0


class TestIteration:
    """Tests for iteration over blob."""
    
    def test_iterate_all(self, temp_blob_dir, sample_data):
        """Test iterating over entire blob."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            collected = []
            for tensor in blob:
                collected.append(tensor)
            
            assert len(collected) == 100
            for i, tensor in enumerate(collected):
                assert torch.allclose(tensor, sample_data[i])
    
    def test_iterate_from_position(self, temp_blob_dir, sample_data):
        """Test that iteration starts from current position."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            blob.seek(50)
            collected = list(blob)
            assert len(collected) == 50  # Only reads from position 50 onwards
            for i, tensor in enumerate(collected):
                assert torch.allclose(tensor, sample_data[50 + i])
    
    def test_iterate_updates_position(self, temp_blob_dir, sample_data):
        """Test that iteration updates position."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data[:10])
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            count = 0
            for tensor in blob:
                count += 1
                if count >= 5:  # Iterate exactly 5 times
                    break
            # After yielding 5 items (positions 0-4), position should be at 5
            assert blob.tell() == 5


class TestSeekAndTell:
    """Tests for seek and tell operations."""
    
    def test_seek_absolute(self, temp_blob_dir, sample_data):
        """Test absolute seeking."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            blob.seek(25)
            assert blob.tell() == 25
            assert torch.allclose(blob[25], sample_data[25])
    
    def test_seek_relative(self, temp_blob_dir, sample_data):
        """Test relative seeking."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            blob.seek(20)
            blob.seek(10, whence=1)  # Relative: +10 from current
            assert blob.tell() == 30
    
    def test_seek_from_end(self, temp_blob_dir, sample_data):
        """Test seeking from end with negative offset."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            # Seek 10 positions back from end
            blob.seek(-10, whence=2)  # whence=2 is SEEK_END
            assert blob.tell() == 90
    
    def test_seek_clamping(self, temp_blob_dir, sample_data):
        """Test that seek clamps to valid range."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            # Seek beyond end - should clamp to length
            blob.seek(200)
            assert blob.tell() == 100
            
            # Seek with relative negative that goes before start - should clamp to 0
            blob.seek(50)
            blob.seek(-100, whence=1)  # From position 50, go back 100 -> clamp to 0
            assert blob.tell() == 0


class TestReadWrite:
    """Tests for combined read/write operations."""
    
    def test_read_plus_write_mode(self, temp_blob_dir):
        """Test reading and writing in r+ mode."""
        # Create initial data
        initial_data = torch.ones(10, 5)
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(initial_data)
        
        # Open in r+ mode and modify
        with TensorBlob.open(temp_blob_dir, "r+") as blob:
            # Read some data
            blob.seek(0)
            first = blob[0]
            assert torch.allclose(first, torch.ones(5))
            
            # Overwrite at position
            blob.seek(5)
            blob.write(torch.ones(3, 5) * 2)
            assert len(blob) == 10  # Length unchanged
            
            # Verify overwrite
            blob.seek(5)
            assert torch.allclose(blob[5], torch.ones(5) * 2)
    
    def test_write_plus_read_mode(self, temp_blob_dir):
        """Test writing and reading in w+ mode."""
        data = torch.randn(20, 5)
        
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(5,)) as blob:
            # Write data
            blob.write(data)
            assert len(blob) == 20
            
            # Seek back and read
            blob.seek(0)
            for i in range(10):
                assert torch.allclose(blob[i], data[i])
    
    def test_overwrite_existing_data(self, temp_blob_dir):
        """Test overwriting data at a specific position."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(3,)) as blob:
            # Write initial data
            blob.write(torch.ones(10, 3))
            
            # Overwrite middle section
            blob.seek(4)
            new_data = torch.ones(3, 3) * 99
            blob.write(new_data)
            
            # Verify
            blob.seek(4)
            for i in range(3):
                assert torch.allclose(blob[4 + i], torch.ones(3) * 99)
            
            # Check length hasn't changed
            assert len(blob) == 10


class TestAppendMode:
    """Tests for append mode operations."""
    
    def test_append_mode_writes_at_end(self, temp_blob_dir):
        """Test that append mode always writes at end."""
        # Create initial blob
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.ones(10, 5))
        
        # Append more data
        with TensorBlob.open(temp_blob_dir, "a") as blob:
            assert blob.tell() == 10  # Position starts at end
            blob.write(torch.ones(5, 5) * 2)
            assert len(blob) == 15
    
    def test_append_ignores_seek(self, temp_blob_dir):
        """Test that writes in append mode always go to end, regardless of seek."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.ones(10, 5))
        
        with TensorBlob.open(temp_blob_dir, "a") as blob:
            # Try to seek to beginning
            blob.seek(0)
            assert blob.tell() == 0
            
            # Write should still go to end
            blob.write(torch.ones(3, 5) * 2)
            assert len(blob) == 13

        # Verify new data is at end, not at position 0
        with TensorBlob.open(temp_blob_dir, "r") as read_blob:
            assert torch.allclose(read_blob[0], torch.ones(5))  # Original
            assert torch.allclose(read_blob[10], torch.ones(5) * 2)  # Appended
    
    def test_append_plus_can_read(self, temp_blob_dir):
        """Test that a+ mode can read existing data."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.ones(10, 5))
        
        with TensorBlob.open(temp_blob_dir, "a+") as blob:
            # Can seek and read
            blob.seek(0)
            assert torch.allclose(blob[0], torch.ones(5))
            
            # Writes still go to end
            blob.write(torch.ones(5, 5) * 2)
            assert len(blob) == 15


class TestFlush:
    """Tests for flush operations."""
    
    def test_flush_persists_data(self, temp_blob_dir):
        """Test that flush persists data to disk."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.ones(10, 5))
            blob.flush()
            
            # Open second instance and verify data is there
            with TensorBlob.open(temp_blob_dir, "r") as blob2:
                assert len(blob2) == 10
    
    def test_auto_flush_on_close(self, temp_blob_dir):
        """Test that data is automatically flushed on close."""
        data = torch.randn(20, 5)
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(data)
            # Don't manually flush
        
        # Reopen and verify
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert len(blob) == 20
            assert torch.allclose(blob[0], data[0])


class TestMultipleBlocks:
    """Tests for operations spanning multiple internal blocks."""
    
    def test_write_across_blocks(self, temp_blob_dir):
        """Test writing data that spans multiple blocks."""
        block_size = 100
        num_tensors = 250
        
        with TensorBlob.open(
            temp_blob_dir, "w", dtype="float32", shape=(5,), block_size=block_size
        ) as blob:
            data = torch.randn(num_tensors, 5)
            blob.write(data)
            assert len(blob) == num_tensors
            assert len(blob._states.bds) == 3  # Should have 3 blocks
    
    def test_read_across_blocks(self, temp_blob_dir):
        """Test reading data that spans multiple blocks."""
        block_size = 50
        data = torch.randn(150, 5)
        
        with TensorBlob.open(
            temp_blob_dir, "w", dtype="float32", shape=(5,), block_size=block_size
        ) as blob:
            blob.write(data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            # Read from first block
            assert torch.allclose(blob[0], data[0])
            # Read from second block
            assert torch.allclose(blob[75], data[75])
            # Read from third block
            assert torch.allclose(blob[149], data[149])
    
    def test_seek_across_blocks(self, temp_blob_dir):
        """Test seeking across block boundaries."""
        block_size = 50
        data = torch.randn(150, 5)
        
        with TensorBlob.open(
            temp_blob_dir, "w+", dtype="float32", shape=(5,), block_size=block_size
        ) as blob:
            blob.write(data)
            
            # Seek to different blocks
            blob.seek(25)  # First block
            assert blob.tell() == 25
            
            blob.seek(75)  # Second block
            assert blob.tell() == 75
            
            blob.seek(125)  # Third block
            assert blob.tell() == 125


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_blob_operations(self, temp_blob_dir):
        """Test operations on empty blob."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(5,)) as blob:
            assert len(blob) == 0
            assert blob.tell() == 0
            
            # Read from empty blob should return None
            result = blob.read()
            assert result is None
    
    def test_single_element_blob(self, temp_blob_dir):
        """Test blob with single element."""
        data = torch.tensor([1.0, 2.0, 3.0])
        
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(3,)) as blob:
            blob.write(data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert len(blob) == 1
            assert torch.allclose(blob[0], data)
    
    def test_write_to_end_extends_blob(self, temp_blob_dir):
        """Test that writing at the end extends the blob."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.ones(10, 5))
            assert len(blob) == 10
            
            # Seek to end and write more
            blob.seek(10)
            blob.write(torch.ones(5, 5) * 2)
            assert len(blob) == 15

