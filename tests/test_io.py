"""Tests for basic I/O operations on TensorBlob."""

import pytest
import torch
from tensorblob import TensorBlob


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
    
    def test_write_updates_position(self, temp_blob_dir):
        """Test that write updates position correctly."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            assert blob.tell() == 0
            blob.write(torch.randn(3, 5))
            assert blob.tell() == 3
            blob.write(torch.randn(7, 5))
            assert blob.tell() == 10
    
    def test_write_with_reshape(self, temp_blob_dir):
        """Test that tensors are correctly reshaped during write."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(2, 3)) as blob:
            data = torch.arange(12, dtype=torch.float32)
            blob.write(data)
            assert len(blob) == 2  # 12 elements = 2 tensors of shape (2,3)


class TestBasicRead:
    """Tests for basic read operations."""
    
    def test_read_all(self, blob_with_data):
        """Test reading all tensors."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob.read()
            assert result.shape == (100, 10)
            assert torch.allclose(result, sample_data)
    
    def test_read_partial(self, blob_with_data):
        """Test reading specific number of tensors."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob.read(size=10)
            assert result.shape == (10, 10)
            assert torch.allclose(result, sample_data[:10])
    
    def test_read_updates_position(self, blob_with_data):
        """Test that read updates position correctly."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            assert blob.tell() == 0
            blob.read(size=10)
            assert blob.tell() == 10
            blob.read(size=20)
            assert blob.tell() == 30
    
    def test_read_from_position(self, blob_with_data):
        """Test reading from non-zero position."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
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
            assert result.size(0) == 0


class TestIndexing:
    """Tests for integer indexing operations."""
    
    def test_positive_index(self, blob_with_data):
        """Test accessing tensors by positive index."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            assert torch.allclose(blob[0], sample_data[0])
            assert torch.allclose(blob[50], sample_data[50])
            assert torch.allclose(blob[99], sample_data[99])

    def test_negative_index(self, blob_with_data):
        """Test accessing tensors by negative index."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            # Negative indices work: -1 is last element, -100 is first element
            assert torch.allclose(blob[-1], sample_data[-1])
            assert torch.allclose(blob[-50], sample_data[-50])
            assert torch.allclose(blob[-100], sample_data[0])
    
    def test_index_out_of_bounds(self, blob_with_data):
        """Test that out-of-bounds indexing raises IndexError."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            with pytest.raises(IndexError, match="out of bounds"):
                _ = blob[100]
            with pytest.raises(IndexError, match="out of bounds"):
                _ = blob[-101]
    
    def test_indexing_returns_clone(self, temp_blob_dir):
        """Test that indexing returns a copy, not a reference."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.ones(5))
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            retrieved = blob[0]
            retrieved[0] = 999.0
            retrieved2 = blob[0]
            assert retrieved2[0] == 1.0


class TestIteration:
    """Tests for iteration over blob."""
    
    def test_iterate_all(self, blob_with_data):
        """Test iterating over entire blob."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            collected = list(blob)
            assert len(collected) == 100
            for i, tensor in enumerate(collected):
                assert torch.allclose(tensor, sample_data[i])
    
    def test_iterate_from_position(self, blob_with_data):
        """Test that iteration starts from current position."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            blob.seek(50)
            collected = list(blob)
            assert len(collected) == 50
            for i, tensor in enumerate(collected):
                assert torch.allclose(tensor, sample_data[50 + i])


class TestSeekAndTell:
    """Tests for seek and tell operations."""
    
    def test_seek_absolute(self, blob_with_data):
        """Test absolute seeking."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            blob.seek(25)
            assert blob.tell() == 25
    
    def test_seek_relative(self, blob_with_data):
        """Test relative seeking."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            blob.seek(20)
            blob.seek(10, whence=1)
            assert blob.tell() == 30
    
    def test_seek_from_end(self, blob_with_data):
        """Test seeking from end."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            blob.seek(-10, whence=2)
            assert blob.tell() == 90
    
    def test_seek_clamping(self, blob_with_data):
        """Test that seek clamps to valid range."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            blob.seek(200)
            assert blob.tell() == 100
            
            blob.seek(50)
            blob.seek(-100, whence=1)
            assert blob.tell() == 0


class TestMixedModes:
    """Tests for combined read/write operations."""
    
    def test_read_plus_mode(self, blob_with_data):
        """Test reading and writing in r+ mode."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r+") as blob:
            first = blob[0]
            assert first.shape == (10,)
            
            blob.seek(5)
            blob.write(torch.ones(3, 10) * 2)
            assert len(blob) == 100
            
            assert torch.allclose(blob[5], torch.ones(10) * 2)
    
    def test_write_plus_mode(self, temp_blob_dir):
        """Test writing and reading in w+ mode."""
        data = torch.randn(20, 5)
        
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(5,)) as blob:
            blob.write(data)
            assert len(blob) == 20
            
            blob.seek(0)
            assert torch.allclose(blob[0], data[0])
    
    def test_append_mode_behavior(self, blob_with_data):
        """Test that append mode writes always go to end."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "a") as blob:
            assert blob.tell() == 100
            blob.seek(0)  # Try to seek
            blob.write(torch.ones(5, 10) * 2)
            assert len(blob) == 105
        
        # Verify data appended at end
        with TensorBlob.open(blob_dir, "r") as blob:
            assert torch.allclose(blob[100], torch.ones(10) * 2)


class TestFlushAndPersistence:
    """Tests for flush and data persistence."""
    
    def test_flush_persists_data(self, temp_blob_dir):
        """Test that flush persists data to disk."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.ones(10, 5))
            blob.flush()
            
            with TensorBlob.open(temp_blob_dir, "r") as blob2:
                assert len(blob2) == 10
    
    def test_auto_flush_on_close(self, temp_blob_dir, sample_data):
        """Test that data is automatically flushed on close."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert len(blob) == 100
            assert torch.allclose(blob[0], sample_data[0])


class TestMultiBlock:
    """Tests for multi-block operations."""
    
    def test_write_across_blocks(self, temp_blob_dir):
        """Test writing data that spans multiple blocks."""
        with TensorBlob.open(
            temp_blob_dir, "w", dtype="float32", shape=(5,), block_size=100
        ) as blob:
            blob.write(torch.randn(250, 5))
            assert len(blob) == 250
            assert len(blob._status.bds) == 3
    
    def test_read_across_blocks(self, multi_block_blob):
        """Test reading data that spans multiple blocks."""
        blob_dir, data, _ = multi_block_blob
        
        with TensorBlob.open(blob_dir, "r") as blob:
            assert torch.allclose(blob[0], data[0])
            assert torch.allclose(blob[75], data[75])
            assert torch.allclose(blob[149], data[149])
