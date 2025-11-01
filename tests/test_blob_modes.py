"""Comprehensive tests for TensorBlob file modes and operations."""

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
    # Cleanup
    if blob_dir.exists():
        shutil.rmtree(blob_dir)


@pytest.fixture
def sample_tensors():
    """Fixture providing sample tensors for testing."""
    return torch.randn(100, 10)


class TestBlobCreation:
    """Tests for creating new TensorBlobs with different modes."""
    
    def test_create_with_w_mode(self, temp_blob_dir):
        """Test creating a new blob with 'w' mode."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            assert blob.writable
            assert not blob.readable
            assert blob.appendable
            assert len(blob) == 0
    
    def test_create_with_w_plus_mode(self, temp_blob_dir):
        """Test creating a new blob with 'w+' mode."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            assert blob.writable
            assert blob.readable
            assert blob.appendable
            assert len(blob) == 0
    
    def test_create_with_x_mode(self, temp_blob_dir):
        """Test exclusive creation with 'x' mode."""
        with TensorBlob.open(temp_blob_dir, "x", dtype="float32", shape=(10,)) as blob:
            assert blob.writable
            assert not blob.readable
        
        # Should fail if blob already exists
        with pytest.raises(FileExistsError):
            TensorBlob.open(temp_blob_dir, "x", dtype="float32", shape=(10,))
    
    def test_create_with_x_plus_mode(self, temp_blob_dir):
        """Test exclusive creation with 'x+' mode."""
        with TensorBlob.open(temp_blob_dir, "x+", dtype="float32", shape=(10,)) as blob:
            assert blob.writable
            assert blob.readable
    
    def test_create_without_dtype_shape_fails(self, temp_blob_dir):
        """Test that creating without dtype/shape fails."""
        with pytest.raises(ValueError, match="dtype"):
            TensorBlob.open(temp_blob_dir, "w", shape=(10,))
        
        with pytest.raises(ValueError, match="shape"):
            TensorBlob.open(temp_blob_dir, "w", dtype="float32")
    
    def test_invalid_mode_fails(self, temp_blob_dir):
        """Test that invalid modes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            TensorBlob.open(temp_blob_dir, "z", dtype="float32", shape=(10,))
        
        with pytest.raises(ValueError, match="Mode must be one of"):
            TensorBlob.open(temp_blob_dir, "rw", dtype="float32", shape=(10,))


class TestBlobOpening:
    """Tests for opening existing TensorBlobs with different modes."""
    
    def test_open_nonexistent_with_r_fails(self, temp_blob_dir):
        """Test that opening nonexistent blob in 'r' mode fails."""
        with pytest.raises(FileNotFoundError, match="mode='r' requires existing blob"):
            TensorBlob.open(temp_blob_dir, "r")
    
    def test_open_nonexistent_with_a_fails(self, temp_blob_dir):
        """Test that opening nonexistent blob in 'a' mode fails."""
        with pytest.raises(FileNotFoundError, match="mode='a' requires existing blob"):
            TensorBlob.open(temp_blob_dir, "a")
    
    def test_open_existing_with_r_mode(self, temp_blob_dir, sample_tensors):
        """Test opening existing blob in 'r' mode."""
        # Create a blob
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        # Open in read mode
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert blob.readable
            assert not blob.writable
            assert not blob.appendable
            assert len(blob) == 100
    
    def test_open_existing_with_r_plus_mode(self, temp_blob_dir, sample_tensors):
        """Test opening existing blob in 'r+' mode."""
        # Create a blob
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        # Open in read+write mode
        with TensorBlob.open(temp_blob_dir, "r+") as blob:
            assert blob.readable
            assert blob.writable
            assert len(blob) == 100
    
    def test_open_existing_with_a_mode(self, temp_blob_dir, sample_tensors):
        """Test opening existing blob in 'a' mode."""
        # Create a blob
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        # Open in append mode
        with TensorBlob.open(temp_blob_dir, "a") as blob:
            assert not blob.readable
            assert not blob.writable
            assert blob.appendable
            assert len(blob) == 100
            assert blob.tell() == 100  # Position at end
    
    def test_open_existing_with_a_plus_mode(self, temp_blob_dir, sample_tensors):
        """Test opening existing blob in 'a+' mode."""
        # Create a blob
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        # Open in append+read mode
        with TensorBlob.open(temp_blob_dir, "a+") as blob:
            assert blob.readable
            assert blob.appendable
            assert blob.tell() == 100  # Position at end
    
    def test_w_mode_truncates_existing(self, temp_blob_dir, sample_tensors):
        """Test that 'w' mode truncates existing blob."""
        # Create a blob with data
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        # Reopen with 'w' should truncate
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            assert len(blob) == 0


class TestReadOperations:
    """Tests for reading from TensorBlobs."""
    
    def test_read_all(self, temp_blob_dir, sample_tensors):
        """Test reading all tensors."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            blob.seek(0)
            data = blob.read()
            assert data is not None
            assert data.shape == sample_tensors.shape
            assert torch.allclose(data, sample_tensors)
    
    def test_read_n_tensors(self, temp_blob_dir, sample_tensors):
        """Test reading specific number of tensors."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            blob.seek(0)
            data = blob.read(10)
            assert data is not None
            assert data.shape == (10, 10)
            assert torch.allclose(data, sample_tensors[:10])
    
    def test_read_advances_position(self, temp_blob_dir, sample_tensors):
        """Test that read advances position correctly."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            blob.seek(0)
            assert blob.tell() == 0
            blob.read(10)
            assert blob.tell() == 10
    
    def test_read_at_end_returns_none(self, temp_blob_dir, sample_tensors):
        """Test reading at end returns None."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            # Position is already at end after write
            data = blob.read()
            assert data is None
    
    def test_read_in_write_only_mode_fails(self, temp_blob_dir):
        """Test that reading in 'w' mode fails."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            with pytest.raises(IOError, match="not open for reading"):
                blob.read()
    
    def test_read_from_closed_blob_fails(self, temp_blob_dir, sample_tensors):
        """Test reading from closed blob fails."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        # Blob is now closed
        with pytest.raises(ValueError, match="closed"):
            blob.read()


class TestWriteOperations:
    """Tests for writing to TensorBlobs."""
    
    def test_write_single_tensor(self, temp_blob_dir):
        """Test writing a single tensor."""
        tensor = torch.randn(10)
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            n = blob.write(tensor)
            assert n == 1
            assert len(blob) == 1
    
    def test_write_multiple_tensors(self, temp_blob_dir, sample_tensors):
        """Test writing multiple tensors."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            n = blob.write(sample_tensors)
            assert n == 100
            assert len(blob) == 100
    
    def test_write_advances_position(self, temp_blob_dir, sample_tensors):
        """Test that write advances position."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            assert blob.tell() == 0
            blob.write(sample_tensors[:10])
            assert blob.tell() == 10
    
    def test_write_in_read_only_mode_fails(self, temp_blob_dir, sample_tensors):
        """Test that writing in 'r' mode fails."""
        # First create a blob
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        # Open in read mode and try to write
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            with pytest.raises(IOError, match="not open for writing"):
                blob.write(torch.randn(10))
    
    def test_write_in_append_mode_fails(self, temp_blob_dir, sample_tensors):
        """Test that write (not append) in 'a' mode fails."""
        # First create a blob
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        # Open in append mode and try to write
        with TensorBlob.open(temp_blob_dir, "a") as blob:
            with pytest.raises(IOError, match="not open for writing"):
                blob.write(torch.randn(10))
    
    def test_write_overwrites_at_position(self, temp_blob_dir, sample_tensors):
        """Test that write can overwrite existing data."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            blob.seek(0)
            new_data = torch.ones(10, 10)
            blob.write(new_data)
            blob.seek(0)
            data = blob.read(10)
            assert data is not None
            assert torch.allclose(data, new_data)
    
    def test_write_extends_blob(self, temp_blob_dir):
        """Test that write can extend blob beyond current length."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(torch.randn(50, 10))
            assert len(blob) == 50
            blob.write(torch.randn(50, 10))
            assert len(blob) == 100


class TestAppendOperations:
    """Tests for appending to TensorBlobs."""
    
    def test_append_single_tensor(self, temp_blob_dir, sample_tensors):
        """Test appending a single tensor."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        with TensorBlob.open(temp_blob_dir, "a") as blob:
            new_tensor = torch.ones(10)
            n = blob.append(new_tensor)
            assert n == 1
            assert len(blob) == 101
    
    def test_append_multiple_tensors(self, temp_blob_dir, sample_tensors):
        """Test appending multiple tensors."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        with TensorBlob.open(temp_blob_dir, "a") as blob:
            new_tensors = torch.ones(20, 10)
            n = blob.append(new_tensors)
            assert n == 20
            assert len(blob) == 120
    
    def test_append_moves_position_to_end(self, temp_blob_dir, sample_tensors):
        """Test that append always works at end."""
        # First create a blob
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            pass  # Just create empty blob
        
        # Now open in append mode and test
        with TensorBlob.open(temp_blob_dir, "a+") as blob:
            blob.append(sample_tensors[:50])
            blob.seek(0)  # Move to beginning
            blob.append(sample_tensors[50:])  # Should still append at end
            assert len(blob) == 100
            assert blob.tell() == 100


class TestSeekTellOperations:
    """Tests for seek and tell operations."""
    
    def test_tell_returns_position(self, temp_blob_dir, sample_tensors):
        """Test that tell returns current position."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            assert blob.tell() == 100
            blob.seek(50)
            assert blob.tell() == 50
    
    def test_seek_absolute(self, temp_blob_dir, sample_tensors):
        """Test absolute seeking (whence=0)."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            pos = blob.seek(25, 0)
            assert pos == 25
            assert blob.tell() == 25
    
    def test_seek_relative(self, temp_blob_dir, sample_tensors):
        """Test relative seeking (whence=1)."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            blob.seek(50)
            pos = blob.seek(10, 1)
            assert pos == 60
            assert blob.tell() == 60
    
    def test_seek_from_end(self, temp_blob_dir, sample_tensors):
        """Test seeking from end (whence=2)."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            pos = blob.seek(-10, 2)
            assert pos == 90
            assert blob.tell() == 90
    
    def test_seek_clamps_to_bounds(self, temp_blob_dir, sample_tensors):
        """Test that seek clamps to valid range."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            
            # Seek before start
            blob.seek(-100)
            assert blob.tell() == 0
            
            # Seek beyond end
            blob.seek(200)
            assert blob.tell() == 100
    
    def test_seek_with_invalid_whence_fails(self, temp_blob_dir, sample_tensors):
        """Test that invalid whence raises ValueError."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            with pytest.raises(ValueError, match="whence must be"):
                blob.seek(0, 3)
    
    def test_seek_on_closed_blob_fails(self, temp_blob_dir, sample_tensors):
        """Test seeking on closed blob fails."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        with pytest.raises(ValueError, match="closed"):
            blob.seek(0)


class TestIndexingAndIteration:
    """Tests for indexing and iteration."""
    
    def test_indexing(self, temp_blob_dir, sample_tensors):
        """Test accessing tensors by index."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            
            # Access first tensor
            t0 = blob[0]
            assert torch.allclose(t0, sample_tensors[0])
            
            # Access middle tensor
            t50 = blob[50]
            assert torch.allclose(t50, sample_tensors[50])
            
            # Access last tensor
            t99 = blob[99]
            assert torch.allclose(t99, sample_tensors[99])
    
    def test_indexing_out_of_bounds_fails(self, temp_blob_dir, sample_tensors):
        """Test that out-of-bounds indexing fails."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            
            with pytest.raises(IndexError, match="out of bound"):
                _ = blob[100]
    
    def test_iteration(self, temp_blob_dir, sample_tensors):
        """Test iterating over blob."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            blob.seek(0)
            
            collected = []
            for i, tensor in enumerate(blob):
                collected.append(tensor)
                if i >= 9:  # Just test first 10
                    break
            
            assert len(collected) == 10
            for i, tensor in enumerate(collected):
                assert torch.allclose(tensor, sample_tensors[i])


class TestContextManager:
    """Tests for context manager functionality."""
    
    def test_context_manager_closes_blob(self, temp_blob_dir):
        """Test that context manager closes blob."""
        blob = TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,))
        assert not blob.closed
        
        with blob:
            assert not blob.closed
        
        assert blob.closed
    
    def test_context_manager_flushes_on_exit(self, temp_blob_dir, sample_tensors):
        """Test that context manager flushes on exit."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
            # Don't manually flush
        
        # Should be able to read data after reopening
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert len(blob) == 100


class TestFlushOperations:
    """Tests for flush operations."""
    
    def test_flush_persists_data(self, temp_blob_dir, sample_tensors):
        """Test that flush persists data to disk."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors[:50])
            blob.flush()
            
            # Open another instance and check
            with TensorBlob.open(temp_blob_dir, "r") as blob2:
                assert len(blob2) == 50
    
    def test_flush_in_read_mode_fails(self, temp_blob_dir, sample_tensors):
        """Test that flush in read-only mode fails."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            with pytest.raises(IOError, match="Cannot flush"):
                blob.flush()
    
    def test_flush_on_closed_blob_fails(self, temp_blob_dir, sample_tensors):
        """Test flushing closed blob fails."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(sample_tensors)
        
        with pytest.raises(ValueError, match="closed"):
            blob.flush()


class TestBlockManagement:
    """Tests for internal block management."""
    
    def test_multiple_blocks_created(self, temp_blob_dir):
        """Test that multiple blocks are created when needed."""
        block_size = 100
        num_tensors = 250  # Should create 3 blocks
        
        with TensorBlob.open(
            temp_blob_dir, "w", dtype="float32", shape=(10,), block_size=block_size
        ) as blob:
            data = torch.randn(num_tensors, 10)
            blob.write(data)
            
            # Check that 3 blocks were created
            assert len(blob._states.bds) == 3
            assert len(blob) == num_tensors
    
    def test_reading_across_blocks(self, temp_blob_dir):
        """Test reading data that spans multiple blocks."""
        block_size = 50
        num_tensors = 150
        
        with TensorBlob.open(
            temp_blob_dir, "w+", dtype="float32", shape=(10,), block_size=block_size
        ) as blob:
            data = torch.randn(num_tensors, 10)
            blob.write(data)
            blob.seek(0)
            
            # Read data spanning multiple blocks
            read_data = blob.read(100)
            assert read_data is not None
            assert read_data.shape == (100, 10)
            assert torch.allclose(read_data, data[:100])


class TestDtypeAndShape:
    """Tests for different dtypes and shapes."""
    
    def test_different_dtypes(self, temp_blob_dir):
        """Test using different tensor dtypes."""
        for dtype in ["float32", "float64", "int32", "int64"]:
            subdir = temp_blob_dir / dtype
            with TensorBlob.open(subdir, "w", dtype=dtype, shape=(5,)) as blob:
                data = torch.randn(10, 5).to(getattr(torch, dtype))
                blob.write(data)
                assert len(blob) == 10
    
    def test_different_shapes(self, temp_blob_dir):
        """Test using different tensor shapes."""
        shapes = [(10,), (5, 5), (2, 3, 4)]
        for i, shape in enumerate(shapes):
            subdir = temp_blob_dir / f"shape_{i}"
            with TensorBlob.open(subdir, "w+", dtype="float32", shape=shape) as blob:
                # Create data matching the shape
                data = torch.randn(20, *shape)
                blob.write(data)
                blob.seek(0)
                read_data = blob.read(20)
                assert read_data is not None
                assert read_data.shape == (20, *shape)
    
    def test_torch_dtype_input(self, temp_blob_dir):
        """Test passing torch.dtype instead of string."""
        with TensorBlob.open(
            temp_blob_dir, "w", dtype=torch.float32, shape=(10,)
        ) as blob:
            data = torch.randn(10, 10)
            blob.write(data)
            assert len(blob) == 10

