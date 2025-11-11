"""Test suite for TensorBlob.unlink() class method."""

import os
import warnings
from pathlib import Path

import pytest
import torch

from tensorblob import TensorBlob


def test_unlink_existing_blob(temp_blob_dir):
    """Test unlinking an existing blob removes all files."""
    blob_path = temp_blob_dir / "test_blob"
    
    # Create and populate a blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(10,)) as blob:
        blob.write(torch.randn(100, 10))
    
    # Verify blob exists
    assert blob_path.exists()
    assert (blob_path / ".conf").exists()
    assert (blob_path / ".stat").exists()
    assert len(list(blob_path.glob("*"))) >= 3  # .conf, .stat, and at least one block
    
    # Unlink the blob
    result = TensorBlob.unlink(blob_path)
    
    # Verify blob is completely removed
    assert result is True
    assert not blob_path.exists()


def test_unlink_nonexistent_blob(temp_blob_dir):
    """Test unlinking a non-existent blob returns True."""
    blob_path = temp_blob_dir / "nonexistent"
    
    assert not blob_path.exists()
    result = TensorBlob.unlink(blob_path)
    
    assert result is True
    assert not blob_path.exists()


def test_unlink_multi_block_blob(temp_blob_dir):
    """Test unlinking a blob with multiple blocks."""
    blob_path = temp_blob_dir / "multi_block"
    
    # Create blob with multiple blocks (block_size=10, 50 tensors = 5 blocks)
    with TensorBlob.open(
        blob_path, "w", dtype="float32", shape=(5,), block_size=10
    ) as blob:
        blob.write(torch.randn(50, 5))
    
    # Count block files
    block_files = [f for f in blob_path.iterdir() if f.name not in [".conf", ".stat"]]
    assert len(block_files) == 5
    
    # Unlink
    result = TensorBlob.unlink(blob_path)
    
    assert result is True
    assert not blob_path.exists()


def test_unlink_with_tilde_path(temp_blob_dir, monkeypatch):
    """Test unlinking with tilde (~) path expansion."""
    blob_path = temp_blob_dir / "tilde_test"
    
    # Create blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    # Mock home directory to temp_blob_dir's parent for testing
    monkeypatch.setenv("HOME", str(temp_blob_dir.parent))
    relative_path = f"~/{temp_blob_dir.name}/tilde_test"
    
    result = TensorBlob.unlink(relative_path)
    
    assert result is True
    assert not blob_path.exists()


def test_unlink_with_relative_path(temp_blob_dir, monkeypatch):
    """Test unlinking with relative path."""
    blob_path = temp_blob_dir / "relative_test"
    
    # Create blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    # Change to parent directory
    monkeypatch.chdir(temp_blob_dir)
    
    result = TensorBlob.unlink("relative_test")
    
    assert result is True
    assert not blob_path.exists()


def test_unlink_empty_blob(temp_blob_dir):
    """Test unlinking an empty blob (no data written)."""
    blob_path = temp_blob_dir / "empty"
    
    # Create empty blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(10,)):
        pass
    
    assert blob_path.exists()
    assert (blob_path / ".conf").exists()
    assert (blob_path / ".stat").exists()
    
    result = TensorBlob.unlink(blob_path)
    
    assert result is True
    assert not blob_path.exists()


def test_unlink_after_read(temp_blob_dir):
    """Test unlinking after reading from a blob."""
    blob_path = temp_blob_dir / "read_test"
    
    # Create and read
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(20, 5))
    
    with TensorBlob.open(blob_path, "r") as blob:
        data = blob.read()
        assert data.shape == (20, 5)
    
    result = TensorBlob.unlink(blob_path)
    
    assert result is True
    assert not blob_path.exists()


def test_unlink_multiple_times(temp_blob_dir):
    """Test unlinking the same path multiple times."""
    blob_path = temp_blob_dir / "multi_unlink"
    
    # Create blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    # First unlink
    result1 = TensorBlob.unlink(blob_path)
    assert result1 is True
    assert not blob_path.exists()
    
    # Second unlink (should still return True)
    result2 = TensorBlob.unlink(blob_path)
    assert result2 is True
    assert not blob_path.exists()


def test_unlink_with_corrupted_status(temp_blob_dir):
    """Test unlinking a blob with missing or corrupted status file."""
    blob_path = temp_blob_dir / "corrupted"
    
    # Create blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    # Corrupt by removing status file
    os.remove(blob_path / ".stat")
    
    # Unlink should handle gracefully
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = TensorBlob.unlink(blob_path)
        
        # Should warn and return False
        assert result is False
        assert len(w) == 1
        assert "Failed to unlink" in str(w[0].message)


def test_unlink_with_missing_config(temp_blob_dir):
    """Test unlinking a blob with missing config file."""
    blob_path = temp_blob_dir / "no_config"
    
    # Create blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    # Remove config file
    os.remove(blob_path / ".conf")
    
    # Unlink should handle gracefully
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = TensorBlob.unlink(blob_path)
        
        assert result is False
        assert len(w) == 1
        assert "Failed to unlink" in str(w[0].message)


def test_unlink_does_not_affect_other_blobs(temp_blob_dir):
    """Test that unlinking one blob doesn't affect others."""
    blob1_path = temp_blob_dir / "blob1"
    blob2_path = temp_blob_dir / "blob2"
    
    # Create two blobs
    with TensorBlob.open(blob1_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    with TensorBlob.open(blob2_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    # Unlink first blob
    result = TensorBlob.unlink(blob1_path)
    
    assert result is True
    assert not blob1_path.exists()
    assert blob2_path.exists()
    
    # Verify second blob is still accessible
    with TensorBlob.open(blob2_path, "r") as blob:
        data = blob.read()
        assert data.shape == (10, 5)


def test_unlink_with_pathlib_path(temp_blob_dir):
    """Test unlinking using pathlib.Path object."""
    blob_path = Path(temp_blob_dir) / "pathlib_test"
    
    # Create blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    # Unlink using Path object
    result = TensorBlob.unlink(blob_path)
    
    assert result is True
    assert not blob_path.exists()


def test_unlink_with_string_path(temp_blob_dir):
    """Test unlinking using string path."""
    blob_path = str(temp_blob_dir / "string_test")
    
    # Create blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    # Unlink using string
    result = TensorBlob.unlink(blob_path)
    
    assert result is True
    assert not Path(blob_path).exists()


def test_unlink_returns_false_on_permission_error(temp_blob_dir):
    """Test that unlink returns False on permission errors."""
    blob_path = temp_blob_dir / "permission_test"
    
    # Create blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    # Make directory read-only (this might not work on all systems)
    original_mode = os.stat(blob_path).st_mode
    try:
        os.chmod(blob_path, 0o444)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = TensorBlob.unlink(blob_path)
            
            # Should warn and return False
            assert result is False
            assert len(w) == 1
            assert "Failed to unlink" in str(w[0].message)
    finally:
        # Restore permissions for cleanup
        os.chmod(blob_path, original_mode)


def test_unlink_recreate_blob(temp_blob_dir):
    """Test creating a new blob after unlinking."""
    blob_path = temp_blob_dir / "recreate_test"
    
    # Create first blob
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(5,)) as blob:
        blob.write(torch.randn(10, 5))
    
    # Unlink
    result = TensorBlob.unlink(blob_path)
    assert result is True
    assert not blob_path.exists()
    
    # Create new blob at same location with different shape
    with TensorBlob.open(blob_path, "w", dtype="float32", shape=(8,)) as blob:
        blob.write(torch.randn(15, 8))
    
    # Verify new blob
    with TensorBlob.open(blob_path, "r") as blob:
        assert len(blob) == 15
        assert blob.shape == (8,)

