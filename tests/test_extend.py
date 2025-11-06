"""Tests for extend and truncate operations on TensorBlob."""

import pytest
import torch
from tensorblob import TensorBlob


class TestTruncateBasic:
    """Tests for basic truncate operations."""
    
    def test_truncate_at_position(self, temp_blob_dir):
        """Test truncating blob at specific position."""
        data = torch.randn(100, 5)
        
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(5,)) as blob:
            blob.write(data)
            assert len(blob) == 100
            
            # Truncate to 50 elements
            blob.truncate(50)
            assert len(blob) == 50
            assert blob.tell() == 50
            
            # Verify data integrity
            blob.seek(0)
            remaining = blob.read()
            assert torch.allclose(remaining, data[:50])
    
    def test_truncate_at_current_position(self, temp_blob_dir):
        """Test truncating at current position (no argument)."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.randn(100, 5))
            
            blob.seek(30)
            blob.truncate()
            assert len(blob) == 30
            assert blob.tell() == 30
    
    def test_truncate_to_zero(self, temp_blob_dir):
        """Test truncating to zero length."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.randn(50, 5))
            
            blob.seek(0)  # Need to seek to position 0 first
            blob.truncate(0)
            assert len(blob) == 0
            result = blob.read()
            assert result.size(0) == 0
    
    def test_truncate_then_write(self, temp_blob_dir):
        """Test writing after truncation."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.randn(100, 5))
            blob.truncate(50)
            
            new_data = torch.ones(10, 5)
            blob.write(new_data)
            assert len(blob) == 60
            
            # Verify new data
            blob.seek(50)
            result = blob.read(size=10)
            assert torch.allclose(result, new_data)
    
    def test_truncate_requires_writable_mode(self, temp_blob_dir):
        """Test that truncate requires writable mode."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.randn(100, 5))
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            with pytest.raises(IOError, match="not open for writing"):
                blob.truncate(50)


class TestTruncateMultiBlock:
    """Tests for truncating across block boundaries."""
    
    def test_truncate_partial_block(self, temp_blob_dir):
        """Test truncating in middle of a block."""
        with TensorBlob.open(
            temp_blob_dir, "w+", dtype="float32", shape=(5,), block_size=50
        ) as blob:
            blob.write(torch.randn(150, 5))
            assert len(blob._status.bds) == 3
            
            # Truncate to middle of second block
            blob.truncate(75)
            assert len(blob) == 75
            assert len(blob._status.bds) == 2  # Should have 2 blocks
    
    def test_truncate_at_block_boundary(self, temp_blob_dir):
        """Test truncating exactly at block boundary."""
        with TensorBlob.open(
            temp_blob_dir, "w+", dtype="float32", shape=(5,), block_size=50
        ) as blob:
            blob.write(torch.randn(150, 5))
            
            # Truncate at exact block boundary
            blob.truncate(100)
            assert len(blob) == 100
            assert len(blob._status.bds) == 2
    
    def test_truncate_removes_blocks(self, temp_blob_dir):
        """Test that truncate removes unused blocks."""
        
        with TensorBlob.open(
            temp_blob_dir, "w+", dtype="float32", shape=(5,), block_size=50
        ) as blob:
            blob.write(torch.randn(150, 5))
            original_blocks = len(blob._status.bds)
            
            blob.truncate(25)
            new_blocks = len(blob._status.bds)
            
            assert new_blocks < original_blocks
            assert new_blocks == 1
    
    def test_truncate_persists_across_sessions(self, temp_blob_dir):
        """Test that truncation is persisted."""
        data = torch.randn(100, 5)
        
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(data)
            blob.truncate(50)
        
        # Reopen and verify
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert len(blob) == 50
            result = blob.read()
            assert torch.allclose(result, data[:50])


class TestExtendBasic:
    """Tests for basic extend operations."""
    
    def test_extend_order_preserving(self, temp_blob_dir):
        """Test extending blob with order preservation."""
        data1 = torch.ones(50, 5)
        data2 = torch.ones(30, 5) * 2
        
        # Create first blob
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(blob1_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(data1)
        
        # Create second blob
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(blob2_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(data2)
        
        # Extend first with second
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                blob1.extend(blob2, maintain_order=True)
        
        # Verify
        with TensorBlob.open(blob1_dir, "r") as blob:
            assert len(blob) == 80
            assert torch.allclose(blob[:50], data1)
            assert torch.allclose(blob[50:], data2)
    
    def test_extend_fast_mode(self, temp_blob_dir):
        """Test extending blob in fast mode (non-order-preserving)."""
        block_size = 50
        data1 = torch.ones(100, 5)
        data2 = torch.ones(75, 5) * 2
        
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(
            blob1_dir, "w", dtype="float32", shape=(5,), block_size=block_size
        ) as blob:
            blob.write(data1)
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(
            blob2_dir, "w", dtype="float32", shape=(5,), block_size=block_size
        ) as blob:
            blob.write(data2)
        
        # Extend in fast mode
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                blob1.extend(blob2, maintain_order=False)
        
        # Verify total length
        with TensorBlob.open(blob1_dir, "r") as blob:
            assert len(blob) == 175
    
    def test_extend_empty_blob(self, temp_blob_dir):
        """Test extending empty blob."""
        data = torch.randn(50, 5)
        
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(blob1_dir, "w", dtype="float32", shape=(5,)) as blob:
            pass  # Empty
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(blob2_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(data)
        
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                blob1.extend(blob2, maintain_order=True)
        
        with TensorBlob.open(blob1_dir, "r") as blob:
            assert len(blob) == 50
            assert torch.allclose(blob[:], data)
    
    def test_extend_with_empty_blob(self, temp_blob_dir):
        """Test extending with empty blob."""
        data = torch.randn(50, 5)
        
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(blob1_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(data)
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(blob2_dir, "w", dtype="float32", shape=(5,)) as blob:
            pass  # Empty
        
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                blob1.extend(blob2, maintain_order=True)
        
        with TensorBlob.open(blob1_dir, "r") as blob:
            assert len(blob) == 50
            assert torch.allclose(blob[:], data)


class TestExtendValidation:
    """Tests for extend validation and error handling."""
    
    def test_extend_dtype_mismatch(self, temp_blob_dir):
        """Test that extend fails with mismatched dtypes."""
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(blob1_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.randn(10, 5))
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(blob2_dir, "w", dtype="float64", shape=(5,)) as blob:
            blob.write(torch.randn(10, 5))
        
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                with pytest.raises(ValueError, match="data types.*must match"):
                    blob1.extend(blob2)
    
    def test_extend_shape_mismatch(self, temp_blob_dir):
        """Test that extend fails with mismatched shapes."""
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(blob1_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.randn(10, 5))
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(blob2_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(torch.randn(10, 10))
        
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                with pytest.raises(ValueError, match="shapes must match"):
                    blob1.extend(blob2)
    
    def test_extend_block_size_mismatch_fast_mode(self, temp_blob_dir):
        """Test that fast extend fails with mismatched block sizes."""
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(
            blob1_dir, "w", dtype="float32", shape=(5,), block_size=50
        ) as blob:
            blob.write(torch.randn(100, 5))
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(
            blob2_dir, "w", dtype="float32", shape=(5,), block_size=100
        ) as blob:
            blob.write(torch.randn(100, 5))
        
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                with pytest.raises(ValueError, match="Block sizes must match"):
                    blob1.extend(blob2, maintain_order=False)
    
    def test_extend_requires_writable_mode(self, temp_blob_dir):
        """Test that extend requires writable mode."""
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(blob1_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.randn(10, 5))
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(blob2_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.randn(10, 5))
        
        with TensorBlob.open(blob1_dir, "r") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                with pytest.raises(IOError, match="not open for writing"):
                    blob1.extend(blob2)


class TestExtendFastMode:
    """Tests for fast (non-order-preserving) extend mode."""
    
    def test_fast_extend_copies_blocks(self, temp_blob_dir):
        """Test that fast extend copies complete blocks."""
        block_size = 50
        
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(
            blob1_dir, "w", dtype="float32", shape=(5,), block_size=block_size
        ) as blob:
            blob.write(torch.randn(100, 5))
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(
            blob2_dir, "w", dtype="float32", shape=(5,), block_size=block_size
        ) as blob:
            blob.write(torch.randn(150, 5))
        
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                blob1.extend(blob2, maintain_order=False)
        
        with TensorBlob.open(blob1_dir, "r") as blob:
            # Should have data from both blobs
            assert len(blob) == 250
            # Note: Order not preserved in fast mode
    
    def test_fast_extend_merges_partial_blocks(self, temp_blob_dir):
        """Test that fast extend merges incomplete blocks."""
        block_size = 50
        
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(
            blob1_dir, "w", dtype="float32", shape=(5,), block_size=block_size
        ) as blob:
            blob.write(torch.randn(75, 5))  # 1.5 blocks
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(
            blob2_dir, "w", dtype="float32", shape=(5,), block_size=block_size
        ) as blob:
            blob.write(torch.randn(80, 5))  # 1.6 blocks
        
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                blob1.extend(blob2, maintain_order=False)
        
        with TensorBlob.open(blob1_dir, "r") as blob:
            assert len(blob) == 155


class TestTruncateAndExtendCombined:
    """Tests combining truncate and extend operations."""
    
    def test_truncate_then_extend(self, temp_blob_dir):
        """Test truncating then extending."""
        data1 = torch.ones(100, 5)
        data2 = torch.ones(50, 5) * 2
        
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(blob1_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(data1)
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(blob2_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(data2)
        
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            # Truncate first
            blob1.truncate(50)
            assert len(blob1) == 50
            
            # Then extend
            with TensorBlob.open(blob2_dir, "r") as blob2:
                blob1.extend(blob2, maintain_order=True)
        
        with TensorBlob.open(blob1_dir, "r") as blob:
            assert len(blob) == 100
            assert torch.allclose(blob[:50], data1[:50])
            assert torch.allclose(blob[50:], data2)
    
    def test_extend_then_truncate(self, temp_blob_dir):
        """Test extending then truncating."""
        blob1_dir = temp_blob_dir / "blob1"
        with TensorBlob.open(blob1_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.ones(50, 5))
        
        blob2_dir = temp_blob_dir / "blob2"
        with TensorBlob.open(blob2_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.ones(50, 5) * 2)
        
        with TensorBlob.open(blob1_dir, "r+") as blob1:
            with TensorBlob.open(blob2_dir, "r") as blob2:
                blob1.extend(blob2, maintain_order=True)
            assert len(blob1) == 100
            
            # Truncate back
            blob1.truncate(75)
            assert len(blob1) == 75
        
        with TensorBlob.open(blob1_dir, "r") as blob:
            assert len(blob) == 75

