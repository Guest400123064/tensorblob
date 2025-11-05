"""Tests for slicing operations on TensorBlob."""

import pytest
import torch
from tensorblob import TensorBlob


class TestBasicSlicing:
    """Tests for basic slicing operations."""
    
    def test_slice_start_stop(self, blob_with_data):
        """Test basic start:stop slicing."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[10:20]
            expected = sample_data[10:20]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
    
    def test_slice_start_only(self, blob_with_data):
        """Test slicing with only start index."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[80:]
            expected = sample_data[80:]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
    
    def test_slice_stop_only(self, blob_with_data):
        """Test slicing with only stop index."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[:30]
            expected = sample_data[:30]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
    
    def test_slice_entire_range(self, blob_with_data):
        """Test slicing entire range with [:] ."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[:]
            expected = sample_data[:]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
    
    def test_empty_slice(self, blob_with_data):
        """Test empty slice."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[50:50]
            assert result.shape == (0, 10)
    
    def test_single_element_slice(self, blob_with_data):
        """Test slice that selects single element."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[42:43]
            expected = sample_data[42:43]
            assert result.shape == (1, 10)
            assert torch.allclose(result, expected)


class TestNegativeSlicing:
    """Tests for slicing with negative indices."""
    
    def test_negative_start(self, blob_with_data):
        """Test slicing with negative start index."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[-20:]
            expected = sample_data[-20:]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
    
    def test_negative_stop(self, blob_with_data):
        """Test slicing with negative stop index."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[:-10]
            expected = sample_data[:-10]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
    
    def test_both_negative(self, blob_with_data):
        """Test slicing with both negative indices."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[-50:-30]
            expected = sample_data[-50:-30]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)


class TestStepSlicing:
    """Tests for slicing with step parameter."""
    
    def test_positive_step(self, blob_with_data):
        """Test slicing with positive step."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[::2]
            expected = sample_data[::2]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
    
    def test_step_with_start_stop(self, blob_with_data):
        """Test slicing with start, stop, and step."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[10:50:3]
            expected = sample_data[10:50:3]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
    
    def test_negative_step(self, blob_with_data):
        """Test slicing with negative step (reverse)."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            # Note: Negative steps work, collecting individual items then reversing
            result = blob[::-1]
            # Build expected by reversing the sample data
            expected = torch.stack([sample_data[i] for i in range(len(sample_data) - 1, -1, -1)])
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
    
    def test_negative_step_with_indices(self, blob_with_data):
        """Test slicing with negative step and explicit indices."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            # Build expected by collecting items with negative step
            result = blob[50:10:-2]
            expected = torch.stack([sample_data[i] for i in range(50, 10, -2)])
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
    
    def test_large_step(self, blob_with_data):
        """Test slicing with large step."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[::10]
            expected = sample_data[::10]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)


class TestSlicingAcrossBlocks:
    """Tests for slicing that spans multiple blocks."""
    
    def test_slice_single_block(self, multi_block_blob):
        """Test slice within a single block."""
        blob_dir, data, _ = multi_block_blob
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[10:30]
            expected = data[10:30]
            assert torch.allclose(result, expected)
    
    def test_slice_across_two_blocks(self, multi_block_blob):
        """Test slice spanning two blocks."""
        blob_dir, data, _ = multi_block_blob
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[40:60]
            expected = data[40:60]
            assert torch.allclose(result, expected)
    
    def test_slice_across_all_blocks(self, multi_block_blob):
        """Test slice spanning all blocks."""
        blob_dir, data, _ = multi_block_blob
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[10:140]
            expected = data[10:140]
            assert torch.allclose(result, expected)
    
    def test_slice_with_step_across_blocks(self, multi_block_blob):
        """Test slice with step spanning multiple blocks."""
        blob_dir, data, _ = multi_block_blob
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[10:140:5]
            expected = data[10:140:5]
            assert torch.allclose(result, expected)


class TestSliceEdgeCases:
    """Tests for edge cases in slicing."""
    
    def test_out_of_bounds_slice(self, blob_with_data):
        """Test that out-of-bounds slices are handled gracefully."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            # Stop beyond length
            result = blob[80:200]
            expected = sample_data[80:200]
            assert result.shape == expected.shape
            assert torch.allclose(result, expected)
            
            # Start beyond length
            result = blob[200:300]
            expected = sample_data[200:300]
            assert result.shape == expected.shape  # Should be (0, 10)
    
    def test_reversed_indices(self, blob_with_data):
        """Test slice with start > stop (positive step)."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[50:30]
            assert result.shape == (0, 10)
    
    def test_slice_empty_blob(self, temp_blob_dir):
        """Test slicing an empty blob."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            pass  # Empty blob
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            result = blob[:]
            assert result.shape == (0, 10)
            
            result = blob[0:10]
            assert result.shape == (0, 10)
    
    def test_invalid_index_type(self, blob_with_data):
        """Test that invalid index types raise TypeError."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            with pytest.raises(TypeError, match="Index must be"):
                _ = blob["invalid"]
            
            with pytest.raises(TypeError, match="Index must be"):
                _ = blob[3.14]


class TestSliceReturnsCopy:
    """Tests that slicing returns copies, not references."""
    
    def test_slice_modification_doesnt_affect_original(self, temp_blob_dir):
        """Test that modifying sliced result doesn't affect original."""
        data = torch.ones(10, 5)
        
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            sliced = blob[0:5]
            sliced[0] = 999.0
            
            # Original should be unchanged
            original = blob[0]
            assert torch.allclose(original, torch.ones(5))


class TestSliceComparison:
    """Tests comparing slicing vs indexing for consistency."""
    
    def test_slice_equals_repeated_indexing(self, blob_with_data):
        """Test that slicing matches repeated integer indexing."""
        blob_dir, _ = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            sliced = blob[20:25]
            indexed = torch.stack([blob[i] for i in range(20, 25)])
            assert torch.allclose(sliced, indexed)
    
    def test_slice_consistency_with_data(self, blob_with_data):
        """Test that all slice forms give consistent results."""
        blob_dir, sample_data = blob_with_data
        
        with TensorBlob.open(blob_dir, "r") as blob:
            # Various equivalent slices
            assert torch.allclose(blob[30:40], sample_data[30:40])
            assert torch.allclose(blob[30:40:1], sample_data[30:40:1])
            
            # Verify individual elements match
            for i in range(30, 40):
                assert torch.allclose(blob[i], sample_data[i])

