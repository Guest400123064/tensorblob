"""Tests for slicing operations on TensorBlob."""

import numpy as np
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



class TestBatchIndexing:
    """Tests for vectorized batch (fancy) indexing."""

    def test_batch_matches_row_by_row(self, blob_with_data):
        """Test batch indexing returns the same rows as repeated indexing."""
        blob_dir, sample_data = blob_with_data
        idxs = [3, 0, 99, 42, 7, 50]

        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[idxs]
            assert result.shape == (6, 10)
            assert torch.allclose(result, sample_data[idxs])

    def test_batch_order_and_duplicates(self, blob_with_data):
        """Test input order is preserved and duplicates are allowed."""
        blob_dir, sample_data = blob_with_data
        idxs = [5, 1, 5, 1, 90]

        with TensorBlob.open(blob_dir, "r") as blob:
            assert torch.allclose(blob[idxs], sample_data[idxs])

    def test_batch_negative_indices(self, blob_with_data):
        """Test negative indices in batch indexing."""
        blob_dir, sample_data = blob_with_data
        idxs = [-1, -100, 0, -50]

        with TensorBlob.open(blob_dir, "r") as blob:
            assert torch.allclose(blob[idxs], sample_data[idxs])

    def test_batch_torch_tensor(self, blob_with_data):
        """Test indexing with a torch tensor of indices."""
        blob_dir, sample_data = blob_with_data
        idxs = torch.tensor([10, 20, 30])

        with TensorBlob.open(blob_dir, "r") as blob:
            assert torch.allclose(blob[idxs], sample_data[idxs])

    def test_batch_numpy_array(self, blob_with_data):
        """Test indexing with a numpy array of indices."""
        blob_dir, sample_data = blob_with_data
        idxs = np.array([4, 0, 77, 4])

        with TensorBlob.open(blob_dir, "r") as blob:
            assert torch.allclose(blob[idxs], sample_data[idxs.tolist()])

    def test_batch_tuple(self, blob_with_data):
        """Test indexing with a tuple of indices."""
        blob_dir, sample_data = blob_with_data

        with TensorBlob.open(blob_dir, "r") as blob:
            assert torch.allclose(blob[(1, 2, 3)], sample_data[[1, 2, 3]])

    def test_batch_single_element(self, blob_with_data):
        """Test single-element batch keeps the batch dimension."""
        blob_dir, sample_data = blob_with_data

        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[[5]]
            assert result.shape == (1, 10)
            assert torch.allclose(result, sample_data[[5]])

    def test_batch_empty(self, blob_with_data):
        """Test empty batch returns an empty tensor."""
        blob_dir, _ = blob_with_data

        with TensorBlob.open(blob_dir, "r") as blob:
            result = blob[[]]
            assert result.shape == (0, 10)

    def test_batch_across_blocks(self, multi_block_blob):
        """Test batch indexing spanning multiple blocks."""
        blob_dir, data, _ = multi_block_blob
        idxs = [0, 49, 50, 51, 149, 75, 0]

        with TensorBlob.open(blob_dir, "r") as blob:
            assert torch.allclose(blob[idxs], data[idxs])

    def test_batch_out_of_bounds(self, blob_with_data):
        """Test that out-of-bounds batch indices raise IndexError."""
        blob_dir, _ = blob_with_data

        with TensorBlob.open(blob_dir, "r") as blob:
            with pytest.raises(IndexError, match="out of bounds"):
                _ = blob[[0, 100]]
            with pytest.raises(IndexError, match="out of bounds"):
                _ = blob[[-101]]

    def test_batch_invalid_dtypes(self, blob_with_data):
        """Test that non-integer batch indices raise TypeError."""
        blob_dir, _ = blob_with_data

        with TensorBlob.open(blob_dir, "r") as blob:
            with pytest.raises(TypeError, match="integer dtype"):
                _ = blob[torch.tensor([True, False])]
            with pytest.raises(TypeError, match="integer dtype"):
                _ = blob[[1.5, 2.0]]
            with pytest.raises(ValueError, match="1-dimensional"):
                _ = blob[torch.tensor([[1, 2], [3, 4]])]

    def test_batch_returns_copy(self, temp_blob_dir):
        """Test that batch results are copies, not references."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(5,)) as blob:
            blob.write(torch.ones(10, 5))

        with TensorBlob.open(temp_blob_dir, "r") as blob:
            batch = blob[[0, 1]]
            batch[0] = 999.0
            assert torch.allclose(blob[0], torch.ones(5))

    def test_batch_sorted_fast_path_equivalence(self, blob_with_data):
        """Test sorted and unsorted queries return the same rows."""
        blob_dir, sample_data = blob_with_data
        idxs = torch.randperm(100)[:40]

        with TensorBlob.open(blob_dir, "r") as blob:
            unsorted_result = blob[idxs]
            order = torch.argsort(idxs)
            sorted_result = blob[idxs[order]]
            assert torch.allclose(sorted_result, sample_data[idxs[order]])
            assert torch.allclose(
                unsorted_result, sorted_result[torch.argsort(order)]
            )
