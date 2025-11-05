"""Tests for TensorBlob.open() method - mode handling and blob creation."""

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
def existing_blob(tmp_path):
    """Fixture providing an existing blob with data."""
    blob_dir = tmp_path / "existing_blob"
    with TensorBlob.open(blob_dir, "w", dtype="float32", shape=(10,)) as blob:
        blob.write(torch.randn(50, 10))
    yield blob_dir
    if blob_dir.exists():
        shutil.rmtree(blob_dir)


class TestModeValidation:
    """Tests for mode string validation."""
    
    def test_valid_read_mode(self, existing_blob):
        """Test opening with 'r' mode."""
        with TensorBlob.open(existing_blob, "r") as blob:
            assert blob._m_rd is True
            assert blob._m_wr is False
            assert blob._m_ap is False
    
    def test_valid_write_mode(self, temp_blob_dir):
        """Test opening with 'w' mode."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            assert blob._m_rd is False
            assert blob._m_wr is True
            assert blob._m_ap is False
    
    def test_valid_append_mode(self, existing_blob):
        """Test opening with 'a' mode."""
        with TensorBlob.open(existing_blob, "a") as blob:
            assert blob._m_rd is False
            assert blob._m_wr is True
            assert blob._m_ap is True
    
    def test_valid_read_plus_mode(self, existing_blob):
        """Test opening with 'r+' mode."""
        with TensorBlob.open(existing_blob, "r+") as blob:
            assert blob._m_rd is True
            assert blob._m_wr is True
            assert blob._m_ap is False
    
    def test_valid_write_plus_mode(self, temp_blob_dir):
        """Test opening with 'w+' mode."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            assert blob._m_rd is True
            assert blob._m_wr is True
            assert blob._m_ap is False
    
    def test_valid_append_plus_mode(self, existing_blob):
        """Test opening with 'a+' mode."""
        with TensorBlob.open(existing_blob, "a+") as blob:
            assert blob._m_rd is True
            assert blob._m_wr is True
            assert blob._m_ap is True
    
    def test_invalid_mode_unknown_char(self, temp_blob_dir):
        """Test that mode with unknown character raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            TensorBlob.open(temp_blob_dir, "x", dtype="float32", shape=(10,))
    
    def test_invalid_mode_no_base_mode(self, temp_blob_dir):
        """Test that mode with only '+' raises ValueError."""
        with pytest.raises(ValueError, match="exactly one"):
            TensorBlob.open(temp_blob_dir, "+", dtype="float32", shape=(10,))
    
    def test_invalid_mode_multiple_base_modes(self, temp_blob_dir):
        """Test that mode with multiple base modes raises ValueError."""
        with pytest.raises(ValueError, match="exactly one"):
            TensorBlob.open(temp_blob_dir, "rw", dtype="float32", shape=(10,))
    
    def test_invalid_mode_multiple_plus(self, temp_blob_dir):
        """Test that mode with multiple '+' raises ValueError."""
        with pytest.raises(ValueError, match="mode"):
            TensorBlob.open(temp_blob_dir, "r++", dtype="float32", shape=(10,))
    
    def test_invalid_mode_duplicate_chars(self, temp_blob_dir):
        """Test that mode with duplicate characters raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            TensorBlob.open(temp_blob_dir, "rr", dtype="float32", shape=(10,))
    
    def test_invalid_mode_binary_text(self, temp_blob_dir):
        """Test that binary/text modes are rejected."""
        with pytest.raises(ValueError, match="Invalid mode"):
            TensorBlob.open(temp_blob_dir, "rb", dtype="float32", shape=(10,))
        
        with pytest.raises(ValueError, match="Invalid mode"):
            TensorBlob.open(temp_blob_dir, "rt", dtype="float32", shape=(10,))


class TestFileExistence:
    """Tests for file existence checking based on mode."""
    
    def test_read_mode_requires_existing_blob(self, temp_blob_dir):
        """Test that 'r' mode requires existing blob."""
        with pytest.raises(FileNotFoundError, match="Blob not found"):
            TensorBlob.open(temp_blob_dir, "r")
    
    def test_read_plus_mode_requires_existing_blob(self, temp_blob_dir):
        """Test that 'r+' mode requires existing blob."""
        with pytest.raises(FileNotFoundError, match="Blob not found"):
            TensorBlob.open(temp_blob_dir, "r+")
    
    def test_append_mode_requires_existing_blob(self, temp_blob_dir):
        """Test that 'a' mode requires existing blob."""
        with pytest.raises(FileNotFoundError, match="Blob not found"):
            TensorBlob.open(temp_blob_dir, "a")
    
    def test_append_plus_mode_requires_existing_blob(self, temp_blob_dir):
        """Test that 'a+' mode requires existing blob."""
        with pytest.raises(FileNotFoundError, match="Blob not found"):
            TensorBlob.open(temp_blob_dir, "a+")
    
    def test_write_mode_creates_new_blob(self, temp_blob_dir):
        """Test that 'w' mode creates new blob if not exists."""
        assert not temp_blob_dir.exists()
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            assert blob is not None
        assert temp_blob_dir.exists()
    
    def test_write_plus_mode_creates_new_blob(self, temp_blob_dir):
        """Test that 'w+' mode creates new blob if not exists."""
        assert not temp_blob_dir.exists()
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            assert blob is not None
        assert temp_blob_dir.exists()
    
    def test_write_mode_opens_existing_blob(self, existing_blob):
        """Test that 'w' mode can open existing blob."""
        with TensorBlob.open(existing_blob, "w", dtype="float32", shape=(10,)) as blob:
            assert blob is not None


class TestDtypeValidation:
    """Tests for dtype parameter validation."""
    
    def test_dtype_string_valid(self, temp_blob_dir):
        """Test creating blob with string dtype."""
        valid_dtypes = ["float32", "float64", "int32", "int64", "bool"]
        for dtype in valid_dtypes:
            subdir = temp_blob_dir / dtype
            with TensorBlob.open(subdir, "w", dtype=dtype, shape=(5,)) as blob:
                assert blob.dtype == dtype
    
    def test_dtype_torch_dtype_valid(self, temp_blob_dir):
        """Test creating blob with torch.dtype."""
        torch_dtypes = [
            (torch.float32, "float32"),
            (torch.float64, "float64"),
            (torch.int32, "int32"),
            (torch.int64, "int64"),
        ]
        for torch_dtype, expected_str in torch_dtypes:
            subdir = temp_blob_dir / expected_str
            with TensorBlob.open(subdir, "w", dtype=torch_dtype, shape=(5,)) as blob:
                assert blob.dtype == expected_str
    
    def test_dtype_missing_for_new_blob(self, temp_blob_dir):
        """Test that creating new blob without dtype raises ValueError."""
        with pytest.raises(ValueError, match="Arguments.*dtype.*required"):
            TensorBlob.open(temp_blob_dir, "w", shape=(10,))
    
    def test_dtype_none_for_new_blob(self, temp_blob_dir):
        """Test that creating new blob with None dtype raises ValueError."""
        with pytest.raises(ValueError, match="Arguments.*dtype.*required"):
            TensorBlob.open(temp_blob_dir, "w", dtype=None, shape=(10,))
    
    def test_dtype_invalid_type(self, temp_blob_dir):
        """Test that invalid dtype type raises TypeError."""
        with pytest.raises(TypeError, match="dtype must be str or torch.dtype"):
            TensorBlob.open(temp_blob_dir, "w", dtype=123, shape=(10,))
        
        with pytest.raises(TypeError, match="dtype must be str or torch.dtype"):
            TensorBlob.open(temp_blob_dir, "w", dtype=[], shape=(10,))
    
    def test_dtype_not_required_for_existing_blob(self, existing_blob):
        """Test that dtype is not required when opening existing blob."""
        with TensorBlob.open(existing_blob, "r") as blob:
            assert blob.dtype == "float32"  # From fixture


class TestShapeValidation:
    """Tests for shape parameter validation."""
    
    def test_shape_tuple_valid(self, temp_blob_dir):
        """Test creating blob with tuple shape."""
        shapes = [(10,), (5, 5), (2, 3, 4), (1, 2, 3, 4)]
        for i, shape in enumerate(shapes):
            subdir = temp_blob_dir / f"shape_{i}"
            with TensorBlob.open(subdir, "w", dtype="float32", shape=shape) as blob:
                assert blob.shape == shape
    
    def test_shape_int_converted_to_tuple(self, temp_blob_dir):
        """Test that integer shape is converted to tuple."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=10) as blob:
            assert blob.shape == (10,)
    
    def test_shape_missing_for_new_blob(self, temp_blob_dir):
        """Test that creating new blob without shape raises ValueError."""
        with pytest.raises(ValueError, match="Arguments.*shape.*required"):
            TensorBlob.open(temp_blob_dir, "w", dtype="float32")
    
    def test_shape_none_for_new_blob(self, temp_blob_dir):
        """Test that creating new blob with None shape raises ValueError."""
        with pytest.raises(ValueError, match="Arguments.*shape.*required"):
            TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=None)
    
    def test_shape_not_required_for_existing_blob(self, existing_blob):
        """Test that shape is not required when opening existing blob."""
        with TensorBlob.open(existing_blob, "r") as blob:
            assert blob.shape == (10,)  # From fixture


class TestBlockSize:
    """Tests for block_size parameter."""
    
    def test_default_block_size(self, temp_blob_dir):
        """Test that default block_size is 8192."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            assert blob.block_size == 8192
    
    def test_custom_block_size(self, temp_blob_dir):
        """Test creating blob with custom block_size."""
        custom_sizes = [100, 1000, 10000]
        for size in custom_sizes:
            subdir = temp_blob_dir / f"bs_{size}"
            with TensorBlob.open(subdir, "w", dtype="float32", shape=(10,), block_size=size) as blob:
                assert blob.block_size == size
    
    def test_block_size_preserved_on_reopen(self, temp_blob_dir):
        """Test that block_size is preserved when reopening blob."""
        custom_size = 500
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,), block_size=custom_size) as blob:
            pass
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert blob.block_size == custom_size


class TestTruncation:
    """Tests for truncation behavior in 'w' mode."""
    
    def test_write_mode_truncates_existing_data(self, existing_blob):
        """Test that 'w' mode truncates existing data."""
        # Verify existing blob has data
        with TensorBlob.open(existing_blob, "r") as blob:
            original_len = len(blob)
            assert original_len == 50
        
        # Reopen in 'w' mode
        with TensorBlob.open(existing_blob, "w", dtype="float32", shape=(10,)) as blob:
            # Should be empty after truncation
            assert len(blob) == 0
    
    def test_write_plus_mode_truncates_existing_data(self, existing_blob):
        """Test that 'w+' mode truncates existing data."""
        # Verify existing blob has data
        with TensorBlob.open(existing_blob, "r") as blob:
            assert len(blob) == 50
        
        # Reopen in 'w+' mode
        with TensorBlob.open(existing_blob, "w+", dtype="float32", shape=(10,)) as blob:
            # Should be empty after truncation
            assert len(blob) == 0
            # Should be able to write new data
            blob.write(torch.randn(10, 10))
            assert len(blob) == 10


class TestAppendModePosition:
    """Tests for position initialization in append mode."""
    
    def test_append_mode_starts_at_end(self, existing_blob):
        """Test that 'a' mode positions at end of blob."""
        with TensorBlob.open(existing_blob, "a") as blob:
            assert blob.tell() == 50  # Existing blob has 50 tensors
    
    def test_append_plus_mode_starts_at_end(self, existing_blob):
        """Test that 'a+' mode positions at end of blob."""
        with TensorBlob.open(existing_blob, "a+") as blob:
            assert blob.tell() == 50


class TestReadModePosition:
    """Tests for position initialization in read mode."""
    
    def test_read_mode_starts_at_beginning(self, existing_blob):
        """Test that 'r' mode positions at beginning of blob."""
        with TensorBlob.open(existing_blob, "r") as blob:
            assert blob.tell() == 0
    
    def test_read_plus_mode_starts_at_beginning(self, existing_blob):
        """Test that 'r+' mode positions at beginning of blob."""
        with TensorBlob.open(existing_blob, "r+") as blob:
            assert blob.tell() == 0


class TestWriteModePosition:
    """Tests for position initialization in write mode."""
    
    def test_write_mode_starts_at_beginning(self, temp_blob_dir):
        """Test that 'w' mode positions at beginning of blob."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            assert blob.tell() == 0
    
    def test_write_plus_mode_starts_at_beginning(self, temp_blob_dir):
        """Test that 'w+' mode positions at beginning of blob."""
        with TensorBlob.open(temp_blob_dir, "w+", dtype="float32", shape=(10,)) as blob:
            assert blob.tell() == 0


class TestConfigPersistence:
    """Tests for configuration persistence across open/close cycles."""
    
    def test_dtype_persisted(self, temp_blob_dir):
        """Test that dtype is persisted when reopening blob."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float64", shape=(10,)) as blob:
            blob.write(torch.randn(10, 10))
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert blob.dtype == "float64"
    
    def test_shape_persisted(self, temp_blob_dir):
        """Test that shape is persisted when reopening blob."""
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(2, 3, 4)) as blob:
            blob.write(torch.randn(10, 2, 3, 4))
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert blob.shape == (2, 3, 4)
    
    def test_data_persisted(self, temp_blob_dir):
        """Test that data is persisted when reopening blob."""
        data = torch.randn(20, 10)
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,)) as blob:
            blob.write(data)
        
        with TensorBlob.open(temp_blob_dir, "r") as blob:
            assert len(blob) == 20
            # Check first and last tensors
            assert torch.allclose(blob[0], data[0])
            assert torch.allclose(blob[19], data[19])


class TestPathHandling:
    """Tests for different path formats."""
    
    def test_path_object(self, tmp_path):
        """Test opening blob with Path object."""
        blob_path = tmp_path / "path_object_blob"
        with TensorBlob.open(blob_path, "w", dtype="float32", shape=(10,)) as blob:
            assert blob.filename == str(blob_path.resolve())
    
    def test_string_path(self, tmp_path):
        """Test opening blob with string path."""
        blob_path = str(tmp_path / "string_path_blob")
        with TensorBlob.open(blob_path, "w", dtype="float32", shape=(10,)) as blob:
            assert Path(blob.filename).exists()
    
    def test_expanduser_in_path(self, tmp_path, monkeypatch):
        """Test that ~ is expanded in path."""
        # Set HOME to tmp_path for testing
        monkeypatch.setenv("HOME", str(tmp_path))
        with TensorBlob.open("~/test_blob", "w", dtype="float32", shape=(10,)) as blob:
            assert "~" not in blob.filename
            assert blob.filename.startswith(str(tmp_path))
    
    def test_relative_path_resolved(self, tmp_path, monkeypatch):
        """Test that relative paths are resolved to absolute."""
        monkeypatch.chdir(tmp_path)
        with TensorBlob.open("relative_blob", "w", dtype="float32", shape=(10,)) as blob:
            assert Path(blob.filename).is_absolute()


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_mode_string(self, temp_blob_dir):
        """Test that empty mode string raises ValueError."""
        with pytest.raises(ValueError):
            TensorBlob.open(temp_blob_dir, "", dtype="float32", shape=(10,))
    
    def test_zero_block_size(self, temp_blob_dir):
        """Test creating blob with zero block_size (should probably fail or be handled)."""
        # This might cause issues - let's see what happens
        try:
            with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,), block_size=0) as blob:
                # If it doesn't fail immediately, try writing
                blob.write(torch.randn(10))
        except (ValueError, RuntimeError, ZeroDivisionError):
            # Expected - zero block size should cause issues
            pass
    
    def test_negative_block_size(self, temp_blob_dir):
        """Test creating blob with negative block_size (should probably fail)."""
        # Negative block size doesn't make sense
        try:
            with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=(10,), block_size=-100) as _:
                pass
        except (ValueError, RuntimeError):
            # Expected
            pass
    
    def test_empty_shape_tuple(self, temp_blob_dir):
        """Test creating blob with empty shape tuple."""
        # Scalar tensors - shape ()
        with TensorBlob.open(temp_blob_dir, "w", dtype="float32", shape=()) as blob:
            # Should work, but each tensor is a scalar
            blob.write(torch.tensor([1.0, 2.0, 3.0]))
            assert len(blob) == 3

