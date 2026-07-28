"""Tests for TensorDB - columnar interface managing aligned TensorBlobs."""

import io

import pytest
import torch

from tensorblob import TensorBlob, TensorDB


@pytest.fixture
def schema():
    """Fixture providing a simple two-field schema."""
    return {"price": ("float32", (1,)), "embed": ("float32", (4,))}


@pytest.fixture
def sample_rows():
    """Fixture providing sample rows (100 rows, two fields)."""
    torch.manual_seed(42)
    return {"price": torch.randn(100, 1), "embed": torch.randn(100, 4)}


@pytest.fixture
def db_with_data(tmp_path, schema, sample_rows):
    """Fixture providing a database pre-populated with sample rows."""
    db_dir = tmp_path / "db_with_data"
    with TensorDB.open(db_dir, "w", schema=schema) as db:
        db.write(sample_rows)
    return db_dir, sample_rows


class TestModeValidation:
    """Tests for mode string validation."""

    def test_valid_modes(self, tmp_path, schema):
        """Test opening with all valid modes sets flags correctly."""
        db_dir = tmp_path / "db"
        with TensorDB.open(db_dir, "w", schema=schema) as db:
            assert db._m_rd is False
            assert db._m_wr is True
            assert db._m_ap is False

        with TensorDB.open(db_dir, "r") as db:
            assert (db._m_rd, db._m_wr, db._m_ap) == (True, False, False)
        with TensorDB.open(db_dir, "a") as db:
            assert (db._m_rd, db._m_wr, db._m_ap) == (False, True, True)
        with TensorDB.open(db_dir, "r+") as db:
            assert (db._m_rd, db._m_wr, db._m_ap) == (True, True, False)
        with TensorDB.open(db_dir, "a+") as db:
            assert (db._m_rd, db._m_wr, db._m_ap) == (True, True, True)

    def test_valid_write_plus_mode(self, tmp_path, schema):
        """Test opening with 'w+' mode."""
        with TensorDB.open(tmp_path / "db", "w+", schema=schema) as db:
            assert (db._m_rd, db._m_wr, db._m_ap) == (True, True, False)

    def test_invalid_mode(self, tmp_path, schema):
        """Test that invalid modes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            TensorDB.open(tmp_path / "db", "x", schema=schema)
        with pytest.raises(ValueError, match="Invalid mode"):
            TensorDB.open(tmp_path / "db", "rb", schema=schema)
        with pytest.raises(ValueError, match="exactly one"):
            TensorDB.open(tmp_path / "db", "rw", schema=schema)
        with pytest.raises(ValueError, match="exactly one"):
            TensorDB.open(tmp_path / "db", "+", schema=schema)

    def test_read_mode_requires_existing_db(self, tmp_path):
        """Test that 'r', 'r+', 'a', 'a+' require an existing database."""
        for mode in ["r", "r+", "a", "a+"]:
            with pytest.raises(FileNotFoundError, match="Database not found"):
                TensorDB.open(tmp_path / "missing", mode)


class TestSchemaValidation:
    """Tests for schema parameter validation."""

    def test_schema_required_for_new_db(self, tmp_path):
        """Test that creating a new database without schema raises ValueError."""
        with pytest.raises(ValueError, match="``schema``.*required"):
            TensorDB.open(tmp_path / "db", "w")

    def test_schema_must_be_nonempty_dict(self, tmp_path):
        """Test that empty or non-dict schema raises ValueError."""
        with pytest.raises(ValueError, match="non-empty dict"):
            TensorDB.open(tmp_path / "db", "w", schema={})
        with pytest.raises(ValueError, match="non-empty dict"):
            TensorDB.open(tmp_path / "db", "w", schema=[("a", ("float32", 1))])

    def test_invalid_field_names(self, tmp_path):
        """Test that unsafe field names raise ValueError."""
        for name in ["a/b", ".hidden", "", ".."]:
            with pytest.raises(ValueError, match="Invalid field name"):
                TensorDB.open(tmp_path / "db", "w", schema={name: ("float32", 1)})

    def test_dtype_torch_dtype_normalized(self, tmp_path):
        """Test that torch.dtype in schema is converted to string."""
        with TensorDB.open(
            tmp_path / "db", "w", schema={"a": (torch.float64, 3)}
        ) as db:
            assert db.schema["a"] == ("float64", (3,))

    def test_dtype_invalid_type(self, tmp_path):
        """Test that invalid dtype type raises TypeError."""
        with pytest.raises(TypeError, match="dtype must be str or torch.dtype"):
            TensorDB.open(tmp_path / "db", "w", schema={"a": (123, 3)})

    def test_shape_int_converted_to_tuple(self, tmp_path):
        """Test that integer shape is converted to tuple."""
        with TensorDB.open(tmp_path / "db", "w", schema={"a": ("float32", 10)}) as db:
            assert db.schema["a"] == ("float32", (10,))

    def test_schema_not_required_for_existing_db(self, db_with_data):
        """Test that schema is loaded from config when reopening."""
        db_dir, _ = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            assert db.schema == {"price": ("float32", (1,)), "embed": ("float32", (4,))}

    def test_block_size_persisted_per_column(self, tmp_path):
        """Test that block_size is persisted when reopening."""
        db_dir = tmp_path / "db"
        schema = {"a": ("float32", (2,)), "b": ("float32", (3,))}
        with TensorDB.open(db_dir, "w", schema=schema, block_size=64) as db:
            pass
        with TensorDB.open(db_dir, "r") as db:
            for col in db._cols.values():
                assert col.block_size == 64

    def test_max_cached_blocks_passed_through(self, tmp_path, schema):
        """Test that max_cached_blocks is forwarded to each column."""
        with TensorDB.open(
            tmp_path / "db", "w", schema=schema, max_cached_blocks=7
        ) as db:
            for col in db._cols.values():
                assert col.max_cached_blocks == 7

    def test_write_mode_truncates_existing(self, db_with_data):
        """Test that reopening with 'w' mode truncates all columns."""
        db_dir, _ = db_with_data
        with TensorDB.open(db_dir, "w") as db:
            assert len(db) == 0
        with TensorDB.open(db_dir, "r") as db:
            assert len(db) == 0


class TestBasicWrite:
    """Tests for write operations."""

    def test_write_returns_row_count(self, tmp_path, schema, sample_rows):
        """Test writing rows updates length and position."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db:
            n = db.write(sample_rows)
            assert n == 100
            assert len(db) == 100
            assert db.tell() == 100

    def test_write_single_row(self, tmp_path, schema):
        """Test writing a single row."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db:
            n = db.write({"price": torch.ones(1), "embed": torch.ones(4)})
            assert n == 1
            assert len(db) == 1

    def test_write_missing_field(self, tmp_path, schema):
        """Test that dense write with a missing field raises ValueError."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db:
            with pytest.raises(ValueError, match="missing"):
                db.write({"price": torch.randn(10, 1)})
            assert len(db) == 0

    def test_write_unexpected_field(self, tmp_path, schema):
        """Test that dense write with an extra field raises ValueError."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db, pytest.raises(
            ValueError, match="unexpected"
        ):
            db.write(
                {
                    "price": torch.randn(10, 1),
                    "embed": torch.randn(10, 4),
                    "extra": torch.randn(10, 2),
                }
            )

    def test_write_mismatched_row_counts(self, tmp_path, schema):
        """Test that fields with different row counts raise ValueError."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db:
            with pytest.raises(ValueError, match="same row count"):
                db.write({"price": torch.randn(10, 1), "embed": torch.randn(12, 4)})
            assert len(db) == 0

    def test_write_non_dict_rows(self, tmp_path, schema):
        """Test that non-dict rows raise TypeError."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db, pytest.raises(
            TypeError, match="dict"
        ):
            db.write(torch.randn(10, 5))

    def test_write_requires_writable_mode(self, db_with_data):
        """Test that write in read mode raises IOError."""
        db_dir, sample_rows = db_with_data
        with TensorDB.open(db_dir, "r") as db, pytest.raises(
            IOError, match="not open for writing"
        ):
            db.write(sample_rows)

    def test_write_with_reshape(self, tmp_path):
        """Test that flat tensors are reshaped like in TensorBlob."""
        with TensorDB.open(
            tmp_path / "db", "w", schema={"a": ("float32", (2, 3))}
        ) as db:
            db.write({"a": torch.arange(12, dtype=torch.float32)})
            assert len(db) == 2


class TestBasicRead:
    """Tests for read operations."""

    def test_read_all(self, db_with_data):
        """Test reading all rows returns a dict of tensors."""
        db_dir, sample_rows = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            result = db.read()
            assert set(result) == {"price", "embed"}
            assert torch.allclose(result["price"], sample_rows["price"])
            assert torch.allclose(result["embed"], sample_rows["embed"])

    def test_read_partial_and_position(self, db_with_data):
        """Test reading a specific number of rows updates position."""
        db_dir, sample_rows = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            result = db.read(size=10)
            assert result["price"].shape == (10, 1)
            assert db.tell() == 10
            result = db.read(size=20)
            assert db.tell() == 30
            assert torch.allclose(result["embed"], sample_rows["embed"][10:30])

    def test_read_empty_db(self, tmp_path, schema):
        """Test reading from an empty database."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db:
            pass
        with TensorDB.open(tmp_path / "db", "r") as db:
            result = db.read()
            assert result["price"].shape == (0, 1)
            assert result["embed"].shape == (0, 4)

    def test_read_requires_readable_mode(self, tmp_path, schema):
        """Test that read in write-only mode raises IOError."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db, pytest.raises(
            IOError, match="not open for reading"
        ):
            db.read()


class TestIndexingAndSlicing:
    """Tests for __getitem__ with int and slice."""

    def test_int_index_returns_row_dict(self, db_with_data):
        """Test integer indexing returns one tensor per field."""
        db_dir, sample_rows = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            row = db[42]
            assert set(row) == {"price", "embed"}
            assert row["price"].shape == (1,)
            assert row["embed"].shape == (4,)
            assert torch.allclose(row["price"], sample_rows["price"][42])
            assert torch.allclose(row["embed"], sample_rows["embed"][42])

    def test_negative_index(self, db_with_data):
        """Test negative indexing."""
        db_dir, sample_rows = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            assert torch.allclose(db[-1]["embed"], sample_rows["embed"][-1])
            assert torch.allclose(db[-100]["price"], sample_rows["price"][0])

    def test_index_out_of_bounds(self, db_with_data):
        """Test that out-of-bounds indexing raises IndexError."""
        db_dir, _ = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            with pytest.raises(IndexError, match="out of bounds"):
                _ = db[100]
            with pytest.raises(IndexError, match="out of bounds"):
                _ = db[-101]

    def test_invalid_index_type(self, db_with_data):
        """Test that invalid index types raise TypeError."""
        db_dir, _ = db_with_data
        with TensorDB.open(db_dir, "r") as db, pytest.raises(
            TypeError, match="Index must be"
        ):
            _ = db["price"]

    def test_slice_returns_batched_dict(self, db_with_data):
        """Test slicing returns batched tensors per field."""
        db_dir, sample_rows = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            result = db[10:50:3]
            assert torch.allclose(result["price"], sample_rows["price"][10:50:3])
            assert torch.allclose(result["embed"], sample_rows["embed"][10:50:3])

    def test_negative_step_slice(self, db_with_data):
        """Test slicing with negative step."""
        db_dir, sample_rows = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            result = db[::-1]
            assert torch.allclose(result["embed"], sample_rows["embed"].flip(0))

    def test_indexing_returns_clone(self, tmp_path, schema):
        """Test that retrieved rows are copies, not references."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db:
            db.write({"price": torch.ones(1), "embed": torch.ones(4)})
        with TensorDB.open(tmp_path / "db", "r") as db:
            row = db[0]
            row["price"][0] = 999.0
            assert db[0]["price"][0] == 1.0


class TestSeekTellAndIteration:
    """Tests for seek, tell, and iteration."""

    def test_seek_and_whence(self, db_with_data):
        """Test absolute, relative, and from-end seeking."""
        db_dir, _ = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            assert db.seek(25) == 25
            assert db.seek(10, whence=io.SEEK_CUR) == 35
            assert db.seek(-10, whence=io.SEEK_END) == 90

    def test_seek_clamping(self, db_with_data):
        """Test that seek clamps to valid range."""
        db_dir, _ = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            assert db.seek(200) == 100
            assert db.seek(-100, whence=io.SEEK_CUR) == 0

    def test_invalid_whence(self, db_with_data):
        """Test that invalid whence raises ValueError."""
        db_dir, _ = db_with_data
        with TensorDB.open(db_dir, "r") as db, pytest.raises(
            ValueError, match="whence"
        ):
            db.seek(0, whence=99)

    def test_iterate_all(self, db_with_data):
        """Test iterating over the entire database yields row dicts."""
        db_dir, sample_rows = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            collected = list(db)
            assert len(collected) == 100
            for i, row in enumerate(collected):
                assert torch.allclose(row["price"], sample_rows["price"][i])

    def test_iterate_from_position(self, db_with_data):
        """Test that iteration starts from the current position."""
        db_dir, _ = db_with_data
        with TensorDB.open(db_dir, "r") as db:
            db.seek(95)
            assert len(list(db)) == 5
            assert db.tell() == 100


class TestAppendAndUpdate:
    """Tests for append mode and in-place updates."""

    def test_append_mode_starts_at_end(self, db_with_data):
        """Test that 'a' mode positions at end."""
        db_dir, _ = db_with_data
        with TensorDB.open(db_dir, "a") as db:
            assert db.tell() == 100

    def test_append_writes_go_to_end(self, db_with_data):
        """Test that append mode ignores seek position on write."""
        db_dir, sample_rows = db_with_data
        with TensorDB.open(db_dir, "a") as db:
            db.seek(0)
            db.write({"price": torch.ones(5, 1) * 2, "embed": torch.ones(5, 4) * 2})
            assert len(db) == 105
        with TensorDB.open(db_dir, "r") as db:
            assert torch.allclose(db[104]["price"], torch.ones(1) * 2)
            assert torch.allclose(db[0]["price"], sample_rows["price"][0])

    def test_read_plus_overwrite(self, db_with_data):
        """Test overwriting rows in the middle with r+ mode."""
        db_dir, sample_rows = db_with_data
        with TensorDB.open(db_dir, "r+") as db:
            db.seek(5)
            db.write({"price": torch.ones(3, 1) * 2, "embed": torch.ones(3, 4) * 2})
            assert len(db) == 100
            assert torch.allclose(db[5]["price"], torch.ones(1) * 2)
            assert torch.allclose(db[8]["price"], sample_rows["price"][8])


class TestTruncate:
    """Tests for truncate operations."""

    def test_truncate_at_position(self, tmp_path, schema, sample_rows):
        """Test truncating all columns at a specific position."""
        with TensorDB.open(tmp_path / "db", "w+", schema=schema) as db:
            db.write(sample_rows)
            assert db.truncate(50) == 50
            assert len(db) == 50
            db.seek(0)
            result = db.read()
            assert torch.allclose(result["price"], sample_rows["price"][:50])
            assert torch.allclose(result["embed"], sample_rows["embed"][:50])

    def test_truncate_at_current_position(self, tmp_path, schema, sample_rows):
        """Test truncating at the current position."""
        with TensorDB.open(tmp_path / "db", "w+", schema=schema) as db:
            db.write(sample_rows)
            db.seek(30)
            db.truncate()
            assert len(db) == 30

    def test_truncate_persists_across_sessions(self, tmp_path, schema, sample_rows):
        """Test that truncation is persisted."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db:
            db.write(sample_rows)
            db.truncate(40)
        with TensorDB.open(tmp_path / "db", "r") as db:
            assert len(db) == 40
            assert torch.allclose(db.read()["embed"], sample_rows["embed"][:40])

    def test_truncate_requires_writable_mode(self, db_with_data):
        """Test that truncate requires writable mode."""
        db_dir, _ = db_with_data
        with TensorDB.open(db_dir, "r") as db, pytest.raises(
            IOError, match="not open for writing"
        ):
            db.truncate(50)


class TestExtend:
    """Tests for extend operations."""

    def _make_db(self, path, schema, rows, block_size=8192):
        with TensorDB.open(path, "w", schema=schema, block_size=block_size) as db:
            db.write(rows)

    def test_extend_order_preserving(self, tmp_path, schema):
        """Test extending preserves row order across fields."""
        rows1 = {"price": torch.ones(50, 1), "embed": torch.ones(50, 4)}
        rows2 = {"price": torch.ones(30, 1) * 2, "embed": torch.ones(30, 4) * 2}
        self._make_db(tmp_path / "db1", schema, rows1)
        self._make_db(tmp_path / "db2", schema, rows2)

        with TensorDB.open(tmp_path / "db1", "r+") as db1, TensorDB.open(
            tmp_path / "db2", "r"
        ) as db2:
            db1.extend(db2, maintain_order=True)

        with TensorDB.open(tmp_path / "db1", "r") as db:
            assert len(db) == 80
            assert torch.allclose(db[:50]["price"], rows1["price"])
            assert torch.allclose(db[50:]["embed"], rows2["embed"])

    def test_extend_fast_mode(self, tmp_path, schema):
        """Test fast (non-order-preserving) extend copies blocks."""
        rows1 = {"price": torch.ones(100, 1), "embed": torch.ones(100, 4)}
        rows2 = {"price": torch.ones(75, 1) * 2, "embed": torch.ones(75, 4) * 2}
        self._make_db(tmp_path / "db1", schema, rows1, block_size=50)
        self._make_db(tmp_path / "db2", schema, rows2, block_size=50)

        with TensorDB.open(tmp_path / "db1", "r+") as db1, TensorDB.open(
            tmp_path / "db2", "r"
        ) as db2:
            db1.extend(db2, maintain_order=False)

        with TensorDB.open(tmp_path / "db1", "r") as db:
            assert len(db) == 175

    def test_extend_fast_mode_small_destination(self, tmp_path, schema):
        """Regression: fast extend when destination is smaller than one block."""
        rows1 = {"price": torch.ones(7, 1), "embed": torch.ones(7, 4)}
        rows2 = {"price": torch.zeros(2, 1), "embed": torch.zeros(2, 4)}
        self._make_db(tmp_path / "db1", schema, rows1)
        self._make_db(tmp_path / "db2", schema, rows2)

        with TensorDB.open(tmp_path / "db1", "r+") as db1, TensorDB.open(
            tmp_path / "db2", "r"
        ) as db2:
            db1.extend(db2, maintain_order=False)

        with TensorDB.open(tmp_path / "db1", "r") as db:
            # Destination's own rows must not be duplicated
            assert len(db) == 9
            assert torch.allclose(db[:7]["price"], rows1["price"])
            assert torch.allclose(db[7:]["embed"], rows2["embed"])

    def test_extend_field_mismatch(self, tmp_path):
        """Test that extending with different fields raises ValueError."""
        self._make_db(
            tmp_path / "db1", {"a": ("float32", (2,))}, {"a": torch.randn(10, 2)}
        )
        self._make_db(
            tmp_path / "db2", {"b": ("float32", (2,))}, {"b": torch.randn(10, 2)}
        )
        with TensorDB.open(tmp_path / "db1", "r+") as db1, TensorDB.open(
            tmp_path / "db2", "r"
        ) as db2, pytest.raises(ValueError, match="Schema fields must match"):
            db1.extend(db2)

    def test_extend_dtype_mismatch(self, tmp_path):
        """Test that extending with mismatched column dtypes raises ValueError."""
        self._make_db(
            tmp_path / "db1", {"a": ("float32", (2,))}, {"a": torch.randn(10, 2)}
        )
        self._make_db(
            tmp_path / "db2", {"a": ("float64", (2,))}, {"a": torch.randn(10, 2)}
        )
        with TensorDB.open(tmp_path / "db1", "r+") as db1, TensorDB.open(
            tmp_path / "db2", "r"
        ) as db2, pytest.raises(ValueError, match="must match"):
            db1.extend(db2)

    def test_extend_requires_writable_mode(self, tmp_path, schema):
        """Test that extend requires writable mode."""
        rows = {"price": torch.randn(10, 1), "embed": torch.randn(10, 4)}
        self._make_db(tmp_path / "db1", schema, rows)
        self._make_db(tmp_path / "db2", schema, rows)
        with TensorDB.open(tmp_path / "db1", "r") as db1, TensorDB.open(
            tmp_path / "db2", "r"
        ) as db2, pytest.raises(IOError, match="not open for writing"):
            db1.extend(db2)


class TestFlushAndClose:
    """Tests for flush, close, and context manager behavior."""

    def test_flush_persists_data(self, tmp_path, schema, sample_rows):
        """Test that flush makes data visible to other sessions."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db:
            db.write(sample_rows)
            db.flush()
            with TensorDB.open(tmp_path / "db", "r") as db2:
                assert len(db2) == 100

    def test_auto_flush_on_close(self, tmp_path, schema, sample_rows):
        """Test that data is automatically flushed on close."""
        with TensorDB.open(tmp_path / "db", "w", schema=schema) as db:
            db.write(sample_rows)
        with TensorDB.open(tmp_path / "db", "r") as db:
            assert len(db) == 100
            assert torch.allclose(db[0]["price"], sample_rows["price"][0])

    def test_operation_on_closed_db(self, tmp_path, schema):
        """Test that operations on a closed database raise IOError."""
        db = TensorDB.open(tmp_path / "db", "w+", schema=schema)
        db.close()
        with pytest.raises(IOError, match="closed"):
            db.tell()
        with pytest.raises(IOError, match="closed"):
            db.read()


class TestUnlink:
    """Tests for TensorDB.unlink() class method."""

    def test_unlink_existing_db(self, db_with_data):
        """Test unlinking removes the whole database directory."""
        db_dir, _ = db_with_data
        assert db_dir.exists()
        assert TensorDB.unlink(db_dir) is True
        assert not db_dir.exists()

    def test_unlink_nonexistent_db(self, tmp_path):
        """Test unlinking a non-existent database returns True."""
        assert TensorDB.unlink(tmp_path / "missing") is True

    def test_unlink_corrupted_db(self, db_with_data):
        """Test unlinking works even on corrupted databases."""
        db_dir, _ = db_with_data
        (db_dir / ".conf").unlink()
        assert TensorDB.unlink(db_dir) is True
        assert not db_dir.exists()

    def test_unlink_then_recreate(self, tmp_path, schema, sample_rows):
        """Test creating a new database after unlinking."""
        db_dir = tmp_path / "db"
        with TensorDB.open(db_dir, "w", schema=schema) as db:
            db.write(sample_rows)
        assert TensorDB.unlink(db_dir) is True

        new_schema = {"volume": ("int64", (1,))}
        with TensorDB.open(db_dir, "w", schema=new_schema) as db:
            db.write({"volume": torch.arange(15)})
        with TensorDB.open(db_dir, "r") as db:
            assert len(db) == 15
            assert db.schema == {"volume": ("int64", (1,))}


class TestConsistencyRepair:
    """Tests for crash-recovery: committed row count is the source of truth."""

    def _simulate_partial_write(self, db_dir, field, extra_rows):
        """Simulate a crash mid-write: one column grew past the committed count."""
        with TensorBlob.open(db_dir / field, "a") as col:
            col.write(extra_rows)

    def test_repair_on_writable_open(self, tmp_path, schema, sample_rows):
        """Test that opening a writable database repairs inconsistent columns."""
        db_dir = tmp_path / "db"
        with TensorDB.open(db_dir, "w", schema=schema) as db:
            db.write(sample_rows)
        self._simulate_partial_write(db_dir, "price", torch.randn(5, 1))

        with pytest.warns(UserWarning, match="Inconsistent"):
            db = TensorDB.open(db_dir, "r+")
        with db:
            assert len(db) == 100

        # Repair persisted: column truncated back and no more warnings
        with TensorBlob.open(db_dir / "price", "r") as col:
            assert len(col) == 100

    def test_read_only_open_reports_committed_length(
        self, tmp_path, schema, sample_rows
    ):
        """Test read-only open reports min length but cannot repair on disk."""
        db_dir = tmp_path / "db"
        with TensorDB.open(db_dir, "w", schema=schema) as db:
            db.write(sample_rows)
        self._simulate_partial_write(db_dir, "embed", torch.randn(5, 4))

        with pytest.warns(UserWarning, match="Inconsistent"):
            db = TensorDB.open(db_dir, "r")
        with db:
            assert len(db) == 100
            result = db.read()
            assert result["price"].shape[0] == 100
            assert result["embed"].shape[0] == 100

        # Not repaired on disk in read-only mode
        with TensorBlob.open(db_dir / "embed", "r") as col:
            assert len(col) == 105
