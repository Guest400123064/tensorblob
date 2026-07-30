from __future__ import annotations

import io
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import orjson
import torch
from configmixin import ConfigMixin, register_to_config

from tensorblob._blob import TensorBlob

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self


@dataclass(slots=True, kw_only=True)
class TensorDBStatus:
    len: int = 0

    @classmethod
    def load(cls, frm):
        with open(frm, "rb") as fs:
            return cls(**orjson.loads(fs.read()))

    def dump(self, to):
        with open(to, "wb") as fs:
            fs.write(orjson.dumps(self))


class TensorDB(ConfigMixin):
    _m_rd = False
    _m_wr = False
    _m_ap = False

    status_name = ".stat"
    config_name = ".conf"
    ignore_for_config: ClassVar[list[str]] = ["filename", "mode", "max_cached_blocks"]

    @classmethod
    def open(
        cls,
        filename,
        mode="r",
        *,
        schema=None,
        block_size=8192,
        max_cached_blocks=None,
    ):
        r"""Open a TensorDB with file-like interface for multi-field tensor storage.

        TensorDB provides persistent, row-aligned storage for heterogeneous
        (multi-field) tensor collections. Each field is stored as a plain
        :class:`TensorBlob` in a subdirectory, and TensorDB keeps the row
        orders of all fields aligned. Rows are dense: every write must supply
        every field with the same row count.

        The database is stored as a directory containing:
        - ``.conf``: Schema file (field names, dtypes, shapes)
        - ``.stat``: State file (committed row count)
        - Field-named subdirectories: One TensorBlob per field

        Parameters
        ----------
        filename : str or Path
            Directory path for database storage. Supports tilde expansion (~)
            and relative paths.
        mode : str, default="r"
            File access mode ('r', 'w', 'a', 'r+', 'w+', 'a+'). Behaves like
            :meth:`TensorBlob.open`; the mode is applied to every field.
        schema : dict, optional
            Mapping of field names to ``(dtype, shape)`` pairs, e.g.,
            ``{"price": ("float32", 1), "embed": (torch.float16, (768,))}``.
            Required for new databases (modes 'w', 'w+'). Fixed at creation;
            loaded automatically when opening existing databases.
        block_size : int, default=8192
            Number of rows per memory-mapped block file, applied to all fields.
        max_cached_blocks : int, optional
            Maximum number of memory-mapped blocks to keep cached per field.
            If None (default), uses 1/16 of system's max_map_count limit.

        Returns
        -------
        TensorDB
            Opened database object. Use with context manager for automatic
            cleanup.

        Raises
        ------
        FileNotFoundError
            If mode is 'r', 'r+', 'a', or 'a+' and database doesn't exist.
        ValueError
            If creating new database without schema, if the schema is
            malformed, or if mode is invalid.
        TypeError
            If a dtype is neither string nor torch.dtype.

        Examples
        --------
        Creating a new database and writing dense rows:

        >>> import torch
        >>> from tensorblob import TensorDB
        >>>
        >>> with TensorDB.open("events.db", "w",
        ...                    schema={"price": ("float32", 1),
        ...                            "embed": ("float32", 768)}) as db:
        ...     db.write({"price": torch.randn(1000, 1),
        ...               "embed": torch.randn(1000, 768)})
        ...     print(f"Wrote {len(db)} rows")
        Wrote 1000 rows

        Reading rows back, aligned across fields:

        >>> with TensorDB.open("events.db", "r") as db:
        ...     row = db[42]        # {"price": (1,), "embed": (768,)}
        ...     batch = db[10:100]  # {"price": (90, 1), "embed": (90, 768)}

        Notes
        -----
        Writes commit the row count only after all fields are written. If a
        crash leaves fields longer than the committed count, the next open
        reports the committed (minimum) length and, for writable modes,
        truncates the stray rows back to restore alignment.
        """
        modes = set(mode)
        if modes - set("raw+") or len(mode) > len(modes):
            raise ValueError(f"Invalid mode: {mode}")
        if sum(c in "raw" for c in mode) != 1 or mode.count("+") > 1:
            raise ValueError(
                f"Must have exactly one of read/write/append mode and at most one plus: {mode}"
            )

        filename = Path(filename).expanduser().resolve()
        if not filename.exists():
            if "r" in modes or "a" in modes:
                raise FileNotFoundError(f"Database not found: {filename!r}")
            if schema is None:
                raise ValueError("Argument ``schema`` is required for new database!")
            schema = cls._normalize_schema(schema)
            return cls(os.fspath(filename), schema, block_size, mode, max_cached_blocks)

        return cls.from_config(
            save_directory=filename,
            runtime_kwargs={
                "mode": mode,
                "filename": os.fspath(filename),
                "max_cached_blocks": max_cached_blocks,
            },
        )

    @classmethod
    def unlink(cls, filename):
        filename = Path(filename).expanduser().resolve()
        if filename.exists():
            try:
                shutil.rmtree(filename)
            except OSError as exc:
                warnings.warn(f"Failed to unlink database at {filename!r}: {exc}")
                return False
        return True

    @classmethod
    def _normalize_schema(cls, schema):
        if not isinstance(schema, dict) or not schema:
            raise ValueError(
                "Schema must be a non-empty dict mapping field names to (dtype, shape)!"
            )
        norm = {}
        for name, spec in schema.items():
            if (
                not isinstance(name, str)
                or not name
                or name.startswith(".")
                or "/" in name
                or os.sep in name
                or (os.altsep and os.altsep in name)
            ):
                raise ValueError(f"Invalid field name: {name!r}")
            dtype, shape = spec
            if isinstance(dtype, torch.dtype):
                dtype = str(dtype).split(".").pop()
            elif not isinstance(dtype, str):
                raise TypeError(
                    f"dtype must be str or torch.dtype, got {type(dtype).__name__!r}"
                )
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            norm[name] = (dtype, shape)
        return norm

    @classmethod
    def apply_param_hooks(cls, jdict):
        jdict["schema"] = {
            name: (dtype, tuple(shape))
            for name, (dtype, shape) in jdict["schema"].items()
        }
        return jdict

    @register_to_config
    def __init__(
        self,
        filename: str,
        schema: dict[str, tuple[str, tuple[int, ...]]],
        block_size: int,
        mode: str,
        max_cached_blocks: int | None = None,
    ) -> None:
        self.filename = filename
        self.schema = schema
        self.block_size = block_size
        self.mode = mode
        self.max_cached_blocks = max_cached_blocks

        self._closed = False

        if "+" in mode:
            self._m_rd = True
            self._m_wr = True
        match mode.replace("+", ""):
            case "r":
                self._m_rd = True
            case "w":
                self._m_wr = True
            case "a":
                self._m_wr = True
                self._m_ap = True

        isnew = not os.path.exists(self.filename)
        if isnew:
            os.makedirs(self.filename)
            self.save_config(save_directory=self.filename)
        self._cols = {
            name: TensorBlob.open(
                os.path.join(self.filename, name),
                mode,
                dtype=dtype,
                shape=shape,
                block_size=block_size,
                max_cached_blocks=max_cached_blocks,
            )
            for name, (dtype, shape) in self.schema.items()
        }

        # For new or truncated databases the committed count starts at zero; no
        # repair is needed since the columns were just (re)initialized above.
        if isnew or "w" in mode:
            self._status = TensorDBStatus()
            self._status.dump(self.statuspath)
        else:
            self._loadstatus()

    @property
    def configpath(self) -> str:
        return os.path.join(self.filename, self.config_name)

    @property
    def statuspath(self) -> str:
        return os.path.join(self.filename, self.status_name)

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __len__(self) -> int:
        return self._status.len

    def __getitem__(
        self, idx: int | slice | list | tuple | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if not isinstance(idx, (int, slice, list, tuple, torch.Tensor)) and not hasattr(
            idx, "__array__"
        ):
            raise TypeError(
                "Index must be int, slice, or a sequence of int, "
                f"got {type(idx).__name__!r}!"
            )
        return {name: col[idx] for name, col in self._cols.items()}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for i in range(self.tell(), len(self)):
            self.seek(i + 1)
            yield self[i]

    def _loadstatus(self) -> None:
        try:
            self._status = TensorDBStatus.load(self.statuspath)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Status file missing for database at {self.statuspath!r}; file corrupted!"
            ) from exc

        # The committed row count is the source of truth. A crash mid-write can
        # leave some columns longer than the committed count; report the minimum
        # and, if writable, truncate stray rows back to restore alignment.
        target = min([self._status.len] + [len(col) for col in self._cols.values()])
        if target != self._status.len or any(
            len(col) != target for col in self._cols.values()
        ):
            warnings.warn(
                f"Inconsistent column lengths detected for database at {self.filename!r}; "
                f"reporting {target} committed rows."
            )
            self._status.len = target
            if self._m_wr:
                for col in self._cols.values():
                    if len(col) != target:
                        col.truncate(target)
                self._status.dump(self.statuspath)

    def _checkclosed(self) -> None:
        if self._closed:
            raise OSError("I/O operation on closed database.")

    def _checkwritable(self) -> None:
        if not self._m_wr:
            raise OSError(f"Database is not open for writing (mode='{self.mode}')")
        self._checkclosed()

    def _checkreadable(self) -> None:
        if not self._m_rd:
            raise OSError(f"Database is not open for reading (mode='{self.mode}')")
        self._checkclosed()

    def tell(self) -> int:
        self._checkclosed()
        return next(iter(self._cols.values())).tell()

    def seek(self, pos: int = 0, whence: int = io.SEEK_SET) -> int:
        self._checkclosed()
        for col in self._cols.values():
            col.seek(pos, whence)
        return self.tell()

    def close(self) -> None:
        if self._closed:
            return
        for col in self._cols.values():
            col.close()
        if self._m_wr:
            self._status.dump(self.statuspath)
        self._closed = True

    def flush(self) -> None:
        self._checkwritable()
        for col in self._cols.values():
            col.flush()
        self._status.dump(self.statuspath)

    def read(self, size: int | None = None) -> dict[str, torch.Tensor]:
        self._checkreadable()
        # Clamp at the committed row count so uncommitted trailing rows left by
        # an interrupted write are never visible.
        remaining = len(self) - self.tell()
        size = remaining if size is None else min(size, remaining)
        if size <= 0:
            return {
                name: torch.empty(0, *shape, dtype=getattr(torch, dtype))
                for name, (dtype, shape) in self.schema.items()
            }
        return {name: col.read(size) for name, col in self._cols.items()}

    def write(self, rows: dict[str, torch.Tensor]) -> int:
        self._checkwritable()
        if not isinstance(rows, dict):
            raise TypeError(
                f"Rows must be a dict mapping field names to tensors, got {type(rows).__name__!r}!"
            )
        missing = sorted(self.schema.keys() - rows.keys())
        extra = sorted(rows.keys() - self.schema.keys())
        if missing or extra:
            raise ValueError(
                "Dense writes require exactly the schema fields; "
                f"missing: {missing!r}, unexpected: {extra!r}"
            )

        nts = {
            name: ts.view(-1, *self.schema[name][1]).size(0)
            for name, ts in rows.items()
        }
        if len(set(nts.values())) != 1:
            raise ValueError(f"All fields must have the same row count; got: {nts!r}")
        nt = next(iter(nts.values()))

        # Columns are written first and the committed row count is bumped only
        # afterwards, so an interrupted write is rolled back on the next open.
        for name, col in self._cols.items():
            col.write(rows[name])
        self._status.len = len(next(iter(self._cols.values())))
        return nt

    def truncate(self, pos: int | None = None) -> int:
        self._checkwritable()
        for col in self._cols.values():
            col.truncate(pos)
        self._status.len = self.tell()
        self._status.dump(self.statuspath)
        return self.tell()

    def extend(self, other: TensorDB, maintain_order: bool = False) -> None:
        if set(self.schema) != set(other.schema):
            raise ValueError("Schema fields must match to extend databases!")
        self._checkwritable()
        for name, col in self._cols.items():
            col.extend(other._cols[name], maintain_order=maintain_order)
        self._status.len = len(next(iter(self._cols.values())))
        self._status.dump(self.statuspath)
