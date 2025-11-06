from __future__ import annotations

import io
import os
import shutil
import uuid
from dataclasses import dataclass, field
from itertools import groupby
from math import ceil
from pathlib import Path
from typing import Iterator

import orjson
import torch
from configmixin import ConfigMixin, register_to_config
from tensordict import MemoryMappedTensor


@dataclass(slots=True, kw_only=True)
class TensorBlobStatus:
    len: int = 0
    bds: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, frm):
        with open(frm, "rb") as fs:
            return cls(**orjson.loads(fs.read()))

    def dump(self, to):
        with open(to, "wb") as fs:
            fs.write(orjson.dumps(self))


class TensorBlob(ConfigMixin):
    _m_rd = False
    _m_wr = False
    _m_ap = False

    status_name = ".stat"
    config_name = ".conf"
    ignore_for_config = ["filename", "mode"]

    @classmethod
    def open(cls, filename, mode="r", *, dtype=None, shape=None, block_size=8192):
        r"""Open a TensorBlob with file-like interface for tensor storage.

        TensorBlob provides persistent, memory-mapped storage for large collections
        of same-shaped tensors. It uses a block-based architecture where tensors are
        organized into fixed-size blocks for efficient I/O and memory management.

        The blob is stored as a directory containing:
        - ``.conf``: Configuration file (dtype, shape, block_size)
        - ``.stat``: State file (length, block list)
        - Block files: UUID-named memory-mapped tensor files

        Parameters
        ----------
        filename : str or Path
            Directory path for blob storage. Supports tilde expansion (~) and
            relative paths.
        mode : str, default="r"
            File access mode ('r', 'w', 'a', 'r+', 'w+', 'a+'). See below for details.
        dtype : str or torch.dtype, optional
            Data type for tensors. Required for new blobs (modes 'w', 'w+').
        shape : tuple of int or int, optional
            Shape of individual tensors. Required for new blobs (modes 'w', 'w+').
        block_size : int, default=8192
            Number of tensors per memory-mapped block file.

        Returns
        -------
        TensorBlob
            Opened blob object. Use with context manager for automatic cleanup.

        Raises
        ------
        FileNotFoundError
            If mode is 'r', 'r+', 'a', or 'a+' and blob doesn't exist.
        ValueError
            If creating new blob without dtype or shape, or if mode is invalid.
        TypeError
            If dtype is neither string nor torch.dtype.

        Examples
        --------
        Creating a new blob and writing data:

        >>> import torch
        >>> from tensorblob import TensorBlob
        >>>
        >>> with TensorBlob.open("data/embeddings", "w",
        ...                       dtype="float32", shape=(768,)) as blob:
        ...     embeddings = torch.randn(1000, 768)
        ...     blob.write(embeddings)
        ...     print(f"Wrote {len(blob)} tensors")
        Wrote 1000 tensors

        Reading from existing blob:

        >>> with TensorBlob.open("data/embeddings", "r") as blob:
        ...     all_data = blob.read()
        ...     print(all_data.shape)
        torch.Size([1000, 768])

        Appending to existing blob:

        >>> with TensorBlob.open("data/embeddings", "a") as blob:
        ...     new_data = torch.randn(100, 768)
        ...     blob.write(new_data)
        ...     print(f"Total: {len(blob)}")
        Total: 1100

        Read and update with r+ mode:

        >>> with TensorBlob.open("data/embeddings", "r+") as blob:
        ...     first_10 = blob.read(size=10)
        ...     blob.seek(5)
        ...     blob.write(torch.ones(3, 768))  # Overwrite at position 5

        Custom block size for large tensors:

        >>> with TensorBlob.open("data/images", "w",
        ...                       dtype=torch.float32,
        ...                       shape=(3, 1024, 1024),
        ...                       block_size=256) as blob:
        ...     images = torch.randn(1000, 3, 1024, 1024)
        ...     blob.write(images)

        File Access Modes
        -----------------
        Similar to Python's built-in open(), supports the following modes:

        Basic modes:
        - 'r'  : Read-only. Blob must exist. Position starts at beginning.
        - 'w'  : Write-only. Creates new or truncates existing. Position at start. **If the blob already exists,
                   truncation will ignore any other parameters supplied and rely on existing configuration.**
        - 'a'  : Append-only. Blob must exist. Position starts at end.
                All writes go to end regardless of seek position.

        Update modes (with '+'):
        - 'r+' : Read and write. Blob must exist. Position at start.
                   Can overwrite existing data or extend at end.
        - 'w+' : Read and write. Creates new or truncates existing. Position at start.
        - 'a+' : Read and append. Blob must exist. Position at end.
                   Reads allowed anywhere, writes always append to end.

        Data Type and Shape
        -------------------
        All tensors in a blob must have the same dtype and shape. These are
        specified when creating a new blob (modes 'w', 'w+') and stored in
        the configuration file. When opening existing blobs, dtype and shape
        are loaded automatically.

        Supported dtypes: "float32", "float64", "int32", "int64", "bool", etc.
        Can also use torch.dtype objects like torch.float32.

        Shape can be:
        - Single integer: shape=10 creates 1D tensors of shape (10,)
        - Tuple: shape=(3, 224, 224) creates 3D tensors
        """
        modes = set(mode)
        if modes - set("raw+") or len(mode) > len(modes):
            raise ValueError("Invalid mode: %s" % mode)
        if sum(c in "raw" for c in mode) != 1 or mode.count("+") > 1:
            raise ValueError(
                "Must have exactly one of read/write/append mode and at most one plus: %s"
                % mode
            )

        filename = Path(filename).expanduser().resolve()
        if not filename.exists():
            if "r" in modes or "a" in modes:
                raise FileNotFoundError("Blob not found: %r" % filename)
            if dtype is None or shape is None:
                raise ValueError(
                    "Arguments ``dtype`` and ``shape`` are required for new blob; got: %r and %r"
                    % (dtype, shape)
                )
            if isinstance(dtype, torch.dtype):
                dtype = str(dtype).split(".").pop()
            elif not isinstance(dtype, str):
                raise TypeError(
                    "dtype must be str or torch.dtype, got %r" % type(dtype).__name__
                )
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            return cls(os.fspath(filename), dtype, shape, block_size, mode)

        return cls.from_config(
            save_directory=filename,
            runtime_kwargs={"mode": mode, "filename": os.fspath(filename)},
        )

    @classmethod
    def apply_param_hooks(cls, d):
        d["shape"] = tuple(d["shape"])
        return d

    @register_to_config
    def __init__(
        self,
        filename: str,
        dtype: str,
        shape: tuple[int, ...],
        block_size: int,
        mode: str,
    ) -> None:
        self.filename = filename
        self.dtype = dtype
        self.shape = shape
        self.block_size = block_size
        self.mode = mode

        self._pos = 0
        self._closed = False

        if "+" in mode:
            self._m_rd = True
            self._m_wr = True
        match mode.replace("+", ""):
            case "r":
                self._m_rd = True
            case "w":
                self._m_wr = True
                self._trunc()
            case "a":
                self._m_wr = True
                self._m_ap = True
                self._create()

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

    def __enter__(self) -> TensorBlob:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __len__(self) -> int:
        return self._status.len

    def __getitem__(self, idx: int | slice) -> torch.Tensor:
        if not isinstance(idx, (int, slice)):
            raise TypeError("Index must be int or slice, got %r!" % type(idx).__name__)
        if isinstance(idx, int):
            if idx >= len(self) or idx < -len(self):
                raise IndexError(
                    "Index out of bounds: %r (length: %d)" % (idx, len(self))
                )
            i, o = divmod(idx + len(self) if idx < 0 else idx, self.block_size)
            return self._getblock(i)[o].clone()

        # Although the current implementation may not be efficient, it is very easy to
        # understand and debug. More efficient implementation requires much more complex
        # edge case handling and is error prone. Also, I think the primary cost here is
        # still the I/O operations, not the Python code.
        ret = [
            self._getblock(bd)[[i % self.block_size for i in _is]]
            for bd, _is in groupby(
                range(*idx.indices(len(self))), key=lambda i: i // self.block_size
            )
        ]
        if not ret:
            return torch.empty(0, *self.shape, dtype=getattr(torch, self.dtype))
        return torch.cat(ret, dim=0)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(self._pos, len(self)):
            self._pos += 1
            yield self[i]

    def _trunc(self) -> None:
        if os.path.exists(self.filename):
            try:
                st = TensorBlobStatus.load(self.statuspath)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Status file missing for blob at %r; file corrupted!"
                    % self.statuspath
                ) from exc
            for bd in st.bds:
                os.remove(os.path.join(self.filename, bd))
        self.save_config(save_directory=self.filename, overwrite=True)
        TensorBlobStatus().dump(self.statuspath)

    def _create(self) -> None:
        if not os.path.exists(self.filename):
            self.save_config(save_directory=self.filename)
            TensorBlobStatus().dump(self.statuspath)

    def _getblock(self, bd: str | int = -1) -> MemoryMappedTensor:
        if not self._status.bds:
            self._addblock()
        if isinstance(bd, int):
            bd = self._status.bds[bd]
        return self._memmap[bd]

    def _isfull(self) -> bool:
        return (not len(self) % self.block_size) and bool(len(self))

    def _addblock(self) -> MemoryMappedTensor:
        if self._status.bds and not self._isfull():
            raise RuntimeError(
                "Attempt to create a new block when working block "
                "is not full: length <%d> < capacity <%d>."
                % (len(self) % self.block_size, self.block_size)
            )
        name = str(uuid.uuid4())
        mmap = MemoryMappedTensor.empty(
            self.block_size,
            *self.shape,
            dtype=getattr(torch, self.dtype),
            filename=os.path.join(self.filename, name),
        )
        self._status.bds.append(name)
        self._memmap[name] = mmap
        return mmap

    def _loadstatus(self) -> None:
        try:
            self._status = TensorBlobStatus.load(self.statuspath)
            self._memmap = {
                name: MemoryMappedTensor.from_filename(
                    os.path.join(self.filename, name),
                    dtype=getattr(torch, self.dtype),
                    shape=(self.block_size, *self.shape),
                )
                for name in self._status.bds
            }
            if self._m_ap:
                self._pos = len(self)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "status file missing for blob at %r; file corrupted!" % self.statuspath
            ) from exc

    def _checkclosed(self) -> None:
        if self._closed:
            raise IOError("I/O operation on closed blob.")

    def _checkwritable(self) -> None:
        if not self._m_wr:
            raise IOError("Blob is not open for writing (mode='%s')" % self.mode)
        self._checkclosed()

    def _checkreadable(self) -> None:
        if not self._m_rd:
            raise IOError("Blob is not open for reading (mode='%s')" % self.mode)
        self._checkclosed()

    def tell(self) -> int:
        self._checkclosed()
        return self._pos

    def seek(self, pos: int = 0, whence: int = io.SEEK_SET) -> int:
        self._checkclosed()
        match whence:
            case io.SEEK_SET:
                _pos = pos
            case io.SEEK_CUR:
                _pos = self._pos + pos
            case io.SEEK_END:
                _pos = len(self) + pos
            case _:
                raise ValueError("Invalid whence: %r" % whence)
        self._pos = max(min(_pos, len(self)), 0)
        return self.tell()

    def close(self) -> None:
        if not self._closed and self._m_wr:
            self.flush()
        self._closed = True

    def flush(self) -> None:
        self._checkwritable()
        self._status.dump(self.statuspath)

    def read(self, size: int | None = None) -> torch.Tensor:
        self._checkreadable()
        end = min(self._pos + (size or len(self)), len(self))
        ret = self[self._pos : end]
        self.seek(end)
        return ret

    def write(self, ts: torch.Tensor) -> int:
        self._checkwritable()
        if self._m_ap:
            self.seek(whence=io.SEEK_END)
        ts = ts.view(-1, *self.shape)
        for t in ts:
            if self._isfull() and self._pos >= len(self):
                self._addblock()
            i, o = divmod(self._pos, self.block_size)
            self._getblock(i)[o] = t
            self._status.len += self._pos >= len(self)
            self._pos += 1
        return len(ts)

    def truncate(self, pos: int | None = None) -> int:
        self._checkwritable()
        self.seek(pos or self.tell())
        brk = ceil(self.tell() / self.block_size)
        for bd in self._status.bds[brk:]:
            os.remove(self._memmap.pop(bd).filename)
        self._status.bds = self._status.bds[:brk]
        self._status.len = self.tell()
        self.flush()
        return self.tell()

    def extend(self, other: TensorBlob, maintain_order: bool = False) -> None:
        if self.dtype != other.dtype or self.shape != other.shape:
            raise ValueError("Blob data types and shapes must match to extend blobs!")

        self._checkwritable()
        self.seek(whence=io.SEEK_END)
        if maintain_order:
            for i in range(len(other)):
                self.write(other[i])
            return

        # If order is not important, we can simply copy over the complete blocks from
        # the other blob and merge incomplete blocks.
        if self.block_size != other.block_size:
            raise ValueError(
                "Block sizes must match to extend blobs in non-order-preserving mode!"
            )

        comb = []
        sbrk = len(self) // self.block_size * self.block_size
        if sbrk < len(self):
            comb.append(self[sbrk:])
        obrk = len(other) // other.block_size * other.block_size
        if obrk < len(other):
            comb.append(other[obrk:])

        # TODO: We are directly accessing internal data structures of the other blob here.
        self.truncate(sbrk)
        for obd in other._status.bds[: len(other) // other.block_size]:
            sbd = str(uuid.uuid4())
            shutil.copy(
                os.path.join(other.filename, obd), os.path.join(self.filename, sbd)
            )
            self._status.bds.append(sbd)
            self._status.len += self.block_size
            self._memmap[sbd] = MemoryMappedTensor.from_filename(
                os.path.join(self.filename, sbd),
                dtype=getattr(torch, self.dtype),
                shape=(self.block_size, *self.shape),
            )

        self.seek(whence=io.SEEK_END)
        if comb:
            self.write(torch.cat(comb, dim=0))
        self.flush()
