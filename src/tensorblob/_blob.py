from __future__ import annotations

import os
import io
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import orjson
import torch
from configmixin import ConfigMixin, register_to_config
from tensordict import MemoryMappedTensor


@dataclass(slots=True, kw_only=True)
class TensorBlobStates:
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

    states_name = ".stat"
    config_name = ".conf"
    ignore_for_config = ["filename", "mode"]

    @classmethod
    def open(cls, filename, mode="r", *, dtype=None, shape=None, block_size=8192):
        r"""Open a TensorBlob with file-like interface.

        Parameters
        ----------
        filename : str
            Path to the blob directory.
        mode : str, default="r"
            Mode to open the blob.
        dtype : str or torch.dtype, default=None
            Data type of each tensor in the blob.
        shape : tuple, default=None
            Shape of each tensor in the blob.
        block_size : int, default=8192
            Number of tensors per block.

        Returns
        -------
        TensorBlob
            The opened TensorBlob.
        """
        modes = set(mode)
        if modes - set("raw+") or len(mode) > len(modes):
            raise ValueError("Invalide mode: %s" % mode)
        if sum(c in "raw" for c in mode) != 1 or mode.count("+") > 1:
            raise ValueError("Must have exactly one of read/write/append mode and at most one plus: %s" % mode)

        filename = Path(filename).expanduser().resolve()
        if not filename.exists():
            if "r" in modes:
                raise FileNotFoundError("File not found: %r" % filename)
            if dtype is None or shape is None:
                raise ValueError("Arguments ``dtype`` and ``shape`` are required for new blob; got: %r and %r" % (dtype, shape))
            if isinstance(dtype, torch.dtype):
                dtype = str(dtype).split(".").pop()
            elif not isinstance(dtype, str):
                raise TypeError("dtype must be str or torch.dtype, got %r" % type(dtype).__name__)
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            return cls(os.fspath(filename), dtype, shape, block_size, mode)

        return cls.from_config(save_directory=filename, runtime_kwargs={"mode": mode, "filename": os.fspath(filename)})

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

        self._loadstates()

    @property
    def configpath(self) -> str:
        return os.path.join(self.filename, self.config_name)

    @property
    def statespath(self) -> str:
        return os.path.join(self.filename, self.states_name)

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> TensorBlob:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __len__(self) -> int:
        return self._states.len

    def __getitem__(self, index: int) -> torch.Tensor:
        if index >= len(self):
            raise IndexError(
                "Index out of bound: index <%d> >= length <%d>" % (index, len(self))
            )
        count, offset = divmod(index, self.block_size)
        return self._getblock(count)[offset].clone()

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(self._pos, len(self)):
            yield self[i]
            self._pos += 1

    def _trunc(self) -> None:
        if os.path.exists(self.filename):
            try:
                st = TensorBlobStates.load(self.statespath)
            except FileNotFoundError as exc:
                raise FileNotFoundError("States file missing for blob at %r; file corrupted!" % self.statespath) from exc
            for bd in st.bds:
                os.remove(os.path.join(self.filename, bd))
        self.save_config(save_directory=self.filename, overwrite=True)
        TensorBlobStates().dump(self.statespath)

    def _create(self) -> None:
        if not os.path.exists(self.filename):
            self.save_config(save_directory=self.filename)
            TensorBlobStates().dump(self.statespath)

    def _getblock(self, bd: str | int = -1) -> MemoryMappedTensor:
        if isinstance(bd, int):
            bd = self._states.bds[bd]
        return self._memmap[bd]

    def _isfull(self) -> bool:
        return (not len(self) % self.block_size) and bool(len(self))

    def _addblock(self) -> MemoryMappedTensor:
        if self._states.bds and not self._isfull():
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
        self._states.bds.append(name)
        self._memmap[name] = mmap
        return mmap

    def _loadstates(self) -> None:
        try:
            self._states = TensorBlobStates.load(self.statespath)
            self._memmap = {
                name: MemoryMappedTensor.from_filename(
                    os.path.join(self.filename, name),
                    dtype=getattr(torch, self.dtype),
                    shape=(self.block_size, *self.shape),
                )
                for name in self._states.bds
            }
            if self._m_ap:
                self._pos = len(self)
        except FileNotFoundError as exc:
            raise FileNotFoundError("States file missing for blob at %r; file corrupted!" % self.statespath) from exc

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

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        self._checkclosed()
        if pos < 0:
            raise ValueError("Negative seek position: %r" % pos)
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
        if not self._closed:
            try:
                self.flush()
            finally:
                self._closed = True

    def flush(self) -> None:
        self._checkwritable()
        self._states.dump(self.statespath)

    def read(self, size: int | None = None, lazy: bool = False) -> torch.Tensor | Iterator[torch.Tensor]:
        self._checkreadable()
        size = size or len(self) - self._pos    
        if lazy:
            return (self[i] for i in range(self._pos, self._pos + size))
        raise NotImplementedError

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
            self._states.len += self._pos >= len(self)
            self._pos += 1
        return len(ts)

    def extend(
        self, other: TensorBlob, copy: bool = True, maintain_order: bool = False
    ) -> None:
        self._checkwritable()
        self.seek(whence=io.SEEK_END)
        if not maintain_order:
            other.seek(whence=io.SEEK_SET)
            for t in other.read(lazy=True):
                self.write(t)
            if not copy:
                raise NotImplementedError
            return
        raise NotImplementedError
