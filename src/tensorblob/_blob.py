from __future__ import annotations

import io
import os
import shutil
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
    states_name = ".stat"
    config_name = ".conf"
    ignore_for_config = ["filename", "mode"]

    @classmethod
    def open(cls, filename, mode="r", *, dtype=None, shape=None, block_size=8192):
        filename = Path(filename).expanduser().resolve()
        if (filename / cls.config_name).exists() or mode in "ar":
            return cls.from_config(
                save_directory=filename,
                runtime_kwargs={"mode": mode, "filename": str(filename)},
            )
        if not filename.exists():
            filename.mkdir(parents=True)
            if dtype is None or shape is None:
                raise ValueError(
                    "Cannot create new blob with missing 'dtype' <%s> or 'shape' <%s>!"
                    % (str(dtype), str(shape))
                )
        filename = str(filename)

        if not isinstance(dtype, torch.dtype):
            if not isinstance(dtype, str):
                raise ValueError(
                    "Expecting str or torch.dtype for 'dtype', got <%s>"
                    % str(type(dtype))
                )

        shape = (shape,) if isinstance(shape, int) else shape
        return cls(filename, dtype, shape, block_size, mode)  # pyright: ignore

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
        self._memmap = {}

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
        if not os.path.exists(self.configpath):
            self.save_config(self.filename)
        if not os.path.exists(self.statespath):
            if self.mode in "ar":
                raise FileNotFoundError(
                    "Cannot read from blob states under <%s>; file corrupted!"
                    % self.filename
                )
            self._states = TensorBlobStates()
            self.flush()

        self._states = TensorBlobStates.load(self.statespath)
        self._memmap = {
            name: MemoryMappedTensor.from_filename(
                os.path.join(self.filename, name),
                dtype=getattr(torch, self.dtype),
                shape=(self.block_size, *self.shape),
            )
            for name in self._states.bds
        }
        if not self._states.bds:
            self._addblock()
        self.flush()

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int):
        if self.closed:
            raise RuntimeError("Seek on a closed file.")
        self._pos = max(min(pos, len(self)), 0)

    def close(self) -> None:
        if self._closed:
            return
        if self.mode in "wa":
            self.flush()
        self._closed = True

    def flush(self) -> None:
        if self._closed:
            raise IOError("Cannot flush closed blob.")
        self._states.dump(self.statespath)

    def read(
        self,
    ):
        pass

    def write(self, items: torch.Tensor) -> int:
        items = items.view(-1, *self.shape)
        for item in items:
            if self._isfull() and self._pos >= len(self):
                self._addblock()
            blk = self._getblock(self._pos // self.block_size)
            blk[self._pos % self.block_size] = item
            self._states.len += self._pos >= len(self)
            self._pos += 1
        return items.size(0)

    def append(self, item: torch.Tensor) -> int:
        self.seek(len(self))
        return self.write(item)

    def extend(
        self, other: TensorBlob, copy: bool = True, maintain_order: bool = False
    ) -> None:
        if not maintain_order:
            for i in range(len(other)):
                self.append(other[i])
            if not copy:
                raise NotImplementedError
            return
        raise NotImplementedError
