from __future__ import annotations

import os
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
        """
        Open a TensorBlob with file-like semantics.
        
        Args:
            filename: Directory path for the blob storage
            mode: Access mode string (similar to built-in open):
                - 'r': read-only, must exist
                - 'w': write-only, truncate if exists
                - 'a': append-only, must exist
                - 'x': exclusive create, fail if exists
                - '+': add to r/w/a for read+write access
            dtype: Data type for tensors (required for new blobs)
            shape: Shape of individual tensors (required for new blobs)
            block_size: Number of tensors per block (default: 8192)
        """
        # Validate and parse mode
        if not mode or not all(c in "rwax+" for c in mode):
            raise ValueError(f"Invalid mode: {mode!r}")
        
        base_mode = mode.replace("+", "")
        if len(base_mode) != 1 or base_mode not in "rwax":
            raise ValueError(
                f"Mode must be one of 'r', 'w', 'a', 'x' (optionally with '+'), got {mode!r}"
            )
        
        filename = Path(filename).expanduser().resolve()
        config_exists = (filename / cls.config_name).exists()
        blob_exists = config_exists and (filename / cls.states_name).exists()
        
        # Handle mode-specific requirements
        if base_mode == "r":
            # Read mode: blob must exist
            if not blob_exists:
                raise FileNotFoundError(
                    f"TensorBlob does not exist at {filename} (mode='r' requires existing blob)"
                )
            return cls.from_config(
                save_directory=filename,
                runtime_kwargs={"mode": mode, "filename": str(filename)},
            )
        
        elif base_mode == "w":
            # Write mode: truncate if exists, create if not
            if blob_exists:
                # Truncate: remove all block files and state file
                states = TensorBlobStates.load(filename / cls.states_name)
                for bd in states.bds:
                    block_path = filename / bd
                    if block_path.exists():
                        block_path.unlink()
                # Remove state file to truly truncate
                (filename / cls.states_name).unlink()
            
            # Create or recreate
            if not filename.exists():
                filename.mkdir(parents=True)
            
            if dtype is None or shape is None:
                raise ValueError(
                    f"Cannot create blob with missing 'dtype' ({dtype!r}) or 'shape' ({shape!r})"
                )
            
            if isinstance(dtype, torch.dtype):
                dtype = str(dtype).replace("torch.", "")
            elif not isinstance(dtype, str):
                raise TypeError(f"dtype must be str or torch.dtype, got {type(dtype).__name__}")
            
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            return cls(str(filename), dtype, shape, block_size, mode)
        
        elif base_mode == "a":
            # Append mode: blob must exist
            if not blob_exists:
                raise FileNotFoundError(
                    f"TensorBlob does not exist at {filename} (mode='a' requires existing blob)"
                )
            return cls.from_config(
                save_directory=filename,
                runtime_kwargs={"mode": mode, "filename": str(filename)},
            )
        
        elif base_mode == "x":
            # Exclusive create: fail if exists
            if blob_exists or config_exists:
                raise FileExistsError(
                    f"TensorBlob already exists at {filename} (mode='x' requires new blob)"
                )
            
            if not filename.exists():
                filename.mkdir(parents=True)
            
            if dtype is None or shape is None:
                raise ValueError(
                    f"Cannot create blob with missing 'dtype' ({dtype!r}) or 'shape' ({shape!r})"
                )
            
            if isinstance(dtype, torch.dtype):
                dtype = str(dtype).replace("torch.", "")
            elif not isinstance(dtype, str):
                raise TypeError(f"dtype must be str or torch.dtype, got {type(dtype).__name__}")
            
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            return cls(str(filename), dtype, shape, block_size, mode)

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
        
        # Parse mode flags
        self._base_mode = mode.replace("+", "")
        self._has_plus = "+" in mode

        self._pos = 0
        self._closed = False
        self._memmap = {}

        self._loadstates()
        
        # Set initial position based on mode
        if self._base_mode == "a":
            self._pos = len(self)

    @property
    def configpath(self) -> str:
        return os.path.join(self.filename, self.config_name)

    @property
    def statespath(self) -> str:
        return os.path.join(self.filename, self.states_name)

    @property
    def closed(self) -> bool:
        return self._closed
    
    @property
    def readable(self) -> bool:
        """Check if the blob is open for reading."""
        return not self._closed and (self._base_mode == "r" or self._has_plus)
    
    @property
    def writable(self) -> bool:
        """Check if the blob is open for writing (excluding append mode)."""
        return not self._closed and self._base_mode in "wx" or (
            self._base_mode in "rw" and self._has_plus
        )
    
    @property
    def appendable(self) -> bool:
        """Check if the blob is open for appending."""
        return not self._closed and (self._base_mode == "a" or self.writable)

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
        """Load or initialize blob states based on mode."""
        # Save config if creating new blob
        if not os.path.exists(self.configpath):
            if self._base_mode in "ra":
                raise FileNotFoundError(
                    f"Config file missing for blob at {self.filename}; file corrupted!"
                )
            self.save_config(self.filename)
        
        # Load or initialize states
        if not os.path.exists(self.statespath):
            if self._base_mode in "ra":
                raise FileNotFoundError(
                    f"States file missing for blob at {self.filename}; file corrupted!"
                )
            self._states = TensorBlobStates()
            if self.writable or self.appendable:
                self.flush()
        else:
            self._states = TensorBlobStates.load(self.statespath)
        
        # Load existing memory-mapped blocks
        self._memmap = {
            name: MemoryMappedTensor.from_filename(
                os.path.join(self.filename, name),
                dtype=getattr(torch, self.dtype),
                shape=(self.block_size, *self.shape),
            )
            for name in self._states.bds
        }
        
        # Create initial block if needed (only for writable modes)
        if not self._states.bds and (self.writable or self.appendable):
            self._addblock()
            self.flush()

    def tell(self) -> int:
        """Return current position in the blob."""
        if self.closed:
            raise ValueError("I/O operation on closed blob.")
        return self._pos

    def seek(self, pos: int, whence: int = 0) -> int:
        """
        Change stream position.
        
        Args:
            pos: Position offset
            whence: Reference point (0=start, 1=current, 2=end)
            
        Returns:
            New absolute position
        """
        if self.closed:
            raise ValueError("I/O operation on closed blob.")
        
        # In append mode with '+', seeking is allowed but writes still append
        # In pure append mode without '+', only seeking within readable range makes sense
        
        if whence == 0:  # Absolute positioning
            new_pos = pos
        elif whence == 1:  # Relative to current
            new_pos = self._pos + pos
        elif whence == 2:  # Relative to end
            new_pos = len(self) + pos
        else:
            raise ValueError(f"whence must be 0, 1, or 2, got {whence}")
        
        self._pos = max(min(new_pos, len(self)), 0)
        return self._pos

    def close(self) -> None:
        """Close the blob and flush any pending writes."""
        if self._closed:
            return
        if self.writable or self.appendable:
            self.flush()
        self._closed = True

    def flush(self) -> None:
        """Flush write buffers to disk."""
        if self._closed:
            raise ValueError("I/O operation on closed blob.")
        if not (self.writable or self.appendable):
            raise IOError(f"Cannot flush blob opened in mode '{self.mode}'")
        self._states.dump(self.statespath)

    def read(self, n: int | None = None) -> torch.Tensor | None:
        """
        Read tensors from the blob.
        
        Args:
            n: Number of tensors to read. If None, read all remaining tensors.
               If 0, read nothing. If negative, read all remaining.
        
        Returns:
            Stacked tensor of shape (n, *self.shape), or None if no data to read.
        """
        if self.closed:
            raise ValueError("I/O operation on closed blob.")
        if not self.readable:
            raise IOError(f"Blob not open for reading (mode='{self.mode}')")
        
        # Determine how many tensors to read
        if n is None or n < 0:
            n = len(self) - self._pos
        else:
            n = min(n, len(self) - self._pos)
        
        if n == 0:
            return None
        
        # Read tensors
        tensors = []
        start_pos = self._pos
        for i in range(n):
            idx = start_pos + i
            count, offset = divmod(idx, self.block_size)
            tensor = self._getblock(count)[offset].clone()
            tensors.append(tensor)
            self._pos += 1
        
        return torch.stack(tensors)

    def write(self, items: torch.Tensor) -> int:
        """
        Write tensors to the blob at current position.
        
        Args:
            items: Tensor(s) to write. Will be reshaped to (-1, *self.shape).
        
        Returns:
            Number of tensors written.
        """
        if self.closed:
            raise ValueError("I/O operation on closed blob.")
        if not self.writable:
            raise IOError(f"Blob not open for writing (mode='{self.mode}')")
        
        items = items.view(-1, *self.shape)
        for item in items:
            # Extend blob if writing beyond current length
            if self._isfull() and self._pos >= len(self):
                self._addblock()
            
            blk = self._getblock(self._pos // self.block_size)
            blk[self._pos % self.block_size] = item
            
            # Update length if writing beyond current end
            if self._pos >= len(self):
                self._states.len += 1
            
            self._pos += 1
        
        return items.size(0)

    def append(self, items: torch.Tensor) -> int:
        """
        Append tensors to the end of the blob.
        
        Args:
            items: Tensor(s) to append. Will be reshaped to (-1, *self.shape).
        
        Returns:
            Number of tensors appended.
        """
        if self.closed:
            raise ValueError("I/O operation on closed blob.")
        if not self.appendable:
            raise IOError(f"Blob not open for appending (mode='{self.mode}')")
        
        items = items.view(-1, *self.shape)
        for item in items:
            # Always append at the end
            pos = len(self)
            
            if self._isfull():
                self._addblock()
            
            blk = self._getblock(pos // self.block_size)
            blk[pos % self.block_size] = item
            self._states.len += 1
        
        # Move position to end after appending
        self._pos = len(self)
        return items.size(0)

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
