from __future__ import annotations

import gc
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensordict import MemoryMappedTensor


class LRUCache:
    r"""An LRU cache for memory-mapped block management.

    This cache uses an LRU (Least Recently Used) eviction policy to limit
    the number of memory-mapped blocks kept in memory. When the cache reaches
    maxsize, the least recently accessed block is evicted and unmapped.

    This is crucial for limiting kernel VMA (Virtual Memory Area) overhead
    when working with blobs containing many blocks. Each MemoryMappedTensor
    creates a VMA structure in the kernel, and systems have limits (typically
    ~65,530 VMAs per process).

    Parameters
    ----------
    maxsize : int
        Maximum number of blocks to keep cached. When exceeded, least recently
        used blocks are evicted and unmapped.

    Notes
    -----
    The cache automatically triggers munmap() on evicted MemoryMappedTensor
    objects by deleting them and forcing garbage collection. This ensures
    kernel VMA structures are freed promptly.

    Examples
    --------
    >>> cache = LRUCache(maxsize=1000)
    >>> cache['block_1'] = MemoryMappedTensor.from_filename(...)
    >>> tensor = cache['block_1']  # Access (moves to end of LRU)
    >>> len(cache)
    1
    """

    def __init__(self, maxsize: int) -> None:
        self._cap = maxsize
        self._map = OrderedDict()

    def __getitem__(self, key: str) -> MemoryMappedTensor:
        self._map.move_to_end(key)
        return self._map[key]

    def __setitem__(self, key: str, value: MemoryMappedTensor) -> None:
        try:
            self._map.move_to_end(key)
        except KeyError:
            if len(self._map) >= self._cap:
                del self[next(iter(self.keys()))]
        self._map[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._map

    def __delitem__(self, key: str) -> None:
        value = self._map.pop(key)
        del value
        gc.collect()

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def keys(self):
        return self._map.keys()

    def values(self):
        return self._map.values()

    def items(self):
        return self._map.items()

    def clear(self) -> None:
        self._map.clear()
        gc.collect()
