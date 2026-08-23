"""
Defines the Chunk container holding 16x16 tile IDs & PyBiwis bitmasks.
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from world.tile_registry import TileRegistry
from world.bitmask_encoder import ChunkBitmaskEncoder


class Chunk:
    """
    Container for a 16x16 spatial chunk tile matrix and PyBiwis bitmask.
    """

    CHUNK_SIZE: int = 16  # tiles

    def __init__(
        self,
        cx: int,
        cy: int,
        grid: NDArray[np.uint8],
        tile_registry: TileRegistry
    ) -> None:
        """
        Initializes chunk spatial coordinates, grid, and PyBiwis bitmask.
        """
        self.cx: int = cx  # chunk_x
        self.cy: int = cy  # chunk_y
        self.world_x: int = cx * self.CHUNK_SIZE  # tile_x
        self.world_y: int = cy * self.CHUNK_SIZE  # tile_y

        self.grid: NDArray[np.uint8] = grid.copy()
        self.bitmask_words: NDArray[np.uint64] = (
            ChunkBitmaskEncoder.encode_grid_to_words(
                self.grid, tile_registry
            )
        )
        self.is_dirty: bool = False  # state

    def get_tile(self, lx: int, ly: int) -> int:
        """
        Retrieves local tile ID at coordinate (lx, ly) in O(1) time.
        """
        if not (0 <= lx < self.CHUNK_SIZE and 0 <= ly < self.CHUNK_SIZE):
            return 0
        return int(self.grid[ly, lx])

    def set_tile(
        self,
        lx: int,
        ly: int,
        tile_id: int,
        tile_registry: TileRegistry
    ) -> None:
        """
        Updates tile ID at (lx, ly) and re-encodes PyBiwis bitmask in-place.
        """
        if not (0 <= lx < self.CHUNK_SIZE and 0 <= ly < self.CHUNK_SIZE):
            return

        self.grid[ly, lx] = np.uint8(tile_id)
        self.is_dirty = True
        self.reencode_bitmask(tile_registry)

    def is_solid_bit(self, lx: int, ly: int) -> bool:
        """
        Performs O(1) register-speed bitwise wall check on PyBiwis words.
        """
        if not (0 <= lx < self.CHUNK_SIZE and 0 <= ly < self.CHUNK_SIZE):
            return True

        flat_idx: int = lx + (ly * self.CHUNK_SIZE)
        word_idx: int = flat_idx // ChunkBitmaskEncoder.WORD_SIZE
        bit_off: int = flat_idx % ChunkBitmaskEncoder.WORD_SIZE

        return bool(
            (self.bitmask_words[word_idx] >> np.uint64(bit_off))
            & np.uint64(1)
        )

    def reencode_bitmask(self, tile_registry: TileRegistry) -> None:
        """
        Re-encodes 4 uint64 PyBiwis bitmask words in-place.
        """
        self.bitmask_words = ChunkBitmaskEncoder.encode_grid_to_words(
            self.grid, tile_registry
        )
        self.is_dirty = False
