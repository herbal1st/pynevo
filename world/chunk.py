"""
Defines the Chunk container holding 16x16 tile IDs & PyBiwis bitmasks.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray
import pygame

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
        tile_registry: TileRegistry,
        tile_size: int = 10
    ) -> None:
        """
        Initializes chunk spatial coordinates, grid, bitmask, & surface.
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
        self.surface: Optional[pygame.Surface] = None
        self.bake_surface(tile_registry, tile_size)

    def bake_surface(
        self,
        tile_registry: TileRegistry,
        tile_size: int = 10
    ) -> pygame.Surface:
        """
        Bakes 16x16 tiles and borders onto a native converted surface.
        """
        px_dim: int = self.CHUNK_SIZE * tile_size
        surf: pygame.Surface = pygame.Surface((px_dim, px_dim))

        for y in range(self.CHUNK_SIZE):
            for x in range(self.CHUNK_SIZE):
                tile_id: int = int(self.grid[y, x])
                t_prof = tile_registry.get_tile(tile_id)
                t_x: int = x * tile_size
                t_y: int = y * tile_size
                t_rect = (t_x, t_y, tile_size, tile_size)

                surf.fill(t_prof.color, t_rect)

                if t_prof.border_width_ratio > 0.0:
                    b_px: int = max(
                        1,
                        int(tile_size * t_prof.border_width_ratio * 0.5)
                    )
                    surf.fill(t_prof.border_color, t_rect)
                    in_rect = (
                        t_x + b_px,
                        t_y + b_px,
                        max(1, tile_size - (2 * b_px)),
                        max(1, tile_size - (2 * b_px))
                    )
                    surf.fill(t_prof.color, in_rect)

        if pygame.display.get_surface() is not None:
            self.surface = surf.convert()
        else:
            self.surface = surf

        self.is_dirty = False
        return self.surface

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
        tile_registry: TileRegistry,
        tile_size: int = 10
    ) -> None:
        """
        Updates tile ID at (lx, ly) and re-bakes surface and bitmask.
        """
        if not (0 <= lx < self.CHUNK_SIZE and 0 <= ly < self.CHUNK_SIZE):
            return

        self.grid[ly, lx] = np.uint8(tile_id)
        self.is_dirty = True
        self.reencode_bitmask(tile_registry)
        self.bake_surface(tile_registry, tile_size)

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
