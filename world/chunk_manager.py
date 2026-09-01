"""
Spatial chunk manager managing memory, O(1) lookups & radial loading.
"""

import math
from typing import Dict, Tuple, Optional, Any

from world.tile_registry import TileRegistry
from world.chunk import Chunk


class ChunkManager:
    """
    Orchestrates infinite spatial chunk storage and radial load/unload.
    """

    CHUNK_SIZE: int = 16  # tiles

    def __init__(
        self,
        tile_registry: TileRegistry,
        world_seed: int = 420
    ) -> None:
        """
        Initializes spatial chunk storage, tile registry, and world seed.
        """
        self.tile_registry: TileRegistry = tile_registry
        self.world_seed: int = world_seed  # seed
        self.chunks: Dict[Tuple[int, int], Chunk] = {}

    def get_chunk(self, cx: int, cy: int) -> Optional[Chunk]:
        """
        Retrieves loaded chunk at grid coordinate (cx, cy) or None.
        """
        return self.chunks.get((cx, cy))

    def get_tile(self, wx: int, wy: int) -> int:
        """
        Retrieves uint8 tile ID at global position (wx, wy) in O(1) time.
        """
        cx: int = math.floor(wx / self.CHUNK_SIZE)
        cy: int = math.floor(wy / self.CHUNK_SIZE)

        chunk: Optional[Chunk] = self.chunks.get((cx, cy))
        if chunk is None:
            return 0

        lx: int = wx % self.CHUNK_SIZE
        ly: int = wy % self.CHUNK_SIZE
        return chunk.get_tile(lx, ly)

    def is_solid(self, wx: int, wy: int) -> bool:
        """
        Performs O(1) register-speed PyBiwis bitmask wall check at (wx, wy).
        """
        cx: int = math.floor(wx / self.CHUNK_SIZE)
        cy: int = math.floor(wy / self.CHUNK_SIZE)

        chunk: Optional[Chunk] = self.chunks.get((cx, cy))
        if chunk is None:
            return True

        lx: int = wx % self.CHUNK_SIZE
        ly: int = wy % self.CHUNK_SIZE
        return chunk.is_solid_bit(lx, ly)

    def set_tile(self, wx: int, wy: int, tile_id: int) -> None:
        """
        Updates tile ID at (wx, wy) and re-encodes PyBiwis bitmask in-place.
        """
        cx: int = math.floor(wx / self.CHUNK_SIZE)
        cy: int = math.floor(wy / self.CHUNK_SIZE)

        chunk: Optional[Chunk] = self.chunks.get((cx, cy))
        if chunk is None:
            return

        lx: int = wx % self.CHUNK_SIZE
        ly: int = wy % self.CHUNK_SIZE
        chunk.set_tile(lx, ly, tile_id, self.tile_registry)

    def update_focus(
        self,
        center_x: float,
        center_y: float,
        viewport_w_tiles: float,
        viewport_h_tiles: float,
        generator: Any
    ) -> None:
        """
        Synchronously loads chunks in circular R_load & purges R_unload.
        """
        center_tile_x: int = math.floor(center_x)
        center_tile_y: int = math.floor(center_y)

        center_cx: float = float(center_tile_x) / float(self.CHUNK_SIZE)
        center_cy: float = float(center_tile_y) / float(self.CHUNK_SIZE)

        int_center_cx: int = math.floor(center_cx)
        int_center_cy: int = math.floor(center_cy)

        diag_tiles: float = math.sqrt(
            (viewport_w_tiles ** 2) + (viewport_h_tiles ** 2)
        )
        rad_load_chunks: float = (
            (diag_tiles / (2.0 * float(self.CHUNK_SIZE))) + 1.0
        )
        rad_unload_chunks: float = rad_load_chunks + 2.0

        r_load_sq: float = rad_load_chunks ** 2
        r_unload_sq: float = rad_unload_chunks ** 2

        search_bound: int = math.ceil(rad_load_chunks)

        # 1. Load missing chunks within circular R_load
        for dy in range(-search_bound, search_bound + 1):
            for dx in range(-search_bound, search_bound + 1):
                target_cx: int = int_center_cx + dx
                target_cy: int = int_center_cy + dy

                dist_sq: float = (
                    (float(target_cx) + 0.5 - center_cx) ** 2 +
                    (float(target_cy) + 0.5 - center_cy) ** 2
                )

                if dist_sq <= r_load_sq:
                    coord_key: Tuple[int, int] = (target_cx, target_cy)
                    if coord_key not in self.chunks:
                        new_chunk: Chunk = generator.generate_chunk(
                            target_cx,
                            target_cy,
                            self.world_seed,
                            self.tile_registry
                        )
                        self.chunks[coord_key] = new_chunk

        # 2. Unload distant chunks outside circular R_unload
        for coord_key in list(self.chunks.keys()):
            cx_idx, cy_idx = coord_key
            dist_sq = (
                (float(cx_idx) + 0.5 - center_cx) ** 2 +
                (float(cy_idx) + 0.5 - center_cy) ** 2
            )

            if dist_sq > r_unload_sq:
                del self.chunks[coord_key]

    def clear(self) -> None:
        """
        Purges all loaded chunks from active RAM memory.
        """
        self.chunks.clear()
