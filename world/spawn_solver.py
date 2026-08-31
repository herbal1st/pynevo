"""
Universal safe spawn solver for infinite procedural noise terrain.
"""

import math
from typing import Tuple, Set, Any

from world.tile_registry import TileRegistry
from world.chunk_manager import ChunkManager


class EndlessSpawnSolver:
    """
    Locates safe, unobstructed floor tiles in infinite noise terrain.
    """

    @classmethod
    def find_safe_spawn(
        cls,
        chunk_manager: ChunkManager,
        tile_registry: TileRegistry,
        generator: Any,
        center_x: float = 0.0,
        center_y: float = 0.0,
        diameter_ratio: float = 0.5,
        min_speed_mult: float = 0.1,
        max_search_radius: int = 100
    ) -> Tuple[float, float]:
        """
        Spirals outward from center to find safe spawn coordinates.
        """
        rad_tiles: float = max(0.01, 0.5 * float(diameter_ratio))
        tx_center: int = math.floor(center_x)
        ty_center: int = math.floor(center_y)

        for r in range(max_search_radius + 1):
            ring_tiles = cls._get_ring_coordinates(
                tx_center, ty_center, r
            )
            for tx, ty in ring_tiles:
                cx: float = float(tx) + 0.5
                cy: float = float(ty) + 0.5

                if cls._is_footprint_clear(
                    cx,
                    cy,
                    rad_tiles,
                    min_speed_mult,
                    chunk_manager,
                    tile_registry,
                    generator
                ):
                    return cx, cy

        return center_x, center_y

    @classmethod
    def _get_ring_coordinates(
        cls,
        tx: int,
        ty: int,
        radius: int
    ) -> Set[Tuple[int, int]]:
        """
        Returns all tile coordinates forming a square ring at radius r.
        """
        if radius == 0:
            return {(tx, ty)}

        coords: Set[Tuple[int, int]] = set()
        for i in range(-radius, radius + 1):
            coords.add((tx + i, ty - radius))
            coords.add((tx + i, ty + radius))
            coords.add((tx - radius, ty + i))
            coords.add((tx + radius, ty + i))

        return coords

    @classmethod
    def _is_footprint_clear(
        cls,
        cx: float,
        cy: float,
        rad_tiles: float,
        min_speed_mult: float,
        chunk_manager: ChunkManager,
        tile_registry: TileRegistry,
        generator: Any
    ) -> bool:
        """
        Validates continuous multi-tile bounding footprint clearance.
        """
        min_tx: int = math.floor(cx - rad_tiles)
        max_tx: int = math.floor(cx + rad_tiles)
        min_ty: int = math.floor(cy - rad_tiles)
        max_ty: int = math.floor(cy + rad_tiles)

        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                cls._ensure_chunk_loaded(
                    tx, ty, chunk_manager, tile_registry, generator
                )
                tile_id: int = chunk_manager.get_tile(tx, ty)
                tile_prof = tile_registry.get_tile(tile_id)

                if (
                    tile_prof.solid or
                    tile_prof.speed_multiplier < min_speed_mult
                ):
                    return False

        return True

    @classmethod
    def _ensure_chunk_loaded(
        cls,
        tx: int,
        ty: int,
        chunk_manager: ChunkManager,
        tile_registry: TileRegistry,
        generator: Any
    ) -> None:
        """
        Ensures chunk containing tile coordinate (tx, ty) is generated.
        """
        c_size: int = ChunkManager.CHUNK_SIZE
        cx_chunk: int = math.floor(tx / c_size)
        cy_chunk: int = math.floor(ty / c_size)
        coord_key: Tuple[int, int] = (cx_chunk, cy_chunk)

        if coord_key not in chunk_manager.chunks:
            chunk = generator.generate_chunk(
                cx_chunk,
                cy_chunk,
                chunk_manager.world_seed,
                tile_registry
            )
            chunk_manager.chunks[coord_key] = chunk
