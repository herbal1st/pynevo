"""
Endless noise chunk generator mapping multi-octave noise to tile profiles.
"""

from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray

from entities.map_endless_profile_registry import ResolvedMapEndlessProfile
from world.tile_registry import TileRegistry
from world.chunk import Chunk
from utils.noise import SimplexNoise, PerlinNoise


class EndlessNoiseGenerator:
    """
    Populates 16x16 chunk matrices deterministically from noise profiles.
    """

    CHUNK_SIZE: int = 16  # tiles

    def __init__(
        self,
        profile: ResolvedMapEndlessProfile,
        tile_registry: TileRegistry
    ) -> None:
        """
        Initializes generator with endless map profile and tile registry.
        """
        self.profile: ResolvedMapEndlessProfile = profile
        self.tile_registry: TileRegistry = tile_registry

        if profile.noise_type.upper() == "PERLIN":
            self.noise_engine: Union[
                SimplexNoise, PerlinNoise
            ] = PerlinNoise(profile.world_seed)
        else:
            self.noise_engine = SimplexNoise(profile.world_seed)

        self._strata_cache: Tuple[Tuple[float, int], ...] = (
            self._precache_strata(profile, tile_registry)
        )

    def generate_chunk(
        self,
        cx: int,
        cy: int,
        world_seed: int,
        tile_registry: TileRegistry
    ) -> Chunk:
        """
        Generates a 16x16 chunk grid at (cx, cy) using noise stratification.
        """
        wx_start: int = cx * self.CHUNK_SIZE
        wy_start: int = cy * self.CHUNK_SIZE

        x_coords: NDArray[np.float32] = (
            np.arange(wx_start, wx_start + self.CHUNK_SIZE, dtype=np.float32)
        )
        y_coords: NDArray[np.float32] = (
            np.arange(wy_start, wy_start + self.CHUNK_SIZE, dtype=np.float32)
        )

        wx_grid, wy_grid = np.meshgrid(x_coords, y_coords)

        noise_field: NDArray[np.float32] = self.noise_engine.sample_grid(
            wx_grid,
            wy_grid,
            scale=self.profile.noise_scale,
            octaves=self.profile.octaves,
            octaves_decay=self.profile.octaves_decay
        )

        grid: NDArray[np.uint8] = np.zeros(
            (self.CHUNK_SIZE, self.CHUNK_SIZE), dtype=np.uint8
        )

        for r in range(self.CHUNK_SIZE):
            for c in range(self.CHUNK_SIZE):
                val: float = float(noise_field[r, c])
                grid[r, c] = np.uint8(self._resolve_tile_id(val))

        return Chunk(cx, cy, grid, tile_registry)

    def _resolve_tile_id(self, noise_val: float) -> int:
        """
        Resolves tile ID by evaluating strata thresholds in ascending order.
        """
        for thresh, t_id in self._strata_cache:
            if noise_val <= thresh:
                return t_id

        return self._strata_cache[-1][1] if self._strata_cache else 0

    def _precache_strata(
        self,
        profile: ResolvedMapEndlessProfile,
        registry: TileRegistry
    ) -> Tuple[Tuple[float, int], ...]:
        """
        Precaches strata thresholds and tile IDs for fast runtime resolution.
        """
        cache: list[Tuple[float, int]] = []
        for thresh, t_name in profile.strata_layers:
            tile_prof = registry.get_tile_by_name(t_name)
            cache.append((thresh, tile_prof.tile_id))

        cache.sort(key=lambda x: x[0])
        return tuple(cache)
