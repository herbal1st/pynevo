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

        self._thresh_arr, self._id_arr = self._precache_strata(
            profile, tile_registry
        )

    def generate_chunk(
        self,
        cx: int,
        cy: int,
        world_seed: int,
        tile_registry: TileRegistry
    ) -> Chunk:
        """
        Generates 16x16 chunk grid at (cx, cy) using 18x18 halo padding.
        """
        wx_start: int = (cx * self.CHUNK_SIZE) - 1
        wy_start: int = (cy * self.CHUNK_SIZE) - 1
        padded_dim: int = self.CHUNK_SIZE + 2

        x_coords: NDArray[np.float32] = np.arange(
            wx_start, wx_start + padded_dim, dtype=np.float32
        )
        y_coords: NDArray[np.float32] = np.arange(
            wy_start, wy_start + padded_dim, dtype=np.float32
        )

        wx_grid, wy_grid = np.meshgrid(x_coords, y_coords)

        noise_field: NDArray[np.float32] = self.noise_engine.sample_grid(
            wx_grid,
            wy_grid,
            scale=self.profile.noise_scale,
            octaves=self.profile.octaves,
            octaves_decay=self.profile.octaves_decay
        )

        indices: NDArray[np.intp] = np.digitize(
            noise_field, self._thresh_arr
        )
        np.clip(indices, 0, len(self._id_arr) - 1, out=indices)

        padded_grid: NDArray[np.uint8] = self._id_arr[indices].astype(
            np.uint8
        )
        inner_grid: NDArray[np.uint8] = padded_grid[1:17, 1:17].copy()

        return Chunk(
            cx,
            cy,
            inner_grid,
            tile_registry,
            tile_size=self.profile.tile_size,
            padded_grid=padded_grid
        )

    def _precache_strata(
        self,
        profile: ResolvedMapEndlessProfile,
        registry: TileRegistry
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint8]]:
        """
        Precaches strata thresholds and tile IDs for fast vectorized math.
        """
        sorted_strata = sorted(profile.strata_layers, key=lambda x: x[0])
        thresh_list: list[float] = [item[0] for item in sorted_strata]
        id_list: list[int] = [
            registry.get_tile_by_name(item[1]).tile_id
            for item in sorted_strata
        ]

        thresh_arr: NDArray[np.float32] = np.array(
            thresh_list, dtype=np.float32
        )
        id_arr: NDArray[np.uint8] = np.array(id_list, dtype=np.uint8)

        return thresh_arr, id_arr
