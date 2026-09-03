"""
Extracts 2D continuous float noise height matrices for active viewports.
"""

import math
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from world.chunk_manager import ChunkManager
from world.generation.endless_noise import EndlessNoiseGenerator


class ViewportHeightSampler:
    """
    Samples float noise fields & tile IDs directly from ChunkManager.
    """

    def sample_viewport_heights(
        self,
        focus_x: float,  # tiles
        focus_y: float,  # tiles
        vw_tiles: float,  # tiles
        vh_tiles: float,  # tiles
        chunk_manager: ChunkManager,
        generator: EndlessNoiseGenerator
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint8], int, int]:
        """
        Extracts 2D height noise matrix & tile ID matrix in C-speed slices.
        """
        half_w: float = vw_tiles / 2.0  # tiles
        half_h: float = vh_tiles / 2.0  # tiles

        min_tx: int = math.floor(focus_x - half_w) - 1  # tile_x
        max_tx: int = math.ceil(focus_x + half_w) + 1  # tile_x
        min_ty: int = math.floor(focus_y - half_h) - 1  # tile_y
        max_ty: int = math.ceil(focus_y + half_h) + 1  # tile_y

        grid_w: int = max_tx - min_tx + 1  # tiles
        grid_h: int = max_ty - min_ty + 1  # tiles

        x_coords: NDArray[np.float32] = np.arange(
            min_tx, max_tx + 1, dtype=np.float32
        )
        y_coords: NDArray[np.float32] = np.arange(
            min_ty, max_ty + 1, dtype=np.float32
        )

        wx_grid, wy_grid = np.meshgrid(x_coords, y_coords)

        height_matrix: NDArray[np.float32] = (
            generator.noise_engine.sample_grid(
                wx_grid,
                wy_grid,
                scale=generator.profile.noise_scale,
                octaves=generator.profile.octaves,
                octaves_decay=generator.profile.octaves_decay
            )
        )

        tile_id_matrix: NDArray[np.uint8] = np.zeros(
            (grid_h, grid_w), dtype=np.uint8
        )

        c_size: int = ChunkManager.CHUNK_SIZE  # tiles
        min_cx: int = math.floor(min_tx / c_size)  # chunk_x
        max_cx: int = math.floor(max_tx / c_size)  # chunk_x
        min_cy: int = math.floor(min_ty / c_size)  # chunk_y
        max_cy: int = math.floor(max_ty / c_size)  # chunk_y

        for cy in range(min_cy, max_cy + 1):
            for cx in range(min_cx, max_cx + 1):
                chunk = chunk_manager.get_chunk(cx, cy)
                if chunk is None:
                    continue

                chk_wx: int = cx * c_size  # tile_x
                chk_wy: int = cy * c_size  # tile_y

                overlap_x0: int = max(min_tx, chk_wx)  # tile_x
                overlap_x1: int = min(max_tx, chk_wx + c_size - 1)
                overlap_y0: int = max(min_ty, chk_wy)  # tile_y
                overlap_y1: int = min(max_ty, chk_wy + c_size - 1)

                if overlap_x0 > overlap_x1 or overlap_y0 > overlap_y1:
                    continue

                dst_x0: int = overlap_x0 - min_tx
                dst_x1: int = overlap_x1 - min_tx + 1
                dst_y0: int = overlap_y0 - min_ty
                dst_y1: int = overlap_y1 - min_ty + 1

                src_x0: int = overlap_x0 - chk_wx
                src_x1: int = overlap_x1 - chk_wx + 1
                src_y0: int = overlap_y0 - chk_wy
                src_y1: int = overlap_y1 - chk_wy + 1

                tile_id_matrix[
                    dst_y0:dst_y1, dst_x0:dst_x1
                ] = chunk.grid[src_y0:src_y1, src_x0:src_x1]

        return height_matrix, tile_id_matrix, min_tx, min_ty
