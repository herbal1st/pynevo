"""
Data structures for 2D map grids and compact PyBiwis bitmask encoding.
"""

import math
import random
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

from core.bitmask_encoder import BitmaskEncoder


@njit(fastmath=True, cache=True)
def march_los_segment_jit(
    cx: float,
    cy: float,
    ex: float,
    ey: float,
    grid_array: NDArray[np.uint8],
    width: int,
    height: int,
    step_size: float = 0.2
) -> bool:
    """
    JIT-compiled ray march checking wall collisions directly over grid_array.
    """
    dx = ex - cx
    dy = ey - cy
    dist = (dx * dx + dy * dy) ** 0.5

    if dist < 1e-4:
        return True

    dir_x = dx / dist
    dir_y = dy / dist
    num_steps = int(dist / step_size)

    for step in range(1, num_steps):
        px = int(math.floor(cx + (dir_x * step * step_size)))
        py = int(math.floor(cy + (dir_y * step * step_size)))

        if px < 0 or px >= width or py < 0 or py >= height:
            return False
        if grid_array[py, px] == 1:
            return False

    return True


class MapData:
    """
    Stores 2D tile layout data with PyBiwis 64-bit packing & target sequences.
    """

    def __init__(
        self,
        width: int,
        height: int,
        start_pos: Tuple[int, int],
        exit_pos: Tuple[int, int]
    ) -> None:
        self.width: int = width
        self.height: int = height
        self.start_pos: Tuple[int, int] = start_pos
        self.exit_pos: Tuple[int, int] = exit_pos
        self.grid: List[List[int]] = [
            [0 for _ in range(width)] for _ in range(height)
        ]
        self.grid_array: NDArray[np.uint8] = np.zeros(
            (height, width), dtype=np.uint8
        )
        self.bitmask_chunks: List[int] = []
        self.los_cache: Optional[List[List[bool]]] = None
        self.target_sequence: List[Tuple[int, int]] = [exit_pos]

    def get_target_pos(self, stage_idx: int) -> Tuple[int, int]:
        """
        Retrieves target coordinates using direct fast indexing.
        """
        seq = self.target_sequence
        n = len(seq)
        if stage_idx < n:
            return seq[stage_idx]
        if n > 0:
            return seq[-1]
        return self.exit_pos

    def append_next_target(
        self,
        seed_offset: int = 0
    ) -> Tuple[int, int]:
        if not self.target_sequence:
            self.target_sequence = [self.exit_pos]

        curr_target = self.target_sequence[-1]
        open_tiles: List[Tuple[int, int]] = [
            (x, y) for y in range(1, self.height - 1)
            for x in range(1, self.width - 1)
            if self.is_walkable(x, y) and (x, y) != curr_target
        ]

        if not open_tiles:
            return curr_target

        rng = random.Random(
            self.width * 1000 + self.height * 100 + seed_offset
            + len(self.target_sequence)
        )
        next_target = rng.choice(open_tiles)
        self.target_sequence.append(next_target)
        return next_target

    def set_wall(self, x: int, y: int, is_wall: bool = True) -> None:
        val: int = 1 if is_wall else 0
        self.grid[y][x] = val
        self.grid_array[y, x] = np.uint8(val)

    def is_wall(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return self.grid_array[y, x] == 1

    def is_walkable(self, x: int, y: int) -> bool:
        return not self.is_wall(x, y)

    def encode_bitmask(self) -> List[int]:
        chunks: List[int] = BitmaskEncoder.encode_grid_to_chunks(
            self.grid, self.width, self.height
        )
        self.bitmask_chunks = chunks
        return chunks

    def decode_bitmask(self, chunks: List[int]) -> None:
        self.bitmask_chunks = chunks
        BitmaskEncoder.decode_chunks_to_grid(
            chunks, self.grid, self.width, self.height
        )
        for y in range(self.height):
            for x in range(self.width):
                self.grid_array[y, x] = np.uint8(self.grid[y][x])

    def compute_exit_los_cache(self) -> None:
        cache: List[List[bool]] = [
            [False for _ in range(self.width)] for _ in range(self.height)
        ]
        ex, ey = self.exit_pos
        exit_targets = [
            (float(ex) + 0.5, float(ey) + 0.5),
            (float(ex) + 0.1, float(ey) + 0.1),
            (float(ex) + 0.9, float(ey) + 0.1),
            (float(ex) + 0.1, float(ey) + 0.9),
            (float(ex) + 0.9, float(ey) + 0.9)
        ]

        for ty in range(self.height):
            for tx in range(self.width):
                if self.is_wall(tx, ty):
                    continue

                tile_probes = [
                    (float(tx) + 0.5, float(ty) + 0.5),
                    (float(tx) + 0.1, float(ty) + 0.1),
                    (float(tx) + 0.9, float(ty) + 0.1),
                    (float(tx) + 0.1, float(ty) + 0.9),
                    (float(tx) + 0.9, float(ty) + 0.9)
                ]

                has_los: bool = False
                for px, py in tile_probes:
                    for tx_target, ty_target in exit_targets:
                        if self._march_los_segment(px, py, tx_target, ty_target):
                            has_los = True
                            break
                    if has_los:
                        break

                cache[ty][tx] = has_los

        self.los_cache = cache

    def has_line_of_sight_to_exit(self, tile_x: int, tile_y: int) -> bool:
        if self.los_cache is None:
            self.compute_exit_los_cache()

        if (
            tile_x < 0 or tile_x >= self.width or
            tile_y < 0 or tile_y >= self.height
        ):
            return False

        if self.los_cache is None:
            return False

        return self.los_cache[tile_y][tile_x]

    def _march_los_segment(
        self,
        cx: float,
        cy: float,
        ex: float,
        ey: float,
        step_size: float = 0.2
    ) -> bool:
        """
        Fast line-of-sight segment probe checking wall obstacles via JIT.
        """
        return march_los_segment_jit(
            cx, cy, ex, ey, self.grid_array, self.width, self.height, step_size
        )