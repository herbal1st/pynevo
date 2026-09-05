"""
Discrete topological BFS grid pathfinder and shortest-path tracer.
"""

from collections import deque
from typing import List, Tuple, Optional, Dict
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

from core.map_data import MapData
from utils.math_utils import CARDINAL_MOVES


# 218,297 calls in trace -> compile down to LLVM machine instructions
@njit(fastmath=True, cache=True)
def get_step_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Direct JIT-compiled distance calculation bypassing Python math.sqrt abstraction.
    """
    dx = x2 - x1
    dy = y2 - y1
    return (dx * dx + dy * dy) ** 0.5


@njit(fastmath=True, cache=True)
def lookup_step_distance_jit(
    matrix_buffer: NDArray[np.int32],
    tile_x: int,
    tile_y: int,
    stage_idx: int,
    width: int,
    height: int,
    max_targets: int,
    unreachable_val: int = 9999
) -> int:
    """
    O(1) register-speed buffer lookup with inlined bounds check.
    """
    if (
        tile_x < 0 or tile_x >= width or
        tile_y < 0 or tile_y >= height or
        stage_idx < 0 or stage_idx >= max_targets
    ):
        return unreachable_val
    return int(matrix_buffer[stage_idx, tile_y, tile_x])


class BFSPathfinder:
    """
    Computes step-distance matrices and traces shortest tile paths.
    """

    def __init__(
        self,
        map_data: MapData,
        max_targets: int = 32
    ) -> None:
        """
        Binds target map data and pre-allocates 3D NumPy distance buffer.
        """
        self.map_data: MapData = map_data
        self.max_targets: int = max_targets
        self.unreachable_val: int = 9999
        self._matrix_buffer: NDArray[np.int32] = np.full(
            (max_targets, map_data.height, map_data.width),
            self.unreachable_val,
            dtype=np.int32
        )
        self._cached_targets: Dict[Tuple[int, int], int] = {}

    @property
    def distance_matrix(self) -> List[List[int]]:
        """
        Backward compatibility property returning Stage 0 distance matrix.
        """
        return self._matrix_buffer[0].tolist()

    def clear_cache(self) -> None:
        """
        Resets 3D distance buffer to 9999 and clears target map in-place.
        """
        self._matrix_buffer.fill(self.unreachable_val)
        self._cached_targets.clear()

    def compute_distance_matrix_for_target(
        self,
        target_pos: Tuple[int, int],
        stage_idx: int = 0
    ) -> NDArray[np.int32]:
        """
        Runs backward BFS from target_pos directly into 3D NumPy buffer.
        """
        if stage_idx >= self.max_targets:
            new_max: int = (
                self.max_targets * 2
                if self.max_targets * 2 > (stage_idx + 1)
                else (stage_idx + 1)
            )
            new_buf = np.full(
                (new_max, self.map_data.height, self.map_data.width),
                self.unreachable_val,
                dtype=np.int32
            )
            new_buf[:self.max_targets] = self._matrix_buffer
            self._matrix_buffer = new_buf
            self.max_targets = new_max

        tx, ty = target_pos
        self._matrix_buffer[stage_idx].fill(self.unreachable_val)
        self._matrix_buffer[stage_idx, ty, tx] = 0

        queue: deque[Tuple[int, int]] = deque([(tx, ty)])

        while queue:
            cx, cy = queue.popleft()
            current_dist: int = int(self._matrix_buffer[stage_idx, cy, cx])

            for dx, dy in CARDINAL_MOVES:
                nx: int = cx + dx
                ny: int = cy + dy

                if not self.map_data.is_walkable(nx, ny):
                    continue

                if (
                    self._matrix_buffer[stage_idx, ny, nx] ==
                    self.unreachable_val
                ):
                    self._matrix_buffer[stage_idx, ny, nx] = (
                        current_dist + 1
                    )
                    queue.append((nx, ny))

        self._cached_targets[target_pos] = stage_idx
        return self._matrix_buffer[stage_idx]

    def compute_distance_matrix(self) -> Optional[List[List[int]]]:
        """
        Computes Stage 0 distance matrix from map exit_pos.
        """
        arr = self.compute_distance_matrix_for_target(
            self.map_data.exit_pos, stage_idx=0
        )
        sx, sy = self.map_data.start_pos
        if arr[sy, sx] >= self.unreachable_val:
            return None
        return arr.tolist()

    def get_step_distance(
        self,
        tile_x: int,
        tile_y: int,
        stage_idx: int = 0
    ) -> int:
        """
        Retrieves step distance to stage target in O(1) time via compiled JIT.
        """
        return lookup_step_distance_jit(
            self._matrix_buffer,
            tile_x,
            tile_y,
            stage_idx,
            self.map_data.width,
            self.map_data.height,
            self.max_targets,
            self.unreachable_val
        )

    def compute_shortest_path_tiles(
        self,
        stage_idx: int = 0
    ) -> List[Tuple[int, int]]:
        """
        Traces step-by-step tile coordinates from start_pos to target.
        """
        sx, sy = self.map_data.start_pos
        ex, ey = (
            self.map_data.get_target_pos(stage_idx)
            if hasattr(self.map_data, "get_target_pos")
            else self.map_data.exit_pos
        )

        curr_x, curr_y = sx, sy
        curr_dist: int = self.get_step_distance(curr_x, curr_y, stage_idx)

        if curr_dist >= self.unreachable_val:
            return []

        path: List[Tuple[int, int]] = [(sx, sy)]

        while (curr_x, curr_y) != (ex, ey) and curr_dist > 0:
            next_pos: Optional[Tuple[int, int]] = None

            for dx, dy in CARDINAL_MOVES:
                tx: int = curr_x + dx
                ty: int = curr_y + dy
                if (
                    self.get_step_distance(tx, ty, stage_idx) ==
                    curr_dist - 1
                ):
                    next_pos = (tx, ty)
                    break

            if next_pos is None:
                break

            path.append(next_pos)
            curr_x, curr_y = next_pos
            curr_dist -= 1

        return path

    def count_shortest_path_turns(self, stage_idx: int = 0) -> int:
        """
        Traces shortest path to target and counts 90-degree turns.
        """
        sx, sy = self.map_data.start_pos
        ex, ey = (
            self.map_data.get_target_pos(stage_idx)
            if hasattr(self.map_data, "get_target_pos")
            else self.map_data.exit_pos
        )

        if (sx, sy) == (ex, ey):
            return 0

        curr_x, curr_y = sx, sy
        curr_dist: int = self.get_step_distance(curr_x, curr_y, stage_idx)
        if curr_dist >= self.unreachable_val:
            return 0

        turns: int = 0
        last_dir: Optional[Tuple[int, int]] = None

        while (curr_x, curr_y) != (ex, ey) and curr_dist > 0:
            next_pos: Optional[Tuple[int, int]] = None
            next_dir: Optional[Tuple[int, int]] = None

            if last_dir is not None:
                lx, ly = last_dir
                tx, ty = curr_x + lx, curr_y + ly
                if (
                    self.get_step_distance(tx, ty, stage_idx) ==
                    curr_dist - 1
                ):
                    next_pos = (tx, ty)
                    next_dir = last_dir

            if next_pos is None:
                for dx, dy in CARDINAL_MOVES:
                    tx, ty = curr_x + dx, curr_y + dy
                    if (
                        self.get_step_distance(tx, ty, stage_idx) ==
                        curr_dist - 1
                    ):
                        next_pos = (tx, ty)
                        next_dir = (dx, dy)
                        break

            if next_pos is None or next_dir is None:
                break

            if last_dir is not None and next_dir != last_dir:
                turns += 1

            last_dir = next_dir
            curr_x, curr_y = next_pos
            curr_dist -= 1

        return turns