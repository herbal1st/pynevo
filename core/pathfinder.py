"""
Discrete topological BFS grid pathfinder and shortest-path tracer.
"""

from collections import deque
from typing import List, Tuple, Optional
import numpy as np

from core.map_data import MapData
from utils.math_utils import CARDINAL_MOVES


class BFSPathfinder:
    """
    Computes step-distance matrices and traces shortest tile paths.
    """

    def __init__(self, map_data: MapData) -> None:
        """
        Binds target map data instance.
        """
        self.map_data: MapData = map_data
        self.distance_matrix: List[List[int]] = []
        self.numpy_dist: Optional[np.ndarray] = None

    def compute_distance_matrix(self) -> Optional[List[List[int]]]:
        """
        Runs BFS backward from the exit tile to compute step distances.
        """
        width: int = self.map_data.width
        height: int = self.map_data.height
        unreachable_val: int = 9999
        dist_grid: List[List[int]] = [
            [unreachable_val for _ in range(width)]
            for _ in range(height)
        ]

        exit_x, exit_y = self.map_data.exit_pos
        dist_grid[exit_y][exit_x] = 0

        queue: deque[Tuple[int, int]] = deque([(exit_x, exit_y)])

        while queue:
            cx, cy = queue.popleft()
            current_dist: int = dist_grid[cy][cx]

            for dx, dy in CARDINAL_MOVES:
                nx: int = cx + dx
                ny: int = cy + dy

                if not self.map_data.is_walkable(nx, ny):
                    continue

                if dist_grid[ny][nx] == unreachable_val:
                    dist_grid[ny][nx] = current_dist + 1
                    queue.append((nx, ny))

        start_x, start_y = self.map_data.start_pos
        if dist_grid[start_y][start_x] == unreachable_val:
            return None

        self.distance_matrix = dist_grid
        self.numpy_dist = np.array(dist_grid, dtype=np.int32)
        return dist_grid

    def get_step_distance(self, tile_x: int, tile_y: int) -> int:
        """
        Retrieves pre-computed step distance to exit in O(1) time.
        """
        if not self.distance_matrix:
            return 9999

        if (
            tile_x < 0 or tile_x >= self.map_data.width or
            tile_y < 0 or tile_y >= self.map_data.height
        ):
            return 9999

        return self.distance_matrix[tile_y][tile_x]

    def compute_shortest_path_tiles(self) -> List[Tuple[int, int]]:
        """
        Traces step-by-step tile coordinates from start_pos to exit_pos.
        """
        if not self.distance_matrix:
            return []

        sx, sy = self.map_data.start_pos
        ex, ey = self.map_data.exit_pos
        curr_x, curr_y = sx, sy
        curr_dist: int = self.get_step_distance(curr_x, curr_y)

        if curr_dist >= 9999:
            return []

        path: List[Tuple[int, int]] = [(sx, sy)]

        while (curr_x, curr_y) != (ex, ey) and curr_dist > 0:
            next_pos: Optional[Tuple[int, int]] = None

            for dx, dy in CARDINAL_MOVES:
                tx: int = curr_x + dx
                ty: int = curr_y + dy
                if self.get_step_distance(tx, ty) == curr_dist - 1:
                    next_pos = (tx, ty)
                    break

            if next_pos is None:
                break

            path.append(next_pos)
            curr_x, curr_y = next_pos
            curr_dist -= 1

        return path

    def count_shortest_path_turns(self) -> int:
        """
        Traces shortest path from start to exit and counts 90-deg turns.
        """
        if not self.distance_matrix:
            return 0

        sx, sy = self.map_data.start_pos
        ex, ey = self.map_data.exit_pos
        if (sx, sy) == (ex, ey):
            return 0

        curr_x, curr_y = sx, sy
        curr_dist: int = self.get_step_distance(curr_x, curr_y)
        if curr_dist >= 9999:
            return 0

        turns: int = 0
        last_dir: Optional[Tuple[int, int]] = None

        while (curr_x, curr_y) != (ex, ey) and curr_dist > 0:
            next_pos: Optional[Tuple[int, int]] = None
            next_dir: Optional[Tuple[int, int]] = None

            if last_dir is not None:
                lx, ly = last_dir
                tx, ty = curr_x + lx, curr_y + ly
                if self.get_step_distance(tx, ty) == curr_dist - 1:
                    next_pos = (tx, ty)
                    next_dir = last_dir

            if next_pos is None:
                for dx, dy in CARDINAL_MOVES:
                    tx, ty = curr_x + dx, curr_y + dy
                    if self.get_step_distance(tx, ty) == curr_dist - 1:
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
