"""
Candidate spawn angle generator with BFS path alignment & cardinal probing.
"""

import math
import random
from typing import Tuple, Optional, List

from core.map_data import MapData
from core.pathfinder import BFSPathfinder
from utils.math_utils import normalize_angle_2pi


class SpawnHeadingGenerator:
    """
    Generates initial spawn headings aligned to orthogonal cardinal corridors.
    """

    @staticmethod
    def generate_random_heading(
        map_data: Optional[MapData] = None,
        start_pos: Optional[Tuple[int, int]] = None,
        use_bfs_spawn_heading: bool = True
    ) -> float:
        """
        Selects spawn angle facing open cardinal corridor, aligned to BFS if set.
        """
        if map_data is None or start_pos is None:
            cardinal_angles = [
                0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0
            ]
            return random.choice(cardinal_angles)

        start_x, start_y = start_pos
        cardinal_moves: List[Tuple[int, int, float]] = [
            (0, -1, 3.0 * math.pi / 2.0),
            (1, 0, 0.0),
            (0, 1, math.pi / 2.0),
            (-1, 0, math.pi)
        ]

        if use_bfs_spawn_heading:
            pathfinder = BFSPathfinder(map_data)
            if not pathfinder.distance_matrix:
                pathfinder.compute_distance_matrix()

            curr_dist: int = pathfinder.get_step_distance(start_x, start_y)
            if curr_dist < 9999:
                bfs_candidates: List[float] = []
                for dx, dy, target_angle in cardinal_moves:
                    nx: int = start_x + dx
                    ny: int = start_y + dy
                    if pathfinder.get_step_distance(nx, ny) == curr_dist - 1:
                        bfs_candidates.append(
                            normalize_angle_2pi(target_angle)
                        )

                if bfs_candidates:
                    return random.choice(bfs_candidates)

        open_cardinal_candidates: List[float] = []
        for dx, dy, target_angle in cardinal_moves:
            nx = start_x + dx
            ny = start_y + dy
            if map_data.is_walkable(nx, ny):
                open_cardinal_candidates.append(
                    normalize_angle_2pi(target_angle)
                )

        if open_cardinal_candidates:
            return random.choice(open_cardinal_candidates)

        return 0.0
