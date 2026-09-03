"""
Reworked random scatter strategy with on-demand diagonal validation.
"""

import random
from typing import List, Tuple, Set

from core.map_data import MapData
from core.map_generation.base_strategy import BaseMapStrategy
from core.map_generation.halo_utils import (
    calculate_snake_corridor_max_walls
)


class RandomScatterStrategy(BaseMapStrategy):
    """
    Grows organic random labyrinths with on-demand diagonal validation.
    """

    def generate_tiles(
        self, map_data: MapData, wall_density: float
    ) -> bool:
        """
        Populates outer borders and places physics-safe random walls.
        """
        clamped_density: float = max(0.0, min(1.0, wall_density))
        width: int = map_data.width
        height: int = map_data.height

        for y in range(height):
            for x in range(width):
                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    map_data.set_wall(x, y, True)

        max_possible: int = calculate_snake_corridor_max_walls(
            width, height
        )
        target_walls: int = int(round(max_possible * clamped_density))

        candidate_set: Set[Tuple[int, int]] = {
            (x, y)
            for y in range(1, height - 1)
            for x in range(1, width - 1)
        }

        placed_walls: Set[Tuple[int, int]] = {
            (x, y)
            for y in range(height)
            for x in range(width)
            if map_data.is_wall(x, y)
        }

        placed_count: int = 0

        while candidate_set and placed_count < target_walls:
            candidates: List[Tuple[int, int]] = list(candidate_set)
            random.shuffle(candidates)
            placed_in_pass: bool = False

            for cx, cy in candidates:
                if map_data.is_wall(cx, cy):
                    candidate_set.discard((cx, cy))
                    continue

                if self._has_isolated_diagonal_touch(
                    cx, cy, placed_walls
                ):
                    candidate_set.discard((cx, cy))
                    continue

                map_data.set_wall(cx, cy, True)
                placed_walls.add((cx, cy))
                candidate_set.discard((cx, cy))
                placed_count += 1
                placed_in_pass = True
                break

            if not placed_in_pass:
                break

        return True

    def _has_isolated_diagonal_touch(
        self, x: int, y: int, placed_walls: Set[Tuple[int, int]]
    ) -> bool:
        """
        Checks if (x, y) touches an existing wall only diagonally.
        """
        diag_offsets: List[Tuple[int, int]] = [
            (-1, -1), (1, -1), (-1, 1), (1, 1)
        ]
        for dx, dy in diag_offsets:
            bx: int = x + dx
            by: int = y + dy
            if (bx, by) in placed_walls:
                ortho_a: Tuple[int, int] = (x + dx, y)
                ortho_b: Tuple[int, int] = (x, y + dy)
                if (
                    ortho_a not in placed_walls
                    and ortho_b not in placed_walls
                ):
                    return True

        return False
