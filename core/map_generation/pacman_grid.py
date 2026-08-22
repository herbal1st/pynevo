"""
Arcade Pacman grid strategy with outer ring corridor & pillar generation.
"""

import random
from typing import List, Tuple, Set, Optional

from core.map_data import MapData
from core.map_generation.base_strategy import BaseMapStrategy
from core.map_generation.halo_utils import (
    is_inner_halo_tile,
    get_inner_halo_tiles,
    calculate_snake_corridor_max_walls
)


class PacmanGridStrategy(BaseMapStrategy):
    """
    Grows arcade pillar arenas with outer ring corridors & diagonal discards.
    """

    def __init__(self, num_anchors: Optional[int] = None) -> None:
        """
        Initializes strategy knob for border anchor stubs.
        """
        self.num_anchors: Optional[int] = num_anchors

    def generate_tiles(
        self, map_data: MapData, wall_density: float
    ) -> bool:
        """
        Fills outer borders and generates spaced central wall pillars.
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

        halo_tiles: Set[Tuple[int, int]] = get_inner_halo_tiles(
            width, height
        )

        if self.num_anchors is None or self.num_anchors == 0:
            candidate_set.difference_update(halo_tiles)

        anchors_placed: int = 0
        placed_count: int = 0

        while candidate_set and placed_count < target_walls:
            candidates: List[Tuple[int, int]] = list(candidate_set)
            random.shuffle(candidates)
            placed_in_pass: bool = False

            for cx, cy in candidates:
                if map_data.is_wall(cx, cy):
                    candidate_set.discard((cx, cy))
                    continue

                is_halo: bool = is_inner_halo_tile(cx, cy, width, height)
                if (
                    self.num_anchors is not None
                    and self.num_anchors > 0
                    and is_halo
                ):
                    if anchors_placed >= self.num_anchors:
                        candidate_set.discard((cx, cy))
                        continue
                    anchors_placed += 1

                map_data.set_wall(cx, cy, True)
                placed_count += 1
                placed_in_pass = True

                candidate_set.discard((cx, cy))
                candidate_set.discard((cx - 1, cy - 1))
                candidate_set.discard((cx + 1, cy - 1))
                candidate_set.discard((cx - 1, cy + 1))
                candidate_set.discard((cx + 1, cy + 1))

                if (
                    self.num_anchors is not None
                    and self.num_anchors > 0
                    and anchors_placed >= self.num_anchors
                ):
                    candidate_set.difference_update(halo_tiles)

                break

            if not placed_in_pass:
                break

        return True
