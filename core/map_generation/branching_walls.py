"""
Organic branching wall crawler strategy supporting dynamic N-anchor modes.
"""

import random
from typing import List, Tuple, Set, Optional

from core.map_data import MapData
from core.map_generation.base_strategy import BaseMapStrategy
from core.map_generation.branching.seed_manager import HaloSeedManager
from core.map_generation.branching.extension_solver import (
    StemExtensionSolver
)


class BranchingWallsStrategy(BaseMapStrategy):
    """
    Grows organic branching wall stems with optional dynamic N-anchor mode.
    """

    def __init__(
        self,
        num_anchors: Optional[int] = None,
        early_term_rate: float = 0.05,
        min_straight_steps: int = 1
    ) -> None:
        """
        Initializes strategy, tuning knobs, and helper sub-components.
        """
        self.num_anchors: Optional[int] = num_anchors
        self.early_term_rate: float = early_term_rate
        self.min_straight_steps: int = min_straight_steps
        self.seed_manager: HaloSeedManager = HaloSeedManager()
        self.extension_solver: StemExtensionSolver = StemExtensionSolver()

    def generate_tiles(
        self,
        map_data: MapData,
        wall_density: float
    ) -> bool:
        """
        Grows branching wall stems until target or seed pool is exhausted.
        """
        clamped_density: float = max(0.0, min(1.0, wall_density))
        width: int = map_data.width
        height: int = map_data.height

        border_walls: Set[Tuple[int, int]] = set()
        placed_internal: Set[Tuple[int, int]] = set()

        for y in range(height):
            for x in range(width):
                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    map_data.set_wall(x, y, True)
                    border_walls.add((x, y))

        candidate_set: Set[Tuple[int, int]] = (
            self.seed_manager.build_inner_halo_candidates(width, height)
        )

        max_possible: int = (
            self.extension_solver.calculate_max_possible_walls(
                width, height
            )
        )
        target_walls: int = int(round(max_possible * clamped_density))

        placed_count: int = 0
        anchors_placed: int = 0
        max_attempts: int = max_possible * 100

        cardinal_dirs: List[Tuple[int, int]] = [
            (0, -1), (0, 1), (-1, 0), (1, 0)
        ]

        active_stem: List[Tuple[int, int]] = []
        current_dir: Tuple[int, int] = (0, 0)
        stem_length: int = 0

        attempts: int = 0
        while placed_count < target_walls and attempts < max_attempts:
            attempts += 1

            if not active_stem:
                seed_tile = self.seed_manager.pop_valid_seed(
                    map_data, width, height, border_walls,
                    placed_internal, candidate_set, anchors_placed,
                    num_anchors=self.num_anchors
                )

                if seed_tile is None:
                    break

                sx, sy = seed_tile
                map_data.set_wall(sx, sy, True)
                active_stem.append((sx, sy))
                placed_internal.add((sx, sy))
                placed_count += 1

                if self.seed_manager.is_inner_halo_tile(
                    sx, sy, width, height
                ):
                    anchors_placed += 1

                self.seed_manager.register_internal_wall_tile(
                    sx, sy, width, height, map_data, candidate_set
                )
                current_dir = random.choice(cardinal_dirs)
                stem_length = 1

            else:
                if (
                    stem_length >= 2 and
                    random.random() < self.early_term_rate
                ):
                    placed_internal.update(active_stem)
                    for wx, wy in active_stem:
                        self.seed_manager.register_internal_wall_tile(
                            wx, wy, width, height, map_data, candidate_set
                        )
                    active_stem.clear()
                    stem_length = 0
                    continue

                last_x, last_y = active_stem[-1]

                turns = [
                    (-current_dir[1], current_dir[0]),
                    (current_dir[1], -current_dir[0])
                ]
                random.shuffle(turns)

                candidate_dirs: List[Tuple[int, int]] = []
                if stem_length < self.min_straight_steps:
                    candidate_dirs = [current_dir] + turns
                else:
                    if random.random() < 0.7:
                        candidate_dirs = [current_dir] + turns
                    else:
                        candidate_dirs = turns + [current_dir]

                valid_next: Optional[
                    Tuple[int, int, Tuple[int, int]]
                ] = None
                for dx, dy in candidate_dirs:
                    nx = last_x + dx
                    ny = last_y + dy

                    if not (
                        1 <= nx < width - 1 and 1 <= ny < height - 1
                    ):
                        continue

                    if map_data.is_wall(nx, ny):
                        continue

                    if self.extension_solver.is_valid_stem_extension(
                        nx, ny, active_stem,
                        border_walls, placed_internal
                    ):
                        valid_next = (nx, ny, (dx, dy))
                        break

                if valid_next is not None:
                    nx, ny, new_dir = valid_next
                    map_data.set_wall(nx, ny, True)
                    active_stem.append((nx, ny))
                    placed_internal.add((nx, ny))
                    self.seed_manager.register_internal_wall_tile(
                        nx, ny, width, height, map_data, candidate_set
                    )
                    placed_count += 1
                    current_dir = new_dir
                    stem_length += 1
                else:
                    placed_internal.update(active_stem)
                    for wx, wy in active_stem:
                        self.seed_manager.register_internal_wall_tile(
                            wx, wy, width, height, map_data, candidate_set
                        )
                    active_stem.clear()
                    stem_length = 0

        if active_stem:
            placed_internal.update(active_stem)
            for wx, wy in active_stem:
                self.seed_manager.register_internal_wall_tile(
                    wx, wy, width, height, map_data, candidate_set
                )
            active_stem.clear()

        return True
