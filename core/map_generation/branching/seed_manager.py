"""
Inner halo candidate seed pool manager and collinear seed validator.
"""

import random
from typing import List, Tuple, Set, Optional

from core.map_data import MapData


class HaloSeedManager:
    """
    Manages inner halo candidate seed pools and collinear seed validation.
    """

    def build_inner_halo_candidates(
        self,
        width: int,
        height: int
    ) -> Set[Tuple[int, int]]:
        """
        Builds initial inner halo candidate set bordering outer walls.
        """
        candidates: Set[Tuple[int, int]] = set()

        for x in range(1, width - 1):
            candidates.add((x, 1))
            candidates.add((x, height - 2))

        for y in range(1, height - 1):
            candidates.add((1, y))
            candidates.add((width - 2, y))

        return candidates

    def is_inner_halo_tile(
        self,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> bool:
        """
        Checks if tile (x, y) lies on the inner halo adjacent to borders.
        """
        return x == 1 or x == width - 2 or y == 1 or y == height - 2

    def pop_valid_seed(
        self,
        map_data: MapData,
        width: int,
        height: int,
        border_walls: Set[Tuple[int, int]],
        placed_internal: Set[Tuple[int, int]],
        candidate_set: Set[Tuple[int, int]],
        anchors_placed: int,
        num_anchors: Optional[int] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Picks and returns a valid collinear seed tile from candidate set.
        """
        if not candidate_set:
            return None

        all_walls: Set[Tuple[int, int]] = border_walls | placed_internal
        candidates: List[Tuple[int, int]] = list(candidate_set)
        random.shuffle(candidates)

        for cx, cy in candidates:
            if map_data.is_wall(cx, cy):
                candidate_set.discard((cx, cy))
                continue

            is_halo: bool = self.is_inner_halo_tile(
                cx, cy, width, height
            )
            if (
                num_anchors is not None and
                anchors_placed >= num_anchors and
                is_halo
            ):
                candidate_set.discard((cx, cy))
                continue

            if self.is_collinear_halo_seed(cx, cy, all_walls):
                candidate_set.discard((cx, cy))
                return (cx, cy)

            candidate_set.discard((cx, cy))

        return None

    def is_collinear_halo_seed(
        self,
        x: int,
        y: int,
        all_walls: Set[Tuple[int, int]]
    ) -> bool:
        """
        Checks if (x, y) has valid collinear wall neighbors in 3x3 grid.
        """
        neighbors = [
            (x + dx, y + dy)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dx != 0 or dy != 0) and (x + dx, y + dy) in all_walls
        ]

        num_walls: int = len(neighbors)
        if num_walls == 1:
            nx, ny = neighbors[0]
            return (nx == x) or (ny == y)

        if num_walls not in (2, 3):
            return False

        same_row: bool = all(
            ny == neighbors[0][1] for _, ny in neighbors
        )
        same_col: bool = all(
            nx == neighbors[0][0] for nx, _ in neighbors
        )

        if same_row:
            xs = [nx for nx, _ in neighbors]
            if max(xs) - min(xs) == len(xs) - 1:
                return True

        if same_col:
            ys = [ny for _, ny in neighbors]
            if max(ys) - min(ys) == len(ys) - 1:
                return True

        return False

    def register_internal_wall_tile(
        self,
        wx: int,
        wy: int,
        width: int,
        height: int,
        map_data: MapData,
        candidate_set: Set[Tuple[int, int]]
    ) -> None:
        """
        Adds orthogonal floor neighbors of placed wall to candidate set.
        """
        candidate_set.discard((wx, wy))
        cardinal_moves: Tuple[Tuple[int, int], ...] = (
            (0, -1), (0, 1), (-1, 0), (1, 0)
        )
        for dx, dy in cardinal_moves:
            nx: int = wx + dx
            ny: int = wy + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                if not map_data.is_wall(nx, ny):
                    candidate_set.add((nx, ny))
