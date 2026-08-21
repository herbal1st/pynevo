"""
Serpentine capacity math and active stem extension clearance solver.
"""

from typing import List, Tuple, Set


class StemExtensionSolver:
    """
    Solves wall capacity limits and validates active stem step clearance.
    """

    def calculate_max_possible_walls(
        self,
        width: int,
        height: int
    ) -> int:
        """
        Calculates maximum placeable walls using snake corridor formula.
        """
        rem_w: int = width - 2
        rem_h: int = height - 2

        if rem_w <= 0 or rem_h <= 0:
            return 0

        max_dim: int = max(rem_w, rem_h)
        min_dim: int = min(rem_w, rem_h)

        wall_lines: int = (max_dim - 1) // 2
        bookshelf_walls: int = wall_lines * min_dim
        snake_openings: int = wall_lines

        max_walls: int = bookshelf_walls - snake_openings
        return max(1, max_walls)

    def is_valid_stem_extension(
        self,
        nx: int,
        ny: int,
        active_stem: List[Tuple[int, int]],
        border_walls: Set[Tuple[int, int]],
        unrelated_walls: Set[Tuple[int, int]]
    ) -> bool:
        """
        Validates extending active stem tile including 90-degree turns.
        """
        allowed_stem: Set[Tuple[int, int]] = set(active_stem[-2:])
        effective_unrelated: Set[Tuple[int, int]] = (
            (border_walls | unrelated_walls) - set(active_stem)
        )

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                bx: int = nx + dx
                by: int = ny + dy

                if (bx, by) in effective_unrelated:
                    return False

                if (
                    (bx, by) in active_stem and
                    (bx, by) not in allowed_stem
                ):
                    return False

        return True
