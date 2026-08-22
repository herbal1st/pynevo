"""
Shared utilities for grid halo detection and snake corridor wall capacity.
"""

from typing import Set, Tuple


def is_inner_halo_tile(
    x: int, y: int, width: int, height: int
) -> bool:
    """
    Checks if a tile coordinate (x, y) lies on the inner halo.
    """
    return x == 1 or x == width - 2 or y == 1 or y == height - 2


def get_inner_halo_tiles(
    width: int, height: int
) -> Set[Tuple[int, int]]:
    """
    Returns a set of all inner halo tile coordinates.
    """
    halo_tiles: Set[Tuple[int, int]] = set()
    for x in range(1, width - 1):
        halo_tiles.add((x, 1))
        halo_tiles.add((x, height - 2))
    for y in range(1, height - 1):
        halo_tiles.add((1, y))
        halo_tiles.add((width - 2, y))
    return halo_tiles


def calculate_snake_corridor_max_walls(
    width: int, height: int
) -> int:
    """
    Calculates maximum placeable walls using snake corridor capacity math.
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
