"""
Static tilemap surface pre-rendering and background surface caching.
"""

from typing import Dict, Tuple
import pygame

import config
from core.map_data import MapData


class MapRenderer:
    """
    Renders static maze tilemap layouts into cached background Surfaces.
    """

    def __init__(self) -> None:
        """
        Initializes empty map surface cache.
        """
        self._map_cache: Dict[Tuple[int, int, int], pygame.Surface] = {}

    def get_rendered_map_surface(
        self,
        map_data: MapData,
        gen_idx: int,
        rw: int,
        rh: int
    ) -> pygame.Surface:
        """
        Retrieves cached pre-rendered map surface or renders background once.
        """
        cache_key: Tuple[int, int, int] = (gen_idx, rw, rh)
        if cache_key in self._map_cache:
            return self._map_cache[cache_key]

        map_surf: pygame.Surface = pygame.Surface((rw, rh))
        map_surf.fill(config.COLOR_BG)

        tile_size: float = min(
            float(rw) / float(map_data.width),
            float(rh) / float(map_data.height)
        )

        for y in range(map_data.height):
            for x in range(map_data.width):
                t_x: int = int(float(x) * tile_size)
                t_y: int = int(float(y) * tile_size)
                t_rect: Tuple[int, int, int, int] = (
                    t_x, t_y, int(tile_size) + 1, int(tile_size) + 1
                )

                if (x, y) == map_data.start_pos:
                    pygame.draw.rect(map_surf, config.COLOR_START, t_rect)
                elif (x, y) == map_data.exit_pos:
                    pygame.draw.rect(map_surf, config.COLOR_EXIT, t_rect)
                elif map_data.is_wall(x, y):
                    pygame.draw.rect(map_surf, config.COLOR_WALL, t_rect)
                    pygame.draw.rect(
                        map_surf, config.COLOR_WALL_BORDER, t_rect, 1
                    )
                else:
                    pygame.draw.rect(map_surf, config.COLOR_FLOOR, t_rect)
                    pygame.draw.rect(
                        map_surf, config.COLOR_FLOOR_BORDER, t_rect, 1
                    )

        self._map_cache[cache_key] = map_surf
        return map_surf

    def clear_cache(self) -> None:
        """
        Clears pre-rendered surface cache.
        """
        self._map_cache.clear()
