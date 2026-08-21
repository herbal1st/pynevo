"""
Renders background tilemap layouts for sub-viewports.
"""

from typing import Tuple, Optional
import pygame

import config
from core.map_data import MapData
from entities.map_profile_registry import (
    MapProfileRegistry,
    ResolvedMapProfile,
)
from visualization.map_renderer import MapRenderer
from visualization.camera_projection import CameraProjection


class ViewportTileRenderer:
    """
    Renders tilemap backgrounds in Map-Centered and Player-Centered views.
    """

    def __init__(self) -> None:
        """
        Initializes map renderer and profile registry.
        """
        self.map_renderer: MapRenderer = MapRenderer()
        self.map_registry: MapProfileRegistry = MapProfileRegistry()
        self.map_profile: ResolvedMapProfile = (
            self.map_registry.get_profile(config.ACTIVE_MAP_PROFILE)
        )

    def draw_tiles(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        map_data: MapData,
        gen_idx: int,
        cx: float,
        cy: float,
        is_player_centered: bool,
        is_zoomed: bool,
        rows: int,
        cols: int,
        player_zoom: float = config.PLAYER_CAMERA_ZOOM,
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Renders tile background and returns (tile_size, origin_pixel).
        """
        rx, ry, rw, rh = rect

        tile_size: float = CameraProjection.calculate_tile_size(
            rw,
            rh,
            map_data.width,
            map_data.height,
            is_player_centered,
            is_zoomed,
            rows,
            cols,
            player_zoom=player_zoom,
            map_profile=self.map_profile,
        )

        origin_pixel: Tuple[int, int] = (
            CameraProjection.calculate_origin_pixel(
                rx, ry, rw, rh, cx, cy, tile_size, is_player_centered
            )
        )

        if is_player_centered:
            self._draw_player_centered_tiles(
                surface, rect, map_data, cx, cy, tile_size
            )
        else:
            bg_surf = self.map_renderer.get_rendered_map_surface(
                map_data, gen_idx, rw, rh
            )
            surface.blit(bg_surf, (rx, ry))

        return tile_size, origin_pixel

    def _draw_player_centered_tiles(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        map_data: MapData,
        cx: float,
        cy: float,
        tile_size: float,
    ) -> None:
        """
        Renders visible tiles dynamically centered around player position.
        """
        rx, ry, rw, rh = rect
        center_px: float = float(rx) + (float(rw) / 2.0)
        center_py: float = float(ry) + (float(rh) / 2.0)

        for y in range(map_data.height):
            for x in range(map_data.width):
                t_rect = (
                    CameraProjection.calculate_player_centered_tile_rect(
                        x, y, cx, cy, center_px, center_py, tile_size
                    )
                )

                if (
                    t_rect[0] + t_rect[2] < rx
                    or t_rect[0] > rx + rw
                    or t_rect[1] + t_rect[3] < ry
                    or t_rect[1] > ry + rh
                ):
                    continue

                if (x, y) == map_data.start_pos:
                    pygame.draw.rect(surface, config.COLOR_START, t_rect)
                elif (x, y) == map_data.exit_pos:
                    pygame.draw.rect(surface, config.COLOR_EXIT, t_rect)
                elif map_data.is_wall(x, y):
                    pygame.draw.rect(surface, config.COLOR_WALL, t_rect)
                    pygame.draw.rect(
                        surface, config.COLOR_WALL_BORDER, t_rect, 1
                    )
                else:
                    pygame.draw.rect(surface, config.COLOR_FLOOR, t_rect)
                    pygame.draw.rect(
                        surface, config.COLOR_FLOOR_BORDER, t_rect, 1
                    )
