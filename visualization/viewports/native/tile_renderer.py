"""
Renders background tilemap layouts for sub-viewports and endless maps.
"""

import math
from typing import Tuple, Optional
import pygame

import config
from core.map_data import MapData
from entities.map_profile_registry import (
    MapProfileRegistry,
    ResolvedMapProfile,
)
from entities.skin_profile_registry import (
    SkinProfileRegistry,
    ResolvedSkinProfile,
)
from world.tile_registry import TileRegistry
from world.chunk_manager import ChunkManager
from visualization.map_renderer import MapRenderer


class ViewportTileRenderer:
    """
    Renders tilemap backgrounds in Map-Centered, Camera, and Endless views.
    """

    def __init__(self) -> None:
        """
        Initializes map renderer and profile registries.
        """
        self.map_renderer: MapRenderer = MapRenderer()
        self.map_registry: MapProfileRegistry = MapProfileRegistry()
        self.skin_registry: SkinProfileRegistry = SkinProfileRegistry()
        self.tile_registry: TileRegistry = TileRegistry()

        self.map_profile: ResolvedMapProfile = (
            self.map_registry.get_profile(config.ACTIVE_MAP_PROFILE)
        )
        self.skin_profile: ResolvedSkinProfile = (
            self.skin_registry.get_skin("DEFAULT")
        )

    def draw_tiles(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        map_data: MapData,
        gen_idx: int,
        cx: float,
        cy: float,
        is_camera_centered: bool,
        is_zoomed: bool,
        rows: int,
        cols: int,
        camera_zoom: Optional[float] = None,
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Renders tile background and returns (tile_size, origin_pixel).
        """
        rx, ry, rw, rh = rect
        zoom_val: float = (
            camera_zoom if camera_zoom is not None
            else self.skin_profile.camera_zoom
        )

        if is_camera_centered:
            tile_size: float = (
                float(self.map_profile.tile_size) * zoom_val
            )
            center_px: float = float(rx) + (float(rw) / 2.0)
            center_py: float = float(ry) + (float(rh) / 2.0)
            self._draw_camera_centered_tiles(
                surface, rect, map_data, cx, cy, tile_size
            )
            origin_pixel = (int(round(center_px)), int(round(center_py)))
        else:
            tile_size = min(
                float(rw) / float(map_data.width),
                float(rh) / float(map_data.height),
            )
            bg_surf = self.map_renderer.get_rendered_map_surface(
                map_data, gen_idx, rw, rh
            )
            surface.blit(bg_surf, (rx, ry))
            origin_pixel = (
                int(rx + (cx * tile_size)),
                int(ry + (cy * tile_size)),
            )

        return tile_size, origin_pixel

    def draw_endless_tiles(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        chunk_manager: ChunkManager,
        focus_x: float,
        focus_y: float,
        tile_size_base: float = 10.0,
        camera_zoom: Optional[float] = None
    ) -> float:
        """
        Blits endless chunks using top-left anchor grid snapping.
        """
        rx, ry, rw, rh = rect
        zoom_val: float = (
            camera_zoom if camera_zoom is not None
            else self.skin_profile.camera_zoom
        )
        base_chunk_px: int = int(round(16 * tile_size_base))
        scaled_chunk_px: int = int(round(base_chunk_px * zoom_val))

        if scaled_chunk_px <= 0:
            return tile_size_base * zoom_val

        center_px: float = float(rx) + (float(rw) / 2.0)
        center_py: float = float(ry) + (float(rh) / 2.0)

        c_size: float = float(ChunkManager.CHUNK_SIZE)
        focus_cx: float = focus_x / c_size
        focus_cy: float = focus_y / c_size

        min_cx: int = math.floor(
            focus_cx - (float(rw) / (2.0 * float(scaled_chunk_px)))
        )
        max_cx: int = math.ceil(
            focus_cx + (float(rw) / (2.0 * float(scaled_chunk_px)))
        )
        min_cy: int = math.floor(
            focus_cy - (float(rh) / (2.0 * float(scaled_chunk_px)))
        )
        max_cy: int = math.ceil(
            focus_cy + (float(rh) / (2.0 * float(scaled_chunk_px)))
        )

        anchor_chk_x: int = int(
            round(
                center_px + (float(min_cx) - focus_cx) * scaled_chunk_px
            )
        )
        anchor_chk_y: int = int(
            round(
                center_py + (float(min_cy) - focus_cy) * scaled_chunk_px
            )
        )

        is_scaled: bool = (scaled_chunk_px != base_chunk_px)

        for cy in range(min_cy, max_cy + 1):
            rel_cy: int = cy - min_cy
            chk_y: int = anchor_chk_y + (rel_cy * scaled_chunk_px)

            for cx in range(min_cx, max_cx + 1):
                chunk = chunk_manager.get_chunk(cx, cy)
                if chunk is None or chunk.surface is None:
                    continue

                rel_cx: int = cx - min_cx
                chk_x: int = anchor_chk_x + (rel_cx * scaled_chunk_px)

                if is_scaled:
                    scaled_surf = pygame.transform.scale(
                        chunk.surface,
                        (scaled_chunk_px, scaled_chunk_px)
                    )
                    surface.blit(scaled_surf, (chk_x, chk_y))
                else:
                    surface.blit(chunk.surface, (chk_x, chk_y))

        return tile_size_base * zoom_val

    def _draw_camera_centered_tiles(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        map_data: MapData,
        cx: float,
        cy: float,
        tile_size: float,
    ) -> None:
        """
        Renders visible tiles dynamically centered around focal position.
        """
        rx, ry, rw, rh = rect
        center_px: float = float(rx) + (float(rw) / 2.0)
        center_py: float = float(ry) + (float(rh) / 2.0)

        for y in range(map_data.height):
            for x in range(map_data.width):
                t_x: int = int(
                    round(center_px + (float(x) - cx) * tile_size)
                )
                t_y: int = int(
                    round(center_py + (float(y) - cy) * tile_size)
                )
                t_rect = (t_x, t_y, int(tile_size) + 1, int(tile_size) + 1)

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
