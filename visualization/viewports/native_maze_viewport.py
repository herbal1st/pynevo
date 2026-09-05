"""
Native PyNevo maze environment sub-viewport renderer facade.
"""

from typing import Tuple, Dict, Any
import pygame

import config
from core.map_data import MapData
from visualization.vision_renderer import VisionRenderer
from visualization.viewports.adapter_interface import IViewportAdapter
from visualization.viewports.native.state_resolver import (
    ViewportStateResolver,
)
from visualization.viewports.native.tile_renderer import (
    ViewportTileRenderer,
)
from visualization.viewports.native.avatar_renderer import (
    ViewportAvatarRenderer,
)
from visualization.viewports.native.hud_overlay_renderer import (
    ViewportHUDOverlayRenderer,
)


class NativeMazeViewport(IViewportAdapter):
    """
    Orchestrates native sub-viewport and full-population overlay rendering.
    """

    def __init__(self, canvas_w: int, canvas_h: int) -> None:
        self.state_resolver: ViewportStateResolver = ViewportStateResolver()
        self.tile_renderer: ViewportTileRenderer = ViewportTileRenderer()
        self.vision_renderer: VisionRenderer = VisionRenderer(canvas_w, canvas_h)
        self.avatar_renderer: ViewportAvatarRenderer = ViewportAvatarRenderer()
        self.hud_renderer: ViewportHUDOverlayRenderer = ViewportHUDOverlayRenderer(
            canvas_w, canvas_h
        )

    def render_viewport(
        self,
        surface: pygame.Surface,
        gen_data: Dict[str, Any],
        cand_idx: int,
        active_step: int,
        rect: Tuple[int, int, int, int],
        is_selected: bool,
        is_zoomed: bool,
        is_camera_centered: bool,
        rows: int,
        cols: int,
    ) -> None:
        frame_state = self.state_resolver.resolve_frame_state(
            gen_data, cand_idx, active_step
        )
        if frame_state is None:
            return

        rx, ry, rw, rh = rect
        map_w: int = gen_data.get("map_width", gen_data.get("map_size", 20))
        map_h: int = gen_data.get("map_height", gen_data.get("map_size", 20))
        map_data = MapData(
            map_w, map_h, gen_data["start_pos"], gen_data["exit_pos"]
        )
        map_data.decode_bitmask(gen_data["bitmask_chunks"])
        map_data.target_sequence = list(
            gen_data.get("target_sequence", [gen_data["exit_pos"]])
        )

        clip_rect = pygame.Rect(rx, ry, rw, rh)
        surface.set_clip(clip_rect)
        pygame.draw.rect(surface, config.COLOR_BG, rect)

        base_sub_w: float = float(surface.get_width()) / float(cols)
        ui_scale: float = (float(rw) / base_sub_w) * 0.5 if is_zoomed else 1.0
        gen_idx: int = int(gen_data.get("generation", 0))

        tile_size, origin_pixel = self.tile_renderer.draw_tiles(
            surface,
            rect,
            map_data,
            gen_idx,
            frame_state.x,
            frame_state.y,
            is_camera_centered,
            is_zoomed,
            rows,
            cols,
            camera_zoom=self.tile_renderer.skin_profile.camera_zoom,
            target_override=frame_state.target_pos,
            checkpoint_override=frame_state.checkpoint_pos
        )

        self.vision_renderer.draw_vision_arc(
            surface,
            rx, ry, rw, rh,
            frame_state.x, frame_state.y, frame_state.heading,
            origin_pixel, tile_size, is_camera_centered, map_data
        )

        self.avatar_renderer.draw_avatar(
            surface, origin_pixel, tile_size, frame_state, is_selected, ui_scale, active_step
        )

        self.hud_renderer.draw_hud_overlays(surface, rect, frame_state, ui_scale)
        surface.set_clip(None)
        self.hud_renderer.draw_viewport_borders(surface, rect, frame_state, is_selected)

    def render_overlay_viewport(
        self,
        surface: pygame.Surface,
        gen_data: Dict[str, Any],
        selected_cand_idx: int,
        active_step: int,
        rect: Tuple[int, int, int, int],
        is_camera_centered: bool
    ) -> None:
        """
        Renders all candidates in the population simultaneously overlaid onto one single map.
        """
        telemetry = gen_data.get("telemetry", None)
        if telemetry is None:
            return

        rx, ry, rw, rh = rect
        pop_size: int = int(telemetry.shape[1])
        safe_sel_idx: int = min(max(0, selected_cand_idx), pop_size - 1)

        focal_state = self.state_resolver.resolve_frame_state(
            gen_data, safe_sel_idx, active_step
        )
        if focal_state is None:
            return

        map_w: int = int(gen_data.get("map_width", 24))
        map_h: int = int(gen_data.get("map_height", 18))
        map_data = MapData(
            map_w, map_h, gen_data["start_pos"], gen_data["exit_pos"]
        )
        map_data.decode_bitmask(gen_data["bitmask_chunks"])
        map_data.target_sequence = list(
            gen_data.get("target_sequence", [gen_data["exit_pos"]])
        )

        clip_rect = pygame.Rect(rx, ry, rw, rh)
        surface.set_clip(clip_rect)
        pygame.draw.rect(surface, config.COLOR_BG, rect)

        gen_idx: int = int(gen_data.get("generation", 0))

        # 1. Draw single map background once
        tile_size, focal_pixel = self.tile_renderer.draw_tiles(
            surface,
            rect,
            map_data,
            gen_idx,
            focal_state.x,
            focal_state.y,
            is_camera_centered,
            is_zoomed=True,
            rows=1,
            cols=1,
            camera_zoom=self.tile_renderer.skin_profile.camera_zoom,
            target_override=focal_state.target_pos,
            checkpoint_override=focal_state.checkpoint_pos
        )

        center_px: float = float(rx) + (float(rw) / 2.0)
        center_py: float = float(ry) + (float(rh) / 2.0)

        alive_count: int = 0
        exits_count: int = 0

        # 2. Pass 1: Render all background swarm candidates
        for c_idx in range(pop_size):
            if c_idx == safe_sel_idx:
                continue

            c_state = self.state_resolver.resolve_frame_state(
                gen_data, c_idx, active_step
            )
            if c_state is None:
                continue

            if c_state.is_alive:
                alive_count += 1
            if c_state.reached_exit:
                exits_count += 1

            if is_camera_centered:
                px = int(round(center_px + (c_state.x - focal_state.x) * tile_size))
                py = int(round(center_py + (c_state.y - focal_state.y) * tile_size))
            else:
                px = int(rx + (c_state.x * tile_size))
                py = int(ry + (c_state.y * tile_size))

            if not (rx - 25 <= px <= rx + rw + 25 and ry - 25 <= py <= ry + rh + 25):
                continue

            self.avatar_renderer.draw_avatar(
                surface,
                (px, py),
                tile_size,
                c_state,
                is_selected=False,
                ui_scale=0.85,
                active_step=active_step
            )

        # 3. Pass 2: Render focal champion on top
        if focal_state.is_alive:
            alive_count += 1
        if focal_state.reached_exit:
            exits_count += 1

        self.vision_renderer.draw_vision_arc(
            surface,
            rx, ry, rw, rh,
            focal_state.x, focal_state.y, focal_state.heading,
            focal_pixel, tile_size, is_camera_centered, map_data
        )

        self.avatar_renderer.draw_avatar(
            surface,
            focal_pixel,
            tile_size,
            focal_state,
            is_selected=True,
            ui_scale=1.1,
            active_step=active_step
        )

        # 4. HUD Overlay
        self.hud_renderer.draw_hud_overlays(
            surface, rect, focal_state, ui_scale=1.0, show_scorecard=False
        )

        # 5. Live Swarm Status Badge
        font_badge = self.hud_renderer.font_manager.get_font(12, bold=True)
        badge_text = f"SWARM OVERLAY: {pop_size} AGENTS  |  ALIVE: {alive_count}  |  EXITS: {exits_count}"
        badge_surf = font_badge.render(badge_text, True, config.COLOR_VIEWPORT_HIGHLIGHT)
        badge_bg = pygame.Surface(
            (badge_surf.get_width() + 12, badge_surf.get_height() + 6),
            pygame.SRCALPHA
        )
        badge_bg.fill((15, 15, 20, 210))
        surface.blit(badge_bg, (rx + 8, ry + rh - badge_bg.get_height() - 8))
        surface.blit(badge_surf, (rx + 14, ry + rh - badge_bg.get_height() - 5))

        surface.set_clip(None)
        pygame.draw.rect(surface, config.COLOR_VIEWPORT_HIGHLIGHT, rect, 1)
