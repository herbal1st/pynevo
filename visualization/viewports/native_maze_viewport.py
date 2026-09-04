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
    Orchestrates native sub-viewport rendering via modular sub-renderers.
    """

    def __init__(self, canvas_w: int, canvas_h: int) -> None:
        """
        Initializes sub-renderers and vision arc sampler.
        """
        self.state_resolver: ViewportStateResolver = (
            ViewportStateResolver()
        )
        self.tile_renderer: ViewportTileRenderer = ViewportTileRenderer()
        self.vision_renderer: VisionRenderer = VisionRenderer(
            canvas_w, canvas_h
        )
        self.avatar_renderer: ViewportAvatarRenderer = (
            ViewportAvatarRenderer()
        )
        self.hud_renderer: ViewportHUDOverlayRenderer = (
            ViewportHUDOverlayRenderer(canvas_w, canvas_h)
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
        """
        Renders a single native candidate sub-viewport into rect bounds.
        """
        frame_state = self.state_resolver.resolve_frame_state(
            gen_data, cand_idx, active_step
        )
        if frame_state is None:
            return

        rx, ry, rw, rh = rect
        map_w: int = gen_data.get(
            "map_width", gen_data.get("map_size", 20)
        )
        map_h: int = gen_data.get(
            "map_height", gen_data.get("map_size", 20)
        )
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
        ui_scale: float = (
            (float(rw) / base_sub_w) * 0.5 if is_zoomed else 1.0
        )

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
            rx,
            ry,
            rw,
            rh,
            frame_state.x,
            frame_state.y,
            frame_state.heading,
            origin_pixel,
            tile_size,
            is_camera_centered,
            map_data,
        )

        self.avatar_renderer.draw_avatar(
            surface,
            origin_pixel,
            tile_size,
            frame_state,
            is_selected,
            ui_scale,
            active_step,
        )

        self.hud_renderer.draw_hud_overlays(
            surface, rect, frame_state, ui_scale
        )

        surface.set_clip(None)

        self.hud_renderer.draw_viewport_borders(
            surface, rect, frame_state, is_selected
        )
