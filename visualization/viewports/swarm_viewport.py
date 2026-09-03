"""
Single-maze swarm viewport rendering all candidate agents simultaneously.
"""

import math
from typing import Tuple, Dict, Any, List
import pygame

import config
from core.map_data import MapData
from visualization.map_renderer import MapRenderer
from visualization.vision_renderer import VisionRenderer
from visualization.viewports.native.state_resolver import (
    ViewportStateResolver,
    ViewportFrameState
)
from visualization.viewports.native.avatar_renderer import ViewportAvatarRenderer
from visualization.viewports.native.hud_overlay_renderer import ViewportHUDOverlayRenderer
from utils.font_manager import FontManager


class SwarmMazeViewport:
    """
    Renders one large shared maze with the entire population swarm overlay.
    """

    def __init__(self, canvas_w: int, canvas_h: int) -> None:
        self.state_resolver: ViewportStateResolver = ViewportStateResolver()
        self.map_renderer: MapRenderer = MapRenderer()
        self.vision_renderer: VisionRenderer = VisionRenderer(canvas_w, canvas_h)
        self.avatar_renderer: ViewportAvatarRenderer = ViewportAvatarRenderer()
        self.hud_renderer: ViewportHUDOverlayRenderer = ViewportHUDOverlayRenderer(canvas_w, canvas_h)
        self.font_manager: FontManager = FontManager()

    def render_swarm(
        self,
        surface: pygame.Surface,
        gen_data: Dict[str, Any],
        selected_cand_idx: int,
        active_step: int,
        rect: Tuple[int, int, int, int],
        is_camera_centered: bool = False
    ) -> None:
        rx, ry, rw, rh = rect
        map_w: int = int(gen_data.get("map_width", 24))
        map_h: int = int(gen_data.get("map_height", 18))
        gen_idx: int = int(gen_data.get("generation", 0))

        map_data = MapData(map_w, map_h, gen_data["start_pos"], gen_data["exit_pos"])
        map_data.decode_bitmask(gen_data["bitmask_chunks"])

        telemetry = gen_data.get("telemetry", None)
        if telemetry is None:
            return

        max_f: int = int(telemetry.shape[0])
        pop_s: int = int(telemetry.shape[1])
        safe_step: int = max(0, min(active_step, max_f - 1))

        # Resolve selected candidate state for focal camera / HUD
        selected_state = self.state_resolver.resolve_frame_state(
            gen_data, selected_cand_idx, safe_step
        )
        if selected_state is None:
            return

        clip_rect = pygame.Rect(rx, ry, rw, rh)
        surface.set_clip(clip_rect)
        pygame.draw.rect(surface, config.COLOR_BG, rect)

        # Calculate map scale
        tile_size: float = min(float(rw) / float(map_w), float(rh) / float(map_h))
        map_pixel_w = int(map_w * tile_size)
        map_pixel_h = int(map_h * tile_size)
        map_offset_x = rx + (rw - map_pixel_w) // 2
        map_offset_y = ry + (rh - map_pixel_h) // 2

        # 1. Draw Shared Background Maze
        bg_surf = self.map_renderer.get_rendered_map_surface(
            map_data, gen_idx, map_pixel_w, map_pixel_h
        )
        surface.blit(bg_surf, (map_offset_x, map_offset_y))

        # 2. Draw Selected Candidate Vision Arc (Only for active focus)
        focus_px = int(map_offset_x + (selected_state.x * tile_size))
        focus_py = int(map_offset_y + (selected_state.y * tile_size))
        self.vision_renderer.draw_vision_arc(
            surface,
            rx, ry, rw, rh,
            selected_state.x, selected_state.y, selected_state.heading,
            (focus_px, focus_py),
            tile_size,
            is_camera_centered=False,
            map_data=map_data
        )

        # 3. Draw All Swarm Agents on the Same Maze
        alive_count = 0
        exit_count = 0
        dead_count = 0

        # Draw non-selected background swarm agents first
        for c_idx in range(pop_s):
            row = telemetry[safe_step, c_idx]
            cx: float = float(row[0])
            cy: float = float(row[1])
            heading: float = float(row[2])
            hp: float = float(row[3])
            is_alive: bool = bool(row[6] > 0.5)
            reached_exit: bool = bool(row[7] > 0.5)

            if reached_exit:
                exit_count += 1
            elif is_alive:
                alive_count += 1
            else:
                dead_count += 1

            if c_idx == selected_cand_idx:
                continue  # Draw selected agent on top

            px = int(map_offset_x + (cx * tile_size))
            py = int(map_offset_y + (cy * tile_size))
            radius = max(2, int(tile_size * 0.22))

            # Swarm Body Colors
            if reached_exit:
                color = (50, 220, 100, 200)    # Green for solvers
            elif not is_alive:
                color = (180, 40, 40, 100)     # Dim red for dead
            else:
                color = (240, 160, 40, 160)    # Orange for active swarm

            pygame.draw.circle(surface, color[:3], (px, py), radius)

            # Draw directional pointer for living agents
            if is_alive and not reached_exit:
                hx = int(px + math.cos(heading) * (radius + 3))
                hy = int(py + math.sin(heading) * (radius + 3))
                pygame.draw.line(surface, (255, 255, 255, 180), (px, py), (hx, hy), 1)

        # 4. Draw Selected Focus Candidate (Highlighted with Face & Status Ring)
        self.avatar_renderer.draw_avatar(
            surface,
            (focus_px, focus_py),
            tile_size,
            selected_state,
            is_selected=True,
            ui_scale=1.0,
            active_step=safe_step
        )

        # 5. Swarm Telemetry HUD Tag (Top Left of Viewport)
        font_hud = self.font_manager.get_font(12, bold=True)
        hud_str = f"SWARM: {pop_s} | ALIVE: {alive_count} | EXITS: {exit_count} | CRASHED: {dead_count}"
        hud_surf = font_hud.render(hud_str, True, config.COLOR_VIEWPORT_HIGHLIGHT[:3])

        bg_rect = pygame.Rect(rx + 8, ry + 8, hud_surf.get_width() + 12, hud_surf.get_height() + 6)
        pygame.draw.rect(surface, (15, 15, 20, 200), bg_rect, border_radius=4)
        pygame.draw.rect(surface, config.COLOR_WALL_BORDER, bg_rect, 1, border_radius=4)
        surface.blit(hud_surf, (rx + 14, ry + 11))

        # Selected Candidate Focus Card (Bottom Right)
        score_surf = font_hud.render(
            f"FOCUS AGENT #{selected_cand_idx} | SCORE: {selected_state.score_val}",
            True, config.COLOR_VIEWPORT_HIGHLIGHT[:3]
        )
        s_rect = score_surf.get_rect(bottomright=(rx + rw - 12, ry + rh - 10))
        bg_card = pygame.Rect(s_rect.x - 6, s_rect.y - 3, s_rect.width + 12, s_rect.height + 6)
        pygame.draw.rect(surface, (15, 15, 20, 200), bg_card, border_radius=4)
        pygame.draw.rect(surface, config.COLOR_WALL_BORDER, bg_card, 1, border_radius=4)
        surface.blit(score_surf, s_rect)

        surface.set_clip(None)
        pygame.draw.rect(surface, config.COLOR_WALL_BORDER[:3], rect, 1)
