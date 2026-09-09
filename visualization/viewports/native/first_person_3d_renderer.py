"""
First-Person 3D Raycasting Perspective Renderer for selected agents.
"""

import math
from typing import Tuple, Dict, Any
import pygame

import config
from core.map_data import MapData
from visualization.viewports.native.state_resolver import ViewportFrameState


class FirstPerson3DRenderer:
    """
    Renders a full-screen pseudo-3D perspective projection from the agent's viewpoint.
    """

    def render_3d_view(
        self,
        surface: pygame.Surface,
        gen_data: Dict[str, Any],
        frame_state: ViewportFrameState
    ) -> None:
        rx, ry, rw, rh = 0, 0, config.VIRTUAL_WIDTH, config.VIRTUAL_HEIGHT
        clip_rect = pygame.Rect(rx, ry, rw, rh)
        surface.set_clip(clip_rect)

        horizon_y = ry + (rh // 2)
        sky_rect = pygame.Rect(rx, ry, rw, horizon_y - ry)
        floor_rect = pygame.Rect(rx, horizon_y, rw, (ry + rh) - horizon_y)

        pygame.draw.rect(surface, (15, 18, 25), sky_rect)
        pygame.draw.rect(surface, (30, 35, 45), floor_rect)

        map_w = gen_data.get("map_width", 24)
        map_h = gen_data.get("map_height", 18)
        map_data = MapData(map_w, map_h, gen_data["start_pos"], gen_data["exit_pos"])
        map_data.decode_bitmask(gen_data["bitmask_chunks"])

        ox, oy = frame_state.x, frame_state.y
        heading = frame_state.heading - math.pi
        fov = math.radians(65.0)
        num_rays = rw // 3

        half_fov = fov / 2.0
        start_angle = heading - half_fov
        angle_step = fov / float(num_rays)
        col_width = max(1, rw // num_rays)
        max_dist = 12.0

        for i in range(num_rays):
            ray_angle = start_angle + (i * angle_step)
            dist = self._cast_dda_distance(ox, oy, ray_angle, max_dist, map_data)
            
            corrected_dist = dist * math.cos(ray_angle - heading)
            corrected_dist = max(0.1, corrected_dist)

            wall_height = int(rh / corrected_dist)
            wall_top = horizon_y - (wall_height // 2)
            wall_bot = horizon_y + (wall_height // 2)

            # Safely clamp RGB values within valid range [0, 255]
            shade = max(20, min(200, int(220 * (1.0 - (corrected_dist / max_dist)))))
            wall_color = (
                max(0, min(255, shade)),
                max(0, min(255, shade + 12)),
                max(0, min(255, shade + 30))
            )

            slice_rect = pygame.Rect(rx + (i * col_width), max(ry, wall_top), col_width, min(rh, wall_bot - wall_top))
            pygame.draw.rect(surface, wall_color, slice_rect)

        pygame.draw.rect(surface, config.COLOR_VIEWPORT_HIGHLIGHT, (rx, ry, rw, rh), 3)
        font = pygame.font.SysFont("monospace", 16, bold=True)
        lbl = font.render(f"[FULLSCREEN 3D FIRST-PERSON MODE] AGENT #{frame_state.cand_idx}", True, config.COLOR_VIEWPORT_HIGHLIGHT)
        surface.blit(lbl, (rx + 16, ry + 16))

        surface.set_clip(None)

    def _cast_dda_distance(self, ox: float, oy: float, angle: float, max_dist: float, map_data: MapData) -> float:
        dir_x = math.cos(angle)
        dir_y = math.sin(angle)
        
        eps = 1e-9
        if abs(dir_x) < eps: dir_x = eps
        if abs(dir_y) < eps: dir_y = eps

        tx = int(math.floor(ox))
        ty = int(math.floor(oy))

        step_x = 1 if dir_x > 0 else -1
        step_y = 1 if dir_y > 0 else -1

        t_delta_x = abs(1.0 / dir_x)
        t_delta_y = abs(1.0 / dir_y)

        t_max_x = (float(tx + 1) - ox) * t_delta_x if dir_x > 0 else (ox - float(tx)) * t_delta_x
        t_max_y = (float(ty + 1) - oy) * t_delta_y if dir_y > 0 else (oy - float(ty)) * t_delta_y

        dist = 0.0
        while dist < max_dist:
            if t_max_x < t_max_y:
                dist = t_max_x
                t_max_x += t_delta_x
                tx += step_x
            else:
                dist = t_max_y
                t_max_y += t_delta_y
                ty += step_y

            if map_data.is_wall(tx, ty):
                return dist

        return max_dist
