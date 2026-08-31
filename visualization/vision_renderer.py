"""
Renders candidate vision arc fans and directional heading indicators.
"""

import math
from typing import List, Tuple
import pygame

import config
from core.map_data import MapData
from entities.agent_profile_registry import (
    AgentProfileRegistry,
    ResolvedAgentProfile
)
from perception.vision_arc import VisionArcSampler
from utils.surface_utils import create_alpha_surface


class VisionRenderer:
    """
    Renders visual perception cone polygons and translucent heading lines.
    """

    def __init__(self, w: int, h: int) -> None:
        """
        Initializes vision arc sampler and alpha scratchpad surface.
        """
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.profile: ResolvedAgentProfile = self.registry.get_profile(
            config.ACTIVE_AGENT_PROFILE
        )
        self.sampler: VisionArcSampler = VisionArcSampler(
            num_rays=self.profile.vision_rays,
            arc_angle_deg=self.profile.vision_arc_angle,
            max_dist=self.profile.vision_max_dist
        )
        self.scratchpad: pygame.Surface = create_alpha_surface(w, h)

    def draw_vision_arc(
        self,
        surface: pygame.Surface,
        rx: int,
        ry: int,
        rw: int,
        rh: int,
        cx: float,
        cy: float,
        heading: float,
        origin_pixel: Tuple[int, int],
        tile_size: float,
        is_camera_centered: bool,
        map_data: MapData,
        radius_ratio: float = 0.25
    ) -> None:
        """
        Renders semi-transparent vision arc cone, ray lines, & heading line.
        """
        center_px: float = float(rx) + (float(rw) / 2.0)
        center_py: float = float(ry) + (float(rh) / 2.0)

        rel_origin: Tuple[int, int] = (
            origin_pixel[0] - rx, origin_pixel[1] - ry
        )

        cone_points: List[Tuple[int, int]] = [rel_origin]
        ray_endpoints: List[Tuple[int, int]] = []

        for rel_angle in self.sampler.relative_angles:
            ray_angle: float = heading + rel_angle
            wall_prox, _ = self.sampler._cast_single_ray(
                cx, cy, ray_angle, map_data
            )
            dist_tiles: float = (
                (1.0 - wall_prox) * self.profile.vision_max_dist
            )
            ex: float = cx + (math.cos(ray_angle) * dist_tiles)
            ey: float = cy + (math.sin(ray_angle) * dist_tiles)

            if is_camera_centered:
                px_e: int = int(round(center_px + (ex - cx) * tile_size))
                py_e: int = int(round(center_py + (ey - cy) * tile_size))
            else:
                px_e = int(rx + (ex * tile_size))
                py_e = int(ry + (ey * tile_size))

            rel_end: Tuple[int, int] = (px_e - rx, py_e - ry)
            cone_points.append(rel_end)
            ray_endpoints.append(rel_end)

        self.scratchpad.fill((0, 0, 0, 0))

        if len(cone_points) > 2:
            pygame.draw.polygon(
                self.scratchpad,
                self.profile.skin.color_vision_arc,
                cone_points
            )

        for rel_end in ray_endpoints:
            pygame.draw.line(
                self.scratchpad,
                self.profile.skin.color_vision_rays,
                rel_origin,
                rel_end,
                1
            )

        body_radius_px: float = tile_size * radius_ratio
        line_ext_px: float = (
            self.profile.skin.heading_line_length * tile_size
        )
        head_line_len: float = body_radius_px + line_ext_px

        hx: int = int(
            rel_origin[0] + (math.cos(heading) * head_line_len)
        )
        hy: int = int(
            rel_origin[1] + (math.sin(heading) * head_line_len)
        )

        pygame.draw.line(
            self.scratchpad,
            self.profile.skin.color_heading_line,
            rel_origin,
            (hx, hy),
            self.profile.skin.heading_line_width
        )

        surface.blit(
            self.scratchpad, (rx, ry), area=pygame.Rect(0, 0, rw, rh)
        )
