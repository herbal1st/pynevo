"""
Renders candidate body sprite, face expressions, status ring, and solved arcs.
"""

import math
from typing import Tuple, Optional
import pygame

import config
from entities.agent_profile_registry import (
    AgentProfileRegistry,
    ResolvedAgentProfile,
)
from entities.skin_profile_registry import ResolvedSkinProfile
from utils.font_manager import FontManager
from utils.math_utils import calculate_spin_angle
from utils.color_utils import resolve_net_delta_color
from utils.surface_utils import scale_text_surface
from visualization.viewports.native.state_resolver import ViewportFrameState


class ViewportAvatarRenderer:
    """
    Renders candidate avatar circle, ASCII face, status ring, & solved arcs.
    """

    def __init__(self) -> None:
        """
        Initializes profile registry and font manager.
        """
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.profile: ResolvedAgentProfile = self.registry.get_profile(
            config.ACTIVE_AGENT_PROFILE
        )
        self.font_manager: FontManager = FontManager()

    def draw_avatar(
        self,
        surface: pygame.Surface,
        origin_pixel: Tuple[int, int],
        tile_size: float,
        frame_state: ViewportFrameState,
        is_selected: bool,
        ui_scale: float,
        active_step: int,
    ) -> None:
        """
        Renders body circle, face text, status ring, and counter-rotating arcs.
        """
        px, py = origin_pixel
        p_radius: int = max(
            3, int(tile_size * frame_state.radius_ratio)
        )

        active_skin: ResolvedSkinProfile = (
            frame_state.skin
            if frame_state.skin is not None
            else self.profile.skin
        )

        body_color = active_skin.color_body[:3]
        pygame.draw.circle(surface, body_color, (px, py), p_radius)

        if active_skin.show_status_ring:
            self._draw_status_ring(
                surface,
                px,
                py,
                p_radius,
                frame_state.reached_exit,
                frame_state.is_alive,
                frame_state.net_delta,
                active_step,
                active_skin=active_skin,
            )

        if active_skin.show_ascii_faces:
            font_norm = self.font_manager.get_font(
                max(10, int(12 * ui_scale))
            )
            raw_face = font_norm.render(
                frame_state.face_str,
                True,
                active_skin.color_text[:3],
            )
            target_side: int = max(
                2,
                int(
                    p_radius * 2 * active_skin.face_text_scale
                ),
            )
            scaled_face = scale_text_surface(raw_face, target_side)
            f_rect = scaled_face.get_rect(center=(px, py))
            surface.blit(scaled_face, f_rect)

    def _draw_status_ring(
        self,
        surface: pygame.Surface,
        px: int,
        py: int,
        p_radius: int,
        reached_exit: bool,
        is_alive: bool,
        net_delta: float,
        active_step: int,
        active_skin: Optional[ResolvedSkinProfile] = None,
    ) -> None:
        """
        Renders status ring or counter-rotating solved arcs.
        """
        skin: ResolvedSkinProfile = (
            active_skin if active_skin is not None else self.profile.skin
        )
        line_w: int = max(
            1, int(float(p_radius) * skin.status_ring_ratio)
        )

        if reached_exit:
            self._draw_solved_counter_rotating_arcs(
                surface, px, py, p_radius, line_w, active_step, active_skin=skin
            )
            return

        ring_color: Tuple[int, int, int] = (
            config.COLOR_FRAME_DEAD[:3]
            if not is_alive
            else resolve_net_delta_color(net_delta)
        )

        pygame.draw.circle(
            surface, ring_color, (px, py), p_radius, line_w
        )

    def _draw_solved_counter_rotating_arcs(
        self,
        surface: pygame.Surface,
        px: int,
        py: int,
        p_radius: int,
        line_w: int,
        active_step: int,
        active_skin: Optional[ResolvedSkinProfile] = None,
    ) -> None:
        """
        Renders concentric counter-rotating arcs for exit solvers.
        """
        skin: ResolvedSkinProfile = (
            active_skin if active_skin is not None else self.profile.skin
        )
        ratio: float = skin.status_ring_ratio
        r_inner: int = p_radius
        r_outer: int = max(2, int(float(p_radius) * (1.0 + ratio)))

        rect_inner = pygame.Rect(
            px - r_inner, py - r_inner, r_inner * 2, r_inner * 2
        )
        rect_outer = pygame.Rect(
            px - r_outer, py - r_outer, r_outer * 2, r_outer * 2
        )

        color: Tuple[int, int, int] = skin.solved_arc_color[:3]
        arc_rad: float = math.radians(skin.solved_arc_segments)

        spin_angle: float = calculate_spin_angle(
            active_step, speed_rate=0.15
        )

        start_cw1: float = spin_angle
        end_cw1: float = spin_angle + arc_rad
        start_cw2: float = spin_angle + math.pi
        end_cw2: float = spin_angle + math.pi + arc_rad

        start_ccw1: float = -spin_angle + (math.pi / 2.0)
        end_ccw1: float = start_ccw1 + arc_rad
        start_ccw2: float = -spin_angle + (3.0 * math.pi / 2.0)
        end_ccw2: float = start_ccw2 + arc_rad

        pygame.draw.arc(
            surface, color, rect_outer, start_cw1, end_cw1, line_w
        )
        pygame.draw.arc(
            surface, color, rect_outer, start_cw2, end_cw2, line_w
        )

        pygame.draw.arc(
            surface, color, rect_inner, start_ccw1, end_ccw1, line_w
        )
        pygame.draw.arc(
            surface, color, rect_inner, start_ccw2, end_ccw2, line_w
        )
