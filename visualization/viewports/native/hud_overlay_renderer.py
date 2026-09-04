"""
Renders sub-viewport HUD text tags, health bars, terminal cards, and borders.
"""

from typing import Tuple, Optional
import pygame

import config
from utils.font_manager import FontManager
from utils.color_utils import resolve_health_color
from utils.surface_utils import create_alpha_surface
from visualization.viewports.native.state_resolver import ViewportFrameState


class ViewportHUDOverlayRenderer:
    """
    Renders sub-viewport HUD indicators, health bar, cards, and status frames.
    """

    def __init__(self, canvas_w: int, canvas_h: int) -> None:
        """
        Initializes font manager and alpha scratchpad surface.
        """
        self.font_manager: FontManager = FontManager()
        self.scratchpad: pygame.Surface = create_alpha_surface(
            canvas_w, canvas_h
        )

    def draw_hud_overlays(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        frame_state: ViewportFrameState,
        ui_scale: float,
        show_scorecard: bool = True,
    ) -> None:
        """
        Renders ID tag, score tag, health bar, and optional score overlay.
        """
        rx, ry, rw, rh = rect

        font_norm = self.font_manager.get_font(
            max(10, int(12 * ui_scale))
        )
        font_small = self.font_manager.get_font(
            max(8, int(10 * ui_scale))
        )

        idx_surf = font_norm.render(
            f"#{frame_state.cand_idx}",
            True,
            config.COLOR_VIEWPORT_HIGHLIGHT[:3],
        )
        surface.blit(idx_surf, (rx + 4, ry + 4))

        score_surf = font_small.render(
            f"{frame_state.score_val}",
            True,
            config.COLOR_VIEWPORT_HIGHLIGHT[:3],
        )
        score_rect = score_surf.get_rect(
            bottomright=(
                rx + rw - int(6 * ui_scale),
                ry + rh - int(4 * ui_scale),
            )
        )
        surface.blit(score_surf, score_rect)

        self._draw_health_bar(
            surface, rect, frame_state.health, ui_scale, font_small
        )

        if show_scorecard:
            self._draw_center_score_overlay(
                surface,
                rect,
                frame_state.score_val,
                frame_state.reached_exit,
                frame_state.is_alive,
            )

    def draw_viewport_borders(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        frame_state: ViewportFrameState,
        is_selected: bool,
    ) -> None:
        """
        Renders 3px inner status border and 1px selection outline.
        """
        if frame_state.reached_exit:
            pygame.draw.rect(
                surface, config.COLOR_FRAME_SOLVED[:3], rect, 3
            )
        elif not frame_state.is_alive:
            pygame.draw.rect(
                surface, config.COLOR_FRAME_DEAD[:3], rect, 3
            )
        else:
            pygame.draw.rect(
                surface, config.COLOR_WALL_BORDER[:3], rect, 1
            )

        if is_selected:
            pygame.draw.rect(
                surface, config.COLOR_VIEWPORT_HIGHLIGHT[:3], rect, 1
            )

    def _draw_health_bar(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        health_val: float,
        ui_scale: float,
        font_small: pygame.font.Font,
    ) -> None:
        """
        Renders health bar fill, percentage label, and bounding frame.
        """
        rx, ry, rw, rh = rect
        pct_val: float = health_val * 100.0
        hp_color = resolve_health_color(health_val)

        bar_w: int = int(rw * 0.20 * ui_scale)
        bar_h: int = max(4, int(6 * ui_scale))
        bar_x: int = rx + rw - bar_w - int(6 * ui_scale)
        bar_y: int = ry + int(6 * ui_scale)

        fill_w: int = int(bar_w * health_val)
        bar_fill_x: int = bar_x + (bar_w - fill_w)

        pct_surf = font_small.render(f"{pct_val:5.1f}%", True, hp_color)
        pct_rect = pct_surf.get_rect(
            midright=(bar_x - 4, bar_y + (bar_h // 2))
        )
        surface.blit(pct_surf, pct_rect)

        pygame.draw.rect(
            surface,
            config.COLOR_WALL_BORDER[:3],
            (bar_x, bar_y, bar_w, bar_h),
        )
        if fill_w > 0:
            pygame.draw.rect(
                surface, hp_color, (bar_fill_x, bar_y, fill_w, bar_h)
            )

    def _draw_center_score_overlay(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        score_val: int,
        reached_exit: bool,
        is_alive: bool,
    ) -> None:
        """
        Renders translucent terminal score card only when candidate is dead.
        """
        if is_alive:
            return

        rx, ry, rw, rh = rect
        text_color: Tuple[int, int, int] = config.COLOR_FRAME_DEAD[:3]

        target_font_size: int = max(14, int(rh * 0.22))
        font = self.font_manager.get_font(target_font_size, bold=True)

        score_str: str = f"{score_val}"
        text_w, text_h = font.size(score_str)

        max_allowed_w: int = int(rw * 0.60)
        if text_w > max_allowed_w and max_allowed_w > 10:
            fit_size: int = max(
                12,
                int(
                    target_font_size * (max_allowed_w / float(text_w))
                ),
            )
            font = self.font_manager.get_font(fit_size, bold=True)
            text_w, text_h = font.size(score_str)

        pad_x: int = max(8, int(text_w * 0.15))
        pad_y: int = max(4, int(text_h * 0.10))

        box_w: int = text_w + (pad_x * 2)
        box_h: int = text_h + (pad_y * 2)

        center_x: int = rx + (rw // 2)
        center_y: int = ry + int(float(rh) * 0.25)

        box_x: int = center_x - (box_w // 2)
        box_y: int = center_y - (box_h // 2)

        rel_box_x: int = box_x - rx
        rel_box_y: int = box_y - ry

        self.scratchpad.fill((0, 0, 0, 0))

        bg_box = pygame.Rect(rel_box_x, rel_box_y, box_w, box_h)
        pygame.draw.rect(
            self.scratchpad, (15, 15, 20, 200), bg_box, border_radius=6
        )
        pygame.draw.rect(
            self.scratchpad,
            (*text_color, 220),
            bg_box,
            1,
            border_radius=6,
        )

        score_surf = font.render(score_str, True, text_color)
        rel_text_rect = score_surf.get_rect(
            center=(center_x - rx, center_y - ry)
        )
        self.scratchpad.blit(score_surf, rel_text_rect)

        surface.blit(
            self.scratchpad, (rx, ry), area=pygame.Rect(0, 0, rw, rh)
        )
