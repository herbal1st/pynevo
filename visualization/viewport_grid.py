"""
Viewport manager rendering the entire population swarm on a single shared maze.
"""

from typing import Tuple, Dict, Any, Optional
import pygame

import config
from visualization.viewports.swarm_viewport import SwarmMazeViewport


class ViewportGrid:
    """
    Coordinates full-screen swarm replay visualization on a single shared maze.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_GRID_RECT,
        rows: int = 1,
        cols: int = 1,
        adapter: Optional[Any] = None
    ) -> None:
        self.x, self.y, self.w, self.h = rect
        self.rows: int = 1
        self.cols: int = 1
        self.selected_idx: int = 0
        self.is_zoomed: bool = False
        self.is_camera_centered: bool = False
        self.swarm_viewport: SwarmMazeViewport = SwarmMazeViewport(rect[2], rect[3])

    def refresh_middle_candidates(self) -> None:
        pass

    def toggle_camera_mode(self) -> None:
        self.is_camera_centered = not self.is_camera_centered

    def navigate_grid(
        self,
        delta_row: int,
        delta_col: int,
        total_candidates: int
    ) -> None:
        if total_candidates <= 0:
            return
        # Cycle through candidates in the population
        delta = delta_col if delta_col != 0 else delta_row
        self.selected_idx = (self.selected_idx + delta) % total_candidates

    def reset_selection(self) -> None:
        self.selected_idx = 0

    def draw_grid(
        self,
        surface: pygame.Surface,
        gen_data: Dict[str, Any],
        active_step: int
    ) -> None:
        """
        Renders the entire population swarm simultaneously on one single maze.
        """
        sub_rect: Tuple[int, int, int, int] = (self.x, self.y, self.w, self.h)
        self.swarm_viewport.render_swarm(
            surface,
            gen_data,
            selected_cand_idx=self.selected_idx,
            active_step=active_step,
            rect=sub_rect,
            is_camera_centered=self.is_camera_centered
        )

    def handle_click(
        self,
        click_pos: Tuple[int, int],
        is_double_click: bool = False,
        mouse_button: int = 1
    ) -> bool:
        cx, cy = click_pos
        if not (self.x <= cx <= self.x + self.w and self.y <= cy <= self.y + self.h):
            return False

        if mouse_button == 3:
            self.toggle_camera_mode()
            return True

        return True
