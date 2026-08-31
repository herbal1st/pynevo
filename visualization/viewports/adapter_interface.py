"""
Abstract interface contract for environment viewport adapters.
"""

from typing import Tuple, Dict, Any
import pygame


class IViewportAdapter:
    """
    Abstract interface contract defining sub-viewport rendering methods.
    """

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
        cols: int
    ) -> None:
        """
        Renders an individual candidate environment frame into rect bounds.
        """
        raise NotImplementedError
