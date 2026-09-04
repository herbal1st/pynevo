"""
Renders shortcut cheat-sheet overlay panel specifically for Live Winner Mode.
"""

from typing import Tuple, List
import pygame

import config
from utils.font_manager import FontManager


class LiveHelpOverlay:
    """
    Renders live mode cheat-sheet overlay in LAYOUT_GRAPH_RECT position.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_GRAPH_RECT,
    ) -> None:
        """
        Initializes bounding rect and font manager.
        """
        self.x, self.y, self.w, self.h = rect
        self.font_manager: FontManager = FontManager()

    def draw_panel(self, surface: pygame.Surface) -> None:
        """
        Renders styled background frame and two-column live shortcut list.
        """
        pygame.draw.rect(
            surface, config.COLOR_BG, (self.x, self.y, self.w, self.h)
        )
        pygame.draw.rect(
            surface,
            config.COLOR_WALL_BORDER,
            (self.x, self.y, self.w, self.h),
            1,
        )

        title_font = self.font_manager.get_font(
            config.HUD_HELP_TITLE_FONT_SIZE, bold=True
        )
        body_font = self.font_manager.get_font(
            config.HUD_HELP_BODY_FONT_SIZE, bold=False
        )
        bold_body_font = self.font_manager.get_font(
            config.HUD_HELP_BODY_FONT_SIZE, bold=True
        )

        title_text: str = "LIVE SOLVER CHEAT-SHEET"
        title_surf = title_font.render(
            title_text, True, config.COLOR_START
        )
        surface.blit(title_surf, (self.x + 12, self.y + 10))

        shortcuts: List[Tuple[str, str, str]] = [
            ("HEADER", "LIVE EVALUATION & NAVIGATION", ""),
            ("KEY", "END", "Toggle Live Winner Mode"),
            ("KEY", "SPACE", "Toggle Play / Pause"),
            ("KEY", "LEFT / RIGHT", "Jump Frame (- / +)"),
            ("KEY", "UP / DOWN", "Cycle Saved Brains"),
            ("KEY", "R", "Regenerate Fresh Maze"),
            ("KEY", "T", "Toggle Scrubber (R/C)"),
            ("HEADER", "CAMERA & VIEWPORT", ""),
            ("KEY", "TAB / R-CLICK", "Toggle Tracking Mode"),
            ("HEADER", "SIMULATION SPEED", ""),
            ("KEY", "PGUP/DN, +/-", "Playback Speed (1/10x..10x)"),
            ("KEY", "WHEEL SCROLL", "Step Speed Up / Down"),
            ("KEY", "PERIOD (.)", "Reset Speed to 1x"),
            ("HEADER", "SYSTEM & HELP", ""),
            ("KEY", "H / R-CLICK", "Toggle Help Overlay"),
            ("KEY", "ESC", "Exit Application"),
        ]

        start_y: int = self.y + 36
        curr_y: int = start_y
        col1_x: int = self.x + 14
        col2_x: int = self.x + 155

        line_h: int = body_font.get_linesize() + 3

        for row_type, key_str, desc_str in shortcuts:
            if row_type == "HEADER":
                curr_y += 4
                hdr_surf = bold_body_font.render(
                    f"-- {key_str} --", True, config.COLOR_VIEWPORT_HIGHLIGHT
                )
                surface.blit(hdr_surf, (col1_x, curr_y))
                curr_y += line_h
            else:
                k_surf = bold_body_font.render(
                    key_str, True, config.COLOR_EXIT
                )
                d_surf = body_font.render(
                    desc_str, True, (220, 220, 220)
                )
                surface.blit(k_surf, (col1_x, curr_y))
                surface.blit(d_surf, (col2_x, curr_y))
                curr_y += line_h
