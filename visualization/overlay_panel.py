"""
Renders header title, telemetry dashboard, and active candidate information.
"""

from typing import Tuple, Dict, Any, Optional
import pygame

import config
from entities.agent_profile_registry import (
    AgentProfileRegistry,
    ResolvedAgentProfile,
)
from utils.font_manager import FontManager


class OverlayPanel:
    """
    Renders dashboard headers, active scores, and winner callouts in 2 columns.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_PANEL_RECT,
    ) -> None:
        """
        Initializes bounding rect, font manager, and active profile lookup.
        """
        self.x, self.y, self.w, self.h = rect
        self.font_manager: FontManager = FontManager()
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.profile: ResolvedAgentProfile = self.registry.get_profile(
            config.ACTIVE_AGENT_PROFILE
        )

    def draw_panel(
        self,
        surface: pygame.Surface,
        gen_data: Dict[str, Any],
        active_cand_idx: int,
        active_step: int,
        total_steps: int,
        total_gens: int,
        active_profile_title: Optional[str] = None,
    ) -> None:
        """
        Renders title header and 2-column scaled dashboard metrics.
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
            config.HUD_PANEL_TITLE_FONT_SIZE
        )
        body_font = self.font_manager.get_font(
            config.HUD_PANEL_BODY_FONT_SIZE
        )

        title_text: str = "PYNEVO - NEUROEVOLUTION"
        title_surf = title_font.render(
            title_text, True, config.COLOR_START
        )
        surface.blit(title_surf, (self.x + 12, self.y + 10))

        gen_num: int = int(gen_data.get("generation", 0)) + 1
        raw_scores = gen_data.get("raw_scores", [])
        top_score: float = max(raw_scores) if raw_scores else 0.0
        avg_score: float = (
            sum(raw_scores) / float(len(raw_scores))
            if raw_scores
            else 0.0
        )
        winner_idx: int = int(gen_data.get("winner_index", 0))

        telemetry = gen_data.get("telemetry", None)
        if telemetry is not None:
            max_f: int = int(telemetry.shape[0])
        else:
            cand_frames = gen_data.get("candidate_frames", [])
            max_f = max(len(cf) for cf in cand_frames) if cand_frames else 0

        col1_x: int = self.x + 12
        col2_x: int = self.x + 215
        row_y1: int = self.y + 40
        row_y2: int = self.y + 70
        row_y3: int = self.y + 100

        if active_profile_title is not None:
            lbl_sel = body_font.render(
                f"SELECTED : {active_profile_title}",
                True,
                config.COLOR_VIEWPORT_HIGHLIGHT,
            )
            lbl_win = body_font.render(
                f"WINNER   : {active_profile_title}",
                True,
                config.COLOR_EXIT,
            )
        else:
            lbl_sel = body_font.render(
                f"SELECTED : #{active_cand_idx}",
                True,
                config.COLOR_VIEWPORT_HIGHLIGHT,
            )
            lbl_win = body_font.render(
                f"WINNER   : #{winner_idx}",
                True,
                config.COLOR_EXIT,
            )

        surface.blit(lbl_sel, (col1_x, row_y1))

        lbl_step = body_font.render(
            f"FRAME STEP: {int(active_step)}/{int(max_f)}",
            True,
            (255, 255, 255),
        )
        surface.blit(lbl_step, (col1_x, row_y2))

        lbl_gen = body_font.render(
            f"GENERATION: {gen_num}/{total_gens}",
            True,
            (255, 255, 255),
        )
        surface.blit(lbl_gen, (col1_x, row_y3))

        surface.blit(lbl_win, (col2_x, row_y1))

        lbl_top = body_font.render(
            f"TOP SCORE : {top_score:.1f}",
            True,
            config.COLOR_EXIT,
        )
        surface.blit(lbl_top, (col2_x, row_y2))

        lbl_avg = body_font.render(
            f"AVG SCORE : {avg_score:.1f}",
            True,
            (200, 200, 200),
        )
        surface.blit(lbl_avg, (col2_x, row_y3))
