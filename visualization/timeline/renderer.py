"""
Pygame rendering engine for timeline scrubber controls, tracks, and markers.
"""

from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import pygame

import config
from utils.font_manager import FontManager
from utils.color_utils import (
    resolve_solve_ratio_color,
    resolve_rank_color
)

if TYPE_CHECKING:
    from visualization.timeline_scrubber import TimelineScrubber


class ScrubberRenderer:
    """
    Renders transport buttons, timeline tracks, heatmaps, and handle tags.
    """

    def __init__(self, font_manager: FontManager) -> None:
        """
        Initializes track renderer and pre-computed solver index cache.
        """
        self.font_manager: FontManager = font_manager
        self._solver_cache: Dict[
            Tuple[int, str],
            Tuple[int, List[Tuple[int, int]]]
        ] = {}

    def draw_buttons(
        self,
        surface: pygame.Surface,
        scrubber: "TimelineScrubber"
    ) -> None:
        """
        Renders Play/Pause, Repeat Mode, Speed, and Scrubber Mode buttons.
        """
        font_btn = self.font_manager.get_font(
            config.HUD_SCRUBBER_BUTTON_FONT_SIZE
        )

        t_color = (
            config.COLOR_BUTTON_ACTIVE if scrubber.is_playing
            else config.COLOR_BUTTON
        )
        pygame.draw.rect(surface, t_color, scrubber.btn_toggle_rect)
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER, scrubber.btn_toggle_rect, 1
        )
        t_str: str = "PAUSE" if scrubber.is_playing else "PLAY"
        t_lbl = font_btn.render(t_str, True, (255, 255, 255))
        surface.blit(
            t_lbl, t_lbl.get_rect(center=scrubber.btn_toggle_rect.center)
        )

        r_color = (
            config.COLOR_BUTTON_ACTIVE if scrubber.repeat_all
            else config.COLOR_BUTTON
        )
        pygame.draw.rect(surface, r_color, scrubber.btn_repeat_rect)
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER, scrubber.btn_repeat_rect, 1
        )
        r_str: str = "REP ALL" if scrubber.repeat_all else "REP ONE"
        r_lbl = font_btn.render(r_str, True, (255, 255, 255))
        surface.blit(
            r_lbl, r_lbl.get_rect(center=scrubber.btn_repeat_rect.center)
        )

        pygame.draw.rect(
            surface, config.COLOR_BUTTON, scrubber.btn_speed_rect
        )
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER, scrubber.btn_speed_rect, 1
        )
        sp_lbl = font_btn.render(
            scrubber.get_formatted_speed_text(), True, (255, 255, 255)
        )
        surface.blit(
            sp_lbl, sp_lbl.get_rect(center=scrubber.btn_speed_rect.center)
        )

        m_color = (
            config.COLOR_BUTTON_ACTIVE if scrubber.scrubber_mode == "C"
            else config.COLOR_BUTTON
        )
        pygame.draw.rect(surface, m_color, scrubber.btn_mode_rect)
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER, scrubber.btn_mode_rect, 1
        )
        m_lbl = font_btn.render(
            scrubber.scrubber_mode, True, (255, 255, 255)
        )
        surface.blit(
            m_lbl, m_lbl.get_rect(center=scrubber.btn_mode_rect.center)
        )

    def draw_tracks(
        self,
        surface: pygame.Surface,
        scrubber: "TimelineScrubber",
        active_gen: int,
        total_gens: int,
        active_frame: int,
        total_frames: int,
        selected_cand_idx: int = 0,
        gen_history: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Renders generation/frame background tracks, solver ticks & markers.
        """
        font_marker = self.font_manager.get_font(
            config.HUD_SCRUBBER_MARKER_FONT_SIZE
        )

        pygame.draw.rect(
            surface, config.COLOR_TIMELINE_BAR, scrubber.frame_bar_rect
        )
        pygame.draw.rect(
            surface, config.COLOR_TIMELINE_BAR, scrubber.gen_bar_rect
        )

        is_block_mode: bool = getattr(
            config, "TIMELINE_BLOCK_GENERATION_BAR", True
        )

        if gen_history:
            self._draw_generation_track(
                surface, scrubber, active_gen, total_gens,
                gen_history, is_block_mode, font_marker
            )

        if gen_history and 0 <= active_gen < len(gen_history):
            self._draw_frame_ticks(
                surface, scrubber, active_gen, total_frames,
                selected_cand_idx, gen_history
            )

        self._draw_markers(
            surface, scrubber, active_gen, total_gens, active_frame,
            total_frames, is_block_mode, font_marker
        )

    def _get_generation_solver_data(
        self,
        g_data: Dict[str, Any],
        is_solve_mode: bool,
        target_hold_frames: int = 15
    ) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Pre-computes or retrieves cached solver count and step indices.
        """
        gen_idx: int = int(g_data.get("generation", 0))
        mode_str: str = "C" if is_solve_mode else "R"
        cache_key: Tuple[int, str] = (gen_idx, mode_str)

        if cache_key in self._solver_cache:
            return self._solver_cache[cache_key]

        telemetry = g_data.get("telemetry", None)
        solvers_info: List[Tuple[int, int]] = []
        solve_cnt: int = 0

        if telemetry is not None:
            pop_s: int = int(telemetry.shape[1])
            for c_idx in range(pop_s):
                arr = telemetry[:, c_idx, 7]
                if not is_solve_mode:
                    match_indices = np.where(arr > 0.5)[0]
                else:
                    if len(arr) < target_hold_frames:
                        match_indices = np.array([], dtype=int)
                    else:
                        conv = np.convolve(
                            arr,
                            np.ones(target_hold_frames, dtype=np.float32),
                            mode="valid"
                        )
                        match_raw = np.where(
                            conv >= float(target_hold_frames) - 0.01
                        )[0]
                        match_indices = match_raw + (target_hold_frames - 1)

                if len(match_indices) > 0:
                    solve_cnt += 1
                    solvers_info.append((c_idx, int(match_indices[0])))
        else:
            c_list = g_data.get("candidate_frames", [])
            for c_idx, cf in enumerate(c_list):
                for step_idx, f_dict in enumerate(cf):
                    if f_dict.get("reached_exit", False):
                        solve_cnt += 1
                        solvers_info.append((c_idx, step_idx))
                        break

        result = (solve_cnt, solvers_info)
        self._solver_cache[cache_key] = result
        return result

    def _draw_generation_track(
        self,
        surface: pygame.Surface,
        scrubber: "TimelineScrubber",
        active_gen: int,
        total_gens: int,
        gen_history: List[Dict[str, Any]],
        is_block_mode: bool,
        font_marker: pygame.font.Font
    ) -> None:
        """
        Renders generation solver density blocks or lines with zero gaps.
        """
        is_solve_mode: bool = (
            getattr(scrubber, "scrubber_mode", "R") == "C"
        )
        session_solver_counts: List[int] = []

        for g_data in gen_history:
            cnt, _ = self._get_generation_solver_data(
                g_data, is_solve_mode
            )
            session_solver_counts.append(cnt)

        max_solvers: int = max(max(session_solver_counts, default=1), 1)
        bar_x: float = float(scrubber.gen_bar_rect.x)
        bar_w: float = float(scrubber.gen_bar_rect.w)
        gen_denom: float = float(max(1, total_gens))

        for g_idx, solvers in enumerate(session_solver_counts):
            if solvers > 0:
                solve_ratio: float = float(solvers) / float(max_solvers)
                line_color = resolve_solve_ratio_color(solve_ratio)

                if is_block_mode:
                    x1: int = int(round(bar_x + (g_idx * bar_w / gen_denom)))
                    x2: int = int(
                        round(bar_x + ((g_idx + 1) * bar_w / gen_denom))
                    )
                    bw: int = max(1, x2 - x1)
                    block_rect = pygame.Rect(
                        x1, scrubber.gen_bar_rect.top, bw,
                        scrubber.gen_bar_rect.h
                    )
                    pygame.draw.rect(surface, line_color, block_rect)
                else:
                    g_r: float = float(g_idx) / float(
                        max(1, total_gens - 1)
                    )
                    gx = int(
                        scrubber.gen_bar_rect.x +
                        (g_r * scrubber.gen_bar_rect.w)
                    )
                    pygame.draw.line(
                        surface,
                        line_color,
                        (gx, scrubber.gen_bar_rect.top),
                        (gx, scrubber.gen_bar_rect.bottom),
                        2
                    )

        if is_block_mode:
            active_x1: int = int(
                round(bar_x + (active_gen * bar_w / gen_denom))
            )
            active_x2: int = int(
                round(bar_x + ((active_gen + 1) * bar_w / gen_denom))
            )
            bw = max(1, active_x2 - active_x1)
            outline_rect = pygame.Rect(
                active_x1 - 1,
                scrubber.gen_bar_rect.top - 1,
                bw + 2,
                scrubber.gen_bar_rect.h + 2
            )
            pygame.draw.rect(surface, (0, 0, 0), outline_rect, 2)

            block_center_x: int = active_x1 + (bw // 2)
            g_lbl = font_marker.render(
                f"# {int(active_gen) + 1}", True, config.COLOR_MARKER
            )
            g_rect = g_lbl.get_rect(
                midtop=(
                    block_center_x,
                    scrubber.gen_bar_rect.bottom + 2
                )
            )
            surface.blit(g_lbl, g_rect)

    def _draw_frame_ticks(
        self,
        surface: pygame.Surface,
        scrubber: "TimelineScrubber",
        active_gen: int,
        total_frames: int,
        selected_cand_idx: int,
        gen_history: List[Dict[str, Any]]
    ) -> None:
        """
        Renders rank-color-graded exit solve tick marks for active frame.
        """
        is_solve_mode: bool = (
            getattr(scrubber, "scrubber_mode", "R") == "C"
        )
        active_g_data = gen_history[active_gen]
        _, solvers_info = self._get_generation_solver_data(
            active_g_data, is_solve_mode
        )

        solvers_info.sort(key=lambda item: item[1])
        num_solvers: int = len(solvers_info)

        for rank_idx, (c_idx, f_step) in enumerate(solvers_info):
            r_color = resolve_rank_color(rank_idx, num_solvers)
            f_r: float = float(f_step) / float(
                max(1, total_frames - 1)
            )
            fx: int = int(
                scrubber.frame_bar_rect.x + (f_r * scrubber.frame_bar_rect.w)
            )
            is_sel: bool = (c_idx == selected_cand_idx)
            alpha_val: int = (
                255 if is_sel
                else config.TIMELINE_UNSELECTED_TICK_ALPHA
            )
            line_w: int = 3 if is_sel else 2

            line_surf = pygame.Surface(
                (line_w, scrubber.frame_bar_rect.h), pygame.SRCALPHA
            )
            line_surf.fill((*r_color, alpha_val))
            surface.blit(
                line_surf,
                (fx - (line_w // 2), scrubber.frame_bar_rect.top)
            )

    def _draw_markers(
        self,
        surface: pygame.Surface,
        scrubber: "TimelineScrubber",
        active_gen: int,
        total_gens: int,
        active_frame: int,
        total_frames: int,
        is_block_mode: bool,
        font_marker: pygame.font.Font
    ) -> None:
        """
        Renders rectangular handle markers and tag text labels.
        """
        m_width: int = config.HUD_SCRUBBER_MARKER_WIDTH
        m_height: int = config.HUD_SCRUBBER_MARKER_HEIGHT

        f_ratio: float = (
            float(active_frame) / float(max(1, total_frames - 1))
        )
        f_marker_x: int = int(
            scrubber.frame_bar_rect.x + (f_ratio * scrubber.frame_bar_rect.w)
        )
        f_handle_rect = pygame.Rect(0, 0, m_width, m_height)
        f_handle_rect.center = (
            f_marker_x, scrubber.frame_bar_rect.centery
        )

        pygame.draw.rect(surface, config.COLOR_MARKER, f_handle_rect)
        pygame.draw.rect(
            surface, config.COLOR_WALL_BORDER, f_handle_rect, 1
        )

        f_lbl = font_marker.render(
            f"# {int(active_frame)}", True, config.COLOR_MARKER
        )
        f_rect = f_lbl.get_rect(
            midbottom=(f_marker_x, scrubber.frame_bar_rect.y - 2)
        )
        surface.blit(f_lbl, f_rect)

        if not is_block_mode:
            g_ratio: float = (
                float(active_gen) / float(max(1, total_gens - 1))
            )
            g_marker_x: int = int(
                scrubber.gen_bar_rect.x + (g_ratio * scrubber.gen_bar_rect.w)
            )
            g_handle_rect = pygame.Rect(0, 0, m_width, m_height)
            g_handle_rect.center = (
                g_marker_x, scrubber.gen_bar_rect.centery
            )

            pygame.draw.rect(surface, config.COLOR_MARKER, g_handle_rect)
            pygame.draw.rect(
                surface, config.COLOR_WALL_BORDER, g_handle_rect, 1
            )

            g_lbl = font_marker.render(
                f"# {int(active_gen) + 1}", True, config.COLOR_MARKER
            )
            g_rect = g_lbl.get_rect(
                midtop=(g_marker_x, scrubber.gen_bar_rect.bottom + 2)
            )
            surface.blit(g_lbl, g_rect)
