"""
Renders real-time headless training progress and metrics curves onto a fixed fullscreen Pygame surface.
"""

from typing import List, Dict, Any, Tuple
import pygame

import config
from utils.font_manager import FontManager


class TrainingHUDOverlay:
    """
    Renders an in-game GUI dashboard visualizing live evolutionary training generation stats,
    fitness curves, and performance metrics on a fixed fullscreen canvas.
    """

    def __init__(self, rect: Tuple[int, int, int, int] = (20, 20, 1240, 680)) -> None:
        self.x, self.y, self.w, self.h = rect
        self.font_manager: FontManager = FontManager()
        self.generation_records: List[Dict[str, Any]] = []

    def log_line(self, line: str) -> None:
        print(line)

    def handle_scroll(self, y_delta: int) -> None:
        pass

    def record_generation(
        self,
        gen_idx: int,
        raw_scores: List[float],
        norm_scores: List[float],
        candidate_states: List[Any],
        running_solve_avg: float,
        elapsed_sec: float,
        map_width: int,
        map_height: int
    ) -> None:
        top_score = max(raw_scores) if raw_scores else 0.0
        avg_score = sum(raw_scores) / float(len(raw_scores)) if raw_scores else 0.0
        solved_count = sum(1 for c in candidate_states if getattr(c, "stages_cleared", 0) > 0 or getattr(c, "first_hold_clear_step", -1) >= 0)
        solve_pct = running_solve_avg * 100.0

        self.generation_records.append({
            "gen": gen_idx + 1,
            "top": top_score,
            "avg": avg_score,
            "solve_pct": solve_pct,
            "solved_count": solved_count,
            "elapsed": elapsed_sec,
            "map_size": f"{map_width}x{map_height}"
        })

    def draw_hud(self, surface: pygame.Surface, active_gen: int, total_gens: int) -> None:
        win_w, win_h = surface.get_size()
        self.w = win_w - 40
        self.h = win_h - 40

        surface.fill(config.COLOR_BG)
        pygame.draw.rect(surface, config.COLOR_WALL_BORDER, (self.x, self.y, self.w, self.h), 2)

        title_font = self.font_manager.get_font(22, bold=True)
        body_font = self.font_manager.get_font(14, bold=False)
        bold_font = self.font_manager.get_font(14, bold=True)

        title_surf = title_font.render("PYNEVO - HEADLESS NEUROEVOLUTION TRAINING MONITOR (FULLSCREEN)", True, config.COLOR_START)
        surface.blit(title_surf, (self.x + 20, self.y + 16))

        if not self.generation_records:
            wait_surf = bold_font.render("Initializing training worker pool & compiling JIT kernels...", True, config.COLOR_VIEWPORT_HIGHLIGHT)
            surface.blit(wait_surf, (self.x + 20, self.y + 60))
            return

        latest = self.generation_records[-1]
        
        # Stat Callouts Box (Top)
        stats_y = self.y + 55
        col_w = self.w // 4
        
        self._draw_stat_box(surface, self.x + 20, stats_y, col_w - 30, 75, "GENERATION", f"{latest['gen']}", config.COLOR_VIEWPORT_HIGHLIGHT)
        self._draw_stat_box(surface, self.x + col_w, stats_y, col_w - 30, 75, "TOP SCORE", f"{latest['top']:.1f}", config.COLOR_EXIT)
        self._draw_stat_box(surface, self.x + (col_w * 2), stats_y, col_w - 30, 75, "POPULATION AVG", f"{latest['avg']:.1f}", (200, 200, 200))
        self._draw_stat_box(surface, self.x + (col_w * 3), stats_y, col_w - 30, 75, "SOLVE RATE", f"{latest['solve_pct']:.1f}%", config.COLOR_START)

        # Full-width Fitness Curve Graph (Middle)
        graph_rect = pygame.Rect(self.x + 20, self.y + 150, self.w - 40, 280)
        pygame.draw.rect(surface, (20, 20, 28), graph_rect)
        pygame.draw.rect(surface, config.COLOR_WALL_BORDER, graph_rect, 1)

        graph_title = bold_font.render("LIVE FITNESS & SOLVE RATE CURVES", True, (180, 180, 200))
        surface.blit(graph_title, (graph_rect.x + 16, graph_rect.y + 12))

        if len(self.generation_records) >= 2:
            self._draw_line_graph(surface, graph_rect)

        # Recent Generation Logs Table (Bottom)
        table_rect = pygame.Rect(self.x + 20, self.y + 450, self.w - 40, self.h - 470)
        pygame.draw.rect(surface, (20, 20, 28), table_rect)
        pygame.draw.rect(surface, config.COLOR_WALL_BORDER, table_rect, 1)

        table_title = bold_font.render("RECENT GENERATION LOGS", True, (180, 180, 200))
        surface.blit(table_title, (table_rect.x + 16, table_rect.y + 12))

        headers = ["GEN", "MAP SIZE", "TOP SCORE", "AVG SCORE", "SOLVED", "SOLVE %", "TIME"]
        hx = table_rect.x + 20
        hy = table_rect.y + 45
        hw = [90, 140, 180, 180, 140, 140, 140]

        for h_text, w_val in zip(headers, hw):
            h_surf = bold_font.render(h_text, True, config.COLOR_VIEWPORT_HIGHLIGHT)
            surface.blit(h_surf, (hx, hy))
            hx += w_val

        pygame.draw.line(surface, config.COLOR_WALL_BORDER, (table_rect.x + 15, hy + 22), (table_rect.right - 15, hy + 22), 1)

        log_y = hy + 32
        recent_records = self.generation_records[-10:]
        for rec in reversed(recent_records):
            hx = table_rect.x + 20
            row_data = [
                f"{rec['gen']}",
                f"{rec['map_size']}",
                f"{rec['top']:.1f}",
                f"{rec['avg']:.1f}",
                f"{rec['solved_count']}",
                f"{rec['solve_pct']:.1f}%",
                f"{rec['elapsed']:.2f}s"
            ]
            for val, w_val in zip(row_data, hw):
                r_surf = body_font.render(val, True, (220, 220, 220))
                surface.blit(r_surf, (hx, log_y))
                hx += w_val
            log_y += 28

    def _draw_stat_box(self, surface: pygame.Surface, x: int, y: int, w: int, h: int, label: str, value: str, val_color: Tuple[int, int, int]) -> None:
        box_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surface, (22, 22, 30), box_rect)
        pygame.draw.rect(surface, config.COLOR_WALL_BORDER, box_rect, 1)

        lbl_font = self.font_manager.get_font(11, bold=True)
        val_font = self.font_manager.get_font(24, bold=True)

        l_surf = lbl_font.render(label, True, (150, 150, 170))
        v_surf = val_font.render(value, True, val_color)

        surface.blit(l_surf, (box_rect.x + 14, box_rect.y + 12))
        surface.blit(v_surf, (box_rect.x + 14, box_rect.y + 36))

    def _draw_line_graph(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        points_top = []
        points_solve = []
        
        max_score = max([r["top"] for r in self.generation_records])
        max_score = max(100.0, max_score)

        n = len(self.generation_records)
        gx_min = rect.x + 70
        gx_max = rect.right - 30
        gy_min = rect.y + 50
        gy_max = rect.bottom - 30

        gw = gx_max - gx_min
        gh = gy_max - gy_min

        for idx, rec in enumerate(self.generation_records):
            px = gx_min + int(gw * (idx / max(1, n - 1)))
            
            py_top = gy_max - int(gh * min(1.0, rec["top"] / max_score))
            points_top.append((px, py_top))

            py_solve = gy_max - int(gh * min(1.0, rec["solve_pct"] / 100.0))
            points_solve.append((px, py_solve))

        if len(points_top) >= 2:
            pygame.draw.lines(surface, config.COLOR_EXIT, False, points_top, 2)
            pygame.draw.lines(surface, config.COLOR_START, False, points_solve, 2)

        bold_font = self.font_manager.get_font(13, bold=True)
        leg1 = bold_font.render("--- TOP SCORE (Green)", True, config.COLOR_EXIT)
        leg2 = bold_font.render("--- SOLVE % (Blue)", True, config.COLOR_START)
        surface.blit(leg1, (rect.right - 260, rect.y + 12))
        surface.blit(leg2, (rect.right - 260, rect.y + 30))
