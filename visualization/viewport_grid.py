"""
Sub-viewport grid coordinator managing grid layout and slot adapters.
"""

from typing import Tuple, Dict, Any, Optional, List
import pygame

import config
from visualization.viewports.grid_layout import GridLayoutManager
from visualization.viewports.candidate_mapper import CandidatePoolMapper
from visualization.viewports.native_maze_viewport import NativeMazeViewport
from visualization.viewports.adapter_interface import IViewportAdapter
from visualization.viewports.native.first_person_3d_renderer import FirstPerson3DRenderer


class ViewportGrid:
    """
    Coordinates sub-viewport grid slots, slot mapping, and camera tracking.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_GRID_RECT,
        rows: int = config.GRID_ROWS,
        cols: int = config.GRID_COLS,
        adapter: Optional[IViewportAdapter] = None
    ) -> None:
        self.layout: GridLayoutManager = GridLayoutManager(rect, rows, cols)
        self.selected_slot: int = 0
        self.is_zoomed: bool = False
        self.is_camera_centered: bool = False
        self.is_3d_mode: bool = False
        self.refresh_seed_offset: int = 0
        self.adapter: IViewportAdapter = adapter or NativeMazeViewport(
            rect[2], rect[3]
        )
        self.renderer_3d: FirstPerson3DRenderer = FirstPerson3DRenderer()
        self._last_mapped_candidates: List[int] = []

    @property
    def selected_idx(self) -> int:
        if (
            0 <= self.selected_slot < len(self._last_mapped_candidates)
        ):
            return self._last_mapped_candidates[self.selected_slot]
        return 0

    @property
    def rows(self) -> int:
        return self.layout.rows

    @property
    def cols(self) -> int:
        return self.layout.cols

    @property
    def x(self) -> int:
        return self.layout.x

    @property
    def y(self) -> int:
        return self.layout.y

    @property
    def w(self) -> int:
        return self.layout.w

    @property
    def h(self) -> int:
        return self.layout.h

    def refresh_middle_candidates(self) -> None:
        self.refresh_seed_offset += 1

    def toggle_camera_mode(self) -> None:
        self.is_camera_centered = not self.is_camera_centered

    def toggle_3d_mode(self) -> None:
        self.is_3d_mode = not self.is_3d_mode

    def navigate_grid(
        self,
        delta_row: int,
        delta_col: int,
        total_candidates: int
    ) -> None:
        if total_candidates <= 0:
            return

        self.selected_slot = self.layout.navigate_slot(
            self.selected_slot, delta_row, delta_col
        )

    def reset_selection(self) -> None:
        self.selected_slot = 0

    def draw_grid(
        self,
        surface: pygame.Surface,
        gen_data: Dict[str, Any],
        active_step: int
    ) -> None:
        if self.is_3d_mode:
            # Fullscreen 3D view overrides all grid / zoom layout
            frame_state = self.adapter.state_resolver.resolve_frame_state(
                gen_data, self.selected_idx, active_step
            )
            if frame_state:
                self.renderer_3d.render_3d_view(surface, gen_data, frame_state)
            return

        telemetry = gen_data.get("telemetry", None)
        num_cand: int = (
            int(telemetry.shape[1]) if telemetry is not None
            else len(gen_data.get("candidate_frames", []))
        )
        if num_cand == 0:
            return

        gen_num: int = int(gen_data.get("generation", 0))
        scores: List[float] = gen_data.get("raw_scores", [])

        effective_seed: int = gen_num + self.refresh_seed_offset
        mapped_candidates = CandidatePoolMapper.map_candidates_to_slots(
            num_cand, self.rows, self.cols, scores=scores, seed=effective_seed
        )
        self._last_mapped_candidates = mapped_candidates

        if self.is_zoomed:
            sub_rect: Tuple[int, int, int, int] = (
                self.x, self.y, self.w, self.h
            )
            self.adapter.render_viewport(
                surface, gen_data, self.selected_idx, active_step,
                sub_rect, is_selected=True, is_zoomed=True,
                is_camera_centered=self.is_camera_centered,
                rows=self.rows, cols=self.cols
            )
            return

        for slot_idx, cand_idx in enumerate(mapped_candidates):
            if cand_idx >= num_cand:
                continue

            sub_rect = self.layout.get_sub_viewport_rect(slot_idx)
            is_sel: bool = (slot_idx == self.selected_slot)

            self.adapter.render_viewport(
                surface, gen_data, cand_idx, active_step,
                sub_rect, is_selected=is_sel, is_zoomed=False,
                is_camera_centered=self.is_camera_centered,
                rows=self.rows, cols=self.cols
            )

    def handle_click(
        self,
        click_pos: Tuple[int, int],
        is_double_click: bool = False,
        mouse_button: int = 1
    ) -> bool:
        cx, cy = click_pos
        slot_idx = self.layout.get_slot_index_from_click(cx, cy)
        if slot_idx is None:
            return False

        if mouse_button == 3:
            self.toggle_camera_mode()
            return True

        if self.is_zoomed:
            if is_double_click:
                self.is_zoomed = False
            return True

        self.selected_slot = slot_idx

        if is_double_click:
            self.is_zoomed = True

        return True
