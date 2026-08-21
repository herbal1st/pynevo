"""
Interactive UI transport controls facade, dual timeline bars, and markers.
"""

from typing import Tuple, List, Optional, Dict, Any
import pygame

import config
from utils.font_manager import FontManager
from visualization.timeline.renderer import ScrubberRenderer


class TimelineScrubber:
    """
    Facade managing transport controls, dual timeline bars, and markers.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_SCRUBBER_RECT,
    ) -> None:
        """
        Initializes bounding rects, playback state, and renderer instance.
        """
        self.x, self.y, self.w, self.h = rect
        self.is_playing: bool = True
        self.repeat_all: bool = True
        self.is_dragging_frame: bool = False
        self.is_dragging_gen: bool = False
        self.playback_speed: float = float(config.DEFAULT_PLAYBACK_SPEED)

        self.speed_options: List[float] = [
            0.1,
            1.0 / 9.0,
            0.125,
            1.0 / 7.0,
            1.0 / 6.0,
            0.2,
            0.25,
            1.0 / 3.0,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
        ]

        btn_h: int = config.HUD_SCRUBBER_BUTTON_HEIGHT
        self.btn_toggle_rect: pygame.Rect = pygame.Rect(
            self.x, self.y, 70, btn_h
        )
        self.btn_repeat_rect: pygame.Rect = pygame.Rect(
            self.x + 75, self.y, 70, btn_h
        )
        self.btn_speed_rect: pygame.Rect = pygame.Rect(
            self.x + 150, self.y, 50, btn_h
        )

        bar_x: int = self.x + 210
        bar_w: int = self.w - 210
        bar_h: int = config.HUD_SCRUBBER_BAR_HEIGHT

        self.frame_bar_rect: pygame.Rect = pygame.Rect(
            bar_x, self.y, bar_w, bar_h
        )
        self.gen_bar_rect: pygame.Rect = pygame.Rect(
            bar_x, self.y + bar_h + 8, bar_w, bar_h
        )

        self.font_manager: FontManager = FontManager()
        self.renderer: ScrubberRenderer = ScrubberRenderer(
            self.font_manager
        )

    def get_formatted_speed_text(self) -> str:
        """
        Returns human-readable text string for active playback speed.
        """
        sp: float = float(self.playback_speed)
        if sp < 0.95:
            denom: int = int(round(1.0 / sp))
            return f"1/{denom}x"

        val_int: int = int(round(sp))
        return f"{val_int}x"

    def reset_speed(self) -> None:
        """
        Resets playback speed multiplier back to default 1.0 speed.
        """
        self.playback_speed = float(config.DEFAULT_PLAYBACK_SPEED)

    def _get_current_speed_index(self) -> int:
        """
        Returns index of active playback speed in speed options list.
        """
        sp: float = self.playback_speed
        best_idx: int = min(
            range(len(self.speed_options)),
            key=lambda i: abs(self.speed_options[i] - sp),
        )
        return best_idx

    def step_speed_up(self) -> None:
        """
        Steps playback speed to the next available multiplier.
        """
        idx: int = self._get_current_speed_index()
        next_idx: int = (idx + 1) % len(self.speed_options)
        self.playback_speed = self.speed_options[next_idx]

    def step_speed_down(self) -> None:
        """
        Steps playback speed to the previous available multiplier.
        """
        idx: int = self._get_current_speed_index()
        next_idx: int = (idx - 1) % len(self.speed_options)
        self.playback_speed = self.speed_options[next_idx]

    def draw_controls(
        self,
        surface: pygame.Surface,
        active_gen: int,
        total_gens: int,
        active_frame: int,
        total_frames: int,
        selected_cand_idx: int = 0,
        gen_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Delegates button and track rendering to renderer engine.
        """
        self.renderer.draw_buttons(surface, self)
        self.renderer.draw_tracks(
            surface,
            self,
            active_gen,
            total_gens,
            active_frame,
            total_frames,
            selected_cand_idx,
            gen_history,
        )

    def handle_click(
        self,
        click_pos: Tuple[int, int],
        total_gens: int,
        total_frames: int,
        mouse_button: int = 1,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Processes transport button clicks and initial timeline dragging.
        """
        cx, cy = click_pos

        if self.btn_toggle_rect.collidepoint(cx, cy):
            self.is_playing = not self.is_playing
            return None, None

        if self.btn_repeat_rect.collidepoint(cx, cy):
            self.repeat_all = not self.repeat_all
            return None, None

        if self.btn_speed_rect.collidepoint(cx, cy):
            if mouse_button == 3:
                self.step_speed_down()
            else:
                self.step_speed_up()
            return None, None

        new_frame: Optional[int] = None
        new_gen: Optional[int] = None

        if self.frame_bar_rect.collidepoint(cx, cy):
            self.is_dragging_frame = True
            new_frame = self._calculate_frame_index(cx, total_frames)

        elif self.gen_bar_rect.collidepoint(cx, cy):
            self.is_dragging_gen = True
            new_gen = self._calculate_generation_index(cx, total_gens)

        return new_gen, new_frame

    def handle_mouse_move(
        self,
        click_pos: Tuple[int, int],
        total_gens: int,
        total_frames: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Updates frame or generation index during mouse drag movement.
        """
        cx, _ = click_pos
        new_frame: Optional[int] = None
        new_gen: Optional[int] = None

        if self.is_dragging_frame:
            new_frame = self._calculate_frame_index(cx, total_frames)

        elif self.is_dragging_gen:
            new_gen = self._calculate_generation_index(cx, total_gens)

        return new_gen, new_frame

    def handle_mouse_up(self) -> None:
        """
        Clears timeline scrubber bar dragging flags on mouse release.
        """
        self.is_dragging_frame = False
        self.is_dragging_gen = False

    def _calculate_frame_index(
        self, click_x: int, total_frames: int
    ) -> int:
        """
        Translates pixel X coordinate to target frame step index.
        """
        rel_x: float = float(click_x - self.frame_bar_rect.x)
        ratio: float = max(
            0.0, min(1.0, rel_x / float(self.frame_bar_rect.w))
        )
        return int(round(ratio * max(0, total_frames - 1)))

    def _calculate_generation_index(
        self, click_x: int, total_gens: int
    ) -> int:
        """
        Translates pixel X coordinate to mode-aware generation index.
        """
        if total_gens <= 1:
            return 0

        rel_x: float = float(click_x - self.gen_bar_rect.x)
        is_block_mode: bool = getattr(
            config, "TIMELINE_BLOCK_GENERATION_BAR", True
        )

        if is_block_mode:
            ratio: float = max(
                0.0, min(0.999999, rel_x / float(self.gen_bar_rect.w))
            )
            return int(ratio * float(total_gens))

        ratio = max(0.0, min(1.0, rel_x / float(self.gen_bar_rect.w)))
        return int(round(ratio * float(total_gens - 1)))
