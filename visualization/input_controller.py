"""
Translates raw Pygame keyboard and mouse events into UI command actions.
"""

from typing import Tuple, Optional
import pygame

import config
from visualization.viewport_grid import ViewportGrid
from visualization.timeline_scrubber import TimelineScrubber


class InputController:
    """
    Processes Pygame events for viewport grid interactions and scrubbers.
    """

    def __init__(self) -> None:
        """
        Initializes click timer and held key repetition timing state.
        """
        self.last_click_time: int = 0
        self.held_nav_key: Optional[int] = None
        self.held_key_press_time: int = 0
        self.held_key_last_repeat_time: int = 0
        self.show_help_overlay: bool = False

    def process_events(
        self,
        viewport_grid: ViewportGrid,
        timeline_scrubber: TimelineScrubber,
        active_gen: int,
        active_frame: int,
        total_gens: int,
        total_frames: int,
        total_candidates: int
    ) -> Tuple[bool, int, int]:
        """
        Polls Pygame events and returns updated (running, active_gen, frame).
        """
        running: bool = True
        new_gen: int = active_gen
        new_frame: int = active_frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )

            elif event.type == pygame.MOUSEWHEEL:
                self._handle_mouse_wheel(event.y, timeline_scrubber)

            elif event.type == pygame.KEYDOWN:
                keep_running, new_g, new_f = self._handle_key_down(
                    event.key,
                    viewport_grid,
                    timeline_scrubber,
                    new_gen,
                    new_frame,
                    total_gens,
                    total_frames,
                    total_candidates
                )
                if not keep_running:
                    return False, new_g, new_f
                new_gen = new_g
                new_frame = new_f

            elif event.type == pygame.KEYUP:
                self._handle_key_up(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 2, 3):
                    new_gen, new_frame = self._handle_mouse_click(
                        event.button,
                        event.pos,
                        viewport_grid,
                        timeline_scrubber,
                        new_gen,
                        new_frame,
                        total_gens,
                        total_frames
                    )

            elif event.type == pygame.MOUSEMOTION:
                new_g_drag, new_f_drag = (
                    timeline_scrubber.handle_mouse_move(
                        event.pos, total_gens, total_frames
                    )
                )
                if new_g_drag is not None:
                    new_gen = new_g_drag
                    new_frame = 0
                if new_f_drag is not None:
                    new_frame = new_f_drag

            elif event.type == pygame.MOUSEBUTTONUP:
                timeline_scrubber.handle_mouse_up()

        new_gen, new_frame = self.update_held_keys(
            viewport_grid,
            timeline_scrubber,
            new_gen,
            new_frame,
            total_gens,
            total_frames
        )

        return running, new_gen, new_frame

    def update_held_keys(
        self,
        viewport_grid: ViewportGrid,
        timeline_scrubber: TimelineScrubber,
        active_gen: int,
        active_frame: int,
        total_gens: int,
        total_frames: int
    ) -> Tuple[int, int]:
        """
        Executes continuous repeats for held navigation keys based on timers.
        """
        if self.held_nav_key is None:
            return active_gen, active_frame

        now_ms: int = pygame.time.get_ticks()
        elapsed_press: int = now_ms - self.held_key_press_time

        if elapsed_press < config.TIMELINE_KEY_REPEAT_DELAY_MS:
            return active_gen, active_frame

        elapsed_interval: int = now_ms - self.held_key_last_repeat_time

        if elapsed_interval < config.TIMELINE_KEY_REPEAT_INTERVAL_MS:
            return active_gen, active_frame

        self.held_key_last_repeat_time = now_ms
        new_gen: int = active_gen
        new_frame: int = active_frame

        if self.held_nav_key == pygame.K_LEFT:
            jump: int = max(
                1,
                int(
                    total_frames *
                    config.TIMELINE_FRAME_JUMP_RATIO *
                    timeline_scrubber.playback_speed
                )
            )
            new_frame = max(0, active_frame - jump)

        elif self.held_nav_key == pygame.K_RIGHT:
            jump = max(
                1,
                int(
                    total_frames *
                    config.TIMELINE_FRAME_JUMP_RATIO *
                    timeline_scrubber.playback_speed
                )
            )
            new_frame = min(total_frames - 1, active_frame + jump)

        elif self.held_nav_key == pygame.K_UP:
            new_gen = min(total_gens - 1, active_gen + 1)
            new_frame = 0

        elif self.held_nav_key == pygame.K_DOWN:
            new_gen = max(0, active_gen - 1)
            new_frame = 0

        return new_gen, new_frame

    def _handle_mouse_wheel(
        self,
        y_delta: int,
        timeline_scrubber: TimelineScrubber
    ) -> None:
        """
        Steps playback speed up or down based on mouse wheel scroll direction.
        """
        if y_delta > 0:
            timeline_scrubber.step_speed_up()
        elif y_delta < 0:
            timeline_scrubber.step_speed_down()

    def _handle_key_down(
        self,
        key: int,
        viewport_grid: ViewportGrid,
        timeline_scrubber: TimelineScrubber,
        active_gen: int,
        active_frame: int,
        total_gens: int,
        total_frames: int,
        total_candidates: int
    ) -> Tuple[bool, int, int]:
        """
        Processes keydown events for viewport, transport, and scrubber.
        """
        new_gen: int = active_gen
        new_frame: int = active_frame

        nav_keys: Tuple[int, ...] = (
            pygame.K_LEFT,
            pygame.K_RIGHT,
            pygame.K_UP,
            pygame.K_DOWN
        )

        if key in nav_keys:
            now_ms: int = pygame.time.get_ticks()
            self.held_nav_key = key
            self.held_key_press_time = now_ms
            self.held_key_last_repeat_time = now_ms

        if key == pygame.K_h:
            self.show_help_overlay = not self.show_help_overlay

        elif key == pygame.K_t:
            timeline_scrubber.toggle_scrubber_mode()

        elif key == pygame.K_r:
            viewport_grid.refresh_middle_candidates()

        elif key in (pygame.K_KP7, pygame.K_7):
            viewport_grid.navigate_grid(-1, -1, total_candidates)
        elif key in (pygame.K_KP8, pygame.K_8):
            viewport_grid.navigate_grid(-1, 0, total_candidates)
        elif key in (pygame.K_KP9, pygame.K_9):
            viewport_grid.navigate_grid(-1, 1, total_candidates)
        elif key in (pygame.K_KP4, pygame.K_4):
            viewport_grid.navigate_grid(0, -1, total_candidates)
        elif key in (pygame.K_KP5, pygame.K_5):
            viewport_grid.reset_selection()
        elif key in (pygame.K_KP6, pygame.K_6):
            viewport_grid.navigate_grid(0, 1, total_candidates)
        elif key in (pygame.K_KP1, pygame.K_1):
            viewport_grid.navigate_grid(1, -1, total_candidates)
        elif key in (pygame.K_KP2, pygame.K_2):
            viewport_grid.navigate_grid(1, 0, total_candidates)
        elif key in (pygame.K_KP3, pygame.K_3):
            viewport_grid.navigate_grid(1, 1, total_candidates)

        if key == pygame.K_ESCAPE:
            return False, new_gen, new_frame

        elif key == pygame.K_TAB:
            viewport_grid.toggle_camera_mode()

        elif key == pygame.K_RETURN:
            viewport_grid.is_zoomed = not viewport_grid.is_zoomed

        elif key == pygame.K_SPACE:
            timeline_scrubber.is_playing = not timeline_scrubber.is_playing

        elif key == pygame.K_LEFT:
            jump: int = max(
                1,
                int(
                    total_frames *
                    config.TIMELINE_FRAME_JUMP_RATIO *
                    timeline_scrubber.playback_speed
                )
            )
            new_frame = max(0, active_frame - jump)

        elif key == pygame.K_RIGHT:
            jump = max(
                1,
                int(
                    total_frames *
                    config.TIMELINE_FRAME_JUMP_RATIO *
                    timeline_scrubber.playback_speed
                )
            )
            new_frame = min(total_frames - 1, active_frame + jump)

        elif key == pygame.K_UP:
            new_gen = min(total_gens - 1, active_gen + 1)
            new_frame = 0

        elif key == pygame.K_DOWN:
            new_gen = max(0, active_gen - 1)
            new_frame = 0

        elif key in (pygame.K_0, pygame.K_KP0):
            timeline_scrubber.repeat_all = not timeline_scrubber.repeat_all

        elif key in (
            pygame.K_PAGEUP,
            pygame.K_PLUS,
            pygame.K_KP_PLUS,
            pygame.K_EQUALS
        ):
            timeline_scrubber.step_speed_up()

        elif key in (
            pygame.K_PAGEDOWN,
            pygame.K_MINUS,
            pygame.K_KP_MINUS
        ):
            timeline_scrubber.step_speed_down()

        elif key in (pygame.K_PERIOD, pygame.K_KP_PERIOD):
            timeline_scrubber.reset_speed()

        return True, new_gen, new_frame

    def _handle_key_up(self, key: int) -> None:
        """
        Clears held key state on key release event.
        """
        if key == self.held_nav_key:
            self.held_nav_key = None

    def _handle_mouse_click(
        self,
        mouse_button: int,
        mouse_pos: Tuple[int, int],
        viewport_grid: ViewportGrid,
        timeline_scrubber: TimelineScrubber,
        active_gen: int,
        active_frame: int,
        total_gens: int,
        total_frames: int
    ) -> Tuple[int, int]:
        """
        Processes mouse click coordinates across viewports and controls.
        """
        vx, vy = mouse_pos
        gx, gy, gw, gh = config.LAYOUT_GRAPH_RECT
        if mouse_button == 3 and (gx <= vx <= gx + gw and gy <= vy <= gy + gh):
            self.show_help_overlay = not self.show_help_overlay
            return active_gen, active_frame

        now_ms: int = pygame.time.get_ticks()
        is_double: bool = (
            mouse_button == 1 and (now_ms - self.last_click_time) < 300
        )
        if mouse_button == 1:
            self.last_click_time = now_ms

        if mouse_button in (1, 3):
            viewport_grid.handle_click(
                mouse_pos,
                is_double_click=is_double,
                mouse_button=mouse_button
            )

        was_playing_before: bool = timeline_scrubber.is_playing

        clicked_gen, clicked_frame = timeline_scrubber.handle_click(
            mouse_pos,
            total_gens,
            total_frames,
            mouse_button=mouse_button
        )

        new_gen: int = active_gen
        new_frame: int = active_frame

        if (
            not was_playing_before and
            timeline_scrubber.is_playing and
            new_frame >= total_frames - 1
        ):
            if new_gen >= total_gens - 1 and timeline_scrubber.repeat_all:
                new_gen = 0
            new_frame = 0

        if clicked_gen is not None:
            new_gen = clicked_gen
            new_frame = 0

        if clicked_frame is not None:
            new_frame = clicked_frame

        return new_gen, new_frame
