"""
Application window lifecycle manager, Pygame event loop, and GUI runner.
"""

import os
import sys
from typing import Tuple
import pygame

import config
from entities.training_profile_registry import TrainingProfileRegistry
from evolution.recorder import FrameRecorder
from visualization.live_view_presenter import LiveViewPresenter
from visualization.viewport_grid import ViewportGrid
from visualization.network_graph.graph_facade import NetworkGraph
from visualization.help_overlay import HelpOverlay
from visualization.timeline_scrubber import TimelineScrubber
from visualization.overlay_panel import OverlayPanel
from visualization.input_controller import InputController


class AppWindow:
    """
    Manages Pygame display window lifecycle, event loop, and render cycle.
    """

    def __init__(self, recorder: FrameRecorder) -> None:
        """
        Initializes Pygame display, clock, UI subsystems, & live presenter.
        """
        self.recorder: FrameRecorder = recorder
        self.recorder.load_temporary_disk_archive()

        self.total_gens: int = len(recorder.generations_history)
        if self.total_gens == 0:
            print("[Error] No generation history recorded or loaded.")
            sys.exit(1)

        self.training_profile = TrainingProfileRegistry().get_profile(
            config.ACTIVE_TRAINING_PROFILE
        )
        self.max_steps: int = self.training_profile.max_simulation_steps

        pygame.init()

        flags: int = (
            pygame.RESIZABLE if config.USE_RESIZABLE_WINDOW else 0
        )
        self.screen: pygame.Surface = pygame.display.set_mode(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), flags
        )
        pygame.display.set_caption("PyNevo - Neuroevolution Visualizer")
        self._load_window_icon()

        self.virtual_surface: pygame.Surface = pygame.Surface(
            (config.VIRTUAL_WIDTH, config.VIRTUAL_HEIGHT)
        )

        self.clock: pygame.time.Clock = pygame.time.Clock()

        self.viewport_grid: ViewportGrid = ViewportGrid(
            config.LAYOUT_GRID_RECT
        )
        self.overlay_panel: OverlayPanel = OverlayPanel(
            config.LAYOUT_PANEL_RECT
        )
        self.network_graph: NetworkGraph = NetworkGraph(
            config.LAYOUT_GRAPH_RECT
        )
        self.help_overlay: HelpOverlay = HelpOverlay(
            config.LAYOUT_GRAPH_RECT
        )
        self.timeline_scrubber: TimelineScrubber = TimelineScrubber(
            config.LAYOUT_SCRUBBER_RECT
        )
        self.input_controller: InputController = InputController()
        self.live_view_presenter: LiveViewPresenter = LiveViewPresenter()

        self.active_gen: int = 0
        self.active_frame: int = 0
        self.active_frame_float: float = 0.0
        self.is_live_mode: bool = False

    def run(self) -> None:
        """
        Executes interactive GUI event loop and rendering cycle.
        """
        running: bool = True
        while running:
            running = self._handle_events()
            self._update_timeline()
            self._draw_frame()

            pygame.display.flip()
            self.clock.tick(config.FPS)

        self.recorder.flush_replay_memory()
        pygame.quit()

    def _load_window_icon(self) -> None:
        """
        Loads and sets window icon from icon.png if present in project root.
        """
        icon_path: str = "icon.png"
        if not os.path.exists(icon_path):
            return

        try:
            icon_surf: pygame.Surface = pygame.image.load(icon_path)
            pygame.display.set_icon(icon_surf)
        except pygame.error:
            pass

    def _get_scale_and_offset(self) -> Tuple[float, int, int]:
        """
        Computes uniform scale factor and pillarbox/letterbox screen offsets.
        """
        win_w, win_h = self.screen.get_size()
        scale_x: float = float(win_w) / float(config.VIRTUAL_WIDTH)
        scale_y: float = float(win_h) / float(config.VIRTUAL_HEIGHT)
        scale: float = min(scale_x, scale_y)

        scaled_w: int = int(round(config.VIRTUAL_WIDTH * scale))
        scaled_h: int = int(round(config.VIRTUAL_HEIGHT * scale))

        offset_x: int = (win_w - scaled_w) // 2
        offset_y: int = (win_h - scaled_h) // 2

        return scale, offset_x, offset_y

    def _handle_events(self) -> bool:
        """
        Translates screen mouse clicks to virtual surface space and dispatches.
        """
        running: bool = True
        scale, offset_x, offset_y = self._get_scale_and_offset()
        total_f: int = self.max_steps

        gen_data = self.recorder.get_generation_data(self.active_gen)
        telemetry = gen_data.get("telemetry", None)
        num_cand: int = (
            int(telemetry.shape[1])
            if telemetry is not None
            else len(gen_data.get("candidate_frames", []))
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_END:
                self.is_live_mode = not self.is_live_mode
                if self.is_live_mode:
                    self.live_view_presenter.activate()
                continue

            elif self.is_live_mode:
                if event.type in (
                    pygame.MOUSEBUTTONDOWN,
                    pygame.MOUSEBUTTONUP,
                    pygame.MOUSEMOTION,
                ):
                    mx, my = (
                        event.pos
                        if hasattr(event, "pos")
                        else (0, 0)
                    )
                    vx: int = int(round((mx - offset_x) / scale))
                    vy: int = int(round((my - offset_y) / scale))

                    event_dict = dict(event.dict)
                    event_dict["pos"] = (vx, vy)
                    virtual_event = pygame.event.Event(
                        event.type, event_dict
                    )
                    keep_running = self.live_view_presenter.handle_event(
                        virtual_event
                    )
                else:
                    keep_running = self.live_view_presenter.handle_event(
                        event
                    )

                if not keep_running:
                    return False

            else:
                if event.type == pygame.MOUSEWHEEL:
                    self.input_controller._handle_mouse_wheel(
                        event.y, self.timeline_scrubber
                    )

                elif event.type == pygame.KEYDOWN:
                    keep_running, new_g, new_f = (
                        self.input_controller._handle_key_down(
                            event.key,
                            self.viewport_grid,
                            self.timeline_scrubber,
                            self.active_gen,
                            self.active_frame,
                            self.total_gens,
                            total_f,
                            num_cand,
                        )
                    )
                    if not keep_running:
                        return False
                    self._update_active_position(new_g, new_f)

                elif event.type == pygame.KEYUP:
                    self.input_controller._handle_key_up(event.key)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button in (1, 2, 3, 4, 5):
                        mx, my = event.pos
                        vx = int(round((mx - offset_x) / scale))
                        vy = int(round((my - offset_y) / scale))

                        new_gen, new_frame = (
                            self.input_controller._handle_mouse_click(
                                event.button,
                                (vx, vy),
                                self.viewport_grid,
                                self.timeline_scrubber,
                                self.active_gen,
                                self.active_frame,
                                self.total_gens,
                                total_f,
                            )
                        )
                        self._update_active_position(new_gen, new_frame)

                elif event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    vx = int(round((mx - offset_x) / scale))
                    vy = int(round((my - offset_y) / scale))

                    new_g_drag, new_f_drag = (
                        self.timeline_scrubber.handle_mouse_move(
                            (vx, vy), self.total_gens, total_f
                        )
                    )
                    if new_g_drag is not None:
                        self._update_active_position(new_g_drag, 0)
                    if new_f_drag is not None:
                        self._update_active_position(
                            self.active_gen, new_f_drag
                        )

                elif event.type == pygame.MOUSEBUTTONUP:
                    self.timeline_scrubber.handle_mouse_up()

        if not self.is_live_mode:
            held_g, held_f = self.input_controller.update_held_keys(
                self.viewport_grid,
                self.timeline_scrubber,
                self.active_gen,
                self.active_frame,
                self.total_gens,
                total_f,
            )
            self._update_active_position(held_g, held_f)

        return running

    def _update_active_position(
        self, new_gen: int, new_frame: int
    ) -> None:
        """
        Updates active generation and frame index while syncing float state.
        """
        if new_gen != self.active_gen or new_frame != self.active_frame:
            self.active_gen = new_gen
            self.active_frame = new_frame
            self.active_frame_float = float(new_frame)

    def _update_timeline(self) -> None:
        """
        Advances timeline playback or delegates live runner physics update.
        """
        if self.is_live_mode:
            self.live_view_presenter.update()
            return

        total_f: int = self.max_steps
        if self.timeline_scrubber.is_playing:
            self.active_frame_float += float(
                self.timeline_scrubber.playback_speed
            )
            self.active_frame = int(self.active_frame_float)

            if self.active_frame >= total_f:
                if self.timeline_scrubber.repeat_all:
                    if self.active_gen < self.total_gens - 1:
                        self.active_gen += 1
                        self.active_frame = 0
                        self.active_frame_float = 0.0
                    else:
                        self.active_gen = 0
                        self.active_frame = 0
                        self.active_frame_float = 0.0
                else:
                    self.active_frame = 0
                    self.active_frame_float = 0.0

    def _draw_frame(self) -> None:
        """
        Renders UI into virtual surface and scales to screen with letterboxing.
        """
        self.virtual_surface.fill(config.COLOR_BG)

        if self.is_live_mode:
            self.live_view_presenter.draw(self.virtual_surface)
        else:
            gen_data = self.recorder.get_generation_data(self.active_gen)
            cand_idx: int = self.viewport_grid.selected_idx

            telemetry = gen_data.get("telemetry", None)
            num_cand: int = (
                telemetry.shape[1]
                if telemetry is not None
                else len(gen_data.get("candidate_frames", []))
            )

            safe_cand_idx: int = min(cand_idx, num_cand - 1)

            self.viewport_grid.draw_grid(
                self.virtual_surface, gen_data, self.active_frame
            )
            self.overlay_panel.draw_panel(
                self.virtual_surface,
                gen_data,
                safe_cand_idx,
                self.active_frame,
                self.max_steps,
                self.total_gens,
            )

            if self.input_controller.show_help_overlay:
                self.help_overlay.draw_panel(self.virtual_surface)
            else:
                self.network_graph.draw_graph(
                    self.virtual_surface,
                    gen_data,
                    safe_cand_idx,
                    self.active_frame,
                )

            self.timeline_scrubber.draw_controls(
                self.virtual_surface,
                self.active_gen,
                self.total_gens,
                self.active_frame,
                self.max_steps,
                selected_cand_idx=safe_cand_idx,
                gen_history=self.recorder.generations_history,
            )

        scale, offset_x, offset_y = self._get_scale_and_offset()

        scaled_w: int = int(round(config.VIRTUAL_WIDTH * scale))
        scaled_h: int = int(round(config.VIRTUAL_HEIGHT * scale))

        scaled_surf: pygame.Surface = pygame.transform.smoothscale(
            self.virtual_surface, (scaled_w, scaled_h)
        )

        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled_surf, (offset_x, offset_y))
