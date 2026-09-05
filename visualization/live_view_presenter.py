"""
Dedicated live winner view presenter managing multi-champion swarm UI and loop rendering.
"""

from typing import Tuple, Dict, Any, List, Optional
import pygame

import config
from bridges.live_winner_runner import LiveWinnerRunner
from visualization.map_renderer import MapRenderer
from visualization.live_help_overlay import LiveHelpOverlay
from visualization.overlay_panel import OverlayPanel
from visualization.timeline_scrubber import TimelineScrubber
from visualization.network_graph.graph_facade import NetworkGraph
from visualization.viewports.native_maze_viewport import NativeMazeViewport


class LiveViewPresenter:
    """
    Manages live multi-champion swarm view, inputs, background cache, & auto-regeneration.
    """

    def __init__(
        self,
        virtual_width: int = config.VIRTUAL_WIDTH,
        virtual_height: int = config.VIRTUAL_HEIGHT,
    ) -> None:
        self.runner: LiveWinnerRunner = LiveWinnerRunner()
        self.map_renderer: MapRenderer = MapRenderer()
        self.viewport: NativeMazeViewport = NativeMazeViewport(virtual_width, virtual_height)

        self.overlay_panel: OverlayPanel = OverlayPanel(config.LAYOUT_PANEL_RECT)
        self.network_graph: NetworkGraph = NetworkGraph(config.LAYOUT_GRAPH_RECT)
        self.help_overlay: LiveHelpOverlay = LiveHelpOverlay(config.LAYOUT_GRAPH_RECT)
        self.timeline_scrubber: TimelineScrubber = TimelineScrubber(config.LAYOUT_SCRUBBER_RECT)

        self.is_camera_centered: bool = False
        self.show_help_overlay: bool = False
        self.active_frame: int = 0
        self.active_frame_float: float = 0.0

        self.held_nav_key: Optional[int] = None
        self.held_key_press_time: int = 0
        self.held_key_last_repeat_time: int = 0

    def activate(self) -> None:
        self.runner.load_winner_brain(verbose=False)
        self.generate_fresh_maze()

    def generate_fresh_maze(self) -> None:
        self.runner.generate_fresh_maze()
        self.map_renderer.clear_cache()
        self.active_frame = 0
        self.active_frame_float = 0.0
        self.held_nav_key = None
        self.timeline_scrubber.is_playing = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        total_steps: int = max(1, self.runner.total_run_steps)
        sp: float = self.timeline_scrubber.playback_speed

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                now_ms = pygame.time.get_ticks()
                self.held_nav_key = event.key
                self.held_key_press_time = now_ms
                self.held_key_last_repeat_time = now_ms
                self._execute_frame_jump(event.key, total_steps, sp)

            elif event.key == pygame.K_UP:
                self._execute_brain_cycle(1)
            elif event.key == pygame.K_DOWN:
                self._execute_brain_cycle(-1)

            # '[' and ']' dynamically scale champion swarm count
            elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS, pygame.K_KP_MINUS):
                self.runner.set_champion_count(self.runner.num_champions - 5)
            elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                self.runner.set_champion_count(self.runner.num_champions + 5)

            elif event.key == pygame.K_SPACE:
                self.timeline_scrubber.is_playing = not self.timeline_scrubber.is_playing
            elif event.key == pygame.K_r:
                self.generate_fresh_maze()
            elif event.key == pygame.K_TAB:
                self.is_camera_centered = not self.is_camera_centered
            elif event.key == pygame.K_h:
                self.show_help_overlay = not self.show_help_overlay
            elif event.key == pygame.K_ESCAPE:
                return False

        elif event.type == pygame.KEYUP:
            if event.key == self.held_nav_key:
                self.held_nav_key = None

        elif event.type == pygame.MOUSEBUTTONDOWN:
            vx, vy = event.pos
            if event.button in (1, 2, 3):
                clicked_gen, clicked_frame = self.timeline_scrubber.handle_click(
                    (vx, vy), total_gens=1, total_frames=total_steps, mouse_button=event.button
                )
                if clicked_frame is not None:
                    self.active_frame = clicked_frame
                    self.active_frame_float = float(clicked_frame)

            if event.button == 3:
                gx, gy, gw, gh = config.LAYOUT_GRAPH_RECT
                if gx <= vx <= gx + gw and gy <= vy <= gy + gh:
                    self.show_help_overlay = not self.show_help_overlay
                else:
                    self.is_camera_centered = not self.is_camera_centered

        elif event.type == pygame.MOUSEMOTION:
            vx, vy = event.pos
            _, new_f_drag = self.timeline_scrubber.handle_mouse_move(
                (vx, vy), total_gens=1, total_frames=total_steps
            )
            if new_f_drag is not None:
                self.active_frame = new_f_drag
                self.active_frame_float = float(new_f_drag)

        elif event.type == pygame.MOUSEBUTTONUP:
            self.timeline_scrubber.handle_mouse_up()

        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0:
                self.timeline_scrubber.step_speed_up()
            elif event.y < 0:
                self.timeline_scrubber.step_speed_down()

        return True

    def update(self) -> None:
        total_steps: int = max(1, self.runner.total_run_steps)
        sp: float = self.timeline_scrubber.playback_speed

        if not self.timeline_scrubber.is_playing:
            return

        self.active_frame_float += sp
        self.active_frame = int(self.active_frame_float)

        # Autogen next map when simulation ends (all died, solved, or hit limit)
        if self.active_frame >= total_steps:
            self.generate_fresh_maze()

    def draw(self, surface: pygame.Surface) -> None:
        live_gen_data = self.runner.to_gen_data_adapter()
        total_steps = max(1, self.runner.total_run_steps)
        safe_frame = max(0, min(self.active_frame, total_steps - 1))

        rect = config.LAYOUT_GRID_RECT
        # Render all champion models overlaid onto the fresh map
        self.viewport.render_overlay_viewport(
            surface,
            live_gen_data,
            selected_cand_idx=live_gen_data.get("winner_index", 0),
            active_step=safe_frame,
            rect=rect,
            is_camera_centered=self.is_camera_centered
        )

        title_str = f"{self.runner.active_brain_title} ({self.runner.num_champions} Clones)"
        self.overlay_panel.draw_panel(
            surface,
            live_gen_data,
            active_cand_idx=live_gen_data.get("winner_index", 0),
            active_step=safe_frame,
            total_steps=total_steps,
            total_gens=1,
            active_profile_title=title_str,
        )

        if self.show_help_overlay:
            self.help_overlay.draw_panel(surface)
        else:
            live_activations = self.runner.get_activations_for_step(safe_frame)
            self.network_graph.draw_live_graph(surface, live_activations)

        self.timeline_scrubber.draw_controls(
            surface,
            active_gen=0,
            total_gens=1,
            active_frame=safe_frame,
            total_frames=total_steps,
            selected_cand_idx=live_gen_data.get("winner_index", 0),
            gen_history=[live_gen_data],
        )

    def _execute_brain_cycle(self, delta: int) -> None:
        self.runner.cycle_brain(delta)
        self.map_renderer.clear_cache()
        self.active_frame = 0
        self.active_frame_float = 0.0
        self.timeline_scrubber.is_playing = True

    def _execute_frame_jump(self, key: int, total_steps: int, speed: float) -> None:
        jump = max(1, int(total_steps * config.TIMELINE_FRAME_JUMP_RATIO * speed))
        if key == pygame.K_LEFT:
            self.active_frame = max(0, self.active_frame - jump)
        elif key == pygame.K_RIGHT:
            self.active_frame = min(total_steps - 1, self.active_frame + jump)
        self.active_frame_float = float(self.active_frame)
