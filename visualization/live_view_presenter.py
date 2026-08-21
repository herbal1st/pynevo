"""
Dedicated live winner view presenter managing UI rendering and events.
"""

from typing import Tuple, Dict, Any, List, Optional
import pygame

import config
from core.map_data import MapData
from bridges.live_winner_runner import LiveWinnerRunner
from visualization.map_renderer import MapRenderer
from visualization.vision_renderer import VisionRenderer
from visualization.live_help_overlay import LiveHelpOverlay
from visualization.overlay_panel import OverlayPanel
from visualization.timeline_scrubber import TimelineScrubber
from visualization.network_graph.graph_facade import NetworkGraph
from visualization.viewports.native.state_resolver import (
    ViewportStateResolver,
)
from visualization.viewports.native.tile_renderer import (
    ViewportTileRenderer,
)
from visualization.viewports.native.avatar_renderer import (
    ViewportAvatarRenderer,
)
from visualization.viewports.native.hud_overlay_renderer import (
    ViewportHUDOverlayRenderer,
)


class LiveViewPresenter:
    """
    Manages live mode UI view, inputs, background cache, & scrubbing.
    """

    def __init__(
        self,
        virtual_width: int = config.VIRTUAL_WIDTH,
        virtual_height: int = config.VIRTUAL_HEIGHT,
    ) -> None:
        """
        Initializes live runner, isolated map renderer, & UI sub-renderers.
        """
        self.runner: LiveWinnerRunner = LiveWinnerRunner()
        self.map_renderer: MapRenderer = MapRenderer()
        self.state_resolver: ViewportStateResolver = (
            ViewportStateResolver()
        )
        self.tile_renderer: ViewportTileRenderer = ViewportTileRenderer()
        self.vision_renderer: VisionRenderer = VisionRenderer(
            virtual_width, virtual_height
        )
        self.avatar_renderer: ViewportAvatarRenderer = (
            ViewportAvatarRenderer()
        )
        self.hud_renderer: ViewportHUDOverlayRenderer = (
            ViewportHUDOverlayRenderer(virtual_width, virtual_height)
        )

        self.overlay_panel: OverlayPanel = OverlayPanel(
            config.LAYOUT_PANEL_RECT
        )
        self.network_graph: NetworkGraph = NetworkGraph(
            config.LAYOUT_GRAPH_RECT
        )
        self.help_overlay: LiveHelpOverlay = LiveHelpOverlay(
            config.LAYOUT_GRAPH_RECT
        )
        self.timeline_scrubber: TimelineScrubber = TimelineScrubber(
            config.LAYOUT_SCRUBBER_RECT
        )

        self.is_player_centered: bool = False
        self.show_help_overlay: bool = False
        self.active_frame: int = 0
        self.active_frame_float: float = 0.0

        self.held_nav_key: Optional[int] = None
        self.held_key_press_time: int = 0
        self.held_key_last_repeat_time: int = 0

    def activate(self) -> None:
        """
        Activates live mode, reloads weights, & flushes background cache.
        """
        self.runner.load_winner_brain(verbose=False)
        self.generate_fresh_maze()

    def generate_fresh_maze(self) -> None:
        """
        Generates fresh maze, executes upfront run, & resumes auto-play.
        """
        self.runner.generate_fresh_maze()
        self.map_renderer.clear_cache()
        self.active_frame = 0
        self.active_frame_float = 0.0
        self.held_nav_key = None
        self.timeline_scrubber.is_playing = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Processes live mode keyboard and mouse input events.
        """
        total_steps: int = max(1, self.runner.total_run_steps)
        sp: float = self.timeline_scrubber.playback_speed

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                now_ms: int = pygame.time.get_ticks()
                self.held_nav_key = event.key
                self.held_key_press_time = now_ms
                self.held_key_last_repeat_time = now_ms
                self._execute_frame_jump(event.key, total_steps, sp)

            elif event.key == pygame.K_UP:
                self._execute_brain_cycle(1)
            elif event.key == pygame.K_DOWN:
                self._execute_brain_cycle(-1)

            elif event.key == pygame.K_SPACE:
                self.timeline_scrubber.is_playing = (
                    not self.timeline_scrubber.is_playing
                )
            elif event.key == pygame.K_r:
                self.generate_fresh_maze()
            elif event.key == pygame.K_TAB:
                self.is_player_centered = not self.is_player_centered
            elif event.key == pygame.K_h:
                self.show_help_overlay = not self.show_help_overlay
            elif event.key == pygame.K_ESCAPE:
                return False
            elif event.key in (
                pygame.K_PAGEUP,
                pygame.K_PLUS,
                pygame.K_KP_PLUS,
                pygame.K_EQUALS,
            ):
                self.timeline_scrubber.step_speed_up()
            elif event.key in (
                pygame.K_PAGEDOWN,
                pygame.K_MINUS,
                pygame.K_KP_MINUS,
            ):
                self.timeline_scrubber.step_speed_down()
            elif event.key in (pygame.K_PERIOD, pygame.K_KP_PERIOD):
                self.timeline_scrubber.reset_speed()

        elif event.type == pygame.KEYUP:
            if event.key == self.held_nav_key:
                self.held_nav_key = None

        elif event.type == pygame.MOUSEBUTTONDOWN:
            vx, vy = event.pos
            if event.button in (1, 2, 3):
                clicked_gen, clicked_frame = (
                    self.timeline_scrubber.handle_click(
                        (vx, vy),
                        total_gens=1,
                        total_frames=total_steps,
                        mouse_button=event.button,
                    )
                )
                if clicked_frame is not None:
                    self.active_frame = clicked_frame
                    self.active_frame_float = float(clicked_frame)

            if event.button == 3:
                gx, gy, gw, gh = config.LAYOUT_GRAPH_RECT
                if gx <= vx <= gx + gw and gy <= vy <= gy + gh:
                    self.show_help_overlay = not self.show_help_overlay
                else:
                    self.is_player_centered = not self.is_player_centered

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
        """
        Advances scrubbed frame pointer during playback and key repeats.
        """
        total_steps: int = max(1, self.runner.total_run_steps)
        sp: float = self.timeline_scrubber.playback_speed

        if (
            self.held_nav_key is not None
            and self.held_nav_key in (pygame.K_LEFT, pygame.K_RIGHT)
        ):
            now_ms: int = pygame.time.get_ticks()
            delay: int = config.TIMELINE_KEY_REPEAT_DELAY_MS
            interval: int = config.TIMELINE_KEY_REPEAT_INTERVAL_MS

            if (
                now_ms - self.held_key_press_time >= delay
                and now_ms - self.held_key_last_repeat_time >= interval
            ):
                self.held_key_last_repeat_time = now_ms
                self._execute_frame_jump(
                    self.held_nav_key, total_steps, sp
                )

        if not self.timeline_scrubber.is_playing:
            return

        self.active_frame_float += sp
        self.active_frame = int(self.active_frame_float)

        if self.active_frame >= total_steps:
            self.active_frame = total_steps - 1
            self.active_frame_float = float(self.active_frame)
            self.timeline_scrubber.is_playing = False

    def draw(self, surface: pygame.Surface) -> None:
        """
        Renders live viewports, telemetry panel, activation graph, & scrubber.
        """
        live_gen_data: Dict[str, Any] = self.runner.to_gen_data_adapter()
        total_steps: int = max(1, self.runner.total_run_steps)
        safe_frame: int = max(0, min(self.active_frame, total_steps - 1))

        rect: Tuple[int, int, int, int] = config.LAYOUT_GRID_RECT
        self._draw_live_viewport(surface, rect, live_gen_data, safe_frame)

        self.overlay_panel.draw_panel(
            surface,
            live_gen_data,
            active_cand_idx=0,
            active_step=safe_frame,
            total_steps=total_steps,
            total_gens=1,
            active_profile_title=self.runner.active_brain_title,
        )

        if self.show_help_overlay:
            self.help_overlay.draw_panel(surface)
        else:
            live_activations: List[
                List[float]
            ] = self.runner.get_activations_for_step(safe_frame)
            self.network_graph.draw_live_graph(
                surface, live_activations
            )

        self.timeline_scrubber.draw_controls(
            surface,
            active_gen=0,
            total_gens=1,
            active_frame=safe_frame,
            total_frames=total_steps,
            selected_cand_idx=0,
            gen_history=[live_gen_data],
        )

    def _execute_brain_cycle(self, delta: int) -> None:
        """
        Triggers runner brain cycling, clears map cache, & resumes auto-play.
        """
        self.runner.cycle_brain(delta)
        self.map_renderer.clear_cache()
        self.active_frame = 0
        self.active_frame_float = 0.0
        self.timeline_scrubber.is_playing = True

    def _execute_frame_jump(
        self, key: int, total_steps: int, speed: float
    ) -> None:
        """
        Jumps active frame index backward or forward based on arrow key.
        """
        jump: int = max(
            1, int(total_steps * config.TIMELINE_FRAME_JUMP_RATIO * speed)
        )
        if key == pygame.K_LEFT:
            self.active_frame = max(0, self.active_frame - jump)
        elif key == pygame.K_RIGHT:
            self.active_frame = min(total_steps - 1, self.active_frame + jump)
        self.active_frame_float = float(self.active_frame)

    def _draw_live_viewport(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        gen_data: Dict[str, Any],
        active_step: int,
    ) -> None:
        """
        Renders native live sub-viewport into bounding rect.
        """
        frame_state = self.state_resolver.resolve_frame_state(
            gen_data, 0, active_step
        )
        if frame_state is None:
            return

        rx, ry, rw, rh = rect
        map_w: int = int(gen_data.get("map_width", 40))
        map_h: int = int(gen_data.get("map_height", 30))
        map_data = MapData(
            map_w, map_h, gen_data["start_pos"], gen_data["exit_pos"]
        )
        map_data.decode_bitmask(gen_data["bitmask_chunks"])

        clip_rect = pygame.Rect(rx, ry, rw, rh)
        surface.set_clip(clip_rect)
        pygame.draw.rect(surface, config.COLOR_BG, rect)

        rows: int = config.GRID_ROWS
        cols: int = config.GRID_COLS
        base_sub_w: float = float(surface.get_width()) / float(cols)
        ui_scale: float = (float(rw) / base_sub_w) * 0.5

        tile_size, origin_pixel = self._draw_live_tiles(
            surface, rect, map_data, frame_state.x, frame_state.y, rows, cols
        )

        self.vision_renderer.draw_vision_arc(
            surface,
            rx,
            ry,
            rw,
            rh,
            frame_state.x,
            frame_state.y,
            frame_state.heading,
            origin_pixel,
            tile_size,
            self.is_player_centered,
            map_data,
        )

        self.avatar_renderer.draw_avatar(
            surface,
            origin_pixel,
            tile_size,
            frame_state,
            is_selected=True,
            ui_scale=ui_scale,
            active_step=active_step,
        )

        self.hud_renderer.draw_hud_overlays(
            surface,
            rect,
            frame_state,
            ui_scale,
            show_scorecard=True,
        )

        surface.set_clip(None)
        self.hud_renderer.draw_viewport_borders(
            surface, rect, frame_state, is_selected=True
        )

    def _draw_live_tiles(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        map_data: MapData,
        cx: float,
        cy: float,
        rows: int,
        cols: int,
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Renders live background tiles and returns (tile_size, origin_pixel).
        """
        rx, ry, rw, rh = rect
        tile_size: float = self.tile_renderer.map_profile.tile_size

        if self.is_player_centered:
            tile_size = float(tile_size) * config.PLAYER_CAMERA_ZOOM
            center_px: float = float(rx) + (float(rw) / 2.0)
            center_py: float = float(ry) + (float(rh) / 2.0)
            self.tile_renderer._draw_player_centered_tiles(
                surface, rect, map_data, cx, cy, tile_size
            )
            origin_pixel = (int(round(center_px)), int(round(center_py)))
        else:
            tile_size = min(
                float(rw) / float(map_data.width),
                float(rh) / float(map_data.height),
            )
            bg_surf = self.map_renderer.get_rendered_map_surface(
                map_data, gen_idx=-1, rw=rw, rh=rh
            )
            surface.blit(bg_surf, (rx, ry))
            origin_pixel = (
                int(rx + (cx * tile_size)),
                int(ry + (cy * tile_size)),
            )

        return tile_size, origin_pixel
