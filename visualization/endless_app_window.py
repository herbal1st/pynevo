"""
Full-canvas endless visualizer window hosting human player.
"""

import math
import os
import sys
from typing import Tuple, Optional
import pygame

import config
from entities.player_profile_registry import (
    PlayerProfileRegistry,
    ResolvedPlayerProfile
)
from entities.skin_profile_registry import (
    SkinProfileRegistry,
    ResolvedSkinProfile
)
from entities.player_controller import PlayerController
from entities.map_endless_profile_registry import (
    MapEndlessProfileRegistry,
    ResolvedMapEndlessProfile
)
from world.tile_registry import TileRegistry
from world.chunk_manager import ChunkManager
from world.spawn_solver import EndlessSpawnSolver
from world.generation.endless_noise import EndlessNoiseGenerator
from world.lighting.atmosphere_overlay import AtmosphereOverlayManager
from visualization.viewports.native.tile_renderer import (
    ViewportTileRenderer
)
from visualization.viewports.native.avatar_renderer import (
    ViewportAvatarRenderer
)
from visualization.viewports.native.state_resolver import (
    ViewportFrameState
)


class EndlessAppWindow:
    """
    Manages full-canvas display, endless map streaming, & player movement.
    """

    def __init__(self) -> None:
        """
        Initializes Pygame, endless engine, safe spawn, & player controller.
        """
        pygame.init()

        flags: int = (
            pygame.RESIZABLE if config.USE_RESIZABLE_WINDOW else 0
        )
        self.screen: pygame.Surface = pygame.display.set_mode(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), flags
        )
        pygame.display.set_caption("PyNevo - Endless World Engine")
        self._load_window_icon()

        self.virtual_surface: pygame.Surface = pygame.Surface(
            (config.VIRTUAL_WIDTH, config.VIRTUAL_HEIGHT)
        )
        self.clock: pygame.time.Clock = pygame.time.Clock()

        self.tile_registry: TileRegistry = TileRegistry()
        self.endless_map_registry: MapEndlessProfileRegistry = (
            MapEndlessProfileRegistry()
        )
        self.player_registry: PlayerProfileRegistry = (
            PlayerProfileRegistry()
        )
        self.skin_registry: SkinProfileRegistry = SkinProfileRegistry()

        p_endless_name: str = getattr(
            config, "ACTIVE_ENDLESS_MAP_PROFILE", "SIMPLEX"
        )
        self.endless_profile: ResolvedMapEndlessProfile = (
            self.endless_map_registry.get_profile(p_endless_name)
        )

        self.endless_generator: EndlessNoiseGenerator = (
            EndlessNoiseGenerator(
                self.endless_profile, self.tile_registry
            )
        )
        self.chunk_manager: ChunkManager = ChunkManager(
            self.tile_registry, self.endless_profile.world_seed
        )

        p_player_name: str = getattr(
            config, "ACTIVE_PLAYER_PROFILE", "DEFAULT"
        )
        self.player_profile: ResolvedPlayerProfile = (
            self.player_registry.get_profile(p_player_name)
        )
        self.skin_profile: ResolvedSkinProfile = (
            self.skin_registry.get_skin(
                self.player_profile.skin_profile
            )
        )

        spawn_x, spawn_y = EndlessSpawnSolver.find_safe_spawn(
            self.chunk_manager,
            self.tile_registry,
            self.endless_generator,
            center_x=0.0,
            center_y=0.0,
            diameter_ratio=self.player_profile.diameter_ratio,
            min_speed_mult=self.player_profile.min_spawn_speed
        )

        self.player: PlayerController = PlayerController(
            self.player_profile, spawn_x, spawn_y
        )

        self.endless_focus_x: float = spawn_x
        self.endless_focus_y: float = spawn_y

        self.tile_renderer: ViewportTileRenderer = ViewportTileRenderer()
        self.avatar_renderer: ViewportAvatarRenderer = (
            ViewportAvatarRenderer()
        )

        p_lighting_name: str = getattr(
            config, "ACTIVE_LIGHTING_PROFILE", "DEFAULT"
        )
        self.lighting_overlay: AtmosphereOverlayManager = (
            AtmosphereOverlayManager(profile_name=p_lighting_name)
        )

    def run(self) -> None:
        """
        Executes event loop and full-canvas endless world render cycle.
        """
        running: bool = True
        while running:
            dt: float = float(self.clock.tick(config.FPS)) / 1000.0
            self.lighting_overlay.update(dt)

            running = self._handle_events()
            self._draw_frame()

            pygame.display.flip()

        pygame.quit()

    def _load_window_icon(self) -> None:
        """
        Loads and sets window icon from icon.png if present in root.
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
        Computes uniform scale factor and screen letterboxing offsets.
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
        Polls events, updates player input movement, & syncs camera focus.
        """
        keys = pygame.key.get_pressed()
        self.player.update(
            keys,
            self.chunk_manager,
            self.tile_registry,
            self.endless_generator,
            fps=config.FPS
        )

        self.endless_focus_x = self.player.x
        self.endless_focus_y = self.player.y

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )
            elif (
                event.type == pygame.KEYDOWN and
                event.key == pygame.K_ESCAPE
            ):
                return False

        return True

    def _draw_frame(self) -> None:
        """
        Renders endless tilemap, avatar, & dynamic atmosphere overlay.
        """
        self.virtual_surface.fill(config.COLOR_BG)

        canvas_rect: Tuple[int, int, int, int] = (
            0, 0, config.VIRTUAL_WIDTH, config.VIRTUAL_HEIGHT
        )
        base_tile_sz: float = float(self.endless_profile.tile_size)
        effective_tile_sz: float = (
            base_tile_sz * self.skin_profile.camera_zoom
        )

        vw_tiles: float = (
            float(config.VIRTUAL_WIDTH) / max(1.0, effective_tile_sz)
        )
        vh_tiles: float = (
            float(config.VIRTUAL_HEIGHT) / max(1.0, effective_tile_sz)
        )

        self.chunk_manager.update_focus(
            self.endless_focus_x,
            self.endless_focus_y,
            vw_tiles,
            vh_tiles,
            self.endless_generator
        )

        self.tile_renderer.draw_endless_tiles(
            self.virtual_surface,
            canvas_rect,
            self.chunk_manager,
            self.endless_focus_x,
            self.endless_focus_y,
            tile_size_base=base_tile_sz,
            camera_zoom=self.skin_profile.camera_zoom
        )

        center_px: int = config.VIRTUAL_WIDTH // 2
        center_py: int = config.VIRTUAL_HEIGHT // 2

        self._draw_player_heading_indicator(
            center_px, center_py, effective_tile_sz
        )

        face_text: str = (
            self.skin_profile.face_wall
            if self.player.last_collided
            else self.skin_profile.face_walk
        )

        frame_state = ViewportFrameState(
            cand_idx=0,
            frame_idx=0,
            x=self.player.x,
            y=self.player.y,
            heading=self.player.heading,
            health=1.0,
            dist=0,
            hit_wall=self.player.last_collided,
            is_alive=True,
            reached_exit=False,
            speed_ratio=1.0,
            is_idle=False,
            is_healing=False,
            net_delta=0.0,
            face_str=face_text,
            score_val=0,
            radius_ratio=self.player_profile.radius_ratio,
            skin=self.skin_profile
        )

        self.avatar_renderer.draw_avatar(
            self.virtual_surface,
            (center_px, center_py),
            effective_tile_sz,
            frame_state,
            is_selected=False,
            ui_scale=1.0,
            active_step=0
        )

        self.lighting_overlay.draw_overlay(
            self.virtual_surface,
            self.endless_focus_x,
            self.endless_focus_y,
            vw_tiles,
            vh_tiles,
            self.chunk_manager,
            self.endless_generator,
            effective_tile_sz
        )

        scale, offset_x, offset_y = self._get_scale_and_offset()

        scaled_w: int = int(round(config.VIRTUAL_WIDTH * scale))
        scaled_h: int = int(round(config.VIRTUAL_HEIGHT * scale))

        scaled_surf: pygame.Surface = pygame.transform.smoothscale(
            self.virtual_surface, (scaled_w, scaled_h)
        )

        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled_surf, (offset_x, offset_y))

    def _draw_player_heading_indicator(
        self,
        center_px: int,
        center_py: int,
        effective_tile_sz: float
    ) -> None:
        """
        Renders directional heading line pointing along player orientation.
        """
        body_radius_px: float = (
            effective_tile_sz * self.player_profile.radius_ratio
        )
        line_ext_px: float = (
            self.skin_profile.heading_line_length * effective_tile_sz
        )
        line_len: float = body_radius_px + line_ext_px

        hx: int = int(
            center_px + (math.cos(self.player.heading) * line_len)
        )
        hy: int = int(
            center_py + (math.sin(self.player.heading) * line_len)
        )

        pygame.draw.line(
            self.virtual_surface,
            self.skin_profile.color_heading_line,
            (center_px, center_py),
            (hx, hy),
            self.skin_profile.heading_line_width
        )
