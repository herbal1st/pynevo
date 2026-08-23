"""
Dedicated full-screen visualizer window runner for endless spatial worlds.
"""

import os
import sys
from typing import Tuple
import pygame

import config
from entities.map_endless_profile_registry import (
    MapEndlessProfileRegistry,
    ResolvedMapEndlessProfile,
)
from world.tile_registry import TileRegistry
from world.chunk_manager import ChunkManager
from world.generation.endless_noise import EndlessNoiseGenerator
from visualization.viewports.native.tile_renderer import (
    ViewportTileRenderer,
)


class EndlessAppWindow:
    """
    Manages display lifecycle and endless map rendering across full canvas.
    """

    def __init__(self) -> None:
        """
        Initializes Pygame window, endless engine, and tile renderer.
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
        self.tile_renderer: ViewportTileRenderer = ViewportTileRenderer()

        self.tile_registry: TileRegistry = TileRegistry()
        self.endless_map_registry: MapEndlessProfileRegistry = (
            MapEndlessProfileRegistry()
        )

        p_name: str = getattr(
            config, "ACTIVE_ENDLESS_MAP_PROFILE", "CAVERN"
        )
        self.endless_profile: ResolvedMapEndlessProfile = (
            self.endless_map_registry.get_profile(p_name)
        )

        self.endless_generator: EndlessNoiseGenerator = (
            EndlessNoiseGenerator(
                self.endless_profile, self.tile_registry
            )
        )
        self.chunk_manager: ChunkManager = ChunkManager(
            self.tile_registry, self.endless_profile.world_seed
        )

        self.endless_focus_x: float = 0.0
        self.endless_focus_y: float = 0.0

    def run(self) -> None:
        """
        Executes event loop and full-canvas endless world render cycle.
        """
        running: bool = True
        while running:
            running = self._handle_events()
            self._draw_frame()

            pygame.display.flip()
            self.clock.tick(config.FPS)

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
        Polls keyboard input for smooth camera panning and window close.
        """
        pan_speed: float = 0.5
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.endless_focus_x -= pan_speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.endless_focus_x += pan_speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.endless_focus_y -= pan_speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.endless_focus_y += pan_speed

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
        Renders endless tilemap across full virtual canvas (1280x720).
        """
        self.virtual_surface.fill(config.COLOR_BG)

        canvas_rect: Tuple[int, int, int, int] = (
            0, 0, config.VIRTUAL_WIDTH, config.VIRTUAL_HEIGHT
        )
        vw_tiles: float = float(config.VIRTUAL_WIDTH) / float(
            self.endless_profile.tile_size
        )
        vh_tiles: float = float(config.VIRTUAL_HEIGHT) / float(
            self.endless_profile.tile_size
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
            tile_size_base=float(self.endless_profile.tile_size)
        )

        scale, offset_x, offset_y = self._get_scale_and_offset()

        scaled_w: int = int(round(config.VIRTUAL_WIDTH * scale))
        scaled_h: int = int(round(config.VIRTUAL_HEIGHT * scale))

        scaled_surf: pygame.Surface = pygame.transform.smoothscale(
            self.virtual_surface, (scaled_w, scaled_h)
        )

        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled_surf, (offset_x, offset_y))
