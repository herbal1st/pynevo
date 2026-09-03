"""
Orchestrates full-canvas atmosphere surface and dynamic light pass.
"""

from typing import Tuple, Optional
import pygame

import config
from entities.lighting_profile_registry import (
    LightingProfileRegistry,
    ResolvedLightingProfile,
)
from world.lighting.time_clock import DayNightClock
from world.lighting.ambient_palette import AmbientPaletteResolver
from world.lighting.viewport_height_sampler import ViewportHeightSampler
from world.lighting.height_shadow_engine import (
    VectorizedHeightShadowEngine
)
from world.generation.endless_noise import EndlessNoiseGenerator
from world.chunk_manager import ChunkManager
from world.tile_registry import TileRegistry


class AtmosphereOverlayManager:
    """
    Manages screenwide atmosphere overlay surface and time progression.
    """

    def __init__(
        self,
        profile_name: Optional[str] = None,
        width: int = config.VIRTUAL_WIDTH,  # pixels
        height: int = config.VIRTUAL_HEIGHT  # pixels
    ) -> None:
        """
        Initializes registry profile, clock, sampler, & alpha canvas.
        """
        self.width: int = width  # pixels
        self.height: int = height  # pixels

        p_name: str = profile_name or getattr(
            config, "ACTIVE_LIGHTING_PROFILE", "DEFAULT"
        )
        self.registry: LightingProfileRegistry = (
            LightingProfileRegistry()
        )
        self.profile: ResolvedLightingProfile = (
            self.registry.get_profile(p_name)
        )

        self.clock: DayNightClock = DayNightClock(
            day_cycle_duration=self.profile.day_cycle_duration,
            start_time_ratio=self.profile.start_time_ratio,
            start_light_angle_deg=self.profile.start_light_angle_deg
        )
        self.tile_registry: TileRegistry = TileRegistry()
        self.surface: pygame.Surface = pygame.Surface(
            (width, height)
        )
        self.sampler: ViewportHeightSampler = ViewportHeightSampler()

    def update(self, delta_time: float) -> None:
        """
        Advances time clock by frame delta time.
        """
        self.clock.update(delta_time)

    def draw_overlay(
        self,
        target_surface: pygame.Surface,
        focus_x: float,  # tiles
        focus_y: float,  # tiles
        vw_tiles: float,  # tiles
        vh_tiles: float,  # tiles
        chunk_manager: ChunkManager,
        generator: EndlessNoiseGenerator,
        effective_tile_sz: float  # pixels
    ) -> None:
        """
        Executes direct 3-pass canvas compositing sequence onto target.
        """
        heights, tile_ids, min_tx, min_ty = (
            self.sampler.sample_viewport_heights(
                focus_x,
                focus_y,
                vw_tiles,
                vh_tiles,
                chunk_manager,
                generator
            )
        )

        active_angle: float = self.clock.solar_angle_rad

        hl_surf, sh_surf = (
            VectorizedHeightShadowEngine.compute_shading_surfaces(
                heights,
                tile_ids,
                active_angle,
                self.profile,
                effective_tile_sz,
                self.tile_registry
            )
        )

        center_px: float = self.width / 2.0  # pixels
        center_py: float = self.height / 2.0  # pixels

        start_tile_x: int = min_tx + 1  # tile_x
        start_tile_y: int = min_ty + 1  # tile_y

        px_x: int = int(
            round(
                center_px + (float(start_tile_x) - focus_x) * effective_tile_sz
            )
        )
        px_y: int = int(
            round(
                center_py + (float(start_tile_y) - focus_y) * effective_tile_sz
            )
        )

        # Pass 2A: Subtractive Mountain Shadows
        target_surface.blit(sh_surf, (px_x, px_y))

        # Pass 2B: Tile-Aware Additive Highlights directly on viewport canvas!
        target_surface.blit(
            hl_surf, (px_x, px_y), special_flags=pygame.BLEND_ADD
        )

        # Pass 3: Multiplicative Ambient Day/Night Tint Pass
        rgb: Tuple[int, int, int] = (
            AmbientPaletteResolver.resolve_ambient_color(
                self.clock.normalized_time, self.profile
            )
        )

        if rgb != (255, 255, 255):
            self.surface.fill((rgb[0], rgb[1], rgb[2]))
            target_surface.blit(
                self.surface, (0, 0), special_flags=pygame.BLEND_MULT
            )
