"""
Computes 2D Tile-Aware base-color highlights and mountain shadow masks.
"""

import math
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import pygame

from entities.lighting_profile_registry import ResolvedLightingProfile
from world.tile_registry import TileRegistry


class VectorizedHeightShadowEngine:
    """
    Calculates Tile-Aware base-color highlights and mountain shadows.
    """

    _tile_rgb_table: Optional[NDArray[np.uint8]] = None

    @classmethod
    def _get_rgb_table(
        cls, tile_registry: TileRegistry
    ) -> NDArray[np.uint8]:
        """
        Pre-caches O(1) RGB lookup table mapping tile IDs to base colors.
        """
        if cls._tile_rgb_table is None:
            table: NDArray[np.uint8] = np.zeros(
                (256, 3), dtype=np.uint8
            )
            for t_id in range(256):
                if t_id in tile_registry._by_id:
                    col = tile_registry.get_tile(t_id).color
                    table[t_id] = [col[0], col[1], col[2]]
            cls._tile_rgb_table = table
        return cls._tile_rgb_table

    @classmethod
    def compute_shading_surfaces(
        cls,
        height_matrix: NDArray[np.float32],
        tile_id_matrix: NDArray[np.uint8],
        solar_angle_rad: float,  # radians
        profile: ResolvedLightingProfile,
        tile_size_px: float,  # pixels
        tile_registry: TileRegistry
    ) -> Tuple[pygame.Surface, pygame.Surface]:
        """
        Generates Tile-Aware additive highlight surface & shadow surface.
        """
        grid_h, grid_w = height_matrix.shape  # tiles
        if grid_h < 3 or grid_w < 3:
            dummy = pygame.Surface((1, 1), pygame.SRCALPHA)
            return dummy, dummy

        gy, gx = np.gradient(height_matrix)

        lx: float = math.cos(solar_angle_rad)  # ratio
        ly: float = math.sin(solar_angle_rad)  # ratio

        dot_prod: NDArray[np.float32] = (
            (gx * lx) + (gy * ly)
        ).astype(np.float32)

        steepness: float = profile.terrain_steepness  # multiplier
        inner_dot = dot_prod[1:-1, 1:-1]
        inner_ids = tile_id_matrix[1:-1, 1:-1]
        inner_h, inner_w = inner_dot.shape  # tiles

        highlights = np.maximum(0.0, inner_dot * steepness)
        shadows = np.maximum(0.0, -inner_dot * steepness)

        surf_w: int = max(1, int(round(inner_w * tile_size_px)))
        surf_h: int = max(1, int(round(inner_h * tile_size_px)))

        highlight_surf = pygame.Surface(
            (surf_w, surf_h), pygame.SRCALPHA
        )
        shadow_surf = pygame.Surface(
            (surf_w, surf_h), pygame.SRCALPHA
        )

        max_sh: float = float(profile.shadow_intensity)  # alpha
        max_hl: float = float(profile.highlight_intensity)  # alpha

        # 1. Render Subtractive Shadow Surface (Vectorized C-Speed)
        if max_sh > 0:
            sh_alpha = np.clip(
                shadows * max_sh, 0.0, max_sh
            ).astype(np.uint8)

            if np.any(sh_alpha > 0):
                sh_rgba = np.zeros(
                    (inner_h, inner_w, 4), dtype=np.uint8
                )
                sh_rgba[:, :, 3] = sh_alpha
                sh_s = pygame.image.frombuffer(
                    sh_rgba.tobytes(), (inner_w, inner_h), "RGBA"
                )
                shadow_surf = pygame.transform.scale(
                    sh_s, (surf_w, surf_h)
                )

        # 2. Render Tile-Aware Additive Highlight Surface (Vectorized C-Speed)
        if max_hl > 0:
            hl_ratio = np.clip(
                highlights * (max_hl / 255.0), 0.0, 1.0
            )

            if np.any(hl_ratio > 0.01):
                rgb_table = cls._get_rgb_table(tile_registry)
                base_rgb = rgb_table[inner_ids]

                hl_ratio_3d = hl_ratio[:, :, np.newaxis]
                hl_rgb = np.clip(
                    base_rgb * hl_ratio_3d * 1.5, 0, 255
                ).astype(np.uint8)
                hl_a = np.clip(
                    hl_ratio * 255.0, 0, 255
                ).astype(np.uint8)[:, :, np.newaxis]

                hl_rgba = np.dstack((hl_rgb, hl_a))
                hl_s = pygame.image.frombuffer(
                    hl_rgba.tobytes(), (inner_w, inner_h), "RGBA"
                )
                highlight_surf = pygame.transform.scale(
                    hl_s, (surf_w, surf_h)
                )

        return highlight_surf, shadow_surf
