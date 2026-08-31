"""
Endless chunk-aware kinematics engine with Circle-to-AABB collision math.
"""

import math
from typing import Tuple, Any

from world.tile_registry import TileRegistry
from world.chunk_manager import ChunkManager
from utils.math_utils import normalize_angle_2pi


class EndlessKinematics:
    """
    Handles 2D entity movement physics and wall ejection in endless terrain.
    """

    @classmethod
    def apply_rotation(
        cls,
        heading_rad: float,
        turn_effort: float,
        move_effort: float = 1.0,
        turn_speed_dpsec: float = 1800.0,
        profile_style: str = "TANK",
        fps: int = 60
    ) -> float:
        """
        Calculates updated heading angle based on active steering style.
        """
        style_upper: str = profile_style.upper()
        clamped_turn: float = max(-1.0, min(1.0, float(turn_effort)))
        clamped_move: float = max(-1.0, min(1.0, float(move_effort)))
        rad_per_frame: float = (
            math.radians(turn_speed_dpsec) / float(fps)
        )

        if style_upper == "CAR":
            effective_turn: float = clamped_turn * abs(clamped_move)
            new_heading: float = heading_rad + (
                effective_turn * rad_per_frame
            )
        else:
            new_heading = heading_rad + (clamped_turn * rad_per_frame)

        return normalize_angle_2pi(new_heading)

    @classmethod
    def calculate_forward_step(
        cls,
        cx: float,
        cy: float,
        heading_rad: float,
        move_effort: float,
        base_move_speed: float,
        diameter_ratio: float,
        chunk_manager: ChunkManager,
        tile_registry: TileRegistry,
        generator: Any,
        passes: int = 2
    ) -> Tuple[float, float, bool]:
        """
        Calculates forward step scaled by terrain friction and resolves walls.
        """
        clamped_effort: float = max(-1.0, min(1.0, float(move_effort)))
        if abs(clamped_effort) < 1e-4:
            return cx, cy, False

        center_tx: int = math.floor(cx)
        center_ty: int = math.floor(cy)
        cls._ensure_chunk_loaded(
            center_tx, center_ty, chunk_manager, tile_registry, generator
        )

        center_tile_id: int = chunk_manager.get_tile(center_tx, center_ty)
        center_tile_prof = tile_registry.get_tile(center_tile_id)
        speed_mult: float = max(
            0.0, float(center_tile_prof.speed_multiplier)
        )

        step_dist: float = clamped_effort * base_move_speed * speed_mult
        next_x: float = cx + (math.cos(heading_rad) * step_dist)
        next_y: float = cy + (math.sin(heading_rad) * step_dist)

        resolved_x, resolved_y, hit = cls._resolve_circle_aabb(
            next_x,
            next_y,
            diameter_ratio,
            chunk_manager,
            tile_registry,
            generator,
            passes=passes
        )
        return resolved_x, resolved_y, hit

    @classmethod
    def _resolve_circle_aabb(
        cls,
        px: float,
        py: float,
        diameter_ratio: float,
        chunk_manager: ChunkManager,
        tile_registry: TileRegistry,
        generator: Any,
        passes: int = 2
    ) -> Tuple[float, float, bool]:
        """
        Resolves circle penetration against surrounding solid tile AABBs.
        """
        r: float = max(0.01, 0.5 * float(diameter_ratio))
        has_collided: bool = False

        for _ in range(passes):
            min_tx: int = math.floor(px - r)
            max_tx: int = math.floor(px + r)
            min_ty: int = math.floor(py - r)
            max_ty: int = math.floor(py + r)

            for ty in range(min_ty, max_ty + 1):
                for tx in range(min_tx, max_tx + 1):
                    cls._ensure_chunk_loaded(
                        tx, ty, chunk_manager, tile_registry, generator
                    )
                    tile_id: int = chunk_manager.get_tile(tx, ty)
                    tile_prof = tile_registry.get_tile(tile_id)

                    if not tile_prof.solid:
                        continue

                    cx: float = max(float(tx), min(px, float(tx) + 1.0))
                    cy: float = max(float(ty), min(py, float(ty) + 1.0))

                    dx: float = px - cx
                    dy: float = py - cy
                    dist_sq: float = (dx * dx) + (dy * dy)

                    if dist_sq < (r * r):
                        has_collided = True
                        dist: float = math.sqrt(dist_sq)

                        if dist > 1e-6:
                            overlap: float = r - dist
                            px += (dx / dist) * overlap
                            py += (dy / dist) * overlap
                        else:
                            tile_cx: float = float(tx) + 0.5
                            tile_cy: float = float(ty) + 0.5
                            push_x: float = (
                                1.0 if px >= tile_cx else -1.0
                            )
                            push_y: float = (
                                1.0 if py >= tile_cy else -1.0
                            )

                            if abs(px - tile_cx) < abs(py - tile_cy):
                                py = (
                                    float(ty + 1) + r if push_y > 0.0
                                    else float(ty) - r
                                )
                            else:
                                px = (
                                    float(tx + 1) + r if push_x > 0.0
                                    else float(tx) - r
                                )

        return px, py, has_collided

    @classmethod
    def _ensure_chunk_loaded(
        cls,
        tx: int,
        ty: int,
        chunk_manager: ChunkManager,
        tile_registry: TileRegistry,
        generator: Any
    ) -> None:
        """
        Ensures chunk containing tile coordinate (tx, ty) is generated.
        """
        c_size: int = ChunkManager.CHUNK_SIZE
        cx_chunk: int = math.floor(tx / c_size)
        cy_chunk: int = math.floor(ty / c_size)
        coord_key: Tuple[int, int] = (cx_chunk, cy_chunk)

        if coord_key not in chunk_manager.chunks:
            chunk = generator.generate_chunk(
                cx_chunk,
                cy_chunk,
                chunk_manager.world_seed,
                tile_registry
            )
            chunk_manager.chunks[coord_key] = chunk
