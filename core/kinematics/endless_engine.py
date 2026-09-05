"""
Endless chunk-aware kinematics engine with Circle-to-AABB collision math.
"""

import math
from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from numba import njit

from world.tile_registry import TileRegistry
from world.chunk_manager import ChunkManager
from utils.math_utils import normalize_angle_2pi


@njit(fastmath=True, cache=True)
def resolve_endless_circle_aabb_jit(
    px: float,
    py: float,
    r: float,
    solid_tx: NDArray[np.int32],
    solid_ty: NDArray[np.int32],
    passes: int = 2
) -> Tuple[float, float, bool]:
    """
    JIT-compiled non-GIL C routine resolving continuous circle overlap against solid tiles.
    """
    has_collided = False
    r_sq = r * r
    n = len(solid_tx)

    for _ in range(passes):
        for i in range(n):
            tx = solid_tx[i]
            ty = solid_ty[i]

            cx = max(float(tx), min(px, float(tx) + 1.0))
            cy = max(float(ty), min(py, float(ty) + 1.0))

            dx = px - cx
            dy = py - cy
            dist_sq = (dx * dx) + (dy * dy)

            if dist_sq < r_sq:
                has_collided = True
                dist = math.sqrt(dist_sq)

                if dist > 1e-6:
                    overlap = r - dist
                    px += (dx / dist) * overlap
                    py += (dy / dist) * overlap
                else:
                    tile_cx = float(tx) + 0.5
                    tile_cy = float(ty) + 0.5
                    push_x = 1.0 if px >= tile_cx else -1.0
                    push_y = 1.0 if py >= tile_cy else -1.0

                    if abs(px - tile_cx) < abs(py - tile_cy):
                        py = (float(ty + 1) + r) if push_y > 0.0 else (float(ty) - r)
                    else:
                        px = (float(tx + 1) + r) if push_x > 0.0 else (float(tx) - r)

    return px, py, has_collided


class EndlessKinematics:
    """
    Handles 2D entity movement physics and wall ejection in endless terrain.
    """

    MAX_SUB_STEP_DIST: float = 0.20  # tiles limit

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
        Calculates forward step scaled by friction with physics sub-stepping.
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

        total_dist: float = clamped_effort * base_move_speed * speed_mult
        if abs(total_dist) < 1e-5:
            return cx, cy, False

        num_sub_steps: int = max(
            1, math.ceil(abs(total_dist) / cls.MAX_SUB_STEP_DIST)
        )
        sub_dist: float = total_dist / float(num_sub_steps)

        dx_sub: float = math.cos(heading_rad) * sub_dist
        dy_sub: float = math.sin(heading_rad) * sub_dist

        curr_x: float = cx
        curr_y: float = cy
        has_any_collision: bool = False

        for _ in range(num_sub_steps):
            curr_x += dx_sub
            curr_y += dy_sub

            curr_x, curr_y, hit = cls._resolve_circle_aabb(
                curr_x,
                curr_y,
                diameter_ratio,
                chunk_manager,
                tile_registry,
                generator,
                passes=passes
            )
            if hit:
                has_any_collision = True

        return curr_x, curr_y, has_any_collision

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
        Resolves circle penetration against surrounding solid tile AABBs via JIT.
        """
        r: float = max(0.01, 0.5 * float(diameter_ratio))
        min_tx: int = math.floor(px - r)
        max_tx: int = math.floor(px + r)
        min_ty: int = math.floor(py - r)
        max_ty: int = math.floor(py + r)

        solid_tx_list = []
        solid_ty_list = []

        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                cls._ensure_chunk_loaded(
                    tx, ty, chunk_manager, tile_registry, generator
                )
                tile_id: int = chunk_manager.get_tile(tx, ty)
                tile_prof = tile_registry.get_tile(tile_id)
                if tile_prof.solid:
                    solid_tx_list.append(tx)
                    solid_ty_list.append(ty)

        if not solid_tx_list:
            return px, py, False

        stx_arr = np.array(solid_tx_list, dtype=np.int32)
        sty_arr = np.array(solid_ty_list, dtype=np.int32)

        return resolve_endless_circle_aabb_jit(
            px, py, r, stx_arr, sty_arr, passes=passes
        )

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