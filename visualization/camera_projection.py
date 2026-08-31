"""
Camera projection math converting world tile coordinates to viewport pixels.
"""

from typing import Tuple, Optional
from entities.map_profile_registry import (
    MapProfileRegistry,
    ResolvedMapProfile,
)


class CameraProjection:
    """
    Computes spatial tile bounding rects and origin pixel coordinates.
    """

    @staticmethod
    def calculate_tile_size(
        rw: int,
        rh: int,
        map_w: int,
        map_h: int,
        is_camera_centered: bool,
        is_zoomed: bool,
        rows: int,
        cols: int,
        camera_zoom: float = 1.0,
        map_profile: Optional[ResolvedMapProfile] = None,
    ) -> float:
        """
        Calculates pixel tile size depending on active camera mode.
        """
        if is_camera_centered:
            base_tile_sz: float = float(
                map_profile.tile_size if map_profile is not None
                else MapProfileRegistry().get_profile("DEFAULT").tile_size
            )
            return base_tile_sz * camera_zoom

        return min(
            float(rw) / float(map_w),
            float(rh) / float(map_h)
        )

    @staticmethod
    def calculate_origin_pixel(
        rx: int,
        ry: int,
        rw: int,
        rh: int,
        cx: float,
        cy: float,
        tile_size: float,
        is_camera_centered: bool,
    ) -> Tuple[int, int]:
        """
        Calculates screen pixel position for candidate center position.
        """
        if is_camera_centered:
            center_px: float = float(rx) + (float(rw) / 2.0)
            center_py: float = float(ry) + (float(rh) / 2.0)
            return int(round(center_px)), int(round(center_py))

        return int(rx + (cx * tile_size)), int(ry + (cy * tile_size))

    @staticmethod
    def calculate_camera_centered_tile_rect(
        tx: int,
        ty: int,
        cx: float,
        cy: float,
        center_px: float,
        center_py: float,
        tile_size: float,
    ) -> Tuple[int, int, int, int]:
        """
        Calculates screen pixel rectangle for tile in camera-centered mode.
        """
        t_x: int = int(round(center_px + (float(tx) - cx) * tile_size))
        t_y: int = int(round(center_py + (float(ty) - cy) * tile_size))
        return (t_x, t_y, int(tile_size) + 1, int(tile_size) + 1)
