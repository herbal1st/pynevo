"""
Abstract base strategy interface for procedural grid level generators.
"""

from core.map_data import MapData


class BaseMapStrategy:
    """
    Abstract strategy defining interface for procedural tilemap generation.
    """

    def generate_tiles(
        self,
        map_data: MapData,
        wall_density: float
    ) -> bool:
        """
        Populates walls and floor tiles within map_data grid bounds.
        """
        raise NotImplementedError
