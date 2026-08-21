"""
Random noise scatter strategy for procedural level generation.
"""

import random
from core.map_data import MapData
from core.map_generation.base_strategy import BaseMapStrategy


class RandomScatterStrategy(BaseMapStrategy):
    """
    Populates map grid with random wall noise based on density ratio.
    """

    def generate_tiles(
        self,
        map_data: MapData,
        wall_density: float
    ) -> bool:
        """
        Fills outer borders and randomly places inner wall tiles.
        """
        width: int = map_data.width
        height: int = map_data.height

        for y in range(height):
            for x in range(width):
                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    map_data.set_wall(x, y, True)
                elif random.random() < wall_density:
                    map_data.set_wall(x, y, True)

        return True
