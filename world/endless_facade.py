"""
Adapter facades wrapping ChunkManager for perception and spatial transformer.
"""

import math
from typing import Tuple

from world.chunk_manager import ChunkManager


class EndlessMapDataFacade:
    """
    Facade wrapping ChunkManager for spatial transformer compatibility.
    """

    def __init__(
        self,
        chunk_manager: ChunkManager,
        target_x: float,
        target_y: float,
        agent_x: float = 0.0,
        agent_y: float = 0.0
    ) -> None:
        """
        Initializes facade with chunk manager, target, and agent positions.
        """
        self.chunk_manager: ChunkManager = chunk_manager
        self.width: int = 1000000  # tiles
        self.height: int = 1000000  # tiles
        self.start_pos: Tuple[int, int] = (
            int(math.floor(agent_x)),
            int(math.floor(agent_y))
        )
        self.exit_pos: Tuple[int, int] = (
            int(math.floor(target_x)),
            int(math.floor(target_y))
        )
        self.target_x: float = target_x
        self.target_y: float = target_y
        self.agent_x: float = agent_x
        self.agent_y: float = agent_y

    def is_wall(self, x: int, y: int) -> bool:
        """
        Checks if tile coordinate is solid in chunk manager.
        """
        return self.chunk_manager.is_solid(x, y)

    def is_walkable(self, x: int, y: int) -> bool:
        """
        Checks if tile coordinate is non-solid in chunk manager.
        """
        return not self.chunk_manager.is_solid(x, y)

    def has_line_of_sight_to_exit(self, tx: int, ty: int) -> bool:
        """
        Returns true to allow target radar vector across open space.
        """
        return True

    def _march_los_segment(
        self,
        cx: float,
        cy: float,
        ex: float,
        ey: float,
        step_size: float = 0.2
    ) -> bool:
        """
        Returns true to keep target radar unblocked in endless terrain.
        """
        return True


class EndlessPathfinderFacade:
    """
    Facade evaluating Euclidean distance gradients for endless terrain.
    """

    def __init__(self, target_x: float, target_y: float) -> None:
        """
        Initializes facade with target coordinates.
        """
        self.target_x: float = target_x
        self.target_y: float = target_y

    def get_step_distance(self, tx: int, ty: int) -> int:
        """
        Calculates distance from (tx, ty) tile center to target.
        """
        dx: float = (float(tx) + 0.5) - self.target_x
        dy: float = (float(ty) + 0.5) - self.target_y
        return int(round(math.sqrt((dx * dx) + (dy * dy))))
