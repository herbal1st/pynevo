"""
Entity physical state and AI candidate evaluation state tracking.
"""

from typing import Tuple


class EntityState:
    """
    Maintains physical 2D entity position, health, and physics flags.
    """

    def __init__(self, start_x: float, start_y: float) -> None:
        """
        Initializes physical entity state at starting tile position.
        """
        self.x: float = start_x
        self.y: float = start_y
        self.heading: float = 0.0
        self.frames_survived: int = 0
        self.has_collided: bool = False
        self.health: float = 1.0
        self.is_alive: bool = True
        self.last_speed_ratio: float = 0.0
        self.last_collided: bool = False
        self.last_idle: bool = False
        self.last_healing: bool = False
        self.last_rot_ratio: float = 0.0

    @property
    def tile_coords(self) -> Tuple[int, int]:
        """
        Returns integer grid coordinates for active entity location.
        """
        return int(self.x), int(self.y)

    def reset(self, start_x: float, start_y: float) -> None:
        """
        Resets physical state variables for new evaluation runs.
        """
        self.x = start_x
        self.y = start_y
        self.heading = 0.0
        self.frames_survived = 0
        self.has_collided = False
        self.health = 1.0
        self.is_alive = True
        self.last_speed_ratio = 0.0
        self.last_collided = False
        self.last_idle = False
        self.last_healing = False
        self.last_rot_ratio = 0.0


class AgentState(EntityState):
    """
    Extends EntityState with maze navigation & fitness evaluation metrics.
    """

    def __init__(self, start_x: float, start_y: float) -> None:
        """
        Initializes candidate state with path distance and stage metrics.
        """
        super().__init__(start_x, start_y)
        self.best_step_dist: int = 9999
        self.touched_exit: bool = False
        self.exit_solved: bool = False
        self.stages_cleared: int = 0
        self.active_target_idx: int = 0
        self.hold_frame_counter: int = 0
        self.first_touch_step: int = -1
        self.first_hold_clear_step: int = -1
        self.total_lifetime_progress: float = 0.0

    @property
    def has_reached_exit(self) -> bool:
        """
        Backward compatibility alias for touched_exit.
        """
        return self.touched_exit

    @has_reached_exit.setter
    def has_reached_exit(self, value: bool) -> None:
        """
        Backward compatibility setter for touched_exit.
        """
        self.touched_exit = value

    @property
    def targets_cleared(self) -> int:
        """
        Backward compatibility alias for stages_cleared.
        """
        return self.stages_cleared

    @targets_cleared.setter
    def targets_cleared(self, value: int) -> None:
        """
        Backward compatibility setter for stages_cleared.
        """
        self.stages_cleared = value

    def reset(self, start_x: float, start_y: float) -> None:
        """
        Resets physical state and evaluation progress metrics.
        """
        super().reset(start_x, start_y)
        self.best_step_dist = 9999
        self.touched_exit = False
        self.exit_solved = False
        self.stages_cleared = 0
        self.active_target_idx = 0
        self.hold_frame_counter = 0
        self.first_touch_step = -1
        self.first_hold_clear_step = -1
        self.total_lifetime_progress = 0.0
