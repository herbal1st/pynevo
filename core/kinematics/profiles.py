"""
Abstract and concrete candidate steering kinematics profiles.
"""

from typing import Tuple

from utils.math_utils import normalize_angle_2pi


class KinematicsProfile:
    """
    Abstract base steering profile defining candidate rotational movement.
    """

    def calculate_rotation(
        self,
        heading_rad: float,
        turn_effort: float,
        move_effort: float,
        rad_per_frame: float
    ) -> Tuple[float, bool]:
        """
        Calculates updated heading and returns stationary turning state.
        """
        raise NotImplementedError


class CarProfile(KinematicsProfile):
    """
    Car dynamics: turning rate is scaled directly by movement effort.
    """

    def calculate_rotation(
        self,
        heading_rad: float,
        turn_effort: float,
        move_effort: float,
        rad_per_frame: float
    ) -> Tuple[float, bool]:
        """
        Applies rotational delta scaled by movement magnitude.
        """
        clamped_turn: float = max(-1.0, min(1.0, turn_effort))
        clamped_move: float = max(0.0, min(1.0, abs(move_effort)))
        effective_turn: float = clamped_turn * clamped_move

        new_heading: float = heading_rad + (effective_turn * rad_per_frame)
        return normalize_angle_2pi(new_heading), False


class TankProfile(KinematicsProfile):
    """
    Tank dynamics: differential steering works in place; idle turns drain HP.
    """

    def calculate_rotation(
        self,
        heading_rad: float,
        turn_effort: float,
        move_effort: float,
        rad_per_frame: float
    ) -> Tuple[float, bool]:
        """
        Applies differential in-place rotation independent of move effort.
        """
        clamped_turn: float = max(-1.0, min(1.0, turn_effort))
        clamped_move: float = max(0.0, min(1.0, abs(move_effort)))

        new_heading: float = heading_rad + (clamped_turn * rad_per_frame)
        is_stationary_turn: bool = (
            abs(clamped_turn) > 0.05 and clamped_move < 0.05
        )
        return normalize_angle_2pi(new_heading), is_stationary_turn
