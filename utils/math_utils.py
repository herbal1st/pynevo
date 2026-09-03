"""
Mathematical calculations and vector transformation utilities.
"""

import math
from typing import Tuple

CARDINAL_MOVES: Tuple[Tuple[int, int], ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0)
)


def normalize_angle_2pi(angle_rad: float) -> float:
    """
    Normalizes an angle in radians to the range [0.0, 2.0 * pi).
    """
    two_pi: float = 2.0 * math.pi
    return angle_rad % two_pi


def normalize_angle_pi(angle_rad: float) -> float:
    """
    Normalizes an angle in radians to the range [-pi, pi].
    """
    two_pi: float = 2.0 * math.pi
    norm_angle: float = angle_rad % two_pi
    if norm_angle > math.pi:
        norm_angle -= two_pi
    return norm_angle


def calculate_angle_delta(
    source_rad: float,
    target_rad: float
) -> float:
    """
    Calculates signed shortest angular delta between source and target angles.
    """
    two_pi: float = 2.0 * math.pi
    delta: float = (target_rad - source_rad) % two_pi
    if delta > math.pi:
        delta -= two_pi
    return delta


def calculate_euclidean_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> float:
    """
    Computes Euclidean distance between two 2D points.
    """
    dx: float = x2 - x1
    dy: float = y2 - y1
    return math.sqrt((dx * dx) + (dy * dy))


def lerp(start_val: float, end_val: float, factor: float) -> float:
    """
    Linear interpolation between start_val and end_val by factor [0.0, 1.0].
    """
    clamped_factor: float = max(0.0, min(1.0, factor))
    return start_val + clamped_factor * (end_val - start_val)


def calculate_spin_angle(
    step_idx: int,
    speed_rate: float = 0.15
) -> float:
    """
    Computes a continuous normalized spin angle in radians [0.0, 2.0 * pi).
    """
    two_pi: float = 2.0 * math.pi
    raw_angle: float = float(step_idx) * speed_rate
    return raw_angle % two_pi
