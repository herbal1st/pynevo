"""
World time-of-day clock tracking solar and lunar orbital angles.
"""

import math


class DayNightClock:
    """
    Tracks elapsed time and calculates 360-degree solar light angles.
    """

    def __init__(
        self,
        day_cycle_duration: float = 120.0,  # seconds
        start_time_ratio: float = 0.25,  # ratio
        start_light_angle_deg: float = 135.0  # degrees
    ) -> None:
        """
        Initializes clock time, cycle duration, and starting light angle.
        """
        self.day_cycle_duration: float = max(
            0.0, day_cycle_duration
        )  # seconds
        self.start_light_angle_deg: float = (
            start_light_angle_deg % 360.0
        )  # degrees
        self.elapsed_time: float = (
            start_time_ratio * self.day_cycle_duration
            if self.day_cycle_duration > 0.0 else 0.0
        )  # seconds

    @property
    def normalized_time(self) -> float:
        """
        Returns time-of-day ratio in range [0.0, 1.0).
        """
        if self.day_cycle_duration <= 0.0:
            return 0.50
        return (
            self.elapsed_time % self.day_cycle_duration
        ) / self.day_cycle_duration

    @property
    def solar_angle_rad(self) -> float:
        """
        Returns 360-degree counterclockwise solar orbital angle in rad.
        """
        base_rad: float = math.radians(self.start_light_angle_deg)
        if self.day_cycle_duration <= 0.0:
            return base_rad % (2.0 * math.pi)

        orbit_rad: float = self.normalized_time * 2.0 * math.pi
        return (base_rad - orbit_rad) % (2.0 * math.pi)

    @property
    def is_daytime(self) -> bool:
        """
        Checks if active solar angle is in daylight phase ratio.
        """
        return 0.25 <= self.normalized_time < 0.75

    def update(self, delta_time: float) -> None:
        """
        Advances elapsed clock time by frame delta in seconds.
        """
        if delta_time <= 0.0 or self.day_cycle_duration <= 0.0:
            return
        self.elapsed_time += delta_time
