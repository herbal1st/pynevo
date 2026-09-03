"""
Player controller translating raw keyboard input to endless kinematics steps.
"""

import math
from typing import Tuple, Any
import pygame

from world.tile_registry import TileRegistry
from world.chunk_manager import ChunkManager
from core.kinematics.endless_engine import EndlessKinematics
from entities.player_profile_registry import ResolvedPlayerProfile
from utils.math_utils import normalize_angle_2pi


class PlayerController:
    """
    Encapsulates human player state and dispatches input movement ticks.
    """

    def __init__(
        self,
        profile: ResolvedPlayerProfile,
        start_x: float,
        start_y: float
    ) -> None:
        """
        Initializes player coordinates, heading, and profile binding.
        """
        self.profile: ResolvedPlayerProfile = profile
        self.x: float = start_x
        self.y: float = start_y
        self.heading: float = 0.0
        self.last_collided: bool = False

    def update(
        self,
        keys: Any,
        chunk_manager: ChunkManager,
        tile_registry: TileRegistry,
        generator: Any,
        fps: int = 60
    ) -> Tuple[float, float, bool]:
        """
        Processes key states and updates player position and heading.
        """
        style: str = self.profile.profile_style.upper()
        move_effort: float = 0.0
        turn_effort: float = 0.0

        if style == "DIRECT_VECTOR":
            dx: float = 0.0
            dy: float = 0.0

            if keys[pygame.K_w] or keys[pygame.K_UP]:
                dy -= 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                dy += 1.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                dx -= 1.0
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                dx += 1.0

            if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                self.heading = normalize_angle_2pi(
                    math.atan2(dy, dx)
                )
                move_effort = 1.0
        else:
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                move_effort += 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                move_effort -= 1.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                turn_effort -= 1.0
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                turn_effort += 1.0

            self.heading = EndlessKinematics.apply_rotation(
                self.heading,
                turn_effort,
                move_effort,
                self.profile.turn_speed,
                self.profile.profile_style,
                fps
            )

        self.x, self.y, self.last_collided = (
            EndlessKinematics.calculate_forward_step(
                self.x,
                self.y,
                self.heading,
                move_effort,
                self.profile.move_speed,
                self.profile.diameter_ratio,
                chunk_manager,
                tile_registry,
                generator
            )
        )

        return self.x, self.y, self.last_collided
