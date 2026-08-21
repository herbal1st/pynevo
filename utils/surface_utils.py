"""
Surface creation, alpha scratchpad, and scaling helper utilities.
"""

import pygame


def create_alpha_surface(
    width: int,
    height: int
) -> pygame.Surface:
    """
    Creates pre-allocated transparent RGBA Pygame surface buffer.
    """
    surface: pygame.Surface = pygame.Surface(
        (width, height), pygame.SRCALPHA
    )
    return surface


def scale_text_surface(
    surface: pygame.Surface,
    target_side: int
) -> pygame.Surface:
    """
    Smoothly scales text surface to square dimensions of target_side pixels.
    """
    clamped_side: int = max(2, target_side)
    return pygame.transform.smoothscale(
        surface, (clamped_side, clamped_side)
    )
