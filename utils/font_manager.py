"""
Centralized font instantiation and point size caching service.
"""

from typing import Dict
import pygame


class FontManager:
    """
    Manages cached Pygame Font instances by font size.
    """

    def __init__(self, font_name: str = "monospace") -> None:
        """
        Initializes empty font cache dictionary.
        """
        self.font_name: str = font_name
        self._cache: Dict[int, pygame.font.Font] = {}

    def get_font(
        self,
        size: int,
        bold: bool = True
    ) -> pygame.font.Font:
        """
        Retrieves cached Font instance or instantiates new Pygame font.
        """
        cache_key: int = size if bold else -size
        if cache_key in self._cache:
            return self._cache[cache_key]

        font: pygame.font.Font = pygame.font.SysFont(
            self.font_name, size, bold=bold
        )
        self._cache[cache_key] = font
        return font

    def clear(self) -> None:
        """
        Clears cached font dictionary instances.
        """
        self._cache.clear()
