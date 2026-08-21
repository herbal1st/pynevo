"""
Geometry and layout calculation engine for neural activation graphs.
"""

from dataclasses import dataclass
from typing import Tuple, List, Sequence, Dict

import pygame

import config
from utils.font_manager import FontManager


@dataclass(frozen=True, slots=True)
class GraphLayout:
    """
    Stores calculated geometry metrics and pre-rendered text surfaces.
    """

    spacing: int
    header_y: int
    header_height: int
    grid_y: int
    content_height: int
    node_start_x: float
    left_label_right: int
    right_label_left: int
    node_headers: Tuple[str, ...]
    node_widths: Tuple[float, ...]
    left_label_surfaces: Tuple[pygame.Surface, ...]
    right_label_surfaces: Tuple[pygame.Surface, ...]
    header_surfaces: Tuple[pygame.Surface, ...]


class GraphLayoutEngine:
    """
    Calculates dynamic layout bounds and fits scaled text font sizes.
    """

    _HEADER_UNIT_COUNT: float = 2.0
    _MIN_FONT_SIZE: int = 1

    def __init__(self, font_manager: FontManager) -> None:
        """
        Initializes layout engine with shared font manager and surface cache.
        """
        self.font_manager: FontManager = font_manager
        self._surface_cache: Dict[
            Tuple[str, int, bool, Tuple[int, int, int]],
            pygame.Surface
        ] = {}

    def build_layout(
        self,
        rect: Tuple[int, int, int, int],
        base_channels: int,
        mem_k: int,
        hidden_counts: Sequence[int],
        input_labels: Sequence[str],
        output_labels: Sequence[str],
        spacing: int
    ) -> GraphLayout:
        """
        Calculates spatial geometry metrics and caches text surfaces.
        """
        rx, ry, rw, rh = rect
        node_headers: List[str] = ["INP"]
        node_headers.extend(
            f"M-{m_idx}" for m_idx in range(1, mem_k + 1)
        )
        node_headers.extend(
            f"H-{h_idx + 1}" for h_idx in range(len(hidden_counts))
        )
        node_headers.append("OUT")

        full_layer_counts: List[int] = [max(1, base_channels)]
        full_layer_counts.extend(
            max(1, int(cnt)) for cnt in hidden_counts
        )
        least_nodes: int = max(1, min(full_layer_counts))

        available_height: int = max(1, rh - (2 * spacing))
        unit_height: float = available_height / float(
            least_nodes + self._HEADER_UNIT_COUNT
        )

        header_height: int = max(
            1, int(round(unit_height * self._HEADER_UNIT_COUNT))
        )
        header_y: int = ry + spacing
        grid_y: int = header_y + header_height
        grid_bottom: int = ry + rh - spacing
        content_height: int = max(1, grid_bottom - grid_y)

        input_spacing: float = (
            content_height / float(max(1, base_channels))
        )
        max_label_h: int = max(1, int(input_spacing))

        label_font_sz: int = self._fit_font_size(
            [*input_labels, *output_labels], max_label_h, bold=False
        )

        in_surfs: Tuple[pygame.Surface, ...] = tuple(
            self._get_text_surface(
                txt,
                label_font_sz,
                config.COLOR_PLAYER_HIGHLIGHT,
                bold=False
            )
            for txt in input_labels
        )
        out_surfs: Tuple[pygame.Surface, ...] = tuple(
            self._get_text_surface(
                txt,
                label_font_sz,
                config.COLOR_PLAYER_HIGHLIGHT,
                bold=False
            )
            for txt in output_labels
        )

        left_w: int = max((s.get_width() for s in in_surfs), default=0)
        right_w: int = max((s.get_width() for s in out_surfs), default=0)

        total_slots: int = len(node_headers) + 2
        spacing_segs: int = total_slots + 1

        node_area_w: float = max(
            1.0, float(rw - left_w - right_w - (spacing_segs * spacing))
        )
        activation_units: float = (
            float(max(1, len(node_headers) - 1)) + 0.5
        )
        unit_w: float = node_area_w / activation_units

        node_widths: Tuple[float, ...] = tuple(
            unit_w * (0.5 if idx == len(node_headers) - 1 else 1.0)
            for idx in range(len(node_headers))
        )

        left_right_x: int = rx + spacing + left_w
        node_start_x: float = float(left_right_x + spacing)
        right_left_x: int = int(
            round(
                node_start_x +
                sum(node_widths) +
                (len(node_widths) * spacing)
            )
        )

        hdr_font_sz: int = self._fit_header_font_size(
            node_headers, node_widths, header_height
        )
        hdr_surfs: Tuple[pygame.Surface, ...] = tuple(
            self._get_text_surface(
                txt, hdr_font_sz, config.COLOR_START, bold=True
            )
            for txt in node_headers
        )

        return GraphLayout(
            spacing=spacing,
            header_y=header_y,
            header_height=header_height,
            grid_y=grid_y,
            content_height=content_height,
            node_start_x=node_start_x,
            left_label_right=left_right_x,
            right_label_left=right_left_x,
            node_headers=tuple(node_headers),
            node_widths=node_widths,
            left_label_surfaces=in_surfs,
            right_label_surfaces=out_surfs,
            header_surfaces=hdr_surfs
        )

    def _fit_font_size(
        self,
        texts: Sequence[str],
        max_height: int,
        bold: bool
    ) -> int:
        """
        Finds the largest font size fitting maximum line height bounds.
        """
        if not texts:
            return self._MIN_FONT_SIZE

        safe_h: int = self._get_scaled_fit_height(max_height)
        largest: int = max(self._MIN_FONT_SIZE, safe_h * 2)

        for sz in range(largest, self._MIN_FONT_SIZE - 1, -1):
            font = self.font_manager.get_font(sz, bold=bold)
            if all(font.get_height() <= safe_h for txt in texts):
                return sz

        return self._MIN_FONT_SIZE

    def _fit_header_font_size(
        self,
        headers: Sequence[str],
        column_widths: Sequence[float],
        max_height: int
    ) -> int:
        """
        Fits header font sizes to height and column width constraints.
        """
        if not headers:
            return self._MIN_FONT_SIZE

        safe_h: int = self._get_scaled_fit_height(max_height)
        largest: int = max(self._MIN_FONT_SIZE, safe_h * 2)

        for sz in range(largest, self._MIN_FONT_SIZE - 1, -1):
            font = self.font_manager.get_font(sz, bold=True)
            if font.get_height() > safe_h:
                continue

            fits_w: bool = all(
                font.size(hdr)[0] <= max(1, int(round(w)))
                for hdr, w in zip(headers, column_widths)
            )
            if fits_w:
                return sz

        return self._MIN_FONT_SIZE

    def _get_scaled_fit_height(self, max_height: int) -> int:
        """
        Applies configured text scale factor to max height constraint.
        """
        scale: float = max(1.0, float(config.HUD_GRAPH_TEXT_SCALE))
        safe_h: int = max(1, int(max_height))
        scaled_h: int = int(round(safe_h * scale))
        return max(safe_h, scaled_h)

    def _get_text_surface(
        self,
        text: str,
        size: int,
        color: Tuple[int, int, int],
        bold: bool
    ) -> pygame.Surface:
        """
        Retrieves cached rendered text surface or generates a new one.
        """
        cache_key = (text, size, bold, color)
        if cache_key not in self._surface_cache:
            font = self.font_manager.get_font(size, bold=bold)
            self._surface_cache[cache_key] = font.render(text, True, color)
        return self._surface_cache[cache_key]
