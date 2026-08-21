"""
Pygame activation node column and label surface rendering module.
"""

from typing import Sequence
import pygame

import config
from utils.color_utils import resolve_activation_color
from visualization.network_graph.layout_engine import GraphLayout


class GraphColumnRenderer:
    """
    Renders rectangular activation nodes, headers, and text label surfaces.
    """

    _ROW_GAP_RATIO: float = 0.20

    def draw_labels(
        self,
        surface: pygame.Surface,
        layout: GraphLayout
    ) -> None:
        """
        Draws input and output text labels beside outer node columns.
        """
        in_cnt: int = max(1, len(layout.left_label_surfaces))
        in_spacing: float = layout.content_height / float(in_cnt)

        for row_idx, surf in enumerate(layout.left_label_surfaces):
            center_y: int = int(
                round(
                    layout.grid_y +
                    (row_idx * in_spacing) +
                    (in_spacing / 2.0)
                )
            )
            rect = surf.get_rect(
                midright=(layout.left_label_right, center_y)
            )
            surface.blit(surf, rect)

        out_cnt: int = max(1, len(layout.right_label_surfaces))
        out_spacing: float = layout.content_height / float(out_cnt)

        for row_idx, surf in enumerate(layout.right_label_surfaces):
            center_y = int(
                round(
                    layout.grid_y +
                    (row_idx * out_spacing) +
                    (out_spacing / 2.0)
                )
            )
            rect = surf.get_rect(
                midleft=(layout.right_label_left, center_y)
            )
            surface.blit(surf, rect)

    def draw_headers(
        self,
        surface: pygame.Surface,
        layout: GraphLayout
    ) -> None:
        """
        Draws header text surfaces centered above each activation column.
        """
        curr_x: float = layout.node_start_x

        for col_idx, surf in enumerate(layout.header_surfaces):
            col_w: float = layout.node_widths[col_idx]
            center_x: int = int(round(curr_x + (col_w / 2.0)))
            center_y: int = layout.header_y + (layout.header_height // 2)

            rect = surf.get_rect(center=(center_x, center_y))
            surface.blit(surf, rect)

            curr_x += col_w + layout.spacing

    def draw_activation_column(
        self,
        surface: pygame.Surface,
        values: Sequence[float],
        x: float,
        width: float,
        grid_y: int,
        content_height: int,
        spacing: int
    ) -> None:
        """
        Renders a vertical column of activation heatmap rectangles.
        """
        node_cnt: int = max(1, len(values))
        row_spacing: float = content_height / float(node_cnt)
        row_gap: int = self._get_row_gap(row_spacing, spacing)
        node_h: int = max(1, int(round(row_spacing)) - row_gap)
        node_w: int = max(1, int(round(width)))
        center_x: int = int(round(x + (width / 2.0)))

        for node_idx, val in enumerate(values):
            center_y: int = int(
                round(
                    grid_y +
                    (node_idx * row_spacing) +
                    (row_spacing / 2.0)
                )
            )
            clamped_val: float = max(0.0, min(1.0, float(val)))
            node_color = resolve_activation_color(clamped_val)

            node_rect = pygame.Rect(0, 0, node_w, node_h)
            node_rect.center = (center_x, center_y)

            pygame.draw.rect(surface, node_color, node_rect)
            pygame.draw.rect(
                surface, config.COLOR_WALL_BORDER, node_rect, 1
            )

    def _get_row_gap(
        self,
        row_spacing: float,
        spacing: int
    ) -> int:
        """
        Derives proportional vertical row gap bounded by layout spacing.
        """
        prop_gap: int = int(round(row_spacing * self._ROW_GAP_RATIO))
        return max(1, min(spacing, prop_gap))
