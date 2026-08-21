"""
Main network graph facade orchestrating label, layout, and rendering modules.
"""

from typing import Tuple, Dict, Any, Optional, List, Sequence
import pygame

import config
from bridges.playback_presenter import PlaybackPresenter
from entities.agent_profile_registry import (
    AgentProfileRegistry,
    ResolvedAgentProfile,
)
from utils.font_manager import FontManager
from visualization.network_graph.label_resolver import GraphLabelResolver
from visualization.network_graph.layout_engine import (
    GraphLayoutEngine,
    GraphLayout,
)
from visualization.network_graph.column_renderer import GraphColumnRenderer


class NetworkGraph:
    """
    Renders neural network activation columns and heatmap overlays.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int] = config.LAYOUT_GRAPH_RECT,
    ) -> None:
        """
        Initializes graph facade, profile registry, and component solvers.
        """
        self.x, self.y, self.w, self.h = rect
        self.font_manager: FontManager = FontManager()
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.profile: ResolvedAgentProfile = self.registry.get_profile(
            config.ACTIVE_AGENT_PROFILE
        )

        self.label_resolver: GraphLabelResolver = GraphLabelResolver()
        self.layout_engine: GraphLayoutEngine = GraphLayoutEngine(
            self.font_manager
        )
        self.column_renderer: GraphColumnRenderer = GraphColumnRenderer()

        self.presenter: Optional[PlaybackPresenter] = None
        self._current_gen: Optional[int] = None
        self._layout_key: Optional[Tuple[Any, ...]] = None
        self._layout: Optional[GraphLayout] = None

    def draw_graph(
        self,
        surface: pygame.Surface,
        gen_data: Dict[str, Any],
        cand_idx: int,
        active_step: int,
    ) -> None:
        """
        Renders background panel and candidate network activation graph.
        """
        pygame.draw.rect(
            surface, config.COLOR_BG, (self.x, self.y, self.w, self.h)
        )
        pygame.draw.rect(
            surface,
            config.COLOR_WALL_BORDER,
            (self.x, self.y, self.w, self.h),
            1,
        )

        gen_num: int = int(gen_data.get("generation", 0))

        if self.presenter is None or self._current_gen != gen_num:
            if self.presenter is None:
                self.presenter = PlaybackPresenter(gen_data)
            else:
                self.presenter.bind_generation(gen_data)
            self._current_gen = gen_num

        if self.presenter is None:
            return

        curr_frame = self.presenter.get_candidate_frame(
            cand_idx, active_step
        )
        if curr_frame is None:
            return

        activations: List[List[float]] = curr_frame["activations"]
        self._draw_activations_layout(surface, activations)

    def draw_live_graph(
        self,
        surface: pygame.Surface,
        activations: List[List[float]],
    ) -> None:
        """
        Renders neural network activation graph directly from live layer list.
        """
        pygame.draw.rect(
            surface, config.COLOR_BG, (self.x, self.y, self.w, self.h)
        )
        pygame.draw.rect(
            surface,
            config.COLOR_WALL_BORDER,
            (self.x, self.y, self.w, self.h),
            1,
        )

        self._draw_activations_layout(surface, activations)

    def _draw_activations_layout(
        self,
        surface: pygame.Surface,
        activations: List[List[float]],
    ) -> None:
        """
        Helper rendering activation columns from structured layer list.
        """
        if len(activations) < 2:
            return

        flat_inputs: List[float] = activations[0]
        hidden_and_outputs: List[List[float]] = activations[1:]

        input_labels: List[str] = (
            self.label_resolver.get_base_shorthand_list(self.profile)
        )
        output_labels: List[str] = (
            self.label_resolver.get_output_label_list(
                len(hidden_and_outputs[-1]), self.profile
            )
        )

        base_channels: int = len(input_labels)
        mem_k: int = max(0, self.profile.memory_frames)
        total_frames_count: int = 1 + mem_k

        memory_columns: List[List[float]] = []
        for mem_idx in range(total_frames_count):
            chunk_idx: int = (total_frames_count - 1) - mem_idx
            start_idx: int = chunk_idx * base_channels
            end_idx: int = start_idx + base_channels
            chunk: List[float] = flat_inputs[start_idx:end_idx]

            if len(chunk) < base_channels:
                chunk.extend([0.0] * (base_channels - len(chunk)))
            memory_columns.append(chunk)

        hidden_layers_data: List[List[float]] = [
            list(layer_data) for layer_data in hidden_and_outputs[:-1]
        ]
        output_layer_data: List[float] = list(hidden_and_outputs[-1])
        if not output_layer_data:
            output_layer_data = [0.0]

        layout: GraphLayout = self._get_layout(
            base_channels=base_channels,
            mem_k=mem_k,
            hidden_layers_data=hidden_layers_data,
            output_count=len(output_layer_data),
            input_labels=input_labels,
            output_labels=output_labels,
        )

        self.column_renderer.draw_labels(surface, layout)
        self.column_renderer.draw_headers(surface, layout)

        col_idx: int = 0
        curr_x: float = layout.node_start_x

        for mem_vals in memory_columns:
            col_w: float = layout.node_widths[col_idx]
            self.column_renderer.draw_activation_column(
                surface,
                mem_vals,
                curr_x,
                col_w,
                layout.grid_y,
                layout.content_height,
                layout.spacing,
            )
            curr_x += col_w + layout.spacing
            col_idx += 1

        for hidden_vals in hidden_layers_data:
            col_w = layout.node_widths[col_idx]
            self.column_renderer.draw_activation_column(
                surface,
                hidden_vals,
                curr_x,
                col_w,
                layout.grid_y,
                layout.content_height,
                layout.spacing,
            )
            curr_x += col_w + layout.spacing
            col_idx += 1

        output_w: float = layout.node_widths[-1]
        self.column_renderer.draw_activation_column(
            surface,
            output_layer_data,
            curr_x,
            output_w,
            layout.grid_y,
            layout.content_height,
            layout.spacing,
        )

    def _get_layout(
        self,
        base_channels: int,
        mem_k: int,
        hidden_layers_data: Sequence[Sequence[float]],
        output_count: int,
        input_labels: Sequence[str],
        output_labels: Sequence[str],
    ) -> GraphLayout:
        """
        Retrieves cached layout or delegates calculation to layout engine.
        """
        spacing: int = max(1, int(config.HUD_GRAPH_SPACING))
        hidden_counts: Tuple[int, ...] = tuple(
            len(layer_data) for layer_data in hidden_layers_data
        )

        layout_key = (
            self.x,
            self.y,
            self.w,
            self.h,
            spacing,
            base_channels,
            mem_k,
            hidden_counts,
            output_count,
            tuple(input_labels),
            tuple(output_labels),
        )

        if self._layout is not None and self._layout_key == layout_key:
            return self._layout

        layout: GraphLayout = self.layout_engine.build_layout(
            (self.x, self.y, self.w, self.h),
            base_channels,
            mem_k,
            hidden_counts,
            input_labels,
            output_labels,
            spacing,
        )

        self._layout_key = layout_key
        self._layout = layout
        return layout
