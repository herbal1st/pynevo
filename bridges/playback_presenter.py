"""
Data presenter formatting recorded playback timelines into UI view models.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np

import config
from core.map_data import MapData
from core.pathfinder import BFSPathfinder
from entities.agent_profile_registry import (
    AgentProfileRegistry,
    ResolvedAgentProfile
)
from entities.agent_factory import AgentFactory
from entities.entity_express import EntityExpress
from perception.spatial_transformer import SpatialTransformer
from neural.brain_persistence import BrainPersistence
from neural.network import NeuralNetwork
from visualization.network_graph.label_resolver import GraphLabelResolver


class PlaybackPresenter:
    def __init__(self, gen_data: Dict[str, Any]) -> None:
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.profile: ResolvedAgentProfile = self.registry.get_profile(
            config.ACTIVE_AGENT_PROFILE
        )
        self.label_resolver: GraphLabelResolver = GraphLabelResolver()
        self._map_data: Optional[MapData] = None
        self._pathfinder: Optional[BFSPathfinder] = None
        self._transformer: Optional[SpatialTransformer] = None
        self._network: Optional[NeuralNetwork] = None
        self.gen_networks: List[NeuralNetwork] = []
        self.gen_data: Dict[str, Any] = gen_data
        self.bind_generation(gen_data)

    def bind_generation(self, gen_data: Dict[str, Any]) -> None:
        self.gen_data = gen_data

        map_w: int = int(self.gen_data.get("map_width", 24))
        map_h: int = int(self.gen_data.get("map_height", 18))
        start_pos = self.gen_data["start_pos"]
        exit_pos = self.gen_data["exit_pos"]

        m_data = MapData(map_w, map_h, start_pos, exit_pos)
        m_data.decode_bitmask(self.gen_data["bitmask_chunks"])
        m_data.compute_exit_los_cache()

        p_finder = BFSPathfinder(m_data)
        p_finder.compute_distance_matrix()

        self._map_data = m_data
        self._pathfinder = p_finder
        self._transformer = SpatialTransformer(self.profile)

        factory = AgentFactory(self.registry, config.ACTIVE_AGENT_PROFILE)

        if self._network is None:
            net = factory.create_network()
            persistence = BrainPersistence()
            persistence.load_brain(
                config.ACTIVE_AGENT_PROFILE,
                net,
                self.profile,
                context="visualizer playback",
                verbose=False,
            )
            self._network = net

        pop_w = self.gen_data.get("pop_weights", None)
        self.gen_networks.clear()

        if pop_w is not None and len(pop_w) > 0:
            if isinstance(pop_w, np.ndarray):
                for c_idx in range(pop_w.shape[0]):
                    cand_net = factory.create_network()
                    cand_net.import_flat_weights(pop_w[c_idx])
                    self.gen_networks.append(cand_net)

    def get_candidate_frame(
        self,
        candidate_idx: int,
        step: int
    ) -> Optional[Dict[str, Any]]:
        telemetry = self.gen_data.get("telemetry", None)
        safe_cand: int = max(0, int(candidate_idx))
        safe_step: int = max(0, int(step))

        if telemetry is None:
            return None

        max_f: int = int(telemetry.shape[0])
        pop_s: int = int(telemetry.shape[1])
        if safe_cand >= pop_s or max_f == 0:
            return None

        frame_idx: int = min(safe_step, max_f - 1)
        row = telemetry[frame_idx, safe_cand]

        c_x: float = float(row[0])
        c_y: float = float(row[1])
        c_head: float = float(row[2])
        c_hp: float = float(row[3])
        c_dist: int = int(row[4])
        hit_wall: bool = bool(row[5] > 0.5)
        is_alive: bool = bool(row[6] > 0.5)
        reached_exit: bool = bool(row[7] > 0.5)

        face_str: str = EntityExpress.resolve_face(
            reached_exit, hit_wall, is_alive, profile=self.profile
        )

        structured_activations = self._compute_live_activations(
            safe_cand, frame_idx, telemetry
        )

        return {
            "step": frame_idx,
            "x": c_x,
            "y": c_y,
            "heading": c_head,
            "face": face_str,
            "hit_wall": hit_wall,
            "health": c_hp,
            "is_alive": is_alive,
            "reached_exit": reached_exit,
            "dist": c_dist,
            "activations": structured_activations
        }

    def _compute_live_activations(
        self,
        cand_idx: int,
        frame_idx: int,
        telemetry: np.ndarray
    ) -> List[List[float]]:
        if self._map_data is None or self._pathfinder is None or self._transformer is None:
            return self._build_empty_activations()

        active_net = (
            self.gen_networks[cand_idx]
            if 0 <= cand_idx < len(self.gen_networks)
            else self._network
        )

        if active_net is None:
            return self._build_empty_activations()

        mem_k: int = self.profile.memory_frames
        total_frames_count: int = 1 + max(0, mem_k)

        base_vectors: List[np.ndarray] = []

        for offset in range(total_frames_count - 1, -1, -1):
            step_k: int = max(0, frame_idx - offset)
            row_k = telemetry[step_k, cand_idx]

            x_k: float = float(row_k[0])
            y_k: float = float(row_k[1])
            head_k: float = float(row_k[2])
            hp_k: float = float(row_k[3])
            hit_k: bool = bool(row_k[5] > 0.5)

            prev_step_k: int = max(0, step_k - 1)
            row_prev = telemetry[prev_step_k, cand_idx]
            prev_x: float = float(row_prev[0])
            prev_y: float = float(row_prev[1])

            dx = x_k - prev_x
            dy = y_k - prev_y
            disp = math.sqrt(dx * dx + dy * dy)
            spd_r = max(0.0, min(1.0, disp / 0.20))

            bv = self._transformer.compile_base_vector(
                x_k, y_k, head_k, spd_r, hp_k,
                self._map_data, self._pathfinder,
                candidate_idx=cand_idx,
                prev_x=prev_x, prev_y=prev_y,
                is_collided=hit_k
            )
            base_vectors.append(bv)

        features = np.concatenate(base_vectors)

        outputs = active_net.forward(features)[0]

        layer_list: List[List[float]] = [features.astype(np.float64).tolist()]
        for layer in active_net.layers[:-1]:
            if layer.output is not None:
                layer_list.append(layer.output.flatten().tolist())
            else:
                layer_list.append([0.0] * layer.weights.shape[1])

        out_vals = [max(0.0, min(1.0, float(v))) for v in outputs]
        layer_list.append(out_vals)
        return layer_list

    def _build_empty_activations(self) -> List[List[float]]:
        base_channels = len(self.label_resolver.get_base_shorthand_list(self.profile))
        total_in = base_channels * (1 + self.profile.memory_frames)
        layer_list = [[0.0] * total_in]
        for _ in range(self.profile.hidden_layers):
            layer_list.append([0.0] * self.profile.neurons)
        layer_list.append([0.0, 0.0, 0.0, 0.0])
        return layer_list
