"""
Data presenter formatting recorded playback timelines into UI view models.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
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
from utils.math_utils import calculate_angle_delta


class PlaybackPresenter:
    """
    Transforms raw recorded generation histories into strongly-typed UI views.
    """

    def __init__(self, gen_data: Dict[str, Any]) -> None:
        """
        Binds generation history data dictionary, profile, and brain loader.
        """
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
        """
        Binds new generation history data dictionary and builds map data.
        """
        self.gen_data = gen_data

        map_w: int = int(self.gen_data.get("map_width", 24))
        map_h: int = int(self.gen_data.get("map_height", 18))
        start_pos = self.gen_data["start_pos"]
        exit_pos = self.gen_data["exit_pos"]

        m_data = MapData(map_w, map_h, start_pos, exit_pos)
        m_data.decode_bitmask(self.gen_data["bitmask_chunks"])
        m_data.compute_exit_los_cache()
        m_data.target_sequence = list(
            self.gen_data.get("target_sequence", [exit_pos])
        )

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
            elif isinstance(pop_w, list):
                for cand_layers in pop_w:
                    cand_net = factory.create_network()
                    if isinstance(cand_layers, np.ndarray):
                        cand_net.import_flat_weights(cand_layers)
                    else:
                        for idx, (w_mat, b_mat) in enumerate(cand_layers):
                            if idx < len(cand_net.layers):
                                cand_net.layers[idx].weights = w_mat.copy()
                                cand_net.layers[idx].biases = b_mat.copy()
                    self.gen_networks.append(cand_net)

    @property
    def generation_number(self) -> int:
        """
        Returns 1-based display generation index.
        """
        return int(self.gen_data.get("generation", 0)) + 1

    @property
    def winner_index(self) -> int:
        """
        Returns winning candidate index for active generation.
        """
        return int(self.gen_data.get("winner_index", 0))

    @property
    def raw_scores(self) -> List[float]:
        """
        Returns raw candidate scores list.
        """
        return self.gen_data.get("raw_scores", [])

    @property
    def top_score(self) -> float:
        """
        Returns maximum score achieved in generation.
        """
        scores = self.raw_scores
        return max(scores) if scores else 0.0

    @property
    def avg_score(self) -> float:
        """
        Returns average population score in generation.
        """
        scores = self.raw_scores
        return sum(scores) / float(len(scores)) if scores else 0.0

    def get_max_frame_count(self) -> int:
        """
        Returns maximum recorded frame count across candidates in generation.
        """
        telemetry = self.gen_data.get("telemetry", None)
        if telemetry is not None:
            return int(telemetry.shape[0])
        cand_frames = self.gen_data.get("candidate_frames", [])
        return max(len(cf) for cf in cand_frames) if cand_frames else 0

    def get_candidate_frame(
        self,
        candidate_idx: int,
        step: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves playback frame dictionary for candidate at specified step.
        """
        telemetry = self.gen_data.get("telemetry", None)
        activations = self.gen_data.get("activations", None)

        safe_cand: int = max(0, int(candidate_idx))
        safe_step: int = max(0, int(step))

        if telemetry is None:
            cand_frames = self.gen_data.get("candidate_frames", [])
            if not cand_frames or safe_cand >= len(cand_frames):
                return None
            frames = cand_frames[safe_cand]
            if not frames:
                return None
            frame_i: int = min(safe_step, len(frames) - 1)
            return frames[frame_i]

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

        t_seq = self.gen_data.get(
            "target_sequence",
            [self._map_data.exit_pos if self._map_data else (0, 0)]
        )
        curr_stage, target_pos = self._resolve_active_target_for_step(
            safe_cand, frame_idx, telemetry, t_seq
        )

        face_str: str = EntityExpress.resolve_face(
            reached_exit, hit_wall, is_alive, profile=self.profile
        )

        if activations is not None:
            act_flat = activations[frame_idx, safe_cand]
            structured_activations = self._unflatten_activations(act_flat)
        else:
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
            "stage_idx": curr_stage,
            "target_pos": target_pos,
            "activations": structured_activations
        }

    def _resolve_active_target_for_step(
        self,
        cand_idx: int,
        frame_idx: int,
        telemetry: np.ndarray,
        target_sequence: List[Tuple[int, int]]
    ) -> Tuple[int, Tuple[int, int]]:
        """
        Calculates active target stage and coordinates for candidate step.
        """
        if not target_sequence:
            return 0, (0, 0)

        hold_thresh: float = (
            self.profile.target_hold_distance_threshold
            if self.profile is not None else 0.25
        )
        hold_thresh_sq: float = hold_thresh * hold_thresh
        target_hold_frames: int = 15

        curr_stage: int = 0
        hold_count: int = 0

        for s in range(frame_idx + 1):
            if curr_stage >= len(target_sequence):
                break

            tx, ty = target_sequence[curr_stage]
            tc_x: float = float(tx) + 0.5
            tc_y: float = float(ty) + 0.5

            cx: float = float(telemetry[s, cand_idx, 0])
            cy: float = float(telemetry[s, cand_idx, 1])

            dx: float = cx - tc_x
            dy: float = cy - tc_y
            dist_sq: float = (dx * dx) + (dy * dy)

            if dist_sq <= hold_thresh_sq:
                hold_count += 1
                if hold_count >= target_hold_frames:
                    curr_stage += 1
                    hold_count = 0
            else:
                hold_count = 0

        safe_stage: int = min(curr_stage, len(target_sequence) - 1)
        return safe_stage, target_sequence[safe_stage]

    def _compute_live_activations(
        self,
        cand_idx: int,
        frame_idx: int,
        telemetry: np.ndarray
    ) -> List[List[float]]:
        """
        Computes real-time sensory feature vector and neural forward pass.
        """
        if (
            self._map_data is None or
            self._pathfinder is None or
            self._transformer is None
        ):
            return self._build_empty_activations()

        active_net: Optional[NeuralNetwork] = None
        if 0 <= cand_idx < len(self.gen_networks):
            active_net = self.gen_networks[cand_idx]
        else:
            active_net = self._network

        if active_net is None:
            return self._build_empty_activations()

        mem_k: int = (
            self.profile.memory_frames if self.profile is not None else 0
        )
        total_frames_count: int = 1 + max(0, mem_k)

        max_speed: float = (
            self.profile.move_speed if self.profile is not None else 0.125
        )
        idle_thresh: float = (
            self.profile.idle_damage_speed_threshold
            if self.profile is not None else 0.05
        )
        heal_thresh: float = (
            self.profile.heal_speed_threshold
            if self.profile is not None else 0.80
        )

        spin_dmg_rate: float = (
            self.profile.health_spin_dmg_per_frame
            if self.profile is not None else 0.0
        )

        base_vectors: List[np.ndarray] = []

        for offset in range(total_frames_count - 1, -1, -1):
            step_k: int = max(0, frame_idx - offset)
            row_k = telemetry[step_k, cand_idx]

            x_k: float = float(row_k[0])
            y_k: float = float(row_k[1])
            head_k: float = float(row_k[2])
            hp_k: float = float(row_k[3])
            hit_k: bool = bool(row_k[5] > 0.5)
            alive_k: bool = bool(row_k[6] > 0.5)

            prev_step_k: int = max(0, step_k - 1)
            row_prev = telemetry[prev_step_k, cand_idx]
            prev_x: float = float(row_prev[0])
            prev_y: float = float(row_prev[1])
            prev_head: float = float(row_prev[2])

            dx: float = x_k - prev_x
            dy: float = y_k - prev_y
            disp_dist: float = math.sqrt((dx * dx) + (dy * dy))
            spd_ratio: float = max(
                0.0, min(1.0, disp_dist / max(1e-4, max_speed))
            )

            is_idle_k: bool = (spd_ratio < idle_thresh)
            is_heal_k: bool = (spd_ratio >= heal_thresh) and alive_k

            d_theta: float = abs(
                calculate_angle_delta(prev_head, head_k)
            )
            max_turn_rad: float = max(
                1e-6,
                math.radians(
                    self.profile.turn_speed
                    if self.profile is not None else 1800.0
                ) / float(config.FPS)
            )
            rot_ratio_k: float = (
                max(0.0, min(1.0, d_theta / max_turn_rad))
                if spin_dmg_rate > 0.0 else 0.0
            )

            bv = self._transformer.compile_base_vector(
                x_k,
                y_k,
                head_k,
                spd_ratio,
                hp_k,
                self._map_data,
                self._pathfinder,
                candidate_idx=cand_idx,
                prev_x=prev_x,
                prev_y=prev_y,
                prev_heading=prev_head,
                is_collided=hit_k,
                is_idle=is_idle_k,
                is_healing=is_heal_k,
                rot_ratio=rot_ratio_k
            )
            base_vectors.append(bv)

        features = np.concatenate(base_vectors)

        outputs = active_net.forward(features)[0]
        l_fwd: float = float(outputs[0])
        l_bwd: float = float(outputs[1])
        r_fwd: float = float(outputs[2])
        r_bwd: float = float(outputs[3])

        layer_list: List[List[float]] = [
            features.astype(np.float64).tolist()
        ]

        for layer in active_net.layers[:-1]:
            if layer.output is not None:
                layer_list.append(layer.output.flatten().tolist())
            else:
                layer_list.append([0.0] * layer.weights.shape[1])

        out_l_fwd: float = max(0.0, min(1.0, l_fwd))
        out_l_bwd: float = max(0.0, min(1.0, l_bwd))
        out_r_fwd: float = max(0.0, min(1.0, r_fwd))
        out_r_bwd: float = max(0.0, min(1.0, r_bwd))
        layer_list.append([out_l_fwd, out_l_bwd, out_r_fwd, out_r_bwd])

        return layer_list

    def _build_empty_activations(self) -> List[List[float]]:
        """
        Constructs empty activations topology fallback structure.
        """
        base_channels: int = len(
            self.label_resolver.get_base_shorthand_list(self.profile)
        )
        total_frames_count: int = 1 + max(0, self.profile.memory_frames)
        input_size: int = base_channels * total_frames_count
        layer_list: List[List[float]] = [[0.0] * input_size]

        for _ in range(self.profile.hidden_layers):
            layer_list.append([0.0] * self.profile.neurons)

        layer_list.append([0.0, 0.0, 0.0, 0.0])
        return layer_list

    def _unflatten_activations(
        self,
        act_flat: np.ndarray
    ) -> List[List[float]]:
        """
        Unflattens 1D activation vector back into multi-layer list structure.
        """
        base_channels: int = len(
            self.label_resolver.get_base_shorthand_list(self.profile)
        )
        total_frames_count: int = 1 + max(0, self.profile.memory_frames)
        input_size: int = base_channels * total_frames_count
        hidden_layers: int = self.profile.hidden_layers
        neurons: int = self.profile.neurons
        output_size: int = 4

        layer_list: List[List[float]] = []
        offset: int = 0

        inp_chunk = act_flat[offset:offset + input_size].tolist()
        layer_list.append(inp_chunk)
        offset += input_size

        for _ in range(hidden_layers):
            h_chunk = act_flat[offset:offset + neurons].tolist()
            layer_list.append(h_chunk)
            offset += neurons

        out_chunk = act_flat[offset:offset + output_size].tolist()
        layer_list.append(out_chunk)

        return layer_list
