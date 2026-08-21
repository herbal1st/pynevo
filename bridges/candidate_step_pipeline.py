"""
Encapsulates single-frame simulation execution for individual candidates.
"""

import math

from core.map_data import MapData
from core.pathfinder import BFSPathfinder
from core.kinematics.engine import CandidateKinematics
from perception.spatial_transformer import SpatialTransformer
from entities.player_state import PlayerState
from neural.network import NeuralNetwork
from evolution.recorder import FrameRecorder
from utils.math_utils import calculate_angle_delta


class CandidateStepPipeline:
    """
    Executes sensory perception, neural inference, & physical steps.
    """

    def __init__(
        self,
        transformer: SpatialTransformer,
        kinematics: CandidateKinematics
    ) -> None:
        """
        Initializes pipeline with spatial transformer and kinematics engine.
        """
        self.transformer: SpatialTransformer = transformer
        self.kinematics: CandidateKinematics = kinematics

    def execute_step(
        self,
        step_idx: int,
        state: PlayerState,
        net: NeuralNetwork,
        map_data: MapData,
        pathfinder: BFSPathfinder,
        recorder: FrameRecorder,
        candidate_idx: int = 0
    ) -> bool:
        """
        Executes candidate tick using task-space or differential thrust outputs.
        """
        profile = self.transformer.profile
        max_speed: float = (
            self.kinematics.move_speed if self.kinematics is not None
            else 0.125
        )

        spin_dmg_rate: float = (
            profile.health_spin_dmg_per_frame if profile is not None
            else 0.0
        )
        effective_rot_ratio: float = (
            state.last_rot_ratio if spin_dmg_rate > 0.0 else 0.0
        )

        features = self.transformer.compile_feature_vector(
            state.x,
            state.y,
            state.heading,
            state.last_speed_ratio,
            state.health,
            map_data,
            pathfinder,
            candidate_idx=candidate_idx,
            is_collided=state.last_collided,
            is_idle=state.last_idle,
            is_healing=state.last_healing,
            rot_ratio=effective_rot_ratio
        )

        gps_progress = self.transformer.last_gps_progress

        use_linear: bool = (
            profile.use_linear_speed_output
            if profile is not None else False
        )

        outputs = net.forward(features)[0]
        if use_linear:
            fwd_eff: float = float(outputs[0])
            bwd_eff: float = float(outputs[1])
            left_eff: float = float(outputs[2])
            right_eff: float = float(outputs[3])

            move_eff: float = fwd_eff - bwd_eff
            turn_eff: float = right_eff - left_eff
        else:
            l_fwd: float = float(outputs[0])
            l_bwd: float = float(outputs[1])
            r_fwd: float = float(outputs[2])
            r_bwd: float = float(outputs[3])

            net_l: float = l_fwd - l_bwd
            net_r: float = r_fwd - r_bwd

            move_eff = (net_r + net_l) / 2.0
            turn_eff = (net_r - net_l) / 2.0

        prev_heading: float = state.heading
        state.heading, _ = self.kinematics.apply_rotation(
            state.heading, turn_eff, move_eff
        )

        d_theta: float = abs(
            calculate_angle_delta(prev_heading, state.heading)
        )
        max_turn_rad: float = max(
            1e-6,
            self.kinematics.rad_per_frame
            if self.kinematics is not None else 0.1
        )
        rot_ratio: float = max(0.0, min(1.0, d_theta / max_turn_rad))

        nx, ny, hit = self.kinematics.calculate_forward_step(
            state.x, state.y, state.heading, move_eff, map_data
        )

        dx: float = nx - state.x
        dy: float = ny - state.y
        disp_dist: float = math.sqrt((dx * dx) + (dy * dy))
        physical_speed_ratio: float = max(
            0.0, min(1.0, disp_dist / max(1e-4, max_speed))
        )

        idle_thresh: float = (
            profile.idle_damage_speed_threshold if profile is not None
            else 0.05
        )
        heal_thresh: float = (
            profile.heal_speed_threshold if profile is not None
            else 0.80
        )

        is_idle: bool = (physical_speed_ratio < idle_thresh)
        is_healing: bool = (
            physical_speed_ratio >= heal_thresh and state.is_alive
        )

        state.x = nx
        state.y = ny
        state.has_collided = hit
        state.frames_survived += 1

        dmg_coll: float = (
            profile.health_coll_dmg_per_frame if profile is not None
            else 0.005
        )
        dmg_idle: float = (
            profile.health_idle_dmg_per_frame if profile is not None
            else 0.005
        )
        move_heal_rate: float = (
            profile.move_heal_per_frame if profile is not None
            else 0.002
        )
        path_heal_rate: float = (
            profile.path_heal_per_frame if profile is not None
            else 0.0
        )

        if hit:
            state.health = max(0.0, state.health - dmg_coll)

        if is_idle:
            state.health = max(0.0, state.health - dmg_idle)

        if spin_dmg_rate > 0.0 and rot_ratio > 0.0:
            state.health = max(
                0.0, state.health - (spin_dmg_rate * rot_ratio)
            )

        if is_healing and move_heal_rate > 0.0:
            state.health = min(1.0, state.health + move_heal_rate)

        use_binoc: bool = (
            profile.use_binocular_gps_compasses if profile is not None
            else True
        )
        if use_binoc and len(gps_progress) >= 2:
            bfsl_pos: float = gps_progress[0]
            bfsr_pos: float = gps_progress[1]
            path_refuel: float = 0.5 * path_heal_rate * (
                bfsl_pos + bfsr_pos
            )
        elif len(gps_progress) >= 1:
            bfs_pos: float = gps_progress[0]
            path_refuel = path_heal_rate * bfs_pos
        else:
            path_refuel = 0.0

        if path_heal_rate > 0.0 and path_refuel > 0.0:
            state.health = min(1.0, state.health + path_refuel)

        if state.health <= 0.0:
            state.is_alive = False

        state.last_speed_ratio = physical_speed_ratio
        state.last_collided = hit
        state.last_idle = is_idle
        state.last_healing = is_healing
        state.last_rot_ratio = rot_ratio if spin_dmg_rate > 0.0 else 0.0

        curr_dist: int = pathfinder.get_step_distance(*state.tile_coords)

        if curr_dist < state.best_step_dist:
            state.best_step_dist = curr_dist

        if state.tile_coords == map_data.exit_pos:
            state.has_reached_exit = True

        recorder.record_step_data(
            step_idx=step_idx,
            cand_idx=candidate_idx,
            x=state.x,
            y=state.y,
            heading=state.heading,
            health=state.health,
            dist=float(curr_dist),
            hit_wall=hit,
            is_alive=state.is_alive,
            reached_exit=state.has_reached_exit
        )

        return state.is_alive and not state.has_reached_exit
