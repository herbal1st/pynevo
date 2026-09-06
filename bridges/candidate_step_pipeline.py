"""
Encapsulates single-frame simulation execution for individual candidates.
"""

import math

from core.map_data import MapData
from core.pathfinder import BFSPathfinder
from core.kinematics.engine import CandidateKinematics
from perception.spatial_transformer import SpatialTransformer
from entities.entity_state import AgentState
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
        state: AgentState,
        net: NeuralNetwork,
        map_data: MapData,
        pathfinder: BFSPathfinder,
        recorder: FrameRecorder,
        candidate_idx: int = 0,
        target_hold_frames: int = 15
    ) -> bool:
        """
        Executes candidate tick with thermodynamic metabolic health updates.
        """
        profile = self.transformer.profile
        max_speed: float = (
            self.kinematics.move_speed if self.kinematics is not None
            else 0.125
        )

        spin_dmg_rate: float = (
            profile.spin_dmg_per_frame if profile is not None
            else 0.0
        )
        effective_rot_ratio: float = (
            state.last_rot_ratio if spin_dmg_rate > 0.0 else 0.0
        )

        if hasattr(map_data, "get_target_pos"):
            ex, ey = map_data.get_target_pos(state.active_target_idx)
        else:
            ex, ey = map_data.exit_pos

        target_cx: float = float(ex) + 0.5
        target_cy: float = float(ey) + 0.5

        hold_dist_thresh: float = (
            profile.target_hold_distance_threshold
            if profile is not None else 0.25
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
            rot_ratio=effective_rot_ratio,
            stage_idx=state.active_target_idx
        )

        gps_progress = self.transformer.last_gps_progress
        f_dist: float = self.transformer.compute_scent_field_intensity(
            state.x,
            state.y,
            map_data,
            stage_idx=state.active_target_idx
        )

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

        dx_t: float = nx - target_cx
        dy_t: float = ny - target_cy
        dist_to_target: float = math.sqrt((dx_t * dx_t) + (dy_t * dy_t))

        state.touched_exit = (dist_to_target <= hold_dist_thresh)
        state.exit_solved = False

        # --- Thermodynamic Metabolic Engine ---
        move_thresh: float = (
            profile.move_dmg_threshold if profile is not None else 0.05
        )
        spin_thresh: float = (
            profile.spin_dmg_threshold if profile is not None else 0.05
        )
        idle_thresh: float = (
            profile.idle_damage_speed_threshold
            if profile is not None else 0.20
        )
        heal_thresh: float = (
            profile.heal_speed_threshold if profile is not None else 0.80
        )

        is_moving: bool = (physical_speed_ratio >= move_thresh)
        is_spinning: bool = (rot_ratio >= spin_thresh)
        is_idle: bool = (physical_speed_ratio < idle_thresh)
        is_collided: bool = hit
        is_in_target_zone: bool = state.touched_exit
        is_fwd_motion: bool = (move_eff >= 0.0)

        use_binoc: bool = (
            profile.use_binocular_gps_compasses if profile is not None
            else True
        )
        if use_binoc and len(gps_progress) >= 2:
            path_intensity: float = 0.5 * (
                max(0.0, gps_progress[0]) + max(0.0, gps_progress[1])
            )
        elif len(gps_progress) >= 1:
            path_intensity = max(0.0, gps_progress[0])
        else:
            path_intensity = 0.0

        eta_behavior: float = max(
            0.0, min(1.0, physical_speed_ratio / max(1e-4, heal_thresh))
        )

        invert_target: bool = (
            profile.invert_target_zone_field if profile is not None
            else False
        )

        # 1. Total Per-Frame Damage Vector
        move_dmg_rate: float = (
            profile.move_fwd_dmg_per_frame if is_fwd_motion
            else profile.move_bwd_dmg_per_frame
        ) if profile is not None else 0.0

        d_base: float = (
            profile.base_dmg_per_frame if profile is not None else 0.0
        )
        d_move: float = (
            (move_dmg_rate * physical_speed_ratio)
            if is_moving else 0.0
        )
        d_spin: float = (
            (profile.spin_dmg_per_frame * rot_ratio)
            if (profile is not None and is_spinning) else 0.0
        )
        d_coll: float = (
            profile.coll_dmg_per_frame
            if (profile is not None and is_collided) else 0.0
        )
        d_idle: float = (
            profile.idle_dmg_per_frame
            if (profile is not None and is_idle) else 0.0
        )
        d_path: float = (
            (profile.path_dmg_per_frame * (1.0 - path_intensity))
            if profile is not None else 0.0
        )

        if invert_target:
            d_target: float = (
                (profile.target_hold_dmg_per_frame * f_dist)
                if profile is not None else 0.0
            )
        else:
            d_target = (
                profile.target_hold_dmg_per_frame
                if (profile is not None and is_in_target_zone) else 0.0
            )

        raw_dmg: float = (
            d_base + d_move + d_spin + d_coll + d_idle + d_path + d_target
        )

        # 2. Total Per-Frame Healing Vector
        move_heal_rate: float = (
            profile.move_fwd_heal_per_frame if is_fwd_motion
            else profile.move_bwd_heal_per_frame
        ) if profile is not None else 0.0

        h_base: float = (
            profile.base_heal_per_frame if profile is not None else 0.0
        )
        h_move: float = (
            (move_heal_rate * eta_behavior)
            if is_moving else 0.0
        )
        h_spin: float = (
            (profile.spin_heal_per_frame * rot_ratio)
            if (profile is not None and is_spinning) else 0.0
        )
        h_coll: float = (
            profile.coll_heal_per_frame
            if (profile is not None and is_collided) else 0.0
        )
        h_idle: float = (
            profile.idle_heal_per_frame
            if (profile is not None and is_idle) else 0.0
        )
        h_path: float = (
            (profile.path_heal_per_frame * path_intensity)
            if profile is not None else 0.0
        )

        if invert_target:
            h_target: float = 0.0
        else:
            h_target = (
                (profile.target_hold_heal_per_frame * f_dist)
                if profile is not None else 0.0
            )

        raw_heal: float = (
            h_base + h_move + h_spin + h_coll + h_idle + h_path + h_target
        )

        # 3. Dynamic Field Capping Math
        if invert_target:
            if is_in_target_zone:
                total_heal: float = min(raw_heal, 0.95 * max(1e-6, raw_dmg))
                total_dmg: float = raw_dmg
            else:
                total_heal = raw_heal
                total_dmg = min(
                    raw_dmg, max(0.0, f_dist) * max(1e-6, raw_heal)
                )
        else:
            if is_in_target_zone:
                total_dmg: float = min(raw_dmg, 0.95 * max(1e-6, raw_heal))
                total_heal: float = raw_heal
            else:
                total_dmg = raw_dmg
                total_heal = min(
                    raw_heal, max(0.0, f_dist) * max(1e-6, raw_dmg)
                )

        # 4. Net Health Update
        net_hp_delta: float = total_heal - total_dmg
        if state.is_alive:
            state.health = max(0.0, min(1.0, state.health + net_hp_delta))

        if state.health <= 0.0:
            state.is_alive = False

        # --- Stage Hold & Clear Logic ---
        is_endless: bool = hasattr(map_data, "chunk_manager")
        if not is_endless:
            if state.is_alive and state.touched_exit:
                if state.first_touch_step < 0:
                    state.first_touch_step = step_idx

                state.hold_frame_counter += 1
                if state.hold_frame_counter >= target_hold_frames:
                    if state.first_hold_clear_step < 0:
                        state.first_hold_clear_step = step_idx

                    state.stages_cleared += 1
                    state.exit_solved = True
                    state.hold_frame_counter = 0
                    state.active_target_idx += 1

                    if (
                        profile is not None
                        and profile.full_heal_on_stage_clear
                    ):
                        state.health = 1.0

                    self.transformer.gps_sensor.reset_candidate_history(
                        candidate_idx
                    )
            else:
                state.hold_frame_counter = 0

        state.x = nx
        state.y = ny
        state.has_collided = hit
        state.frames_survived += 1

        state.last_speed_ratio = physical_speed_ratio
        state.last_collided = hit
        state.last_idle = is_idle
        state.last_healing = (total_heal > total_dmg and state.is_alive)
        state.last_rot_ratio = (
            rot_ratio if spin_dmg_rate > 0.0 else 0.0
        )

        curr_dist: int = pathfinder.get_step_distance(
            *state.tile_coords, stage_idx=state.active_target_idx
        )

        if curr_dist < state.best_step_dist:
            state.best_step_dist = curr_dist

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
            reached_exit=state.touched_exit
        )

        return state.is_alive
