"""
Curriculum Neuroevolution Trainer with Programmable Multi-Metric Gatekeepers.
Dynamically validates custom combinations of DONE, EXITS, CNTR, EFFC, PACE, and EXPL per stage.
"""

import time
from typing import List, Optional
import numpy as np

import config
from core.map_generation.generator import MapGenerator
from core.pathfinder import BFSPathfinder
from entities.agent_profile_registry import AgentProfileRegistry
from entities.training_profile_registry import (
    TrainingProfileRegistry,
    ResolvedTrainingProfile,
    CurriculumStage
)
from entities.map_profile_registry import MapProfileRegistry, ResolvedMapProfile
from entities.agent_factory import AgentFactory
from entities.entity_state import AgentState
from evolution.fitness import FitnessEvaluator
from evolution.recorder import FrameRecorder
from bridges.cli_presenter import CLIPresenter
from neural.brain_persistence import BrainPersistence
from core.batch_engine import UniversalBatchRunner


class HeadlessTrainer:
    def __init__(
        self,
        active_profile_name: str = config.ACTIVE_AGENT_PROFILE,
        active_training_name: str = config.ACTIVE_TRAINING_PROFILE,
        active_map_name: str = config.ACTIVE_MAP_PROFILE
    ) -> None:
        self.active_profile_name: str = active_profile_name
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.factory: AgentFactory = AgentFactory(self.registry, active_profile_name)

        self.training_registry: TrainingProfileRegistry = TrainingProfileRegistry()
        self.training_profile: ResolvedTrainingProfile = (
            self.training_registry.get_profile(active_training_name)
        )

        self.map_registry: MapProfileRegistry = MapProfileRegistry()
        self.map_profile: ResolvedMapProfile = self.map_registry.get_profile(active_map_name)

        self.pop_size: int = self.training_profile.population_size
        self.global_max_steps: int = self.training_profile.max_simulation_steps
        self.curriculum: List[CurriculumStage] = list(self.training_profile.curriculum)

        self.map_generator: MapGenerator = MapGenerator(map_profile=self.map_profile)
        self.recorder: FrameRecorder = FrameRecorder()
        self.persistence: BrainPersistence = BrainPersistence()

        template_net = self.factory.create_network()
        self.param_count = template_net.param_count
        self.total_in_dim = template_net.layers[0].weights.shape[0]
        self.hidden_dim = template_net.layers[0].weights.shape[1]

        self.pop_weights = np.zeros((self.pop_size, self.param_count), dtype=np.float32)
        for i in range(self.pop_size):
            net = self.factory.create_network()
            self.pop_weights[i] = net.export_flat_weights().astype(np.float32)

        seed_net = self.factory.create_network()
        if self.persistence.load_brain(
            active_profile_name, seed_net, self.factory.profile, context="training"
        ):
            flat_seed = seed_net.export_flat_weights().astype(np.float32)
            self.pop_weights[:] = flat_seed
            noise = np.random.normal(0.0, 0.125, self.pop_weights[1:].shape)
            self.pop_weights[1:] += noise.astype(np.float32)

        self.cli_presenter: CLIPresenter = CLIPresenter(
            self.pop_size, self.global_max_steps
        )

        self.batch_runner: UniversalBatchRunner = UniversalBatchRunner(
            pop_size=self.pop_size,
            max_steps=self.global_max_steps,
            num_rays=self.factory.profile.vision_rays,
            hidden_dim=self.factory.profile.neurons
        )

    def run_training_session(
        self,
        num_generations: Optional[int] = None
    ) -> FrameRecorder:
        max_buffer_gens = 1500
        replay_pop_size: int = min(25, self.pop_size)
        self.recorder.allocate_session_buffers(
            self.global_max_steps, replay_pop_size, max_buffer_gens, self.param_count
        )

        self.cli_presenter.print_start_banner(profile_name=self.active_profile_name)

        current_global_gen = 0
        stage_list = self.curriculum
        best_overall_weights = None

        for s_idx, stage in enumerate(stage_list):
            gate_info = []
            if stage.min_done_pct > 0: gate_info.append(f"DONE>={stage.min_done_pct}%")
            if stage.min_exits > 0: gate_info.append(f"EXITS>={stage.min_exits}")
            if stage.min_cntr_pct > 0: gate_info.append(f"CNTR>={stage.min_cntr_pct}%")
            if stage.min_effc_pct > 0: gate_info.append(f"EFFC>={stage.min_effc_pct}%")
            if stage.min_pace_pct > 0: gate_info.append(f"PACE>={stage.min_pace_pct}%")
            if stage.min_expl_pct > 0: gate_info.append(f"EXPL>={stage.min_expl_pct}%")
            gate_str = ", ".join(gate_info) if gate_info else "EXITS>=1"

            print(
                f"\n>>> {stage.name} [{s_idx + 1}/{len(stage_list)}] "
                f"| Grid: {stage.map_width}x{stage.map_height} (Density: {stage.wall_density}) "
                f"| Steps: {stage.max_simulation_steps} | Min Rounds: {stage.min_mandatory_generations} "
                f"| Gate: [{gate_str}] x{stage.consecutive_rounds_needed}"
            )
            print("-" * 85)

            self.map_generator.width = stage.map_width
            self.map_generator.height = stage.map_height
            self.map_generator.map_profile = ResolvedMapProfile(
                profile_name="DYNAMIC",
                map_type=self.map_profile.map_type,
                map_width=stage.map_width,
                map_height=stage.map_height,
                tile_size=self.map_profile.tile_size,
                wall_density=stage.wall_density,
                stem_early_termination_rate=self.map_profile.stem_early_termination_rate,
                min_straight_start_steps=self.map_profile.min_straight_start_steps
            )

            self.batch_runner.max_steps = stage.max_simulation_steps
            mutation_rate = stage.mutation_rate
            mutation_scale = stage.mutation_scale

            stage_gen_count = 0
            consecutive_passes = 0

            while True:
                stage_gen_count += 1
                gen_start_time = time.perf_counter()

                map_data = self.map_generator.generate_solvable_map()
                pathfinder = BFSPathfinder(map_data)
                pathfinder.compute_distance_matrix()

                start_x, start_y = map_data.start_pos
                initial_bfs_dist = pathfinder.get_step_distance(start_x, start_y)
                num_turns = pathfinder.count_shortest_path_turns()

                theoretical_max = FitnessEvaluator.calculate_theoretical_max_score(
                    initial_bfs_dist,
                    stage.max_simulation_steps,
                    move_speed=self.factory.profile.move_speed,
                    dist_ratio=self.training_profile.dist_to_time_bonus_ratio,
                    num_turns=num_turns
                )

                # 1. Primary Maze Rollout
                raw_scores_1, summary_matrix, telemetry_matrix = self.batch_runner.run_generation_from_tensor(
                    map_data,
                    pathfinder,
                    self.pop_weights,
                    profile=self.factory.profile,
                    move_speed=self.factory.profile.move_speed
                )

                # 2. Validation Maze (Eliminates Luck)
                map_data_val = self.map_generator.generate_solvable_map()
                pathfinder_val = BFSPathfinder(map_data_val)
                pathfinder_val.compute_distance_matrix()
                raw_scores_2, _, _ = self.batch_runner.run_generation_from_tensor(
                    map_data_val,
                    pathfinder_val,
                    self.pop_weights,
                    profile=self.factory.profile,
                    move_speed=self.factory.profile.move_speed
                )

                combined_scores = (np.array(raw_scores_1, dtype=np.float32) + np.array(raw_scores_2, dtype=np.float32)) * 0.5
                raw_scores = combined_scores.tolist()

                # Vectorized Metrics
                scores_arr = np.array(raw_scores, dtype=np.float32)
                min_s = float(np.min(scores_arr))
                max_s = float(np.max(scores_arr))
                span = max_s - min_s
                norm_scores = ((scores_arr - min_s) / (span if span > 1e-6 else 1.0)).tolist()

                scaled_scores = [
                    float(min(1000.0, max(0.0, (s / theoretical_max) * 1000.0)))
                    for s in raw_scores
                ]

                ranked_indices = np.argsort(scores_arr)[::-1]
                winner_idx = int(ranked_indices[0])
                best_overall_weights = self.pop_weights[winner_idx].copy()

                min_step_dist = int(summary_matrix[winner_idx, 0])
                dist_reduced = max(0.0, float(initial_bfs_dist - min_step_dist))
                done_pct = int(round((dist_reduced / max(1e-6, initial_bfs_dist)) * 100))

                # Diagnostics
                total_open_tiles = sum(1 for y in range(map_data.height) for x in range(map_data.width) if map_data.is_walkable(x, y))
                tiles_explored_winner = int(summary_matrix[winner_idx, 2])
                expl_pct = int(round((tiles_explored_winner / max(1, total_open_tiles)) * 100))
                cntr_pct = int(summary_matrix[winner_idx, 3])
                effc_pct = int(summary_matrix[winner_idx, 4])
                pace_pct = int(summary_matrix[winner_idx, 5])

                winner_solved = (summary_matrix[winner_idx, 1] == 1)
                total_exits = int(np.sum(summary_matrix[:, 1]))
                if winner_solved:
                    done_pct = 100

                # 3. Save Top 25 to Replay Cache
                top_k_indices = ranked_indices[:replay_pop_size]
                sliced_telemetry = np.zeros(
                    (self.global_max_steps, replay_pop_size, 8), dtype=np.float32
                )
                sliced_telemetry[:stage.max_simulation_steps] = telemetry_matrix[
                    :stage.max_simulation_steps, :, :
                ]

                sliced_scaled = [scaled_scores[i] for i in top_k_indices]
                sliced_norm = [norm_scores[i] for i in top_k_indices]

                sliced_nets = []
                for k_idx in top_k_indices:
                    net = self.factory.create_network()
                    net.import_flat_weights(self.pop_weights[k_idx])
                    sliced_nets.append(net)

                if self.recorder.telemetry_bundler is not None:
                    self.recorder.telemetry_bundler._curr_buffer = sliced_telemetry.copy()

                self.recorder.finalize_generation(
                    current_global_gen,
                    map_data,
                    sliced_scaled,
                    sliced_norm,
                    stage.max_simulation_steps,
                    pop_networks=sliced_nets
                )

                elapsed_sec = time.perf_counter() - gen_start_time

                streak_str = f"{consecutive_passes}/{stage.consecutive_rounds_needed}"

                self.cli_presenter.print_generation_row(
                    current_global_gen,
                    scaled_scores,
                    norm_scores,
                    done_pct,
                    expl_pct,
                    effc_pct,
                    pace_pct,
                    cntr_pct,
                    total_exits,
                    streak_str,
                    elapsed_sec
                )

                # =============================================================
                # Programmable Multi-Metric Gatekeeper Evaluation
                # =============================================================
                gate_passed = True
                if done_pct < stage.min_done_pct: gate_passed = False
                if total_exits < stage.min_exits: gate_passed = False
                if cntr_pct < stage.min_cntr_pct: gate_passed = False
                if effc_pct < stage.min_effc_pct: gate_passed = False
                if pace_pct < stage.min_pace_pct: gate_passed = False
                if expl_pct < stage.min_expl_pct: gate_passed = False

                if stage_gen_count > stage.min_mandatory_generations:
                    if gate_passed:
                        consecutive_passes += 1
                    else:
                        consecutive_passes = 0
                else:
                    if stage_gen_count % 25 == 0:
                        print(f"    --> [Mandatory Warmup: {stage_gen_count}/{stage.min_mandatory_generations} rounds completed]")

                # 4. Vectorized Genetic Algorithm (k=8 Tournament)
                k_tourn = 8
                num_elites = max(2, int(self.pop_size * 0.03))

                new_pop_weights = np.zeros_like(self.pop_weights)
                new_pop_weights[:num_elites] = self.pop_weights[ranked_indices[:num_elites]]

                t_idx_a = np.random.randint(0, self.pop_size, size=(self.pop_size - num_elites, k_tourn))
                t_scores_a = scores_arr[t_idx_a]
                best_a = t_idx_a[np.arange(self.pop_size - num_elites), np.argmax(t_scores_a, axis=1)]

                t_idx_b = np.random.randint(0, self.pop_size, size=(self.pop_size - num_elites, k_tourn))
                t_scores_b = scores_arr[t_idx_b]
                best_b = t_idx_b[np.arange(self.pop_size - num_elites), np.argmax(t_scores_b, axis=1)]

                parent_a_w = self.pop_weights[best_a]
                parent_b_w = self.pop_weights[best_b]
                cross_mask = np.random.rand(*parent_a_w.shape) < 0.5
                children = np.where(cross_mask, parent_a_w, parent_b_w)

                mut_mask = np.random.rand(*children.shape) < mutation_rate
                noise = np.random.normal(0.0, mutation_scale, size=children.shape)
                children += (mut_mask * noise).astype(np.float32)

                new_pop_weights[num_elites:] = children
                self.pop_weights = new_pop_weights

                current_global_gen += 1

                # Check if Stage is Conquered
                if stage_gen_count >= stage.min_mandatory_generations and consecutive_passes >= stage.consecutive_rounds_needed:
                    if s_idx < len(stage_list) - 1:
                        print(
                            f"\n[Curriculum Gatekeeper] STAGE CONQUERED! "
                            f"{stage_gen_count} rounds completed with all {stage.consecutive_rounds_needed} consecutive gates passed. "
                            f"Graduating immediately to next stage...\n"
                        )
                    break

        if best_overall_weights is not None:
            final_net = self.factory.create_network()
            final_net.import_flat_weights(best_overall_weights)
            self.persistence.save_brain(
                self.active_profile_name,
                final_net,
                self.factory.profile,
                context="programmable multi-gate curriculum"
            )

        self.cli_presenter.print_finish_footer()
        self.recorder.save_temporary_disk_archive()
        return self.recorder
