"""
Headless neuroevolution simulation trainer running progressive curriculum scaling across CPU cores.
Accelerated with parallel multi-threaded Numba JIT kernels and pure SLAM embodied perception.
Zero-leak, constant multi-gen/s throughput regardless of generation count or maze size.
"""

import time
import os
import gc
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple, Deque
from collections import deque
import numpy as np
import pygame

import config
from core.warmup import warmup_jit
from core.map_generation.generator import MapGenerator
from entities.agent_profile_registry import AgentProfileRegistry
from entities.training_profile_registry import (
    TrainingProfileRegistry,
    ResolvedTrainingProfile
)
from entities.map_profile_registry import (
    MapProfileRegistry,
    ResolvedMapProfile
)
from entities.agent_factory import AgentFactory
from entities.entity_state import AgentState
from evolution.fitness import FitnessEvaluator
from evolution.population import PopulationManager
from evolution.recorder import FrameRecorder
from evolution.fast_simulation import simulate_population_parallel_jit
from perception.spawn_heading import SpawnHeadingGenerator
from neural.brain_persistence import BrainPersistence
from visualization.training_hud_overlay import TrainingHUDOverlay


class HeadlessTrainer:
    def __init__(
        self,
        active_profile_name: str = config.ACTIVE_AGENT_PROFILE,
        active_training_name: str = config.ACTIVE_TRAINING_PROFILE,
        active_map_name: str = config.ACTIVE_MAP_PROFILE
    ) -> None:
        self.completed_generations = 0
        warmup_jit()

        self.active_profile_name: str = active_profile_name
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.factory: AgentFactory = AgentFactory(
            self.registry, active_profile_name
        )

        self.training_registry: TrainingProfileRegistry = TrainingProfileRegistry()
        self.training_profile: ResolvedTrainingProfile = (
            self.training_registry.get_profile(active_training_name)
        )

        self.map_registry: MapProfileRegistry = MapProfileRegistry()
        self.is_prog_mode: bool = (active_map_name.upper() == "PROG")
        self.solve_ratio_history: Deque[float] = deque(maxlen=5)

        if self.is_prog_mode:
            self.persistence = BrainPersistence()
            (
                self.prog_stage,
                self.prog_width,
                self.prog_height,
                self.prog_generation,
            ) = self._load_curriculum_state()

            self.map_profile: ResolvedMapProfile = ResolvedMapProfile(
                profile_name="PROG",
                map_type="BRANCHING_WALLS",
                map_width=self.prog_width,
                map_height=self.prog_height,
                tile_size=max(12, int(40 * (16 / self.prog_width))),
                wall_density=min(0.78, 0.70 + (self.prog_width * 0.001)),
                stem_early_termination_rate=0.05,
                min_straight_start_steps=1
            )
        else:
            self.map_profile = self.map_registry.get_profile(active_map_name)

        self.pop_size: int = self.training_profile.population_size
        self.max_steps: int = self.training_profile.max_simulation_steps
        self.map_generator: MapGenerator = MapGenerator(
            map_profile=self.map_profile
        )

        min_diff = self.training_profile.min_path_difficulty_ratio
        max_diff = self.training_profile.max_path_difficulty_ratio
        self.locked_map_data = self.map_generator.generate_solvable_map(
            min_difficulty_ratio=min_diff,
            max_difficulty_ratio=max_diff
        )
        self.locked_map_data.target_sequence = [self.locked_map_data.exit_pos]

        self.persistence: BrainPersistence = BrainPersistence()
        self.population: PopulationManager = PopulationManager(
            factory=self.factory,
            pop_size=self.pop_size,
            mutation_rate=self.training_profile.mutation_rate,
            mutation_scale=self.training_profile.mutation_scale,
            elitism_ratio=self.training_profile.elitism_ratio
        )
        self._restore_evolution_state()
        self.recorder: FrameRecorder = FrameRecorder()

        pygame.init()
        try:
            self.screen: pygame.Surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        except Exception:
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

        fw, fh = self.screen.get_size()
        pygame.display.set_caption("PyNevo - Headless Neuroevolution Training Monitor")
        self.hud_overlay: TrainingHUDOverlay = TrainingHUDOverlay((20, 20, fw - 40, fh - 40))

    def _restore_evolution_state(self) -> None:
        try:
            restored = self.population.load_checkpoint(
                self.active_profile_name, curriculum_name="PROG"
            )
        except TypeError:
            restored = self.population.load_checkpoint(self.active_profile_name)

        if not restored:
            return

        self.prog_stage = int(restored.get("stage", 0))
        self.prog_width = int(restored.get("width", 16))
        self.prog_height = int(restored.get("height", 12))
        self.completed_generations = int(restored.get("completed_generations", 0))
        self._rebuild_prog_map()

    def _curriculum_state_path(self) -> Path:
        safe_p = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.active_profile_name)
        safe_t = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.training_profile.profile_name)
        return self.persistence.storage_dir / f"curriculum_{safe_p}_{safe_t}_PROG.json"

    def _load_curriculum_state(self) -> Tuple[int, int, int, int]:
        default = (0, 16, 12, 0)
        path = self._curriculum_state_path()
        if not path.exists():
            return default

        try:
            with path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            w = int(state["width"])
            h = int(state["height"])
            st = int(state.get("stage", max(0, (w - 16) // 4)))
            gen = int(state.get("generation", 0))
            return st, w, h, gen
        except Exception:
            return default

    def _save_curriculum_state(self, generation: int) -> None:
        if not self.is_prog_mode:
            return

        path = self._curriculum_state_path()
        tmp_path = path.with_suffix(".json.tmp")
        state = {
            "version": 1,
            "stage": int(self.prog_stage),
            "width": int(self.prog_width),
            "height": int(self.prog_height),
            "generation": max(0, int(generation)),
        }

        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(path)
        except OSError:
            pass

    def _rebuild_prog_map(self) -> None:
        self.map_profile = ResolvedMapProfile(
            profile_name="PROG",
            map_type="BRANCHING_WALLS",
            map_width=self.prog_width,
            map_height=self.prog_height,
            tile_size=max(12, int(40 * (16 / self.prog_width))),
            wall_density=min(0.78, 0.70 + (self.prog_width * 0.001)),
            stem_early_termination_rate=0.05,
            min_straight_start_steps=1
        )
        self.map_generator = MapGenerator(map_profile=self.map_profile)
        min_diff = self.training_profile.min_path_difficulty_ratio
        max_diff = self.training_profile.max_path_difficulty_ratio
        self.locked_map_data = self.map_generator.generate_solvable_map(
            min_difficulty_ratio=min_diff,
            max_difficulty_ratio=max_diff
        )
        self.locked_map_data.target_sequence = [self.locked_map_data.exit_pos]

    def _save_evolution_checkpoint(self) -> None:
        if not self.is_prog_mode:
            return

        self.population.save_checkpoint(
            profile_name=self.active_profile_name,
            stage=self.prog_stage,
            width=self.prog_width,
            height=self.prog_height,
            completed_generations=self.completed_generations,
            curriculum_name="PROG"
        )
        self._save_curriculum_state(self.completed_generations)

    def _handle_window_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def run_training_session(
        self,
        num_generations: Optional[int] = None
    ) -> FrameRecorder:
        gens_count: int = (
            999999 if self.is_prog_mode
            else (num_generations if num_generations is not None else self.training_profile.learning_generations)
        )

        param_cnt: int = self.population.networks[0].param_count
        initial_alloc = 100
        self.recorder.allocate_session_buffers(
            self.max_steps, self.pop_size, initial_alloc, param_cnt
        )

        prof = self.factory.profile
        v_rays = int(prof.vision_rays)
        base_dim = v_rays + 15
        memory_frames = int(prof.memory_frames)
        mem_frames = memory_frames
        in_dim = base_dim * (1 + mem_frames)
        hidden_layers = int(prof.hidden_layers)
        neurons = int(prof.neurons)
        output_size = 4

        half_arc = math.radians(prof.vision_arc_angle / 2.0)
        rel_angles = np.linspace(-half_arc, half_arc, v_rays, dtype=np.float64)
        vision_max_dist = float(prof.vision_max_dist)
        agent_radius = float(prof.agent_radius_ratio)
        move_speed = float(prof.move_speed)
        rad_per_frame = float(math.radians(prof.turn_speed) / config.FPS)
        profile_style_is_tank = (prof.profile_style.upper() == "TANK")
        use_linear_speed_output = bool(prof.use_linear_speed_output)

        idle_damage_speed_thresh = float(prof.idle_damage_speed_threshold)
        heal_speed_thresh = float(prof.heal_speed_threshold)
        coll_dmg = float(prof.health_coll_dmg_per_frame)
        idle_dmg = float(prof.health_idle_dmg_per_frame)
        spin_dmg_rate = float(prof.health_spin_dmg_per_frame)
        move_heal_rate = float(prof.move_heal_per_frame)
        hold_heal_rate = float(prof.target_hold_heal_per_frame)
        target_hold_frames = int(self.training_profile.target_hold_frames)

        # Pre-allocated single-instance arrays (Zero-leak reused memory)
        pop_weights = np.empty((self.pop_size, param_cnt), dtype=np.float32)
        out_final_x = np.empty(self.pop_size, dtype=np.float64)
        out_final_y = np.empty(self.pop_size, dtype=np.float64)
        out_final_heading = np.empty(self.pop_size, dtype=np.float64)
        out_final_health = np.empty(self.pop_size, dtype=np.float64)
        out_is_alive = np.empty(self.pop_size, dtype=np.bool_)
        out_touched_exit = np.empty(self.pop_size, dtype=np.bool_)
        out_first_touch_step = np.empty(self.pop_size, dtype=np.int32)
        out_first_hold_clear_step = np.empty(self.pop_size, dtype=np.int32)
        out_stages_cleared = np.empty(self.pop_size, dtype=np.int32)
        out_total_lifetime_progress = np.empty(self.pop_size, dtype=np.float64)
        out_distance_traveled = np.empty(self.pop_size, dtype=np.float64)
        out_collision_count = np.empty(self.pop_size, dtype=np.int32)
        out_frames_survived = np.empty(self.pop_size, dtype=np.int32)
        out_unique_visited = np.empty(self.pop_size, dtype=np.int32)
        out_max_disp = np.empty(self.pop_size, dtype=np.float64)
        out_visited_grids = np.zeros(
            (self.pop_size, self.locked_map_data.height, self.locked_map_data.width),
            dtype=np.uint8
        )

        # Pre-allocated thread scratch buffers (zero NRT allocations)
        mem_bufs = np.zeros((self.pop_size, in_dim), dtype=np.float32)
        base_vecs = np.zeros((self.pop_size, base_dim), dtype=np.float32)
        h_as = np.zeros((self.pop_size, neurons), dtype=np.float32)
        h_bs = np.zeros((self.pop_size, neurons), dtype=np.float32)

        initial_headings = np.empty(self.pop_size, dtype=np.float64)
        candidate_states = [AgentState(0.0, 0.0) for _ in range(self.pop_size)]

        gen_idx = 0
        last_hud_render = 0.0

        try:
            while gen_idx < gens_count:
                if not self._handle_window_events():
                    break

                gen_start_time = time.perf_counter()
                map_data = self.locked_map_data
                start_x, start_y = map_data.start_pos
                target_x, target_y = map_data.exit_pos
                target_cx = float(target_x) + 0.5
                target_cy = float(target_y) + 0.5

                # Dynamically scale step budget with labyrinth perimeter so large mazes (56x42+) can be solved
                curriculum_steps = int((map_data.width + map_data.height) * 32)
                step_limit = max(self.max_steps, curriculum_steps)

                if out_visited_grids.shape[1] != map_data.height or out_visited_grids.shape[2] != map_data.width:
                    out_visited_grids = np.zeros(
                        (self.pop_size, map_data.height, map_data.width), dtype=np.uint8
                    )
                else:
                    out_visited_grids.fill(0)

                # Divide swarm deterministically across open cardinal corridors
                open_headings = []
                for dx, dy, ang in ((0, -1, 3.0 * math.pi / 2.0), (1, 0, 0.0), (0, 1, math.pi / 2.0), (-1, 0, math.pi)):
                    if map_data.is_walkable(start_x + dx, start_y + dy):
                        open_headings.append(ang)
                if not open_headings:
                    open_headings = [0.0]

                n_open = len(open_headings)
                for c_idx in range(self.pop_size):
                    initial_headings[c_idx] = open_headings[c_idx % n_open]

                for c_idx in range(self.pop_size):
                    pop_weights[c_idx] = self.population.networks[c_idx].param_buffer

                simulate_population_parallel_jit(
                    pop_weights,
                    float(start_x) + 0.5, float(start_y) + 0.5,
                    initial_headings,
                    map_data.grid_array, map_data.width, map_data.height,
                    target_cx, target_cy,
                    in_dim, hidden_layers, neurons, output_size,
                    v_rays, rel_angles, vision_max_dist, memory_frames,
                    agent_radius, move_speed, rad_per_frame,
                    profile_style_is_tank, use_linear_speed_output,
                    idle_damage_speed_thresh, heal_speed_thresh,
                    coll_dmg, idle_dmg, spin_dmg_rate, move_heal_rate, hold_heal_rate,
                    target_hold_frames, step_limit,
                    out_final_x, out_final_y, out_final_heading, out_final_health,
                    out_is_alive, out_touched_exit, out_first_touch_step,
                    out_first_hold_clear_step, out_stages_cleared, out_total_lifetime_progress,
                    out_distance_traveled, out_collision_count, out_frames_survived,
                    out_unique_visited, out_max_disp, out_visited_grids,
                    mem_bufs, base_vecs, h_as, h_bs
                )

                actual_steps = int(np.max(out_frames_survived))
                if actual_steps < 1:
                    actual_steps = 1

                for c_idx in range(self.pop_size):
                    st = candidate_states[c_idx]
                    st.start_x = float(start_x) + 0.5
                    st.start_y = float(start_y) + 0.5
                    st.x = out_final_x[c_idx]
                    st.y = out_final_y[c_idx]
                    st.heading = out_final_heading[c_idx]
                    st.health = out_final_health[c_idx]
                    st.is_alive = bool(out_is_alive[c_idx])
                    st.touched_exit = bool(out_touched_exit[c_idx])
                    st.first_touch_step = int(out_first_touch_step[c_idx])
                    st.first_hold_clear_step = int(out_first_hold_clear_step[c_idx])
                    st.stages_cleared = int(out_stages_cleared[c_idx])
                    st.total_lifetime_progress = float(out_total_lifetime_progress[c_idx])
                    st.distance_traveled = float(out_distance_traveled[c_idx])
                    st.collision_count = int(out_collision_count[c_idx])
                    st.frames_survived = int(out_frames_survived[c_idx])
                    st.unique_visited_count = int(out_unique_visited[c_idx])
                    st.max_disp = float(out_max_disp[c_idx])
                    st.visited_tiles = set()

                raw_scores = [
                    FitnessEvaluator.calculate_raw_score(
                        c_state,
                        max_steps=step_limit,
                        stage_bonus=2500.0,
                        lost_hp_impact=self.training_profile.lost_hp_score_impact_ratio
                    )
                    for c_state in candidate_states
                ]
                norm_scores = FitnessEvaluator.normalize_scores(raw_scores)
                winner_idx = int(np.argmax(norm_scores))
                winner_net = self.population.networks[winner_idx]

                ys, xs = np.where(out_visited_grids[winner_idx])
                candidate_states[winner_idx].visited_tiles = set(zip(xs.tolist(), ys.tolist()))

                self.recorder.finalize_generation(
                    gen_idx,
                    map_data,
                    raw_scores,
                    norm_scores,
                    actual_steps,
                    pop_networks=self.population.networks
                )

                elapsed_sec = time.perf_counter() - gen_start_time
                solve_cnt = sum(
                    1 for c in candidate_states
                    if c.stages_cleared > 0 or c.first_touch_step >= 0 or c.touched_exit
                )
                gen_solve_ratio = float(solve_cnt) / float(self.pop_size)
                self.solve_ratio_history.append(gen_solve_ratio)
                running_mean_ratio = sum(self.solve_ratio_history) / float(len(self.solve_ratio_history))

                self.hud_overlay.record_generation(
                    gen_idx, raw_scores, norm_scores, candidate_states,
                    running_solve_avg=running_mean_ratio, elapsed_sec=elapsed_sec,
                    map_width=self.prog_width if self.is_prog_mode else self.map_profile.map_width,
                    map_height=self.prog_height if self.is_prog_mode else self.map_profile.map_height
                )

                now = time.perf_counter()
                if now - last_hud_render >= 0.033 or gen_idx % 10 == 0:
                    self.screen.fill(config.COLOR_BG)
                    self.hud_overlay.draw_hud(self.screen, gen_idx, gens_count)
                    pygame.display.flip()
                    last_hud_render = now

                if gen_idx % 25 == 0:
                    top_s = max(raw_scores)
                    avg_s = sum(raw_scores) / float(len(raw_scores))
                    fps_val = 1.0 / max(1e-5, elapsed_sec)
                    print(
                        f"[Train] Gen {gen_idx + 1:5d} | Size: {map_data.width}x{map_data.height} | "
                        f"Top: {top_s:6.1f} | Avg: {avg_s:6.1f} | "
                        f"Solves: {solve_cnt}/{self.pop_size} ({gen_solve_ratio * 100:4.1f}%) | "
                        f"{fps_val:5.1f} gen/s"
                    )

                if self.is_prog_mode and (solve_cnt >= 1 or gen_solve_ratio >= 0.02 or running_mean_ratio >= 0.01):
                    print(
                        f"[Curriculum] Stage {self.prog_stage} mastered ({map_data.width}x{map_data.height})! "
                        f"Transferring champion to larger maze..."
                    )
                    self.persistence.save_brain(
                        self.active_profile_name,
                        winner_net,
                        self.factory.profile,
                        context=f"curriculum {self.prog_width}x{self.prog_height}"
                    )
                    self.population.seed_population_from_brain(winner_net)

                    self.prog_width += 4
                    self.prog_height += 3
                    self.prog_stage += 1
                    self.solve_ratio_history.clear()
                    self.stage_gen_counter = 0
                    self._rebuild_prog_map()

                elif not self.is_prog_mode and gen_idx == gens_count - 1:
                    self.persistence.save_brain(
                        self.active_profile_name,
                        winner_net,
                        self.factory.profile,
                        context="training"
                    )

                # Re-roll procedural seed if stuck on a single degenerate maze layout for 80 generations
                if not hasattr(self, "stage_gen_counter"):
                    self.stage_gen_counter = 0
                self.stage_gen_counter += 1

                if self.is_prog_mode and self.stage_gen_counter >= 80 and solve_cnt == 0:
                    print(f"[Curriculum] Stage {self.prog_stage} ({map_data.width}x{map_data.height}) re-rolling procedural maze layout...")
                    self._rebuild_prog_map()
                    self.stage_gen_counter = 0

                self.population.evolve_next_generation(norm_scores)
                self.completed_generations += 1

                if gen_idx % 50 == 0:
                    gc.collect()
                    if self.is_prog_mode:
                        self._save_evolution_checkpoint()

                gen_idx += 1

        except KeyboardInterrupt:
            if self.is_prog_mode:
                self._save_evolution_checkpoint()
            elif "norm_scores" in locals():
                winner_idx = int(np.argmax(norm_scores))
                winner_net = self.population.networks[winner_idx]
                self.persistence.save_brain(
                    self.active_profile_name,
                    winner_net,
                    self.factory.profile,
                    context="final training"
                )

        self.recorder.save_temporary_disk_archive()
        return self.recorder
