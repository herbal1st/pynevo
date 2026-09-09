"""
Headless neuroevolution simulation trainer running progressive curriculum scaling across CPU cores.
"""

import time
import os
import json
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
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
from bridges.candidate_step_pipeline import CandidateStepPipeline
from neural.brain_persistence import BrainPersistence
from visualization.training_hud_overlay import TrainingHUDOverlay


_WORKER_FACTORY = None
_WORKER_KINEMATICS = None
_WORKER_TRANSFORMER = None


def _get_worker_env(profile_name: str):
    global _WORKER_FACTORY, _WORKER_KINEMATICS, _WORKER_TRANSFORMER
    if _WORKER_FACTORY is None:
        registry = AgentProfileRegistry()
        _WORKER_FACTORY = AgentFactory(registry, profile_name)
        _WORKER_KINEMATICS = _WORKER_FACTORY.create_kinematics()
        _WORKER_TRANSFORMER = _WORKER_FACTORY.create_transformer()
    return _WORKER_FACTORY, _WORKER_KINEMATICS, _WORKER_TRANSFORMER


class _DummyWorkerRecorder:
    def record_step_data(self, *args, **kwargs):
        pass


def _worker_init():
    pass


def _simulate_candidate_substep(args) -> Tuple[int, AgentState, np.ndarray]:
    (
        c_idx, state, flat_weights, map_data,
        max_steps, profile_name, target_hold_frames
    ) = args

    factory, kinematics, transformer = _get_worker_env(profile_name)
    net = factory.create_network()
    net.import_flat_weights(flat_weights)
    pipeline = CandidateStepPipeline(transformer, kinematics)
    transformer.reset_candidate_history(c_idx)

    telemetry_rows = np.zeros((1, 8), dtype=np.float32)
    dummy_rec = _DummyWorkerRecorder()

    for step in range(max_steps):
        if not state.is_alive:
            break

        pipeline.execute_step(
            step,
            state,
            net,
            map_data,
            None,
            dummy_rec,
            candidate_idx=c_idx,
            target_hold_frames=target_hold_frames
        )
        if state.touched_exit:
            break

    return c_idx, state, telemetry_rows


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

        total_cpus = multiprocessing.cpu_count()
        self.workers_count = max(1, total_cpus - 1) if total_cpus > 1 else 1

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
        initial_alloc = 200 if self.is_prog_mode else gens_count
        self.recorder.allocate_session_buffers(
            self.max_steps, self.pop_size, initial_alloc, param_cnt
        )

        gen_idx = 0
        try:
            with ProcessPoolExecutor(max_workers=self.workers_count, initializer=_worker_init) as executor:
                while gen_idx < gens_count:
                    if not self._handle_window_events():
                        break

                    gen_start_time = time.perf_counter()
                    map_data = self.locked_map_data
                    start_x, start_y = map_data.start_pos

                    candidate_states = [
                        AgentState(float(start_x) + 0.5, float(start_y) + 0.5)
                        for _ in range(self.pop_size)
                    ]

                    transformer = self.factory.create_transformer()
                    for c_idx, state in enumerate(candidate_states):
                        state.heading = transformer.generate_random_heading(
                            map_data, map_data.start_pos
                        )

                    tasks = [
                        (
                            c_idx,
                            candidate_states[c_idx],
                            self.population.networks[c_idx].export_flat_weights(),
                            map_data,
                            self.max_steps,
                            self.active_profile_name,
                            self.training_profile.target_hold_frames
                        )
                        for c_idx in range(self.pop_size)
                    ]

                    results = list(executor.map(_simulate_candidate_substep, tasks, chunksize=4))

                    actual_steps = 1
                    for c_idx, final_state, _ in results:
                        candidate_states[c_idx] = final_state
                        actual_steps = max(actual_steps, final_state.frames_survived)

                    raw_scores = [
                        FitnessEvaluator.calculate_raw_score(
                            c_state,
                            max_steps=self.max_steps,
                            stage_bonus=2500.0,
                            lost_hp_impact=self.training_profile.lost_hp_score_impact_ratio
                        )
                        for c_state in candidate_states
                    ]
                    norm_scores = FitnessEvaluator.normalize_scores(raw_scores)

                    if self.recorder.weight_bundler and gen_idx >= self.recorder.weight_bundler.num_generations:
                        new_cap = self.recorder.weight_bundler.num_generations + 100
                        new_tensor = np.zeros((new_cap, self.pop_size, param_cnt), dtype=np.float16)
                        new_tensor[:self.recorder.weight_bundler.num_generations] = self.recorder.weight_bundler.master_tensor
                        self.recorder.weight_bundler._tensor = new_tensor
                        self.recorder.weight_bundler.num_generations = new_cap

                    self.recorder.finalize_generation(
                        gen_idx,
                        map_data,
                        raw_scores,
                        norm_scores,
                        actual_steps,
                        pop_networks=self.population.networks
                    )

                    elapsed_sec = time.perf_counter() - gen_start_time
                    winner_idx = int(np.argmax(norm_scores))
                    winner_net = self.population.networks[winner_idx]

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

                    self.screen.fill(config.COLOR_BG)
                    self.hud_overlay.draw_hud(self.screen, gen_idx, gens_count)
                    pygame.display.flip()

                    # Dynamic curriculum upgrade with Champion Policy Transfer
                    if self.is_prog_mode and (gen_solve_ratio >= 0.12 or running_mean_ratio >= 0.08):
                        self.persistence.save_brain(
                            self.active_profile_name,
                            winner_net,
                            self.factory.profile,
                            context=f"curriculum {self.prog_width}x{self.prog_height}"
                        )
                        # Seed new population from champion network
                        self.population.seed_population_from_brain(winner_net)

                        self.prog_width += 4
                        self.prog_height += 3
                        self.prog_stage += 1
                        self.solve_ratio_history.clear()
                        self._rebuild_prog_map()

                    elif not self.is_prog_mode and gen_idx == gens_count - 1:
                        self.persistence.save_brain(
                            self.active_profile_name,
                            winner_net,
                            self.factory.profile,
                            context="training"
                        )

                    self.population.evolve_next_generation(norm_scores)
                    self.completed_generations += 1

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
