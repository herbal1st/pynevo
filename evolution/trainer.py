"""
Headless neuroevolution simulation trainer running progressive curriculum scaling across CPU cores.
"""

import time
import os
import json
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple
import numpy as np

import config
from core.warmup import warmup_jit
from core.map_generation.generator import MapGenerator
from core.pathfinder import BFSPathfinder
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
from bridges.cli_presenter import CLIPresenter
from neural.brain_persistence import BrainPersistence


def _worker_init(core_id: int):
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core_id})
        except Exception:
            pass


def _simulate_candidate_substep(args) -> Tuple[int, AgentState, np.ndarray]:
    (
        c_idx, state, flat_weights, map_data, pathfinder_buf,
        max_steps, profile_name, target_hold_frames
    ) = args

    registry = AgentProfileRegistry()
    factory = AgentFactory(registry, profile_name)
    transformer = factory.create_transformer()
    kinematics = factory.create_kinematics()
    net = factory.create_network()
    net.import_flat_weights(flat_weights)
    pipeline = CandidateStepPipeline(transformer, kinematics)

    pathfinder = BFSPathfinder(map_data)
    pathfinder._matrix_buffer = pathfinder_buf

    telemetry_rows = np.zeros((max_steps, 8), dtype=np.float32)

    class FastWorkerRecorder:
        def record_step_data(self, step_idx, cand_idx, x, y, heading, health, dist, hit_wall, is_alive, reached_exit):
            if step_idx < max_steps:
                telemetry_rows[step_idx, 0] = x
                telemetry_rows[step_idx, 1] = y
                telemetry_rows[step_idx, 2] = heading
                telemetry_rows[step_idx, 3] = health
                telemetry_rows[step_idx, 4] = dist
                telemetry_rows[step_idx, 5] = 1.0 if hit_wall else 0.0
                telemetry_rows[step_idx, 6] = 1.0 if is_alive else 0.0
                telemetry_rows[step_idx, 7] = 1.0 if reached_exit else 0.0

    dummy_rec = FastWorkerRecorder()

    for step in range(1, max_steps + 1):
        step_idx = step - 1
        if not state.is_alive:
            if step_idx > 0:
                telemetry_rows[step_idx] = telemetry_rows[step_idx - 1]
            continue

        pipeline.execute_step(
            step_idx,
            state,
            net,
            map_data,
            pathfinder,
            dummy_rec,
            candidate_idx=c_idx,
            target_hold_frames=target_hold_frames
        )

    return c_idx, state, telemetry_rows


class HeadlessTrainer:
    """
    Coordinates headless training loops with Progressive Grid Expansion ('PROG').
    """

    def __init__(
        # Persistent evolutionary checkpoint progress.

        self,
        active_profile_name: str = config.ACTIVE_AGENT_PROFILE,
        active_training_name: str = config.ACTIVE_TRAINING_PROFILE,
        active_map_name: str = config.ACTIVE_MAP_PROFILE
    ) -> None:
        # Persistent evolutionary progress.
        self.completed_generations = 0
        # Persistent evolutionary progress.
        warmup_jit()

        self.active_profile_name: str = active_profile_name
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.factory: AgentFactory = AgentFactory(
            self.registry, active_profile_name
        )

        self.training_registry: TrainingProfileRegistry = (
            TrainingProfileRegistry()
        )
        self.training_profile: ResolvedTrainingProfile = (
            self.training_registry.get_profile(active_training_name)
        )

        self.map_registry: MapProfileRegistry = MapProfileRegistry()
        self.is_prog_mode: bool = (active_map_name.upper() == "PROG")

        # Progressive mode setup.
        #
        # Curriculum state is persistent and independent from the neural
        # checkpoint. This means Ctrl+C/restart resumes the same maze
        # difficulty instead of resetting to 16x12.
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
                tile_size=max(
                    12,
                    int(40 * (16 / self.prog_width))
                ),
                wall_density=min(
                    0.78,
                    0.70 + (self.prog_width * 0.001)
                ),
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
        if not self.is_prog_mode:
            self.persistence = BrainPersistence()


        self.cli_presenter: CLIPresenter = CLIPresenter(
            self.pop_size, self.max_steps
        )

        total_cpus = multiprocessing.cpu_count()
        self.usable_cores = list(range(1, total_cpus)) if total_cpus > 1 else [0]
        print(f"[Parallel Engine] Multi-core worker pool initialized across {len(self.usable_cores)} cores: {self.usable_cores}")

    def _restore_evolution_state(self) -> None:
        """Restore the full evolutionary population/checkpoint if one exists."""
        try:
            restored = self.population.load_checkpoint(
                self.active_profile_name,
                curriculum_name="PROG",
            )
        except TypeError:
            # Compatibility with implementations where the profile is implicit.
            restored = self.population.load_checkpoint(
                self.active_profile_name
            )

        if not restored:
            print("[Checkpoint] No full population checkpoint found; starting fresh.")
            return

        self.prog_stage = int(restored.get("stage", 0))
        self.prog_width = int(restored.get("width", 16))
        self.prog_height = int(restored.get("height", 12))
        self.completed_generations = int(
            restored.get("completed_generations", 0)
        )

        print(
            f"[Checkpoint] Restored full population: "
            f"stage={self.prog_stage}, "
            f"map={self.prog_width}x{self.prog_height}, "
            f"completed_generations={self.completed_generations}"
        )

        # Rebuild the active PROG map at the restored curriculum size.
        if hasattr(self, "_rebuild_prog_map"):
            self._rebuild_prog_map()

    def _curriculum_state_path(self) -> Path:
        """
        Returns the persistent curriculum-state file.

        State is kept beside the saved brain checkpoints but is separate
        from the .npz network format.
        """
        safe_profile = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in self.active_profile_name
        )
        safe_training = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in self.training_profile.profile_name
        )

        return (
            self.persistence.storage_dir
            / f"curriculum_{safe_profile}_{safe_training}_PROG.json"
        )

    def _load_curriculum_state(self) -> Tuple[int, int, int, int]:
        """
        Loads persistent progressive-curriculum state.

        Returns:
            stage, width, height, completed_generation_count

        Invalid/corrupt state safely falls back to the initial curriculum.
        """
        default = (0, 16, 12, 0)

        path = self._curriculum_state_path()

        if not path.exists():
            return default

        try:
            with path.open("r", encoding="utf-8") as f:
                state = json.load(f)

            width = int(state["width"])
            height = int(state["height"])
            stage = int(state.get("stage", max(0, (width - 16) // 4)))
            generation = int(state.get("generation", 0))

            # Curriculum geometry is deliberately constrained to the
            # progression used by the trainer.
            if width < 16 or width > 80:
                return default

            if height < 12 or height > 60:
                return default

            if (width - 16) % 4 != 0:
                return default

            if (height - 12) % 3 != 0:
                return default

            expected_stage = (width - 16) // 4

            if stage != expected_stage:
                stage = expected_stage

            generation = max(0, generation)

            print(
                f"[Curriculum] Resuming persistent curriculum: "
                f"stage {stage} | {width}x{height} | "
                f"{generation} completed generations"
            )

            return stage, width, height, generation

        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(
                f"[Curriculum] Warning: could not load curriculum state "
                f"({exc}); starting at 16x12."
            )
            return default

    def _save_curriculum_state(self, generation: int) -> None:
        """
        Atomically persists curriculum position.

        The generation number is informational/checkpoint metadata.
        The important restart invariant is the stage and its dimensions.
        """
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
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())

            tmp_path.replace(path)

            print(
                f"[Curriculum] State saved: "
                f"stage {self.prog_stage} | "
                f"{self.prog_width}x{self.prog_height} | "
                f"generation {generation}"
            )

        except OSError as exc:
            print(
                f"[Curriculum] Warning: failed to save curriculum state: "
                f"{exc}"
            )
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


    # ============================================================
    # Persistent curriculum / population state
    # ============================================================

    def _curriculum_state_path(self) -> Path:
        path = Path("saved_brains")
        path.mkdir(parents=True, exist_ok=True)

        return path / (
            f"curriculum_{self.active_profile_name}_PROG.json"
        )

    def _rebuild_prog_map(self) -> None:

        self.map_profile = ResolvedMapProfile(
            profile_name="PROG",
            map_type="BRANCHING_WALLS",
            map_width=self.prog_width,
            map_height=self.prog_height,
            tile_size=max(
                12,
                int(
                    40 *
                    (16 / self.prog_width)
                )
            ),
            wall_density=min(
                0.78,
                0.70 +
                (self.prog_width * 0.001)
            ),
            stem_early_termination_rate=0.05,
            min_straight_start_steps=1
        )

        self.map_generator = MapGenerator(
            map_profile=self.map_profile
        )

    def _load_evolution_checkpoint(self) -> bool:

        if not self.is_prog_mode:
            return False

        metadata = self.population.load_checkpoint(
            self.active_profile_name,
            "PROG"
        )

        if metadata is None:
            return False

        self.prog_stage = metadata["stage"]
        self.prog_width = metadata["width"]
        self.prog_height = metadata["height"]
        self.completed_generations = (
            metadata["completed_generations"]
        )

        self._rebuild_prog_map()
        self._save_curriculum_state(self.completed_generations)

        return True

    def _save_evolution_checkpoint(self) -> None:

        if not self.is_prog_mode:
            return

        path = self.population.save_checkpoint(
            profile_name=self.active_profile_name,
            stage=self.prog_stage,
            width=self.prog_width,
            height=self.prog_height,
            completed_generations=(
                self.completed_generations
            ),
            curriculum_name="PROG"
        )

        self._save_curriculum_state(self.completed_generations)

        print(
            "[Persistence] Checkpoint saved: "
            f"{path.name} | "
            f"stage={self.prog_stage} | "
            f"map={self.prog_width}x"
            f"{self.prog_height} | "
            f"generation="
            f"{self.completed_generations}"
        )

    def _upgrade_curriculum(self) -> None:

        old_w = self.prog_width
        old_h = self.prog_height

        # NO HARD CEILING.
        self.prog_width += 4
        self.prog_height += 3
        self.prog_stage += 1

        self._rebuild_prog_map()
        self._save_curriculum_state(self.completed_generations)

        print()
        print("=" * 80)
        print(
            "[CURRICULUM UPGRADE] "
            f"{old_w}x{old_h} conquered!"
        )
        print(
            "Expanding maze -> "
            f"{self.prog_width}x"
            f"{self.prog_height}"
        )
        print(
            "Curriculum stage -> "
            f"{self.prog_stage}"
        )
        print("=" * 80)
        print()

    def run_training_session(
        self,
        num_generations: Optional[int] = None
    ) -> FrameRecorder:
        # In PROG mode, runs indefinitely until user presses Ctrl+C
        gens_count: int = (
            999999 if self.is_prog_mode
            else (num_generations if num_generations is not None else self.training_profile.learning_generations)
        )

        param_cnt: int = self.population.networks[0].param_count
        # Allocate buffers for session
        initial_alloc = 200 if self.is_prog_mode else gens_count
        self.recorder.allocate_session_buffers(
            self.max_steps, self.pop_size, initial_alloc, param_cnt
        )

        self.cli_presenter.print_start_banner(
            agent_profile=self.active_profile_name,
            training_profile=self.training_profile.profile_name,
            map_profile=f"PROG ({self.prog_width}x{self.prog_height})" if self.is_prog_mode else self.map_profile.profile_name,
            hold_frames=self.training_profile.target_hold_frames
        )
        for w_str in AgentProfileRegistry.get_clamped_warning_strings():
            print(w_str)

        min_diff_ratio = self.training_profile.min_path_difficulty_ratio
        max_diff_ratio = self.training_profile.max_path_difficulty_ratio

        gen_idx = 0
        try:
            with ProcessPoolExecutor(max_workers=len(self.usable_cores)) as executor:
                while gen_idx < gens_count:
                    gen_start_time = time.perf_counter()

                    map_data = self.map_generator.generate_solvable_map(
                        min_difficulty_ratio=min_diff_ratio,
                        max_difficulty_ratio=max_diff_ratio
                    )
                    map_data.target_sequence = [map_data.exit_pos]

                    pathfinder = BFSPathfinder(map_data)
                    pathfinder.clear_cache()
                    pathfinder.compute_distance_matrix_for_target(
                        map_data.exit_pos, stage_idx=0
                    )

                    start_x, start_y = map_data.start_pos
                    initial_bfs_dist = pathfinder.get_step_distance(
                        start_x, start_y, stage_idx=0
                    )

                    candidate_states = [
                        AgentState(float(start_x) + 0.5, float(start_y) + 0.5)
                        for _ in range(self.pop_size)
                    ]

                    transformer = self.factory.create_transformer()
                    for c_idx, state in enumerate(candidate_states):
                        state.heading = transformer.generate_random_heading(
                            map_data, map_data.start_pos
                        )
                        state.best_step_dist = initial_bfs_dist

                    tasks = []
                    for c_idx in range(self.pop_size):
                        flat_w = self.population.networks[c_idx].export_flat_weights()
                        task_args = (
                            c_idx,
                            candidate_states[c_idx],
                            flat_w,
                            map_data,
                            pathfinder._matrix_buffer,
                            self.max_steps,
                            self.active_profile_name,
                            self.training_profile.target_hold_frames
                        )
                        tasks.append(task_args)

                    results = list(executor.map(_simulate_candidate_substep, tasks))

                    actual_steps = 1
                    for c_idx, final_state, t_rows in results:
                        candidate_states[c_idx] = final_state
                        if self.recorder.telemetry_bundler is not None and self.recorder.telemetry_bundler._curr_buffer is not None:
                            self.recorder.telemetry_bundler._curr_buffer[:, c_idx, :] = t_rows
                        actual_steps = max(actual_steps, final_state.frames_survived)

                    raw_scores = [
                        FitnessEvaluator.calculate_raw_score(
                            c_state,
                            initial_bfs_dist,
                            max_steps=self.max_steps,
                            stage_bonus=50.0,
                            lost_hp_impact=self.training_profile.lost_hp_score_impact_ratio
                        )
                        for c_state in candidate_states
                    ]
                    norm_scores = FitnessEvaluator.normalize_scores(raw_scores)

                    # Expand bundler buffer if running past initial allocation
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

                    self.cli_presenter.print_generation_row(
                        gen_idx,
                        raw_scores,
                        norm_scores,
                        candidate_states,
                        elapsed_sec
                    )

                    winner_idx = int(np.argmax(norm_scores))
                    winner_net = self.population.networks[winner_idx]

                    # Check 50% exits threshold
                    exits_cnt = sum(
                        1 for c in candidate_states
                        if c.first_touch_step >= 0 or c.touched_exit or c.stages_cleared > 0
                    )
                    solve_cnt = sum(1 for c in candidate_states if c.stages_cleared > 0)
                    threshold = int(self.pop_size * 0.50)

                    # Progressive Curriculum Scaling Trigger
                    if self.is_prog_mode and (exits_cnt >= threshold or solve_cnt >= max(2, int(threshold * 0.5))):
                        # 1. Save winning model checkpoint immediately
                        saved_path = self.persistence.save_brain(
                            self.active_profile_name,
                            winner_net,
                            self.factory.profile,
                            context=f"curriculum {self.prog_width}x{self.prog_height}"
                        )

                        # 2. Expand map dimensions
                        old_w, old_h = self.prog_width, self.prog_height
                        self.prog_width += 4
                        self.prog_height += 3

                        # Advance persistent curriculum stage.
                        self.prog_stage += 1

                        print("\n" + "=" * 80)
                        print(f"🎉 [CURRICULUM UPGRADE] {exits_cnt}/{self.pop_size} (>=50%) agents conquered {old_w}x{old_h}!")
                        print(f"💾 Checkpoint saved: {saved_path.name}")
                        print(f"🗺️  Expanding maze bounds to -> {self.prog_width}x{self.prog_height}")
                        print("=" * 80 + "\n")

                        # Rebuild map profile and generator for larger size
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

                    elif not self.is_prog_mode and gen_idx == gens_count - 1:
                        self.persistence.save_brain(
                            self.active_profile_name,
                            winner_net,
                            self.factory.profile,
                            context="training"
                        )

                    self.population.evolve_next_generation(
                        norm_scores
                    )

                    self.completed_generations += 1

                    if self.is_prog_mode:
                        self._save_evolution_checkpoint()

                    gen_idx += 1

                    if self.is_prog_mode:
                        self.prog_generation += 1
                        self._save_curriculum_state(self.prog_generation)

        except KeyboardInterrupt:
            print()
            print(
                "[Curriculum] Ctrl+C detected! "
                "Saving evolutionary checkpoint..."
            )

            if self.is_prog_mode:
                # The last checkpoint is the last fully evolved
                # generation. Do not invent a partially evolved one.
                self._save_evolution_checkpoint()

            elif "norm_scores" in locals():
                winner_idx = int(
                    np.argmax(norm_scores)
                )

                winner_net = (
                    self.population.networks[
                        winner_idx
                    ]
                )

                self.persistence.save_brain(
                    self.active_profile_name,
                    winner_net,
                    self.factory.profile,
                    context="final training"
                )

        self.cli_presenter.print_finish_footer()
        for w_str in AgentProfileRegistry.get_clamped_warning_strings():
            print(w_str)

        self.recorder.save_temporary_disk_archive()
        return self.recorder
