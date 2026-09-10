"""
Genetic algorithm population manager with recombination crossover, elitism, and checkpointing.
Zero-allocation double-buffering with multi-tier mutation niches to break out of maze plateaus.
"""

from typing import List, Optional, Dict
from pathlib import Path
import pickle
import random
import re
import numpy as np

from entities.agent_factory import AgentFactory
from neural.network import NeuralNetwork
from evolution.operators.selection import TournamentSelection


CHECKPOINT_VERSION = 2
CHECKPOINT_DIR = Path("saved_brains")


class PopulationManager:
    def __init__(
        self,
        factory: Optional[AgentFactory] = None,
        pop_size: int = 25,
        mutation_rate: float = 0.15,
        mutation_scale: float = 0.04,
        elitism_ratio: float = 0.20
    ) -> None:
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elitism_ratio = elitism_ratio
        self.factory = factory

        if factory is not None:
            self.networks = [factory.create_network() for _ in range(pop_size)]
            self._next_networks = [factory.create_network() for _ in range(pop_size)]
        else:
            self.networks = [NeuralNetwork() for _ in range(pop_size)]
            self._next_networks = [NeuralNetwork() for _ in range(pop_size)]

    def seed_population_from_brain(
        self,
        seed_network: NeuralNetwork
    ) -> None:
        """
        Seeds population around champion network to smoothly carry forward policies across curriculum stages.
        """
        if not self.networks:
            return

        self.networks[0].copy_weights_from(seed_network)
        self.networks[1].copy_weights_from(seed_network)

        pop_len = self.pop_size
        for idx in range(2, pop_len):
            self.networks[idx].copy_weights_from(seed_network)
            if idx < pop_len // 3:
                noise_scale = 0.020
            elif idx < (2 * pop_len) // 3:
                noise_scale = 0.050
            else:
                noise_scale = 0.100

            noise = np.random.normal(
                0.0, noise_scale, size=self.networks[idx].param_buffer.shape
            ).astype(np.float32)
            self.networks[idx].param_buffer += noise

    def export_population(self) -> np.ndarray:
        if not self.networks:
            return np.empty((0, 0), dtype=np.float16)

        return np.ascontiguousarray(
            np.stack([net.export_flat_weights() for net in self.networks], axis=0),
            dtype=np.float16
        )

    def import_population(self, weights: np.ndarray) -> None:
        weights = np.asarray(weights)
        if weights.ndim != 2 or weights.shape[0] != self.pop_size:
            raise ValueError("Population weights shape mismatch")

        expected = self.networks[0].param_count
        if weights.shape[1] != expected:
            raise ValueError("Parameter count mismatch")

        for idx, network in enumerate(self.networks):
            network.import_flat_weights(weights[idx])

    @staticmethod
    def checkpoint_path(profile_name: str, curriculum_name: str = "PROG") -> Path:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        safe_p = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile_name)
        safe_c = re.sub(r"[^A-Za-z0-9_.-]+", "_", curriculum_name)
        return CHECKPOINT_DIR / f"evolution_{safe_p}_{safe_c}.npz"

    def save_checkpoint(
        self,
        profile_name: str,
        stage: int,
        width: int,
        height: int,
        completed_generations: int,
        curriculum_name: str = "PROG"
    ) -> Path:
        target = self.checkpoint_path(profile_name, curriculum_name)
        temp = target.with_suffix(".npz.tmp")

        payload = {
            "checkpoint_version": np.array(CHECKPOINT_VERSION, dtype=np.int32),
            "stage": np.array(stage, dtype=np.int32),
            "width": np.array(width, dtype=np.int32),
            "height": np.array(height, dtype=np.int32),
            "completed_generations": np.array(completed_generations, dtype=np.int64),
            "population_size": np.array(self.pop_size, dtype=np.int32),
            "param_count": np.array(self.networks[0].param_count, dtype=np.int64),
            "population_weights": self.export_population(),
            "python_random_state": np.frombuffer(pickle.dumps(random.getstate()), dtype=np.uint8),
            "numpy_random_state": np.frombuffer(pickle.dumps(np.random.get_state()), dtype=np.uint8),
        }
        np.savez_compressed(temp, **payload)
        actual = temp if temp.exists() else Path(str(temp) + ".npz")
        actual.replace(target)
        return target

    def load_checkpoint(
        self,
        profile_name: str,
        curriculum_name: str = "PROG"
    ) -> Optional[Dict[str, int]]:
        target = self.checkpoint_path(profile_name, curriculum_name)
        if not target.exists():
            return None

        try:
            archive = np.load(target, allow_pickle=False)
            if int(archive["checkpoint_version"]) != CHECKPOINT_VERSION:
                archive.close()
                return None

            if int(archive["population_size"]) != self.pop_size or int(archive["param_count"]) != self.networks[0].param_count:
                archive.close()
                return None

            self.import_population(archive["population_weights"])
            random.setstate(pickle.loads(archive["python_random_state"].tobytes()))
            np.random.set_state(pickle.loads(archive["numpy_random_state"].tobytes()))

            meta = {
                "stage": int(archive["stage"]),
                "width": int(archive["width"]),
                "height": int(archive["height"]),
                "completed_generations": int(archive["completed_generations"]),
            }
            archive.close()
            return meta
        except Exception:
            return None

    def evolve_next_generation(self, fitness_scores: List[float]) -> None:
        indexed = list(enumerate(fitness_scores))
        indexed.sort(key=lambda item: item[1], reverse=True)

        num_elites = max(2, int(self.pop_size * self.elitism_ratio))
        elite_indices = [idx for idx, _ in indexed[:num_elites]]

        # 1. Exact Elites (zero mutation preservation)
        for slot, idx in enumerate(elite_indices):
            np.copyto(self._next_networks[slot].param_buffer, self.networks[idx].param_buffer)

        # 2. Multi-tier mutation niches to continually test alternate corridor branches
        pop_len = self.pop_size
        scores_arr = np.asarray(fitness_scores, dtype=np.float64)

        for slot in range(num_elites, pop_len):
            target_buf = self._next_networks[slot].param_buffer

            cand_a = np.random.choice(pop_len, 4, replace=False)
            p1_idx = cand_a[np.argmax(scores_arr[cand_a])]
            cand_b = np.random.choice(pop_len, 4, replace=False)
            p2_idx = cand_b[np.argmax(scores_arr[cand_b])]

            p1_buf = self.networks[p1_idx].param_buffer
            p2_buf = self.networks[p2_idx].param_buffer

            if random.random() < 0.50:
                mask = np.random.random(target_buf.shape) < 0.5
                np.copyto(target_buf, np.where(mask, p1_buf, p2_buf))
            else:
                np.copyto(target_buf, p1_buf)

            # 4 distinct exploratory niches
            if slot < pop_len * 0.40:
                scale = 0.020
                gene_rate = 0.07
            elif slot < pop_len * 0.70:
                scale = 0.050
                gene_rate = 0.15
            elif slot < pop_len * 0.90:
                scale = 0.100
                gene_rate = 0.25
            else:
                # Top 10% frontier scouts: high mutation to escape dead-end valleys
                scale = 0.160
                gene_rate = 0.40

            mut_mask = np.random.random(target_buf.shape) < gene_rate
            noise = np.random.normal(0.0, scale, size=target_buf.shape).astype(np.float32)
            target_buf += mut_mask * noise

        self.networks, self._next_networks = self._next_networks, self.networks
