"""
Genetic algorithm population manager with recombination crossover, elitism, and checkpointing.
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
        else:
            self.networks = [NeuralNetwork() for _ in range(pop_size)]

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
        for idx in range(1, self.pop_size):
            self.networks[idx].copy_weights_from(seed_network)
            noise_scale = 0.015 if idx < self.pop_size // 2 else 0.040
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

        new_networks = []

        # 1. Exact Elites (zero mutation preservation)
        for idx in elite_indices:
            child = (
                self.factory.create_network()
                if self.factory is not None
                else NeuralNetwork()
            )
            child.copy_weights_from(self.networks[idx])
            new_networks.append(child)

        # 2. Uniform Crossover & Adaptive Mutation
        while len(new_networks) < self.pop_size:
            p1 = TournamentSelection.select(self.networks, fitness_scores, k=4)
            p2 = TournamentSelection.select(self.networks, fitness_scores, k=4)

            child = (
                self.factory.create_network()
                if self.factory is not None
                else NeuralNetwork()
            )

            if random.random() < 0.50:
                mask = np.random.random(child.param_buffer.shape) < 0.5
                np.copyto(
                    child.param_buffer,
                    np.where(mask, p1.param_buffer, p2.param_buffer)
                )
            else:
                child.copy_weights_from(p1)

            slot = len(new_networks)
            if slot < self.pop_size * 0.45:
                scale = 0.020
                gene_rate = 0.08
            elif slot < self.pop_size * 0.80:
                scale = 0.045
                gene_rate = 0.15
            else:
                scale = 0.090
                gene_rate = 0.25

            mut_mask = np.random.random(child.param_buffer.shape) < gene_rate
            noise = np.random.normal(0.0, scale, size=child.param_buffer.shape).astype(np.float32)
            child.param_buffer += mut_mask * noise

            new_networks.append(child)

        self.networks = new_networks
