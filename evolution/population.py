"""
Genetic algorithm population manager with full population checkpointing.
"""

from typing import List, Tuple, Optional, Dict
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
        mutation_rate: float = 0.25,
        mutation_scale: float = 0.08,
        elitism_ratio: float = 0.15
    ) -> None:

        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elitism_ratio = elitism_ratio
        self.factory = factory

        if factory is not None:
            self.networks = [
                factory.create_network()
                for _ in range(pop_size)
            ]
        else:
            self.networks = [
                NeuralNetwork()
                for _ in range(pop_size)
            ]

    def seed_population_from_brain(
        self,
        seed_network: NeuralNetwork
    ) -> None:
        """
        Legacy champion-only restoration.
        Used only when no full population checkpoint exists.
        """

        if not self.networks:
            return

        self.networks[0].copy_weights_from(seed_network)

        for idx in range(1, self.pop_size):
            self.networks[idx].copy_weights_from(seed_network)

            noise = np.random.normal(
                0.0,
                0.08,
                size=self.networks[idx].param_buffer.shape
            ).astype(np.float32)

            self.networks[idx].param_buffer += noise

    # ========================================================
    # Full population serialization
    # ========================================================

    def export_population(self) -> np.ndarray:

        if not self.networks:
            return np.empty(
                (0, 0),
                dtype=np.float16
            )

        return np.ascontiguousarray(
            np.stack(
                [
                    network.export_flat_weights()
                    for network in self.networks
                ],
                axis=0
            ),
            dtype=np.float16
        )

    def import_population(
        self,
        weights: np.ndarray
    ) -> None:

        weights = np.asarray(weights)

        if weights.ndim != 2:
            raise ValueError(
                f"Invalid population checkpoint shape: "
                f"{weights.shape}"
            )

        if weights.shape[0] != self.pop_size:
            raise ValueError(
                f"Population size mismatch: "
                f"checkpoint={weights.shape[0]}, "
                f"current={self.pop_size}"
            )

        expected = self.networks[0].param_count

        if weights.shape[1] != expected:
            raise ValueError(
                f"Parameter count mismatch: "
                f"checkpoint={weights.shape[1]}, "
                f"current={expected}"
            )

        for idx, network in enumerate(self.networks):
            network.import_flat_weights(weights[idx])

    @staticmethod
    def checkpoint_path(
        profile_name: str,
        curriculum_name: str = "PROG"
    ) -> Path:

        CHECKPOINT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        safe_profile = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            profile_name
        )

        safe_curriculum = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            curriculum_name
        )

        return CHECKPOINT_DIR / (
            f"evolution_{safe_profile}_{safe_curriculum}.npz"
        )

    def save_checkpoint(
        self,
        profile_name: str,
        stage: int,
        width: int,
        height: int,
        completed_generations: int,
        curriculum_name: str = "PROG"
    ) -> Path:

        target = self.checkpoint_path(
            profile_name,
            curriculum_name
        )

        temporary = target.with_suffix(".npz.tmp")

        payload = {
            "checkpoint_version": np.array(
                CHECKPOINT_VERSION,
                dtype=np.int32
            ),
            "stage": np.array(
                stage,
                dtype=np.int32
            ),
            "width": np.array(
                width,
                dtype=np.int32
            ),
            "height": np.array(
                height,
                dtype=np.int32
            ),
            "completed_generations": np.array(
                completed_generations,
                dtype=np.int64
            ),
            "population_size": np.array(
                self.pop_size,
                dtype=np.int32
            ),
            "param_count": np.array(
                self.networks[0].param_count,
                dtype=np.int64
            ),
            "population_weights": self.export_population(),
            "python_random_state": np.frombuffer(
                pickle.dumps(random.getstate()),
                dtype=np.uint8
            ),
            "numpy_random_state": np.frombuffer(
                pickle.dumps(np.random.get_state()),
                dtype=np.uint8
            ),
        }

        np.savez_compressed(
            temporary,
            **payload
        )

        actual_tmp = temporary

        if not actual_tmp.exists():
            appended = Path(
                str(temporary) + ".npz"
            )

            if appended.exists():
                actual_tmp = appended

        actual_tmp.replace(target)

        return target

    def load_checkpoint(
        self,
        profile_name: str,
        curriculum_name: str = "PROG"
    ) -> Optional[Dict[str, int]]:

        target = self.checkpoint_path(
            profile_name,
            curriculum_name
        )

        if not target.exists():
            return None

        try:
            archive = np.load(
                target,
                allow_pickle=False
            )

            version = int(
                archive["checkpoint_version"]
            )

            if version != CHECKPOINT_VERSION:
                print(
                    f"[Persistence] Ignoring incompatible "
                    f"checkpoint {target.name}: "
                    f"version={version}"
                )
                archive.close()
                return None

            checkpoint_pop = int(
                archive["population_size"]
            )

            checkpoint_params = int(
                archive["param_count"]
            )

            current_params = (
                self.networks[0].param_count
            )

            if checkpoint_pop != self.pop_size:
                print(
                    f"[Persistence] Population mismatch: "
                    f"checkpoint={checkpoint_pop}, "
                    f"current={self.pop_size}"
                )
                archive.close()
                return None

            if checkpoint_params != current_params:
                print(
                    f"[Persistence] Network mismatch: "
                    f"checkpoint={checkpoint_params}, "
                    f"current={current_params}"
                )
                archive.close()
                return None

            self.import_population(
                archive["population_weights"]
            )

            random.setstate(
                pickle.loads(
                    archive[
                        "python_random_state"
                    ].tobytes()
                )
            )

            np.random.set_state(
                pickle.loads(
                    archive[
                        "numpy_random_state"
                    ].tobytes()
                )
            )

            metadata = {
                "stage": int(
                    archive["stage"]
                ),
                "width": int(
                    archive["width"]
                ),
                "height": int(
                    archive["height"]
                ),
                "completed_generations": int(
                    archive[
                        "completed_generations"
                    ]
                ),
            }

            archive.close()

            print(
                "[Persistence] FULL EVOLUTION "
                "CHECKPOINT RESTORED: "
                f"stage={metadata['stage']} "
                f"map={metadata['width']}x"
                f"{metadata['height']} "
                f"generations="
                f"{metadata['completed_generations']} "
                f"population={self.pop_size}"
            )

            return metadata

        except Exception as exc:
            print(
                f"[Persistence] Failed to load "
                f"{target.name}: {exc}"
            )
            return None

    # ========================================================
    # Evolution
    # ========================================================

    def evolve_next_generation(
        self,
        fitness_scores: List[float]
    ) -> None:

        indexed_scores = list(
            enumerate(fitness_scores)
        )

        indexed_scores.sort(
            key=lambda item: item[1],
            reverse=True
        )

        num_elites = max(
            2,
            int(
                self.pop_size *
                self.elitism_ratio
            )
        )

        elite_indices = [
            idx
            for idx, _ in
            indexed_scores[:num_elites]
        ]

        new_networks = []

        # Exact elites.
        for idx in elite_indices:

            child = (
                self.factory.create_network()
                if self.factory is not None
                else NeuralNetwork()
            )

            child.copy_weights_from(
                self.networks[idx]
            )

            new_networks.append(child)

        # Fill remaining population.
        while len(new_networks) < self.pop_size:

            parent = TournamentSelection.select(
                self.networks,
                fitness_scores,
                k=4
            )

            child = (
                self.factory.create_network()
                if self.factory is not None
                else NeuralNetwork()
            )

            child.copy_weights_from(parent)

            slot = len(new_networks)

            if slot < self.pop_size * 0.50:
                scale = 0.03
                gene_rate = 0.15

            elif slot < self.pop_size * 0.85:
                scale = 0.08
                gene_rate = 0.25

            else:
                scale = 0.16
                gene_rate = 0.40

            mask = (
                np.random.random(
                    child.param_buffer.shape
                ) < gene_rate
            )

            noise = np.random.normal(
                0.0,
                scale,
                size=child.param_buffer.shape
            ).astype(np.float32)

            child.param_buffer += (
                mask * noise
            )

            new_networks.append(child)

        self.networks = new_networks
