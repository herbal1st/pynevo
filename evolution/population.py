"""
Genetic algorithm population manager executing reproduction and mutation.
"""

from typing import List, Tuple, Optional

from entities.agent_factory import AgentFactory
from neural.network import NeuralNetwork
from evolution.operators.selection import TournamentSelection
from evolution.operators.crossover import UniformCrossover
from evolution.operators.mutation import GaussianMutation


class PopulationManager:
    """
    Manages neural network weight matrices across evolutionary generations.
    """

    def __init__(
        self,
        factory: Optional[AgentFactory] = None,
        pop_size: int = 25,
        mutation_rate: float = 0.25,
        mutation_scale: float = 0.125,
        elitism_ratio: float = 0.15
    ) -> None:
        """
        Initializes population parameters and candidate network pool.
        """
        self.pop_size: int = pop_size
        self.mutation_rate: float = mutation_rate
        self.mutation_scale: float = mutation_scale
        self.elitism_ratio: float = elitism_ratio
        self.factory: Optional[AgentFactory] = factory

        if factory is not None:
            self.networks: List[NeuralNetwork] = [
                factory.create_network() for _ in range(pop_size)
            ]
        else:
            self.networks = [NeuralNetwork() for _ in range(pop_size)]

    def seed_population_from_brain(
        self,
        seed_network: NeuralNetwork
    ) -> None:
        """
        Seeds population using cloned seed weights with exploration noise.
        """
        if not self.networks:
            return

        self.networks[0].copy_weights_from(seed_network)

        for idx in range(1, self.pop_size):
            self.networks[idx].copy_weights_from(seed_network)
            GaussianMutation.apply_mutation(
                self.networks[idx],
                mutation_rate=1.0,
                mutation_scale=self.mutation_scale
            )

    def evolve_next_generation(
        self,
        fitness_scores: List[float]
    ) -> None:
        """
        Performs elitism, tournament selection, crossover, and mutation.
        """
        indexed_scores: List[Tuple[int, float]] = list(
            enumerate(fitness_scores)
        )
        indexed_scores.sort(key=lambda item: item[1], reverse=True)

        num_elites: int = max(1, int(self.pop_size * self.elitism_ratio))
        elite_indices: List[int] = [
            idx for idx, _ in indexed_scores[:num_elites]
        ]

        new_networks: List[NeuralNetwork] = []

        for idx in elite_indices:
            elite_net = (
                self.factory.create_network() if self.factory is not None
                else NeuralNetwork()
            )
            elite_net.copy_weights_from(self.networks[idx])
            new_networks.append(elite_net)

        while len(new_networks) < self.pop_size:
            parent_a = TournamentSelection.select(
                self.networks, fitness_scores
            )
            parent_b = TournamentSelection.select(
                self.networks, fitness_scores
            )

            child_net = (
                self.factory.create_network() if self.factory is not None
                else NeuralNetwork()
            )

            if parent_a is parent_b:
                child_net.copy_weights_from(parent_a)
                GaussianMutation.apply_mutation(
                    child_net,
                    mutation_rate=1.0,
                    mutation_scale=self.mutation_scale
                )
            else:
                UniformCrossover.apply_crossover(
                    parent_a, parent_b, child_net
                )
                GaussianMutation.apply_mutation(
                    child_net, self.mutation_rate, self.mutation_scale
                )

            new_networks.append(child_net)

        self.networks = new_networks
