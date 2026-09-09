"""
Continuous multi-objective fitness evaluation module rewarding genuine autonomous maze exploration.
Fitness drives emergence of corridor navigation, corner turning, and exit seeking without oracles.
"""

from typing import List
from entities.entity_state import AgentState


class FitnessEvaluator:
    @staticmethod
    def calculate_raw_score(
        state: AgentState,
        initial_bfs_dist: int = 0,
        max_steps: int = 1000,
        stage_bonus: float = 2500.0,
        lost_hp_impact: float = 0.0
    ) -> float:
        # 1. Unique territory explored (evaluated externally by simulation, NOT known by agent brain)
        unique_visited = len(getattr(state, "visited_tiles", set()))
        exploration_reward = float(unique_visited) * 45.0

        # 2. Smooth locomotion (rewarding forward corridor traversal)
        dist_traveled = float(getattr(state, "distance_traveled", 0.0))
        locomotion_reward = min(250.0, dist_traveled * 3.5)

        # 3. Collision penalty (penalizing wall-ramming)
        collisions = float(getattr(state, "collision_count", 0))
        collision_penalty = collisions * 3.0

        # 4. Exit / Target Discovery Bonus (massive reward scaled by speed of finding exit)
        exit_bonus = 0.0
        if state.first_touch_step >= 0 or state.touched_exit:
            remaining = max(1, max_steps - (state.first_touch_step if state.first_touch_step >= 0 else max_steps // 2))
            exit_bonus = 15000.0 + (float(remaining) * 25.0)

        # 5. Stage Clear Bonus
        clear_bonus = float(state.stages_cleared) * stage_bonus

        total_score = (
            exploration_reward +
            locomotion_reward +
            exit_bonus +
            clear_bonus +
            state.total_lifetime_progress -
            collision_penalty
        )

        return max(1.0, total_score)

    @staticmethod
    def normalize_scores(raw_scores: List[float]) -> List[float]:
        if not raw_scores:
            return []

        min_s: float = min(raw_scores)
        max_s: float = max(raw_scores)
        span: float = max_s - min_s

        if span < 1e-6:
            return [1.0 for _ in raw_scores]

        return [(s - min_s) / span for s in raw_scores]
