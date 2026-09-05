"""
Continuous multi-objective fitness evaluation module for maze exploration mastery.
"""

from typing import List
from entities.entity_state import AgentState


class FitnessEvaluator:
    """
    Computes fitness rewarding distance reduction, unique maze exploration (curiosity),
    time-to-solve efficiency, target hold progression, and clean driving.
    """

    @staticmethod
    def calculate_raw_score(
        state: AgentState,
        initial_bfs_dist: int = 0,
        max_steps: int = 1000,
        stage_bonus: float = 50.0,
        lost_hp_impact: float = 0.1
    ) -> float:
        # 1. Topological Distance Progress (toward Target 0)
        dist_reduced: float = 0.0
        if initial_bfs_dist > 0:
            d = float(initial_bfs_dist - state.best_step_dist)
            dist_reduced = d if d > 0.0 else 0.0

        # 2. Curiosity Exploration Reward (Unique tiles mapped)
        # Prevents agents from getting stuck pacing in spawn rooms
        unique_tiles_visited = len(getattr(state, "visited_tiles", set()))
        curiosity_bonus = float(unique_tiles_visited) * 0.40

        # 3. Speed / Time Efficiency Bonus
        time_bonus: float = 0.0
        if state.first_touch_step >= 0:
            den = max_steps if max_steps > 1 else 1
            remaining_fraction = float(max_steps - state.first_touch_step) / float(den)
            time_bonus = (remaining_fraction if remaining_fraction > 0.0 else 0.0) * 25.0

        # 4. Continuous Hold Zone Ramp (1..15 frames)
        hold_ramp: float = 0.0
        if state.stages_cleared == 0 and state.max_hold_frames > 0:
            capped_hold = state.max_hold_frames if state.max_hold_frames < 15 else 15
            hold_ratio = float(capped_hold) / 15.0
            hold_ramp = hold_ratio * 15.0

        # 5. Multi-Stage Discrete Clear Bonus (+50 pts per stage)
        clear_bonus: float = float(state.stages_cleared) * stage_bonus

        # 6. Health / Clean Driving Factor
        clamped_hp = 1.0 if state.health > 1.0 else (0.0 if state.health < 0.0 else state.health)
        hp_factor: float = (1.0 - lost_hp_impact) + (lost_hp_impact * clamped_hp)

        raw_total: float = (
            dist_reduced +
            curiosity_bonus +
            state.total_lifetime_progress +
            hold_ramp +
            time_bonus +
            clear_bonus
        ) * hp_factor

        return raw_total if raw_total > 0.1 else 0.1

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
