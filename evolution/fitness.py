"""
Continuous min-max fitness evaluation module for candidate ranking.
"""

from typing import List

from entities.entity_state import AgentState


class FitnessEvaluator:
    """
    Computes candidate raw scores and normalized fitness ratios.
    """

    @staticmethod
    def calculate_raw_score(
        state: AgentState,
        initial_bfs_dist: int = 0,
        lost_hp_impact: float = 0.1,
        stage_bonus: float = 20.0
    ) -> float:
        """
        Computes unconstrained cumulative lifetime progress distance score.
        """
        progress_score: float = state.total_lifetime_progress
        if progress_score <= 0.0 and initial_bfs_dist > 0:
            dist_reduced: float = max(
                0.0, float(initial_bfs_dist - state.best_step_dist)
            )
            progress_score = dist_reduced

        clear_bonus: float = float(state.stages_cleared) * stage_bonus
        raw_total: float = progress_score + clear_bonus

        clamped_hp: float = max(0.0, min(1.0, state.health))
        hp_factor: float = (1.0 - lost_hp_impact) + (
            lost_hp_impact * clamped_hp
        )

        return raw_total * hp_factor

    @staticmethod
    def normalize_scores(raw_scores: List[float]) -> List[float]:
        """
        Normalizes a list of raw scores to [0.0, 1.0] ratios.
        """
        if not raw_scores:
            return []

        min_s: float = min(raw_scores)
        max_s: float = max(raw_scores)
        span: float = max_s - min_s

        if span < 1e-6:
            return [1.0 for _ in raw_scores]

        return [(s - min_s) / span for s in raw_scores]
