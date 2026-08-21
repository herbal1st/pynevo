"""
Continuous min-max fitness evaluation module for candidate ranking.
"""

from typing import List

from entities.player_state import PlayerState


class FitnessEvaluator:
    """
    Computes candidate raw scores and normalized fitness ratios.
    """

    @staticmethod
    def calculate_raw_score(
        state: PlayerState,
        initial_bfs_dist: int,
        max_steps: int = 1000,
        move_speed: float = 0.15,
        dist_ratio: float = 0.6,
        lost_hp_impact: float = 0.1
    ) -> float:
        """
        Computes path progress plus time bonus weighted by health.
        """
        clamped_r: float = max(0.0, min(1.0, dist_ratio))
        w_dist: float = 1000.0 * clamped_r
        w_time: float = 1000.0 * (1.0 - clamped_r)

        total_path_len: float = float(initial_bfs_dist)

        dist_reduced: float = max(
            0.0, float(initial_bfs_dist - state.best_step_dist)
        )
        prog_ratio: float = max(
            0.0, min(1.0, dist_reduced / max(1e-6, total_path_len))
        )
        score_dist: float = w_dist * prog_ratio

        if state.has_reached_exit:
            min_frames: float = total_path_len / max(1e-6, move_speed)
            max_saved: float = max(1.0, float(max_steps) - min_frames)
            actual_saved: float = max(
                0.0, float(max_steps - state.frames_survived)
            )
            time_ratio: float = max(
                0.0, min(1.0, actual_saved / max_saved)
            )
            score_time: float = w_time * time_ratio
        else:
            score_time = 0.0

        raw_total: float = score_dist + score_time

        clamped_hp: float = max(0.0, min(1.0, state.health))
        hp_factor: float = (1.0 - lost_hp_impact) + (
            lost_hp_impact * clamped_hp
        )

        return raw_total * hp_factor

    @staticmethod
    def calculate_theoretical_max_score(
        initial_bfs_dist: int,
        max_steps: int = 1000,
        move_speed: float = 0.15,
        dist_ratio: float = 0.6,
        num_turns: int = 0,
        corner_savings_per_turn: float = 0.586
    ) -> float:
        """
        Returns theoretical max score (normalized peak 1000.0).
        """
        return 1000.0

    @staticmethod
    def calculate_scaled_score(
        raw_score: float,
        theoretical_max: float
    ) -> float:
        """
        Normalizes raw score to [0.0, 1000.0] range.
        """
        if theoretical_max < 1e-6:
            return 0.0

        ratio: float = max(0.0, raw_score / theoretical_max)
        return min(1000.0, ratio * 1000.0)

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
