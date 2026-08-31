"""
Formats and presents training generation metrics in a console progress table.
"""

import time
from typing import List, Tuple
import numpy as np

import config
from entities.entity_state import AgentState


class CLIPresenter:
    """
    Renders real-time evolutionary progress tables to standard output.
    """

    def __init__(self, pop_size: int, max_steps: int) -> None:
        """
        Initializes column formatting bounds and header definitions.
        """
        self.pop_size: int = pop_size
        self.max_steps: int = max_steps
        self.header_str: str = (
            f"{'GEN':>5s} | {'TOP':>5s} | {'AVG':>6s} | "
            f"{'FIRST':>5s} | {'DONE':>5s} | {'DIST':>5s} | "
            f"{'FRAME':>5s} | {'EXITS':>5s} | {'TIME':>6s}"
        )

    def print_start_banner(
        self,
        profile_name: str = config.ACTIVE_AGENT_PROFILE
    ) -> None:
        """
        Outputs training session parameters and table column header.
        """
        print("\n=== NEUROEVOLUTION SIMULATION RUN ===")
        print(
            f"Population: {self.pop_size} | Max Steps: {self.max_steps} | "
            f"Profile: {profile_name} | Target Score: 1000\n"
        )
        print(self.header_str)
        print("-" * len(self.header_str))

    def print_generation_row(
        self,
        gen_idx: int,
        scaled_scores: List[float],
        initial_bfs_dist: int,
        norm_scores: List[float],
        candidate_states: List[AgentState],
        elapsed_sec: float = 0.0,
        done_pct: int = 0
    ) -> None:
        """
        Formats and prints a single generation progress metrics row.
        """
        top_int: int = int(round(max(scaled_scores)))
        avg_scaled: float = (
            sum(scaled_scores) / float(len(scaled_scores))
        )
        winner_idx: int = int(np.argmax(norm_scores))

        solvers: List[Tuple[int, int]] = [
            (c_idx, c_state.frames_survived)
            for c_idx, c_state in enumerate(candidate_states)
            if c_state.has_reached_exit
        ]
        solve_count: int = len(solvers)
        exits_str: str = f"{solve_count}" if solve_count > 0 else "-"

        if solve_count > 0:
            fastest_step: int = min(
                step_cnt for _, step_cnt in solvers
            )
            frame_str: str = str(fastest_step)
        else:
            frame_str = "-"

        winner_str: str = f"# {winner_idx}"
        done_str: str = f"{done_pct}%"
        time_str: str = f"{elapsed_sec:5.2f}s"

        row_str: str = (
            f"{gen_idx + 1:>5d} | {top_int:>5d} | {avg_scaled:>6.1f} | "
            f"{winner_str:>5s} | {done_str:>5s} | {initial_bfs_dist:>5d} | "
            f"{frame_str:>5s} | {exits_str:>5s} | {time_str:>6s}"
        )
        print(row_str)

    def print_finish_footer(self) -> None:
        """
        Outputs completion footer message.
        """
        print("-" * len(self.header_str))
        print("Training complete! Booting interactive visualizer GUI...\n")
