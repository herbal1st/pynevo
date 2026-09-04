"""
Formats and presents training generation metrics in a console progress table.
"""

import time
from typing import List
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
            f"{'FIRST':>5s} | {'STAGE':>5s} | {'TOUCH':>5s} | "
            f"{'SOLVE':>5s} | {'EXITS':>5s} | {'PROS':>5s} | {'TIME':>6s}"
        )

    def print_start_banner(
        self,
        agent_profile: str = config.ACTIVE_AGENT_PROFILE,
        training_profile: str = config.ACTIVE_TRAINING_PROFILE,
        map_profile: str = config.ACTIVE_MAP_PROFILE,
        hold_frames: int = 15
    ) -> None:
        """
        Outputs training session parameters and table column header.
        """
        print("\n=== NEUROEVOLUTION SIMULATION RUN ===")
        print(
            f"Population: {self.pop_size} | Max Steps: {self.max_steps} | "
            f"Target Hold: {hold_frames}f"
        )
        print(
            f"Agent: {agent_profile} | Training: {training_profile} | "
            f"Map: {map_profile}\n"
        )
        print(self.header_str)
        print("-" * len(self.header_str))

    def print_generation_row(
        self,
        gen_idx: int,
        raw_scores: List[float],
        norm_scores: List[float],
        candidate_states: List[AgentState],
        elapsed_sec: float = 0.0
    ) -> None:
        """
        Formats and prints a single generation progress metrics row.
        """
        top_int: int = int(round(max(raw_scores)))
        avg_raw: float = (
            sum(raw_scores) / float(len(raw_scores))
            if raw_scores else 0.0
        )
        winner_idx: int = int(np.argmax(norm_scores))
        winner_state: AgentState = candidate_states[winner_idx]

        winner_stage: int = winner_state.stages_cleared

        touch_str: str = (
            str(winner_state.first_touch_step)
            if winner_state.first_touch_step >= 0 else "-"
        )
        solve_str: str = (
            str(winner_state.first_hold_clear_step)
            if winner_state.first_hold_clear_step >= 0 else "-"
        )

        exits_cnt: int = sum(
            1 for c in candidate_states
            if c.first_touch_step >= 0 or c.touched_exit
        )
        pros_cnt: int = sum(
            1 for c in candidate_states
            if c.stages_cleared > 0 or c.first_hold_clear_step >= 0
        )

        exits_str: str = f"{exits_cnt}" if exits_cnt > 0 else "-"
        pros_str: str = f"{pros_cnt}" if pros_cnt > 0 else "-"
        winner_str: str = f"# {winner_idx}"
        time_str: str = f"{elapsed_sec:5.2f}s"

        row_str: str = (
            f"{gen_idx + 1:>5d} | {top_int:>5d} | {avg_raw:>6.1f} | "
            f"{winner_str:>5s} | {winner_stage:>5d} | {touch_str:>5s} | "
            f"{solve_str:>5s} | {exits_str:>5s} | {pros_str:>5s} | "
            f"{time_str:>6s}"
        )
        print(row_str)

    def print_finish_footer(self) -> None:
        """
        Outputs completion footer message.
        """
        print("-" * len(self.header_str))
        print("Training complete! Booting interactive visualizer GUI...\n")
