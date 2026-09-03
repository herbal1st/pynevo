"""
Formats and presents real-time neural evolutionary diagnostics in a console progress table.
Tracks Path Freshness (Zero-Revisits) and Pacing (Zero-Dwell).
"""

from typing import List
import config


class CLIPresenter:
    def __init__(self, pop_size: int, max_steps: int) -> None:
        self.pop_size: int = pop_size
        self.max_steps: int = max_steps
        self.header_str: str = (
            f"{'GEN':>5s} | {'TOP':>5s} | {'AVG':>5s} | "
            f"{'DONE':>5s} | {'EXPL':>5s} | {'EFFC':>5s} | {'PACE':>5s} | {'CNTR':>5s} | "
            f"{'EXITS':>5s} | {'STRK':>4s} | {'TIME':>6s}"
        )

    def print_start_banner(
        self,
        profile_name: str = config.ACTIVE_AGENT_PROFILE
    ) -> None:
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
        norm_scores: List[float],
        done_pct: int,
        expl_pct: int,
        effc_pct: int,
        pace_pct: int,
        cntr_pct: int,
        exit_count: int,
        streak_str: str,
        elapsed_sec: float = 0.0
    ) -> None:
        top_int: int = int(round(max(scaled_scores)))
        avg_scaled: float = sum(scaled_scores) / float(len(scaled_scores))

        exits_str: str = f"{exit_count}" if exit_count > 0 else "-"
        done_str: str = f"{done_pct}%"
        expl_str: str = f"{expl_pct}%"
        effc_str: str = f"{effc_pct}%"
        pace_str: str = f"{pace_pct}%"
        cntr_str: str = f"{cntr_pct}%"
        time_str: str = f"{elapsed_sec:5.2f}s"

        row_str: str = (
            f"{gen_idx + 1:>5d} | {top_int:>5d} | {avg_scaled:>5.1f} | "
            f"{done_str:>5s} | {expl_str:>5s} | {effc_str:>5s} | {pace_str:>5s} | {cntr_str:>5s} | "
            f"{exits_str:>5s} | {streak_str:>4s} | {time_str:>6s}"
        )
        print(row_str)

    def print_finish_footer(self) -> None:
        print("-" * len(self.header_str))
        print("Training complete! Booting interactive visualizer GUI...\n")
