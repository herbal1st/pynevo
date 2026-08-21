"""
Stratified candidate pool mapper for multi-candidate viewport grid slots.
"""

import random
from typing import List, Optional


class CandidatePoolMapper:
    """
    Maps population candidates to viewport slots using stratified ranking.
    """

    @staticmethod
    def map_candidates_to_slots(
        pop_size: int,
        rows: int,
        cols: int,
        scores: Optional[List[float]] = None,
        seed: int = 0
    ) -> List[int]:
        """
        Maps candidate indices to R x C grid slots sorted by performance rank.
        """
        total_slots: int = rows * cols
        if pop_size <= 0:
            return []

        if scores and len(scores) == pop_size:
            indexed = list(enumerate(scores))
            indexed.sort(key=lambda item: item[1], reverse=True)
            ranked: List[int] = [idx for idx, _ in indexed]
        else:
            ranked = list(range(pop_size))

        if pop_size <= total_slots:
            return ranked

        if rows == 1:
            return ranked[:cols]

        mapped_slots: List[int] = [0] * total_slots

        # Top row: Best C ranked candidates (Winner ranked[0] at top-left)
        for c in range(cols):
            mapped_slots[c] = ranked[c]

        # Bottom row: Worst C ranked candidates
        bottom_start_slot: int = (rows - 1) * cols
        worst_start_rank: int = pop_size - cols
        for c in range(cols):
            mapped_slots[bottom_start_slot + c] = ranked[worst_start_rank + c]

        # Middle rows: Stratified selection across remaining middle ranks
        middle_slots_count: int = (rows - 2) * cols
        mid_rank_start: int = cols
        mid_rank_end: int = pop_size - cols
        mid_rank_count: int = mid_rank_end - mid_rank_start

        if middle_slots_count <= 0 or mid_rank_count <= 0:
            return mapped_slots

        rng = random.Random(seed)
        stratum_size: float = float(mid_rank_count) / float(
            middle_slots_count
        )

        for m in range(middle_slots_count):
            slot_idx: int = cols + m
            s_low: int = mid_rank_start + int(m * stratum_size)
            s_high: int = mid_rank_start + int((m + 1) * stratum_size) - 1
            s_high = max(s_low, min(s_high, mid_rank_end - 1))

            chosen_rank: int = rng.randint(s_low, s_high)
            mapped_slots[slot_idx] = ranked[chosen_rank]

        return mapped_slots
