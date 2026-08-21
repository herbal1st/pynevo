"""
Pure spatial layout manager for multi-viewport grid coordinate math.
"""

from typing import Tuple, Optional


class GridLayoutManager:
    """
    Computes R x C viewport grid bounding rectangles and click targeting.
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        rows: int = 4,
        cols: int = 4
    ) -> None:
        """
        Initializes grid bounds and dimensions.
        """
        self.x, self.y, self.w, self.h = rect
        self.rows: int = max(1, min(8, rows))
        self.cols: int = max(1, min(8, cols))

    @property
    def total_slots(self) -> int:
        """
        Returns total number of grid viewport slots.
        """
        return self.rows * self.cols

    def get_sub_viewport_rect(
        self,
        slot_idx: int
    ) -> Tuple[int, int, int, int]:
        """
        Calculates pixel bounding rectangle for a sub-viewport slot index.
        """
        sub_w: int = self.w // self.cols
        sub_h: int = self.h // self.rows

        row: int = slot_idx // self.cols
        col: int = slot_idx % self.cols

        return (
            self.x + (col * sub_w),
            self.y + (row * sub_h),
            sub_w,
            sub_h
        )

    def get_slot_index_from_click(
        self,
        cx: int,
        cy: int
    ) -> Optional[int]:
        """
        Translates pixel click coordinates to sub-viewport grid slot index.
        """
        if not (
            self.x <= cx <= self.x + self.w and
            self.y <= cy <= self.y + self.h
        ):
            return None

        sub_w: int = self.w // self.cols
        sub_h: int = self.h // self.rows

        col: int = max(0, min(self.cols - 1, (cx - self.x) // sub_w))
        row: int = max(0, min(self.rows - 1, (cy - self.y) // sub_h))

        return int((row * self.cols) + col)

    def navigate_slot(
        self,
        curr_slot_idx: int,
        delta_row: int,
        delta_col: int
    ) -> int:
        """
        Calculates updated slot index from 2D directional navigation deltas.
        """
        curr_row: int = curr_slot_idx // self.cols
        curr_col: int = curr_slot_idx % self.cols

        target_row: int = max(0, min(self.rows - 1, curr_row + delta_row))
        target_col: int = max(0, min(self.cols - 1, curr_col + delta_col))

        return int((target_row * self.cols) + target_col)
