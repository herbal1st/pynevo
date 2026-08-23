"""
Deterministic 2D Simplex & Perlin noise engine seeded by world_seed.
"""

import math
import random
from typing import List, Final
import numpy as np
from numpy.typing import NDArray

# Skewing and unskewing factors for 2D Simplex noise
F2: Final[float] = 0.5 * (math.sqrt(3.0) - 1.0)  # ratio
G2: Final[float] = (3.0 - math.sqrt(3.0)) / 6.0  # ratio

GRAD2: Final[List[List[int]]] = [
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1]
]


class SimplexNoise:
    """
    Deterministic 2D Simplex noise generator using a seeded permutation.
    """

    def __init__(self, seed: int = 420) -> None:
        """
        Initializes permutation table deterministically from world_seed.
        """
        self.seed: int = seed  # seed
        rng: random.Random = random.Random(seed)
        perm: List[int] = list(range(256))
        rng.shuffle(perm)

        # Double the permutation table to eliminate modulo wraps
        self.p: List[int] = perm + perm
        self.perm_grad_idx: List[int] = [i % 8 for i in self.p]

    def noise_2d(self, x: float, y: float) -> float:
        """
        Samples 2D Simplex field at (x, y). Returns value in [0.0, 1.0].
        """
        s_val: float = (x + y) * F2
        i: int = math.floor(x + s_val)
        j: int = math.floor(y + s_val)

        t_val: float = (i + j) * G2
        x0: float = x - (i - t_val)
        y0: float = y - (j - t_val)

        i1, j1 = (1, 0) if x0 > y0 else (0, 1)

        x1: float = x0 - i1 + G2
        y1: float = y0 - j1 + G2
        x2: float = x0 - 1.0 + (2.0 * G2)
        y2: float = y0 - 1.0 + (2.0 * G2)

        ii: int = i & 255
        jj: int = j & 255

        n0 = self._get_corner_contrib(x0, y0, self.p[ii + self.p[jj]])
        n1 = self._get_corner_contrib(
            x1, y1, self.p[ii + i1 + self.p[jj + j1]]
        )
        n2 = self._get_corner_contrib(
            x2, y2, self.p[ii + 1 + self.p[jj + 1]]
        )

        raw_val: float = 70.0 * (n0 + n1 + n2)
        norm_val: float = (raw_val + 1.0) * 0.5
        return max(0.0, min(1.0, norm_val))

    def sample_grid(
        self,
        x_grid: NDArray[np.float32],
        y_grid: NDArray[np.float32],
        scale: float = 36.0,
        octaves: int = 1,
        octaves_decay: float = 0.5
    ) -> NDArray[np.float32]:
        """
        Samples 2D multi-octave noise across coordinate grids in NumPy.
        """
        height, width = x_grid.shape
        result: NDArray[np.float32] = np.zeros(
            (height, width), dtype=np.float32
        )

        total_amplitude: float = 0.0
        freq: float = 1.0 / max(0.1, scale)
        amp: float = 1.0

        for _ in range(max(1, octaves)):
            for r in range(height):
                for c in range(width):
                    nx: float = float(x_grid[r, c]) * freq
                    ny: float = float(y_grid[r, c]) * freq
                    result[r, c] += self.noise_2d(nx, ny) * amp

            total_amplitude += amp
            freq *= 2.0
            amp *= max(0.0, min(1.0, octaves_decay))

        if total_amplitude > 0.0:
            result /= total_amplitude

        return np.clip(result, 0.0, 1.0)

    def _get_corner_contrib(
        self, x: float, y: float, hash_val: int
    ) -> float:
        """
        Calculates the noise contribution from a single simplex corner.
        """
        t_val: float = 0.5 - (x * x) - (y * y)
        if t_val < 0.0:
            return 0.0

        grad: List[int] = GRAD2[self.perm_grad_idx[hash_val]]
        t_sq: float = t_val * t_val
        return (t_sq * t_sq) * ((grad[0] * x) + (grad[1] * y))
