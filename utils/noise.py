"""
Deterministic 2D Simplex & classical Perlin noise engines.
"""

import math
import random
from typing import List, Final
import numpy as np
from numpy.typing import NDArray

# Skewing and unskewing factors for 2D Simplex noise
F2: Final[float] = 0.5 * (math.sqrt(3.0) - 1.0)
G2: Final[float] = (3.0 - math.sqrt(3.0)) / 6.0

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
        self.seed: int = seed
        rng: random.Random = random.Random(seed)
        perm: List[int] = list(range(256))
        rng.shuffle(perm)

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


class PerlinNoise:
    """
    Deterministic classical 2D Perlin noise generator.
    """

    def __init__(self, seed: int = 420) -> None:
        """
        Initializes permutation table deterministically from world_seed.
        """
        self.seed: int = seed
        rng: random.Random = random.Random(seed)
        perm: List[int] = list(range(256))
        rng.shuffle(perm)

        self.p: List[int] = perm + perm
        self.perm_grad_idx: List[int] = [i % 8 for i in self.p]

    def noise_2d(self, x: float, y: float) -> float:
        """
        Samples 2D Perlin field at (x, y). Returns value in [0.0, 1.0].
        """
        cell_x: int = math.floor(x) & 255
        cell_y: int = math.floor(y) & 255

        xf: float = x - math.floor(x)
        yf: float = y - math.floor(y)

        u: float = self._fade(xf)
        v: float = self._fade(yf)

        aa: int = self.p[self.p[cell_x] + cell_y]
        ab: int = self.p[self.p[cell_x] + cell_y + 1]
        ba: int = self.p[self.p[cell_x + 1] + cell_y]
        bb: int = self.p[self.p[cell_x + 1] + cell_y + 1]

        g00: float = self._grad(self.perm_grad_idx[aa], xf, yf)
        g10: float = self._grad(self.perm_grad_idx[ba], xf - 1.0, yf)
        g01: float = self._grad(self.perm_grad_idx[ab], xf, yf - 1.0)
        g11: float = self._grad(
            self.perm_grad_idx[bb], xf - 1.0, yf - 1.0
        )

        x1: float = self._lerp(g00, g10, u)
        x2: float = self._lerp(g01, g11, u)
        raw_val: float = self._lerp(x1, x2, v)

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
        Samples 2D multi-octave Perlin noise across coordinate grids.
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

    @staticmethod
    def _fade(t: float) -> float:
        """
        Applies Quintic smoothstep curve 6t^5 - 15t^4 + 10t^3.
        """
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """
        Linear interpolation between a and b.
        """
        return a + t * (b - a)

    @staticmethod
    def _grad(hash_idx: int, x: float, y: float) -> float:
        """
        Calculates dot product between chosen gradient and distance vector.
        """
        g = GRAD2[hash_idx]
        return (g[0] * x) + (g[1] * y)
