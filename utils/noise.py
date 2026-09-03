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

GRAD2_ARR: Final[NDArray[np.float32]] = np.array(
    GRAD2, dtype=np.float32
)


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
        self.p_arr: NDArray[np.int32] = np.array(
            self.p, dtype=np.int32
        )
        self.perm_grad_idx: List[int] = [i % 8 for i in self.p]
        self.perm_grad_arr: NDArray[np.int32] = np.array(
            self.perm_grad_idx, dtype=np.int32
        )

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

        p = self.p
        gi0 = self.perm_grad_idx[p[ii + p[jj]]]
        gi1 = self.perm_grad_idx[p[ii + i1 + p[jj + j1]]]
        gi2 = self.perm_grad_idx[p[ii + 1 + p[jj + 1]]]

        g0 = GRAD2[gi0]
        g1 = GRAD2[gi1]
        g2 = GRAD2[gi2]

        t0 = max(0.0, 0.5 - (x0 * x0) - (y0 * y0))
        t1 = max(0.0, 0.5 - (x1 * x1) - (y1 * y1))
        t2 = max(0.0, 0.5 - (x2 * x2) - (y2 * y2))

        n0 = (t0 ** 4) * ((g0[0] * x0) + (g0[1] * y0))
        n1 = (t1 ** 4) * ((g1[0] * x1) + (g1[1] * y1))
        n2 = (t2 ** 4) * ((g2[0] * x2) + (g2[1] * y2))

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
            noise_layer = self._vectorized_simplex_2d(
                x_grid * freq, y_grid * freq
            )
            result += (noise_layer * amp)
            total_amplitude += amp
            freq *= 2.0
            amp *= max(0.0, min(1.0, octaves_decay))

        if total_amplitude > 0.0:
            result /= total_amplitude

        return np.clip(result, 0.0, 1.0)

    def _vectorized_simplex_2d(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Evaluates 2D Simplex noise across arrays using pure NumPy math.
        """
        s_val = (x + y) * F2
        i = np.floor(x + s_val).astype(np.int32)
        j = np.floor(y + s_val).astype(np.int32)

        t_val = (i + j) * G2
        x0 = x - (i - t_val)
        y0 = y - (j - t_val)

        i1 = np.where(x0 > y0, 1, 0)
        j1 = np.where(x0 > y0, 0, 1)

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + (2.0 * G2)
        y2 = y0 - 1.0 + (2.0 * G2)

        ii = i & 255
        jj = j & 255

        p = self.p_arr
        g_arr = GRAD2_ARR
        p_g = self.perm_grad_arr

        gi0 = p_g[p[ii + p[jj]]]
        gi1 = p_g[p[ii + i1 + p[jj + j1]]]
        gi2 = p_g[p[ii + 1 + p[jj + 1]]]

        g0 = g_arr[gi0]
        g1 = g_arr[gi1]
        g2 = g_arr[gi2]

        t0 = np.maximum(0.0, 0.5 - (x0 * x0) - (y0 * y0))
        t1 = np.maximum(0.0, 0.5 - (x1 * x1) - (y1 * y1))
        t2 = np.maximum(0.0, 0.5 - (x2 * x2) - (y2 * y2))

        t0_sq = t0 * t0
        t1_sq = t1 * t1
        t2_sq = t2 * t2

        n0 = (t0_sq * t0_sq) * (g0[..., 0] * x0 + g0[..., 1] * y0)
        n1 = (t1_sq * t1_sq) * (g1[..., 0] * x1 + g1[..., 1] * y1)
        n2 = (t2_sq * t2_sq) * (g2[..., 0] * x2 + g2[..., 1] * y2)

        raw_val = 70.0 * (n0 + n1 + n2)
        norm_val = (raw_val + 1.0) * 0.5
        return np.clip(norm_val, 0.0, 1.0)


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
        self.p_arr: NDArray[np.int32] = np.array(
            self.p, dtype=np.int32
        )
        self.perm_grad_idx: List[int] = [i % 8 for i in self.p]
        self.perm_grad_arr: NDArray[np.int32] = np.array(
            self.perm_grad_idx, dtype=np.int32
        )

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

        p = self.p
        aa: int = p[p[cell_x] + cell_y]
        ab: int = p[p[cell_x] + cell_y + 1]
        ba: int = p[p[cell_x + 1] + cell_y]
        bb: int = p[p[cell_x + 1] + cell_y + 1]

        gi00 = self.perm_grad_idx[aa]
        gi10 = self.perm_grad_idx[ba]
        gi01 = self.perm_grad_idx[ab]
        gi11 = self.perm_grad_idx[bb]

        g00 = GRAD2[gi00]
        g10 = GRAD2[gi10]
        g01 = GRAD2[gi01]
        g11 = GRAD2[gi11]

        dot00 = g00[0] * xf + g00[1] * yf
        dot10 = g10[0] * (xf - 1.0) + g10[1] * yf
        dot01 = g01[0] * xf + g01[1] * (yf - 1.0)
        dot11 = g11[0] * (xf - 1.0) + g11[1] * (yf - 1.0)

        x1: float = dot00 + u * (dot10 - dot00)
        x2: float = dot01 + u * (dot11 - dot01)
        raw_val: float = x1 + v * (x2 - x1)

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
            noise_layer = self._vectorized_perlin_2d(
                x_grid * freq, y_grid * freq
            )
            result += (noise_layer * amp)
            total_amplitude += amp
            freq *= 2.0
            amp *= max(0.0, min(1.0, octaves_decay))

        if total_amplitude > 0.0:
            result /= total_amplitude

        return np.clip(result, 0.0, 1.0)

    def _vectorized_perlin_2d(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Evaluates 2D Perlin noise across arrays using pure NumPy math.
        """
        cell_x = np.floor(x).astype(np.int32) & 255
        cell_y = np.floor(y).astype(np.int32) & 255

        xf = x - np.floor(x)
        yf = y - np.floor(y)

        u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
        v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

        p = self.p_arr
        g_arr = GRAD2_ARR
        p_g = self.perm_grad_arr

        aa = p_g[p[p[cell_x] + cell_y]]
        ab = p_g[p[p[cell_x] + cell_y + 1]]
        ba = p_g[p[p[cell_x + 1] + cell_y]]
        bb = p_g[p[p[cell_x + 1] + cell_y + 1]]

        g00 = g_arr[aa]
        g10 = g_arr[ba]
        g01 = g_arr[ab]
        g11 = g_arr[bb]

        dot00 = g00[..., 0] * xf + g00[..., 1] * yf
        dot10 = g10[..., 0] * (xf - 1.0) + g10[..., 1] * yf
        dot01 = g01[..., 0] * xf + g01[..., 1] * (yf - 1.0)
        dot11 = g11[..., 0] * (xf - 1.0) + g11[..., 1] * (yf - 1.0)

        x1 = dot00 + u * (dot10 - dot00)
        x2 = dot01 + u * (dot11 - dot01)
        raw_val = x1 + v * (x2 - x1)

        norm_val = (raw_val + 1.0) * 0.5
        return np.clip(norm_val, 0.0, 1.0)

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
