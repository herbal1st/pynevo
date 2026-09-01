"""
Interpolates RGB ambient colors across profile time keyframes.
"""

from typing import Tuple
from entities.lighting_profile_registry import ResolvedLightingProfile


class AmbientPaletteResolver:
    """
    Maps normalized time-of-day ratios to RGB overlay colors.
    """

    @classmethod
    def resolve_ambient_color(
        cls,
        time_ratio: float,
        profile: ResolvedLightingProfile
    ) -> Tuple[int, int, int]:
        """
        Interpolates ambient RGB tuple for normalized time ratio.
        """
        keyframes = profile.ambient_keyframes
        if not keyframes:
            return 255, 255, 255

        clamped_time: float = max(0.0, min(1.0, float(time_ratio)))

        for i in range(len(keyframes) - 1):
            t0, r0, g0, b0 = keyframes[i]
            t1, r1, g1, b1 = keyframes[i + 1]

            if t0 <= clamped_time <= t1:
                span: float = max(1e-6, t1 - t0)
                factor: float = (clamped_time - t0) / span

                r: int = int(round(r0 + factor * (r1 - r0)))
                g: int = int(round(g0 + factor * (g1 - g0)))
                b: int = int(round(b0 + factor * (b1 - b0)))

                return r, g, b

        last_kf = keyframes[-1]
        return last_kf[1], last_kf[2], last_kf[3]
