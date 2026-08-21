"""
Color interpolation and status palette resolution utilities.
"""

from typing import Tuple
import config


def interpolate_rgb(
    color_a: Tuple[int, ...],
    color_b: Tuple[int, ...],
    ratio: float
) -> Tuple[int, int, int]:
    """
    Linearly interpolates between two RGB color tuples by ratio [0.0, 1.0].
    """
    clamped_ratio: float = max(0.0, min(1.0, ratio))
    r_val: int = int(
        color_a[0] + clamped_ratio * (color_b[0] - color_a[0])
    )
    g_val: int = int(
        color_a[1] + clamped_ratio * (color_b[1] - color_a[1])
    )
    b_val: int = int(
        color_a[2] + clamped_ratio * (color_b[2] - color_a[2])
    )
    return r_val, g_val, b_val


def resolve_health_color(
    health_ratio: float
) -> Tuple[int, int, int]:
    """
    Returns health bar fill RGB color based on remaining health ratio.
    """
    clamped_ratio: float = max(0.0, min(1.0, health_ratio))
    if clamped_ratio >= 0.50:
        return config.COLOR_HEALTH_FULL[:3]
    if clamped_ratio >= 0.20:
        return config.COLOR_HEALTH_MID[:3]
    return config.COLOR_HEALTH_LOW[:3]


def resolve_net_delta_color(
    net_delta: float
) -> Tuple[int, int, int]:
    """
    Maps Net Delta (Bad - Good) across Red <- Yellow -> Green scale.
    """
    c_neutral: Tuple[int, int, int] = config.COLOR_PLAYER_HIGHLIGHT[:3]
    c_damage: Tuple[int, int, int] = config.COLOR_HEALTH_LOW[:3]
    c_heal: Tuple[int, int, int] = config.COLOR_EXIT[:3]

    if net_delta > 0.0:
        ratio: float = min(1.0, net_delta / 2.0)
        return interpolate_rgb(c_neutral, c_damage, ratio)

    if net_delta < 0.0:
        ratio = min(1.0, abs(net_delta) / 1.0)
        return interpolate_rgb(c_neutral, c_heal, ratio)

    return c_neutral


def resolve_activation_color(
    activation_intensity: float
) -> Tuple[int, int, int]:
    """
    Resolves node color gradient from inactive RGB to active RGB.
    """
    return interpolate_rgb(
        config.COLOR_NODE_INACTIVE[:3],
        config.COLOR_NODE_ACTIVE[:3],
        activation_intensity
    )


def resolve_solve_ratio_color(
    solve_ratio: float
) -> Tuple[int, int, int]:
    """
    Maps solve ratio [0.0, 1.0] across Red -> Orange -> Yellow -> Green.
    """
    clamped: float = max(0.0, min(1.0, solve_ratio))
    c_red: Tuple[int, int, int] = (220, 50, 50)
    c_orange: Tuple[int, int, int] = (255, 140, 0)
    c_yellow: Tuple[int, int, int] = (255, 220, 50)
    c_green: Tuple[int, int, int] = (50, 200, 100)

    if clamped <= 0.33:
        sub_ratio: float = clamped / 0.33
        return interpolate_rgb(c_red, c_orange, sub_ratio)
    if clamped <= 0.66:
        sub_ratio = (clamped - 0.33) / 0.33
        return interpolate_rgb(c_orange, c_yellow, sub_ratio)

    sub_ratio = (clamped - 0.66) / 0.34
    return interpolate_rgb(c_yellow, c_green, sub_ratio)


def resolve_rank_color(
    rank_index: int,
    total_solvers: int
) -> Tuple[int, int, int]:
    """
    Maps candidate speed rank (0 = fastest) across Green -> Yellow -> Red.
    """
    c_green: Tuple[int, int, int] = (50, 200, 100)
    c_yellow: Tuple[int, int, int] = (255, 220, 50)
    c_orange: Tuple[int, int, int] = (255, 140, 0)
    c_red: Tuple[int, int, int] = (220, 50, 50)

    if total_solvers <= 1:
        return c_green

    if total_solvers == 2:
        return c_green if rank_index == 0 else c_red

    rank_ratio: float = float(rank_index) / float(total_solvers - 1)

    if rank_ratio <= 0.33:
        sub_ratio: float = rank_ratio / 0.33
        return interpolate_rgb(c_green, c_yellow, sub_ratio)
    if rank_ratio <= 0.66:
        sub_ratio = (rank_ratio - 0.33) / 0.33
        return interpolate_rgb(c_yellow, c_orange, sub_ratio)

    sub_ratio = (rank_ratio - 0.66) / 0.34
    return interpolate_rgb(c_orange, c_red, sub_ratio)
