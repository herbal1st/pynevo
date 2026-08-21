"""
Label resolution module for neural network observation channels and outputs.
"""

from typing import List, Optional

from entities.agent_profile_registry import ResolvedAgentProfile


class GraphLabelResolver:
    """
    Resolves shorthand observation channel labels and semantic output labels.
    """

    def get_base_shorthand_list(
        self,
        profile: ResolvedAgentProfile
    ) -> List[str]:
        """
        Generates shorthand labels for single-frame observation channels.
        """
        labels: List[str] = []
        num_rays: int = profile.vision_rays
        half_arc: float = profile.vision_arc_angle / 2.0

        if num_rays > 1:
            step: float = (2.0 * half_arc) / float(num_rays - 1)
            for ray_idx in range(num_rays):
                deg: int = int(round(-half_arc + (ray_idx * step)))
                labels.append(f"{deg:+d}°")
        else:
            labels.append("0°")

        labels.extend(["SPD", "HP", "DMG-C", "DMG-I", "DMG-S", "HEAL"])

        use_binocular: bool = profile.use_binocular_gps_compasses
        if not use_binocular:
            labels.extend(["BFS-", "BFS+"])
        else:
            labels.extend(["BFSL-", "BFSR-", "BFSL+", "BFSR+"])

        labels.extend(["C-N", "C-E", "C-S", "C-W"])
        labels.extend(["NFL", "NFR", "NPL", "NPR"])
        labels.extend(["EFL", "EFR", "EPL", "EPR"])

        return labels

    def get_output_label_list(
        self,
        output_count: int,
        profile: Optional[ResolvedAgentProfile] = None
    ) -> List[str]:
        """
        Generates semantic output labels matching active actuation mode.
        """
        use_linear: bool = (
            profile.use_linear_speed_output
            if profile is not None else False
        )

        if use_linear:
            base_labels: List[str] = ["FWD", "BWD", "S-L", "S-R"]
        else:
            base_labels = ["L-FWD", "L-BWD", "R-FWD", "R-BWD"]

        labels: List[str] = []
        for output_idx in range(max(1, output_count)):
            if output_idx < len(base_labels):
                labels.append(base_labels[output_idx])
            else:
                labels.append(f"OUT-{output_idx + 1}")

        return labels
