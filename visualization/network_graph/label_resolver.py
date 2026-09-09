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

        # 3 clearance channels
        labels.extend(["CLR-L", "CLR-F", "CLR-R"])
        # 7 proprioception channels
        labels.extend(["SPD", "HP", "DMG-C", "DMG-I", "DMG-S", "HEAL", "A-VEL"])
        # 2 compass channels
        labels.extend(["CMP-X", "CMP-Y"])
        # 3 visual exit channels (LOS only)
        labels.extend(["EX-VIS", "EX-FWD", "EX-LAT"])

        return labels

    def get_output_label_list(
        self,
        output_count: int,
        profile: Optional[ResolvedAgentProfile] = None
    ) -> List[str]:
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
