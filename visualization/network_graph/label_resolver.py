"""
Label resolution module for neural network observation channels with Cul-de-sac & Optical LOS detectors.
"""

from typing import List, Optional
from entities.agent_profile_registry import ResolvedAgentProfile


class GraphLabelResolver:
    def get_base_shorthand_list(self, profile: ResolvedAgentProfile) -> List[str]:
        labels: List[str] = []
        num_rays: int = profile.vision_rays
        half_arc: float = profile.vision_arc_angle / 2.0

        # 1. Wall Distance Rays
        if num_rays > 1:
            step: float = (2.0 * half_arc) / float(num_rays - 1)
            for ray_idx in range(num_rays):
                deg: int = int(round(-half_arc + (ray_idx * step)))
                labels.append(f"W{deg:+d}°")
        else:
            labels.append("W0°")

        # 2. SLAM Breadcrumb Scent Memory Rays
        if num_rays > 1:
            step = (2.0 * half_arc) / float(num_rays - 1)
            for ray_idx in range(num_rays):
                deg = int(round(-half_arc + (ray_idx * step)))
                labels.append(f"T{deg:+d}°")
        else:
            labels.append("T0°")

        # 3. Proprioception & Cul-de-sac Pocket Detection
        labels.extend(["SPD", "HP", "CUL_DE_SAC", "OPEN_FWD", "EXIT_LOS", "CNTR"])

        # 4. Compasses
        use_binocular: bool = profile.use_binocular_gps_compasses
        if not use_binocular:
            labels.extend(["BFS-", "BFS+"])
        else:
            labels.extend(["BFSL-", "BFSR-", "BFSL+", "BFSR+"])

        labels.extend(["C-N", "C-E", "C-S", "C-W"])
        labels.extend(["NFR", "NFL", "NPR", "NPL"])
        labels.extend(["EFR", "EFL", "EPR", "EPL"])

        return labels

    def get_output_label_list(
        self,
        output_count: int,
        profile: Optional[ResolvedAgentProfile] = None
    ) -> List[str]:
        return ["L-FWD", "L-BWD", "R-FWD", "R-BWD"]
