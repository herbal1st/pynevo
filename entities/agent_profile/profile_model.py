"""
Immutable container holding resolved agent attributes and skin delegations.
"""

from dataclasses import dataclass

import config
from entities.skin_profile_registry import ResolvedSkinProfile


@dataclass(frozen=True, slots=True)
class ResolvedAgentProfile:
    """
    Immutable container holding behavior attributes and composed skin.
    """

    profile_name: str
    skin: ResolvedSkinProfile
    profile_style: str
    use_linear_speed_output: bool
    move_speed: float
    turn_speed: float
    player_diameter_ratio: float
    health_coll_dmg_per_frame: float
    health_idle_dmg_per_frame: float
    health_spin_dmg_per_frame: float
    idle_damage_speed_threshold: float
    heal_speed_threshold: float
    move_heal_per_frame: float
    path_heal_per_frame: float
    vision_rays: int
    vision_arc_angle: float
    vision_max_dist: float
    activate_exit_compass: bool
    exit_compass_los_gating: bool
    range_exit_compass: float
    activate_north_compass: bool
    activate_cardinal_compass: bool
    activate_gps_compass: bool
    use_binocular_gps_compasses: bool
    range_gps_compass: float
    use_bfs_spawn_heading: bool
    target_compasses_offset_angle: float
    focus_field_of_view: float
    periphere_field_of_view: float
    memory_frames: int
    hidden_layers: int
    neurons: int

    @property
    def health_recovery_ratio(self) -> float:
        """
        Backward compatibility alias for move_heal_per_frame.
        """
        return self.move_heal_per_frame

    @property
    def player_camera_zoom(self) -> float:
        """
        Forwards camera zoom scale from config.
        """
        return config.PLAYER_CAMERA_ZOOM

    @property
    def player_radius_ratio(self) -> float:
        """
        Returns true physical body radius ratio (0.5 * diameter).
        """
        return 0.5 * self.player_diameter_ratio

    @property
    def activate_bfs_way(self) -> bool:
        """
        Backward compatibility alias for activate_gps_compass.
        """
        return self.activate_gps_compass

    @property
    def range_bfs_way(self) -> float:
        """
        Backward compatibility alias for range_gps_compass.
        """
        return self.range_bfs_way

    @property
    def use_binocular_compasses(self) -> bool:
        """
        Backward compatibility alias for use_binocular_gps_compasses.
        """
        return self.use_binocular_gps_compasses
