"""
Immutable container holding resolved agent attributes and skin delegations.
"""

from dataclasses import dataclass
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
    endless_move_speed: float
    turn_speed: float
    agent_diameter_ratio: float
    endless_agent_diameter_ratio: float
    base_dmg_per_frame: float
    move_fwd_dmg_per_frame: float
    move_bwd_dmg_per_frame: float
    spin_dmg_per_frame: float
    coll_dmg_per_frame: float
    idle_dmg_per_frame: float
    path_dmg_per_frame: float
    target_hold_dmg_per_frame: float
    move_dmg_threshold: float
    spin_dmg_threshold: float
    idle_dmg_threshold: float
    heal_speed_threshold: float
    target_hold_distance_threshold: float
    target_zone_plateau_intensity: float
    invert_target_zone_field: bool
    full_heal_on_stage_clear: bool
    base_heal_per_frame: float
    path_heal_per_frame: float
    move_fwd_heal_per_frame: float
    move_bwd_heal_per_frame: float
    spin_heal_per_frame: float
    coll_heal_per_frame: float
    idle_heal_per_frame: float
    target_hold_heal_per_frame: float
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
    def move_dmg_per_frame(self) -> float:
        """
        Backward compatibility alias for move_fwd_dmg_per_frame.
        """
        return self.move_fwd_dmg_per_frame

    @property
    def move_heal_per_frame(self) -> float:
        """
        Backward compatibility alias for move_fwd_heal_per_frame.
        """
        return self.move_fwd_heal_per_frame

    @property
    def health_coll_dmg_per_frame(self) -> float:
        """
        Backward compatibility alias for coll_dmg_per_frame.
        """
        return self.coll_dmg_per_frame

    @property
    def health_idle_dmg_per_frame(self) -> float:
        """
        Backward compatibility alias for idle_dmg_per_frame.
        """
        return self.idle_dmg_per_frame

    @property
    def health_spin_dmg_per_frame(self) -> float:
        """
        Backward compatibility alias for spin_dmg_per_frame.
        """
        return self.spin_dmg_per_frame

    @property
    def idle_damage_speed_threshold(self) -> float:
        """
        Backward compatibility alias for idle_dmg_threshold.
        """
        return self.idle_dmg_threshold

    @property
    def health_recovery_ratio(self) -> float:
        """
        Backward compatibility alias for move_fwd_heal_per_frame.
        """
        return self.move_fwd_heal_per_frame

    @property
    def camera_zoom(self) -> float:
        """
        Forwards camera zoom scale from composed skin profile.
        """
        return self.skin.camera_zoom

    @property
    def agent_radius_ratio(self) -> float:
        """
        Returns true physical body radius ratio (0.5 * diameter).
        """
        return 0.5 * self.agent_diameter_ratio

    @property
    def endless_agent_radius_ratio(self) -> float:
        """
        Returns endless physical body radius ratio (0.5 * endless_diameter).
        """
        return 0.5 * self.endless_agent_diameter_ratio

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
        return self.range_gps_compass

    @property
    def use_binocular_compasses(self) -> bool:
        """
        Backward compatibility alias for use_binocular_gps_compasses.
        """
        return self.use_binocular_gps_compasses
