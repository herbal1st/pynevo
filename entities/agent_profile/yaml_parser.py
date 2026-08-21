"""
Parses profiles/agent.yaml with strict fail-fast validation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import sys

from entities.skin_profile_registry import SkinProfileRegistry
from entities.agent_profile.profile_model import ResolvedAgentProfile

try:
    import yaml
except ImportError:
    yaml = None


class AgentProfileParser:
    """
    Parses agent library YAML files enforcing strict fail-fast validation.
    """

    clamped_warnings: Dict[str, Tuple[int, int]] = {}

    @classmethod
    def get_clamped_warning_strings(cls) -> List[str]:
        """
        Returns formatted memory frame safety cap warning strings.
        """
        warnings: List[str] = []
        for p_name, (raw_val, cap_val) in cls.clamped_warnings.items():
            warnings.append(
                f"[Warning] Agent profile '{p_name}' memory_frames "
                f"({raw_val}) exceeds hard safety cap ({cap_val}). "
                f"Clamping to {cap_val}."
            )
        return warnings

    @staticmethod
    def parse_library(
        library_path: Path,
        skin_registry: SkinProfileRegistry
    ) -> Dict[str, ResolvedAgentProfile]:
        """
        Parses YAML configuration file and builds resolved profile map.
        """
        if not library_path.exists():
            print(
                f"[Error] Agent library YAML file missing: {library_path}"
            )
            sys.exit(1)

        if yaml is None:
            print(
                "[Error] PyYAML library is not installed. Please install "
                "via 'pip install pyyaml'."
            )
            sys.exit(1)

        try:
            with open(library_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                raw_profiles: Dict[str, Any] = data.get("profiles", {})
        except Exception as e:
            print(
                f"[Error] Failed to parse {library_path.name}: {e}"
            )
            sys.exit(1)

        profiles: Dict[str, ResolvedAgentProfile] = {}
        for p_name, p_dict in raw_profiles.items():
            resolved = AgentProfileParser._parse_profile_dict(
                p_name, p_dict, library_path, skin_registry
            )
            profiles[p_name] = resolved

        return profiles

    @staticmethod
    def _parse_profile_dict(
        p_name: str,
        p_dict: Dict[str, Any],
        library_path: Path,
        skin_registry: SkinProfileRegistry
    ) -> ResolvedAgentProfile:
        """
        Extracts behavior fields, resolves skin, & builds Agent profile.
        """
        file_name: str = library_path.name
        kin = AgentProfileParser._get_required_section(
            p_dict, "kinematics", p_name, file_name
        )
        perc = AgentProfileParser._get_required_section(
            p_dict, "perception", p_name, file_name
        )
        neural = AgentProfileParser._get_required_section(
            p_dict, "neural", p_name, file_name
        )

        skin_key: str = str(
            AgentProfileParser._get_required_val(
                p_dict, "skin_profile", "root", p_name, file_name
            )
        )
        resolved_skin = skin_registry.get_skin(skin_key)

        style: str = str(
            AgentProfileParser._get_required_val(
                kin, "profile_style", "kinematics", p_name, file_name
            )
        )
        use_linear: bool = bool(
            AgentProfileParser._get_required_val(
                kin, "use_linear_speed_output", "kinematics",
                p_name, file_name
            )
        )
        move_speed: float = float(
            AgentProfileParser._get_required_val(
                kin, "move_speed", "kinematics", p_name, file_name
            )
        )
        turn_speed: float = float(
            AgentProfileParser._get_required_val(
                kin, "turn_speed", "kinematics", p_name, file_name
            )
        )
        diam_ratio: float = float(
            AgentProfileParser._get_required_val(
                kin, "player_diameter_ratio", "kinematics", p_name, file_name
            )
        )
        coll_dmg: float = float(
            AgentProfileParser._get_required_val(
                kin, "health_coll_dmg_per_frame", "kinematics",
                p_name, file_name
            )
        )
        idle_dmg: float = float(
            AgentProfileParser._get_required_val(
                kin, "health_idle_dmg_per_frame", "kinematics",
                p_name, file_name
            )
        )
        spin_dmg: float = float(
            AgentProfileParser._get_required_val(
                kin, "health_spin_dmg_per_frame", "kinematics",
                p_name, file_name
            )
        )
        idle_thresh: float = float(
            AgentProfileParser._get_required_val(
                kin, "idle_damage_speed_threshold", "kinematics",
                p_name, file_name
            )
        )
        heal_thresh: float = float(
            AgentProfileParser._get_required_val(
                kin, "heal_speed_threshold", "kinematics", p_name, file_name
            )
        )
        move_heal: float = float(
            AgentProfileParser._get_required_val(
                kin, "move_heal_per_frame", "kinematics", p_name, file_name
            )
        )
        path_heal: float = float(
            AgentProfileParser._get_required_val(
                kin, "path_heal_per_frame", "kinematics", p_name, file_name
            )
        )

        v_rays: int = int(
            AgentProfileParser._get_required_val(
                perc, "vision_rays", "perception", p_name, file_name
            )
        )
        v_arc: float = float(
            AgentProfileParser._get_required_val(
                perc, "vision_arc_angle", "perception", p_name, file_name
            )
        )
        v_dist: float = float(
            AgentProfileParser._get_required_val(
                perc, "vision_max_dist", "perception", p_name, file_name
            )
        )

        act_exit: bool = bool(
            AgentProfileParser._get_required_val(
                perc, "activate_exit_compass", "perception", p_name, file_name
            )
        )
        act_exit_los: bool = bool(
            AgentProfileParser._get_required_val(
                perc, "exit_compass_los_gating", "perception",
                p_name, file_name
            )
        )
        rng_exit: float = float(
            AgentProfileParser._get_required_val(
                perc, "range_exit_compass", "perception", p_name, file_name
            )
        )
        act_north: bool = bool(
            AgentProfileParser._get_required_val(
                perc, "activate_north_compass", "perception",
                p_name, file_name
            )
        )
        act_cardinal: bool = bool(
            AgentProfileParser._get_required_val(
                perc, "activate_cardinal_compass", "perception",
                p_name, file_name
            )
        )
        act_gps: bool = bool(
            AgentProfileParser._get_required_val(
                perc, "activate_gps_compass", "perception", p_name, file_name
            )
        )
        use_binoc_gps: bool = bool(
            AgentProfileParser._get_required_val(
                perc, "use_binocular_gps_compasses", "perception",
                p_name, file_name
            )
        )
        rng_gps: float = float(
            AgentProfileParser._get_required_val(
                perc, "range_gps_compass", "perception", p_name, file_name
            )
        )
        use_bfs_spawn_heading: bool = bool(
            AgentProfileParser._get_required_val(
                perc, "use_bfs_spawn_heading", "perception",
                p_name, file_name
            )
        )
        offset_angle: float = float(
            AgentProfileParser._get_required_val(
                perc, "target_compasses_offset_angle", "perception",
                p_name, file_name
            )
        )
        focus_fov: float = float(
            AgentProfileParser._get_required_val(
                perc, "focus_field_of_view", "perception", p_name, file_name
            )
        )
        periphere_fov: float = float(
            AgentProfileParser._get_required_val(
                perc, "periphere_field_of_view", "perception",
                p_name, file_name
            )
        )

        raw_mem: int = int(
            AgentProfileParser._get_required_val(
                neural, "memory_frames", "neural", p_name, file_name
            )
        )
        hidden_l: int = int(
            AgentProfileParser._get_required_val(
                neural, "hidden_layers", "neural", p_name, file_name
            )
        )
        neu_cnt: int = int(
            AgentProfileParser._get_required_val(
                neural, "neurons", "neural", p_name, file_name
            )
        )

        max_mem_cap: int = 10
        if raw_mem > max_mem_cap:
            AgentProfileParser.clamped_warnings[p_name] = (
                raw_mem, max_mem_cap
            )
            mem_frames_val: int = max_mem_cap
        else:
            mem_frames_val = max(1, raw_mem)

        return ResolvedAgentProfile(
            profile_name=p_name,
            skin=resolved_skin,
            profile_style=style,
            use_linear_speed_output=use_linear,
            move_speed=move_speed,
            turn_speed=turn_speed,
            player_diameter_ratio=diam_ratio,
            health_coll_dmg_per_frame=coll_dmg,
            health_idle_dmg_per_frame=idle_dmg,
            health_spin_dmg_per_frame=spin_dmg,
            idle_damage_speed_threshold=idle_thresh,
            heal_speed_threshold=heal_thresh,
            move_heal_per_frame=move_heal,
            path_heal_per_frame=path_heal,
            vision_rays=v_rays,
            vision_arc_angle=v_arc,
            vision_max_dist=v_dist,
            activate_exit_compass=act_exit,
            exit_compass_los_gating=act_exit_los,
            range_exit_compass=rng_exit,
            activate_north_compass=act_north,
            activate_cardinal_compass=act_cardinal,
            activate_gps_compass=act_gps,
            use_binocular_gps_compasses=use_binoc_gps,
            range_gps_compass=rng_gps,
            use_bfs_spawn_heading=use_bfs_spawn_heading,
            target_compasses_offset_angle=offset_angle,
            focus_field_of_view=focus_fov,
            periphere_field_of_view=periphere_fov,
            memory_frames=mem_frames_val,
            hidden_layers=hidden_l,
            neurons=neu_cnt
        )

    @staticmethod
    def _get_required_section(
        p_dict: Dict[str, Any],
        section_name: str,
        profile_name: str,
        file_name: str
    ) -> Dict[str, Any]:
        """
        Enforces section existence or fails fast with an explicit error.
        """
        if section_name not in p_dict:
            print(
                f"[Error] Profile '{profile_name}' in profiles/{file_name} "
                f"is missing required section '{section_name}'."
            )
            sys.exit(1)
        return p_dict[section_name]

    @staticmethod
    def _get_required_val(
        d: Dict[str, Any],
        key_name: str,
        section_name: str,
        profile_name: str,
        file_name: str
    ) -> Any:
        """
        Enforces key existence in dict or fails fast with an explicit error.
        """
        if key_name not in d:
            print(
                f"[Error] Profile '{profile_name}' in section "
                f"'{section_name}' of profiles/{file_name} is missing "
                f"required key '{key_name}'."
            )
            sys.exit(1)
        return d[key_name]
