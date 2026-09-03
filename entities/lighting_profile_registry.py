"""
Parses profiles/lighting.yaml and resolves immutable lighting profiles.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import sys

try:
    import yaml
except ImportError:
    yaml = None


@dataclass(frozen=True, slots=True)
class ResolvedLightingProfile:
    """
    Immutable data model holding lighting and atmosphere settings.
    """

    profile_name: str
    day_cycle_duration: float
    start_time_ratio: float
    start_light_angle_deg: float
    terrain_steepness: float
    shadow_intensity: int
    highlight_intensity: int
    ambient_keyframes: Tuple[Tuple[float, int, int, int], ...]


class LightingProfileRegistry:
    """
    Parses profiles/lighting.yaml and manages profile resolution.
    """

    def __init__(self, library_path: Optional[Path] = None) -> None:
        """
        Initializes registry and loads lighting library from YAML.
        """
        self._profiles: Dict[str, ResolvedLightingProfile] = {}
        root_dir: Path = Path(__file__).resolve().parents[1]
        self.library_path: Path = library_path or (
            root_dir / "profiles" / "lighting.yaml"
        )
        self.load_library()

    def get_profile(self, profile_name: str) -> ResolvedLightingProfile:
        """
        Retrieves resolved lighting profile or fails fast with error.
        """
        if profile_name not in self._profiles:
            print(
                f"[Error] Lighting profile '{profile_name}' "
                f"is not defined in {self.library_path.name}. "
                f"Available profiles: {list(self._profiles.keys())}"
            )
            sys.exit(1)
        return self._profiles[profile_name]

    def load_library(self) -> None:
        """
        Parses YAML configuration file and caches resolved profiles.
        """
        if not self.library_path.exists():
            print(
                f"[Error] Lighting library YAML missing: "
                f"{self.library_path}"
            )
            sys.exit(1)

        if yaml is None:
            print(
                "[Error] PyYAML library is not installed. Please install "
                "via 'pip install pyyaml'."
            )
            sys.exit(1)

        try:
            with open(self.library_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                raw_profiles: Dict[str, Any] = data.get("profiles", {})
        except Exception as e:
            print(
                f"[Error] Failed to parse {self.library_path.name}: {e}"
            )
            sys.exit(1)

        self._profiles.clear()
        for p_name, p_dict in raw_profiles.items():
            resolved = self._parse_profile_dict(p_name, p_dict)
            self._profiles[p_name] = resolved

    def _parse_profile_dict(
        self,
        p_name: str,
        p_dict: Dict[str, Any]
    ) -> ResolvedLightingProfile:
        """
        Extracts lighting fields and builds ResolvedLightingProfile.
        """
        file_name: str = self.library_path.name

        duration: float = float(
            self._get_required_val(
                p_dict, "day_cycle_duration", p_name, file_name
            )
        )
        start_ratio: float = float(
            self._get_required_val(
                p_dict, "start_time_ratio", p_name, file_name
            )
        )
        start_angle: float = float(
            p_dict.get("start_light_angle_deg", 135.0)
        )
        steepness: float = float(
            self._get_required_val(
                p_dict, "terrain_steepness", p_name, file_name
            )
        )

        sh_intensity: int = int(
            self._get_required_val(
                p_dict, "shadow_intensity", p_name, file_name
            )
        )
        hl_intensity: int = int(
            self._get_required_val(
                p_dict, "highlight_intensity", p_name, file_name
            )
        )

        raw_keyframes: List[Any] = list(
            self._get_required_val(
                p_dict, "ambient_keyframes", p_name, file_name
            )
        )

        parsed_keyframes: List[Tuple[float, int, int, int]] = []
        for item in raw_keyframes:
            t_rat: float = float(item[0])
            r_val: int = int(item[1])
            g_val: int = int(item[2])
            b_val: int = int(item[3])
            parsed_keyframes.append((t_rat, r_val, g_val, b_val))

        parsed_keyframes.sort(key=lambda x: x[0])

        return ResolvedLightingProfile(
            profile_name=p_name,
            day_cycle_duration=max(0.0, duration),
            start_time_ratio=max(0.0, min(1.0, start_ratio)),
            start_light_angle_deg=start_angle % 360.0,
            terrain_steepness=max(0.1, steepness),
            shadow_intensity=max(0, min(255, sh_intensity)),
            highlight_intensity=max(0, min(255, hl_intensity)),
            ambient_keyframes=tuple(parsed_keyframes)
        )

    def _get_required_val(
        self,
        d: Dict[str, Any],
        key_name: str,
        profile_name: str,
        file_name: str
    ) -> Any:
        """
        Enforces key existence in dict or fails fast with error message.
        """
        if key_name not in d:
            print(
                f"[Error] Profile '{profile_name}' in profiles/{file_name} "
                f"is missing required key '{key_name}'."
            )
            sys.exit(1)
        return d[key_name]
