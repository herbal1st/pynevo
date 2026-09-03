"""
Parses profiles/player.yaml and resolves immutable player profiles.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sys

try:
    import yaml
except ImportError:
    yaml = None


@dataclass(frozen=True, slots=True)
class ResolvedPlayerProfile:
    """
    Immutable container holding human player mechanics and settings.
    """

    profile_name: str
    skin_profile: str
    profile_style: str
    move_speed: float
    turn_speed: float
    diameter_ratio: float
    min_spawn_speed: float = 0.1
    companion_offset: Tuple[float, float] = (-2.0, 0.0)

    @property
    def radius_ratio(self) -> float:
        """
        Returns physical body radius ratio relative to tile width.
        """
        return 0.5 * self.diameter_ratio


class PlayerProfileRegistry:
    """
    Parses profiles/player.yaml and provides player profile resolution.
    """

    def __init__(self, library_path: Optional[Path] = None) -> None:
        """
        Initializes registry and loads player library from YAML.
        """
        self._profiles: Dict[str, ResolvedPlayerProfile] = {}
        root_dir: Path = Path(__file__).resolve().parents[1]
        self.library_path: Path = library_path or (
            root_dir / "profiles" / "player.yaml"
        )
        self.load_library()

    def get_profile(self, profile_name: str) -> ResolvedPlayerProfile:
        """
        Retrieves resolved player profile or fails fast with clear error.
        """
        if profile_name not in self._profiles:
            print(
                f"[Error] Player profile '{profile_name}' is not defined in "
                f"{self.library_path.name}. Available profiles: "
                f"{list(self._profiles.keys())}"
            )
            sys.exit(1)
        return self._profiles[profile_name]

    def load_library(self) -> None:
        """
        Parses YAML configuration file and caches resolved player profiles.
        """
        if not self.library_path.exists():
            print(
                f"[Error] Player library YAML file missing: "
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
    ) -> ResolvedPlayerProfile:
        """
        Extracts player movement fields and builds ResolvedPlayerProfile.
        """
        file_name: str = self.library_path.name
        skin_key: str = str(
            self._get_required_val(
                p_dict, "skin_profile", p_name, file_name
            )
        )
        style: str = str(
            self._get_required_val(
                p_dict, "profile_style", p_name, file_name
            )
        )
        move_sp: float = float(
            self._get_required_val(p_dict, "move_speed", p_name, file_name)
        )
        turn_sp: float = float(
            self._get_required_val(p_dict, "turn_speed", p_name, file_name)
        )
        diam_ratio: float = float(
            self._get_required_val(
                p_dict, "diameter_ratio", p_name, file_name
            )
        )
        min_sp: float = float(p_dict.get("min_spawn_speed", 0.1))

        raw_offset = p_dict.get("companion_offset", [-2.0, 0.0])
        off_x: float = float(raw_offset[0])
        off_y: float = float(raw_offset[1])

        return ResolvedPlayerProfile(
            profile_name=p_name,
            skin_profile=skin_key,
            profile_style=style,
            move_speed=move_sp,
            turn_speed=turn_sp,
            diameter_ratio=diam_ratio,
            min_spawn_speed=min_sp,
            companion_offset=(off_x, off_y)
        )

    def _get_required_val(
        self,
        d: Dict[str, Any],
        key_name: str,
        profile_name: str,
        file_name: str
    ) -> Any:
        """
        Enforces key existence in dict or fails fast with clear error.
        """
        if key_name not in d:
            print(
                f"[Error] Profile '{profile_name}' in profiles/{file_name} "
                f"is missing required key '{key_name}'."
            )
            sys.exit(1)
        return d[key_name]
