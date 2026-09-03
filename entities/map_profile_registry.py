"""
Parses profiles/map.yaml and resolves immutable map layout profiles.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import sys

try:
    import yaml
except ImportError:
    yaml = None


@dataclass(frozen=True, slots=True)
class ResolvedMapProfile:
    """
    Immutable container holding map dimensions and generator parameters.
    """

    profile_name: str
    map_type: str
    map_width: int
    map_height: int
    tile_size: int
    wall_density: float
    stem_early_termination_rate: float
    min_straight_start_steps: int


class MapProfileRegistry:
    """
    Parses profiles/map.yaml and provides map profile resolution.
    """

    def __init__(self, library_path: Optional[Path] = None) -> None:
        """
        Initializes registry and loads map library from YAML.
        """
        self._profiles: Dict[str, ResolvedMapProfile] = {}
        root_dir: Path = Path(__file__).resolve().parents[1]
        self.library_path: Path = library_path or (
            root_dir / "profiles" / "map.yaml"
        )
        self.load_library()

    def get_profile(self, profile_name: str) -> ResolvedMapProfile:
        """
        Retrieves resolved map profile or fails fast with clear error.
        """
        if profile_name not in self._profiles:
            print(
                f"[Error] Map profile '{profile_name}' is not defined in "
                f"{self.library_path.name}. Available map profiles: "
                f"{list(self._profiles.keys())}"
            )
            sys.exit(1)
        return self._profiles[profile_name]

    def load_library(self) -> None:
        """
        Parses YAML configuration file and caches resolved map profiles.
        """
        if not self.library_path.exists():
            print(
                f"[Error] Map library YAML file missing: "
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
    ) -> ResolvedMapProfile:
        """
        Extracts map geometry fields and builds ResolvedMapProfile.
        """
        return ResolvedMapProfile(
            profile_name=p_name,
            map_type=str(p_dict.get("map_type", "BRANCHING_WALLS")),
            map_width=int(p_dict.get("map_width", 40)),
            map_height=int(p_dict.get("map_height", 30)),
            tile_size=int(p_dict.get("tile_size", 40)),
            wall_density=float(p_dict.get("wall_density", 1.0)),
            stem_early_termination_rate=float(
                p_dict.get("stem_early_termination_rate", 0.05)
            ),
            min_straight_start_steps=int(
                p_dict.get("min_straight_start_steps", 1)
            )
        )
