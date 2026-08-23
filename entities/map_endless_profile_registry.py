"""
Parses profiles/map_endless.yaml and resolves endless map profiles.
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
class ResolvedMapEndlessProfile:
    """
    Immutable data model holding endless map generation settings.
    """

    profile_name: str
    map_type: str
    world_seed: int
    noise_type: str
    noise_scale: float
    octaves: int
    octaves_decay: float
    tile_size: int
    strata_layers: Tuple[Tuple[float, str], ...]


class MapEndlessProfileRegistry:
    """
    Parses profiles/map_endless.yaml and manages profile resolution.
    """

    def __init__(self, library_path: Optional[Path] = None) -> None:
        """
        Initializes registry and loads endless map library from YAML.
        """
        self._profiles: Dict[str, ResolvedMapEndlessProfile] = {}
        root_dir: Path = Path(__file__).resolve().parents[1]
        self.library_path: Path = library_path or (
            root_dir / "profiles" / "map_endless.yaml"
        )
        self.load_library()

    def get_profile(self, profile_name: str) -> ResolvedMapEndlessProfile:
        """
        Retrieves resolved endless profile or fails fast with error.
        """
        if profile_name not in self._profiles:
            print(
                f"[Error] Endless map profile '{profile_name}' "
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
                f"[Error] Endless map library YAML missing: "
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
    ) -> ResolvedMapEndlessProfile:
        """
        Extracts hyperparameter fields and builds ResolvedMapEndlessProfile.
        """
        file_name: str = self.library_path.name

        m_type: str = str(
            self._get_required_val(p_dict, "map_type", p_name, file_name)
        )
        seed: int = int(
            self._get_required_val(p_dict, "world_seed", p_name, file_name)
        )
        n_type: str = str(
            self._get_required_val(p_dict, "noise_type", p_name, file_name)
        )
        scale: float = float(
            self._get_required_val(p_dict, "noise_scale", p_name, file_name)
        )
        octs: int = int(
            self._get_required_val(p_dict, "octaves", p_name, file_name)
        )
        decay: float = float(
            self._get_required_val(
                p_dict, "octaves_decay", p_name, file_name
            )
        )
        t_size: int = int(
            self._get_required_val(p_dict, "tile_size", p_name, file_name)
        )
        raw_strata: List[Any] = list(
            self._get_required_val(
                p_dict, "strata_layers", p_name, file_name
            )
        )

        parsed_strata: List[Tuple[float, str]] = []
        for item in raw_strata:
            thresh: float = float(item[0])
            tile_name: str = str(item[1]).upper()
            parsed_strata.append((thresh, tile_name))

        parsed_strata.sort(key=lambda x: x[0])

        return ResolvedMapEndlessProfile(
            profile_name=p_name,
            map_type=m_type,
            world_seed=seed,
            noise_type=n_type.upper(),
            noise_scale=max(0.1, scale),
            octaves=max(1, octs),
            octaves_decay=max(0.0, min(1.0, decay)),
            tile_size=max(1, t_size),
            strata_layers=tuple(parsed_strata)
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
