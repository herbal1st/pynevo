"""
Parses profiles/tiles.yaml and provides O(1) tile property lookups.
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
class ResolvedTileProfile:
    """
    Immutable data model holding tile physical properties and styling.
    """

    tile_id: int
    name: str
    solid: bool
    indestructible: bool
    speed_multiplier: float
    color: tuple[int, int, int]
    border_color: tuple[int, int, int]
    border_width_ratio: float


class TileRegistry:
    """
    Parses tiles.yaml and manages fast O(1) tile attribute resolution.
    """

    def __init__(self, library_path: Optional[Path] = None) -> None:
        """
        Initializes registry cache and loads tile configuration from disk.
        """
        self._by_id: Dict[int, ResolvedTileProfile] = {}
        self._by_name: Dict[str, ResolvedTileProfile] = {}
        root_dir: Path = Path(__file__).resolve().parents[1]
        self.library_path: Path = library_path or (
            root_dir / "profiles" / "tiles.yaml"
        )
        self.load_library()

    def get_tile(self, tile_id: int) -> ResolvedTileProfile:
        """
        Retrieves resolved tile profile by ID or fails fast with error.
        """
        if tile_id not in self._by_id:
            print(
                f"[Error] Tile ID '{tile_id}' is not defined in "
                f"{self.library_path.name}. Registered IDs: "
                f"{list(self._by_id.keys())}"
            )
            sys.exit(1)
        return self._by_id[tile_id]

    def get_tile_by_name(self, name: str) -> ResolvedTileProfile:
        """
        Retrieves resolved tile profile by name or fails fast with error.
        """
        upper_name: str = name.upper()
        if upper_name not in self._by_name:
            print(
                f"[Error] Tile name '{name}' is not defined in "
                f"{self.library_path.name}. Registered names: "
                f"{list(self._by_name.keys())}"
            )
            sys.exit(1)
        return self._by_name[upper_name]

    def load_library(self) -> None:
        """
        Parses YAML configuration file and caches resolved tile profiles.
        """
        if not self.library_path.exists():
            print(
                f"[Error] Tile library YAML file missing: "
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
                raw_tiles: Dict[str, Any] = data.get("profiles", {})
        except Exception as e:
            print(
                f"[Error] Failed to parse {self.library_path.name}: {e}"
            )
            sys.exit(1)

        self._by_id.clear()
        self._by_name.clear()

        for t_key, t_dict in raw_tiles.items():
            resolved = self._parse_tile_dict(t_key, t_dict)
            self._by_id[resolved.tile_id] = resolved
            self._by_name[resolved.name.upper()] = resolved

    def _parse_tile_dict(
        self,
        t_key: str,
        t_dict: Dict[str, Any]
    ) -> ResolvedTileProfile:
        """
        Extracts required tile fields and constructs ResolvedTileProfile.
        """
        file_name: str = self.library_path.name
        t_id: int = int(
            self._get_required_val(t_dict, "id", t_key, file_name)
        )
        name: str = str(
            self._get_required_val(t_dict, "name", t_key, file_name)
        )
        solid: bool = bool(
            self._get_required_val(t_dict, "solid", t_key, file_name)
        )
        indestructible: bool = bool(
            self._get_required_val(
                t_dict, "indestructible", t_key, file_name
            )
        )
        speed_mult: float = float(
            self._get_required_val(
                t_dict, "speed_multiplier", t_key, file_name
            )
        )
        color_raw = self._get_required_val(
            t_dict, "color", t_key, file_name
        )
        border_col_raw = self._get_required_val(
            t_dict, "border_color", t_key, file_name
        )
        border_ratio: float = float(
            self._get_required_val(
                t_dict, "border_width_ratio", t_key, file_name
            )
        )

        return ResolvedTileProfile(
            tile_id=t_id,
            name=name,
            solid=solid,
            indestructible=indestructible,
            speed_multiplier=speed_mult,
            color=(int(color_raw[0]), int(color_raw[1]), int(color_raw[2])),
            border_color=(
                int(border_col_raw[0]),
                int(border_col_raw[1]),
                int(border_col_raw[2])
            ),
            border_width_ratio=max(0.0, min(1.0, border_ratio))
        )

    def _get_required_val(
        self,
        d: Dict[str, Any],
        key_name: str,
        tile_key: str,
        file_name: str
    ) -> Any:
        """
        Enforces key existence in dict or fails fast with error message.
        """
        if key_name not in d:
            print(
                f"[Error] Tile '{tile_key}' in profiles/{file_name} "
                f"is missing required key '{key_name}'."
            )
            sys.exit(1)
        return d[key_name]
