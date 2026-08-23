"""
Parses profiles/skin.yaml and resolves immutable skin visual profiles.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import sys

try:
    import yaml
except ImportError:
    yaml = None


@dataclass(frozen=True, slots=True)
class ResolvedSkinProfile:
    """
    Immutable container holding skin visual rendering attributes.
    """

    skin_name: str
    camera_zoom: float
    color_player: Tuple[int, int, int]
    color_player_text: Tuple[int, int, int]
    color_player_vision_arc: Tuple[int, int, int, int]
    color_player_vision_rays: Tuple[int, int, int, int]
    face_walk: str
    face_wall: str
    face_dead: str
    face_exit: str
    show_ascii_faces: bool
    show_status_ring: bool
    status_ring_ratio: float
    solved_arc_segments: float
    solved_arc_color: Tuple[int, int, int]
    player_face_text_scale: float
    player_heading_line_length: float
    player_heading_line_width: int
    color_player_heading_line: Tuple[int, int, int, int]


class SkinProfileRegistry:
    """
    Parses profiles/skin.yaml and provides skin profile resolution.
    """

    def __init__(self, library_path: Optional[Path] = None) -> None:
        """
        Initializes registry and loads skin library from YAML.
        """
        self._skins: Dict[str, ResolvedSkinProfile] = {}
        root_dir: Path = Path(__file__).resolve().parents[1]
        self.library_path: Path = library_path or (
            root_dir / "profiles" / "skin.yaml"
        )
        self.load_library()

    def get_skin(self, skin_name: str) -> ResolvedSkinProfile:
        """
        Retrieves resolved skin profile or fails fast with clear error.
        """
        if skin_name not in self._skins:
            print(
                f"[Error] Skin profile '{skin_name}' is not defined in "
                f"{self.library_path.name}. Available skins: "
                f"{list(self._skins.keys())}"
            )
            sys.exit(1)
        return self._skins[skin_name]

    def load_library(self) -> None:
        """
        Parses YAML configuration file and caches resolved skin profiles.
        """
        if not self.library_path.exists():
            print(
                f"[Error] Skin library YAML file missing: "
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
                raw_skins: Dict[str, Any] = data.get("profiles", {})
        except Exception as e:
            print(
                f"[Error] Failed to parse {self.library_path.name}: {e}"
            )
            sys.exit(1)

        self._skins.clear()
        for s_name, s_dict in raw_skins.items():
            resolved = self._parse_skin_dict(s_name, s_dict)
            self._skins[s_name] = resolved

    def _parse_skin_dict(
        self,
        s_name: str,
        s_dict: Dict[str, Any]
    ) -> ResolvedSkinProfile:
        """
        Extracts required skin fields and builds ResolvedSkinProfile.
        """
        p_color = s_dict.get("color_player", [240, 180, 50])
        default_arc = [p_color[0], p_color[1], p_color[2], 15]
        default_rays = [p_color[0], p_color[1], p_color[2], 80]
        arc_col_raw = s_dict.get("solved_arc_color", [40, 160, 240])

        return ResolvedSkinProfile(
            skin_name=s_name,
            camera_zoom=float(s_dict.get("camera_zoom", 1.0)),
            color_player=tuple(p_color),
            color_player_text=tuple(
                s_dict.get("color_player_text", [10, 10, 15])
            ),
            color_player_vision_arc=tuple(
                s_dict.get("color_player_vision_arc", default_arc)
            ),
            color_player_vision_rays=tuple(
                s_dict.get("color_player_vision_rays", default_rays)
            ),
            face_walk=str(s_dict.get("face_walk", "o_o")),
            face_wall=str(s_dict.get("face_wall", ">_<")),
            face_dead=str(s_dict.get("face_dead", "T_T")),
            face_exit=str(s_dict.get("face_exit", "^_^")),
            show_ascii_faces=bool(s_dict.get("show_ascii_faces", True)),
            show_status_ring=bool(s_dict.get("show_status_ring", True)),
            status_ring_ratio=float(s_dict.get("status_ring_ratio", 1.0)),
            solved_arc_segments=float(
                s_dict.get("solved_arc_segments", 60.0)
            ),
            solved_arc_color=tuple(arc_col_raw),
            player_face_text_scale=float(
                s_dict.get("player_face_text_scale", 0.80)
            ),
            player_heading_line_length=float(
                s_dict.get("player_heading_line_length", 1.25)
            ),
            player_heading_line_width=int(
                s_dict.get("player_heading_line_width", 1)
            ),
            color_player_heading_line=tuple(
                s_dict.get(
                    "color_player_heading_line", [255, 40, 20, 220]
                )
            )
        )
