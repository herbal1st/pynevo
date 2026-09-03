"""
Parses profiles/agent.yaml and composes resolved agent profiles.
"""

from pathlib import Path
from typing import Dict, Optional, List
import sys

from entities.skin_profile_registry import SkinProfileRegistry
from entities.agent_profile.profile_model import ResolvedAgentProfile
from entities.agent_profile.yaml_parser import AgentProfileParser


class AgentProfileRegistry:
    """
    Registry facade managing agent profile loading and caching.
    """

    def __init__(
        self,
        library_path: Optional[Path] = None,
        skin_registry: Optional[SkinProfileRegistry] = None
    ) -> None:
        """
        Initializes agent registry and loads agent profiles from YAML.
        """
        self._profiles: Dict[str, ResolvedAgentProfile] = {}
        self.skin_registry: SkinProfileRegistry = (
            skin_registry or SkinProfileRegistry()
        )
        root_dir: Path = Path(__file__).resolve().parents[1]
        self.library_path: Path = library_path or (
            root_dir / "profiles" / "agent.yaml"
        )
        self.load_library()

    @classmethod
    def get_clamped_warning_strings(cls) -> List[str]:
        """
        Forwards memory frame safety warning strings from parser.
        """
        return AgentProfileParser.get_clamped_warning_strings()

    def get_profile(self, profile_name: str) -> ResolvedAgentProfile:
        """
        Retrieves resolved agent profile or fails fast with clear error.
        """
        if profile_name not in self._profiles:
            print(
                f"[Error] Agent profile '{profile_name}' is not defined in "
                f"{self.library_path.name}. Available profiles: "
                f"{list(self._profiles.keys())}"
            )
            sys.exit(1)
        return self._profiles[profile_name]

    def load_library(self) -> None:
        """
        Delegates YAML parsing to AgentProfileParser and caches profiles.
        """
        self._profiles = AgentProfileParser.parse_library(
            self.library_path, self.skin_registry
        )
