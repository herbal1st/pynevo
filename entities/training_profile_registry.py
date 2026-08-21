"""
Parses profiles/training.yaml and resolves immutable training profiles.
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
class ResolvedTrainingProfile:
    """
    Immutable container holding training hyperparameters and GA settings.
    """

    profile_name: str
    min_path_difficulty_ratio: float
    max_path_difficulty_ratio: float
    learning_generations: int
    population_size: int
    max_simulation_steps: int
    elitism_ratio: float
    mutation_rate: float
    mutation_scale: float
    dist_to_time_bonus_ratio: float
    lost_hp_score_impact_ratio: float


class TrainingProfileRegistry:
    """
    Parses profiles/training.yaml and provides training profile resolution.
    """

    def __init__(self, library_path: Optional[Path] = None) -> None:
        """
        Initializes registry and loads training library from YAML.
        """
        self._profiles: Dict[str, ResolvedTrainingProfile] = {}
        root_dir: Path = Path(__file__).resolve().parents[1]
        self.library_path: Path = library_path or (
            root_dir / "profiles" / "training.yaml"
        )
        self.load_library()

    def get_profile(self, profile_name: str) -> ResolvedTrainingProfile:
        """
        Retrieves resolved training profile or fails fast with clear error.
        """
        if profile_name not in self._profiles:
            print(
                f"[Error] Training profile '{profile_name}' is not defined "
                f"in {self.library_path.name}. Available profiles: "
                f"{list(self._profiles.keys())}"
            )
            sys.exit(1)
        return self._profiles[profile_name]

    def load_library(self) -> None:
        """
        Parses YAML configuration file and caches resolved training profiles.
        """
        if not self.library_path.exists():
            print(
                f"[Error] Training library YAML file missing: "
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
    ) -> ResolvedTrainingProfile:
        """
        Extracts hyperparameter fields and builds ResolvedTrainingProfile.
        """
        file_name: str = self.library_path.name

        raw_min_diff: float = float(
            self._get_required_val(
                p_dict, "min_path_difficulty_ratio", p_name, file_name
            )
        )
        raw_max_diff: float = float(
            self._get_required_val(
                p_dict, "max_path_difficulty_ratio", p_name, file_name
            )
        )

        min_diff: float = max(0.0, min(1.0, raw_min_diff))
        max_diff: float = max(0.0, min(1.0, raw_max_diff))

        if min_diff > max_diff:
            print(
                f"[Error] Profile '{p_name}' in profiles/{file_name} has "
                f"min_path_difficulty_ratio ({min_diff}) greater than "
                f"max_path_difficulty_ratio ({max_diff})."
            )
            sys.exit(1)

        learn_gens: int = int(
            self._get_required_val(
                p_dict, "learning_generations", p_name, file_name
            )
        )
        pop_sz: int = int(
            self._get_required_val(
                p_dict, "population_size", p_name, file_name
            )
        )
        max_steps: int = int(
            self._get_required_val(
                p_dict, "max_simulation_steps", p_name, file_name
            )
        )
        elitism: float = float(
            self._get_required_val(
                p_dict, "elitism_ratio", p_name, file_name
            )
        )
        mut_rate: float = float(
            self._get_required_val(
                p_dict, "mutation_rate", p_name, file_name
            )
        )
        mut_scale: float = float(
            self._get_required_val(
                p_dict, "mutation_scale", p_name, file_name
            )
        )
        dist_bonus: float = float(
            self._get_required_val(
                p_dict, "dist_to_time_bonus_ratio", p_name, file_name
            )
        )
        hp_impact: float = float(
            self._get_required_val(
                p_dict, "lost_hp_score_impact_ratio", p_name, file_name
            )
        )

        return ResolvedTrainingProfile(
            profile_name=p_name,
            min_path_difficulty_ratio=min_diff,
            max_path_difficulty_ratio=max_diff,
            learning_generations=learn_gens,
            population_size=pop_sz,
            max_simulation_steps=max_steps,
            elitism_ratio=elitism,
            mutation_rate=mut_rate,
            mutation_scale=mut_scale,
            dist_to_time_bonus_ratio=dist_bonus,
            lost_hp_score_impact_ratio=hp_impact
        )

    def _get_required_val(
        self,
        d: Dict[str, Any],
        key_name: str,
        profile_name: str,
        file_name: str
    ) -> Any:
        """
        Enforces key existence in dict or fails fast with an explicit error.
        """
        if key_name not in d:
            print(
                f"[Error] Profile '{profile_name}' in profiles/{file_name} "
                f"is missing required key '{key_name}'."
            )
            sys.exit(1)
        return d[key_name]
