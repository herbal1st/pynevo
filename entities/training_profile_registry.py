"""
Parses profiles/training.yaml with programmable multi-metric curriculum mastery gates.
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
class CurriculumStage:
    """
    Specification for a curriculum stage with programmable multi-metric gate conditions.
    """
    name: str
    generations: int
    map_width: int
    map_height: int
    wall_density: float
    max_simulation_steps: int
    mutation_rate: float
    mutation_scale: float
    min_mandatory_generations: int
    consecutive_rounds_needed: int
    min_done_pct: int
    min_exits: int
    min_cntr_pct: int
    min_effc_pct: int
    min_pace_pct: int
    min_expl_pct: int


@dataclass(frozen=True, slots=True)
class ResolvedTrainingProfile:
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
    curriculum: Tuple[CurriculumStage, ...]


class TrainingProfileRegistry:
    def __init__(self, library_path: Optional[Path] = None) -> None:
        self._profiles: Dict[str, ResolvedTrainingProfile] = {}
        root_dir: Path = Path(__file__).resolve().parents[1]
        self.library_path: Path = library_path or (
            root_dir / "profiles" / "training.yaml"
        )
        self.load_library()

    def get_profile(self, profile_name: str) -> ResolvedTrainingProfile:
        if profile_name not in self._profiles:
            print(
                f"[Error] Training profile '{profile_name}' is not defined "
                f"in {self.library_path.name}. Available profiles: "
                f"{list(self._profiles.keys())}"
            )
            sys.exit(1)
        return self._profiles[profile_name]

    def load_library(self) -> None:
        if not self.library_path.exists():
            print(f"[Error] Training library YAML missing: {self.library_path}")
            sys.exit(1)

        try:
            with open(self.library_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                raw_profiles: Dict[str, Any] = data.get("profiles", {})
        except Exception as e:
            print(f"[Error] Failed to parse {self.library_path.name}: {e}")
            sys.exit(1)

        self._profiles.clear()
        for p_name, p_dict in raw_profiles.items():
            resolved = self._parse_profile_dict(p_name, p_dict)
            self._profiles[p_name] = resolved

    def _parse_profile_dict(self, p_name: str, p_dict: Dict[str, Any]) -> ResolvedTrainingProfile:
        file_name: str = self.library_path.name

        pop_sz = int(p_dict.get("population_size", 1024))
        elitism = float(p_dict.get("elitism_ratio", 0.05))
        dist_bonus = float(p_dict.get("dist_to_time_bonus_ratio", 0.4))
        hp_impact = float(p_dict.get("lost_hp_score_impact_ratio", 0.1))

        raw_curriculum = p_dict.get("curriculum", [])
        parsed_stages: List[CurriculumStage] = []

        if raw_curriculum:
            for s in raw_curriculum:
                stage = CurriculumStage(
                    name=str(s.get("name", "STAGE")),
                    generations=int(s.get("generations", 50)),
                    map_width=int(s.get("map_width", 24)),
                    map_height=int(s.get("map_height", 18)),
                    wall_density=float(s.get("wall_density", 0.5)),
                    max_simulation_steps=int(s.get("max_simulation_steps", 2000)),
                    mutation_rate=float(s.get("mutation_rate", 0.25)),
                    mutation_scale=float(s.get("mutation_scale", 0.125)),
                    min_mandatory_generations=int(s.get("min_mandatory_generations", 30)),
                    consecutive_rounds_needed=int(s.get("consecutive_rounds_needed", 3)),
                    min_done_pct=int(s.get("min_done_pct", 100)),
                    min_exits=int(s.get("min_exits", 1)),
                    min_cntr_pct=int(s.get("min_cntr_pct", 0)),
                    min_effc_pct=int(s.get("min_effc_pct", 0)),
                    min_pace_pct=int(s.get("min_pace_pct", 0)),
                    min_expl_pct=int(s.get("min_expl_pct", 0))
                )
                parsed_stages.append(stage)

            total_gens = sum(st.generations for st in parsed_stages)
            max_steps = max(st.max_simulation_steps for st in parsed_stages)
            mut_rate = parsed_stages[0].mutation_rate
            mut_scale = parsed_stages[0].mutation_scale
        else:
            total_gens = int(p_dict.get("learning_generations", 100))
            max_steps = int(p_dict.get("max_simulation_steps", 1000))
            mut_rate = float(p_dict.get("mutation_rate", 0.25))
            mut_scale = float(p_dict.get("mutation_scale", 0.125))

        return ResolvedTrainingProfile(
            profile_name=p_name,
            min_path_difficulty_ratio=0.7,
            max_path_difficulty_ratio=1.0,
            learning_generations=total_gens,
            population_size=pop_sz,
            max_simulation_steps=max_steps,
            elitism_ratio=elitism,
            mutation_rate=mut_rate,
            mutation_scale=mut_scale,
            dist_to_time_bonus_ratio=dist_bonus,
            lost_hp_score_impact_ratio=hp_impact,
            curriculum=tuple(parsed_stages)
        )
