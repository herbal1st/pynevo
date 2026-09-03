"""
Procedural map generator orchestrator validating solvability via BFS.
"""

import re
import random
import sys
from collections import deque
from typing import List, Tuple, Optional, Set

import config
from core.map_data import MapData
from core.pathfinder import BFSPathfinder
from core.map_generation.base_strategy import BaseMapStrategy
from core.map_generation.branching_walls import BranchingWallsStrategy
from core.map_generation.random_scatter import RandomScatterStrategy
from core.map_generation.pacman_grid import PacmanGridStrategy
from entities.map_profile_registry import (
    MapProfileRegistry,
    ResolvedMapProfile
)


class MapGenerator:
    """
    Coordinates level generation strategies and enforces 100% solvability.
    """

    def __init__(
        self,
        map_profile: Optional[ResolvedMapProfile] = None
    ) -> None:
        """
        Initializes map bounds and strategy registry from profile.
        """
        if map_profile is None:
            registry = MapProfileRegistry()
            map_profile = registry.get_profile(config.ACTIVE_MAP_PROFILE)

        self.map_profile: ResolvedMapProfile = map_profile
        self.width: int = map_profile.map_width
        self.height: int = map_profile.map_height
        self.map_type: str = map_profile.map_type

    def generate_solvable_map(
        self,
        min_difficulty_ratio: float = 0.9,
        max_difficulty_ratio: float = 1.0
    ) -> MapData:
        """
        Generates layout in a single pass or fails fast with an error.
        """
        map_data: Optional[MapData] = self._try_generate_map(
            self.map_profile.wall_density,
            min_difficulty_ratio,
            max_difficulty_ratio
        )
        if map_data is not None:
            map_data.encode_bitmask()
            map_data.compute_exit_los_cache()
            return map_data

        print(
            f"[Fatal Error] Map generation failed to satisfy solvability "
            f"and difficulty criteria for map_type '{self.map_type}'."
        )
        sys.exit(1)

    def _resolve_strategy(self) -> BaseMapStrategy:
        """
        Resolves BaseMapStrategy matching normalized map_type string.
        """
        map_str: str = self.map_type.upper().replace("_", " ")

        early_term: float = self.map_profile.stem_early_termination_rate
        min_steps: int = self.map_profile.min_straight_start_steps

        if "PACMAN" in map_str or "PILLAR" in map_str or "ARENA" in map_str:
            num_anchors = (
                self._parse_anchor_count(map_str)
                if "ANCHOR" in map_str else None
            )
            return PacmanGridStrategy(num_anchors=num_anchors)

        if "RANDOM" in map_str:
            return RandomScatterStrategy()

        if "ANCHOR" in map_str:
            num_anchors = self._parse_anchor_count(map_str)
            return BranchingWallsStrategy(
                num_anchors=num_anchors,
                early_term_rate=early_term,
                min_straight_steps=min_steps
            )

        return BranchingWallsStrategy(
            num_anchors=None,
            early_term_rate=early_term,
            min_straight_steps=min_steps
        )

    def _parse_anchor_count(self, map_str: str) -> int:
        """
        Parses integer anchor count N from configuration string.
        """
        match = re.search(r"(\d+)", map_str)
        if match:
            return max(1, int(match.group(1)))
        return 1

    def _try_generate_map(
        self,
        wall_density: float,
        min_difficulty_ratio: float,
        max_difficulty_ratio: float
    ) -> Optional[MapData]:
        """
        Applies strategy and executes pocket filling & BFS exit selection.
        """
        dummy_start: Tuple[int, int] = (1, 1)
        dummy_exit: Tuple[int, int] = (self.width - 2, self.height - 2)
        map_data: MapData = MapData(
            self.width, self.height, dummy_start, dummy_exit
        )

        strategy: BaseMapStrategy = self._resolve_strategy()
        quota_met: bool = strategy.generate_tiles(map_data, wall_density)

        if not quota_met:
            return None

        open_tiles: List[Tuple[int, int]] = [
            (x, y) for y in range(1, self.height - 1)
            for x in range(1, self.width - 1)
            if map_data.is_walkable(x, y)
        ]

        if len(open_tiles) < 10:
            return None

        components: List[Set[Tuple[int, int]]] = (
            self._find_all_connected_components(open_tiles, map_data)
        )
        if not components:
            return None

        main_region: Set[Tuple[int, int]] = max(components, key=len)
        if len(main_region) < 10:
            return None

        for pos in open_tiles:
            if pos not in main_region:
                map_data.set_wall(pos[0], pos[1], True)

        main_open_tiles: List[Tuple[int, int]] = list(main_region)
        start_pos: Tuple[int, int] = random.choice(main_open_tiles)
        map_data.start_pos = start_pos

        dist_from_start: Optional[List[List[int]]] = (
            self._compute_start_bfs(start_pos, map_data)
        )
        if dist_from_start is None:
            return None

        reachable_distances: List[int] = [
            dist_from_start[pos[1]][pos[0]]
            for pos in main_open_tiles
            if pos != start_pos and dist_from_start[pos[1]][pos[0]] < 9999
        ]

        if not reachable_distances:
            return None

        max_bfs_dist: int = max(reachable_distances)
        target_min_dist: int = int(max_bfs_dist * min_difficulty_ratio)
        target_max_dist: int = int(max_bfs_dist * max_difficulty_ratio)

        valid_exits: List[Tuple[int, int]] = [
            pos for pos in main_open_tiles
            if pos != start_pos
            and target_min_dist <= dist_from_start[pos[1]][pos[0]] <= target_max_dist
        ]

        if not valid_exits:
            valid_exits = [
                pos for pos in main_open_tiles
                if pos != start_pos and dist_from_start[pos[1]][pos[0]] < 9999
            ]

        if not valid_exits:
            return None

        map_data.exit_pos = random.choice(valid_exits)

        pathfinder: BFSPathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()

        return map_data

    def _find_all_connected_components(
        self,
        open_tiles: List[Tuple[int, int]],
        map_data: MapData
    ) -> List[Set[Tuple[int, int]]]:
        """
        Groups all open floor tiles into connected region sets.
        """
        unvisited: Set[Tuple[int, int]] = set(open_tiles)
        components: List[Set[Tuple[int, int]]] = []
        cardinal_moves: List[Tuple[int, int]] = [
            (0, -1), (0, 1), (-1, 0), (1, 0)
        ]

        while unvisited:
            start_node: Tuple[int, int] = next(iter(unvisited))
            visited_component: Set[Tuple[int, int]] = {start_node}
            queue: deque[Tuple[int, int]] = deque([start_node])
            unvisited.remove(start_node)

            while queue:
                cx, cy = queue.popleft()
                for dx, dy in cardinal_moves:
                    nx: int = cx + dx
                    ny: int = cy + dy
                    node: Tuple[int, int] = (nx, ny)
                    if node in unvisited:
                        unvisited.remove(node)
                        visited_component.add(node)
                        queue.append(node)

            components.append(visited_component)

        return components

    def _compute_start_bfs(
        self,
        start_pos: Tuple[int, int],
        map_data: MapData
    ) -> Optional[List[List[int]]]:
        """
        Computes BFS step distance grid outward from start_pos.
        """
        unreachable_val: int = 9999
        dist_grid: List[List[int]] = [
            [unreachable_val for _ in range(self.width)]
            for _ in range(self.height)
        ]

        sx, sy = start_pos
        dist_grid[sy][sx] = 0

        queue: deque[Tuple[int, int]] = deque([(sx, sy)])
        cardinal_moves: List[Tuple[int, int]] = [
            (0, -1), (0, 1), (-1, 0), (1, 0)
        ]

        while queue:
            cx, cy = queue.popleft()
            current_dist: int = dist_grid[cy][cx]

            for dx, dy in cardinal_moves:
                nx: int = cx + dx
                ny: int = cy + dy
                if map_data.is_walkable(nx, ny):
                    if dist_grid[ny][nx] == unreachable_val:
                        dist_grid[ny][nx] = current_dist + 1
                        queue.append((nx, ny))

        return dist_grid
