"""
Dedicated presenter managing companion AI state, target tracking, & rendering.
"""

import math
from typing import Tuple, Optional
import pygame

import config
from entities.agent_profile_registry import (
    AgentProfileRegistry,
    ResolvedAgentProfile
)
from entities.skin_profile_registry import ResolvedSkinProfile
from entities.agent_factory import AgentFactory
from entities.entity_state import AgentState
from entities.player_controller import PlayerController
from entities.player_profile_registry import ResolvedPlayerProfile
from neural.brain_persistence import BrainPersistence
from neural.network import NeuralNetwork
from perception.spatial_transformer import SpatialTransformer
from core.kinematics.endless_engine import EndlessKinematics
from world.tile_registry import TileRegistry
from world.chunk_manager import ChunkManager
from world.spawn_solver import EndlessSpawnSolver
from world.generation.endless_noise import EndlessNoiseGenerator
from world.endless_facade import (
    EndlessMapDataFacade,
    EndlessPathfinderFacade
)
from visualization.viewports.native.avatar_renderer import (
    ViewportAvatarRenderer
)
from visualization.viewports.native.state_resolver import (
    ViewportFrameState
)
from visualization.vision_renderer import VisionRenderer


class CompanionPresenter:
    """
    Manages companion AI brain loading, dynamic target tracking, & rendering.
    """

    def __init__(
        self,
        player_x: float,
        player_y: float,
        player_heading: float,
        chunk_manager: ChunkManager,
        tile_registry: TileRegistry,
        generator: EndlessNoiseGenerator,
        profile_name: Optional[str] = None
    ) -> None:
        """
        Initializes companion profile, brain persistence, AI state, & views.
        """
        p_agent_name: str = profile_name or getattr(
            config, "ACTIVE_AGENT_PROFILE", "TANK_1"
        )
        self.agent_registry: AgentProfileRegistry = AgentProfileRegistry()
        self.agent_profile: ResolvedAgentProfile = (
            self.agent_registry.get_profile(p_agent_name)
        )
        self.agent_factory: AgentFactory = AgentFactory(
            self.agent_registry, p_agent_name
        )
        self.transformer: SpatialTransformer = (
            self.agent_factory.create_transformer()
        )
        self.kinematics = self.agent_factory.create_kinematics()
        self.network: NeuralNetwork = self.agent_factory.create_network()
        self.persistence: BrainPersistence = BrainPersistence()

        self.persistence.load_brain(
            p_agent_name,
            self.network,
            self.agent_profile,
            context="endless companion",
            verbose=True
        )

        c_spawn_x, c_spawn_y = EndlessSpawnSolver.find_safe_spawn(
            chunk_manager,
            tile_registry,
            generator,
            center_x=player_x - 2.0,
            center_y=player_y,
            diameter_ratio=(
                self.agent_profile.endless_agent_diameter_ratio
            ),
            min_speed_mult=0.1
        )

        self.state: AgentState = AgentState(c_spawn_x, c_spawn_y)
        self.state.heading = player_heading

        self.avatar_renderer: ViewportAvatarRenderer = (
            ViewportAvatarRenderer()
        )
        self.vision_renderer: VisionRenderer = VisionRenderer(
            config.VIRTUAL_WIDTH, config.VIRTUAL_HEIGHT
        )

    def update(
        self,
        player: PlayerController,
        player_profile: ResolvedPlayerProfile,
        chunk_manager: ChunkManager,
        generator: EndlessNoiseGenerator
    ) -> None:
        """
        Calculates dynamic target offset and steps companion AI forward.
        """
        rad: float = player.heading
        cos_a: float = math.cos(rad)
        sin_a: float = math.sin(rad)
        off_x, off_y = player_profile.companion_offset

        world_off_x: float = (off_x * cos_a) - (off_y * sin_a)
        world_off_y: float = (off_x * sin_a) + (off_y * cos_a)

        target_x: float = player.x + world_off_x
        target_y: float = player.y + world_off_y

        c_state = self.state
        map_facade = EndlessMapDataFacade(
            chunk_manager,
            target_x,
            target_y,
            agent_x=c_state.x,
            agent_y=c_state.y
        )
        pathfinder_facade = EndlessPathfinderFacade(
            target_x, target_y
        )

        features = self.transformer.compile_feature_vector(
            c_state.x,
            c_state.y,
            c_state.heading,
            c_state.last_speed_ratio,
            c_state.health,
            map_facade,
            pathfinder_facade,
            candidate_idx=0,
            is_collided=c_state.last_collided,
            is_idle=c_state.last_idle,
            is_healing=c_state.last_healing,
            rot_ratio=c_state.last_rot_ratio
        )

        use_linear: bool = self.agent_profile.use_linear_speed_output
        outputs = self.network.forward(features)[0]

        if use_linear:
            move_eff: float = float(outputs[0]) - float(outputs[1])
            turn_eff: float = float(outputs[3]) - float(outputs[2])
        else:
            net_l: float = float(outputs[0]) - float(outputs[1])
            net_r: float = float(outputs[2]) - float(outputs[3])
            move_eff = (net_r + net_l) / 2.0
            turn_eff = (net_r - net_l) / 2.0

        c_state.heading = EndlessKinematics.apply_rotation(
            c_state.heading,
            turn_eff,
            move_eff,
            self.agent_profile.turn_speed,
            self.agent_profile.profile_style,
            fps=config.FPS
        )

        c_state.x, c_state.y, hit = (
            EndlessKinematics.calculate_forward_step(
                c_state.x,
                c_state.y,
                c_state.heading,
                move_eff,
                self.agent_profile.endless_move_speed,
                self.agent_profile.endless_agent_diameter_ratio,
                chunk_manager,
                tile_registry=generator.tile_registry,
                generator=generator
            )
        )

        c_state.last_collided = hit

    def draw(
        self,
        surface: pygame.Surface,
        focus_x: float,
        focus_y: float,
        center_px: int,
        center_py: int,
        effective_tile_sz: float,
        player: PlayerController,
        player_profile: ResolvedPlayerProfile,
        chunk_manager: ChunkManager
    ) -> None:
        """
        Renders companion vision fan, avatar sprite, & target vector marker.
        """
        c_state = self.state
        comp_screen_x: int = int(
            round(center_px + (c_state.x - focus_x) * effective_tile_sz)
        )
        comp_screen_y: int = int(
            round(center_py + (c_state.y - focus_y) * effective_tile_sz)
        )

        rad: float = player.heading
        cos_a: float = math.cos(rad)
        sin_a: float = math.sin(rad)
        off_x, off_y = player_profile.companion_offset
        target_x: float = player.x + (off_x * cos_a) - (off_y * sin_a)
        target_y: float = player.y + (off_x * sin_a) + (off_y * cos_a)

        map_facade = EndlessMapDataFacade(
            chunk_manager,
            target_x,
            target_y,
            agent_x=c_state.x,
            agent_y=c_state.y
        )

        self.vision_renderer.draw_vision_arc(
            surface,
            0, 0, config.VIRTUAL_WIDTH, config.VIRTUAL_HEIGHT,
            c_state.x, c_state.y, c_state.heading,
            (comp_screen_x, comp_screen_y),
            effective_tile_sz,
            is_camera_centered=True,
            map_data=map_facade,
            radius_ratio=self.agent_profile.endless_agent_radius_ratio
        )

        comp_frame_state = ViewportFrameState(
            cand_idx=0,
            frame_idx=0,
            x=c_state.x,
            y=c_state.y,
            heading=c_state.heading,
            health=c_state.health,
            dist=0,
            hit_wall=c_state.last_collided,
            is_alive=True,
            reached_exit=False,
            speed_ratio=1.0,
            is_idle=False,
            is_healing=False,
            net_delta=0.0,
            face_str=self.agent_profile.skin.face_walk,
            score_val=0,
            radius_ratio=self.agent_profile.endless_agent_radius_ratio,
            skin=self.agent_profile.skin
        )

        self.avatar_renderer.draw_avatar(
            surface,
            (comp_screen_x, comp_screen_y),
            effective_tile_sz,
            comp_frame_state,
            is_selected=True,
            ui_scale=1.0,
            active_step=0
        )

        if getattr(config, "SHOW_TARGET_INDICATOR", True):
            target_screen_x: int = int(
                round(center_px + (target_x - focus_x) * effective_tile_sz)
            )
            target_screen_y: int = int(
                round(center_py + (target_y - focus_y) * effective_tile_sz)
            )

            pygame.draw.line(
                surface,
                (255, 60, 60, 180),
                (comp_screen_x, comp_screen_y),
                (target_screen_x, target_screen_y),
                2
            )

            target_r: int = max(3, int(round(effective_tile_sz * 0.3)))
            pygame.draw.circle(
                surface,
                (255, 60, 60),
                (target_screen_x, target_screen_y),
                target_r,
                2
            )
            pygame.draw.circle(
                surface,
                (255, 220, 80),
                (target_screen_x, target_screen_y),
                2
            )
