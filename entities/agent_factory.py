"""
Factory for instantiating profile-driven networks, sensors, and kinematics.
"""

import config
from entities.agent_profile_registry import (
    AgentProfileRegistry,
    ResolvedAgentProfile
)
from neural.network import NeuralNetwork
from perception.spatial_transformer import SpatialTransformer
from core.kinematics.engine import CandidateKinematics
from visualization.network_graph.label_resolver import GraphLabelResolver


class AgentFactory:
    """
    Instantiates agent components configured via agent_library.yaml.
    """

    def __init__(
        self,
        registry: AgentProfileRegistry,
        active_profile_name: str = config.ACTIVE_AGENT_PROFILE
    ) -> None:
        """
        Binds registry and active agent profile.
        """
        self.registry: AgentProfileRegistry = registry
        self.profile: ResolvedAgentProfile = registry.get_profile(
            active_profile_name
        )
        self.label_resolver: GraphLabelResolver = GraphLabelResolver()

    def create_network(self) -> NeuralNetwork:
        """
        Instantiates NeuralNetwork matching active profile topology.
        """
        base_channels: int = len(
            self.label_resolver.get_base_shorthand_list(self.profile)
        )
        total_input_frames: int = 1 + max(0, self.profile.memory_frames)
        total_input_channels: int = base_channels * total_input_frames
        return NeuralNetwork(
            input_size=total_input_channels,
            hidden_layers=self.profile.hidden_layers,
            neurons=self.profile.neurons,
            output_size=4
        )

    def create_transformer(self) -> SpatialTransformer:
        """
        Instantiates SpatialTransformer with profile sensory parameters.
        """
        return SpatialTransformer(self.profile)

    def create_kinematics(self) -> CandidateKinematics:
        """
        Instantiates CandidateKinematics with profile physics settings.
        """
        return CandidateKinematics(
            move_speed=self.profile.move_speed,
            turn_speed_dpsec=self.profile.turn_speed,
            player_diameter_ratio=self.profile.player_diameter_ratio,
            fps=config.FPS,
            profile_style=self.profile.profile_style
        )
