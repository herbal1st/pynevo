"""
Candidate emotional state face expression engine.
"""

from typing import Optional
from entities.agent_profile_registry import ResolvedAgentProfile


class PlayerExpress:
    """
    Determines ASCII face strings based on candidate physics states.
    """

    @staticmethod
    def resolve_face(
        has_reached_exit: bool,
        has_collided: bool,
        is_alive: bool,
        profile: Optional[ResolvedAgentProfile] = None
    ) -> str:
        """
        Evaluates active physics flags to select ASCII expression.
        """
        if profile is not None:
            if has_reached_exit:
                return profile.skin.face_exit
            if not is_alive:
                return profile.skin.face_dead
            if has_collided:
                return profile.skin.face_wall
            return profile.skin.face_walk

        if has_reached_exit:
            return "^_^"
        if not is_alive:
            return "T_T"
        if has_collided:
            return ">_<"
        return "o_o"
