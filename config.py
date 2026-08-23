"""
Global configuration settings for PyNevo engine.
"""

from typing import Tuple

# ------ Display & Resolution Settings ------
SCREEN_WIDTH: int = 1280  # pixels
SCREEN_HEIGHT: int = 720  # pixels
VIRTUAL_WIDTH: int = 1280  # pixels
VIRTUAL_HEIGHT: int = 720  # pixels
FPS: int = 60  # hertz
USE_RESIZABLE_WINDOW: bool = True  # toggle
MAX_TEMP_CACHE_SIZE_MB: int = 500  # megabytes limit

# ------ Multi-Candidate Viewport Grid ------
GRID_ROWS: int = 3  # count
GRID_COLS: int = 3  # count

# ------ Active Profile Selectors ------
ACTIVE_AGENT_PROFILE: str = "TANK_1"  # profile
ACTIVE_TRAINING_PROFILE: str = "DEFAULT"  # profile
ACTIVE_MAP_PROFILE: str = "DEFAULT"  # profile

# ------ Active Endless Profile Selectors ------
USE_ENDLESS_MODE: bool = True
ACTIVE_ENDLESS_MAP_PROFILE: str = "DEFAULT"

# ------ Live Winner Mode Defaults ------
LIVE_RUNNER_MAX_STEPS: int = 1000  # steps
LIVE_RUNNER_AUTO_RESET: bool = True  # toggle

# ------ Interactive Scrubber & Playback Defaults ------
DEFAULT_PLAYBACK_SPEED: float = 1.0  # multiplier
TIMELINE_FRAME_JUMP_RATIO: float = 0.025  # ratio
TIMELINE_KEY_REPEAT_DELAY_MS: int = 125  # ms
TIMELINE_KEY_REPEAT_INTERVAL_MS: int = 50  # ms
TIMELINE_UNSELECTED_TICK_ALPHA: int = 80  # alpha
TIMELINE_BLOCK_GENERATION_BAR: bool = True  # toggle

# ------ GUI Element Layout Rectangles ------
LAYOUT_GRID_RECT: Tuple[int, int, int, int] = (20, 20, 800, 600)  # rect
LAYOUT_PANEL_RECT: Tuple[int, int, int, int] = (840, 20, 420, 140)  # rect
LAYOUT_GRAPH_RECT: Tuple[int, int, int, int] = (840, 180, 420, 440)  # rect
LAYOUT_SCRUBBER_RECT: Tuple[int, int, int, int] = (20, 650, 1240, 60)  # rect

# ------ HUD Typography & Element Sizing ------
HUD_PANEL_TITLE_FONT_SIZE: int = 18  # pt
HUD_PANEL_BODY_FONT_SIZE: int = 15  # pt
HUD_HELP_TITLE_FONT_SIZE: int = 18  # pt
HUD_HELP_BODY_FONT_SIZE: int = 15  # pt
HUD_SCRUBBER_BUTTON_HEIGHT: int = 36  # pixels
HUD_SCRUBBER_BUTTON_FONT_SIZE: int = 13  # pt
HUD_SCRUBBER_MARKER_FONT_SIZE: int = 15  # pt
HUD_SCRUBBER_BAR_HEIGHT: int = 16  # pixels
HUD_SCRUBBER_MARKER_WIDTH: int = 6  # pixels
HUD_SCRUBBER_MARKER_HEIGHT: int = 20  # pixels
HUD_GRAPH_SPACING: int = 5  # pixels
HUD_GRAPH_TEXT_SCALE: float = 1.3  # ratio

# ------ Visual Theme & Palette Colors ------
COLOR_BG: Tuple[int, int, int] = (15, 15, 20)  # rgb
COLOR_WALL: Tuple[int, int, int] = (45, 45, 55)  # rgb
COLOR_WALL_BORDER: Tuple[int, int, int] = (45, 45, 55)  # rgb
COLOR_FLOOR: Tuple[int, int, int] = (25, 25, 32)  # rgb
COLOR_FLOOR_BORDER: Tuple[int, int, int] = (25, 25, 32)  # rgb
COLOR_START: Tuple[int, int, int] = (40, 160, 220)  # rgb
COLOR_EXIT: Tuple[int, int, int] = (50, 200, 100)  # rgb
COLOR_PLAYER_HIGHLIGHT: Tuple[int, int, int] = (255, 220, 80)  # rgb

# ------ Health Bar & Status Frame Colors ------
COLOR_HEALTH_FULL: Tuple[int, int, int] = (50, 200, 100)  # rgb
COLOR_HEALTH_MID: Tuple[int, int, int] = (255, 140, 0)  # rgb
COLOR_HEALTH_LOW: Tuple[int, int, int] = (220, 50, 50)  # rgb
COLOR_FRAME_SOLVED: Tuple[int, int, int] = (20, 140, 50)  # rgb
COLOR_FRAME_DEAD: Tuple[int, int, int] = (160, 20, 20)  # rgb

# ------ HUD & Activation Graph Colors ------
COLOR_NODE_INACTIVE: Tuple[int, int, int] = (80, 15, 15)  # rgb
COLOR_NODE_ACTIVE: Tuple[int, int, int] = (255, 140, 0)  # rgb
COLOR_TIMELINE_BAR: Tuple[int, int, int] = (60, 60, 75)  # rgb
COLOR_MARKER: Tuple[int, int, int] = (255, 140, 0)  # rgb
COLOR_BUTTON: Tuple[int, int, int] = (45, 45, 55)  # rgb
COLOR_BUTTON_ACTIVE: Tuple[int, int, int] = (80, 80, 100)  # rgb
