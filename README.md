```
 ____           __  __                                              
/\  _`\        /\ \/\ \                                             
\ \ \L\ \__  __\ \ `\\ \     __   __  __    ___                     
 \ \ ,__/\ \/\ \\ \ , ` \  /'__`\/\ \/\ \  / __`\                   
  \ \ \/\ \ \_\ \\ \ \`\ \/\  __/\ \ \_/ |/\ \L\ \                  
   \ \_\ \/`____ \\ \_\ \_\ \____\\ \___/ \ \____/                  
    \/_/  `/___/> \\/_/\/_/\/____/ \/__/   \/___/                   
             /\___/                                                 
             \/__/                                                  
===============================================================================
                   PYNEVO - SYSTEM SPECIFICATIONS & GUIDE
===============================================================================

[1.0 SYSTEM OVERVIEW]
-------------------------------------------------------------------------------
Core Philosophy: Sovereign Compute, Matrix-Isolated, Zero-Dependency.
Architecture   : PyNevo is a self-contained 2D neuroevolution simulation
                 engine and infinite world visualizer built using vectorized
                 NumPy matrix math, PyBiwis 64-bit integer bitmask map
                 compression (packing 64 grid tiles into single uint64 words,
                 and 16x16 tile chunks into 4 uint64 words), pre-rendered
                 hardware-formatted chunk surface caching (.convert()),
                 spatial infinite chunking with Euclidean circular hysteresis
                 loading, top-left anchor grid snapping for 100% seam-free
                 rendering, 18x18 halo-padded tile sampling with vectorized
                 2x2 vertex corner-smoothing (turning blocky 90-degree
                 strata borders into chamfered regular octagons), fully
                 vectorized multi-octave Simplex and classical Perlin noise
                 terrain generation with fast array-threshold strata mapping
                 (np.digitize), continuous Circle-to-AABB smooth wall physics
                 with Minimum Translation Vector (MTV) tile ejection, physics
                 sub-stepping for anti-tunneling at high velocities
                 (EndlessKinematics), a 3-pass direct-canvas dynamic lighting
                 and 3D hillshading pipeline featuring a 360-degree
                 counterclockwise solar orbit clock (DayNightClock),
                 vectorized height field sampling (ViewportHeightSampler),
                 Tile-Aware base-color additive highlights and mountain
                 shadows (VectorizedHeightShadowEngine), multiplicative
                 day/night ambient tinting (AmbientPaletteResolver),
                 Amanatides-Woo fast voxel grid traversal raycasting,
                 pre-allocated contiguous sensor array caches, flexible
                 steering kinematics (Car, Tank, and Direct Vector
                 profiles), a data-driven profile library (profiles/), a
                 configurable dual-mode 4-neuron motor actuation switch
                 (Task-Space vs Direct Differential Wheels via
                 use_linear_speed_output), physical turning tax (DMG-S)
                 with strict zero-gating, dual-mode orientation compasses
                 (Focus and Peripheral North & Exit with optional Line-of-
                 Sight wall gating), topological BFS GPS path progress
                 sensors (Mono Progress vs Stereo Binocular Progress), a
                 dual-metabolic survival engine (Kinetic Move Heal & Invisible
                 Topological Path Refuel), orthogonal cardinal spawn heading
                 alignment, a multi-tile safe spawn solver
                 (EndlessSpawnSolver), a human player input controller
                 (PlayerController), a decoupled Universal Entity Layer
                 (EntityState, AgentState, and ViewportFrameState) supporting
                 both AI companions and human players, and a dual Pygame
                 visualizer pipeline featuring a 3x3 candidate replay window
                 (AppWindow) and a dedicated full-canvas 1280x720 endless
                 world window (EndlessAppWindow).
Primary Goal   : Train autonomous 2D AI agents to navigate procedural
                 labyrinths and infinite terrain from randomized start tiles
                 to targets using a multi-ray visual fan, orientation
                 compasses, BFS GPS path progress triggers, rotation health
                 taxes, and a physical topological progress refuel loop, while
                 providing an extensible entity foundation for human-player
                 and companion AI interactions in living, atmospheric
                 environments.
Presentation   : Interactive Pygame visualizers featuring dual camera tracking
                 modes (Map-Centered and Camera-Centered) toggleable via
                 TAB or right-clicking viewports, skin-driven camera zoom
                 scales (camera_zoom in profiles/skin.yaml), 1/10x slow-motion
                 to 10x turbo speeds controllable via keys, buttons, or
                 global mouse wheel scrolling, ergonomic keyboard navigation
                 (Left/Right for frame scrubbing, Up/Down for generations or
                 saved brain cycling), active generation block outlines,
                 8-directional candidate selection via Numpad keys,
                 interactive candidate resample shortcut (R key), toggleable
                 cheat-sheet overlays (H key or right-clicking panels),
                 dedicated Live Winner Evaluation Mode (END key) with upfront
                 pre-calculation and on-the-fly brain hot-swapping for testing
                 trained agents on fresh infinite mazes, real-time WASD and
                 Arrow key human player movement across endless spatial
                 worlds (EndlessAppWindow), pre-rendered background surface
                 caching, automatic 16:9 letterboxed screen projection,
                 standardized [0, 1000] fitness scoring, rank-colored
                 timeline tick markers, a dynamic inner shell status ring
                 with a continuous Libra Balance Engine, gyroscopic counter-
                 rotating blue exit arcs, non-blocking upper terminal score
                 cards, real-time neural activation graph heatmaps driven by
                 live network forward passes, dynamic 3D relief terrain,
                 rotating light and shadow fields, and profile-agnostic
                 entity avatar rendering.

[2.0 MEMORY, MAPS & PROCEDURAL GENERATION (PYBIWIS & STRATEGIES)]
-------------------------------------------------------------------------------
Grid Storage   : Rectangular tile grids represented internally as 2D integer
                 matrix arrays. Bounded grid bounds and tile sizes are
                 configured in profiles/map.yaml, while endless spatial
                 worlds use profiles/map_endless.yaml.
PyBiwis Chunks : Isolated in core/bitmask_encoder.py, world/bitmask_encoder.py,
                 and world/chunk.py. Packs 64 grid tiles into single 64-bit
                 unsigned integers (np.uint64). In endless mode, each 16x16
                 tile chunk (256 tiles) is packed into exactly 4 64-bit integer
                 words.
                 Pre-Rendered Surface Caching: Each 16x16 chunk bakes its
                 visual tiles, borders, and corner-smoothed transitions into a
                 cached hardware-formatted image buffer (.convert()) upon
                 creation. During rendering, the engine blits whole chunk
                 images directly instead of drawing thousands of individual
                 tile rectangles, reducing draw overhead by over 95%.
                 Plain Explanation: Think of mapping 64 light switches to a
                 single master number. This compresses level maps into tiny
                 integer arrays for register-speed binary lookups. At the
                 same time, painting the chunk once onto a reusable canvas
                 lets the computer draw entire landscape sections in single
                 fast image sweeps rather than drawing every tile one by one.
Endless Chunking: Managed by world/chunk_manager.py. Uses a spatial dictionary
                 mapping chunk coordinates (cx, cy) to 16x16 tile chunks.
                 Tracks active camera focus and enforces a single-threaded
                 Euclidean circular loading loop with hysteresis:
                 - Load Radius (R_load): Calculated from the screen diagonal
                   distance + 1 safety perimeter chunk.
                 - Unload Radius (R_unload): Chunks beyond R_load + 2 are
                   purged from RAM to cap memory consumption (~7.5 MB RAM)
                   and prevent memory thrashing when stepping back and forth
                   across chunk seams.
                 - Seam Trickle Capping: The circular perimeter prevents
                   massive single-frame chunk bursts when moving diagonally,
                   capping chunk load requests to a smooth trickle of 1 to 3
                   chunks per step.
                 Plain Explanation: Imagine walking through a dark landscape
                 with a circular spotlight. Chunks just outside your screen
                 diagonal light up before you step onto them, while distant
                 chunks far behind you turn off to save memory. Using a
                 round spotlight instead of a square box stops huge bursts
                 of terrain from loading all at once when moving sideways.
Anchor-Based Grid: Managed by visualization/viewports/native/tile_renderer.py.
                 Locks viewport chunk projection to a single top-left anchor
                 chunk (min_cx, min_cy) evaluated each frame. All visible
                 chunks project from this anchor using whole-number tile
                 offsets (anchor + offset * chunk_width).
                 Plain Explanation: Floating-point camera coordinates can
                 cause nearby chunks to round their pixel positions
                 independently, leaving ugly 1-pixel black cracks (seams)
                 between them. Anchoring the grid to the top-left chunk and
                 placing every other chunk using exact integer steps snaps
                 the entire world map together with zero gaps.
Corner Smoothing: Managed by world/generation/endless_noise.py and world/
                 chunk.py. Transforms blocky 90-degree terrain layer borders
                 into chamfered regular octagons using high-speed vectorized
                 NumPy array shifts:
                 - 18x18 Halo Padding: When generating a 16x16 chunk, noise
                   is sampled across an 18x18 grid (a 1-tile perimeter halo
                   x = -1..16, y = -1..16). This allows border tiles (x=0 or
                   x=15) to evaluate their surrounding neighbors across chunk
                   seams without making cross-chunk lookups or creating
                   seam artifacts.
                 - Vectorized 2x2 Vertex Masking: Evaluates 4 cardinal
                   neighbors (Top, Bottom, Left, Right). If two adjacent
                   neighbors match (e.g., Top and Right match), a 29.3%
                   chamfered triangle patch (leg length x = 0.293 * tile_size)
                   is drawn in that corner, turning square tile intersections
                   into neat regular octagons.
                 - Unidirectional Strata Spillage: At 2x2 diagonal cross
                   junctions (checkerboards), higher ground strata automatically
                   overrides lower strata. This prevents miniature inverted
                   patch artifacts while fully preserving 360-degree corner
                   rounding on 1-tile islands.
                 - Dynamic Border Color Resolution: If a neighbor tile has an
                   outline border (border_width_ratio > 0.0), the smoothing
                   patch uses its border_color; otherwise, it uses fill color.
                   This maintains unbroken framing rings around lakes and
                   cliffs.
                 Plain Explanation: Instead of square blocks meeting like
                 stair-steps, the engine looks at 4-tile corners during map
                 creation. If two adjacent sides belong to a higher ground
                 type like Grass, it fills the corner with a small 45-degree
                 triangle. The math is tuned so that when all 4 corners of a
                 tile are trimmed, the square turns into a neat regular
                 octagon, making coasts and hills look smooth and natural.
Deterministic Noise: Managed by utils/noise.py and world/generation/
                 endless_noise.py. Generates continuous 2D Simplex and
                 classical Perlin noise fields seeded by world_seed. Simplex
                 noise uses skewed triangular grids for fast multi-
                 dimensional sampling, while classical Perlin noise evaluates
                 gradient vectors at grid corners using smooth quintic curves.
                 Multi-octave passes combine coarse terrain features with
                 detailed surface textures.
                 Vectorized NumPy Math: Noise field sampling and strata layer
                 classification run across entire 2D matrix arrays in
                 parallel using C-speed NumPy math (np.digitize), eliminating
                 scalar Python loops and accelerating generation by 10x-20x.
                 Data-Driven Strata: Noise values in [0.0, 1.0] are mapped to
                 tile IDs via strata_layers thresholds defined in
                 profiles/map_endless.yaml (e.g., Water ponds, Sand beaches,
                 Grass fields, Thicket bushes, Forest trees, Dirt trails,
                 Rock walls, Bedrock cores, and Snow peaks).
                 Plain Explanation: A mathematical algorithm generates smooth
                 hills and valleys of numbers between 0 and 1. The engine
                 checks height thresholds in parallel to assign terrain
                 types: low spots become lakes, medium spots become grass,
                 and high spots become mountain rocks. Because the math uses
                 a fixed seed and precise permutation hashing, the world
                 generates identically every time.
Safe Spawn Solver: Managed by world/spawn_solver.py (EndlessSpawnSolver).
                 Executes a 2D outward spiral search starting from target
                 coordinates (0, 0) to locate the nearest safe, non-solid,
                 walkable floor tile.
                 Multi-Tile Clearance Footprint: Evaluates the full continuous
                 bounding box footprint based on the entity's diameter ratio.
                 For an entity with diameter_ratio = 4.0 (4 tiles wide), the
                 solver ensures every tile covered by the entity's physical
                 radius (R = 2.0 tiles) is non-solid and passable
                 (speed_multiplier >= min_spawn_speed), guaranteeing large
                 entities never spawn trapped inside solid rock or bedrock.
                 Plain Explanation: When placing an entity in an endless
                 world, the spawner checks the ground in an expanding circle.
                 If the entity is small (0.5 tiles), it needs 1 walkable tile.
                 If the entity is huge (4.0 tiles wide), it checks a 4x4 box
                 of tiles to make sure no rock walls poke into its body when
                 it appears.
Tile Registry  : Managed by world/tile_registry.py. Loads profiles/tiles.yaml
                 defining tile attributes (id, name, solid collision flag,
                 indestructible protection flag, speed_multiplier, base fill
                 color, border_color, and border_width_ratio with a 1-pixel
                 minimum outline safeguard).
100% Solvability & Pocket Filling: All bounded procedural map generators
                 guarantee 100% floor connectivity using a post-generation
                 floodfill pass. The system identifies all open floor
                 regions, retains the largest main connected walking area,
                 and automatically turns any isolated unreachable floor
                 pockets into solid walls. This guarantees every generated
                 bounded map is fully walkable without isolated dead ends.
Fail-Fast Pass : Map generation operates in a single pass. If a map fails to
                 meet floor count or BFS path difficulty bounds, the system
                 fails fast with an explicit CLI error message, preventing
                 empty map fallbacks or invalid training runs.
Snake Corridor : Calculates maximum physically placeable wall capacity while
                 guaranteeing a continuous 1-tile-wide serpentine corridor:
                 Max = ((max_dim - 1) // 2) * min_dim - ((max_dim - 1) // 2)
                 where max_dim and min_dim are inner bounds (width - 2,
                 height - 2). Capping wall counts to this capacity ensures
                 that even at high density settings (e.g. 0.75), maps always
                 retain at least ~50%+ open floor space to breathe.
BFS Distance   : Every generated bounded level builds an O(1) step-distance
                 matrix originating backwards from the exit tile to calculate
                 exact topological path distances and shortest-path turn
                 counts.
                 Plain Explanation: Imagine pouring water at the exit tile—it
                 spreads tile by tile throughout the maze. The step count to
                 reach any floor tile is recorded in a matrix, giving agents an
                 instant measure of topological distance to the goal.
Multi-Point LOS: Pre-computes a 16-ray corner-to-corner Line-of-Sight (LOS)
                 visibility matrix (los_cache) from the exit tile to all
                 walkable floor tiles, ensuring agents receive immediate exit
                 radar signals upon peeking around corners.
Map Strategies : Modularized under core/map_generation/ and world/generation/:
                 - "BRANCHING_WALLS": Organic maze generator facade
                   (branching_walls.py) backed by branching/seed_manager.py
                   and branching/extension_solver.py. Grows continuous wall
                   stems, sharp 90-degree turns, and T-junctions.
                 - "BRANCHING_WALLS_N_ANCHOR": Dynamic N-anchor maze
                   generator. Parses integer N from map name string (e.g.
                   BRANCHING_WALLS_1_ANCHOR, BRANCHING_WALLS_6_ANCHOR). Seeds
                   exactly N border-anchored stems directly off outer border
                   walls into the inner maze, followed by central stems.
                 - "RANDOM": Physics-safe random scatter labyrinth strategy
                   (random_scatter.py). Uses on-demand diagonal validation to
                   place walls up to the snake corridor cap while rejecting
                   isolated diagonal wall touches. Produces organic random
                   labyrinths free of diagonal physics traps.
                 - "PACMAN" & "PACMAN_N_ANCHOR": Arcade pillar arena strategy
                   (pacman_grid.py). Keeps the inner halo free of wall
                   placements to guarantee an unbroken 1-tile outer ring
                   corridor around central wall pillars. Uses permanent
                   diagonal candidate discards to space out pillars cleanly.
                   Optional N-anchor mode (e.g. PACMAN_25_ANCHOR) seeds up to
                   N border stubs before clearing halo tiles for internal
                   pillar growth.
                 - "ENDLESS_NOISE": Endless spatial world generator
                   (endless_noise.py). Evaluates multi-octave 2D Simplex/Perlin
                   noise fields across 16x16 chunk coordinate grids, mapping
                   noise values to data-driven strata layers defined in
                   profiles/map_endless.yaml.

[3.0 SPATIAL PERCEPTION, DUAL COMPASSES & BFS GPS TRIGGERS]
-------------------------------------------------------------------------------
Vision Fan     : Managed by perception/vision_arc.py. Casts probe rays
                 evenly across a field of view (vision_arc_angle) using the
                 Amanatides-Woo Fast Voxel Traversal algorithm. Instead of
                 stepping forward in small arbitrary increments, it computes
                 exact geometric grid-line intersections to step directly from
                 tile boundary to tile boundary. This guarantees 100% boundary
                 accuracy, zero wall penetration, zero corner clipping, and
                 significantly faster execution speed.
                 Plain Explanation: Think of this like light rays emitted from
                 the agent's eyes. Instead of constantly guessing where a wall
                 is by taking tiny baby steps, it calculates the exact distance
                 to the next wall grid line in one mathematical jump.
Exit Lock Radar: Managed by perception/exit_compass.py. Computes goal
                 orientation signals using 5-point inset visibility (center +
                 4 inset corners) to eliminate corner signal flicker. Features
                 an optional Line-of-Sight gating toggle
                 (exit_compass_los_gating in profiles/agent.yaml). When
                 gating is enabled (true), exit signals are hidden behind
                 walls. When disabled (false), exit signals pass through
                 solid walls, providing 360-degree spatial target awareness
                 even when navigating dead ends or backing out of corridors.
                 Always outputs 4 Focus/Peripheral eye channels (EFL, EFR,
                 EPL, EPR) with Euclidean distance scaling across both modes.
                 Eye offset angle is configured via
                 target_compasses_offset_angle.
                 Plain Explanation: Think of this like a target radar. When
                 gating is ON, walls block the signal like solid obstacles.
                 When gating is OFF, it acts like a directional magnetic
                 compass tuned directly to the exit coordinate even through
                 walls.
North Compass  : Managed by perception/cardinal_compass.py. Computes world
                 North alignment signals. Always outputs 4 Focus/Peripheral
                 eye channels (NFL, NFR, NPL, NPR). Eye offset angle is
                 configured via target_compasses_offset_angle.
Cardinal Needles: Managed by perception/cardinal_compass.py. Computes four
                 view-facing 90-degree linear decay needle signals (C-N,
                 C-E, C-S, C-W) providing 4-way orthogonal grid alignment
                 feedback ideal for tile-based mazes.
BFS Path GPS   : Managed by perception/spatial_transformer.py. Computes
                 high-speed O(1) topological BFS step-distance progress using
                 a pre-allocated 2D NumPy array cache (gps_sensor.py) that
                 eliminates runtime Python dictionary lookups:
                 - Mono Mode (use_binocular_gps_compasses: false): Evaluates
                   progress at a single front-nose probe point offset by body
                   radius r_body in direction theta, outputting 2 progress
                   signals (BFS-, BFS+). Turning in place swings the nose
                   probe in an arc, providing instant rotational progress
                   feedback during turns.
                 - Stereo Mode (use_binocular_gps_compasses: true): Evaluates
                   independent progress for Left Eye and Right Eye skin offset
                   points, providing pure differential steering feedback
                   (BFSL-, BFSR-, BFSL+, BFSR+).
                 - Corner Drop-Off Differential: In Stereo Mode, when an
                   agent negotiates a 90-degree corner or T-junction, the
                   inside eye loses contact with the optimal BFS step-
                   distance gradient exactly 1 frame earlier than the
                   outside eye. This 1-frame signal drop-off provides a
                   high-contrast pivot boundary that agents use for
                   decisive corner turns.
                 - Full-Map Unconstrained Navigation: When range_gps_compass
                   is 1.0, progress triggers cover unconstrained map
                   distance, maintaining continuous GPS signals during
                   full-map detours across all profiles.
                 - Smooth Wall Safety Clamp: Probe fallback uses candidate
                   body center tile distance, keeping GPS signals smooth when
                   steering near wall boundaries.
                 Plain Explanation: Imagine GPS navigation for a car. It tracks
                 whether the agent is getting closer to or further from the
                 goal each step. Using two sensors (left and right) lets the
                 agent feel which way the path curves before it even turns.
Proprioception : Tracks 6 core physical state channels in Layer 0:
                 - SPD    : Continuous physical displacement speedometer
                            ratio (delta d / move_speed in [0.0, 1.0]).
                 - HP     : Overall health ratio (0.0 .. 1.0).
                 - DMG-C  : Wall collision impact pulse (1.0 or 0.0).
                 - DMG-I  : Stalling / idle damage pulse (1.0 or 0.0).
                 - DMG-S  : Physical rotation tax ratio (|delta theta| /
                            max_rad_per_frame in [0.0, 1.0]). Suppresses
                            erratic wiggling on straightaways while rewarding
                            clean, straight trajectories.
                 - HEAL   : High-speed kinetic recovery pulse (1.0 or 0.0).
                 Plain Explanation: These 6 channels tell the brain how fast
                 it is moving, its remaining health, and whether it is
                 currently taking wall collision damage, stalling damage, or
                 turning damage, or earning kinetic healing.
Optimal Spawn  : Managed by perception/spawn_heading.py. Spawns candidates
                 strictly aligned to orthogonal cardinal directions (0°, 90°,
                 180°, 270°) facing an open corridor, eliminating diagonal
                 wall tilting on frame 0. Controlled by use_bfs_spawn_heading:
                 - use_bfs_spawn_heading: true -> Aligns spawn heading facing
                   the optimal BFS shortest path neighbor tile to the exit.
                 - use_bfs_spawn_heading: false -> Selects randomly among all
                   open cardinal corridors.
Tactical Turning: While the turning tax (DMG-S) cleanly suppresses wasteful,
                 high-frequency wiggling on open straightaways, tactical
                 micro-steering during turns and at T/X junctions is vital.
                 These small rotational adjustments allow the stereo BFS eye
                 probes to swing across corridor openings, sensing the 1-frame
                 corner drop-off differential needed to confirm correct turn
                 decisions.
Zero-Gating    : Strict sensory zero-gating: if health_spin_dmg_per_frame is
                 0.0, DMG-S is locked to strictly 0.0 for both health deduction
                 and the Layer 0 neural input channel. This ensures 100%
                 sensory isolation and zero interference when turning tax is
                 disabled.
Base Vector    : Standardized input vector dynamically matching active mode:
                 - Mono Mode (_bg0): 20 state channels (6 Proprioceptive +
                   2 BFS GPS + 4 Cardinal + 4 North + 4 Exit) + Vision Rays.
                 - Stereo Mode (_bg1): 22 state channels (6 Proprioceptive +
                   4 BFS GPS + 4 Cardinal + 4 North + 4 Exit) + Vision Rays.
Memory Stream  : Managed by perception/spatial/memory_stacker.py. Stacks past
                 observation frames into a pre-allocated 3D NumPy array cache
                 that eliminates runtime Python heap allocations and dictionary
                 hashing. Configured via memory_frames, clamped to a hard
                 safety cap of 10.
                 - Conceptual Stack: 1 Active Frame (INP) + K past memory
                   frames (M-1 .. M-K), creating 1 + K observation slots.
                 - Single-frame event pulses (DMG-C, DMG-I, DMG-S, HEAL) shift
                   naturally across INP -> M-1 -> M-2, creating a smooth
                   temporal history trail without artificial decay math.
                 Plain Explanation: This acts like short-term memory. It allows
                 the agent's brain to remember what it saw and felt a few steps
                 ago, helping it detect movement trends and past collisions.

[4.0 DATA-DRIVEN PROFILES, NEURAL TOPOLOGY & PERSISTENCE]
-------------------------------------------------------------------------------
Data-Driven YAML: System configurations are decoupled into dedicated profile
                 files under the profiles/ directory:
                 - profiles/agent.yaml    : Defines physical kinematics (Car
                   vs Tank, use_linear_speed_output toggle, speed,
                   agent_diameter_ratio, collision/idle/spin damage, kinetic
                   move heal rate move_heal_per_frame, and invisible
                   topological path refuel rate path_heal_per_frame),
                   perception parameters (exit_compass_los_gating toggle,
                   use_bfs_spawn_heading toggle), neural hidden topology,
                   and references a visual skin.
                 - profiles/player.yaml   : Defines human player profiles
                   (DEFAULT, TANK, CAR), steering mechanics (DIRECT_VECTOR,
                   TANK, CAR), translation speed, turn speed, body
                   diameter_ratio, min_spawn_speed, and skin profile binding.
                 - profiles/skin.yaml     : Defines avatar skins, camera_zoom,
                   color_body, color_text, color_vision_arc, color_vision_rays,
                   ASCII facial expressions, heading lines, status ring
                   rules, and solved arc graphics.
                 - profiles/lighting.yaml : Defines environmental lighting
                   profiles (day_cycle_duration, start_time_ratio,
                   start_light_angle_deg, terrain_steepness, shadow_intensity,
                   highlight_intensity, and ambient_keyframes lists).
                 - profiles/training.yaml : Defines genetic algorithm
                   hyperparameters, generation counts, population sizes, step
                   caps, mutation/elitism rates, and min/max path difficulty
                   ratio bounds.
                 - profiles/map.yaml      : Defines procedural level bounds
                   (width, height), tile pixel sizes, wall densities, and
                   bounded map strategies ("BRANCHING_WALLS", "RANDOM",
                   "PACMAN").
                 - profiles/map_endless.yaml : Defines endless noise map
                   profiles ("SIMPLEX", "PERLIN"), world seeds, noise types,
                   noise scales, octaves, octaves_decay, base tile sizes, and
                   strata_layers lists.
                 - profiles/tiles.yaml    : Defines tile properties (id, name,
                   solid collision flag, indestructible flag, speed_multiplier,
                   fill color, border_color, border_width_ratio).
Master Selectors: Global config (config.py) contains active profile selectors
                 (ACTIVE_AGENT_PROFILE, ACTIVE_PLAYER_PROFILE,
                 ACTIVE_LIGHTING_PROFILE, ACTIVE_TRAINING_PROFILE,
                 ACTIVE_MAP_PROFILE, and ACTIVE_ENDLESS_MAP_PROFILE), master
                 toggles (USE_ENDLESS_MODE), Live Mode defaults
                 (LIVE_RUNNER_MAX_STEPS, LIVE_RUNNER_AUTO_RESET), UI layout
                 bounds, safety limits (MAX_TEMP_CACHE_SIZE_MB), and theme
                 colors.
Fail-Fast      : Dedicated registry modules under entities/ and world/ parse
                 YAML files and validate required parameters at boot with clear
                 CLI error messages.
Player Registry: Managed by entities/player_profile_registry.py. Loads
                 profiles/player.yaml and resolves immutable player profiles
                 (ResolvedPlayerProfile). Operates as an isolated registry,
                 ensuring zero coupling or interference with the MLP agent
                 neuroevolution pipelines.
Lighting Registry: Managed by entities/lighting_profile_registry.py. Loads
                 profiles/lighting.yaml and resolves immutable lighting
                 profiles (ResolvedLightingProfile) governing time progression
                 and terrain hillshading.
Agent Factory  : entities/agent_factory.py instantiates neural networks,
                 spatial transformers, and kinematics engines dynamically
                 matching the active profile's input shape derived from
                 label_resolver.py and output shape (4 outputs).
Neural Topology: Multi-Layer Perceptron (MLP) with dense hidden layers using
                 ReLU activations and Xavier weight initialization.
Dual-Mode Motor: Configured via use_linear_speed_output in profiles/agent.yaml:
                 1. Task-Space Mode (use_linear_speed_output: true):
                    - FWD (Output 0): Forward effort in [0.0, 1.0].
                    - BWD (Output 1): Backward effort in [0.0, 1.0].
                    - S-L (Output 2): Spin Left effort in [0.0, 1.0].
                    - S-R (Output 3): Spin Right effort in [0.0, 1.0].
                    - Net Translation: move_effort = FWD - BWD in [-1.0, 1.0].
                    - Net Steering: turn_effort = S-R - S-L in [-1.0, 1.0].
                    Plain Explanation: Decouples acceleration from steering.
                    A single neuron controls forward motion, allowing the AI
                    to solve mazes in ~30 generations.
                 2. Direct Differential Wheel Mode (use_linear_speed_output: false):
                    - L-FWD (Output 0): Left wheel forward effort in [0.0, 1.0].
                    - L-BWD (Output 1): Left wheel reverse effort in [0.0, 1.0].
                    - R-FWD (Output 2): Right wheel forward effort in [0.0, 1.0].
                    - R-BWD (Output 3): Right wheel reverse effort in [0.0, 1.0].
                    - Net Left Thrust: net_l = L-FWD - L-BWD in [-1.0, 1.0].
                    - Net Right Thrust: net_r = R-FWD - R-BWD in [-1.0, 1.0].
                    - Kinematic Translation: move_effort = (net_r + net_l) / 2.
                    - Kinematic Steering: turn_effort = (net_r - net_l) / 2.
                    Plain Explanation: Provides direct, un-abstracted 2-wheel
                    differential drive control ideal for real-world robotics.
Disk Persistence: neural/brain_persistence.py automatically saves winning
                 candidate weight matrices using signature-tagged filenames
                 (saved_brains/PROFILE_vV_mM_hH_nN_bB_linX.npz), incorporating
                 vision rays (V), memory frames (M), hidden layers (H),
                 neurons (N), compass mode (b1/b0), and actuation mode
                 (lin1 for Task-Space vs lin0 for Direct Wheels).
                 Signature Isolation: Brain persistence signatures automatically
                 isolate task-space brains (_lin1) from direct wheel brains
                 (_lin0). Switching use_linear_speed_output in YAML never
                 overwrites or corrupts saved weights from the other mode!
Brain Discovery: BrainPersistence.discover_saved_brains() scans saved_brains/
                 using a named Regular Expression (Regex) parser. It extracts
                 clean display titles (e.g. "TANK_1", "CAR_2") and embedded
                 topology parameters into cached metadata objects without
                 costly re-scans during visualizer playback.

[5.0 KINEMATICS, HEALTH & DUAL-METABOLIC REFUEL ENGINE]
-------------------------------------------------------------------------------
Kinematics     : Decoupled into core/kinematics/:
                 - profiles.py: Defines "CAR" dynamics (turning rate scaled
                   by movement magnitude abs(move_effort)) and "TANK" dynamics
                   (in-place differential turning incurring idle health drain
                   when standing still).
                 - engine.py: Handles 2D translation and continuous Circle-
                   to-AABB penetration resolution using Minimum Translation
                   Vectors (MTV) for bounded maps. Accepts signed translation
                   effort move_effort in [-1.0, 1.0].
                 - endless_engine.py (EndlessKinematics): Universal physics
                   engine for infinite noise chunk terrain. Provides 2D
                   translation, steering mode math ("DIRECT_VECTOR", "TANK",
                   "CAR"), terrain friction scaling, physics sub-stepping
                   anti-tunneling safeguards, and Circle-to-AABB MTV wall
                   collision ejection directly against ChunkManager.
                 Plain Explanation: The kinematics engine acts like the laws of
                 physics. It calculates where an entity moves when pushed by
                 keys or a neural network, slows it down when walking in deep
                 water or mud, and pushes it smoothly out of rock walls so it
                 slides cleanly along corners instead of getting stuck.
Physics Sub-Stepping: Managed by EndlessKinematics in core/kinematics/
                 endless_engine.py. Prevents high-speed entities from slipping
                 or tunneling through mountain walls. When movement step
                 displacement exceeds MAX_SUB_STEP_DIST (0.20 tiles), the
                 engine divides the frame step into smaller micro-steps (up
                 to 20 sub-steps for high velocities).
                 Plain Explanation: Imagine a fast player moving 4 tiles in a
                 single frame tick. Instead of teleporting through a 2-tile
                 thick rock wall in 1 jump, the engine checks collisions in
                 tiny micro-steps (0.2 tiles each), guaranteeing the player
                 hits the front face of the cliff cleanly and slides along it.
Terrain Friction: Read directly from TileRegistry for the tile underneath the
                 entity's center (X, Y). Scales step distance dynamically:
                 Step Distance = move_effort * base_move_speed * speed_multiplier
                 For example, walking on Grass (1.0) moves at full speed, while
                 walking through Water (0.5) or Mud (0.1) feels physically
                 heavy and sluggish.
Player Controller: Managed by entities/player_controller.py (PlayerController).
                 Translates raw Pygame keyboard input (WASD / Arrow keys) into
                 normalized movement and rotational effort based on the active
                 steering style (DIRECT_VECTOR 8-directional, TANK in-place,
                 or CAR motion-scaled), delegating physical step execution
                 to EndlessKinematics.
Wall Physics   : Both kinematics engines execute robust Circle-to-AABB wall
                 collision checks with zero-distance AABB tile ejection.
                 Entities slide smoothly along walls and corners without
                 tunneling or clipping through obstacles.
Optimal Spawn  : perception/spawn_heading.py automatically aligns candidate
                 spawn headings strictly along orthogonal cardinal directions
                 (0°, 90°, 180°, 270°) facing an open corridor, ensuring zero
                 frame-0 diagonal wall scraping. Driven by use_bfs_spawn_heading.
Dual Metabolism: Candidates start at 100% health (1.0). Health represents the
                 agent's physical stamina and survival timer:
                 - Collision Damage: Wall impacts deduct
                   health_coll_dmg_per_frame and trigger the DMG-C pulse.
                 - Idle Damage: Stalling (speed_ratio < threshold) deducts
                   health_idle_dmg_per_frame and triggers the DMG-I pulse.
                 - Rotation Tax: Physical turning effort deducts
                   health_spin_dmg_per_frame x rot_ratio per tick and carries
                   rot_ratio on the DMG-S channel.
                 - Kinetic Movement Heal (move_heal_per_frame): Cruising at
                   high physical speed (speed_ratio >= heal_speed_threshold)
                   restores move_heal_per_frame per frame, triggering the
                   HEAL pulse on the neural input layer.
                 - Invisible Topological Refuel (path_heal_per_frame): A pure
                   physical/environmental mechanic (invisible to Layer 0).
                   Executed inside CandidateStepPipeline.execute_step(), the
                   system reads stereo BFS progress intensities (BFSL+, BFSR+)
                   and applies an instant health refuel.
                 - Hard Health Cap: Health is strictly capped at 1.0 (100%),
                   preventing agents from building over-healing shield buffers.
Libra Balance  : utils/color_utils.py resolves a continuous Net Delta score:
                 Net Delta = (Damage Sources Count) - (Kinetic Heal Count)
                 - Net Delta = 0.0 (Neutral)  : Yellow highlight ring.
                 - Net Delta > 0.0 (Damage)   : Smoothly blends Yellow -> Red.
                 - Net Delta < 0.0 (Recovery) : Smoothly blends Yellow -> Green.
Gyroscopic Arcs: Solved candidates render concentric dual-radius blue arcs
                 (solved_arc_color). The inner arcs glide along the inside edge
                 of the body skin, while the outer arcs orbit along the outside
                 edge, counter-rotating in opposite directions.
Expressive UI  : entities/entity_express.py maps physical flags to custom
                 ASCII facial expressions defined in YAML:
                 - face_walk (o_o) : Normal walking traversal.
                 - face_wall (>_<) : Active wall impact collision.
                 - face_dead (X_X) : Depleted health state.
                 - face_exit (^_^) : Reached target exit tile.

[5.1 DYNAMIC LIGHTING, 3D HILLSHADING & ATMOSPHERE PIPELINE]
-------------------------------------------------------------------------------
Architecture   : Isolated under world/lighting/. Implements a direct 3-pass
                 screenwide atmosphere and 3D terrain relief engine that
                 operates on the 1280x720 main canvas without altering or
                 re-baking static 16x16 chunk surfaces.
                 Plain Explanation: Instead of drawing shadows directly onto
                 the ground graphics (which is slow), the world draws normally
                 first. Then, light, shadows, and night colors are painted
                 directly over the screen in fast, single-sweep passes.
360° Solar Orbit: Managed by world/lighting/time_clock.py (DayNightClock).
                 Tracks elapsed time and computes a continuous 360-degree
                 counterclockwise solar light vector theta = start_light_angle
                 - (2 * pi * normalized_time). 
                 - Counterclockwise Trajectory: The sun rises in the East,
                   travels across North, sets in the West, and loops smoothly
                   through the night without directional snaps or jumps.
                 - Static Relief Mode: Setting day_cycle_duration to 0.0
                   freezes orbital rotation, locking the light source at
                   start_light_angle_deg (e.g. 135° South-East) for classic,
                   frozen 2D relief maps.
Height Sampler : Managed by world/lighting/viewport_height_sampler.py
                 (ViewportHeightSampler). Queries EndlessNoiseGenerator for
                 the raw, continuous float noise field (0.0 to 1.0) of
                 visible screen tiles plus a 1-tile safety perimeter. Uses
                 C-speed 2D chunk slice array copies (zero Python loops) to
                 sample 14,000+ screen tiles in ~0.02 milliseconds.
                 Plain Explanation: To make mountain slopes look smooth, the
                 light engine samples the underlying continuous height numbers
                 of the landscape rather than flat tile blocks. This makes
                 light and shadow flow naturally down entire mountain faces.
Dual Hillshading: Managed by world/lighting/height_shadow_engine.py
                 (VectorizedHeightShadowEngine). Calculates 2D spatial
                 gradients (dH/dx, dH/dy) and computes the dot product
                 S = dH . L against the active solar light vector:
                 - Tile-Aware Additive Highlights (S > 0): Slopes facing the
                   light source generate an additive highlight surface. In
                   contrast to flat white overlays that wash out contrast,
                   highlights query each tile's natural base RGB color from
                   TileRegistry (grass glints sunny green, water glints cyan,
                   sand glints gold, snow glints white). Blitted directly to
                   the canvas using pygame.BLEND_ADD.
                 - Subtractive Mountain Shadows (S < 0): Slopes facing away
                   from the light source generate a dark alpha shadow mask,
                   darkening valley floors and cliff backs.
Multiplicative Tint: Managed by world/lighting/ambient_palette.py
                 (AmbientPaletteResolver) and AtmosphereOverlayManager.
                 Interpolates pure RGB color tuples across keyframes defined
                 in profiles/lighting.yaml (e.g., [255, 255, 255] for Noon,
                 [40, 50, 110] for Midnight Blue). Blits the ambient surface
                 onto the viewport canvas using pygame.BLEND_MULT.
                 Plain Explanation: Multiplicative blending behaves like
                 real-world physics. Daylight white leaves terrain untouched,
                 while night blue darkens the world while preserving 100% of
                 the underlying grass, water, and rock texture details without
                 creating a foggy, washed-out haze.
The 3-Pass Sequence: Every frame executes in 3 distinct, un-choked steps:
                 1. Pass 1: Draw base terrain chunks and entities (Player/AI).
                 2. Pass 2A & 2B: Blit subtractive mountain shadows and
                    Tile-Aware additive highlights directly onto the canvas.
                 3. Pass 3: Blit time-of-day ambient color using BLEND_MULT.

[6.0 ZERO-ALLOCATION TELEMETRY, CONTIGUOUS TENSORS & LIVE BRAIN REPLAY]
-------------------------------------------------------------------------------
Zero Allocation: Simulation histories write physical telemetry values directly
                 into flat, pre-allocated NumPy memory matrices via
                 TelemetryBundler. Eliminates intermediate Python object
                 allocations and keeps memory consumption flat (~20 MB).
Contiguous Tensors: Generational population neural network parameters are
                 flattened into a master 3D float16 tensor (WeightBundler) of
                 shape (num_generations, population_size, total_params).
                 Casting weight matrices to float16 cuts parameter memory by
                 50% without any loss of visual or numerical accuracy.
Active Truncation: TelemetryBundler automatically tracks the maximum frame
                 step where any candidate was still active, trimming unused
                 trailing frames before serialization.
Uncompressed Doorman: ArchiveBridge writes both telemetry and weight tensors
                 to disk in a single uncompressed .npz archive using np.savez.
                 Writing uncompressed binary dumps takes less than 0.03s,
                 eliminating CPU compression pauses between training and GUI
                 boot. ArchiveBridge enforces MAX_TEMP_CACHE_SIZE_MB safety
                 limits before saving.
Immediate Unlinking: As soon as ArchiveBridge.load_archive() reads arrays
                 into RAM, it unlinks (deletes) the temporary .runtime_cache.npz
                 file from the root directory. Zero temporary files linger on
                 disk during visualizer playback or after shutdown!
Double-Cleanup Protocol: 
                 - 1st Cleanup (Post-Training): Training-side buffers are
                   cleared and gc.collect() is called BEFORE Pygame initializes,
                   preventing RAM bleeding between training and GUI boot.
                 - 2nd Cleanup (Post-Shutdown): Replay tensors are cleared
                   and gc.collect() is called on visualizer exit, returning
                   all memory cleanly to the OS.
True Historical Replay: bridges/playback_presenter.py reconstructs candidate C's
                 exact local neural network for generation G from the flat
                 float16 weight tensor using NeuralNetwork.import_flat_weights().
                 When scrubbing or playing frame T, the activation graph HUD
                 evaluates candidate C's actual local brain on its spatial
                 telemetry slice, displaying 100% historically true hidden
                 layer heatmaps and exact motor decisions!
Live Evaluation : In addition to historical replay, PyNevo features a
                 decoupled Live Winner Solver (bridges/live_winner_runner.py
                 and visualization/live_view_presenter.py). Toggling the END
                 key loads trained winner weights from saved_brains/ into a
                 live NeuralNetwork and spawns the agent on a fresh procedural
                 solvable maze.
                 - Upfront Pre-Calculation: Executes a fast headless pass
                   (~1ms) up to max steps or until solve/death, recording
                   telemetry into a flat single-candidate buffer.
                 - Saved Brain Hot-Swapping: Pressing UP or DOWN arrow keys
                   cycles through all brain files in saved_brains/, hot-swapping
                   network layers and kinematics dynamically.
Decoupled Avatar Renderer: visualization/viewports/native/avatar_renderer.py
                 renders candidate body sprites, ASCII faces, status rings, and
                 directional heading indicators driven by ViewportFrameState.
                 Decouples entity radius (radius_ratio) and skin palette
                 (skin) from profile registries, allowing human players,
                 neural agents, or custom entities to render accurately at
                 their exact configured sizes and colors.

[7.0 NEUROEVOLUTION, UNCONSTRAINED POOLS & SLOT STRATIFICATION]
-------------------------------------------------------------------------------
Unconstrained GA : Population size (population_size) is fully unconstrained
                   (e.g., 9, 25, 50, 100+ candidates per generation) and set in
                   profiles/training.yaml. Training is no longer clamped to
                   the number of viewport grid slots.
Slot Stratification: Managed by visualization/viewports/candidate_mapper.py.
                   Maps candidate indices to grid slots sorted by rank:
                   - Top Rows: Displays the highest-scoring candidates (Winner
                     at Slot 0).
                   - Bottom Rows: Displays the lowest-scoring candidates.
                   - Middle Rows: Stratified sampling across middle pools.
Self-Breeding Rule: evolution/population.py checks whether Parent A == Parent B.
                 When a dominant candidate breeds with itself, the child is
                 forced to undergo Gaussian weight mutation (mutation_rate = 1.0),
                 transforming what would have been an identical clone into a
                 useful local hill-climbing offshoot (Mutated Elite Variant).
Step Pipeline  : bridges/candidate_step_pipeline.py encapsulates candidate
                 ticks (Sensory Input -> Neural Forward -> Kinematics Step ->
                 Health Update -> BFS Recovery -> Direct Array Write).
In-Place GA Pool: evolution/population.py maintains clean memory isolation,
                 instantiating fresh NeuralNetwork instances every generation.
Fitness Math   : Standardized [0, 1000] fitness scoring in evolution/fitness.py.

[7.1 DISPLAY HUD, CLI METRICS & INTERACTIVE CONTROLS]
-------------------------------------------------------------------------------
CLI Progress Table: Real-time console table outputting:
  GEN | TOP | AVG | FIRST | DONE | DIST | FRAME | EXITS | TIME
  - FIRST: ID string of highest-scoring candidate (e.g., # 2).
  - DONE : Percentage of initial BFS path completed (e.g., 100%).
  - DIST : BFS topological step distance from start to exit tile.
  - FRAME: Step tick index when first exit solver arrived (or -).
  - EXITS: Total candidate exit solver count for the generation.
  - TIME : Wall-clock execution time per generation in seconds (e.g., 1.42s).

Interactive Replay & Live Mode Controls:
  - END          : Toggle between Historical Replay Mode and Live Winner Mode.
  - SPACE        : Toggle Timeline Playback (Replay Mode) OR Pause/Resume
                   live physics simulation ticks (Live Mode).
  - ENTER        : Toggle Viewport Zoom Mode (Expands selected candidate view).
  - TAB / R-CLICK: Toggle Camera Tracking Mode (Map-Centered vs Camera-
                   Centered) on any viewport or zoomed view.
  - LEFT / RIGHT : Jump frame scrubber backward / forward. Jump step size is
                   scaled by current playback speed. Continuous scrubbing is
                   supported by holding Left or Right arrow keys.
  - UP / DOWN    : Switch active Generation forward / backward (Replay Mode)
                   OR cycle through saved brain archives in saved_brains/
                   (Live Winner Mode). Single discrete tap per switch.
  - 0 / NUMPAD 0 : Toggle Repeat Mode (Loop All Generations vs Loop Active Gen).
  - PGUP/DN, +/- : Step Playback Speed multiplier up or down through a 19-step
                   scale (1/10x to 10x).
  - WHEEL SCROLL : Step Playback Speed multiplier up or down globally.
  - PERIOD (.)   : Reset Playback Speed back to default 1x.
  - R            : Resample middle-row viewport candidates (Replay Mode) OR
                   generate a fresh maze for the live solver and auto-resume
                   playback (Live Winner Mode).
  - NUMPAD 1..9  : 8-directional grid navigation for candidate selection.
  - NUMPAD 5     : Reset active candidate selection to Candidate #0.
  - H / R-CLICK  : Toggle interactive cheat-sheet overlay panel.
  - ESCAPE       : Exit Visualizer GUI.

Endless World Human Player Controls (USE_ENDLESS_MODE = True):
  - W / A / S / D: Drive human player character (Up, Left, Down, Right).
  - ARROW KEYS   : Alternate directional driving keys.
  - Steering Mode: DIRECT_VECTOR (8-directional auto-facing), TANK (in-place
                   differential turning), or CAR (motion-scaled turning) as
                   configured in profiles/player.yaml.
  - ESCAPE       : Exit Endless World Application.

[8.0 COMPLETE MODULAR CODEBASE STRUCTURE]
-------------------------------------------------------------------------------

PyNevo/
│
├── config.py                       # Global selectors & window layout
├── main.py                         # Application entry point
├── icon.png                        # Application window icon
├── MANUAL.txt                      # Detailed configuration field guide
├── README.md                       # Master system specifications & guide
│
├── profiles/                       # Data-Driven Profile Library
│   ├── agent.yaml                  # Agent kinematics, perception & neural
│   ├── player.yaml                 # Human player steering, speed & skins
│   ├── skin.yaml                   # Visual rendering skins, zoom & colors
│   ├── lighting.yaml               # Environmental lighting & hillshading
│   ├── training.yaml               # Genetic algorithm & training profiles
│   ├── map.yaml                    # Bounded procedural map profiles
│   ├── map_endless.yaml            # Endless noise world profiles
│   └── tiles.yaml                  # Tile properties, colors & wireframes
│
├── utils/                          # Shared Infrastructure
│   ├── math_utils.py               # Angle normalization, spin math & vectors
│   ├── geometry_utils.py           # Continuous ray-AABB clearance math
│   ├── color_utils.py              # Color interpolation & health palettes
│   ├── font_manager.py             # Font caching service by point size
│   ├── surface_utils.py            # Alpha scratchpads & surface scaling
│   └── noise.py                    # Deterministic 2D Simplex/Perlin noise
│
├── world/                          # Spatial World & Endless Engine
│   ├── bitmask_encoder.py          # PyBiwis 64-bit uint64 chunk encoder
│   ├── chunk.py                    # 16x16 chunk container & corner-smoothing
│   ├── chunk_manager.py            # Spatial chunk manager & circular loader
│   ├── tile_registry.py            # O(1) tile property & color registry
│   ├── spawn_solver.py             # Multi-tile safe spawn solver
│   ├── generation/                 # Endless Procedural Generation
│   │   └── endless_noise.py        # Vectorized 18x18 halo noise generator
│   └── lighting/                   # Dynamic Atmosphere & Hillshading
│       ├── time_clock.py           # 360° counterclockwise orbital clock
│       ├── ambient_palette.py      # RGB keyframe palette resolver
│       ├── viewport_height_sampler.py # Vectorized float noise height sampler
│       ├── height_shadow_engine.py # Tile-Aware additive highlights & shadows
│       └── atmosphere_overlay.py   # Direct 3-pass canvas overlay manager
│
├── core/                           # Physics & World Systems
│   ├── bitmask_encoder.py          # PyBiwis 64-bit integer bitmask encoder
│   ├── map_data.py                 # Grid layout with multi-point LOS cache
│   ├── pathfinder.py               # Topological BFS pathfinder coordinator
│   ├── kinematics/                 # Kinematics Subsystem
│   │   ├── profiles.py             # Car and Tank steering profiles
│   │   ├── engine.py               # Bounded candidate kinematics & MTV
│   │   └── endless_engine.py       # Endless kinematics & sub-stepping
│   └── map_generation/             # Procedural Generation Package
│       ├── base_strategy.py        # Abstract generator strategy interface
│       ├── branching_walls.py      # Organic branching wall crawler facade
│       ├── generator.py            # Map generator facade & flood-fill
│       ├── halo_utils.py           # Shared halo geometry & capacity math
│       ├── pacman_grid.py          # Arcade Pacman pillar arena strategy
│       ├── random_scatter.py       # Physics-safe random scatter strategy
│       └── branching/              # Branching Walls Sub-Package
│           ├── seed_manager.py     # Halo candidate seed pool manager
│           └── extension_solver.py # Serpentine wall capacity solver
│
├── entities/                       # Physical Entities & Factory
│   ├── agent_profile_registry.py   # Agent profile registry facade
│   ├── player_profile_registry.py  # Human player YAML profile registry
│   ├── lighting_profile_registry.py# Lighting YAML profile registry
│   ├── player_controller.py        # Human player input dispatcher
│   ├── skin_profile_registry.py    # Visual Skin YAML profile registry
│   ├── training_profile_registry.py# Training YAML profile registry
│   ├── map_profile_registry.py     # Map Geometry YAML profile registry
│   ├── map_endless_profile_registry.py # Endless Map YAML profile registry
│   ├── agent_factory.py            # Profile-driven agent component factory
│   ├── entity_state.py             # Entity & Agent state data containers
│   ├── entity_express.py           # ASCII face expression resolver
│   └── agent_profile/              # Agent Profile Sub-Package
│       ├── profile_model.py        # ResolvedAgentProfile data model
│       └── yaml_parser.py          # Fail-fast YAML loader & validator
│
├── perception/                     # Sensory Perception
│   ├── vision_arc.py               # Amanatides-Woo fast voxel raycaster
│   ├── exit_compass.py             # Stereo exit radar with 5-point LoS
│   ├── cardinal_compass.py         # Binocular North & 4-needle compass
│   ├── spawn_heading.py            # BFS optimal spawn heading generator
│   ├── spatial_transformer.py      # Spatial transformer coordinator facade
│   └── spatial/                    # Spatial Transformer Sub-Package
│       ├── gps_sensor.py           # Pre-allocated 2D array GPS progress
│       ├── memory_stacker.py       # Pre-allocated 3D memory queue cache
│       └── feature_compiler.py     # Single-frame sensory feature compiler
│
├── neural/                         # Neural MLP Engine
│   ├── brain_persistence.py        # Signature-based weight save/load manager
│   ├── weight_initializer.py       # Xavier, He, & Gaussian initializers
│   ├── layers.py                   # Dense layer with batch forward support
│   ├── activations.py              # ReLU, Tanh, & Sigmoid activations
│   └── network.py                  # MLP with batch forward pass
│
├── evolution/                      # Neuroevolution & GA Engine
│   ├── fitness.py                  # Unit-aligned [0, 1000] score evaluator
│   ├── population.py               # Persistent pool & in-place reproduction
│   ├── recorder.py                 # Contiguous tensor recorder & cleanup
│   ├── trainer.py                  # Headless trainer with batch step loop
│   └── operators/                  # Genetic Operators Package
│       ├── selection.py            # Tournament selection strategy
│       ├── crossover.py            # In-place uniform crossover strategy
│       └── mutation.py             # In-place Gaussian noise mutation
│
├── bridges/                        # Pipeline Bridges & Presenters
│   ├── candidate_step_pipeline.py  # Single-frame tick & telemetry pipeline
│   ├── cli_presenter.py            # Real-time console table renderer
│   ├── live_winner_runner.py       # Real-time live winner maze solver
│   ├── playback_presenter.py       # Array-slicing UI view model presenter
│   ├── weight_bundler.py           # Contiguous float16 weight bundler
│   ├── telemetry_bundler.py        # Truncated float32 telemetry bundler
│   └── archive_bridge.py           # Uncompressed disk Doorman
│
└── visualization/                  # GUI Visualizer & Rendering
    ├── map_renderer.py             # Static tilemap surface pre-renderer
    ├── camera_projection.py        # Viewport zoom & camera coordinate math
    ├── vision_renderer.py          # Vision fan & heading line renderer
    ├── help_overlay.py             # Interactive replay shortcut cheat-sheet
    ├── live_help_overlay.py        # Dedicated live solver cheat-sheet overlay
    ├── live_view_presenter.py      # Live winner view presenter
    ├── overlay_panel.py            # Dashboard HUD overlay panel
    ├── input_controller.py         # Pygame keyboard & mouse event dispatcher
    ├── app_window.py               # Historical 3x3 replay visualizer window
    ├── endless_app_window.py       # Full-canvas 1280x720 endless world window
    ├── network_graph/              # Network Activation Graph Sub-Package
    │   ├── label_resolver.py       # Sensory shorthand & output label resolver
    │   ├── layout_engine.py        # Graph geometry & font fitting solver
    │   ├── column_renderer.py      # Activation node heatmap & header
    │   └── graph_facade.py         # Top-level NetworkGraph coordinator
    ├── timeline_scrubber.py        # Transport scrubber facade
    ├── timeline/                   # Timeline Scrubber Sub-Package
    │   └── renderer.py             # Transport buttons & track renderer
    ├── viewport_grid.py            # Viewport grid coordinator facade
    └── viewports/                  # Modular Viewports Sub-Package
        ├── adapter_interface.py    # IViewportAdapter abstract contract
        ├── grid_layout.py          # Spatial R x C layout geometry & bounds
        ├── candidate_mapper.py     # Stratified candidate slot mapper
        ├── native_maze_viewport.py # Native 2D maze viewport facade
        └── native/                 # Native Viewport Sub-Package
            ├── state_resolver.py   # Telemetry row & physics delta resolver
            ├── tile_renderer.py    # Anchor grid snapped tilemap renderer
            ├── avatar_renderer.py  # Body sprite, face, ring & solved arcs
            └── hud_overlay_renderer.py # Health bar, ID/score tags & cards


===============================================================================
[!] PYNEVO | SOVEREIGN NEUROEVOLUTION ENGINE
===============================================================================
Distributed under the PyNevo Source-Available End User License Agreement.
Copyright (c) 2026 herbal1st. All Rights Reserved.
Strictly for personal evaluation, education, private editing, and non-commercial
research.