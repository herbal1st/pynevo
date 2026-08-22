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
Architecture   : PyNevo is a self-contained 2D neuroevolution
                 simulation engine built using vectorized NumPy matrix math,
                 PyBiwis 64-bit integer bitmask map compression, continuous
                 Circle-to-AABB smooth wall physics with Minimum Translation
                 Vector (MTV) tile ejection, dual steering kinematics (Car
                 and Tank profiles), data-driven YAML profiles (profiles/), a
                 configurable dual-mode 4-neuron motor actuation switch
                 (Task-Space vs Direct Differential Wheels via
                 use_linear_speed_output), physical turning tax (DMG-S) with
                 strict zero-gating, dual-mode orientation compasses (Focus
                 and Peripheral North & Exit with optional Line-of-Sight wall
                 gating via exit_compass_los_gating), topological BFS GPS
                 path progress sensors (Mono Progress vs Stereo Binocular
                 Progress), a dual-metabolic survival engine (Kinetic Move Heal
                 & Invisible Topological Path Refuel), orthogonal cardinal
                 spawn heading alignment (use_bfs_spawn_heading), 5-point
                 multi-corner exit radar, live physical speedometer (SPD),
                 actuation-isolated brain weight persistence (saved_brains/),
                 contiguous float16 generational weight tensors (WeightBundler),
                 active-step truncated float32 telemetry tensors
                 (TelemetryBundler), uncompressed disk serialization with
                 instant unlinking (ArchiveBridge), a Double-Cleanup
                 Protocol for zero RAM bleeding, an unconstrained population
                 candidate mapper with symmetric slot-anchored selection, a
                 decoupled real-time Live Winner Solver (LiveWinnerRunner &
                 LiveViewPresenter), dynamic procedural map strategies
                 (Branching Walls, Labyrinth Random Scatter, Arcade Pacman
                 Grid, and N-Anchor variants), unified BFS floodfill pocket
                 filling, and an interactive Pygame visualizer with a retro
                 K.I.T.T.-style matrix HUD.
Primary Goal   : Train autonomous 2D AI agents to navigate procedural
                 labyrinths from randomized start tiles to exit tiles using a
                 multi-ray visual fan, orientation compasses, BFS GPS path
                 progress triggers, rotation health taxes, and a physical
                 topological progress refuel loop.
Presentation   : Interactive Pygame visualizer featuring dual camera tracking
                 modes (Map-Centered and Player-Centered) toggleable via
                 TAB or right-clicking viewports, 1/10x slow-motion to 10x
                 turbo speeds controllable via keys, buttons, or global mouse
                 wheel scrolling, ergonomic keyboard navigation (Left/Right
                 for frame scrubbing, Up/Down for generations or saved brain
                 cycling), active generation block outlines, 8-directional
                 candidate selection via Numpad keys, interactive candidate
                 resample shortcut (R key), toggleable cheat-sheet overlays
                 (H key or right-clicking panels), dedicated Live Winner
                 Evaluation Mode (END key) with upfront pre-calculation and
                 on-the-fly brain hot-swapping for testing trained agents on
                 fresh infinite mazes, pre-rendered background surface
                 caching with isolated live cache clearing, automatic 16:9
                 letterboxed screen projection, direct telemetry array slicing
                 for lag-free 60+ FPS viewports, standardized [0, 1000]
                 fitness scoring, rank-colored timeline tick markers, a
                 dynamic inner shell status ring with a continuous Libra
                 Balance Engine, gyroscopic counter-rotating blue exit arcs,
                 non-blocking upper terminal score cards, real-time neural
                 activation graph heatmaps driven by live network forward
                 passes, and stateless telemetry slice memory streams.

[2.0 MEMORY, MAPS & PROCEDURAL GENERATION (PYBIWIS & STRATEGIES)]
-------------------------------------------------------------------------------
Grid Storage   : Rectangular tile grids represented internally as 2D
                 integer matrix arrays. Grid bounds and tile sizes are
                 configured in profiles/map.yaml.
PyBiwis Chunks : Isolated in core/bitmask_encoder.py. Packs 64 grid tiles
                 into single 64-bit unsigned integers (np.uint64).
                 Plain Explanation: Think of it like mapping 64 light switches
                 to a single master number. This compresses level maps into
                 tiny integer arrays, allowing register-speed binary lookups
                 and lightweight memory snapshots without bit overflow.
100% Solvability & Pocket Filling: All procedural map generators guarantee 100%
                 floor connectivity using a post-generation floodfill pass.
                 The system identifies all open floor regions, retains the
                 largest main connected walking area, and automatically turns
                 any isolated unreachable floor pockets into solid walls.
                 This guarantees every generated map is fully walkable without
                 isolated dead-end traps or map generation failures.
Fail-Fast Pass : Map generation operates in a single pass. If a map fails to
                 meet floor count or BFS path difficulty bounds, the system
                 fails fast with an explicit CLI error message, preventing
                 empty map fallbacks or invalid training runs.
Snake Corridor : Calculates maximum physically placeable wall capacity while
                 guaranteeing a continuous 1-tile-wide serpentine corridor:
                 Max = ((max_dim - 1) // 2) * min_dim - ((max_dim - 1) // 2)
                 where max_dim and min_dim are inner bounds (width - 2,
                 height - 2). Capping wall counts to this capacity ensures that
                 even at high density settings (e.g. 0.75), maps always retain
                 at least ~50%+ open floor space to breathe.
BFS Distance   : Every generated level builds an O(1) step-distance matrix
                 originating backwards from the exit tile to calculate exact
                 topological path distances and shortest-path turn counts.
                 Plain Explanation: Imagine pouring water at the exit tile—it
                 spreads tile by tile throughout the maze. The step count to
                 reach any floor tile is recorded in a matrix, giving agents an
                 instant measure of topological distance to the goal.
Multi-Point LOS: Pre-computes a 16-ray corner-to-corner Line-of-Sight (LOS)
                 visibility matrix (los_cache) from the exit tile to all
                 walkable floor tiles, ensuring agents receive immediate exit
                 radar signals upon peeking around corners.
Map Strategies : Modularized under core/map_generation/:
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
                   (pacman_grid.py). Keeps the inner halo free of wall placements
                   to guarantee an unbroken 1-tile outer ring corridor around
                   central wall pillars. Uses permanent diagonal candidate
                   discards to space out pillars cleanly. Optional N-anchor
                   mode (e.g. PACMAN_25_ANCHOR) seeds up to N border stubs
                   before clearing halo tiles for internal pillar growth.

[3.0 SPATIAL PERCEPTION, DUAL COMPASSES & BFS GPS TRIGGERS]
-------------------------------------------------------------------------------
Vision Fan     : Managed by perception/vision_arc.py. Casts probe rays
                 evenly across a field of view (vision_arc_angle) using
                 exact DDA grid-line intersection math for 100% boundary
                 accuracy with zero wall penetration. Rays measure distance to
                 the nearest wall tile border from 0.0 (open space) up to 1.0
                 (point-blank wall contact).
                 Plain Explanation: Think of this like light rays emitted from
                 the agent's eyes. They measure how close a wall is in front of
                 and beside the agent.
Exit Lock Radar: Managed by perception/exit_compass.py. Computes goal
                 orientation signals using 5-point inset visibility (center +
                 4 inset corners) to eliminate corner signal flicker. Features
                 an optional Line-of-Sight gating toggle (exit_compass_los_gating
                 in profiles/agent.yaml). When gating is enabled (true), exit
                 signals are hidden behind walls. When disabled (false), exit
                 signals pass through solid walls, providing 360-degree spatial
                 target awareness even when navigating dead ends or backing out
                 of corridors. Always outputs 4 Focus/Peripheral eye channels
                 (EFL, EFR, EPL, EPR) with Euclidean distance scaling across
                 both modes. Eye offset angle is configured via
                 target_compasses_offset_angle.
                 Plain Explanation: Think of this like a target radar. When
                 gating is ON, walls block the signal like solid obstacles.
                 When gating is OFF, it acts like a directional magnetic compass
                 tuned directly to the exit coordinate even through walls.
North Compass  : Managed by perception/cardinal_compass.py. Computes world
                 North alignment signals. Always outputs 4 Focus/Peripheral eye
                 channels (NFL, NFR, NPL, NPR).
                 Eye offset angle is configured via
                 target_compasses_offset_angle.
Cardinal Needles: Managed by perception/cardinal_compass.py. Computes four
                 view-facing 90-degree linear decay needle signals (C-N,
                 C-E, C-S, C-W) providing 4-way orthogonal grid alignment
                 feedback ideal for tile-based mazes.
BFS Path GPS   : Managed by perception/spatial_transformer.py. Computes
                 high-speed O(1) topological BFS step-distance progress:
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
Proprioception : Tracks 6 core physical state channels in Layer 0:
                 - SPD    : Continuous physical displacement speedometer ratio
                            (delta d / move_speed in [0.0, 1.0]).
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
Memory Stream  : memory_frames configures temporal observation stacking
                 clamped to a hard safety cap of 10.
                 - Conceptual Stack: 1 Active Frame (INP) + K past memory
                   frames (M-1 .. M-K), creating 1 + K observation slots.
                 - Single-frame event pulses (DMG-C, DMG-I, DMG-S, HEAL) shift
                   naturally across INP -> M-1 -> M-2, creating a smooth
                   temporal history trail without artificial decay math.

[4.0 DATA-DRIVEN PROFILES, NEURAL TOPOLOGY & PERSISTENCE]
-------------------------------------------------------------------------------
Data-Driven YAML: System configurations are decoupled into dedicated profile
                 files under the profiles/ directory:
                 - profiles/agent.yaml    : Defines physical kinematics (Car
                   vs Tank, use_linear_speed_output toggle, speed, diameter,
                   collision/idle/spin damage, kinetic move heal rate
                   move_heal_per_frame, and invisible topological path refuel
                   rate path_heal_per_frame), perception parameters
                   (exit_compass_los_gating toggle, use_bfs_spawn_heading
                   toggle), neural hidden topology, and references a visual skin.
                 - profiles/skin.yaml     : Defines avatar skins, colors,
                   ASCII facial expressions, heading lines, status ring rules,
                   and solved arc graphics.
                 - profiles/training.yaml : Defines genetic algorithm
                   hyperparameters, generation counts, population sizes, step
                   caps, mutation/elitism rates, and min/max path difficulty
                   ratio bounds (min_path_difficulty_ratio &
                   max_path_difficulty_ratio).
                 - profiles/map.yaml      : Defines procedural level bounds
                   (width, height), tile pixel sizes, wall densities, and
                   map strategies ("BRANCHING_WALLS", "RANDOM", "PACMAN").
Master Selectors: Global config (config.py) contains active profile selectors
                 (ACTIVE_AGENT_PROFILE, ACTIVE_TRAINING_PROFILE, and
                 ACTIVE_MAP_PROFILE), Live Mode defaults (LIVE_RUNNER_MAX_STEPS,
                 LIVE_RUNNER_AUTO_RESET), UI layout bounds, safety limits
                 (MAX_TEMP_CACHE_SIZE_MB), and theme colors.
Fail-Fast      : Dedicated registry modules under entities/ parse YAML files
                 and validate required parameters at boot with clear CLI
                 error messages.
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
                    differential drive control ideal for real-world robotics
                    like the Arduino Alvik.
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
                 topology parameters (rays, memory frames, layers, neurons,
                 compass, actuation) into cached metadata objects without
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
                   Vectors (MTV). Accepts signed translation effort move_effort
                   in [-1.0, 1.0]. Body radius is calculated as r_body = 0.5 *
                   player_diameter_ratio (0.25 tiles), matching the visual
                   body sprite 1-to-1.
Wall Physics   : core/kinematics/engine.py executes robust Circle-to-AABB
                 wall collision checks with zero-distance AABB tile ejection
                 and strict outer map border clamping. Candidates slide
                 smoothly along walls and corners without tunneling or
                 walking past outer map boundaries.
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
                   and applies an instant health refuel:
                   Refuel = 0.5 * path_heal_per_frame * (BFSL+ + BFSR+)
                   Candidates driving along the optimal BFS path constantly
                   refuel their health tanks, keeping them alive and healthy.
                   Candidates standing still or circling in dead-ends drain
                   health and die rapidly.
                 - Hard Health Cap: Health is strictly capped at 1.0 (100%),
                   preventing agents from building over-healing shield buffers.
Libra Balance  : utils/color_utils.py resolves a continuous Net Delta score:
                 Net Delta = (Damage Sources Count) - (Kinetic Heal Count)
                 - Net Delta = 0.0 (Neutral)  : Yellow highlight ring.
                 - Net Delta > 0.0 (Damage)   : Smoothly blends Yellow -> Red.
                 - Net Delta < 0.0 (Recovery) : Smoothly blends Yellow -> Green.
                 Plain Explanation: Like a balanced scale, active wall impacts,
                 stalling, and turning damage push the status ring toward Red,
                 while high-speed kinetic healing pulls it toward Green.
Gyroscopic Arcs: Solved candidates render concentric dual-radius blue arcs
                 (solved_arc_color). The inner arcs glide along the inside edge
                 of the body skin, while the outer arcs orbit along the outside
                 edge, counter-rotating in opposite directions. Rotation
                 angles derive from live timeline steps via calculate_spin_angle,
                 ensuring continuous rotation during scrubber playback.
Expressive UI  : entities/player_express.py maps physical flags to custom
                 ASCII facial expressions defined in YAML:
                 - face_walk (o_o) : Normal walking traversal.
                 - face_wall (>_<) : Active wall impact collision.
                 - face_dead (X_X) : Depleted health state.
                 - face_exit (^_^) : Reached target exit tile.

[6.0 ZERO-ALLOCATION TELEMETRY, CONTIGUOUS TENSORS & LIVE BRAIN REPLAY]
-------------------------------------------------------------------------------
Zero Allocation: Simulation histories write physical telemetry values (x, y,
                 heading, health, distance, wall hits) directly into flat,
                 pre-allocated NumPy memory matrices via TelemetryBundler.
                 Eliminates intermediate Python object allocations and keeps
                 memory consumption completely flat (~20 MB).
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
                 - Upfront Pre-Calculation: When a fresh maze is generated,
                   the solver executes a fast headless pass (~1ms) up to max
                   steps or until solve/death, recording telemetry into a
                   flat single-candidate buffer.
                 - Timeline Scrubbing: Pre-calculation gives full interactive
                   scrubbing across all frames (0 to total run steps). Dragging
                   or stepping the scrubber immediately updates candidate
                   position, vision fan, and neural heatmaps for any frame.
                 - Auto-Pause & Reset: Reaching the final step automatically
                   pauses playback and displays the terminal scorecard card.
                   Pressing R generates a fresh maze and auto-resumes play.
                 - Saved Brain Hot-Swapping: Pressing UP or DOWN arrow keys
                   cycles through all brain files in saved_brains/. The runner
                   inspects the brain signature, hot-swaps network layers and
                   kinematics dynamically matching the target brain's exact
                   topology, generates a fresh maze, and auto-resumes play.
                 - HUD Title Display: Displays clean brain titles (e.g.
                   "SELECTED : TANK_1") in the top-right telemetry dashboard,
                   replacing generic candidate slot numbers.
Dynamic Graph HUD: visualization/network_graph/graph_facade.py derives
                 base_channels dynamically from len(input_labels) using
                 label_resolver.py. The node layout, slot heights, and text
                 labels adapt 1-to-1 on-the-fly, dynamically displaying
                 FWD/BWD/S-L/S-R or L-FWD/L-BWD/R-FWD/R-BWD matching the
                 profile's use_linear_speed_output setting.
Pre-Rendering  : visualization/map_renderer.py pre-renders static maze tile
                 graphics ONCE into cached background surfaces for 60 FPS
                 performance.
Camera Math    : visualization/camera_projection.py converts world tile
                 coordinates into viewport pixel coordinates for both
                 Map-Centered and Player-Centered tracking modes using
                 config.PLAYER_CAMERA_ZOOM as the clean single ground truth.
Canvas Projection: visualization/app_window.py renders all UI elements onto
                 a fixed 1280x720 virtual canvas, smoothly scaled to fit any
                 screen resolution with automatic letterboxing/pillarboxing.
Vision Drawing : visualization/vision_renderer.py draws semi-transparent
                 vision fan polygons and heading lines onto alpha scratchpads.
Input Controller: visualization/input_controller.py dispatches mouse clicks,
                 mouse wheel scroll events, and keyboard shortcuts to control
                 UI viewports and transport scrubbing.

[7.0 NEUROEVOLUTION, UNCONSTRAINED POOLS & SLOT STRATIFICATION]
-------------------------------------------------------------------------------
Unconstrained GA : Population size (population_size) is fully unconstrained
                   (e.g., 9, 25, 50, 100+ candidates per generation) and set in
                   profiles/training.yaml. Training is no longer clamped to
                   the number of viewport grid slots.
Slot Stratification: Managed by visualization/viewports/candidate_mapper.py.
                   Plain Explanation: Think of it like an honor roll display.
                   The top row always showcases the highest-performing
                   candidates, the bottom row shows struggling candidates,
                   and middle slots sample evenly from stratified success-rate
                   middle pools.
                   - Top Rows: Displays the highest-scoring candidates in
                     descending rank order (Slot 0 = True Winner).
                   - Bottom Rows: Displays the lowest-scoring candidates (Slot
                     R-1 = Absolute Worst).
                   - Middle Rows: Symmetric 1/2 row logic restricts random
                     stratum sampling to 1 row for odd screen rows (R > 2) and
                     2 rows for even screen rows (R > 2), centered cleanly in
                     the middle of the viewport grid.
Slot Selection : visualization/viewport_grid.py anchors user selection to
                 grid cell slot indices (0 .. R x C - 1) rather than
                 candidate ID numbers. When switching generations in both
                 multiscreen and fullscreen zoomed views, the yellow selection
                 frame stays locked to the same viewport grid cell, dynamically
                 displaying whichever candidate occupies that slot.
Interactive Refresh: Pressing the R key increments a seed offset counter to
                 instantly re-sample fresh candidates for middle-row viewports,
                 allowing on-demand candidate exploration without losing the
                 active selection or changing generations.
Self-Breeding Rule: evolution/population.py checks whether Parent A == Parent B.
                 When a dominant candidate breeds with itself, the child is
                 forced to undergo Gaussian weight mutation (mutation_rate = 1.0),
                 transforming what would have been an identical clone into a
                 useful local hill-climbing offshoot (Mutated Elite Variant).
Step Pipeline  : bridges/candidate_step_pipeline.py encapsulates candidate
                 ticks (Sensory Input -> Neural Forward -> Kinematics Step ->
                 Health Update -> BFS Recovery -> Direct Array Write).
In-Place GA Pool: evolution/population.py maintains clean memory isolation,
                 instantiating fresh NeuralNetwork instances every generation
                 for elites and offspring to prevent array reference aliasing.
Evolution Ops  : Modularized under evolution/operators/:
                 - selection.py: Tournament selection selecting top candidates.
                 - crossover.py: In-place uniform crossover strategy.
                 - mutation.py: In-place Gaussian noise mutation.
Fitness Math   : Standardized [0, 1000] fitness scoring in evolution/fitness.py:
                 - Configurable Re-balancing: Split by dist_to_time_bonus_ratio
                   (R): Distance Weight = 1000 x R, Time Weight = 1000 x (1-R).
                 - Path Progress: Distance score equals W_dist x (Progress / L_min).
                 - Time Bonus: Time score equals W_time x (Saved Frames / Max
                   Saved Frames) awarded strictly upon reaching exit tile.
                 - Health Weighting: Final raw score is scaled by health:
                   Final Score = Raw Score x [(1 - Impact) + (Impact x HP)].

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

Interactive Controls:
  - END          : Toggle between Historical Replay Mode and Live Winner Mode.
  - SPACE        : Toggle Timeline Playback (Replay Mode) OR Pause/Resume
                   live physics simulation ticks (Live Mode).
  - ENTER        : Toggle Viewport Zoom Mode (Expands selected candidate view).
  - TAB / R-CLICK: Toggle Camera Tracking Mode (Map-Centered vs Player-
                   Centered) on any viewport or zoomed view.
  - LEFT / RIGHT : Jump frame scrubber backward / forward. Jump step size is
                   scaled by current playback speed (TIMELINE_FRAME_JUMP_RATIO x
                   playback_speed) for consistent scrubbing feel. Continuous
                   scrubbing is supported by holding Left or Right arrow keys.
  - UP / DOWN    : Switch active Generation forward / backward (Replay Mode)
                   OR cycle through saved brain archives in saved_brains/
                   (Live Winner Mode). Single discrete tap per switch.
  - 0 / NUMPAD 0 : Toggle Repeat Mode (Loop All Generations vs Loop Active Gen).
  - PGUP/DN, +/- : Step Playback Speed multiplier up or down through a 19-step
                   granular linear scale (1/10x, 1/9x, 1/8x, 1/7x, 1/6x,
                   1/5x, 1/4x, 1/3x, 1/2x, 1x, 2x, 3x, 4x, 5x, 6x, 7x, 8x,
                   9x, 10x).
  - WHEEL SCROLL : Step Playback Speed multiplier up (scroll forward) or
                   down (scroll backward) globally anywhere in visualizer.
  - PERIOD (.)   : Reset Playback Speed back to default 1x.
  - R            : Resample middle-row viewport candidates (Replay Mode) OR
                   generate a fresh maze for the live solver and auto-resume
                   playback (Live Winner Mode).
  - NUMPAD 1..9  : 8-directional grid navigation for candidate viewport selection
                   (7/9/1/3 diagonal, 8/2/4/6 cardinal).
  - NUMPAD 5     : Reset active candidate selection to Candidate #0.
  - H / R-CLICK  : Toggle interactive cheat-sheet overlay panel (HelpOverlay
                   in Replay Mode, LiveHelpOverlay in Live Mode).
  - ESCAPE       : Exit Visualizer GUI.
  - Left Mouse   : Click candidate viewport to select agent; double-click to zoom;
                   click or click-and-drag timeline bars to scrub frames or
                   generations continuously.
  - Right Mouse  : Right-click viewport to toggle tracking mode; right-click
                   graph panel to toggle help overlay; right-click Speed
                   Button to step playback speed downward.

Color Indicators & Scrubber Modes:
  - Terminal Cards : High-contrast score cards rendered in the upper 25% of the
                     sub-viewport upon candidate solve (Green) or death (Red),
                     leaving central agents and spinning arcs 100% visible.
                     Suppressed in Live Winner Mode for clean navigation.
  - Generation Bar : Renders continuous rectangular blocks when
                     TIMELINE_BLOCK_GENERATION_BAR: true. Block colors are
                     normalized relative to the session's peak solver count
                     (Red -> Orange -> Yellow -> Green). Unsolved generations
                     show default grey track. Active generation renders a
                     centered 2px black outline box.
  - Frame Scrubber : Exit solve tick marks color-graded by candidate arrival
                     speed rank (Green for fastest solver down to Red for slower
                     solvers). Selected candidate tick retains 100% opacity,
                     while unselected ticks render semi-transparently.
  - Viewport Borders: Solved (Green) / Dead (Red) status frames render as an
                     inner 3px border along the outer edge. The Yellow selection
                     frame renders on top with 1px stroke, allowing the inner
                     2px status border to peek out clearly.
  - Viewport Scores: Candidate raw score renders as an integer in the lower-right
                     corner of each viewport (e.g., 1000 or 823).

Telemetry HUD    : visualization/overlay_panel.py displays active agent stats,
                   integer frame step count (e.g. 956/1000), generation index,
                   top score, and winner callout in a clear two-column dashboard.
                   In Live Winner Mode, displays clean display titles (e.g.
                   "SELECTED : TANK_1") in place of generic candidate numbers.
Graph & Help HUD : Bounded to the bottom-right panel (LAYOUT_GRAPH_RECT):
                   - Activation Sub-Package (visualization/network_graph/):
                     Decoupled into label_resolver, layout_engine, column_renderer,
                     and graph_facade. Displays real-time neural activations
                     as dynamic rectangular heatmaps (dark red to bright orange).
                     Uses a unified fractional column unit layout
                     (U_total = 1.0 + M + H + 0.5 + 1.0) where standard
                     node columns take 1.0 unit width and the output column
                     takes 0.5 unit width.
                     Inter-column gaps are uniformly spaced via HUD_GRAPH_SPACING.
                     Font sizing automatically adapts via _fit_font_size and
                     HUD_GRAPH_TEXT_SCALE. Displays uniform headers (INP,
                     M-1..M-k, H-1..H-3, OUT), observation shorthands
                     (-120°..+120°, SPD, HP, DMG-C, DMG-I, DMG-S, HEAL,
                     BFSL-..BFSR+, C-N..C-W, NFL..NPR, EFL..EPR), and
                     4-character semantic output labels matching active actuation
                     mode (FWD/BWD/S-L/S-R or L-FWD/L-BWD/R-FWD/R-BWD). Rendered
                     node heatmaps represent candidate C's exact 100% true
                     historical or real-time live candidate neural activations.
                   - Cheat-Sheet Overlays: Replaces graph when toggled with H
                     key or right-clicking the graph panel, displaying a
                     tailored two-column key legend (help_overlay.py for
                     Replay Mode, live_help_overlay.py for Live Mode).

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
│   ├── skin.yaml                   # Visual rendering skins & colors
│   ├── training.yaml               # Genetic algorithm & training profiles
│   └── map.yaml                    # Procedural level geometry & map profiles
│
├── utils/                          # Shared Infrastructure
│   ├── math_utils.py               # Angle normalization, spin math & vectors
│   ├── geometry_utils.py           # Continuous ray-AABB clearance math
│   ├── color_utils.py              # Color interpolation & health palettes
│   ├── font_manager.py             # Font caching service by point size
│   └── surface_utils.py            # Alpha scratchpads & surface scaling
│
├── core/                           # Physics & World Systems
│   ├── bitmask_encoder.py          # PyBiwis 64-bit integer bitmask encoder
│   ├── map_data.py                 # Grid layout with multi-point LOS cache
│   ├── pathfinder.py               # Topological BFS pathfinder coordinator
│   ├── kinematics/                 # Kinematics Subsystem
│   │   ├── profiles.py             # Car and Tank steering profiles
│   │   └── engine.py               # Candidate kinematics & MTV collision
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
│   ├── skin_profile_registry.py    # Visual Skin YAML profile registry
│   ├── training_profile_registry.py# Training YAML profile registry
│   ├── map_profile_registry.py     # Map Geometry YAML profile registry
│   ├── agent_factory.py            # Profile-driven agent component factory
│   ├── player_state.py             # Candidate status data container
│   ├── player_express.py           # ASCII face expression resolver
│   └── agent_profile/              # Agent Profile Sub-Package
│       ├── profile_model.py        # ResolvedAgentProfile data model
│       └── yaml_parser.py          # Fail-fast YAML loader & validator
│
├── perception/                     # Sensory Perception
│   ├── vision_arc.py               # Pure DDA continuous raycaster
│   ├── exit_compass.py             # Stereo exit radar with 5-point LoS
│   ├── cardinal_compass.py         # Binocular North & 4-needle cardinal compass
│   ├── spawn_heading.py            # BFS optimal spawn heading generator
│   ├── spatial_transformer.py      # Spatial transformer coordinator facade
│   └── spatial/                    # Spatial Transformer Sub-Package
│       ├── gps_sensor.py           # Topological BFS distance & GPS channels
│       ├── memory_stacker.py       # Temporal observation stacking queues
│       └── feature_compiler.py     # Single-frame sensory feature compiler
│
├── neural/                         # Neural MLP Engine
│   ├── brain_persistence.py        # Signature-based weight save/load manager
│   ├── weight_initializer.py       # Xavier, He, & Gaussian initializers
│   ├── layers.py                   # Dense layer with batch forward support & flat I/O
│   ├── activations.py              # ReLU, Tanh, & Sigmoid activations
│   └── network.py                  # MLP with batch forward pass & flat weight I/O
│
├── evolution/                      # Neuroevolution & GA Engine
│   ├── fitness.py                  # Unit-aligned [0, 1000] score evaluator
│   ├── population.py               # Persistent pool & in-place reproduction
│   ├── recorder.py                 # Contiguous tensor recorder & double cleanup
│   ├── trainer.py                  # Headless trainer with batch step loop
│   └── operators/                  # Genetic Operators Package
│       ├── selection.py            # Tournament selection strategy
│       ├── crossover.py            # In-place uniform crossover strategy
│       └── mutation.py             # In-place Gaussian noise mutation
│
├── bridges/                        # Pipeline Bridges & Presenters
│   ├── candidate_step_pipeline.py  # Single-frame tick & telemetry pipeline
│   ├── cli_presenter.py            # Real-time console table renderer
│   ├── live_winner_runner.py       # Real-time live winner maze solver runner
│   ├── playback_presenter.py       # Array-slicing UI view model presenter
│   ├── weight_bundler.py           # Contiguous float16 weight bundler
│   ├── telemetry_bundler.py        # Truncated float32 telemetry bundler
│   └── archive_bridge.py           # Uncompressed disk Doorman & instant unlinker
│
└── visualization/                  # GUI Visualizer & Rendering
    ├── map_renderer.py             # Static tilemap surface pre-renderer
    ├── camera_projection.py        # Viewport zoom & camera coordinate math
    ├── vision_renderer.py          # Vision fan & heading line renderer
    ├── help_overlay.py             # Interactive replay shortcut cheat-sheet
    ├── live_help_overlay.py        # Dedicated live solver cheat-sheet overlay
    ├── live_view_presenter.py      # Live winner view presenter & dispatcher
    ├── overlay_panel.py            # Dashboard HUD overlay panel
    ├── input_controller.py         # Pygame keyboard & mouse event dispatcher
    ├── app_window.py               # Pygame window runner & master loop
    ├── network_graph/              # Network Activation Graph Sub-Package
    │   ├── label_resolver.py       # Sensory shorthand & output label resolver
    │   ├── layout_engine.py        # Graph geometry & font fitting solver
    │   ├── column_renderer.py      # Activation node heatmap & header renderer
    │   └── graph_facade.py         # Top-level NetworkGraph coordinator facade
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
            ├── tile_renderer.py    # Map & player centered tile background renderer
            └── avatar_renderer.py  # Body sprite, face, status ring & solved arcs
            └── hud_overlay_renderer.py # Health bar, ID/score tags & cards


===============================================================================
[!] PYNEVO | SOVEREIGN NEUROEVOLUTION ENGINE
===============================================================================
Distributed under the PyNevo Source-Available End User License Agreement.
Copyright (c) 2026 herbal1st. All Rights Reserved.
Strictly for personal evaluation, education, private editing, and non-commercial
research.