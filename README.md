# <p align="center"><code>PYNEVO</code></p>

```ascii
 ____           __  __                                              
/\  _`\        /\ \/\ \                                             
\ \ \L\ \__  __\ \ `\\ \     __   __  __    ___                     
 \ \ ,__/\ \/\ \\ \ , ` \  /'__`\/\ \/\ \  / __`\                   
  \ \ \/\ \ \_\ \\ \ \`\ \/\  __/\ \ \_/ |/\ \L\ \                  
   \ \_\ \/`____ \\ \_\ \_\ \____\\ \___/ \ \____/                  
    \/_/  `/___/> \\/_/\/_/\/____/ \/__/   \/___/                   
             /\___/                                                 
             \/__/                                                  
====================================================================
           SOVEREIGN 2D NEUROEVOLUTION & ENDLESS ENGINE
====================================================================
```

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/Compute-Pure_CPU_Sovereign-orange?style=for-the-badge&logo=cpu&logoColor=white" alt="CPU Only" />
  <img src="https://img.shields.io/badge/JIT-Numba_Accelerated-yellow?style=for-the-badge&logo=numba&logoColor=black" alt="Numba JIT" />
  <img src="https://img.shields.io/badge/Physics-Circle--to--AABB_MTV-red?style=for-the-badge" alt="Physics" />
  <img src="https://img.shields.io/badge/Bitmask-PyBiwis_uint64-blueviolet?style=for-the-badge" alt="PyBiwis" />
  <img src="https://img.shields.io/badge/Engine-Pygame_CE-green?style=for-the-badge&logo=pygame&logoColor=white" alt="Pygame" />
</p>

---

## 📑 Table of Contents
1. [System Overview](#10-system-overview)
2. [Memory, Maps & Procedural Generation (PyBiwis)](#20-memory-maps--procedural-generation-pybiwis--strategies)
3. [Spatial Perception, Dual Compasses & BFS GPS](#30-spatial-perception-dual-compasses--bfs-gps-triggers)
4. [Data-Driven Profiles, Neural Topology & Persistence](#40-data-driven-profiles-neural-topology--persistence)
5. [Kinematics, Health & Triple-Metabolic Engine](#50-kinematics-health--triple-metabolic-refuel-engine)
6. [Dynamic Lighting, 3D Hillshading & Atmosphere](#51-dynamic-lighting-3d-hillshading--atmosphere-pipeline)
7. [Zero-Allocation Telemetry & Contiguous Tensors](#60-zero-allocation-telemetry-contiguous-tensors--live-brain-replay)
8. [Neuroevolution, GA Engine & Fitness Math](#70-neuroevolution-unconstrained-pools--fitness-math)
9. [Display HUD, Telemetry & Interactive Controls](#71-display-hud-cli-metrics--interactive-controls)
10. [Repository Directory Structure](#80-complete-modular-codebase-structure)

---

## 1.0 System Overview

```
                      ┌──────────────────────────────────────────────┐
                      │              PYNEVO ARCHITECTURE             │
                      └──────────────────────┬───────────────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
┌─────────────────────────┐                                     ┌─────────────────────────┐
│   HEADLESS GA TRAINER   │                                     │   ENDLESS WORLD ENGINE  │
├─────────────────────────┤                                     ├─────────────────────────┤
│ • Zero-Alloc Telemetry  │                                     │ • 16x16 Chunking + Ring │
│ • 3D BFS Distance Cache │                                     │ • Anti-Tunnel Substeps  │
│ • Mutated Elite Pools   │                                     │ • 3-Pass 3D Lighting    │
│ • Fast Disk Serialization│                                    │ • Companion AI Follower │
└────────────┬────────────┘                                     └────────────┬────────────┘
             │                                                               │
             └───────────────────────────────┬───────────────────────────────┘
                                             ▼
                                ┌─────────────────────────┐
                                │   REAL-TIME PYGAME UI   │
                                ├─────────────────────────┤
                                │ • 3x3 Stratified Grid   │
                                │ • Live Winner Solver    │
                                │ • Dual Track Scrubber   │
                                │ • Live Forward Heatmap  │
                                └─────────────────────────┘
```

* **Core Philosophy**: **Sovereign Compute**, **Matrix-Isolated**, **Zero-GPU Dependencies**.
* **Engine Foundations**: Built upon vectorized NumPy matrix mathematics, non-blocking JIT-compiled C-loops via Numba, PyBiwis 64-bit integer bitmask map compression, pre-rendered hardware-formatted chunk caching (`.convert()`), continuous Circle-to-AABB collision physics with Minimum Translation Vector (MTV) resolution, and multi-threaded CPU SIMD execution.
* **Autonomous Goal**: Train 2D neural agents to solve non-trivial labyrinth topologies, execute **15-frame stationary holds** on dynamic target zones, and chain multi-stage relocations across infinite terrain.

---

## 2.0 Memory, Maps & Procedural Generation (PyBiwis & Strategies)

> [!NOTE]
> **PyBiwis Compression Rule:** 64 discrete binary grid tiles are bit-packed into a single `uint64` machine word. In Endless Mode, each `16x16` tile chunk (256 tiles) consumes exactly **4 integer words (32 bytes)** of raw state memory.

```
       TILE GRID (16x16 = 256 tiles)               PYBIWIS 64-BIT BITMASK
┌─────────────────────────────────────────┐     ┌────────────────────────────┐
│ 0 1 0 0 1 1 0 1 ... (Tiles 0..63)       │ ──► │ Word 0: 0xA4F1902C... uint64│
│ 1 1 0 1 0 0 1 0 ... (Tiles 64..127)     │ ──► │ Word 1: 0x5D884B1E... uint64│
│ 0 0 1 1 1 0 1 0 ... (Tiles 128..191)    │ ──► │ Word 2: 0x90F2C110... uint64│
│ 1 0 1 0 0 1 1 1 ... (Tiles 192..255)    │ ──► │ Word 3: 0xEE7614A9... uint64│
└─────────────────────────────────────────┘     └────────────────────────────┘
```

### 🔹 Spatial Chunking & Radial Hysteresis
* **Circular Perimeter**: Operates with a circular load radius $R_{\text{load}}$ based on the screen diagonal plus a 1-chunk perimeter guard.
* **Hysteresis Buffer**: Unload radius $R_{\text{unload}} = R_{\text{load}} + 2$. Prevents memory thrashing when an agent steps across chunk seams, capping active chunk RAM footprint to **~7.5 MB**.
* **Seam-Free Snapping**: All viewport chunk rendering anchors to the top-left chunk index `(min_cx, min_cy)` with integer tile offsets to eliminate floating-point pixel-cracking.

### 🔹 Vectorized 2x2 Corner Smoothing (Octagon Strata)
* **18x18 Halo Padding**: Samples a 1-tile perimeter border (`x = -1..16, y = -1..16`) around chunks to allow cross-seam evaluation without inter-chunk queries.
* **Chamfer Mathematics**: Identifies cardinal neighbor matches and draws a 29.3% corner triangle patch ($x = 0.293 \times \text{tile\_size}$), rounding 90° corners into clean regular octagons.
* **Strata Priority**: Higher-elevation ground layers override lower strata at $2 \times 2$ checkerboard intersections to prevent inverted patch artifacts.

```
BLOCKY TILE INTERSECTION                     CHAMFERED REGULAR OCTAGON
       ┌──────┬──────┐                              ┌──────┬──────┐
       │ Grass│ Rock │                              │ Grass╱ Rock │
       ├──────┼──────┤       ───────────►           ├─────╱───────┤
       │ Grass│ Grass│                              │ Grass  Grass│
       └──────┴──────┘                              └─────────────┘
```

### 🔹 3D BFS Distance Pathfinder Cache
* **Array Geometry**: Pre-allocates a contiguous 3D NumPy array of shape `(max_targets, map_height, map_width)` populated with `9999` (unreachable flag).
* **On-Demand BFS Floodfill**: When target $T_K$ is assigned, a backward BFS expands from $T_K$ into index `[K, :, :]`.
* **$\mathcal{O}(1)$ Distance Queries**: Evaluated via single memory lookups: `_matrix_buffer[stage_idx, y, x]`.

---

## 3.0 Spatial Perception, Dual Compasses & BFS GPS Triggers

```
                         [ 0° Front Vision Ray ]
                        \           |           /
                         \          |          /
                          \         |         /
                           \        |        /
      [ Left Eye -22.5° ]    \      |      /    [ Right Eye +22.5° ]
                     \        \     |     /        /
                      ▼        \    |    /        ▼
                  [BFSL / EFL]  \   |   /   [BFSR / EFR]
                                 \  |  /
                                  \ | /
                                 ┌─────┐
                                 │ AI  │ ──► [ Heading θ ]
                                 └─────┘
```

### 🔹 Perception Matrix Composition

| Channel Group | Keys | Channels | Functional Description |
| :--- | :--- | :---: | :--- |
| **Amanatides-Woo Raycast** | `RAY_0`..`RAY_N` | $N$ | Geometric boundary-to-boundary DDA voxel traversal across `vision_arc_angle`. |
| **Proprioception** | `SPD`, `HP`, `DMG-C`, `DMG-I`, `DMG-S`, `HEAL` | **6** | Displacement speedometer, remaining stamina, collision hit, idle flag, turn tax, and kinetic recovery. |
| **Topological BFS GPS** | `BFSL-`, `BFSR-`, `BFSL+`, `BFSR+` | **4** *(stereo)*<br>**2** *(mono)* | Eye progress gradients toward active target $T_K$. Saturates to **1.0** inside target zone. |
| **Cardinal Needles** | `C-N`, `C-E`, `C-S`, `C-W` | **4** | 90° view-facing linear decay needles for orthogonal maze grid alignment. |
| **Binocular North** | `NFL`, `NFR`, `NPL`, `NPR` | **4** | Dual Focus & Peripheral stereo orientation channels to magnetic North. |
| **Target Lock Radar** | `EFL`, `EFR`, `EPL`, `EPR` | **4** | Focus/Peripheral stereo radar channels with 5-point inset visibility and optional LOS wall gating. |

> [!TIP]
> **Total Input Vector Size:**
> * **Stereo GPS Mode (`_b1`)**: `N_rays + 22` channels $\times (\text{memory\_frames} + 1)$
> * **Mono GPS Mode (`_b0`)**: `N_rays + 20` channels $\times (\text{memory\_frames} + 1)$

---

## 4.0 Data-Driven Profiles, Neural Topology & Persistence

All mechanics, sensors, kinematics, and visual palettes are decoupled into cleanly structured YAML files under `profiles/`:

```
profiles/
├── agent.yaml          # Kinematics, perception ranges, damages, and MLP architecture
├── player.yaml         # Human player steering, speed & skins
├── skin.yaml           # Visual palettes, camera zoom scales, ASCII faces, and ring styles
├── lighting.yaml       # Day/night orbital speed, terrain steepness, shadow/highlight levels
├── training.yaml       # Population size, generation caps, mutation/elitism hyper-parameters
├── map.yaml            # Bounded level dimensions, wall densities, and procedural strategies
├── map_endless.yaml    # Noise seeds, multi-octave frequencies, and terrain strata thresholds
└── tiles.yaml          # Physical friction, speed multipliers, passability, and borders
```

### 🔹 Motor Actuation Switch (`use_linear_speed_output`)
```python
# 1. Task-Space Linear Mode (use_linear_speed_output: true)
move_effort = float(outputs[0]) - float(outputs[1])  # FWD - BWD in [-1.0, 1.0]
turn_effort = float(outputs[3]) - float(outputs[2])  # S-R - S-L in [-1.0, 1.0]

# 2. Direct Differential Wheel Mode (use_linear_speed_output: false)
net_left  = float(outputs[0]) - float(outputs[1])   # L_FWD - L_BWD
net_right = float(outputs[2]) - float(outputs[3])   # R_FWD - R_BWD
move_effort = (net_right + net_left) / 2.0
turn_effort = (net_right - net_left) / 2.0
```

### 🔹 Brain Persistence Signatures
Trained neural networks are serialized with topology and actuation mode metadata baked into the filename:
```
saved_brains/TANK_1_v13_m2_h3_n15_b1_lin1.npz
             │      │   │  │   │  │  └─ lin1 = Task-Space / lin0 = Direct Differential
             │      │   │  │   │  └──── b1 = Stereo GPS / b0 = Mono GPS
             │      │   │  │   └─────── n15 = 15 Neurons per hidden layer
             │      │   │  └─────────── h3 = 3 Hidden Layers
             │      │   └────────────── m2 = 2 Temporal Memory Frames
             │      └────────────────── v13 = 13 Vision Rays
             └───────────────────────── Profile Identifier
```

---

## 5.0 Kinematics, Health & Triple-Metabolic Refuel Engine

```
       ┌─────────────────────────────── HEALTH METABOLISM ──────────────────────────────┐
       │                                                                                │
       │   [ Wall Hit ]   ───► Deduct health_coll_dmg_per_frame ──► Trigger DMG-C pulse │
       │   [ Stalling ]   ───► Deduct health_idle_dmg_per_frame ──► Trigger DMG-I pulse │
       │   [ Spin Tax ]   ───► Deduct health_spin_dmg * rot_ratio ► Trigger DMG-S pulse │
       │                                                                                │
       │   [ High Speed ] ───► Recover move_heal_per_frame       ──► Trigger HEAL pulse │
       │   [ Target Hold] ───► Recover target_hold_heal_per_frame──► Trigger HEAL pulse │
       │   [ Path GPS ]   ───► Recover path_heal * (BFSL+ + BFSR+) ──► Pure Refuel      │
       │                                                                                │
       │                         * HARD CAP: Health <= 1.00 (100%)                      │
       └────────────────────────────────────────────────────────────────────────────────┘
```

* **Continuous Run Enforcement**: Agents run until physical death or reaching `max_simulation_steps` (1000 steps). Clearing a stage advances the waypoint index without halting timeline logging.
* **Libra Balance Engine (Dynamic Status Ring)**:
  $$\text{Net Delta} = \sum \text{Damage Sources} - \sum \text{Recovery Sources}$$
  * $\text{Net Delta} = 0.0$ $\rightarrow$ **Yellow Status Ring** (Neutral balance)
  * $\text{Net Delta} > 0.0$ $\rightarrow$ Smoothly interpolates **Yellow $\rightarrow$ Red** (Damage state)
  * $\text{Net Delta} < 0.0$ $\rightarrow$ Smoothly interpolates **Yellow $\rightarrow$ Green** (Stamina recovery)
* **Gyroscopic Counter-Rotating Arcs**: When within $0.25$ tiles of a target center, concentric dual-radius blue arcs spin in opposite directions to confirm target lock.

---

## 5.1 Dynamic Lighting, 3D Hillshading & Atmosphere Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        3-PASS CANVAS COMPOSITOR                        │
├────────────────────────────────────────────────────────────────────────┤
│ Pass 1  │ Draw base terrain chunks & entity avatars                   │
├─────────┼──────────────────────────────────────────────────────────────┤
│ Pass 2A │ Blit subtractive mountain shadow surface                     │
├─────────┼──────────────────────────────────────────────────────────────┤
│ Pass 2B │ Blit Tile-Aware additive highlights (pygame.BLEND_ADD)       │
├─────────┼──────────────────────────────────────────────────────────────┤
│ Pass 3  │ Blit time-of-day ambient tint surface (pygame.BLEND_MULT)    │
└────────────────────────────────────────────────────────────────────────┘
```

* **360° Solar Orbit Clock**: Tracks an astronomical light angle vector:
  $$\theta_{\text{solar}} = \theta_{\text{start}} - (2\pi \times \text{time\_ratio})$$
* **Tile-Aware Highlights**: Highlights take on the natural base RGB palette of the underlying terrain (sand glints gold, grass glints green, snow glints bright white) using `pygame.BLEND_ADD`.
* **Subtractive Shadows**: Inverse normal products darken valley floors opposite the orbital light source.

---

## 6.0 Zero-Allocation Telemetry, Contiguous Tensors & Live Brain Replay

```
  STEP EXECUTION (Headless)               TELEMETRY BUNDLER               UNCOMPRESSED ARCHIVE
┌───────────────────────────┐         ┌─────────────────────────┐         ┌───────────────────────┐
│ Candidate Step Pipeline   │ ──O(1)─►│ Flat np.float32 Buffer  │ ──.npz─►│ .runtime_cache.npz    │
│ (X, Y, Heading, Health...)│         │ Shape: [T, Pop, 8]      │         │ (< 0.03s Fast Dump)   │
└───────────────────────────┘         └─────────────────────────┘         └───────────┬───────────┘
                                                                                      │
                                                                         Read & Unlink Instantly
                                                                                      │
                                                                                      ▼
                                                                          ┌───────────────────────┐
                                                                          │ Playback Presenter    │
                                                                          │ (Interactive Scrubber)│
                                                                          └───────────────────────┘
```

* **Zero Memory Allocation**: Physical telemetry writes directly into pre-allocated NumPy array slices without intermediate dictionary allocations during simulation loops.
* **Fast Disk Handshake**: Tensors and map bitmasks are dumped to `.runtime_cache.npz` using uncompressed binary formatting (`np.savez`). The cache file is deleted immediately after the visualizer loads it into RAM.

---

## 7.0 Neuroevolution, Unconstrained Pools & Fitness Math

$$\text{Raw Fitness} = \text{Lifetime Progress Distance} + (\text{Stages Cleared} \times \text{Stage Bonus})$$

* **Unconstrained Lifespan**: Every tile traversed toward active target waypoints awards cumulative scalar distance.
* **Stage Clear Reward**: A discrete **+20.0 bonus** is awarded per 15-frame hold completion, incentivizing target capture.
* **Self-Breeding Variant Trigger**: If tournament selection matches identical parents (`Parent A == Parent B`), the child undergoes forced mutation (`mutation_rate = 1.0`), preventing genetic stagnation.

---

## 7.1 Display HUD, CLI Metrics & Interactive Controls

### 🔹 CLI Generational Table
```
  GEN |   TOP |    AVG | FIRST | STAGE | TOUCH | SOLVE | EXITS |  PROS |   TIME
-------------------------------------------------------------------------------
    1 |    42 |   12.4 |  #  3 |     0 |   142 |     - |     2 |     - |  0.28s
   10 |   180 |   78.2 |  #  0 |     1 |    88 |   103 |     8 |     3 |  0.31s
   25 |   420 |  210.5 |  #  7 |     3 |    45 |    60 |    19 |    12 |  0.34s
```

### 🔹 Interactive Controls & Keybindings

```
┌───────────────────────────┬────────────────────────────────────────────────────────────────┐
│ Key / Input               │ Action                                                         │
├───────────────────────────┼────────────────────────────────────────────────────────────────┤
│ END                       │ Toggle Live Winner Evaluation Mode vs Replay Mode              │
│ SPACE                     │ Play / Pause timeline scrubber or live simulation ticks        │
│ LEFT / RIGHT              │ Step frame scrubber backward / forward (Hold for continuous)   │
│ UP / DOWN                 │ Cycle Generation (Replay Mode) OR cycle Saved Brains (Live)    │
│ T                         │ Toggle Scrubber Mode: [R] Reached Ticks vs [C] Cleared Ticks  │
│ ENTER                     │ Toggle Sub-Viewport Zoom Mode                                  │
│ TAB / RIGHT-CLICK         │ Toggle Camera Mode: Map-Centered vs Camera-Centered Tracking  │
│ 0 / NUMPAD 0              │ Toggle Repeat Mode: Loop Active Gen vs Loop Entire History     │
│ PGUP / PGDN / +, -        │ Step Simulation Speed multiplier (1/10x to 10x)                │
│ MOUSE WHEEL               │ Dynamically scale Simulation Speed up / down                   │
│ PERIOD (.)                │ Reset Playback Speed to 1.0x                                   │
│ R                         │ Resample middle candidates (Replay) OR Fresh Maze (Live Mode)  │
│ NUMPAD 1..9               │ 8-Directional candidate viewport selection grid                │
│ NUMPAD 5                  │ Reset selection directly to Candidate #0                       │
│ H                         │ Toggle HUD Help Shortcut Cheat-Sheet Overlay                   │
│ ESCAPE                    │ Safely shut down engine and exit application                   │
└───────────────────────────┴────────────────────────────────────────────────────────────────┘
```

---

## 8.0 Complete Modular Codebase Structure

<details open>
<summary><b>📂 Click to expand/collapse directory tree</b></summary>

```
PyNevo/
├── config.py                       # Global configuration, selectors & window geometry
├── main.py                         # Application entry point & thread/CUDA isolation
├── icon.png                        # Application window icon
├── MANUAL.txt                      # Detailed configuration guide
├── README.md                       # Master system specifications & guide
│
├── profiles/                       # Data-Driven Profile Library (YAML)
│   ├── agent.yaml                  # Kinematics, vision arcs, compasses & neural topology
│   ├── player.yaml                 # Human player mechanics, speeds & companion offsets
│   ├── skin.yaml                   # Visual palettes, camera zoom, ASCII faces & status rings
│   ├── lighting.yaml               # Day/night duration, hillshading & ambient keyframes
│   ├── training.yaml               # GA population bounds, mutation rates & step limits
│   ├── map.yaml                    # Bounded map dimensions, densities & maze strategies
│   ├── map_endless.yaml            # Endless noise world seeds, frequencies & strata layers
│   └── tiles.yaml                  # Tile friction, passability, colors & border ratios
│
├── utils/                          # Hardware & Mathematical Foundations
│   ├── math_utils.py               # Radians normalization, spin angles & vector distances
│   ├── geometry_utils.py           # Continuous line-segment ray-to-AABB clearance math
│   ├── color_utils.py              # RGB interpolation, health palettes & Net Delta mapping
│   ├── font_manager.py             # Global Pygame Font caching service by point size
│   ├── surface_utils.py            # Transparent alpha scratchpads & smooth surface scaling
│   └── noise.py                    # Deterministic vectorized Simplex & Perlin noise engines
│
├── world/                          # Spatial World & Infinite Terrain Engine
│   ├── bitmask_encoder.py          # PyBiwis 64-bit uint64 chunk encoder (4 words / chunk)
│   ├── chunk.py                    # 16x16 chunk container, surface baking & corner smoothing
│   ├── chunk_manager.py            # Spatial chunk storage & Euclidean circular loader
│   ├── endless_facade.py           # Adapters wrapping ChunkManager for neural perception
│   ├── tile_registry.py            # O(1) tile property lookup & color cache
│   ├── spawn_solver.py             # Multi-tile clearance safe spawn solver
│   ├── generation/                 # Endless Procedural Generation
│   │   └── endless_noise.py        # Vectorized 18x18 halo-padded terrain generator
│   └── lighting/                   # Dynamic Atmosphere & Hillshading Pipeline
│       ├── time_clock.py           # 360° counterclockwise orbital solar clock
│       ├── ambient_palette.py      # RGB keyframe palette resolver
│       ├── viewport_height_sampler.py # Vectorized float noise height sampler
│       ├── height_shadow_engine.py # Tile-Aware additive highlights & mountain shadows
│       └── atmosphere_overlay.py   # Direct 3-pass canvas overlay compositor
│
├── core/                           # Physics Systems & Map Generators
│   ├── bitmask_encoder.py          # PyBiwis 64-bit level bitmask encoder
│   ├── map_data.py                 # Grid layout, target sequences & LOS matrices
│   ├── pathfinder.py               # Contiguous 3D NumPy BFS matrix cache
│   ├── kinematics/                 # Physics & Steering Engines
│   │   ├── profiles.py             # Car and Tank steering profiles
│   │   ├── engine.py               # Circle-to-AABB MTV wall collision solver
│   │   └── endless_engine.py       # Endless kinematics & anti-tunneling micro-stepping
│   └── map_generation/             # Procedural Labyrinth Generator Suite
│       ├── base_strategy.py        # Abstract generator strategy interface
│       ├── branching_walls.py      # Organic branching wall crawler facade
│       ├── generator.py            # Map generator facade & connected component flood-fill
│       ├── halo_utils.py           # Shared halo geometry & snake corridor capacity math
│       ├── pacman_grid.py          # Arcade Pacman pillar arena strategy
│       ├── random_scatter.py       # Physics-safe random scatter strategy
│       └── branching/              # Branching Walls Sub-Package
│           ├── seed_manager.py     # Halo candidate seed pool manager
│           └── extension_solver.py # Serpentine capacity & stem clearance solver
│
├── entities/                       # Entity Definitions & Registries
│   ├── agent_profile_registry.py   # Agent profile registry facade
│   ├── player_profile_registry.py  # Human player YAML profile registry
│   ├── lighting_profile_registry.py# Lighting YAML profile registry
│   ├── player_controller.py        # Human player input dispatcher
│   ├── skin_profile_registry.py    # Visual Skin YAML profile registry
│   ├── training_profile_registry.py# Training YAML profile registry
│   ├── map_profile_registry.py     # Map Geometry YAML profile registry
│   ├── map_endless_profile_registry.py # Endless Map YAML profile registry
│   ├── agent_factory.py            # Profile-driven agent component factory
│   ├── entity_state.py             # Decoupled entity state containers
│   ├── entity_express.py           # ASCII facial expression resolver
│   └── agent_profile/              # Agent Profile Implementation
│       ├── profile_model.py        # ResolvedAgentProfile data model
│       └── yaml_parser.py          # Strict fail-fast YAML loader
│
├── perception/                     # Sensory Perception & Raycasting
│   ├── vision_arc.py               # Amanatides-Woo fast voxel raycaster
│   ├── exit_compass.py             # Stage-aware stereo exit radar
│   ├── cardinal_compass.py         # Binocular North & 4-needle compass
│   ├── spawn_heading.py            # Orthogonal cardinal spawn heading generator
│   ├── spatial_transformer.py      # Spatial perception coordinator facade
│   └── spatial/                    # Sensory Sub-Package
│       ├── gps_sensor.py           # Target-saturated BFS progress GPS
│       ├── memory_stacker.py       # Pre-allocated 3D temporal memory queue cache
│       └── feature_compiler.py     # Single-frame sensory feature compiler
│
├── neural/                         # Neural Perceptron Engine
│   ├── brain_persistence.py        # Signature-based weight archive manager
│   ├── weight_initializer.py       # Xavier, He, & Gaussian initializers
│   ├── layers.py                   # Dense layer with contiguous forward passes
│   ├── activations.py              # ReLU, Tanh, & Sigmoid activations
│   └── network.py                  # MLP with contiguous parameter buffer
│
├── evolution/                      # Neuroevolution & GA Engine
│   ├── fitness.py                  # Unconstrained cumulative progress evaluator
│   ├── population.py               # Persistent candidate pool & reproduction
│   ├── recorder.py                 # Contiguous tensor recorder & disk archiver
│   ├── trainer.py                  # Headless simulation loop & step pipeline
│   └── operators/                  # Genetic Operators
│       ├── selection.py            # Tournament selection strategy
│       ├── crossover.py            # Vectorized uniform crossover
│       └── mutation.py             # Vectorized Gaussian noise mutation
│
├── bridges/                        # Pipeline Bridges & Presenters
│   ├── candidate_step_pipeline.py  # Single-frame tick & telemetry logger
│   ├── cli_presenter.py            # Real-time console metrics table
│   ├── live_winner_runner.py       # Real-time live winner maze solver
│   ├── playback_presenter.py       # Array-slicing UI view model presenter
│   ├── weight_bundler.py           # Contiguous float16 weight tensor bundler
│   ├── telemetry_bundler.py        # Flat float32 telemetry bundler
│   └── archive_bridge.py           # Uncompressed disk I/O Doorman
│
└── visualization/                  # Pygame Visualizers & GUI
    ├── map_renderer.py             # Static clean tilemap surface renderer
    ├── camera_projection.py        # Camera zoom & coordinate projection math
    ├── vision_renderer.py          # Vision cone polygon & heading line renderer
    ├── companion_presenter.py      # Companion AI state, target & rendering
    ├── help_overlay.py             # Interactive shortcut cheat-sheet
    ├── live_help_overlay.py        # Live solver shortcut cheat-sheet
    ├── live_view_presenter.py      # Live winner view presenter
    ├── overlay_panel.py            # Dashboard HUD overlay panel
    ├── input_controller.py         # Pygame event dispatcher
    ├── app_window.py               # Historical 3x3 replay visualizer
    ├── endless_app_window.py       # Full-canvas 1280x720 endless world window
    ├── network_graph/              # Neural Activation Graph
    │   ├── label_resolver.py       # Sensory shorthand & output label resolver
    │   ├── layout_engine.py        # Graph geometry & font fitting solver
    │   ├── column_renderer.py      # Activation node heatmap & header
    │   └── graph_facade.py         # Top-level NetworkGraph coordinator
    ├── timeline_scrubber.py        # Transport scrubber facade
    ├── timeline/                   # Timeline Scrubber Sub-Package
    │   └── renderer.py             # Pre-computed RAM solver cache renderer
    ├── viewport_grid.py            # Viewport grid coordinator facade
    └── viewports/                  # Sub-Viewport Rendering Engine
        ├── adapter_interface.py    # IViewportAdapter abstract contract
        ├── grid_layout.py          # Spatial R x C layout geometry & bounds
        ├── candidate_mapper.py     # Stratified candidate slot mapper
        ├── native_maze_viewport.py # Native 2D maze viewport facade
        └── native/                 # Native Viewport Sub-Package
            ├── state_resolver.py   # Telemetry row & physics delta resolver
            ├── tile_renderer.py    # Dynamic active checkpoint/target renderer
            ├── avatar_renderer.py  # Body sprite, face, ring & solved arcs
            └── hud_overlay_renderer.py # Health bar, ID/score tags & cards
```

</details>

---

## 📄 License & Attribution

```ascii
===============================================================================
[!] PYNEVO | SOVEREIGN NEUROEVOLUTION ENGINE
===============================================================================
Distributed under the PyNevo Source-Available End User License Agreement.
Copyright (c) 2026 herbal1st. All Rights Reserved.
Strictly for personal evaluation, education, private editing, and non-commercial research.
```
