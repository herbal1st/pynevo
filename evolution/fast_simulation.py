"""
Vectorized Numba JIT simulation engine for neuroevolution.
Runs population candidates in parallel across CPU threads with zero IPC and zero heap allocation.
Optimized with contiguous SIMD cache layout for multi-generation performance on 60x45+ mazes.
Completely SLAM-compliant: strictly first-person embodied perception (LiDAR, clearance, proprioception,
vestibular compass, and line-of-sight visual exit beacon). Zero omniscience.
"""

import math
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from core.kinematics.engine import resolve_circle_aabb_jit
from core.map_data import march_los_segment_jit
from perception.vision_arc import cast_single_ray_jit


@njit(fastmath=True, inline="always")
def _mlp_forward_jit(
    weights: NDArray[np.float32],
    inp: NDArray[np.float32],
    in_dim: int,
    hidden_layers: int,
    neurons: int,
    h_a: NDArray[np.float32],
    h_b: NDArray[np.float32]
) -> Tuple[float, float, float, float]:
    offset = 0

    # Layer 0: in_dim -> neurons (Contiguous memory access for AVX vectorization)
    for j in range(neurons):
        h_a[j] = weights[offset + in_dim * neurons + j]  # bias

    for i in range(in_dim):
        in_val = inp[i]
        if in_val != 0.0:
            w_row = offset + i * neurons
            for j in range(neurons):
                h_a[j] += in_val * weights[w_row + j]

    for j in range(neurons):
        if h_a[j] < 0.0:
            h_a[j] = 0.0
    offset += in_dim * neurons + neurons

    use_a = True
    # Hidden layers: neurons -> neurons (Contiguous cache-line streaming)
    for l in range(1, hidden_layers):
        if use_a:
            for j in range(neurons):
                h_b[j] = weights[offset + neurons * neurons + j]
            for i in range(neurons):
                v = h_a[i]
                if v > 0.0:
                    w_row = offset + i * neurons
                    for j in range(neurons):
                        h_b[j] += v * weights[w_row + j]
            for j in range(neurons):
                if h_b[j] < 0.0:
                    h_b[j] = 0.0
            use_a = False
        else:
            for j in range(neurons):
                h_a[j] = weights[offset + neurons * neurons + j]
            for i in range(neurons):
                v = h_b[i]
                if v > 0.0:
                    w_row = offset + i * neurons
                    for j in range(neurons):
                        h_a[j] += v * weights[w_row + j]
            for j in range(neurons):
                if h_a[j] < 0.0:
                    h_a[j] = 0.0
            use_a = True
        offset += neurons * neurons + neurons

    # Output layer: neurons -> 4
    src_h = h_a if use_a else h_b
    o0 = weights[offset + neurons * 4 + 0]
    o1 = weights[offset + neurons * 4 + 1]
    o2 = weights[offset + neurons * 4 + 2]
    o3 = weights[offset + neurons * 4 + 3]

    for i in range(neurons):
        v = src_h[i]
        if v > 0.0:
            w_row = offset + i * 4
            o0 += v * weights[w_row + 0]
            o1 += v * weights[w_row + 1]
            o2 += v * weights[w_row + 2]
            o3 += v * weights[w_row + 3]

    if o0 >= 0.0:
        val0 = 1.0 / (1.0 + math.exp(-o0))
    else:
        ez0 = math.exp(o0)
        val0 = ez0 / (1.0 + ez0)

    if o1 >= 0.0:
        val1 = 1.0 / (1.0 + math.exp(-o1))
    else:
        ez1 = math.exp(o1)
        val1 = ez1 / (1.0 + ez1)

    if o2 >= 0.0:
        val2 = 1.0 / (1.0 + math.exp(-o2))
    else:
        ez2 = math.exp(o2)
        val2 = ez2 / (1.0 + ez2)

    if o3 >= 0.0:
        val3 = 1.0 / (1.0 + math.exp(-o3))
    else:
        ez3 = math.exp(o3)
        val3 = ez3 / (1.0 + ez3)

    return val0, val1, val2, val3


@njit(fastmath=True)
def simulate_single_candidate_jit(
    flat_weights: NDArray[np.float32],
    start_x: float,
    start_y: float,
    init_heading: float,
    grid_array: NDArray[np.uint8],
    map_width: int,
    map_height: int,
    target_cx: float,
    target_cy: float,
    in_dim: int,
    hidden_layers: int,
    neurons: int,
    v_rays: int,
    rel_angles: NDArray[np.float64],
    vision_max_dist: float,
    memory_frames: int,
    agent_radius: float,
    move_speed: float,
    rad_per_frame: float,
    profile_style_is_tank: bool,
    use_linear_speed_output: bool,
    idle_damage_speed_thresh: float,
    heal_speed_thresh: float,
    coll_dmg: float,
    idle_dmg: float,
    spin_dmg_rate: float,
    move_heal_rate: float,
    hold_heal_rate: float,
    target_hold_frames: int,
    max_steps: int,
    visited_grid: NDArray[np.uint8],
    mem_buf: NDArray[np.float32],
    base_vec: NDArray[np.float32],
    h_a: NDArray[np.float32],
    h_b: NDArray[np.float32]
) -> Tuple[
    float, float, float, float, bool, bool, int, int, int, float, float, int, int, int, float
]:
    base_dim = v_rays + 15
    total_frames = 1 + memory_frames

    mem_buf.fill(0.0)
    base_vec.fill(0.0)

    curr_x = start_x
    curr_y = start_y
    heading = init_heading
    health = 1.0
    is_alive = True
    touched_exit = False
    first_touch_step = -1
    first_hold_clear_step = -1
    hold_frame_counter = 0
    max_hold_frames = 0
    stages_cleared = 0
    total_lifetime_progress = 0.0
    distance_traveled = 0.0
    collision_count = 0
    frames_survived = 0
    max_disp_sq = 0.0

    hit = False
    is_idle = False
    is_healing = False
    speed_ratio = 0.0
    rot_ratio = 0.0
    angular_velocity = 0.0

    hold_dist_thresh_sq = 0.2304
    third = max(1, v_rays // 3)

    # Initial Observation (Step 0)
    for r in range(v_rays):
        ray_angle = heading + rel_angles[r]
        wall_prox, _ = cast_single_ray_jit(
            curr_x, curr_y, ray_angle, vision_max_dist, grid_array, map_width, map_height
        )
        base_vec[r] = np.float32(wall_prox)

    sum_l = 0.0
    for k in range(third):
        sum_l += base_vec[k]
    sum_f = 0.0
    for k in range(third, 2 * third):
        sum_f += base_vec[k]
    sum_r = 0.0
    for k in range(2 * third, v_rays):
        sum_r += base_vec[k]

    base_vec[v_rays]     = np.float32(max(0.0, min(1.0, 1.0 - (sum_l / third))))
    base_vec[v_rays + 1] = np.float32(max(0.0, min(1.0, 1.0 - (sum_f / third))))
    base_vec[v_rays + 2] = np.float32(max(0.0, min(1.0, 1.0 - (sum_r / max(1, v_rays - 2 * third)))))

    base_vec[v_rays + 3] = np.float32(0.0)
    base_vec[v_rays + 4] = np.float32(1.0)
    base_vec[v_rays + 5] = np.float32(0.0)
    base_vec[v_rays + 6] = np.float32(0.0)
    base_vec[v_rays + 7] = np.float32(0.0)
    base_vec[v_rays + 8] = np.float32(0.0)
    base_vec[v_rays + 9] = np.float32(0.0)

    base_vec[v_rays + 10] = np.float32(math.cos(heading))
    base_vec[v_rays + 11] = np.float32(math.sin(heading))

    dx_t = target_cx - curr_x
    dy_t = target_cy - curr_y
    dist_t = math.sqrt(dx_t * dx_t + dy_t * dy_t)
    beacon_range = max(vision_max_dist, 30.0)
    if dist_t <= beacon_range:
        has_los = march_los_segment_jit(
            curr_x, curr_y, target_cx, target_cy, grid_array, map_width, map_height, 0.2
        )
        if has_los:
            target_ang = math.atan2(dy_t, dx_t)
            t_delta = (target_ang - heading) % (2.0 * math.pi)
            if t_delta > math.pi:
                t_delta -= 2.0 * math.pi
            base_vec[v_rays + 12] = np.float32(1.0 - (dist_t / beacon_range))
            base_vec[v_rays + 13] = np.float32(math.cos(t_delta))
            base_vec[v_rays + 14] = np.float32(math.sin(t_delta))

    for f_idx in range(total_frames):
        off = f_idx * base_dim
        for i in range(base_dim):
            mem_buf[off + i] = base_vec[i]

    tx = int(math.floor(curr_x))
    ty = int(math.floor(curr_y))
    if 0 <= tx < map_width and 0 <= ty < map_height:
        visited_grid[ty, tx] = 1

    # Main Simulation Loop
    for step in range(max_steps):
        o0, o1, o2, o3 = _mlp_forward_jit(
            flat_weights, mem_buf, in_dim, hidden_layers, neurons, h_a, h_b
        )

        if use_linear_speed_output:
            move_eff = o0 - o1
            turn_eff = o3 - o2
        else:
            net_l = o0 - o1
            net_r = o2 - o3
            move_eff = (net_r + net_l) * 0.5
            turn_eff = (net_r - net_l) * 0.5

        if touched_exit:
            move_eff *= 0.35
            turn_eff *= 0.35

        prev_heading = heading
        clamped_turn = max(-1.0, min(1.0, turn_eff))
        clamped_move = max(-1.0, min(1.0, move_eff))

        if profile_style_is_tank:
            heading = (heading + clamped_turn * rad_per_frame) % (2.0 * math.pi)
        else:
            effective_turn = clamped_turn * abs(clamped_move)
            heading = (heading + effective_turn * rad_per_frame) % (2.0 * math.pi)

        d_th = abs((heading - prev_heading) % (2.0 * math.pi))
        if d_th > math.pi:
            d_th = 2.0 * math.pi - d_th
        rot_ratio = min(1.0, d_th / max(1e-6, rad_per_frame)) if spin_dmg_rate > 0.0 else 0.0

        step_dist = clamped_move * move_speed
        next_x = curr_x + math.cos(heading) * step_dist
        next_y = curr_y + math.sin(heading) * step_dist

        nx, ny, hit = resolve_circle_aabb_jit(
            next_x, next_y, agent_radius, map_width, map_height, grid_array, 2
        )

        dx = nx - curr_x
        dy = ny - curr_y
        disp_dist = math.sqrt(dx * dx + dy * dy)
        speed_ratio = min(1.0, max(0.0, disp_dist / max(1e-4, move_speed)))

        curr_x = nx
        curr_y = ny
        distance_traveled += disp_dist
        if hit:
            collision_count += 1
        angular_velocity = max(-1.0, min(1.0, turn_eff))

        cur_disp_sq = (curr_x - start_x) * (curr_x - start_x) + (curr_y - start_y) * (curr_y - start_y)
        if cur_disp_sq > max_disp_sq:
            max_disp_sq = cur_disp_sq

        dx_t = curr_x - target_cx
        dy_t = curr_y - target_cy
        dist_sq = dx_t * dx_t + dy_t * dy_t

        touched_exit = (dist_sq <= hold_dist_thresh_sq)
        is_idle = (speed_ratio < idle_damage_speed_thresh) and (not touched_exit)
        is_cruise_healing = (speed_ratio >= min(heal_speed_thresh, 0.30)) and is_alive
        is_hold_healing = touched_exit and is_alive
        is_healing = is_cruise_healing or is_hold_healing

        if hit:
            health -= coll_dmg
        if is_idle:
            health -= idle_dmg
        if spin_dmg_rate > 0.0 and rot_ratio > 0.0:
            health -= spin_dmg_rate * rot_ratio
        if is_cruise_healing and move_heal_rate > 0.0:
            health = min(1.0, health + max(move_heal_rate, 0.003))
        if is_hold_healing and hold_heal_rate > 0.0:
            health = min(1.0, health + hold_heal_rate)

        if health <= 0.0:
            health = 0.0
            is_alive = False

        if touched_exit:
            if first_touch_step < 0:
                first_touch_step = step
            hold_frame_counter += 1
            if hold_frame_counter > max_hold_frames:
                max_hold_frames = hold_frame_counter
            if hold_frame_counter >= target_hold_frames or first_hold_clear_step < 0:
                if first_hold_clear_step < 0:
                    first_hold_clear_step = step
                stages_cleared += 1
                total_lifetime_progress += 50.0
                frames_survived = step + 1
                break
        else:
            hold_frame_counter = 0

        tx = int(math.floor(curr_x))
        ty = int(math.floor(curr_y))
        if 0 <= tx < map_width and 0 <= ty < map_height:
            visited_grid[ty, tx] = 1

        frames_survived = step + 1
        if not is_alive:
            break

        # Observations for next step
        for r in range(v_rays):
            ray_angle = heading + rel_angles[r]
            wall_prox, _ = cast_single_ray_jit(
                curr_x, curr_y, ray_angle, vision_max_dist, grid_array, map_width, map_height
            )
            base_vec[r] = np.float32(wall_prox)

        sum_l = 0.0
        for k in range(third):
            sum_l += base_vec[k]
        sum_f = 0.0
        for k in range(third, 2 * third):
            sum_f += base_vec[k]
        sum_r = 0.0
        for k in range(2 * third, v_rays):
            sum_r += base_vec[k]

        base_vec[v_rays]     = np.float32(max(0.0, min(1.0, 1.0 - (sum_l / third))))
        base_vec[v_rays + 1] = np.float32(max(0.0, min(1.0, 1.0 - (sum_f / third))))
        base_vec[v_rays + 2] = np.float32(max(0.0, min(1.0, 1.0 - (sum_r / max(1, v_rays - 2 * third)))))

        base_vec[v_rays + 3] = np.float32(speed_ratio)
        base_vec[v_rays + 4] = np.float32(health)
        base_vec[v_rays + 5] = np.float32(1.0 if hit else 0.0)
        base_vec[v_rays + 6] = np.float32(1.0 if is_idle else 0.0)
        base_vec[v_rays + 7] = np.float32(rot_ratio)
        base_vec[v_rays + 8] = np.float32(1.0 if is_healing else 0.0)
        base_vec[v_rays + 9] = np.float32(angular_velocity)

        base_vec[v_rays + 10] = np.float32(math.cos(heading))
        base_vec[v_rays + 11] = np.float32(math.sin(heading))

        dist_t = math.sqrt(dist_sq)
        if dist_t <= vision_max_dist:
            has_los = march_los_segment_jit(
                curr_x, curr_y, target_cx, target_cy, grid_array, map_width, map_height, 0.2
            )
            if has_los:
                target_ang = math.atan2(dy_t, dx_t)
                t_delta = (target_ang - heading) % (2.0 * math.pi)
                if t_delta > math.pi:
                    t_delta -= 2.0 * math.pi
                base_vec[v_rays + 12] = np.float32(1.0 - (dist_t / beacon_range))
                base_vec[v_rays + 13] = np.float32(math.cos(t_delta))
                base_vec[v_rays + 14] = np.float32(math.sin(t_delta))
            else:
                base_vec[v_rays + 12] = 0.0
                base_vec[v_rays + 13] = 0.0
                base_vec[v_rays + 14] = 0.0
        else:
            base_vec[v_rays + 12] = 0.0
            base_vec[v_rays + 13] = 0.0
            base_vec[v_rays + 14] = 0.0

        if total_frames > 1:
            shift_len = (total_frames - 1) * base_dim
            for i in range(shift_len):
                mem_buf[i] = mem_buf[base_dim + i]
            for i in range(base_dim):
                mem_buf[shift_len + i] = base_vec[i]
        else:
            for i in range(base_dim):
                mem_buf[i] = base_vec[i]

    unique_visited = 0
    for y in range(map_height):
        for x in range(map_width):
            if visited_grid[y, x] > 0:
                unique_visited += 1

    max_disp = math.sqrt(max_disp_sq)

    return (
        curr_x, curr_y, heading, health, is_alive, touched_exit,
        first_touch_step, first_hold_clear_step, stages_cleared,
        total_lifetime_progress, distance_traveled, collision_count,
        frames_survived, unique_visited, max_disp
    )


@njit(parallel=True, fastmath=True, cache=True)
def simulate_population_parallel_jit(
    pop_weights: NDArray[np.float32],
    start_x: float,
    start_y: float,
    initial_headings: NDArray[np.float64],
    grid_array: NDArray[np.uint8],
    map_width: int,
    map_height: int,
    target_cx: float,
    target_cy: float,
    in_dim: int,
    hidden_layers: int,
    neurons: int,
    output_size: int,
    v_rays: int,
    rel_angles: NDArray[np.float64],
    vision_max_dist: float,
    memory_frames: int,
    agent_radius: float,
    move_speed: float,
    rad_per_frame: float,
    profile_style_is_tank: bool,
    use_linear_speed_output: bool,
    idle_damage_speed_thresh: float,
    heal_speed_thresh: float,
    coll_dmg: float,
    idle_dmg: float,
    spin_dmg_rate: float,
    move_heal_rate: float,
    hold_heal_rate: float,
    target_hold_frames: int,
    max_steps: int,
    # Outputs:
    out_final_x: NDArray[np.float64],
    out_final_y: NDArray[np.float64],
    out_final_heading: NDArray[np.float64],
    out_final_health: NDArray[np.float64],
    out_is_alive: NDArray[np.bool_],
    out_touched_exit: NDArray[np.bool_],
    out_first_touch_step: NDArray[np.int32],
    out_first_hold_clear_step: NDArray[np.int32],
    out_stages_cleared: NDArray[np.int32],
    out_total_lifetime_progress: NDArray[np.float64],
    out_distance_traveled: NDArray[np.float64],
    out_collision_count: NDArray[np.int32],
    out_frames_survived: NDArray[np.int32],
    out_unique_visited: NDArray[np.int32],
    out_max_disp: NDArray[np.float64],
    out_visited_grids: NDArray[np.uint8],
    mem_bufs: NDArray[np.float32],
    base_vecs: NDArray[np.float32],
    h_as: NDArray[np.float32],
    h_bs: NDArray[np.float32]
) -> None:
    pop_size = len(initial_headings)
    for c_idx in prange(pop_size):
        (
            cx, cy, head, hp, alive, touched,
            fts, fhcs, st_clr, tlp, dist, colls,
            survived, u_vis, disp
        ) = simulate_single_candidate_jit(
            pop_weights[c_idx],
            start_x, start_y, initial_headings[c_idx],
            grid_array, map_width, map_height,
            target_cx, target_cy,
            in_dim, hidden_layers, neurons,
            v_rays, rel_angles, vision_max_dist, memory_frames,
            agent_radius, move_speed, rad_per_frame,
            profile_style_is_tank, use_linear_speed_output,
            idle_damage_speed_thresh, heal_speed_thresh,
            coll_dmg, idle_dmg, spin_dmg_rate, move_heal_rate, hold_heal_rate,
            target_hold_frames, max_steps,
            out_visited_grids[c_idx],
            mem_bufs[c_idx],
            base_vecs[c_idx],
            h_as[c_idx],
            h_bs[c_idx]
        )
        out_final_x[c_idx] = cx
        out_final_y[c_idx] = cy
        out_final_heading[c_idx] = head
        out_final_health[c_idx] = hp
        out_is_alive[c_idx] = alive
        out_touched_exit[c_idx] = touched
        out_first_touch_step[c_idx] = fts
        out_first_hold_clear_step[c_idx] = fhcs
        out_stages_cleared[c_idx] = st_clr
        out_total_lifetime_progress[c_idx] = tlp
        out_distance_traveled[c_idx] = dist
        out_collision_count[c_idx] = colls
        out_frames_survived[c_idx] = survived
        out_unique_visited[c_idx] = u_vis
        out_max_disp[c_idx] = disp


def warmup_fast_simulation() -> None:
    dummy_grid = np.zeros((10, 10), dtype=np.uint8)
    dummy_grid[0, :] = 1
    dummy_grid[-1, :] = 1
    dummy_grid[:, 0] = 1
    dummy_grid[:, -1] = 1

    v_rays = 15
    base_dim = v_rays + 15
    mem_frames = 2
    in_dim = base_dim * (1 + mem_frames)
    hidden_layers = 3
    neurons = 40
    out_dim = 4

    total_params = (in_dim * neurons + neurons) + (hidden_layers - 1) * (neurons * neurons + neurons) + (neurons * out_dim + out_dim)
    dummy_weights = np.zeros((2, total_params), dtype=np.float32)
    dummy_headings = np.array([0.0, 1.57], dtype=np.float64)
    dummy_rel_angles = np.linspace(-1.5, 1.5, v_rays, dtype=np.float64)

    out_fx = np.zeros(2, dtype=np.float64)
    out_fy = np.zeros(2, dtype=np.float64)
    out_fh = np.zeros(2, dtype=np.float64)
    out_fhp = np.zeros(2, dtype=np.float64)
    out_al = np.zeros(2, dtype=np.bool_)
    out_te = np.zeros(2, dtype=np.bool_)
    out_fts = np.zeros(2, dtype=np.int32)
    out_fhcs = np.zeros(2, dtype=np.int32)
    out_sc = np.zeros(2, dtype=np.int32)
    out_tlp = np.zeros(2, dtype=np.float64)
    out_dist = np.zeros(2, dtype=np.float64)
    out_colls = np.zeros(2, dtype=np.int32)
    out_surv = np.zeros(2, dtype=np.int32)
    out_uvis = np.zeros(2, dtype=np.int32)
    out_mdisp = np.zeros(2, dtype=np.float64)
    out_vgrids = np.zeros((2, 10, 10), dtype=np.uint8)

    mem_bufs = np.zeros((2, in_dim), dtype=np.float32)
    base_vecs = np.zeros((2, base_dim), dtype=np.float32)
    h_as = np.zeros((2, neurons), dtype=np.float32)
    h_bs = np.zeros((2, neurons), dtype=np.float32)

    simulate_population_parallel_jit(
        dummy_weights,
        5.0, 5.0, dummy_headings,
        dummy_grid, 10, 10,
        8.5, 8.5,
        in_dim, hidden_layers, neurons, out_dim,
        v_rays, dummy_rel_angles, 6.0, mem_frames,
        0.225, 0.15, 0.1,
        True, True,
        0.05, 0.80, 0.005, 0.005, 0.0, 0.002, 0.005,
        15, 5,
        out_fx, out_fy, out_fh, out_fhp, out_al, out_te, out_fts,
        out_fhcs, out_sc, out_tlp, out_dist, out_colls, out_surv,
        out_uvis, out_mdisp, out_vgrids,
        mem_bufs, base_vecs, h_as, h_bs
    )
