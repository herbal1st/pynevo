"""
Universal Hardware-Agnostic Batched Simulation Engine for PyNevo.
Features Absolute Volume Exploration Scoring to eliminate ratio-gaming local minima.
"""

import math
import warnings
import numpy as np

try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
except Exception:
    pass

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ==========================================================================
# 1. CPU MULTI-CORE KERNEL
# ==========================================================================

@njit(parallel=True, fastmath=True)
def batch_simulate_cpu(
    map_grid: np.ndarray,
    bfs_grid: np.ndarray,
    start_pos: np.ndarray,
    exit_pos: np.ndarray,
    w1: np.ndarray, b1: np.ndarray,
    w2: np.ndarray, b2: np.ndarray,
    w3: np.ndarray, b3: np.ndarray,
    w_out: np.ndarray, b_out: np.ndarray,
    scores_out: np.ndarray,
    summary_out: np.ndarray,
    telemetry_out: np.ndarray,
    initial_dist: int,
    max_steps: int,
    num_rays: int,
    base_channels: int,
    total_in_dim: int,
    memory_frames: int,
    hidden_dim: int,
    num_hidden_layers: int,
    half_arc_rad: float,
    ray_step_rad: float,
    vision_max_dist: float,
    move_speed: float,
    turn_rad: float,
    r_body: float,
    offset_rad: float,
    act_gps: float,
    act_cardinal: float,
    act_north: float,
    act_exit: float
):
    pop_size = scores_out.shape[0]
    map_h = map_grid.shape[0]
    map_w = map_grid.shape[1]
    two_pi = 2.0 * math.pi
    total_frames = 1 + memory_frames

    sx_f = float(start_pos[0]) + 0.5
    sy_f = float(start_pos[1]) + 0.5

    for cand_idx in prange(pop_size):
        cx = sx_f
        cy = sy_f
        heading = 0.0
        health = 1.0
        is_alive = True
        reached_exit = False
        best_dist = initial_dist
        frames_survived = 0
        tiles_explored = 1
        total_tile_steps = 1
        hover_frames = 0
        current_tile_frames = 0
        last_tx = int(start_pos[0])
        last_ty = int(start_pos[1])

        centering_accum = 0.0
        prev_x = cx
        prev_y = cy

        trail_bits = np.zeros(64, dtype=np.uint64)
        start_flat = last_ty * map_w + last_tx
        trail_bits[start_flat // 64] |= np.uint64(1 << (start_flat % 64))

        mem_buf = np.zeros(512, dtype=np.float32)
        base_buf = np.zeros(64, dtype=np.float32)
        in_buf = np.zeros(256, dtype=np.float32)
        h1_buf = np.zeros(hidden_dim, dtype=np.float32)
        h2_buf = np.zeros(hidden_dim, dtype=np.float32)
        h3_buf = np.zeros(hidden_dim, dtype=np.float32)
        out_buf = np.zeros(4, dtype=np.float32)

        for step in range(max_steps):
            if not is_alive or reached_exit:
                if step > 0 and cand_idx < 25:
                    for ch in range(8):
                        telemetry_out[step, cand_idx, ch] = telemetry_out[step - 1, cand_idx, ch]
                continue

            # 1. Dual-Layer LiDAR
            left_wall_d = vision_max_dist
            right_wall_d = vision_max_dist

            for r in range(num_rays):
                if r < num_rays - 2:
                    r_ang = heading - half_arc_rad + (float(r) * ray_step_rad)
                elif r == num_rays - 2:
                    r_ang = heading - (math.pi * 0.5)
                else:
                    r_ang = heading + (math.pi * 0.5)

                dir_x = math.cos(r_ang)
                dir_y = math.sin(r_ang)
                if abs(dir_x) < 1e-9: dir_x = 1e-9 if dir_x >= 0.0 else -1e-9
                if abs(dir_y) < 1e-9: dir_y = 1e-9 if dir_y >= 0.0 else -1e-9

                tx = int(math.floor(cx))
                ty = int(math.floor(cy))
                if tx < 0 or tx >= map_w or ty < 0 or ty >= map_h or map_grid[ty, tx] == 1:
                    base_buf[r] = 1.0
                    base_buf[num_rays + r] = 1.0
                else:
                    step_x = 1 if dir_x > 0.0 else -1
                    step_y = 1 if dir_y > 0.0 else -1
                    t_delta_x = abs(1.0 / dir_x)
                    t_delta_y = abs(1.0 / dir_y)
                    t_max_x = (float(tx + 1) - cx) * t_delta_x if dir_x > 0.0 else (cx - float(tx)) * t_delta_x
                    t_max_y = (float(ty + 1) - cy) * t_delta_y if dir_y > 0.0 else (cy - float(ty)) * t_delta_y

                    curr_d = 0.0
                    hit_val = 0.0
                    scent_accum = 0.0
                    scent_samples = 0
                    while curr_d < vision_max_dist:
                        if t_max_x < t_max_y:
                            curr_d = t_max_x
                            t_max_x += t_delta_x
                            tx += step_x
                        else:
                            curr_d = t_max_y
                            t_max_y += t_delta_y
                            ty += step_y

                        if 0 <= tx < map_w and 0 <= ty < map_h:
                            flat_t = ty * map_w + tx
                            if (trail_bits[flat_t // 64] & np.uint64(1 << (flat_t % 64))) != np.uint64(0):
                                scent_accum += 1.0
                            scent_samples += 1

                        if tx < 0 or tx >= map_w or ty < 0 or ty >= map_h or map_grid[ty, tx] == 1:
                            hit_dist = curr_d if curr_d < vision_max_dist else vision_max_dist
                            val = 1.0 - (hit_dist / vision_max_dist)
                            hit_val = val if val > 0.0 else 0.0
                            if r == num_rays - 2: left_wall_d = hit_dist
                            if r == num_rays - 1: right_wall_d = hit_dist
                            break

                    base_buf[r] = hit_val
                    base_buf[num_rays + r] = (scent_accum / float(scent_samples)) if scent_samples > 0 else 0.0

            # 2. Proprioception & Dead-Reckoning
            idx = num_rays * 2
            vel_x = (cx - prev_x) / (move_speed if move_speed > 1e-4 else 1e-4)
            vel_y = (cy - prev_y) / (move_speed if move_speed > 1e-4 else 1e-4)
            prev_x = cx
            prev_y = cy

            center_error = (left_wall_d - right_wall_d) / vision_max_dist
            diff_ce = 1.0 - abs(center_error)
            centering_accum += (diff_ce if diff_ce > 0.0 else 0.0)

            base_buf[idx] = math.sqrt(vel_x * vel_x + vel_y * vel_y)
            base_buf[idx + 1] = health
            base_buf[idx + 2] = (cx - sx_f) / float(map_w)
            base_buf[idx + 3] = (cy - sy_f) / float(map_h)
            base_buf[idx + 4] = vel_x
            base_buf[idx + 5] = center_error
            idx += 6

            # 3. Cardinal Needles
            if act_cardinal > 0.5:
                d_c0 = abs(((-math.pi * 0.5) - heading) % two_pi)
                if d_c0 > math.pi: d_c0 = abs(d_c0 - two_pi)
                val_0 = 1.0 - d_c0 / (math.pi * 0.5)
                base_buf[idx] = val_0 if val_0 > 0.0 else 0.0

                d_c1 = abs((0.0 - heading) % two_pi)
                if d_c1 > math.pi: d_c1 = abs(d_c1 - two_pi)
                val_1 = 1.0 - d_c1 / (math.pi * 0.5)
                base_buf[idx + 1] = val_1 if val_1 > 0.0 else 0.0

                d_c2 = abs(((math.pi * 0.5) - heading) % two_pi)
                if d_c2 > math.pi: d_c2 = abs(d_c2 - two_pi)
                val_2 = 1.0 - d_c2 / (math.pi * 0.5)
                base_buf[idx + 2] = val_2 if val_2 > 0.0 else 0.0

                d_c3 = abs((math.pi - heading) % two_pi)
                if d_c3 > math.pi: d_c3 = abs(d_c3 - two_pi)
                val_3 = 1.0 - d_c3 / (math.pi * 0.5)
                base_buf[idx + 3] = val_3 if val_3 > 0.0 else 0.0
            else:
                base_buf[idx] = 0.0; base_buf[idx+1] = 0.0; base_buf[idx+2] = 0.0; base_buf[idx+3] = 0.0
            idx += 4

            # 4. North Compass
            if act_north > 0.5:
                d_north = ((-math.pi * 0.5) - heading) % two_pi
                if d_north > math.pi: d_north -= two_pi
                d_n_r = abs((d_north + offset_rad) % two_pi)
                if d_n_r > math.pi: d_n_r = abs(d_n_r - two_pi)
                d_n_l = abs((d_north - offset_rad) % two_pi)
                if d_n_l > math.pi: d_n_l = abs(d_n_l - two_pi)

                val_nr = 1.0 - d_n_r / (math.pi * 0.5)
                base_buf[idx] = val_nr if val_nr > 0.0 else 0.0
                val_nl = 1.0 - d_n_l / (math.pi * 0.5)
                base_buf[idx + 1] = val_nl if val_nl > 0.0 else 0.0
                val_npr = 1.0 - d_n_r / math.pi
                base_buf[idx + 2] = val_npr if val_npr > 0.0 else 0.0
                val_npl = 1.0 - d_n_l / math.pi
                base_buf[idx + 3] = val_npl if val_npl > 0.0 else 0.0
            else:
                base_buf[idx] = 0.0; base_buf[idx+1] = 0.0; base_buf[idx+2] = 0.0; base_buf[idx+3] = 0.0
            idx += 4

            base_buf[idx] = 0.0; base_buf[idx+1] = 0.0; base_buf[idx+2] = 0.0; base_buf[idx+3] = 0.0
            base_buf[idx+4] = 0.0; base_buf[idx+5] = 0.0; base_buf[idx+6] = 0.0; base_buf[idx+7] = 0.0

            if step == 0:
                for f in range(total_frames):
                    for b in range(base_channels):
                        mem_buf[f * base_channels + b] = base_buf[b]
            else:
                for f in range(total_frames - 1):
                    for b in range(base_channels):
                        mem_buf[f * base_channels + b] = mem_buf[(f + 1) * base_channels + b]
                for b in range(base_channels):
                    mem_buf[(total_frames - 1) * base_channels + b] = base_buf[b]

            for i in range(total_in_dim):
                in_buf[i] = mem_buf[i]

            # Dynamic MLP Forward
            for h in range(hidden_dim):
                acc = b1[cand_idx, h]
                for i in range(total_in_dim):
                    acc += in_buf[i] * w1[cand_idx, i, h]
                h1_buf[h] = acc if acc > 0.0 else 0.0

            for h in range(hidden_dim):
                acc = b2[cand_idx, h]
                for p in range(hidden_dim):
                    acc += h1_buf[p] * w2[cand_idx, p, h]
                h2_buf[h] = acc if acc > 0.0 else 0.0

            if num_hidden_layers >= 3:
                for h in range(hidden_dim):
                    acc = b3[cand_idx, h]
                    for p in range(hidden_dim):
                        acc += h2_buf[p] * w3[cand_idx, p, h]
                    h3_buf[h] = acc if acc > 0.0 else 0.0
                last_hidden = h3_buf
            else:
                last_hidden = h2_buf

            for o in range(4):
                acc = b_out[cand_idx, o]
                for p in range(hidden_dim):
                    acc += last_hidden[p] * w_out[cand_idx, p, o]
                clamped = acc if acc > -500.0 else -500.0
                if clamped > 500.0: clamped = 500.0
                out_buf[o] = 1.0 / (1.0 + math.exp(-clamped))

            l_eff = out_buf[0] - out_buf[1]
            r_eff = out_buf[2] - out_buf[3]
            m_eff = (r_eff + l_eff) * 0.5
            t_eff = (r_eff - l_eff) * 0.5

            heading = (heading + (t_eff * turn_rad)) % two_pi
            step_len = m_eff * move_speed
            nx = cx + math.cos(heading) * step_len
            ny = cy + math.sin(heading) * step_len

            tx = int(math.floor(nx))
            ty = int(math.floor(ny))
            hit = False
            if tx < 0 or tx >= map_w or ty < 0 or ty >= map_h or map_grid[ty, tx] == 1:
                hit = True
                h_sub = health - 0.005
                health = h_sub if h_sub > 0.0 else 0.0
            else:
                cx = nx
                cy = ny

                if tx != last_tx or ty != last_ty:
                    total_tile_steps += 1
                    current_tile_frames = 0
                    last_tx = tx
                    last_ty = ty
                else:
                    current_tile_frames += 1
                    if current_tile_frames > 8:
                        hover_frames += 1

                flat_tile = ty * map_w + tx
                if (trail_bits[flat_tile // 64] & np.uint64(1 << (flat_tile % 64))) == np.uint64(0):
                    trail_bits[flat_tile // 64] |= np.uint64(1 << (flat_tile % 64))
                    tiles_explored += 1

            if health <= 0.0:
                is_alive = False

            frames_survived += 1
            curr_tx = int(math.floor(cx))
            curr_ty = int(math.floor(cy))
            curr_d = bfs_grid[curr_ty, curr_tx] if (0 <= curr_tx < map_w and 0 <= curr_ty < map_h) else 9999
            if curr_d < best_dist:
                best_dist = curr_d

            if curr_tx == exit_pos[0] and curr_ty == exit_pos[1]:
                reached_exit = True

            if cand_idx < 25:
                telemetry_out[step, cand_idx, 0] = cx
                telemetry_out[step, cand_idx, 1] = cy
                telemetry_out[step, cand_idx, 2] = heading
                telemetry_out[step, cand_idx, 3] = health
                telemetry_out[step, cand_idx, 4] = float(curr_d)
                telemetry_out[step, cand_idx, 5] = 1.0 if hit else 0.0
                telemetry_out[step, cand_idx, 6] = 1.0 if is_alive else 0.0
                telemetry_out[step, cand_idx, 7] = 1.0 if reached_exit else 0.0

        # ======================================================================
        # ABSOLUTE VOLUME EXPLORATION FITNESS (No Ratio Loopholes)
        # ======================================================================
        diff_best = float(initial_dist - best_dist)
        dist_done = diff_best if diff_best > 0.0 else 0.0
        denom_dist = float(initial_dist) if float(initial_dist) > 1.0 else 1.0
        score_dist = 500.0 * (dist_done / denom_dist)

        # Volume Reward: 100 tiles = 300 pts! 5 tiles = only 15 pts!
        score_explore = float(tiles_explored) * 3.0
        if score_explore > 300.0: score_explore = 300.0

        # Excessive looping penalty (Backtracking once is free, looping 3x+ is penalized)
        excess_steps = float(total_tile_steps - tiles_explored * 2)
        penalty_cycling = (excess_steps * 1.5) if excess_steps > 0.0 else 0.0
        if penalty_cycling > 150.0: penalty_cycling = 150.0

        # Exit Solve Jackpot (+500 flat)
        score_exit = 0.0
        if reached_exit:
            time_bonus = 300.0 * (1.0 - (float(frames_survived) / float(max_steps)))
            score_exit = 500.0 + time_bonus

        total_fitness = score_dist + score_explore + score_exit - penalty_cycling
        if total_fitness < 0.0: total_fitness = 0.0

        scores_out[cand_idx] = total_fitness * (0.8 + 0.2 * health)

        denom_fs = float(frames_survived) if float(frames_survived) > 1.0 else 1.0
        avg_cntr = int((centering_accum / denom_fs) * 100.0)
        denom_steps = float(total_tile_steps) if float(total_tile_steps) > 1.0 else 1.0
        effc_pct = int((float(tiles_explored) / denom_steps) * 100.0)
        dwell_ratio = float(hover_frames) / denom_fs
        sub_p = 1.0 - (dwell_ratio * 2.0)
        pace_pct = int((sub_p if sub_p > 0.0 else 0.0) * 100.0)

        summary_out[cand_idx, 0] = best_dist
        summary_out[cand_idx, 1] = 1 if reached_exit else 0
        summary_out[cand_idx, 2] = tiles_explored
        summary_out[cand_idx, 3] = avg_cntr
        summary_out[cand_idx, 4] = effc_pct
        summary_out[cand_idx, 5] = pace_pct


# ==========================================================================
# 2. GPU CUDA KERNEL (100% Inlined - Zero Python Device Calls)
# ==========================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def batch_simulate_cuda(
        map_grid,
        bfs_grid,
        start_pos,
        exit_pos,
        w1, b1, w2, b2, w3, b3, w_out, b_out,
        scores_out,
        summary_out,
        telemetry_out,
        initial_dist,
        max_steps,
        num_rays,
        base_channels,
        total_in_dim,
        memory_frames,
        hidden_dim,
        num_hidden_layers,
        half_arc_rad,
        ray_step_rad,
        vision_max_dist,
        move_speed,
        turn_rad,
        r_body,
        offset_rad,
        act_gps,
        act_cardinal,
        act_north,
        act_exit
    ):
        cand_idx = cuda.grid(1)
        pop_size = scores_out.shape[0]

        s_map = cuda.shared.array((48, 48), dtype=np.uint8)
        map_h = map_grid.shape[0]
        map_w = map_grid.shape[1]

        tx_id = cuda.threadIdx.x
        b_dim = cuda.blockDim.x
        total_tiles = map_h * map_w
        for idx in range(tx_id, total_tiles, b_dim):
            my_y = idx // map_w
            my_x = idx % map_w
            s_map[my_y, my_x] = map_grid[my_y, my_x]

        cuda.syncthreads()

        if cand_idx >= pop_size:
            return

        two_pi = 2.0 * math.pi
        total_frames = 1 + memory_frames

        sx_f = float(start_pos[0]) + 0.5
        sy_f = float(start_pos[1]) + 0.5
        cx = sx_f
        cy = sy_f
        heading = 0.0
        health = 1.0
        is_alive = True
        reached_exit = False
        best_dist = initial_dist
        frames_survived = 0
        tiles_explored = 1
        total_tile_steps = 1
        hover_frames = 0
        current_tile_frames = 0
        last_tx = int(start_pos[0])
        last_ty = int(start_pos[1])

        centering_accum = 0.0
        prev_x = cx
        prev_y = cy

        trail_bits = cuda.local.array(64, dtype=np.uint64)
        for w_i in range(64):
            trail_bits[w_i] = np.uint64(0)
        start_flat = last_ty * map_w + last_tx
        trail_bits[start_flat // 64] |= np.uint64(1 << (start_flat % 64))

        mem_buf = cuda.local.array(512, dtype=np.float32)
        base_buf = cuda.local.array(64, dtype=np.float32)
        in_buf = cuda.local.array(256, dtype=np.float32)
        h1_buf = cuda.local.array(64, dtype=np.float32)
        h2_buf = cuda.local.array(64, dtype=np.float32)
        h3_buf = cuda.local.array(64, dtype=np.float32)
        out_buf = cuda.local.array(4, dtype=np.float32)

        for step in range(max_steps):
            if not is_alive or reached_exit:
                if step > 0 and cand_idx < 25:
                    for ch in range(8):
                        telemetry_out[step, cand_idx, ch] = telemetry_out[step - 1, cand_idx, ch]
                continue

            left_wall_d = vision_max_dist
            right_wall_d = vision_max_dist

            for r in range(num_rays):
                if r < num_rays - 2:
                    r_ang = heading - half_arc_rad + (float(r) * ray_step_rad)
                elif r == num_rays - 2:
                    r_ang = heading - (math.pi * 0.5)
                else:
                    r_ang = heading + (math.pi * 0.5)

                dir_x = math.cos(r_ang)
                dir_y = math.sin(r_ang)
                if abs(dir_x) < 1e-7: dir_x = 1e-7 if dir_x >= 0.0 else -1e-7
                if abs(dir_y) < 1e-7: dir_y = 1e-7 if dir_y >= 0.0 else -1e-7

                tx = int(math.floor(cx))
                ty = int(math.floor(cy))
                if tx < 0 or tx >= map_w or ty < 0 or ty >= map_h or s_map[ty, tx] == 1:
                    base_buf[r] = 1.0
                    base_buf[num_rays + r] = 1.0
                else:
                    step_x = 1 if dir_x > 0.0 else -1
                    step_y = 1 if dir_y > 0.0 else -1
                    t_delta_x = abs(1.0 / dir_x)
                    t_delta_y = abs(1.0 / dir_y)
                    t_max_x = (float(tx + 1) - cx) * t_delta_x if dir_x > 0.0 else (cx - float(tx)) * t_delta_x
                    t_max_y = (float(ty + 1) - cy) * t_delta_y if dir_y > 0.0 else (cy - float(ty)) * t_delta_y

                    curr_d = 0.0
                    hit_val = 0.0
                    scent_accum = 0.0
                    scent_samples = 0
                    while curr_d < vision_max_dist:
                        if t_max_x < t_max_y:
                            curr_d = t_max_x
                            t_max_x += t_delta_x
                            tx += step_x
                        else:
                            curr_d = t_max_y
                            t_max_y += t_delta_y
                            ty += step_y

                        if 0 <= tx < map_w and 0 <= ty < map_h:
                            flat_tile = ty * map_w + tx
                            if (trail_bits[flat_tile // 64] & np.uint64(1 << (flat_tile % 64))) != np.uint64(0):
                                scent_accum += 1.0
                            scent_samples += 1

                        if tx < 0 or tx >= map_w or ty < 0 or ty >= map_h or s_map[ty, tx] == 1:
                            hit_dist = curr_d if curr_d < vision_max_dist else vision_max_dist
                            val = 1.0 - (hit_dist / vision_max_dist)
                            hit_val = val if val > 0.0 else 0.0
                            if r == num_rays - 2: left_wall_d = hit_dist
                            if r == num_rays - 1: right_wall_d = hit_dist
                            break

                    base_buf[r] = hit_val
                    base_buf[num_rays + r] = (scent_accum / float(scent_samples)) if scent_samples > 0 else 0.0

            idx = num_rays * 2
            vel_x = (cx - prev_x) / (move_speed if move_speed > 1e-4 else 1e-4)
            vel_y = (cy - prev_y) / (move_speed if move_speed > 1e-4 else 1e-4)
            prev_x = cx
            prev_y = cy

            center_error = (left_wall_d - right_wall_d) / vision_max_dist
            diff_ce = 1.0 - abs(center_error)
            centering_accum += (diff_ce if diff_ce > 0.0 else 0.0)

            base_buf[idx] = math.sqrt(vel_x * vel_x + vel_y * vel_y)
            base_buf[idx + 1] = health
            base_buf[idx + 2] = (cx - sx_f) / float(map_w)
            base_buf[idx + 3] = (cy - sy_f) / float(map_h)
            base_buf[idx + 4] = vel_x
            base_buf[idx + 5] = center_error
            idx += 6

            if act_cardinal > 0.5:
                d_c0 = abs(((-math.pi * 0.5) - heading) % two_pi)
                if d_c0 > math.pi: d_c0 = abs(d_c0 - two_pi)
                val_0 = 1.0 - d_c0 / (math.pi * 0.5)
                base_buf[idx] = val_0 if val_0 > 0.0 else 0.0

                d_c1 = abs((0.0 - heading) % two_pi)
                if d_c1 > math.pi: d_c1 = abs(d_c1 - two_pi)
                val_1 = 1.0 - d_c1 / (math.pi * 0.5)
                base_buf[idx + 1] = val_1 if val_1 > 0.0 else 0.0

                d_c2 = abs(((math.pi * 0.5) - heading) % two_pi)
                if d_c2 > math.pi: d_c2 = abs(d_c2 - two_pi)
                val_2 = 1.0 - d_c2 / (math.pi * 0.5)
                base_buf[idx + 2] = val_2 if val_2 > 0.0 else 0.0

                d_c3 = abs((math.pi - heading) % two_pi)
                if d_c3 > math.pi: d_c3 = abs(d_c3 - two_pi)
                val_3 = 1.0 - d_c3 / (math.pi * 0.5)
                base_buf[idx + 3] = val_3 if val_3 > 0.0 else 0.0
            else:
                base_buf[idx] = 0.0; base_buf[idx+1] = 0.0; base_buf[idx+2] = 0.0; base_buf[idx+3] = 0.0
            idx += 4

            if act_north > 0.5:
                d_north = ((-math.pi * 0.5) - heading) % two_pi
                if d_north > math.pi: d_north -= two_pi
                d_n_r = abs((d_north + offset_rad) % two_pi)
                if d_n_r > math.pi: d_n_r = abs(d_n_r - two_pi)
                d_n_l = abs((d_north - offset_rad) % two_pi)
                if d_n_l > math.pi: d_n_l = abs(d_n_l - two_pi)

                val_nr = 1.0 - d_n_r / (math.pi * 0.5)
                base_buf[idx] = val_nr if val_nr > 0.0 else 0.0
                val_nl = 1.0 - d_n_l / (math.pi * 0.5)
                base_buf[idx + 1] = val_nl if val_nl > 0.0 else 0.0
                val_npr = 1.0 - d_n_r / math.pi
                base_buf[idx + 2] = val_npr if val_npr > 0.0 else 0.0
                val_npl = 1.0 - d_n_l / math.pi
                base_buf[idx + 3] = val_npl if val_npl > 0.0 else 0.0
            else:
                base_buf[idx] = 0.0; base_buf[idx+1] = 0.0; base_buf[idx+2] = 0.0; base_buf[idx+3] = 0.0
            idx += 4

            base_buf[idx] = 0.0; base_buf[idx+1] = 0.0; base_buf[idx+2] = 0.0; base_buf[idx+3] = 0.0
            base_buf[idx+4] = 0.0; base_buf[idx+5] = 0.0; base_buf[idx+6] = 0.0; base_buf[idx+7] = 0.0

            if step == 0:
                for f in range(total_frames):
                    for b in range(base_channels):
                        mem_buf[f * base_channels + b] = base_buf[b]
            else:
                for f in range(total_frames - 1):
                    for b in range(base_channels):
                        mem_buf[f * base_channels + b] = mem_buf[(f + 1) * base_channels + b]
                for b in range(base_channels):
                    mem_buf[(total_frames - 1) * base_channels + b] = base_buf[b]

            for i in range(total_in_dim):
                in_buf[i] = mem_buf[i]

            # Dynamic MLP Forward
            for h in range(hidden_dim):
                acc = b1[cand_idx, h]
                for i in range(total_in_dim):
                    acc += in_buf[i] * w1[cand_idx, i, h]
                h1_buf[h] = acc if acc > 0.0 else 0.0

            for h in range(hidden_dim):
                acc = b2[cand_idx, h]
                for p in range(hidden_dim):
                    acc += h1_buf[p] * w2[cand_idx, p, h]
                h2_buf[h] = acc if acc > 0.0 else 0.0

            if num_hidden_layers >= 3:
                for h in range(hidden_dim):
                    acc = b3[cand_idx, h]
                    for p in range(hidden_dim):
                        acc += h2_buf[p] * w3[cand_idx, p, h]
                    h3_buf[h] = acc if acc > 0.0 else 0.0
                last_hidden = h3_buf
            else:
                last_hidden = h2_buf

            for o in range(4):
                acc = b_out[cand_idx, o]
                for p in range(hidden_dim):
                    acc += last_hidden[p] * w_out[cand_idx, p, o]
                clamped = acc if acc > -500.0 else -500.0
                if clamped > 500.0: clamped = 500.0
                out_buf[o] = 1.0 / (1.0 + math.exp(-clamped))

            l_eff = out_buf[0] - out_buf[1]
            r_eff = out_buf[2] - out_buf[3]
            m_eff = (r_eff + l_eff) * 0.5
            t_eff = (r_eff - l_eff) * 0.5

            heading = (heading + (t_eff * turn_rad)) % two_pi
            step_len = m_eff * move_speed
            nx = cx + math.cos(heading) * step_len
            ny = cy + math.sin(heading) * step_len

            tx = int(math.floor(nx))
            ty = int(math.floor(ny))
            hit = False
            if tx < 0 or tx >= map_w or ty < 0 or ty >= map_h or s_map[ty, tx] == 1:
                hit = True
                h_sub = health - 0.005
                health = h_sub if h_sub > 0.0 else 0.0
            else:
                cx = nx
                cy = ny

                if tx != last_tx or ty != last_ty:
                    total_tile_steps += 1
                    current_tile_frames = 0
                    last_tx = tx
                    last_ty = ty
                else:
                    current_tile_frames += 1
                    if current_tile_frames > 8:
                        hover_frames += 1

                flat_tile = ty * map_w + tx
                if (trail_bits[flat_tile // 64] & np.uint64(1 << (flat_tile % 64))) == np.uint64(0):
                    trail_bits[flat_tile // 64] |= np.uint64(1 << (flat_tile % 64))
                    tiles_explored += 1

            if health <= 0.0:
                is_alive = False

            frames_survived += 1
            curr_tx = int(math.floor(cx))
            curr_ty = int(math.floor(cy))
            curr_d = bfs_grid[curr_ty, curr_tx] if (0 <= curr_tx < map_w and 0 <= curr_ty < map_h) else 9999
            if curr_d < best_dist:
                best_dist = curr_d

            if curr_tx == exit_pos[0] and curr_ty == exit_pos[1]:
                reached_exit = True

            if cand_idx < 25:
                telemetry_out[step, cand_idx, 0] = cx
                telemetry_out[step, cand_idx, 1] = cy
                telemetry_out[step, cand_idx, 2] = heading
                telemetry_out[step, cand_idx, 3] = health
                telemetry_out[step, cand_idx, 4] = float(curr_d)
                telemetry_out[step, cand_idx, 5] = 1.0 if hit else 0.0
                telemetry_out[step, cand_idx, 6] = 1.0 if is_alive else 0.0
                telemetry_out[step, cand_idx, 7] = 1.0 if reached_exit else 0.0

        # ======================================================================
        # ABSOLUTE VOLUME EXPLORATION FITNESS (No Ratio Loopholes)
        # ======================================================================
        diff_best = float(initial_dist - best_dist)
        dist_done = diff_best if diff_best > 0.0 else 0.0
        denom_dist = float(initial_dist) if float(initial_dist) > 1.0 else 1.0
        score_dist = 500.0 * (dist_done / denom_dist)

        # Volume Reward: 100 tiles = 300 pts! 5 tiles = only 15 pts!
        score_explore = float(tiles_explored) * 3.0
        if score_explore > 300.0: score_explore = 300.0

        # Excessive looping penalty (Backtracking once is free, looping 3x+ is penalized)
        excess_steps = float(total_tile_steps - tiles_explored * 2)
        penalty_cycling = (excess_steps * 1.5) if excess_steps > 0.0 else 0.0
        if penalty_cycling > 150.0: penalty_cycling = 150.0

        # Exit Solve Jackpot (+500 flat)
        score_exit = 0.0
        if reached_exit:
            time_bonus = 300.0 * (1.0 - (float(frames_survived) / float(max_steps)))
            score_exit = 500.0 + time_bonus

        total_fitness = score_dist + score_explore + score_exit - penalty_cycling
        if total_fitness < 0.0: total_fitness = 0.0

        scores_out[cand_idx] = total_fitness * (0.8 + 0.2 * health)

        denom_fs = float(frames_survived) if float(frames_survived) > 1.0 else 1.0
        avg_cntr = int((centering_accum / denom_fs) * 100.0)
        denom_steps = float(total_tile_steps) if float(total_tile_steps) > 1.0 else 1.0
        effc_pct = int((float(tiles_explored) / denom_steps) * 100.0)
        dwell_ratio = float(hover_frames) / denom_fs
        sub_p = 1.0 - (dwell_ratio * 2.0)
        pace_pct = int((sub_p if sub_p > 0.0 else 0.0) * 100.0)

        summary_out[cand_idx, 0] = best_dist
        summary_out[cand_idx, 1] = 1 if reached_exit else 0
        summary_out[cand_idx, 2] = tiles_explored
        summary_out[cand_idx, 3] = avg_cntr
        summary_out[cand_idx, 4] = effc_pct
        summary_out[cand_idx, 5] = pace_pct


# ==========================================================================
# 3. Universal Runner Class
# ==========================================================================

class UniversalBatchRunner:
    def __init__(
        self,
        pop_size: int,
        max_steps: int,
        num_rays: int,
        hidden_dim: int,
        memory_frames: int = 2
    ):
        self.pop_size = pop_size
        self.max_steps = max_steps
        self.num_rays = num_rays
        self.hidden_dim = hidden_dim
        self.memory_frames = memory_frames
        self.use_cuda = CUDA_AVAILABLE

        self.allocated_in_dim = -1
        self.d_scores = None
        self.d_summary = None
        self.d_telemetry = None
        self.d_w1 = None; self.d_b1 = None
        self.d_w2 = None; self.d_b2 = None
        self.d_w3 = None; self.d_b3 = None
        self.d_w_out = None; self.d_b_out = None

        if self.use_cuda:
            dev_name = cuda.get_current_device().name
            if isinstance(dev_name, bytes):
                dev_name = dev_name.decode("utf-8")
            print(f"[Engine] CUDA GPU Detected: Accelerating on {dev_name}")
            self.d_scores = cuda.device_array(pop_size, dtype=np.float32)
            self.d_summary = cuda.device_array((pop_size, 6), dtype=np.int32)
            self.d_telemetry = cuda.device_array((max_steps, 25, 8), dtype=np.float32)

    def _ensure_buffers(self, in_dim: int, h_dim: int, pop_sz: int):
        if self.allocated_in_dim == in_dim:
            return

        self.allocated_in_dim = in_dim
        self.h_w1 = np.zeros((pop_sz, in_dim, h_dim), dtype=np.float32)
        self.h_b1 = np.zeros((pop_sz, h_dim), dtype=np.float32)
        self.h_w2 = np.zeros((pop_sz, h_dim, h_dim), dtype=np.float32)
        self.h_b2 = np.zeros((pop_sz, h_dim), dtype=np.float32)
        self.h_w3 = np.zeros((pop_sz, h_dim, h_dim), dtype=np.float32)
        self.h_b3 = np.zeros((pop_sz, h_dim), dtype=np.float32)
        self.h_w_out = np.zeros((pop_sz, h_dim, 4), dtype=np.float32)
        self.h_b_out = np.zeros((pop_sz, 4), dtype=np.float32)

        if self.use_cuda:
            self.d_w1 = cuda.device_array((pop_sz, in_dim, h_dim), dtype=np.float32)
            self.d_b1 = cuda.device_array((pop_sz, h_dim), dtype=np.float32)
            self.d_w2 = cuda.device_array((pop_sz, h_dim, h_dim), dtype=np.float32)
            self.d_b2 = cuda.device_array((pop_sz, h_dim), dtype=np.float32)
            self.d_w3 = cuda.device_array((pop_sz, h_dim, h_dim), dtype=np.float32)
            self.d_b3 = cuda.device_array((pop_sz, h_dim), dtype=np.float32)
            self.d_w_out = cuda.device_array((pop_sz, h_dim, 4), dtype=np.float32)
            self.d_b_out = cuda.device_array((pop_sz, 4), dtype=np.float32)

    def run_generation_from_tensor(self, map_data, pathfinder, pop_weights, profile, move_speed=0.20):
        pop_sz = self.pop_size
        h_dim = profile.neurons
        num_hidden = profile.hidden_layers
        mem_f = profile.memory_frames

        base_channels = (profile.vision_rays * 2) + 6 + (4 if profile.use_binocular_gps_compasses else 2) + 4 + 4 + 4
        total_in_dim = base_channels * (1 + mem_f)

        self._ensure_buffers(total_in_dim, h_dim, pop_sz)

        w1_sz = total_in_dim * h_dim
        b1_sz = h_dim
        w2_sz = h_dim * h_dim
        b2_sz = h_dim

        o = 0
        self.h_w1[:] = pop_weights[:, o:o+w1_sz].reshape((pop_sz, total_in_dim, h_dim)); o += w1_sz
        self.h_b1[:] = pop_weights[:, o:o+b1_sz]; o += b1_sz
        self.h_w2[:] = pop_weights[:, o:o+w2_sz].reshape((pop_sz, h_dim, h_dim)); o += w2_sz
        self.h_b2[:] = pop_weights[:, o:o+b2_sz]; o += b2_sz

        if num_hidden >= 3:
            w3_sz = h_dim * h_dim
            b3_sz = h_dim
            self.h_w3[:] = pop_weights[:, o:o+w3_sz].reshape((pop_sz, h_dim, h_dim)); o += w3_sz
            self.h_b3[:] = pop_weights[:, o:o+b3_sz]; o += b3_sz
        else:
            self.h_w3.fill(0.0)
            self.h_b3.fill(0.0)

        w_out_sz = h_dim * 4
        self.h_w_out[:] = pop_weights[:, o:o+w_out_sz].reshape((pop_sz, h_dim, 4)); o += w_out_sz
        self.h_b_out[:] = pop_weights[:, o:o+4]

        grid = map_data.numpy_grid
        bfs = pathfinder.numpy_dist
        start_pos = np.array(map_data.start_pos, dtype=np.int32)
        exit_pos = np.array(map_data.exit_pos, dtype=np.int32)
        initial_dist = int(pathfinder.get_step_distance(*map_data.start_pos))

        half_arc_rad = float(math.radians(profile.vision_arc_angle * 0.5))
        ray_step_rad = float((2.0 * half_arc_rad) / max(1, profile.vision_rays - 1))
        offset_rad = float(math.radians(profile.target_compasses_offset_angle))
        turn_rad = float(math.radians(profile.turn_speed / 60.0))

        act_gps = 1.0 if getattr(profile, "activate_gps_compass", False) else 0.0
        act_cardinal = 1.0 if getattr(profile, "activate_cardinal_compass", False) else 0.0
        act_north = 1.0 if getattr(profile, "activate_north_compass", False) else 0.0
        act_exit = 1.0 if getattr(profile, "activate_exit_compass", False) else 0.0

        if self.use_cuda:
            d_grid = cuda.to_device(grid)
            d_bfs = cuda.to_device(bfs)
            d_start = cuda.to_device(start_pos)
            d_exit = cuda.to_device(exit_pos)

            self.d_w1.copy_to_device(self.h_w1)
            self.d_b1.copy_to_device(self.h_b1)
            self.d_w2.copy_to_device(self.h_w2)
            self.d_b2.copy_to_device(self.h_b2)
            self.d_w3.copy_to_device(self.h_w3)
            self.d_b3.copy_to_device(self.h_b3)
            self.d_w_out.copy_to_device(self.h_w_out)
            self.d_b_out.copy_to_device(self.h_b_out)

            threads = 128
            blocks = (pop_sz + threads - 1) // threads

            batch_simulate_cuda[blocks, threads](
                d_grid, d_bfs, d_start, d_exit,
                self.d_w1, self.d_b1, self.d_w2, self.d_b2, self.d_w3, self.d_b3, self.d_w_out, self.d_b_out,
                self.d_scores, self.d_summary, self.d_telemetry,
                initial_dist, int(self.max_steps), int(profile.vision_rays),
                int(base_channels), int(total_in_dim), int(mem_f), int(h_dim), int(num_hidden),
                half_arc_rad, ray_step_rad, float(profile.vision_max_dist), float(move_speed), turn_rad,
                float(profile.agent_diameter_ratio * 0.5), offset_rad,
                act_gps, act_cardinal, act_north, act_exit
            )
            cuda.synchronize()

            scores = self.d_scores.copy_to_host().tolist()
            summary = self.d_summary.copy_to_host()
            telemetry = self.d_telemetry.copy_to_host()
            return scores, summary, telemetry
        else:
            scores_out = np.zeros(pop_sz, dtype=np.float32)
            summary_out = np.zeros((pop_sz, 6), dtype=np.int32)
            telemetry_out = np.zeros((self.max_steps, 25, 8), dtype=np.float32)

            batch_simulate_cpu(
                grid, bfs, start_pos, exit_pos,
                self.h_w1, self.h_b1, self.h_w2, self.h_b2, self.h_w3, self.h_b3, self.h_w_out, self.h_b_out,
                scores_out, summary_out, telemetry_out,
                initial_dist, int(self.max_steps), int(profile.vision_rays),
                int(base_channels), int(total_in_dim), int(mem_f), int(h_dim), int(num_hidden),
                half_arc_rad, ray_step_rad, float(profile.vision_max_dist), float(move_speed), turn_rad,
                float(profile.agent_diameter_ratio * 0.5), offset_rad,
                act_gps, act_cardinal, act_north, act_exit
            )
            return scores_out.tolist(), summary_out, telemetry_out
