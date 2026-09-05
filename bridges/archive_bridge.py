"""
Uncompressed disk serialization, fail-fast deserialization, & unlinking.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

import config
from bridges.weight_bundler import WeightBundler
from bridges.telemetry_bundler import TelemetryBundler


class ArchiveBridge:
    """
    Manages atomic uncompressed disk I/O and instant file unlinking.
    """

    @staticmethod
    def save_archive(
        cache_path: Path,
        weight_bundler: WeightBundler,
        telemetry_bundler: TelemetryBundler,
        gen_metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Serializes weight tensor, telemetry arrays, & metadata uncompressed.
        """
        num_gens: int = len(gen_metadata)
        if num_gens == 0:
            print("[Error] Cannot save empty generation metadata to archive!")
            sys.exit(1)

        w_tensor = weight_bundler.master_tensor
        est_bytes: int = int(w_tensor.nbytes)
        for gen_idx in range(num_gens):
            t_arr = telemetry_bundler.get_generation_telemetry(gen_idx)
            est_bytes += int(t_arr.nbytes)

        est_mb: float = est_bytes / (1024.0 * 1024.0)
        max_allowed_mb: float = float(config.MAX_TEMP_CACHE_SIZE_MB)

        if est_mb > max_allowed_mb:
            print(
                f"[Warning] Estimated archive size "
                f"({est_mb:.2f} MB) exceeds "
                f"MAX_TEMP_CACHE_SIZE_MB "
                f"({max_allowed_mb:.2f} MB). "
                f"Continuing because the cache is optional."
            )

        archive_dict: Dict[str, Any] = {
            "num_generations": np.array(num_gens, dtype=np.int64),
            "pop_weights": w_tensor
        }

        for idx, g_data in enumerate(gen_metadata):
            pfx: str = f"gen_{idx}_"
            archive_dict[pfx + "telemetry"] = (
                telemetry_bundler.get_generation_telemetry(idx)
            )
            archive_dict[pfx + "chunks"] = np.array(
                g_data["bitmask_chunks"], dtype=np.uint64
            )
            archive_dict[pfx + "start"] = np.array(
                g_data["start_pos"], dtype=np.int64
            )
            archive_dict[pfx + "exit"] = np.array(
                g_data["exit_pos"], dtype=np.int64
            )
            archive_dict[pfx + "target_seq"] = np.array(
                g_data.get("target_sequence", [g_data["exit_pos"]]),
                dtype=np.int64
            )
            archive_dict[pfx + "width"] = np.array(
                g_data["map_width"], dtype=np.int64
            )
            archive_dict[pfx + "height"] = np.array(
                g_data["map_height"], dtype=np.int64
            )
            archive_dict[pfx + "raw_scores"] = np.array(
                g_data["raw_scores"], dtype=np.float64
            )
            archive_dict[pfx + "norm_scores"] = np.array(
                g_data["normalized_scores"], dtype=np.float64
            )
            archive_dict[pfx + "winner"] = np.array(
                g_data["winner_index"], dtype=np.int64
            )

        try:
            np.savez(cache_path, **archive_dict)
        except Exception as e:
            print(f"[Error] Failed to write uncompressed archive: {e}")
            sys.exit(1)

    @staticmethod
    def load_archive(
        cache_path: Path
    ) -> Tuple[WeightBundler, TelemetryBundler, List[Dict[str, Any]]]:
        """
        Deserializes archive into RAM and unlinks disk file immediately.
        """
        if not cache_path.exists():
            print(
                f"[Error] Archive file missing for load: {cache_path.name}"
            )
            sys.exit(1)

        try:
            archive = np.load(cache_path)
        except Exception as e:
            print(f"[Error] Failed to load archive {cache_path.name}: {e}")
            sys.exit(1)

        if "num_generations" not in archive or "pop_weights" not in archive:
            print(
                f"[Error] Corrupted archive {cache_path.name}: "
                f"Missing required master keys."
            )
            sys.exit(1)

        num_gens: int = int(archive["num_generations"])
        pop_weights = archive["pop_weights"]

        pop_size: int = int(pop_weights.shape[1])
        param_count: int = int(pop_weights.shape[2])

        w_bundler = WeightBundler(num_gens, pop_size, param_count)
        w_bundler.set_master_tensor(pop_weights)

        gen_0_t = archive["gen_0_telemetry"]
        max_steps_est: int = int(gen_0_t.shape[0])

        t_bundler = TelemetryBundler(max_steps_est, pop_size)
        t_bundler.clear_all()

        generations_history: List[Dict[str, Any]] = []

        for idx in range(num_gens):
            pfx: str = f"gen_{idx}_"

            if (pfx + "telemetry") not in archive:
                print(
                    f"[Error] Archive {cache_path.name} missing key "
                    f"'{pfx}telemetry'."
                )
                sys.exit(1)

            t_arr = archive[pfx + "telemetry"]
            t_bundler.all_generations_telemetry.append(t_arr)

            chunks_raw = archive[pfx + "chunks"]
            chunks_list: List[int] = [int(c) for c in chunks_raw]

            start_raw = archive[pfx + "start"].astype(int)
            exit_raw = archive[pfx + "exit"].astype(int)

            if (pfx + "target_seq") in archive:
                seq_raw = archive[pfx + "target_seq"].astype(int)
                t_seq = [
                    (int(seq_raw[i][0]), int(seq_raw[i][1]))
                    for i in range(len(seq_raw))
                ]
            else:
                t_seq = [(int(exit_raw[0]), int(exit_raw[1]))]

            g_data: Dict[str, Any] = {
                "generation": idx,
                "bitmask_chunks": chunks_list,
                "start_pos": (int(start_raw[0]), int(start_raw[1])),
                "exit_pos": (int(exit_raw[0]), int(exit_raw[1])),
                "target_sequence": t_seq,
                "map_width": int(archive[pfx + "width"]),
                "map_height": int(archive[pfx + "height"]),
                "telemetry": t_arr,
                "raw_scores": archive[pfx + "raw_scores"].tolist(),
                "normalized_scores": archive[pfx + "norm_scores"].tolist(),
                "winner_index": int(archive[pfx + "winner"]),
                "pop_weights": pop_weights[idx]
            }
            generations_history.append(g_data)

        archive.close()
        ArchiveBridge.unlink_archive(cache_path)

        return w_bundler, t_bundler, generations_history

    @staticmethod
    def unlink_archive(cache_path: Path) -> None:
        """
        Instantly deletes temporary archive file from disk if present.
        """
        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError as e:
                print(f"[Warning] Could not unlink cache file: {e}")
