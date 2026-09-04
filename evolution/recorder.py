"""
Contiguous tensor recorder logging simulation timelines for playback.
"""

import gc
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from core.map_data import MapData
from neural.network import NeuralNetwork
from bridges.weight_bundler import WeightBundler
from bridges.telemetry_bundler import TelemetryBundler
from bridges.archive_bridge import ArchiveBridge


class FrameRecorder:
    """
    Stores playback frame data using contiguous tensor bundlers.
    """

    def __init__(
        self,
        cache_filename: str = ".runtime_cache.npz"
    ) -> None:
        """
        Initializes history storage and temporary cache file paths.
        """
        self.cache_path: Path = Path(cache_filename)
        self.generations_history: List[Dict[str, Any]] = []
        self.gen_metadata: List[Dict[str, Any]] = []
        self.telemetry_bundler: Optional[TelemetryBundler] = None
        self.weight_bundler: Optional[WeightBundler] = None

    def allocate_session_buffers(
        self,
        max_steps: int,
        pop_size: int,
        num_generations: int,
        param_count: int
    ) -> None:
        """
        Initializes zero-allocation telemetry & weight bundlers.
        """
        self.telemetry_bundler = TelemetryBundler(max_steps, pop_size)
        self.weight_bundler = WeightBundler(
            num_generations, pop_size, param_count
        )
        self.gen_metadata.clear()
        self.generations_history.clear()

    def record_step_data(
        self,
        step_idx: int,
        cand_idx: int,
        x: float,
        y: float,
        heading: float,
        health: float,
        dist: float,
        hit_wall: bool,
        is_alive: bool,
        reached_exit: bool
    ) -> None:
        """
        Writes step outputs directly into telemetry bundler.
        """
        if self.telemetry_bundler is None:
            return

        self.telemetry_bundler.record_step_data(
            step_idx=step_idx,
            cand_idx=cand_idx,
            x=x,
            y=y,
            heading=heading,
            health=health,
            dist=dist,
            hit_wall=hit_wall,
            is_alive=is_alive,
            reached_exit=reached_exit
        )

    def finalize_generation(
        self,
        gen_idx: int,
        map_data: MapData,
        raw_scores: List[float],
        norm_scores: List[float],
        actual_steps: int,
        pop_networks: List[NeuralNetwork]
    ) -> None:
        """
        Truncates telemetry and records generation candidate weights.
        """
        if (
            self.telemetry_bundler is None or
            self.weight_bundler is None
        ):
            return

        self.telemetry_bundler.finalize_generation(actual_steps)
        self.weight_bundler.record_generation_weights(
            gen_idx, pop_networks
        )

        winner_idx: int = int(np.argmax(norm_scores)) if norm_scores else 0
        g_data: Dict[str, Any] = {
            "generation": gen_idx,
            "bitmask_chunks": map_data.bitmask_chunks,
            "start_pos": map_data.start_pos,
            "exit_pos": map_data.exit_pos,
            "target_sequence": list(map_data.target_sequence),
            "map_width": map_data.width,
            "map_height": map_data.height,
            "raw_scores": raw_scores,
            "normalized_scores": norm_scores,
            "winner_index": winner_idx
        }
        self.gen_metadata.append(g_data)

        if gen_idx < self.weight_bundler.num_generations - 1:
            self.telemetry_bundler.allocate_generation_buffer()

    def save_temporary_disk_archive(self) -> None:
        """
        Flushes tensors to uncompressed disk archive and releases RAM.
        """
        if (
            self.telemetry_bundler is None or
            self.weight_bundler is None or
            not self.gen_metadata
        ):
            return

        ArchiveBridge.save_archive(
            self.cache_path,
            self.weight_bundler,
            self.telemetry_bundler,
            self.gen_metadata
        )
        self.flush_training_memory()

    def load_temporary_disk_archive(self) -> bool:
        """
        Loads uncompressed archive into RAM and unlinks disk file.
        """
        if not self.cache_path.exists():
            return False

        w_b, t_b, history = ArchiveBridge.load_archive(self.cache_path)
        self.weight_bundler = w_b
        self.telemetry_bundler = t_b
        self.generations_history = history
        return True

    def flush_training_memory(self) -> None:
        """
        Frees training-side references and triggers garbage collection.
        """
        self.gen_metadata.clear()
        if self.telemetry_bundler is not None:
            self.telemetry_bundler._curr_buffer = None
        gc.collect()

    def flush_replay_memory(self) -> None:
        """
        Frees visualizer-side references and triggers garbage collection.
        """
        self.generations_history.clear()
        if self.telemetry_bundler is not None:
            self.telemetry_bundler.clear_all()
        self.telemetry_bundler = None
        self.weight_bundler = None
        gc.collect()

    def remove_temporary_disk_archive(self) -> None:
        """
        Unlinks temporary cache file if present.
        """
        ArchiveBridge.unlink_archive(self.cache_path)

    def get_generation_data(self, gen_idx: int) -> Dict[str, Any]:
        """
        Retrieves recorded history data for a specific generation.
        """
        safe_idx: int = max(
            0, min(gen_idx, len(self.generations_history) - 1)
        )
        return self.generations_history[safe_idx]
