"""
Contiguous tensor recorder logging simulation timelines for playback without RAM bloat,
keeping archives extremely compact by discarding full frame telemetry during training.
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
    Stores playback frame data using lightweight metadata and weight bundlers,
    preventing massive temporary archive size warnings and OOM bloat.
    """

    def __init__(
        self,
        cache_filename: str = ".runtime_cache.npz"
    ) -> None:
        self.cache_path: Path = Path(cache_filename)
        self.generations_history: List[Dict[str, Any]] = []
        self.gen_metadata: List[Dict[str, Any]] = []
        self.telemetry_bundler: Optional[TelemetryBundler] = None
        self.weight_bundler: Optional[WeightBundler] = None
        self._max_ram_generations: int = 10

    def allocate_session_buffers(
        self,
        max_steps: int,
        pop_size: int,
        num_generations: int,
        param_count: int
    ) -> None:
        # Allocate minimal 1-step telemetry buffers (shape: 1 x pop_size x 8) to keep archives super tiny
        self.telemetry_bundler = TelemetryBundler(1, pop_size)
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
        # Fully disabled during training to keep memory footprint close to zero and archives tiny
        pass

    def finalize_generation(
        self,
        gen_idx: int,
        map_data: MapData,
        raw_scores: List[float],
        norm_scores: List[float],
        actual_steps: int,
        pop_networks: List[NeuralNetwork]
    ) -> None:
        if (
            self.telemetry_bundler is None or
            self.weight_bundler is None
        ):
            return

        # Force clamp to 1 step in bundler to eliminate memory allocation
        self.telemetry_bundler.finalize_generation(1)
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

        if len(self.gen_metadata) >= self._max_ram_generations:
            self.save_temporary_disk_archive(append=True)

        if self.weight_bundler and gen_idx < self.weight_bundler.num_generations - 1:
            self.telemetry_bundler.allocate_generation_buffer()

    def save_temporary_disk_archive(self, append: bool = False) -> None:
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
        if not append:
            self.flush_training_memory()

    def load_temporary_disk_archive(self) -> bool:
        if not self.cache_path.exists():
            return False

        w_b, t_b, history = ArchiveBridge.load_archive(self.cache_path)
        self.weight_bundler = w_b
        self.telemetry_bundler = t_b
        self.generations_history = history
        return True

    def flush_training_memory(self) -> None:
        self.gen_metadata.clear()
        if self.telemetry_bundler is not None:
            self.telemetry_bundler._curr_buffer = None
            self.telemetry_bundler._generations_telemetry.clear()
        gc.collect()

    def flush_replay_memory(self) -> None:
        self.generations_history.clear()
        if self.telemetry_bundler is not None:
            self.telemetry_bundler.clear_all()
        self.telemetry_bundler = None
        self.weight_bundler = None
        gc.collect()

    def remove_temporary_disk_archive(self) -> None:
        ArchiveBridge.unlink_archive(self.cache_path)

    def get_generation_data(self, gen_idx: int) -> Dict[str, Any]:
        safe_idx: int = max(
            0, min(gen_idx, len(self.generations_history) - 1)
        )
        return self.generations_history[safe_idx]
