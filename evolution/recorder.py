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
    def __init__(
        self,
        cache_filename: str = ".runtime_cache.npz"
    ) -> None:
        self.cache_path: Path = Path(cache_filename)
        self.generations_history: List[Dict[str, Any]] = []
        self.gen_metadata: List[Dict[str, Any]] = []
        self.telemetry_bundler: Optional[TelemetryBundler] = None
        self.weight_bundler: Optional[WeightBundler] = None
        self.record_pop_size: int = 16
        self._max_history: int = 100

    def allocate_session_buffers(
        self,
        max_steps: int,
        pop_size: int,
        num_generations: int,
        param_count: int
    ) -> None:
        self.record_pop_size = min(pop_size, 16)
        self.telemetry_bundler = TelemetryBundler(1, self.record_pop_size)
        self.weight_bundler = WeightBundler(
            min(num_generations, self._max_history), self.record_pop_size, param_count
        )
        self.gen_metadata.clear()
        self.generations_history.clear()

    def record_step_data(self, *args, **kwargs) -> None:
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
        if self.telemetry_bundler is None or self.weight_bundler is None:
            return

        rec_size = getattr(self, "record_pop_size", min(len(pop_networks), 16))
        if norm_scores and len(norm_scores) == len(pop_networks):
            ranked_indices = np.argsort(norm_scores)[::-1][:rec_size]
            top_networks = [pop_networks[i] for i in ranked_indices]
            top_raw_scores = [raw_scores[i] for i in ranked_indices]
            top_norm_scores = [norm_scores[i] for i in ranked_indices]
        else:
            top_networks = pop_networks[:rec_size]
            top_raw_scores = raw_scores[:rec_size]
            top_norm_scores = norm_scores[:rec_size]

        self.telemetry_bundler.finalize_generation(1)

        # Slot into rolling buffer
        slot_idx = gen_idx % self.weight_bundler.num_generations
        self.weight_bundler.record_generation_weights(slot_idx, top_networks)

        g_data: Dict[str, Any] = {
            "generation": gen_idx,
            "bitmask_chunks": map_data.bitmask_chunks,
            "start_pos": map_data.start_pos,
            "exit_pos": map_data.exit_pos,
            "target_sequence": list(map_data.target_sequence),
            "map_width": map_data.width,
            "map_height": map_data.height,
            "raw_scores": top_raw_scores,
            "normalized_scores": top_norm_scores,
            "winner_index": 0
        }
        self.gen_metadata.append(g_data)
        if len(self.gen_metadata) > self._max_history:
            self.gen_metadata.pop(0)

        # Zero disk I/O during training hot loop

    def save_temporary_disk_archive(self, append: bool = False) -> None:
        if self.telemetry_bundler is None or self.weight_bundler is None or not self.gen_metadata:
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
        safe_idx = max(0, min(gen_idx, len(self.generations_history) - 1))
        return self.generations_history[safe_idx]
