"""
Disk persistence manager for saving and loading trained neural weights.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Any, Optional, List
import numpy as np

from entities.agent_profile_registry import ResolvedAgentProfile
from neural.network import NeuralNetwork


@dataclass(frozen=True, slots=True)
class SavedBrainMetadata:
    """
    Immutable container holding discovered brain path and signature metadata.
    """

    file_path: Path
    file_name: str
    clean_title: str
    vision_rays: int
    memory_frames: int
    hidden_layers: int
    neurons: int
    use_binocular_gps: bool
    use_linear_speed: bool
    is_legacy: bool


class BrainPersistence:
    """
    Saves, discovers, & loads candidate network weights using signature tags.
    """

    SIG_REGEX: re.Pattern = re.compile(
        r"^(?P<title>.+?)_v(?P<v>\d+)_m(?P<m>\d+)_h(?P<h>\d+)_n(?P<n>\d+)_"
        r"(?P<b>b[01])_(?P<lin>lin[01])\.npz$"
    )

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        """
        Initializes storage directory in project root and metadata cache.
        """
        root_dir: Path = Path(__file__).resolve().parents[1]
        self.storage_dir: Path = storage_dir or (root_dir / "saved_brains")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cached_brain_list: Optional[List[SavedBrainMetadata]] = None

    def discover_saved_brains(
        self, force_refresh: bool = False
    ) -> List[SavedBrainMetadata]:
        """
        Scans saved_brains directory and extracts cached brain metadata list.
        """
        if self._cached_brain_list is not None and not force_refresh:
            return self._cached_brain_list

        discovered: List[SavedBrainMetadata] = []
        if not self.storage_dir.exists():
            self._cached_brain_list = discovered
            return discovered

        for file_path in self.storage_dir.glob("*.npz"):
            if not file_path.is_file():
                continue

            match = self.SIG_REGEX.match(file_path.name)
            if match:
                title_str: str = match.group("title")
                v_val: int = int(match.group("v"))
                m_val: int = int(match.group("m"))
                h_val: int = int(match.group("h"))
                n_val: int = int(match.group("n"))
                b_val: bool = match.group("b") == "b1"
                lin_val: bool = match.group("lin") == "lin1"

                meta = SavedBrainMetadata(
                    file_path=file_path,
                    file_name=file_path.name,
                    clean_title=title_str,
                    vision_rays=v_val,
                    memory_frames=m_val,
                    hidden_layers=h_val,
                    neurons=n_val,
                    use_binocular_gps=b_val,
                    use_linear_speed=lin_val,
                    is_legacy=False,
                )
            else:
                meta = SavedBrainMetadata(
                    file_path=file_path,
                    file_name=file_path.name,
                    clean_title=file_path.stem,
                    vision_rays=15,
                    memory_frames=2,
                    hidden_layers=3,
                    neurons=40,
                    use_binocular_gps=True,
                    use_linear_speed=True,
                    is_legacy=True,
                )
            discovered.append(meta)

        discovered.sort(key=lambda item: (item.clean_title, item.file_name))
        self._cached_brain_list = discovered
        return discovered

    def get_brain_file_path(self, profile: ResolvedAgentProfile) -> Path:
        """
        Returns file path built from profile name and topology signature.
        """
        v: int = profile.vision_rays
        m: int = profile.memory_frames
        h: int = profile.hidden_layers
        n: int = profile.neurons
        b_tag: str = (
            "b1" if profile.use_binocular_gps_compasses else "b0"
        )
        lin_tag: str = (
            "lin1" if profile.use_linear_speed_output else "lin0"
        )
        sig_filename: str = (
            f"{profile.profile_name}_v{v}_m{m}_h{h}_n{n}_"
            f"{b_tag}_{lin_tag}.npz"
        )
        return self.storage_dir / sig_filename

    def save_brain(
        self,
        profile_name: str,
        network: NeuralNetwork,
        profile: ResolvedAgentProfile,
        context: str = "training",
    ) -> Path:
        """
        Saves network layer weight and bias matrices as compressed archive.
        """
        file_path: Path = self.get_brain_file_path(profile)

        save_dict: Dict[str, Any] = {
            "metadata_input_size": np.array(
                network.layers[0].weights.shape[0]
            ),
            "metadata_hidden_layers": np.array(profile.hidden_layers),
            "metadata_neurons": np.array(profile.neurons),
            "metadata_output_size": np.array(
                network.layers[-1].weights.shape[1]
            ),
            "num_layers": np.array(len(network.layers)),
        }

        for idx, layer in enumerate(network.layers):
            save_dict[f"layer_{idx}_weights"] = layer.weights
            save_dict[f"layer_{idx}_biases"] = layer.biases

        try:
            np.savez_compressed(file_path, **save_dict)
            ctx_str: str = f" from {context}" if context else ""
            print(
                f"[Persistence] Saved winning brain weights{ctx_str} -> "
                f"{file_path.name}"
            )
            self.discover_saved_brains(force_refresh=True)
        except Exception as e:
            print(f"[Warning] Could not save brain weights: {e}")

        return file_path

    def load_brain(
        self,
        profile_name: str,
        target_network: NeuralNetwork,
        profile: ResolvedAgentProfile,
        context: str = "visualizer playback",
        verbose: bool = True,
    ) -> bool:
        """
        Loads saved weights into target_network with signature validation.
        """
        file_path: Path = self.get_brain_file_path(profile)
        legacy_path: Path = self.storage_dir / f"{profile_name}.npz"

        target_path: Path = file_path
        if not file_path.exists():
            if legacy_path.exists():
                target_path = legacy_path
            else:
                return False

        try:
            archive = np.load(target_path)
        except Exception as e:
            if verbose:
                print(
                    f"[Warning] Failed to load brain archive "
                    f"{target_path.name}: {e}"
                )
            return False

        saved_in: int = int(archive.get("metadata_input_size", 0))
        saved_hl: int = int(archive.get("metadata_hidden_layers", 0))
        saved_neu: int = int(archive.get("metadata_neurons", 0))
        saved_out: int = int(archive.get("metadata_output_size", 0))

        target_in: int = target_network.layers[0].weights.shape[0]
        target_hl: int = profile.hidden_layers
        target_neu: int = profile.neurons
        target_out: int = target_network.layers[-1].weights.shape[1]

        mismatch: bool = (
            saved_in != target_in
            or saved_hl != target_hl
            or saved_neu != target_neu
            or saved_out != target_out
        )

        if mismatch:
            if verbose:
                print(
                    f"[Persistence] Topology signature mismatch for "
                    f"'{target_path.name}'. Starting fresh Generation 0 "
                    f"evolution for active profile."
                )
            return False

        num_layers: int = int(
            archive.get("num_layers", len(target_network.layers))
        )
        for idx in range(num_layers):
            target_network.layers[idx].weights = archive[
                f"layer_{idx}_weights"
            ]
            target_network.layers[idx].biases = archive[
                f"layer_{idx}_biases"
            ]

        if verbose:
            ctx_str: str = f" for {context}" if context else ""
            print(
                f"[Persistence] Loaded brain weights{ctx_str} from "
                f"{target_path.name}"
            )
        return True
