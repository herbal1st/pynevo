"""
Application entry point executing pre-training or launching visualizer.
"""

import os
import multiprocessing

# Phase 1: Environment & Hardware Isolation
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Allocate CPU thread pools across OpenMP, MKL, OpenBLAS, and Numba
_cpu_count = str(max(1, multiprocessing.cpu_count()))
os.environ.setdefault("OMP_NUM_THREADS", _cpu_count)
os.environ.setdefault("MKL_NUM_THREADS", _cpu_count)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _cpu_count)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _cpu_count)
os.environ.setdefault("NUMBA_NUM_THREADS", _cpu_count)

import config
from core.warmup import warmup_jit
from evolution.trainer import HeadlessTrainer
from visualization.app_window import AppWindow
from visualization.endless_app_window import EndlessAppWindow


def main() -> None:
    """
    Runs endless window, or legacy pre-training and replay visualizer.
    """
    # Phase 4: Compile JIT kernels before simulation / GUI loops
    warmup_jit()

    use_endless: bool = getattr(config, "USE_ENDLESS_MODE", False)
    if use_endless:
        app: EndlessAppWindow = EndlessAppWindow()
        app.run()
    else:
        trainer: HeadlessTrainer = HeadlessTrainer()
        recorder = trainer.run_training_session()

        replay_app: AppWindow = AppWindow(recorder)
        replay_app.run()


if __name__ == "__main__":
    main()