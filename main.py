"""
Application entry point executing pre-training or launching visualizer.
"""

import os
import sys
import multiprocessing

# Phase 1: Environment & Hardware Isolation (Exclude Core 0, use all other cores)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

total_cpus = multiprocessing.cpu_count()
if total_cpus > 1:
    # Pin process affinity to all cores EXCEPT Core 0 (cores 1, 2, 3, ... N-1)
    if hasattr(os, "sched_setaffinity"):
        try:
            target_cores = set(range(1, total_cpus))
            os.sched_setaffinity(0, target_cores)
            print(f"[Hardware] CPU Affinity locked to cores: {sorted(list(target_cores))} (Core 0 excluded).")
        except Exception as e:
            print(f"[Warning] Could not set CPU affinity: {e}")

    usable_workers = str(total_cpus - 1)
else:
    usable_workers = "1"

os.environ["OMP_NUM_THREADS"] = usable_workers
os.environ["MKL_NUM_THREADS"] = usable_workers
os.environ["OPENBLAS_NUM_THREADS"] = usable_workers
os.environ["NUMEXPR_NUM_THREADS"] = usable_workers
os.environ["NUMBA_NUM_THREADS"] = usable_workers

import config
from core.warmup import warmup_jit
from evolution.trainer import HeadlessTrainer
from visualization.app_window import AppWindow
from visualization.endless_app_window import EndlessAppWindow


def main() -> None:
    """
    Runs endless window, or legacy pre-training and replay visualizer.
    """
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
