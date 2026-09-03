"""
Application entry point executing pre-training or launching visualizer.
"""

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import config
from evolution.trainer import HeadlessTrainer
from visualization.app_window import AppWindow
from visualization.endless_app_window import EndlessAppWindow


def main() -> None:
    """
    Runs endless window, or legacy pre-training and replay visualizer.
    """
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
