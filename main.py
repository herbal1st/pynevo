"""
Application entry point executing pre-training and launching visualizer GUI.
"""

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from evolution.trainer import HeadlessTrainer
from visualization.app_window import AppWindow


def main() -> None:
    """
    Runs headless neuroevolution, then launches visualizer GUI.
    """
    trainer: HeadlessTrainer = HeadlessTrainer()
    recorder = trainer.run_training_session()

    app = AppWindow(recorder)
    app.run()


if __name__ == "__main__":
    main()
