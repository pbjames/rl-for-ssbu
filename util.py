from pathlib import Path
import rich
from gymnasium.wrappers import FlattenObservation, TimeLimit
from sb3_contrib import RecurrentPPO

from env import SkipStepWrapper, SSBUEnv, SSBUSelfPlay
from gamepad import ControllerAgent
from model import default_model


def make_env(self_play: str = ""):
    env = SSBUEnv()
    input("Press enter after going to the smash character selection menu.")
    if self_play:
        sp_controller = ControllerAgent()
        sp_controller.marth_selection_sequence()
        env.controller.marth_selection_sequence()
        input("Press enter after starting the game")
        return TimeLimit(
            SSBUSelfPlay(
                SkipStepWrapper(
                    FlattenObservation(env),
                ),
                sp_controller,
                self_play,
            ),
            max_episode_steps=24000,
        )
    else:
        env.controller.marth_selection_sequence()
        input("Press enter after starting the game")
        return TimeLimit(
            SkipStepWrapper(
                FlattenObservation(env),
            ),
            max_episode_steps=24000,
        )


def safe_load_model(path: Path | str, self_play: str = "") -> RecurrentPPO:
    env = make_env(self_play)
    model = default_model(env)
    try:
        model = RecurrentPPO.load(path, env=env)
    except FileNotFoundError:
        rich.print("[green] Creating new model! 🤸")
    except BaseException as e:
        rich.print(f"[bold red] Model loading exception: {e}")
    finally:
        return model
