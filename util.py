import logging
from pathlib import Path
import rich
from gymnasium.wrappers import FlattenObservation, TimeLimit
from sb3_contrib import RecurrentPPO

from env import SkipStepWrapper, SSBUEnv, SSBUSelfPlay
from gamepad import ControllerAgent
from model import default_model


logger = logging.getLogger(__name__)


def make_env(path: Path, self_play: bool, experiment: bool):
    env = SSBUEnv(us_p1 = experiment)
    if experiment:
        env.controller.simulate_classroom_with_cpu()
    else:
        input("Press enter after going to the smash character selection menu.")
    if self_play:
        logger.info("Starting self-play")
        sp_controller = ControllerAgent()
        sp_controller.marth_selection_sequence()
        env.controller.marth_selection_sequence()
        input("Press enter after starting the game")
        return TimeLimit(
            SSBUSelfPlay(
                SkipStepWrapper(
                    FlattenObservation(env),
                ),
                path,
                sp_controller,
            ),
            max_episode_steps=24000,
        )
    else:
        if not experiment:
            env.controller.marth_selection_sequence()
            input("Press enter after starting the game")
        return TimeLimit(
            SkipStepWrapper(
                FlattenObservation(env),
            ),
            max_episode_steps=24000,
        )


def safe_load_model(
    path: Path, self_play: bool = False, experiment: bool = False
) -> RecurrentPPO:
    env = make_env(path, self_play, experiment)
    model = default_model(env)
    try:
        model = RecurrentPPO.load(path, env=env)
    except FileNotFoundError:
        rich.print("[green] Creating new model! 🤸")
    except BaseException as e:
        rich.print(f"[bold red] Model loading exception: {e}")
    finally:
        return model
