import logging
import logging.config
from pathlib import Path

import rich
import typer
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from consts import LOGGING_CONFIG, PLUGIN_FILE_NAME, PLUGINS_BASE
from env import RewardComponentLoggingCallback
from info_server import InfoServer
from util import safe_load_model

app = typer.Typer()
Path("logs").mkdir(exist_ok=True)
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


@app.command()
def debug():
    logger.debug("start debugging")
    info = InfoServer()
    queue = info.subscribe()
    while True:
        try:
            info.step_game()
            *_, state = queue.get()
            print((state))
        except (KeyboardInterrupt, EOFError):
            rich.print("[yellow] Exit? (Y/n)")
            if input().startswith("y"):
                return


@app.command()
def train(
    name: str,
    timesteps: float = 24000,
    infinite: bool = False,
    self_play: bool = False,
    experiment: bool = False,
):
    def learning_config(model: RecurrentPPO):
        model.learn(
            total_timesteps=int(timesteps),
            progress_bar=True,
            callback=RewardComponentLoggingCallback(),
            log_interval=1,
        )
    path = Path(name)
    model = safe_load_model(path, self_play, experiment=experiment)
    learning_config(model)
    model.save(name)
    while infinite:
        learning_config(model)
        model.save(name)


@app.command()
def evaluate(name: str, episodes: int = 10):
    model = safe_load_model(Path(name))
    env = model.get_env()
    if env is None:
        raise ValueError("Model with no VecEnv")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes)
    rich.print(f"[bold green]{mean_reward=} {std_reward=}")


@app.command()
def toggle():
    original = PLUGINS_BASE / PLUGIN_FILE_NAME
    other = PLUGINS_BASE.parent / PLUGIN_FILE_NAME
    if original.is_file() and original.exists():
        original.rename(other)
        rich.print("[bold yellow]Plugin OFF")
    else:
        other.rename(original)
        rich.print("[bold green]Plugin ON")


if __name__ == "__main__":
    app()
