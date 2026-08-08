from pathlib import Path

import rich
import typer
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy  # pyright: ignore[reportUnknownVariableType] fmt: skip

from consts import PLUGIN_FILE_NAME, PLUGINS_BASE
from env import RewardComponentLoggingCallback, make_env
from model import default_model

app = typer.Typer()


def safe_load_model(path: Path | str, self_play: bool = False) -> RecurrentPPO:
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


@app.command()
def train(timesteps: float = 2e6, infinite: bool = False, self_play: bool = False):
    def learning_config(model: RecurrentPPO):
        model.learn(
            total_timesteps=int(timesteps),
            progress_bar=True,
            callback=RewardComponentLoggingCallback(),
        )

    model = safe_load_model("ppo_lstm", self_play)
    learning_config(model)
    model.save("ppo_lstm")
    while infinite:
        learning_config(model)
        model.save("ppo_lstm")


@app.command()
def evaluate(episodes: int = 10):
    model = safe_load_model("ppo_lstm")
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
        print("[bold green]Plugin ON")


# @app.callback(invoke_without_command=True)
# def main(ctx: typer.Context): ...


if __name__ == "__main__":
    app()
