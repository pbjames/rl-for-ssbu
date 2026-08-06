from pathlib import Path

import numpy as np
import rich
import typer
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy  # pyright: ignore[reportUnknownVariableType] fmt: skip

from env import make_env

app = typer.Typer()
default_model = lambda e: RecurrentPPO(
    "MlpLstmPolicy", e, verbose=1
)

def safe_load_model(path: Path | str) -> RecurrentPPO:
    env = make_env()
    model = default_model(env)
    try:
        model = RecurrentPPO.load(path, env=env)
    except FileNotFoundError:
        rich.print("[green] Creating new model! 🤸")
    except BaseException as e:
        rich.print(f"[bold red] Model loading exception: {e}")
    finally:
        return model


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is not None:
        return
    model = safe_load_model("ppo_lstm")
    vec_env = model.get_env()
    if not vec_env:
        raise ValueError("Model with no vec env")
    obs, lstm_states, num_envs = vec_env.reset(), None, 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        try:
            action, lstm_states = model.predict(
                obs,  # pyright: ignore[reportArgumentType]
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, _rewards, _dones, _info = vec_env.step(action)
        except (KeyboardInterrupt, EOFError):
            break


@app.command()
def train(timesteps: float = 2e5):
    model = safe_load_model("ppo_lstm")
    model.learn(total_timesteps=int(timesteps), progress_bar=True)
    model.save("ppo_lstm")


@app.command()
def evaluate(episodes: int = 10):
    model = safe_load_model("ppo_lstm")
    env = model.get_env()
    if env is None:
        raise ValueError("Model with no VecEnv")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes)
    rich.print(f"[bold green]{mean_reward=} {std_reward=}")


if __name__ == "__main__":
    app()
