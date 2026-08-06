from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Self, final, override

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Space, flatten
from gymnasium.wrappers import FlattenObservation
from numpy.typing import NDArray
from sb3_contrib import RecurrentPPO

from consts import GAMEPAD_STICK_ARR, REWARD_DMG_SCALE, REWARD_HIT, REWARD_KO
from gamepad import Command, ControllerAgent
from info_server import InfoEvent, InfoServer
from structs import Message, Situation, Status, into_dict


@final
class SSBUSelfPlayModule:
    def __init__(self):
        self.controller = ControllerAgent()
        self._model = RecurrentPPO.load("ppo_lstm")
        self._lstm_states = None
        self.observation_space = Dict(into_dict_obs(into_dict(Message.default())))
        self._last_seen_obs = {}
        self._num_envs = 1
        self._episode_starts = np.ones((self._num_envs,), dtype=bool)

    def _execute_action(self, action: NDArray[np.integer]):
        command: Command = Command.by_index(action[0])  # pyright: ignore[reportAny]
        self.controller.execute(
            command,
            int(GAMEPAD_STICK_ARR[action[1]]),
            int(GAMEPAD_STICK_ARR[action[2]]),
        )

    def reload(self):
        self._model = RecurrentPPO.load("ppo_lstm")
        self._last_seen_obs = {}
        self._episode_starts = np.ones((self._num_envs,), dtype=bool)
        self._lstm_states = None

    def observe(self, d: dict[str, Any]):
        self._last_seen_obs = deepcopy(d)
        self._last_seen_obs["cpu"], self._last_seen_obs["opp"] = (
            self._last_seen_obs["opp"],
            self._last_seen_obs["cpu"],
        )

    def step(self):
        if not self._last_seen_obs:
            return
        action, self._lstm_states = self._model.predict(
            flatten(self.observation_space, self._last_seen_obs),
            state=self._lstm_states,
            episode_start=self._episode_starts,
            deterministic=False,
        )
        self._execute_action(action)


@final
class SSBUEnv(Env[dict[str, Any], NDArray[np.integer]]):
    def __init__(
        self, fighter_index: int = 1, self_play: SSBUSelfPlayModule | None = None
    ):
        super(SSBUEnv, self).__init__()
        self.fighter_index = fighter_index
        self.controller = ControllerAgent()
        self._info_server = InfoServer()
        self._events = self._info_server.subscribe()
        self.action_space = MultiDiscrete(np.array([len(list(Command)), 256, 256]))
        self.observation_space = Dict(into_dict_obs(into_dict(Message.default())))
        self.self_play = self_play

    def _execute_action(self, action: NDArray[np.integer]):
        command: Command = Command.by_index(action[0])  # pyright: ignore[reportAny]
        self.controller.execute(
            command,
            int(GAMEPAD_STICK_ARR[action[1]]),
            int(GAMEPAD_STICK_ARR[action[2]]),
        )

    def _harness(self, f: Callable[[], None]):
        try:
            f()
        except BaseException as e:
            with open("log", "+a") as fp:
                fp.writelines([f"{e}"])
                raise e

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, None] | None = None
    ) -> tuple[dict[str, Any], dict[str, None]]:
        super().reset(seed=seed)
        d = into_dict(Message.default(), int_enums=True)
        if self.self_play is not None:
            self.self_play.reload()
        return d, {}

    @override
    def step(
        self, action: NDArray[np.integer]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._execute_action(action)
        if self.self_play:
            self.self_play.step()
        self._harness(self._info_server.step_game)
        reward = 0
        terminate = truncated = False
        info = {
            "opponent": {"wins": 0},
            "learner": {"wins": 0},
        }
        *events, state = self._events.get()
        for event in events:
            match event:
                case InfoEvent.CPU_KO:
                    info["opponent"]["wins"] += 1
                    reward -= REWARD_KO * 2
                    terminate = True
                case InfoEvent.OPP_KO:
                    info["learner"]["wins"] += 1
                    reward += REWARD_KO
                    terminate = True
                case InfoEvent.CPU_TOOK_DMG:
                    reward -= -REWARD_HIT * (state.cpu.damage * REWARD_DMG_SCALE)
                case InfoEvent.OPP_TOOK_DMG:
                    reward += REWARD_HIT * (state.opp.damage * REWARD_DMG_SCALE)
                case InfoEvent.GAME_OVER:
                    terminate = True
                case InfoEvent.STATE_CHANGE:
                    continue
        if reward <= 0:
            reward -= 0.01
        d = into_dict(state, int_enums=True)
        if self.self_play is not None:
            self.self_play.observe(d)
        return d, reward, terminate, truncated, info


def into_dict_obs(d: dict[str, Any]) -> dict[str, Space[Any]]:
    for k, v in d.items():
        if isinstance(v, int):
            d[k] = Box(0, 1000, dtype=np.int64)
        elif isinstance(v, dict):
            d[k] = Dict(into_dict_obs(v))
        elif isinstance(v, tuple):
            d[k] = Box(-1000, 1000, shape=(3,), dtype=np.float64)
        elif isinstance(v, float):
            d[k] = Box(-1000, 1000, shape=(1,), dtype=np.float64)
        elif isinstance(v, bool):
            d[k] = Discrete(2)
        elif isinstance(v, str):
            if k == "situation":
                d[k] = Discrete(len(Situation.values()))
            elif k == "status":
                d[k] = Discrete(len(Status.values()))
            else:
                e = f"Unknown key for string value '{k}' when decoding into Dict space"
                raise ValueError(e)
        else:
            e = f"Unknown value type '{type(v)}' when decoding into Dict space"
            raise ValueError(e)
    return d


def make_env(self_play: bool = False):
    module = SSBUSelfPlayModule() if self_play else None
    env = SSBUEnv(self_play=module)
    input("Press enter after going to the smash character selection menu.")
    if self_play and module is not None:
        module.controller.marth_selection_sequence()
    env.controller.marth_selection_sequence()
    return FlattenObservation(env)
