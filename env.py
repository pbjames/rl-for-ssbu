import logging
from copy import deepcopy
from typing import Any, TypedDict, final, override

import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Space, flatten
from numpy.typing import NDArray
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

from consts import (GAMEPAD_STICK_ARR, GAMEPAD_STICK_RES, REWARD_DMG_SCALE,
                    REWARD_HIT, REWARD_KO)
from gamepad import ControllerAgent
from info_server import InfoServer
from typedefs import EventInfo
from structs import Message, Situation, Status, into_dict
from typedefs import Command, Info, InfoRewardComponents

logger = logging.getLogger(__name__)


@final
class SSBUEnv(Env[dict[str, Any], NDArray[np.integer]]):
    def __init__(self):
        super(SSBUEnv, self).__init__()
        self.controller = ControllerAgent()
        self._info_server = InfoServer()
        self._events = self._info_server.subscribe()
        self.action_space = MultiDiscrete(
            np.array([len(list(Command)), GAMEPAD_STICK_RES, GAMEPAD_STICK_RES])
        )
        self.observation_space = Dict(into_dict_obs(into_dict(Message.default())))
        self._info: Info = {
            "reward_components": {
                "continuous": 0.0,
                "death": 0.0,
                "kill": 0.0,
                "damage_taken": 0.0,
                "damage_dealt": 0.0,
            }
        }

    def _execute_action(self, action: NDArray[np.integer]):
        command: Command = Command.by_index(action[0])  # pyright: ignore[reportAny]
        self.controller.execute(
            command,
            int(GAMEPAD_STICK_ARR[action[1]]),
            int(GAMEPAD_STICK_ARR[action[2]]),
        )

    def process_event_rewards(
        self, events: list[EventInfo], state: Message
    ) -> tuple[float, bool]:
        reward = 0
        for event in events:
            match event:
                case EventInfo.CPU_KO:
                    reward -= REWARD_KO * 2
                    self._info["reward_components"]["death"] -= REWARD_KO * 2
                case EventInfo.OPP_KO:
                    reward += REWARD_KO
                    self._info["reward_components"]["kill"] += REWARD_KO * 2
                case EventInfo.CPU_TOOK_DMG:
                    reward -= -REWARD_HIT
                    self._info["reward_components"]["damage_taken"] -= REWARD_HIT
                case EventInfo.OPP_TOOK_DMG:
                    reward += REWARD_HIT
                    self._info["reward_components"]["damage_dealt"] += REWARD_HIT
                case EventInfo.GAME_OVER:
                    return reward, True
                case EventInfo.STATE_CHANGE:
                    pass
        return reward, False

    def process_state_rewards(self, old: Message, new: Message) -> float:
        reward = 0.0
        old_dist_from_center = sum(x**2 for x in old.cpu.location)
        dist_from_center = sum(x**2 for x in new.cpu.location)
        if dist_from_center < old_dist_from_center:
            reward += 0.001
        opp_damage_taken = new.opp.damage - old.opp.damage
        cpu_damage_taken = new.cpu.damage - old.cpu.damage
        reward += (cpu_damage_taken - opp_damage_taken) * REWARD_DMG_SCALE
        return reward

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, None] | None = None
    ) -> tuple[dict[str, Any], Info]:
        super().reset(seed=seed)
        d = into_dict(Message.default(), int_enums=True)
        for k in self._info["reward_components"]:
            self._info["reward_components"][k] = 0.0
        return d, self._info

    @override
    def step(
        self, action: NDArray[np.integer]
    ) -> tuple[dict[str, Any], float, bool, bool, Info]:
        old_state = self._info_server.state
        self._execute_action(action)
        self._info_server.step_game()
        terminate = truncated = False
        *events, state = self._events.get()
        reward, terminate = self.process_event_rewards(events, state)
        reward += self.process_state_rewards(old_state, state)
        d = into_dict(state, int_enums=True)
        return d, reward, terminate, truncated, self._info


@final
class SSBUSelfPlay(Wrapper):
    def __init__[T, U](self, env: Env[T, U], controller: ControllerAgent, name: str):
        super().__init__(env)
        self.controller = controller
        self.name = name
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
        self._model = RecurrentPPO.load(self.name)
        self._last_seen_obs = {}
        self._episode_starts = np.ones((self._num_envs,), dtype=bool)
        self._lstm_states = None

    def observe(self, d: dict[str, Any]):
        self._last_seen_obs = deepcopy(d)
        self._last_seen_obs["cpu"], self._last_seen_obs["opp"] = (
            self._last_seen_obs["opp"],
            self._last_seen_obs["cpu"],
        )

    @override
    def reset(self, *, seed=None, options=None):
        d, info = self.env.reset()
        self.reload()
        return d, info

    @override
    def step(self, action: NDArray[np.integer]):
        if not self._last_seen_obs:
            return self.env.step(action)
        action, self._lstm_states = self._model.predict(
            flatten(self.observation_space, self._last_seen_obs),  # pyright: ignore[reportArgumentType]
            state=self._lstm_states,
            episode_start=self._episode_starts,
            deterministic=False,
        )
        self._execute_action(action)
        d, reward, terminated, truncated, info = self.env.step(action)
        self.observe(d)
        return d, reward, terminated, truncated, info


@final
class SkipStepWrapper(Wrapper):
    def __init__[T, U](self, env: Env[T, U], skip: int = 5):
        super().__init__(env)
        self.skip = skip

    @override
    def step(self, action: NDArray[np.integer]):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        obs = into_dict(Message.default(), int_enums=True)
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward  # pyright: ignore[reportOperatorIssue]
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class RewardComponentLoggingCallback(BaseCallback):
    @override
    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals["infos"]

        for key in InfoRewardComponents.__required_keys__:
            values = [info["reward_components"][key] for info in infos]
            if values:
                self.logger.record(f"reward/{key}", np.sum(values))

        return True


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
