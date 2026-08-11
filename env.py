import logging
from copy import deepcopy
from functools import cache
from typing import Any, final, override

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
from structs import Message, Situation, Status, StructDict, into_dict
from typedefs import Command, EventInfo, InfoDict, default_info


logger = logging.getLogger(__name__)


@final
class SSBUEnv(Env[StructDict, NDArray[np.integer]]):
    def __init__(self):
        super(SSBUEnv, self).__init__()
        self.controller = ControllerAgent()
        self._info_server = InfoServer()
        self._events = self._info_server.subscribe()
        self.action_space = MultiDiscrete(
            [len(list(Command)), GAMEPAD_STICK_RES, GAMEPAD_STICK_RES]
        )
        self.observation_space = Dict(into_dict_obs(into_dict(Message.default())))

    def _execute_action(self, action: NDArray[np.integer]):
        command: Command = Command.by_index(int(action[0]))  # pyright: ignore[reportAny]
        self.controller.execute(
            command,
            int(GAMEPAD_STICK_ARR[action[1]]),
            int(GAMEPAD_STICK_ARR[action[2]]),
        )

    def _process_event_rewards(
        self, events: list[EventInfo], info: InfoDict
    ) -> tuple[float, bool]:
        reward = 0
        for event in events:
            match event:
                case EventInfo.CPU_KO:
                    reward -= REWARD_KO * 2
                    info["reward_components"]["death"] -= REWARD_KO * 2
                case EventInfo.OPP_KO:
                    reward += REWARD_KO * 2
                    info["reward_components"]["kill"] += REWARD_KO * 2
                case EventInfo.CPU_TOOK_DMG:
                    reward -= -REWARD_HIT
                    info["reward_components"]["damage_taken"] -= REWARD_HIT
                case EventInfo.OPP_TOOK_DMG:
                    reward += REWARD_HIT
                    info["reward_components"]["damage_dealt"] += REWARD_HIT
                case EventInfo.GAME_OVER:
                    return reward, True
                case EventInfo.STATE_CHANGE:
                    pass
        return reward, False

    def _process_state_rewards(
        self, old: Message, new: Message, info: InfoDict
    ) -> float:
        reward = 0.0
        old_dist_from_center = sum(x**2 for x in old.cpu.location)
        dist_from_center = sum(x**2 for x in new.cpu.location)
        if dist_from_center < old_dist_from_center:
            center_reward = 0.01
            if old.cpu.location[0] < 0:
                center_reward *= 5
            info["reward_components"]["center_control"] += center_reward
        opp_damage_taken = (
            new.opp.damage - old.opp.damage if new.opp.situation != "Outfield" else 0
        )
        cpu_damage_taken = (
            new.cpu.damage - old.cpu.damage if new.cpu.situation != "Outfield" else 0
        )
        reward += (cpu_damage_taken - opp_damage_taken) * REWARD_DMG_SCALE
        return reward

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, None] | None = None
    ) -> tuple[StructDict, InfoDict]:
        super().reset(seed=seed, options=options)
        d = into_dict(Message.default(), int_enums=True)
        return d, default_info()

    @override
    def step(
        self, action: NDArray[np.integer]
    ) -> tuple[StructDict, float, bool, bool, InfoDict]:
        old_state = self._info_server.state
        self._execute_action(action)
        self._info_server.step_game()
        terminate = truncated = False
        *events, state = self._events.get()
        info = default_info()
        reward, terminate = self._process_event_rewards(events, info)
        # reward *= -1 if self.us_p1 else 1
        reward += self._process_state_rewards(old_state, state, info)
        d = into_dict(state, int_enums=True)
        return d, reward, terminate, truncated, info


@final
class SSBUSelfPlay(
    Wrapper[StructDict, NDArray[np.integer], StructDict, NDArray[np.integer]]
):
    def __init__(
        self,
        env: Env[StructDict, NDArray[np.integer]],
        controller: ControllerAgent,
        name: str,
    ):
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

    def _reload(self):
        self._model = RecurrentPPO.load(self.name)
        self._last_seen_obs = {}
        self._episode_starts = np.ones((self._num_envs,), dtype=bool)
        self._lstm_states = None

    def _observe(self, d: StructDict):
        self._last_seen_obs = deepcopy(d)
        self._last_seen_obs["cpu"], self._last_seen_obs["opp"] = (
            self._last_seen_obs["opp"],
            self._last_seen_obs["cpu"],
        )

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, None] | None = None):
        d, info = self.env.reset()
        self._reload()
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
        self._observe(d)
        return d, reward, terminated, truncated, info


@final
class SkipStepWrapper(Wrapper[StructDict, NDArray[np.integer], StructDict, NDArray[np.integer]]):
    def __init__(self, env: Env[StructDict, NDArray[np.integer]], skip: int = 6):
        super().__init__(env)
        self.skip = skip

    @override
    def step(self, action: NDArray[np.integer]):
        total_reward = 0.0
        terminated = truncated = False
        info: InfoDict = default_info()
        obs = into_dict(Message.default(), int_enums=True)
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class RewardComponentLoggingCallback(BaseCallback):
    @staticmethod
    @cache
    def info_keys():
        return default_info()["reward_components"].keys()
    
    @override
    def _on_step(self) -> bool:
        infos: list[InfoDict] = self.locals["infos"]  # pyright: ignore[reportAny]
        for key in self.info_keys():
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
