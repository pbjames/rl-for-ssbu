from typing import Any, Final, final, override

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Dict, MultiDiscrete
from numpy.typing import NDArray

from gamepad import Command, ControllerAgent
from info_server import InfoEvent, InfoServer
from structs import Message, Situation, Status, into_dict

ACTIONS_LIST: Final[list[Command]] = list(Command)
SITUATION_LIST: Final[list[Situation]] = list(Situation)
STATUS_LIST: Final[list[Status]] = list(Status)
GAMEPAD_STICK_BOUNDS: Final[tuple[int, int]] = -32768, 32767
REWARD_LOSS: Final[int] = -3
REWARD_KO: Final[int] = 3
REWARD_WIN: Final[int] = 10
DEFAULT_MESSAGE: Final[Message] = Message.default()


@final
class SSBUEnv(Env[dict[str, Any], NDArray[np.integer]]):
    def __init__(self, fighter_index: int = 1):
        super(SSBUEnv, self).__init__()
        self.fighter_index = fighter_index
        self._controller = ControllerAgent()
        self._info_server = InfoServer()
        self._events = self._info_server.subscribe()
        self.action_space = MultiDiscrete(
            np.array(
                [
                    len(ACTIONS_LIST),
                    np.linspace(*GAMEPAD_STICK_BOUNDS, num=256),
                    np.linspace(*GAMEPAD_STICK_BOUNDS, num=256),
                ]
            )
        )
        self.observation_space = Dict(into_dict(Message.default()))

    def _execute_action(self, action: NDArray[np.integer]):
        command = ACTIONS_LIST[action[0]], action[1], action[2]  # pyright: ignore[reportUnknownVariableType]
        self._controller.execute_commands([command])

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, None] | None = None
    ) -> tuple[dict[str, Any], dict[str, None]]:
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)
        return into_dict(Message.default()), {}

    @override
    def step(
        self, action: NDArray[np.integer]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._execute_action(action)
        self._info_server.step_game()
        reward = 0
        terminate = truncated = False
        info = {}
        *events, state = self._events.get()
        for event in events:
            match event:
                case InfoEvent.CPU_KO:
                    reward -= REWARD_KO
                case InfoEvent.OPP_KO:
                    reward += REWARD_KO
                case InfoEvent.CPU_TOOK_DMG:
                    reward -= -0.05 * (state.cpu.damage / 20)
                case InfoEvent.OPP_TOOK_DMG:
                    reward += 0.05 * (state.opp.damage / 20)
                case InfoEvent.GAME_OVER:
                    terminate = True
                case InfoEvent.STATE_CHANGE:
                    continue
        return into_dict(state), reward, terminate, truncated, info


gym.register("SSBUEnv-v0", entry_point="model:SSBUEnv", max_episode_steps=500000)
