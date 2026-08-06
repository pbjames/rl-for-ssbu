import time
from typing import Any, Final, final, override

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Space
from msgspec import Struct
from numpy.typing import NDArray
import rich
from vgamepad import XUSB_BUTTON

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
GAMEPAD_STICK_ARR: Final[NDArray[np.float64]] = np.linspace(
    *GAMEPAD_STICK_BOUNDS, num=256
)


@final
class SSBUEnv(Env[dict[str, Any], NDArray[np.integer]]):
    def __init__(self, fighter_index: int = 1):
        super(SSBUEnv, self).__init__()
        self.fighter_index = fighter_index
        self._controller = ControllerAgent()
        self._info_server = InfoServer()
        self._events = self._info_server.subscribe()
        self.action_space = MultiDiscrete(np.array([len(ACTIONS_LIST), 256, 256]))
        self.observation_space = Dict(into_dict_obs(into_dict(Message.default())))

    def _execute_action(self, action: NDArray[np.integer]):
        command = (
            ACTIONS_LIST[action[0]],
            int(GAMEPAD_STICK_ARR[action[1]]),
            int(GAMEPAD_STICK_ARR[action[2]]),
        )
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
        d =into_dict(Message.default(), int_enums=True)
        return d, {}

    @override
    def step(
        self, action: NDArray[np.integer]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._execute_action(action)
        try:
            self._info_server.step_game()
        except BaseException as e:
            with open("log", "+a") as fp:
                fp.writelines([f"{e}"])
                raise e
        reward = 0
        terminate = truncated = False
        info = {}
        *events, state = self._events.get()
        for event in events:
            match event:
                case InfoEvent.CPU_KO:
                    rich.print("I died")
                    reward -= REWARD_KO
                case InfoEvent.OPP_KO:
                    rich.print("Opponent died")
                    reward += REWARD_KO
                case InfoEvent.CPU_TOOK_DMG:
                    reward -= -0.05 * (state.cpu.damage / 20)
                    rich.print("I took damage")
                case InfoEvent.OPP_TOOK_DMG:
                    reward += 0.05 * (state.opp.damage / 20)
                    rich.print("Opponent took damage")
                case InfoEvent.GAME_OVER:
                    rich.print("Game over")
                    terminate = True
                case InfoEvent.STATE_CHANGE:
                    continue
        return into_dict(state, int_enums=True), reward, terminate, truncated, info


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
                d[k] = Discrete(len(SITUATION_LIST))
            elif k == "status":
                d[k] = Discrete(len(STATUS_LIST))
            else:
                e = f"Unknown key for string value '{k}' when decoding into Dict space"
                raise ValueError(e)
        else:
            e = f"Unknown value type '{type(v)}' when decoding into Dict space"
            raise ValueError(e)
    return d


def make_env():
    env = SSBUEnv()
    print("Welcome to controller setup")
    input("Press enter after setting up controller in Eden.")
    env._controller.gamepad.reset()
    env._controller.gamepad.update()
    input("Press enter after going to the smash character selection menu.")
    env._controller.press_lr()
    env._controller.gamepad.left_joystick(-30000, -30000)
    env._controller.gamepad.update()
    time.sleep(2.5)
    env._controller.gamepad.left_joystick(30000, 8333)
    env._controller.gamepad.update()
    time.sleep(1.2)
    env._controller.gamepad.left_joystick(0, 0)
    env._controller.gamepad.update()
    env._controller.gamepad.press_button(XUSB_BUTTON.XUSB_GAMEPAD_B)
    env._controller.gamepad.update()
    input("Press enter when the game halts.")
    return FlattenObservation(env)
