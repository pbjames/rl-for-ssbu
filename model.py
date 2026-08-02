from typing import Final
from gamepad import Command, ControllerAgent
from info_server import EventQueue, InfoEvent
from structs import Message


PUNISH: Final[int] = -3
REWARD_KO: Final[int] = 3
REWARD_WIN: Final[int] = 10


async def model_agent(queue: EventQueue):
    gamepad = ControllerAgent()
    while True:
        event, state = await queue.get()
        match event:
            case InfoEvent.STATE_CHANGE:  # execute
                commands = model_react(state)
                gamepad.execute_commands(commands)
            case InfoEvent.CPU_KO:  # punish
                ...
            case InfoEvent.OPP_KO:  # reward
                ...
            case InfoEvent.GAME_OVER:
                ...


def model_react(state: Message) -> list[tuple[Command, float, float]]:
    return []
