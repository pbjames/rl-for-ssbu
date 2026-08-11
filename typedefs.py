from __future__ import annotations

from enum import Enum
from functools import cache
from typing import Self, TypedDict

from vgamepad import XUSB_BUTTON  # pyright: ignore[reportMissingTypeStubs]
import queue

from structs import Message


class EventInfo(Enum):
    CPU_KO = "cpu_ko"
    OPP_KO = "opp_ko"
    CPU_TOOK_DMG = "cpu_take_damage"
    OPP_TOOK_DMG = "opp_take_damage"
    GAME_OVER = "game_over"
    STATE_CHANGE = "state_change"


def default_info() -> InfoDict:
    return {
        "reward_components": {
            "center_control": 0.0,
            "death": 0.0,
            "kill": 0.0,
            "damage_taken": 0.0,
            "damage_dealt": 0.0,
            "recovery": 0.0,
        }
    }


EventQueue = queue.Queue[tuple[*tuple[EventInfo, ...], Message]]
InfoDict = dict[str, dict[str, float]]


class Command(Enum):
    HOLD_NEXT = "HOLD_NEXT"
    STOP_HOLDING_NEXT = "RELEASE_NEXT"
    ATTACK = XUSB_BUTTON.XUSB_GAMEPAD_B
    SPECIAL = XUSB_BUTTON.XUSB_GAMEPAD_A
    JUMP = XUSB_BUTTON.XUSB_GAMEPAD_Y
    GRAB = XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
    SHIELD = XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
    LSTICK = "LSTICK"
    RSTICK = "RSTICK"

    @classmethod
    @cache
    def by_index(cls, idx: int) -> Self:
        return list(cls)[idx]
