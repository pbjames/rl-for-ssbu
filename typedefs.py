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


EventQueue = queue.Queue[tuple[*tuple[EventInfo, ...], Message]]


class Info(TypedDict):
    reward_components: InfoRewardComponents


class InfoRewardComponents(TypedDict):
    center_control: float
    death: float
    kill: float
    damage_taken: float
    damage_dealt: float


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
