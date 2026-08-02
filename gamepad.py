from enum import Enum
from typing import Final

import vgamepad as vg  # pyright: ignore[reportMissingTypeStubs]
import time

from vgamepad.win.vigem_commons import XUSB_BUTTON  # pyright: ignore[reportMissingTypeStubs] fmt: skip

ONE_FRAME: Final[float] = 0.016

LEFT = (1, 0)
RIGHT = (-1, 0)
UP = (0, 1)
DOWN = (0, -1)


class Command(Enum):
    HOLD_NEXT = "HOLD_NEXT"
    STOP_HOLDING_NEXT = "RELEASE_NEXT"
    ATTACK = XUSB_BUTTON.XUSB_GAMEPAD_A
    SPECIAL = XUSB_BUTTON.XUSB_GAMEPAD_B
    JUMP = XUSB_BUTTON.XUSB_GAMEPAD_X
    GRAB = XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
    SHIELD = XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
    LSTICK = "LSTICK"
    RSTICK = "RSTICK"


class ControllerAgent:
    def __init__(self):
        self.gamepad: vg.VX360Gamepad = vg.VX360Gamepad()

    def execute_commands(self, commands: list[tuple[Command, float, float]]):
        hold_next = False
        release_next = False
        while commands:
            command, stick_x, stick_y = commands.pop(0)
            value = command.value
            match value:
                case "HOLD_NEXT":
                    hold_next = True
                    continue
                case "RELEASE_NEXT":
                    release_next = True
                    continue
                case XUSB_BUTTON() as button:
                    self.use_button(button, hold_next, release_next)
                case "RSTICK":
                    self.use_rstick(stick_x, stick_y, hold_next, release_next)
                case "LSTICK":
                    self.use_lstick(stick_x, stick_y, hold_next, release_next)
            hold_next = release_next = False

    def use_button(self, button: XUSB_BUTTON, hold: bool, release: bool):
        if not release:
            self.gamepad.press_button(button)
            self.gamepad.update()
            if hold:
                return
            time.sleep(ONE_FRAME)
        self.gamepad.release_button(button)
        self.gamepad.update()

    def use_rstick(self, x: float, y: float, hold: bool, release: bool):
        if not release:
            self.gamepad.right_joystick_float(x, y)
            self.gamepad.update()
            if hold:
                return
            time.sleep(ONE_FRAME)
        self.gamepad.right_joystick_float(0, 0)
        self.gamepad.update()

    def use_lstick(self, x: float, y: float, hold: bool, release: bool):
        if not release:
            self.gamepad.left_joystick_float(x, y)
            self.gamepad.update()
            if hold:
                return
            time.sleep(ONE_FRAME)
        self.gamepad.left_joystick_float(0, 0)
        self.gamepad.update()
