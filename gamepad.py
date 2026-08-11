import time
from typing import final

import vgamepad as vg
from vgamepad.win.vigem_commons import XUSB_BUTTON

from typedefs import Command


@final
class ControllerAgent:
    def __init__(self):
        self.gamepad: vg.VX360Gamepad = vg.VX360Gamepad()
        self._tick: XUSB_BUTTON | Command | None = None
        self._hold_next = self._release_next = False

    def execute(self, command: Command, stick_x: int, stick_y: int):
        match command.value:
            case "HOLD_NEXT":
                self._hold_next = True
                return
            case "RELEASE_NEXT":
                self._release_next = True
                return
            case XUSB_BUTTON() as button:
                self.use_button(button, self._hold_next, self._release_next)
            case "RSTICK":
                self.use_rstick(stick_x, stick_y, self._hold_next, self._release_next)
            case "LSTICK":
                self.use_lstick(stick_x, stick_y, self._hold_next, self._release_next)
        self._release_next = self._hold_next = False

    def use_button(self, button: XUSB_BUTTON, hold: bool, release: bool):
        self._release_ticked()
        if not release:
            self.gamepad.press_button(button)
            self.gamepad.update()
            if not hold:
                self._tick = button
            return
        self.gamepad.release_button(button)
        self.gamepad.update()

    def use_rstick(self, x: int, y: int, hold: bool, release: bool):
        self._release_ticked()
        if not release:
            self.gamepad.right_joystick(x, y)
            self.gamepad.update()
            if not hold:
                self._tick = Command.RSTICK
            return
        self.gamepad.right_joystick(0, 0)
        self.gamepad.update()

    def use_lstick(self, x: int, y: int, hold: bool, release: bool):
        self._release_ticked()
        if not release:
            self.gamepad.left_joystick(x, y)
            self.gamepad.update()
            if not hold:
                self._tick = Command.LSTICK
            return
        self.gamepad.left_joystick(0, 0)
        self.gamepad.update()

    def _release_ticked(self):
        match self._tick:
            case XUSB_BUTTON() as button:
                self.gamepad.release_button(button)
            case Command.LSTICK:
                self.gamepad.left_joystick(0, 0)
            case Command.RSTICK:
                self.gamepad.right_joystick(0, 0)
            case _:
                return
        self._tick = None
        self.gamepad.update()

    def press_lr(self):
        self.gamepad.press_button(XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        self.gamepad.press_button(XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        self.gamepad.update()
        self.gamepad.release_button(XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        self.gamepad.release_button(XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        self.gamepad.update()

    def marth_selection_sequence(self):
        self.gamepad.reset()
        self.gamepad.update()
        self.press_lr()
        self.use_lstick(-32768, -32768, hold=False, release=False)
        time.sleep(3)
        self.use_lstick(30000, 8900, hold=False, release=False)
        time.sleep(1.2)
        self.use_lstick(0, 0, hold=False, release=False)
        self.use_button(XUSB_BUTTON.XUSB_GAMEPAD_B, hold=False, release=False)

    def simulate_classroom_with_cpu(self):
        self.gamepad.reset()
        self.gamepad.update()
        self.use_button(XUSB_BUTTON.XUSB_GAMEPAD_A, hold=False, release=False)
        time.sleep(1)
        self.use_button(XUSB_BUTTON.XUSB_GAMEPAD_A, hold=False, release=False)
        time.sleep(1)
        self.use_button(XUSB_BUTTON.XUSB_GAMEPAD_A, hold=False, release=False)
        self.use_button(XUSB_BUTTON.XUSB_GAMEPAD_A, hold=False, release=False)
        time.sleep(3)
        self.use_button(XUSB_BUTTON.XUSB_GAMEPAD_A, hold=False, release=False)
        self.use_rstick(32767, 0, hold=False, release=False)
        time.sleep(0.63)
        self.use_rstick(0, 0, hold=False, release=False)
        self.use_button(XUSB_BUTTON.XUSB_GAMEPAD_X, hold=False, release=False)
        self.use_button(XUSB_BUTTON.XUSB_GAMEPAD_A, hold=False, release=False)
