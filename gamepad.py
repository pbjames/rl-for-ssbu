from typing import Final

import vgamepad as vg  # pyright: ignore[reportMissingTypeStubs]
import time

# fmt: off
from vgamepad.win.vigem_commons import XUSB_BUTTON  # pyright: ignore[reportMissingTypeStubs]
# fmt: on

ONE_FRAME: Final[float] = 0.016

BINDINGS: dict[str, XUSB_BUTTON] = {
    "attack": XUSB_BUTTON.XUSB_GAMEPAD_A,
    "special": XUSB_BUTTON.XUSB_GAMEPAD_B,
    "jump": XUSB_BUTTON.XUSB_GAMEPAD_X,
    "grab": XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
    "shield": XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
}


class ControllerAgent:
    def __init__(self):
        self.gamepad: vg.VX360Gamepad = vg.VX360Gamepad()

    def press_and_release(self, button: XUSB_BUTTON):
        self.gamepad.press_button(button)
        self.gamepad.update()
        time.sleep(ONE_FRAME)
        self.gamepad.release_button(button)
        self.gamepad.update()

    def press_attack(self):
        self.press_and_release(BINDINGS["attack"])

    def press_special(self):
        self.press_and_release(BINDINGS["special"])

    def press_jump(self):
        self.press_and_release(BINDINGS["jump"])

    def press_grab(self):
        self.press_and_release(BINDINGS["grab"])

    def press_shield(self):
        self.press_and_release(BINDINGS["shield"])

    def press_lr(self):
        self.gamepad.press_button(XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        self.gamepad.press_button(XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        self.gamepad.update()
        time.sleep(ONE_FRAME)
        self.gamepad.release_button(XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        self.gamepad.release_button(XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        self.gamepad.update()

    def join_and_select_bowser(self):
        self.press_lr()
        self.gamepad.left_joystick(1, 1)
        self.gamepad.right_joystick(1, 1)
        self.gamepad.update()
        time.sleep(ONE_FRAME * 50)
        self.gamepad.left_joystick_float(-1, -1)
        self.gamepad.update()
        time.sleep(ONE_FRAME * 50)
        self.gamepad.left_joystick_float(-1, 0)
        self.gamepad.update()
        time.sleep(ONE_FRAME * 22)
        self.gamepad.left_joystick(0, 0)
        self.gamepad.right_joystick(0, 0)
        self.gamepad.update()
