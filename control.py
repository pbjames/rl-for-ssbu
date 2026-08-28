from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import final
from statemachine import StateChart, State

from consts import GAME_PATH
from gamepad import ControllerAgent


def get_eden_qualified() -> Path:
    if (path := shutil.which("eden")) is None:
        e = f"Cannot find Eden executable"
        raise ValueError(e)
    else:
        return Path(path)


def start_eden():
    eden_path = get_eden_qualified()
    subprocess.run([eden_path, GAME_PATH])


@final
class EdenModel:
    def __init__(self, controller: ControllerAgent):
        self.controller = controller
        self.thread = threading.Thread(target=start_eden, daemon=True)

    def before_started(self):
        self.thread.start()
        time.sleep(15)
        self.controller.goto_main_menu()
        time.sleep(3)

    def before_rules_selection(self):
        self.controller.goto_ruleset()



@final
class EdenControl(StateChart[EdenModel]):
    unstarted = State(initial=True)
    started = State()
    main_menu = State()
    rules_selection = State()
    stage_selection = State()
    chara_selection = State()
    fight_with_cpu = State()
    fight_with_self = State()

    start = unstarted.to(started)
    game_over = fight_with_cpu.to(chara_selection) | fight_with_self.to(chara_selection)
