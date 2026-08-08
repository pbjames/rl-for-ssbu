from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

REMOTE_HOST: Final[tuple[str, int]] = "127.0.0.1", 65002
HOST: Final[tuple[str, int]] = "127.0.0.1", 7878
END_MAGIC: Final[bytes] = b"END"
PLUGINS_BASE: Final[Path] = (
    Path("~").expanduser()
    / ".local"
    / "share"
    / "eden"
    / "sdmc"
    / "atmosphere"
    / "contents"
    / "01006A800016E000"
    / "romfs"
    / "skyline"
    / "plugins"
)
PLUGIN_FILE_NAME: Final[str] = "libsmash_cpu_info.nro"
REWARD_LOSS: Final[int] = -30
REWARD_KO: Final[int] = 30
REWARD_WIN: Final[int] = 100
REWARD_HIT: Final[float] = 1
REWARD_DMG_SCALE: Final[float] = 0.08
GAMEPAD_STICK_BOUNDS: Final[tuple[int, int]] = -32768, 32767
GAMEPAD_STICK_RES: Final[int] = 3
GAMEPAD_STICK_ARR: Final[NDArray[np.float64]] = np.linspace(
    *GAMEPAD_STICK_BOUNDS, num=GAMEPAD_STICK_RES
)
