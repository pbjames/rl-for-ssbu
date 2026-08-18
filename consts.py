from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

LOG_PATH: Final[Path] = Path("logs")
REMOTE_HOST: Final[tuple[str, int]] = "127.0.0.1", 65002
HOST: Final[tuple[str, int]] = "127.0.0.1", 7878
GAME_OVER_STAGE: Final[int] = 310
END_MAGIC: Final[bytes] = b"END"
ACK_MAGIC: Final[bytes] = b"AC\n"
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
REWARD_KO: Final[int] = 30
REWARD_WIN: Final[int] = 100
REWARD_HIT: Final[float] = 1
REWARD_DMG_SCALE: Final[float] = 0.02
GAMEPAD_STICK_BOUNDS: Final[tuple[int, int]] = -32768, 32767
GAMEPAD_STICK_RES: Final[int] = 3
GAMEPAD_STICK_ARR: Final[NDArray[np.float64]] = np.linspace(
    *GAMEPAD_STICK_BOUNDS, num=GAMEPAD_STICK_RES
)
LOGGING_CONFIG: Final[dict[str, Any]] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        }
    },
    "handlers": {
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "simple",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "logs/cpu_smash.log",
            "maxBytes": 10**7,
            "backupCount": 3,
        },
    },
    "loggers": {"root": {"level": "DEBUG", "handlers": ["stderr", "file"]}},
}
