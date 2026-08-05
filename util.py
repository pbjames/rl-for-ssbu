from pathlib import Path
from typing import Final

BASE: Final[Path] = (
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
FILE_NAME: Final[str] = "libsmash_cpu_info.nro"


def main():
    original = BASE / FILE_NAME
    other = BASE.parent / FILE_NAME
    print(other, other.is_file(), other.exists())
    if original.is_file() and original.exists():
        original.rename(other)
        print("Moving plugin away")
    else:
        other.rename(original)
        print("Reinstate plugin")


if __name__ == "__main__":
    main()
