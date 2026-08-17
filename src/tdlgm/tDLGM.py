# ruff: noqa: N999

from .main import main
from .model import TDLGM, TDLGMConfig

__all__ = ["TDLGM", "TDLGMConfig"]


if __name__ == "__main__":
    main()
