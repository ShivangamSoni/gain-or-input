r"""
Robust image opening.

Several WBMS memes store their (long) caption as the filename, so the absolute
path can exceed Windows' legacy 260-char MAX_PATH limit -- a plain open() then
raises FileNotFoundError and the image would silently become a placeholder. We
prepend the \\?\ extended-length prefix on Windows for long paths.
"""
from __future__ import annotations

import os

from PIL import Image

from nesymis import config


def fspath(relpath: str) -> str:
    s = str(config.abspath(relpath))
    if os.name == "nt" and len(s) >= 255 and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def open_rgb(relpath: str) -> Image.Image:
    return Image.open(fspath(relpath)).convert("RGB")
