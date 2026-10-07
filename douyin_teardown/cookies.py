"""Read the user's own exported Douyin cookies without ever printing them."""
from __future__ import annotations

import json
from pathlib import Path
import stat


class CookieFileError(RuntimeError):
    """The cookie file is missing, unsafe, or invalid."""


def load_cookie_file(path: Path) -> dict[str, str]:
    """Load a browser-export cookie JSON file without logging its values."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise CookieFileError(f"cookie file does not exist: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise CookieFileError(
            f"cookie file must be private (mode 600 or stricter): {path}"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CookieFileError(f"cookie file is not valid JSON: {path}") from exc

    if isinstance(raw, list):
        cookies = {
            str(item["name"]).strip(): str(item["value"]).strip()
            for item in raw
            if isinstance(item, dict) and item.get("name") and item.get("value")
        }
    elif isinstance(raw, dict):
        cookies = {
            str(name).strip(): str(value).strip()
            for name, value in raw.items()
            if name and value
        }
    else:
        cookies = {}

    if not cookies:
        raise CookieFileError(f"cookie file contains no usable cookies: {path}")
    return cookies
