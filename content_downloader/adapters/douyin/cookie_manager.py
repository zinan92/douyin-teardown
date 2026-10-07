"""Cookie manager for Douyin adapter.

Migrated from douyin-downloader-1/auth/cookie_manager.py.
Simplified: loads from JSON file or dict only. No Playwright/browser dependency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _normalize_cookies(raw) -> Dict[str, str]:
    """Convert cookies to a flat {name: value} dict.

    Accepts two formats:
    - dict: {"key": "value", ...} — standard format
    - list: [{"name": "key", "value": "value", ...}, ...] — browser export format
    """
    if isinstance(raw, list):
        return {
            str(entry["name"]).strip(): str(entry["value"]).strip()
            for entry in raw
            if isinstance(entry, dict) and "name" in entry and "value" in entry
        }
    if isinstance(raw, dict):
        return {str(k).strip(): str(v).strip() for k, v in raw.items() if k}
    return {}


def _sanitize_cookies(raw) -> Dict[str, str]:
    """Keep the historical helper name used by the Douyin API client."""
    return _normalize_cookies(raw)


class CookieManager:
    """Manages Douyin cookies from a JSON file or injected dict.

    No browser automation — credentials are loaded from a pre-exported
    cookies.json file or provided programmatically.
    """

    def __init__(self, cookie_file: Optional[str] = None) -> None:
        self._cookie_file: Optional[Path] = Path(cookie_file) if cookie_file else None
        self._cookies: Dict[str, str] = {}

    def load_from_dict(self, cookies: Dict[str, str]) -> None:
        """Load cookies from a dict (replaces any previously loaded cookies)."""
        self._cookies = _normalize_cookies(cookies)

    def load_from_file(self, path: Optional[str] = None) -> bool:
        """Load cookies from a JSON file.

        Args:
            path: Path to the cookies JSON file. Falls back to the path
                  provided at construction time.

        Returns:
            True if cookies were loaded successfully, False otherwise.
        """
        target = Path(path) if path else self._cookie_file
        if not target or not target.exists():
            logger.debug("Cookie file not found: %s", target)
            return False

        try:
            with open(target, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._cookies = _normalize_cookies(raw)
            logger.debug("Loaded %d cookies from %s", len(self._cookies), target)
            return True
        except Exception as exc:
            logger.error("Failed to load cookies from %s: %s", target, exc)
            return False

    def save_to_file(self, path: Optional[str] = None) -> bool:
        """Persist current cookies to a JSON file."""
        target = Path(path) if path else self._cookie_file
        if not target:
            return False
        try:
            target.write_text(
                json.dumps(self._cookies, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except Exception as exc:
            logger.error("Failed to save cookies to %s: %s", target, exc)
            return False

    def get_cookies(self) -> Dict[str, str]:
        """Return current cookie dict."""
        return dict(self._cookies)

    def get_cookie_string(self) -> str:
        """Return cookies as a semicolon-separated header string."""
        return "; ".join(f"{k}={v}" for k, v in self._cookies.items())

    def validate_cookies(self) -> bool:
        """Return True if essential Douyin cookies are present."""
        required = {"ttwid", "odin_tt", "passport_csrf_token"}
        cookies = self._cookies
        missing = [k for k in required if k not in cookies or not cookies[k]]
        if missing:
            logger.warning("Cookie validation failed, missing: %s", ", ".join(missing))
            return False
        if not cookies.get("msToken"):
            logger.info("msToken not found — will be generated automatically")
        return True

    def clear(self) -> None:
        """Clear all in-memory cookies."""
        self._cookies = {}
