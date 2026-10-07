"""URL router — classifies URLs and routes them to the correct adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from content_downloader.adapters.base import PlatformAdapter


class UnsupportedPlatformError(ValueError):
    """Raised when no adapter can handle the given URL."""

    def __init__(self, url: str) -> None:
        self.url = url
        supported = ", ".join(sorted(_SUPPORTED_PLATFORMS))
        super().__init__(
            f"No adapter found for URL: {url!r}\n"
            f"Supported platforms: {supported}\n"
            f"Supported URL patterns:\n" + _format_supported_patterns()
        )


# URL pattern registry
# Each entry: (platform, url_type, regex_pattern)
_URL_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    # Douyin
    ("douyin", "single", re.compile(r"https?://(?:www\.)?douyin\.com/video/\S+")),
    ("douyin", "single", re.compile(r"https?://v\.douyin\.com/\S+")),
    # /user/self?modal_id=xxx → treat as single video (modal_id has the aweme_id)
    ("douyin", "single", re.compile(r"https?://(?:www\.)?douyin\.com/user/self\?.*modal_id=\d+")),
    ("douyin", "profile", re.compile(r"https?://(?:www\.)?douyin\.com/user/\S+")),
    # Xiaohongshu (XHS)
    ("xhs", "single", re.compile(r"https?://(?:www\.)?xiaohongshu\.com/explore/\S+")),
    ("xhs", "single", re.compile(r"https?://(?:www\.)?xiaohongshu\.com/discovery/\S+")),
    ("xhs", "single", re.compile(r"https?://xhslink\.com/\S+")),
    ("xhs", "profile", re.compile(r"https?://(?:www\.)?xiaohongshu\.com/user/profile/\S+")),
    # WeChat Official Account
    ("wechat_oa", "single", re.compile(r"https?://mp\.weixin\.qq\.com/s/\S+")),
    # X (Twitter)
    ("x", "single", re.compile(r"https?://(?:www\.)?x\.com/[^/]+/status/\d+")),
    ("x", "single", re.compile(r"https?://(?:www\.)?twitter\.com/[^/]+/status/\d+")),
    # Fixture (testing)
    ("fixture", "single", re.compile(r"https?://fixture\.test/(?:video|image)/\S+")),
    ("fixture", "profile", re.compile(r"https?://fixture\.test/user/\S+")),
]

_SUPPORTED_PLATFORMS = {"douyin", "xhs", "wechat_oa", "x", "fixture"}


def _format_supported_patterns() -> str:
    lines = [
        "  douyin   : douyin.com/video/*, v.douyin.com/*, douyin.com/user/*",
        "  xhs      : xiaohongshu.com/explore/*, xiaohongshu.com/discovery/*, xhslink.com/*, xiaohongshu.com/user/profile/*",
        "  wechat_oa: mp.weixin.qq.com/s/*",
        "  x        : x.com/*/status/*, twitter.com/*/status/*",
        "  fixture  : fixture.test/video/*, fixture.test/image/*, fixture.test/user/*",
    ]
    return "\n".join(lines)


def classify_url(url: str) -> tuple[str, str]:
    """Classify a URL into (platform, url_type).

    Args:
        url: The URL to classify.

    Returns:
        A tuple ``(platform, url_type)`` where ``url_type`` is ``'single'`` or
        ``'profile'``.

    Raises:
        UnsupportedPlatformError: If no pattern matches the URL.
    """
    for platform, url_type, pattern in _URL_PATTERNS:
        if pattern.match(url):
            return platform, url_type
    raise UnsupportedPlatformError(url)


def get_adapter(
    url: str,
    cookies_path: Optional[Path] = None,
) -> "PlatformAdapter":
    """Return the appropriate adapter instance for the given URL.

    Imports adapters lazily to avoid circular dependencies and allow
    real adapters to be optional until installed.

    Args:
        url: The URL to route.
        cookies_path: Optional path to a cookies JSON file (used by platforms
                      that require authentication, e.g. Douyin).

    Returns:
        An instantiated adapter that ``can_handle(url)`` returns True.

    Raises:
        UnsupportedPlatformError: If no adapter exists for the URL's platform.
    """
    platform, _url_type = classify_url(url)  # raises if unsupported

    if platform == "fixture":
        from content_downloader.adapters.fixture import FixtureAdapter
        return FixtureAdapter()

    if platform == "douyin":
        from content_downloader.adapters.douyin.adapter import DouyinAdapter
        from content_downloader.adapters.douyin.cookie_manager import _normalize_cookies

        # Auto-discover cookies if not explicitly provided
        resolved_path = cookies_path
        if not resolved_path or not resolved_path.exists():
            for candidate in [
                Path.home() / ".douyin-cookies.json",
                Path("cookies.json"),
            ]:
                if candidate.exists():
                    resolved_path = candidate
                    break

        cookies: dict = {}
        if resolved_path and resolved_path.exists():
            try:
                raw = json.loads(resolved_path.read_text(encoding="utf-8"))
                cookies = _normalize_cookies(raw)
            except Exception:
                pass
        return DouyinAdapter(cookies=cookies)

    if platform == "xhs":
        from content_downloader.adapters.xhs.adapter import XHSAdapter
        return XHSAdapter()

    if platform == "x":
        from content_downloader.adapters.x.adapter import XAdapter
        return XAdapter()

    if platform == "wechat_oa":
        from content_downloader.adapters.wechat_oa.adapter import WeChatOAAdapter
        return WeChatOAAdapter()

    # Real platform adapters are stubs — they raise NotImplementedError when called.
    # They are registered here so `platforms` command can list them.
    from content_downloader.adapters.stub import StubAdapter
    return StubAdapter(platform=platform)


def list_supported_platforms() -> list[str]:
    """Return a sorted list of all supported platform names."""
    return sorted(_SUPPORTED_PLATFORMS)
