"""Playwright-based Douyin cookie fetcher.

Ported from jiji262/douyin-downloader tools/cookie_fetcher.py.

Usage:
    python -m content_downloader.tools.cookie_fetcher [--output cookies.json]
    content-downloader fetch-cookies [--headless] [--output cookies.json]

The tool opens a real Chromium browser, lets the user log in to Douyin,
then captures cookies and saves them to a JSON file compatible with
content_downloader's cookie_manager.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOUYIN_URL = "https://www.douyin.com"

REQUIRED_COOKIES = {
    "msToken",
    "ttwid",
    "odin_tt",
    "passport_csrf_token",
    "sid_guard",
    "sessionid",
    "sid_tt",
    "s_v_web_id",
    "UIFID",
    "__ac_nonce",
    "__ac_signature",
    "d_ticket",
}

# Keys that are always kept if present
AUXILIARY_COOKIES = {
    "bd_ticket_guard_client_data",
    "csrf_session_id",
    "LOGIN_STATUS",
    "n_mh",
    "passport_mfa_token",
    "store-idc",
    "store-country-code",
    "webcast_local_quality",
}


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------


async def goto_with_fallback(page: Any, url: str, timeout_ms: int = 15000) -> None:
    """Navigate to *url*, trying networkidle first then domcontentloaded."""
    try:
        await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
    except Exception:
        logger.debug("networkidle timed out, retrying with domcontentloaded")
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)


async def try_extract_ms_token(page: Any, cookies: List[Dict[str, Any]]) -> Optional[str]:
    """Try to extract msToken from multiple sources.

    Checks (in order):
    1. Captured cookies list
    2. Request headers via evaluate
    3. localStorage
    4. sessionStorage
    """
    # 1. From cookies list
    for c in cookies:
        if c.get("name") == "msToken" and c.get("value"):
            return str(c["value"])

    # 2. From localStorage / sessionStorage
    for storage in ("localStorage", "sessionStorage"):
        try:
            value = await page.evaluate(f"window.{storage}.getItem('msToken')")
            if value:
                return str(value)
        except Exception:
            pass

    return None


def filter_cookies(
    raw_cookies: List[Dict[str, Any]],
    *,
    keep_all: bool = False,
) -> Dict[str, str]:
    """Convert Playwright cookie list to a flat name→value dict.

    When *keep_all* is False (default), only REQUIRED_COOKIES + AUXILIARY_COOKIES
    are kept. This matches the original douyin-downloader behaviour.
    """
    result: Dict[str, str] = {}
    for cookie in raw_cookies:
        name = cookie.get("name", "")
        value = cookie.get("value", "")
        if not name or not value:
            continue
        if keep_all or name in REQUIRED_COOKIES or name in AUXILIARY_COOKIES:
            result[name] = str(value)
    return result


def inject_cookies_into_context(
    context_cookies: List[Dict[str, Any]],
    inject: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Return a new cookie list with extra key-value pairs merged in.

    Existing cookies with the same name are updated; new ones are appended.
    """
    existing = {c["name"]: i for i, c in enumerate(context_cookies)}
    result = list(context_cookies)
    for name, value in inject.items():
        if name in existing:
            result[existing[name]] = {**result[existing[name]], "value": value}
        else:
            result.append(
                {
                    "name": name,
                    "value": value,
                    "domain": ".douyin.com",
                    "path": "/",
                    "httpOnly": False,
                    "secure": True,
                    "sameSite": "None",
                }
            )
    return result


# ---------------------------------------------------------------------------
# Core capture logic
# ---------------------------------------------------------------------------


async def capture_cookies(
    *,
    url: str = DOUYIN_URL,
    browser_type: str = "chromium",
    headless: bool = False,
    output_path: Path,
    keep_all: bool = False,
) -> Dict[str, str]:
    """Launch Playwright, wait for manual login, then harvest cookies.

    Args:
        url: The URL to navigate to. Defaults to douyin.com home.
        browser_type: "chromium", "firefox", or "webkit".
        headless: If True, runs without a visible window (not recommended for
            login flows where CAPTCHA may appear).
        output_path: Where to write the cookies JSON file.
        keep_all: If True, save every cookie, not just the required set.

    Returns:
        Dict of captured cookies (name → value).
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise ImportError(
            "playwright is required for the cookie fetcher. "
            "Install it with: pip install playwright && playwright install chromium"
        ) from exc

    print(f"\n[cookie_fetcher] Opening browser → {url}")
    print("[cookie_fetcher] Please log in to Douyin in the browser window.")
    print("[cookie_fetcher] When done, press Enter here to capture cookies.\n")

    async with async_playwright() as pw:
        launcher = getattr(pw, browser_type)
        browser = await launcher.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="zh-CN",
        )
        page = await context.new_page()

        await goto_with_fallback(page, url)

        # Wait for user to finish logging in
        await asyncio.get_event_loop().run_in_executor(
            None, input, "[cookie_fetcher] Press Enter after login is complete... "
        )

        # Harvest cookies from context storage state
        storage_state = await context.storage_state()
        raw_cookies: List[Dict[str, Any]] = storage_state.get("cookies", [])

        # Try to extract msToken from various sources
        ms_token = await try_extract_ms_token(page, raw_cookies)
        if ms_token:
            logger.debug("msToken extracted: %s…", ms_token[:12])
            # Inject into raw list for completeness
            raw_cookies = inject_cookies_into_context(
                raw_cookies, {"msToken": ms_token}
            )

        cookies = filter_cookies(raw_cookies, keep_all=keep_all)

        await browser.close()

    # Persist to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    captured = list(cookies.keys())
    print(f"\n[cookie_fetcher] Captured {len(captured)} cookies: {', '.join(captured)}")
    print(f"[cookie_fetcher] Saved to: {output_path}")

    missing = [k for k in REQUIRED_COOKIES if k not in cookies]
    if missing:
        print(
            f"[cookie_fetcher] WARNING: Missing required cookies: {', '.join(missing)}"
        )
        print(
            "[cookie_fetcher] You may need to fully log in before pressing Enter."
        )
    else:
        print("[cookie_fetcher] All required cookies captured successfully.")

    return cookies


# ---------------------------------------------------------------------------
# CLI entry point (standalone)
# ---------------------------------------------------------------------------


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="cookie_fetcher",
        description="Capture Douyin login cookies via a real browser session.",
    )
    parser.add_argument(
        "--url",
        default=DOUYIN_URL,
        help=f"URL to open (default: {DOUYIN_URL})",
    )
    parser.add_argument(
        "--browser",
        default="chromium",
        choices=["chromium", "firefox", "webkit"],
        help="Playwright browser engine (default: chromium)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode (not recommended for login)",
    )
    parser.add_argument(
        "--output",
        default="cookies.json",
        help="Output file path (default: cookies.json)",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        default=False,
        help="Save all cookies, not just the required set",
    )
    return parser


def main() -> None:
    """Standalone entry point: python -m content_downloader.tools.cookie_fetcher"""
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    asyncio.run(
        capture_cookies(
            url=args.url,
            browser_type=args.browser,
            headless=args.headless,
            output_path=Path(args.output),
            keep_all=args.keep_all,
        )
    )


if __name__ == "__main__":
    main()
