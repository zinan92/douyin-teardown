"""Douyin API client — migrated from douyin-downloader-1/core/api_client.py.

Key changes from original:
- httpx replaces aiohttp for HTTP calls (consistent with project deps)
- Playwright browser fallback added (get_video_detail_via_browser)
- Removed get_user_like/music/mix/collects (Phase 2: post mode only)
- Error handling maps to DownloadError conventions
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

try:
    from playwright.async_api import async_playwright as _async_playwright

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _async_playwright = None  # type: ignore[assignment]
    _PLAYWRIGHT_AVAILABLE = False

from content_downloader.adapters.douyin.cookie_manager import _sanitize_cookies
from content_downloader.adapters.douyin.ms_token import MsTokenManager
from content_downloader.adapters.douyin.xbogus import XBogus

try:
    from content_downloader.adapters.douyin.abogus import ABogus, BrowserFingerprintGenerator
except Exception:  # pragma: no cover
    ABogus = None  # type: ignore[assignment,misc]
    BrowserFingerprintGenerator = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# All Chrome/Mac user agents — consistent with the macOS runtime environment
# and with the fingerprint used in _default_query().
_USER_AGENT_POOL = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
]


def _landed_on_video(page_url: str, aweme_id: str) -> bool:
    """Whether the browser is still showing this video.

    A video that is private, removed or restricted does not open on the web: Douyin
    redirects to the 精选 feed (``previous_page=web_video_404_link``) and autoplays
    something else. Anything captured from that page belongs to another video.
    """
    parsed = urlparse(page_url or "")
    if "404" in parsed.query:
        return False
    if parsed.path.rstrip("/").endswith(f"/video/{aweme_id}"):
        return True
    return parse_qs(parsed.query).get("modal_id", [""])[0] == aweme_id


class DouyinAPIClient:
    """Async Douyin web API client with XBogus/ABogus request signing."""

    BASE_URL = "https://www.douyin.com"

    def __init__(
        self,
        cookies: Optional[Dict[str, str]] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self.cookies = _sanitize_cookies(cookies or {})
        self.proxy = str(proxy or "").strip() or None
        self._client: Optional[httpx.AsyncClient] = None
        selected_ua = random.choice(_USER_AGENT_POOL)
        self.headers = {
            "User-Agent": selected_ua,
            "Referer": "https://www.douyin.com/",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
        }
        self._signer = XBogus(self.headers["User-Agent"])
        self._ms_token_manager = MsTokenManager(user_agent=self.headers["User-Agent"])
        self._ms_token = (self.cookies.get("msToken") or "").strip()
        self._abogus_enabled = (
            ABogus is not None and BrowserFingerprintGenerator is not None
        )

    async def __aenter__(self) -> "DouyinAPIClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _ensure_client(self) -> None:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                cookies=self.cookies,
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
                proxy=self.proxy,
            )

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _ensure_ms_token(self) -> str:
        if self._ms_token:
            return self._ms_token

        token = await asyncio.to_thread(
            self._ms_token_manager.ensure_ms_token,
            self.cookies,
        )
        self._ms_token = token.strip()
        if self._ms_token:
            self.cookies["msToken"] = self._ms_token
        return self._ms_token

    async def _default_query(self) -> Dict[str, Any]:
        ms_token = await self._ensure_ms_token()
        return {
            "device_platform": "webapp",
            "aid": "6383",
            "channel": "channel_pc_web",
            "pc_client_type": "1",
            "version_code": "170400",
            "version_name": "17.4.0",
            "cookie_enabled": "true",
            "screen_width": "1440",
            "screen_height": "900",
            "browser_language": "zh-CN",
            "browser_platform": "MacIntel",
            "browser_name": "Chrome",
            "browser_version": "131.0.0.0",
            "browser_online": "true",
            "engine_name": "Blink",
            "engine_version": "131.0.0.0",
            "os_name": "Mac",
            "os_version": "10_15_7",
            "cpu_core_num": "8",
            "device_memory": "8",
            "platform": "PC",
            "downlink": "10",
            "effective_type": "4g",
            "round_trip_time": "50",
            "msToken": ms_token,
        }

    def sign_url(self, url: str) -> Tuple[str, str]:
        """Sign URL with XBogus. Returns (signed_url, user_agent)."""
        signed_url, _xbogus, ua = self._signer.build(url)
        return signed_url, ua

    def build_signed_path(self, path: str, params: Dict[str, Any]) -> Tuple[str, str]:
        """Build a signed URL for the given API path and params."""
        query = urlencode(params)
        base_url = f"{self.BASE_URL}{path}"
        ab_signed = self._build_abogus_url(base_url, query)
        if ab_signed:
            return ab_signed
        return self.sign_url(f"{base_url}?{query}")

    def _build_abogus_url(
        self, base_url: str, query: str
    ) -> Optional[Tuple[str, str]]:
        if not self._abogus_enabled:
            return None
        try:
            browser_fp = BrowserFingerprintGenerator.generate_fingerprint("Chrome")
            signer = ABogus(fp=browser_fp, user_agent=self.headers["User-Agent"])
            params_with_ab, _ab, ua, _body = signer.generate_abogus(query, "")
            return f"{base_url}?{params_with_ab}", ua
        except Exception as exc:
            logger.warning("Failed to generate a_bogus, fallback to X-Bogus: %s", exc)
            return None

    async def _request_json(
        self,
        path: str,
        params: Dict[str, Any],
        *,
        suppress_error: bool = False,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        await self._ensure_client()
        delays = [1, 2, 5]
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            signed_url, ua = self.build_signed_path(path, params)
            try:
                assert self._client is not None
                response = await self._client.get(
                    signed_url,
                    headers={**self.headers, "User-Agent": ua},
                )
                if response.status_code == 200:
                    data = response.json()
                    return data if isinstance(data, dict) else {}
                if response.status_code < 500 and response.status_code != 429:
                    log_fn = logger.debug if suppress_error else logger.error
                    log_fn(
                        "Request failed: path=%s, status=%s",
                        path,
                        response.status_code,
                    )
                    return {}
                last_exc = RuntimeError(
                    f"HTTP {response.status_code} for {path}"
                )
            except Exception as exc:
                last_exc = exc

            if attempt < max_retries - 1:
                delay = delays[min(attempt, len(delays) - 1)]
                logger.debug(
                    "Request retry %d/%d for %s in %ds",
                    attempt + 1, max_retries, path, delay,
                )
                await asyncio.sleep(delay)

        log_fn = logger.debug if suppress_error else logger.error
        log_fn(
            "Request failed after %d attempts: path=%s, error=%s",
            max_retries, path, last_exc,
        )
        return {}

    @staticmethod
    def _normalize_paged_response(
        raw_data: Any,
        *,
        item_keys: Optional[List[str]] = None,
        source: str = "api",
    ) -> Dict[str, Any]:
        raw = raw_data if isinstance(raw_data, dict) else {}
        keys = item_keys or []
        keys = ["items", *keys, "aweme_list", "mix_list", "music_list"]

        items: List[Dict[str, Any]] = []
        for key in keys:
            value = raw.get(key)
            if isinstance(value, list):
                items = value
                break

        has_more_value = raw.get("has_more", False)
        try:
            has_more = bool(int(has_more_value))
        except (TypeError, ValueError):
            has_more = bool(has_more_value)

        max_cursor_value = raw.get("max_cursor")
        if max_cursor_value is None:
            max_cursor_value = raw.get("cursor", 0)
        try:
            max_cursor = int(max_cursor_value or 0)
        except (TypeError, ValueError):
            max_cursor = 0

        status_code_value = raw.get("status_code", 0)
        try:
            status_code = int(status_code_value or 0)
        except (TypeError, ValueError):
            status_code = 0

        risk_flags = {
            "login_tip": bool(
                ((raw.get("not_login_module") or {}).get("guide_login_tip_exist"))
                if isinstance(raw.get("not_login_module"), dict)
                else False
            ),
            "verify_page": bool(raw.get("verify_ticket")),
        }

        normalized = {
            "items": items,
            "aweme_list": items,
            "has_more": has_more,
            "max_cursor": max_cursor,
            "status_code": status_code,
            "source": source,
            "risk_flags": risk_flags,
            "raw": raw,
        }
        for key, value in raw.items():
            if key not in normalized:
                normalized[key] = value
        return normalized

    async def _build_user_page_params(
        self, sec_uid: str, max_cursor: int, count: int
    ) -> Dict[str, Any]:
        params = await self._default_query()
        params.update(
            {
                "sec_user_id": sec_uid,
                "max_cursor": max_cursor,
                "count": count,
                "locate_query": "false",
            }
        )
        return params

    async def _get_video_detail_api(
        self, aweme_id: str, *, suppress_error: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Attempt to fetch video metadata via the signed API (XBogus/ABogus)."""
        params = await self._default_query()
        params.update(
            {
                "aweme_id": aweme_id,
                "aid": "6383",
            }
        )
        data = await self._request_json(
            "/aweme/v1/web/aweme/detail/",
            params,
            suppress_error=suppress_error,
        )
        if data:
            return data.get("aweme_detail")
        return None

    async def get_video_detail_via_browser(
        self, aweme_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch video metadata using a real browser.

        Three extraction strategies (tried in order):
        1. Intercept /aweme/v1/web/aweme/detail/ XHR response
        2. Extract from SSR data in page script tags (RENDER_DATA / __NEXT_DATA__)
        3. Extract video URL directly from page DOM
        """
        if not _PLAYWRIGHT_AVAILABLE:
            logger.warning(
                "playwright is not installed — browser fallback unavailable. "
                "Install with: pip install playwright && playwright install chromium"
            )
            return None

        video_url = f"{self.BASE_URL}/video/{aweme_id}"
        logger.info("Browser fallback: opening %s", video_url)

        captured_detail: Optional[Dict[str, Any]] = None
        captured_video_urls: List[str] = []
        detail_event = asyncio.Event()

        async with _async_playwright() as pw:
            try:
                browser = await pw.chromium.launch(
                    headless=False,
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except Exception as exc:
                logger.warning("Browser fallback unavailable: %s", exc)
                return None
            context = await browser.new_context(
                user_agent=self.headers["User-Agent"],
                viewport={"width": 1280, "height": 800},
                locale="zh-CN",
            )

            if self.cookies:
                playwright_cookies = [
                    {
                        "name": name,
                        "value": value,
                        "domain": ".douyin.com",
                        "path": "/",
                        "httpOnly": False,
                        "secure": True,
                        "sameSite": "None",
                    }
                    for name, value in self.cookies.items()
                    if name and value
                ]
                await context.add_cookies(playwright_cookies)

            page = await context.new_page()

            # Strategy 1: Intercept API response + capture video CDN URLs
            async def _on_response(response: Any) -> None:
                nonlocal captured_detail
                url = response.url or ""

                # Capture video CDN URLs from network requests
                if "douyinvod.com" in url and ("video" in url or ".mp4" in url):
                    content_type = response.headers.get("content-type", "")
                    if "video" in content_type or url.endswith(".mp4") or "mime_type=video" in url:
                        captured_video_urls.append(url)
                        logger.debug("Browser fallback: captured CDN video URL")

                if "/aweme/v1/web/aweme/detail/" not in url and "/aweme/detail/" not in url:
                    return
                try:
                    body = await response.body()
                    data = json.loads(body)
                    detail = data.get("aweme_detail")
                    if detail and str(detail.get("aweme_id") or "") == str(aweme_id):
                        captured_detail = detail
                        detail_event.set()
                        logger.info("Browser fallback: intercepted API response for %s", aweme_id)
                except Exception as exc:
                    logger.debug("Browser fallback: failed to parse API response: %s", exc)

            page.on("response", _on_response)

            try:
                await page.goto(video_url, wait_until="domcontentloaded", timeout=30000)
            except Exception as exc:
                logger.warning("Browser fallback: navigation error (continuing): %s", exc)

            # Wait a few seconds for API intercept
            try:
                await asyncio.wait_for(detail_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.info("Browser fallback: no API intercept, trying SSR extraction...")

            if not captured_detail and not _landed_on_video(page.url, aweme_id):
                # Redirected away (private / removed / restricted): every video on this
                # page is someone else's. Give up instead of downloading the wrong one.
                logger.warning(
                    "Browser fallback: %s does not open on the web (landed on %s)",
                    aweme_id, page.url,
                )
                await browser.close()
                return None

            # Strategy 2: Extract from SSR script tags
            if not captured_detail:
                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass

                ssr_detail = await self._extract_ssr_data(page, aweme_id)
                if ssr_detail:
                    captured_detail = ssr_detail
                    logger.info("Browser fallback: extracted from SSR data for %s", aweme_id)

            # Strategy 3: Extract video URL from DOM or captured network requests
            if not captured_detail:
                dom_detail = await self._extract_from_dom(page, aweme_id)
                if dom_detail:
                    captured_detail = dom_detail
                    logger.info("Browser fallback: extracted from DOM for %s", aweme_id)

            # Strategy 4: Use captured CDN video URLs from network traffic
            if not captured_detail and captured_video_urls:
                captured_detail = {
                    "aweme_id": aweme_id,
                    "desc": "",
                    "video": {
                        "play_addr": {
                            "url_list": captured_video_urls,
                        },
                    },
                    "_source": "network_intercept",
                }
                try:
                    title = await page.title()
                    captured_detail["desc"] = title or ""
                except Exception:
                    pass
                logger.info("Browser fallback: using %d captured CDN URLs for %s", len(captured_video_urls), aweme_id)

            # Sync cookies back
            try:
                storage = await context.storage_state()
                for c in storage.get("cookies", []):
                    name = c.get("name", "")
                    value = c.get("value", "")
                    if name and value:
                        self.cookies[name] = value
            except Exception as exc:
                logger.debug("Browser fallback: could not sync cookies: %s", exc)

            await browser.close()

        return captured_detail

    async def _extract_ssr_data(
        self, page: Any, aweme_id: str
    ) -> Optional[Dict[str, Any]]:
        """Extract video detail from SSR data embedded in page script tags."""
        js = """
() => {
    // Try RENDER_DATA (URL-encoded JSON in script#RENDER_DATA)
    const renderEl = document.getElementById('RENDER_DATA');
    if (renderEl) {
        try {
            const decoded = decodeURIComponent(renderEl.textContent);
            return { type: 'RENDER_DATA', data: decoded };
        } catch(e) {}
    }

    // Try __NEXT_DATA__
    const nextEl = document.getElementById('__NEXT_DATA__');
    if (nextEl) {
        return { type: '__NEXT_DATA__', data: nextEl.textContent };
    }

    // Try window._SSR_DATA or window.__SSR_DATA__
    if (window._SSR_DATA) {
        return { type: '_SSR_DATA', data: JSON.stringify(window._SSR_DATA) };
    }
    if (window.__SSR_DATA__) {
        return { type: '__SSR_DATA__', data: JSON.stringify(window.__SSR_DATA__) };
    }

    // Scan all script tags for aweme_detail or playAddr
    const scripts = document.querySelectorAll('script');
    for (const s of scripts) {
        const t = s.textContent || '';
        if (t.includes('aweme_detail') || t.includes('playAddr') || t.includes('play_addr')) {
            if (t.length > 100 && t.length < 500000) {
                return { type: 'script_tag', data: t };
            }
        }
    }

    return null;
}
"""
        try:
            result = await page.evaluate(js)
            if not result:
                return None

            data_str = result.get("data", "")
            source = result.get("type", "unknown")
            logger.debug("Browser fallback: found SSR source: %s (%d chars)", source, len(data_str))

            data = json.loads(data_str)

            # Navigate the nested structure to find aweme_detail
            detail = self._find_aweme_detail(data, aweme_id)
            return detail
        except Exception as exc:
            logger.debug("Browser fallback: SSR extraction failed: %s", exc)
            return None

    @staticmethod
    def _find_aweme_detail(data: Any, aweme_id: str) -> Optional[Dict[str, Any]]:
        """Recursively search for aweme_detail in nested data structures."""
        if not isinstance(data, dict):
            return None

        # Direct match
        if "aweme_detail" in data:
            detail = data["aweme_detail"]
            if isinstance(detail, dict):
                return detail

        # RENDER_DATA format: nested under numbered keys
        for key, value in data.items():
            if not isinstance(value, dict):
                continue

            # Check this level
            if "aweme_detail" in value:
                detail = value["aweme_detail"]
                if isinstance(detail, dict):
                    return detail

            # Check awemeDetail (camelCase variant)
            if "awemeDetail" in value:
                detail = value["awemeDetail"]
                if isinstance(detail, dict):
                    return detail

            # Go one more level deep
            for k2, v2 in value.items():
                if not isinstance(v2, dict):
                    continue
                if "aweme_detail" in v2:
                    detail = v2["aweme_detail"]
                    if isinstance(detail, dict):
                        return detail
                if "awemeDetail" in v2:
                    detail = v2["awemeDetail"]
                    if isinstance(detail, dict):
                        return detail

        return None

    async def _extract_from_dom(
        self, page: Any, aweme_id: str
    ) -> Optional[Dict[str, Any]]:
        """Last resort: extract video URL from page DOM and network requests."""
        js = """
() => {
    // Strategy A: Find non-blob video src
    const videos = document.querySelectorAll('video, video source');
    for (const v of videos) {
        const src = v.src || v.getAttribute('src') || '';
        if (src && src.startsWith('http') && !src.startsWith('blob:')) {
            return { src: src, title: document.title || '' };
        }
    }

    // Strategy B: Search page HTML for CDN video URLs
    const html = document.documentElement.innerHTML || '';
    const cdnPatterns = [
        /https?:\/\/v[0-9a-z-]+\.douyinvod\.com\/[^\s"'<>]+/g,
        /https?:\/\/[^\s"'<>]*douyinvod[^\s"'<>]+\.mp4[^\s"'<>]*/g,
        /https?:\/\/[^\s"'<>]*play_addr[^\s"'<>]*/g,
    ];
    for (const pattern of cdnPatterns) {
        const matches = html.match(pattern);
        if (matches && matches.length > 0) {
            // Pick the longest URL (most likely the full CDN URL)
            const best = matches.sort((a, b) => b.length - a.length)[0];
            return { src: best, title: document.title || '' };
        }
    }

    // Strategy C: Check window.__playAddr or similar globals
    try {
        const globals = ['__playAddr', '__videoUrl'];
        for (const g of globals) {
            if (window[g] && typeof window[g] === 'string' && window[g].startsWith('http')) {
                return { src: window[g], title: document.title || '' };
            }
        }
    } catch(e) {}

    return null;
}
"""
        try:
            result = await page.evaluate(js)
            if not result or not result.get("src"):
                return None

            # Build a minimal aweme_detail-like dict
            return {
                "aweme_id": aweme_id,
                "desc": result.get("title", ""),
                "video": {
                    "play_addr": {
                        "url_list": [result["src"]],
                    },
                },
                "_source": "dom_extraction",
            }
        except Exception as exc:
            logger.debug("Browser fallback: DOM extraction failed: %s", exc)
            return None

    async def get_video_detail(
        self, aweme_id: str, *, suppress_error: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a single video by aweme_id.

        Tries the signed API first; if that returns nothing, falls back to
        intercepting the response in a real Playwright browser session.
        """
        detail = await self._get_video_detail_api(aweme_id, suppress_error=suppress_error)
        if detail:
            return detail
        logger.info(
            "API signing rejected for aweme_id=%s — trying browser fallback", aweme_id
        )
        return await self.get_video_detail_via_browser(aweme_id)

    async def get_user_post(
        self, sec_uid: str, max_cursor: int = 0, count: int = 20
    ) -> Dict[str, Any]:
        """Fetch paginated list of a user's posts."""
        params = await self._build_user_page_params(sec_uid, max_cursor, count)
        params.update(
            {
                "show_live_replay_strategy": "1",
                "need_time_list": "1",
                "time_list_query": "0",
                "whale_cut_token": "",
                "cut_version": "1",
                "publish_video_strategy_type": "2",
            }
        )
        raw = await self._request_json("/aweme/v1/web/aweme/post/", params)
        return self._normalize_paged_response(raw, item_keys=["aweme_list"])

    async def resolve_short_url(self, short_url: str) -> Optional[str]:
        """Resolve a v.douyin.com short URL to its canonical long URL."""
        try:
            await self._ensure_client()
            assert self._client is not None
            response = await self._client.get(
                short_url,
                follow_redirects=True,
            )
            return str(response.url)
        except Exception as exc:
            logger.error("Failed to resolve short URL: %s, error: %s", short_url, exc)
            return None
