"""HTTP API client for XHS-Downloader sidecar service."""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:5556"
_DEFAULT_TIMEOUT = 30.0
_HEALTH_TIMEOUT = 3.0


class XHSAPIClient:
    """Async HTTP client for the XHS-Downloader FastAPI sidecar.

    The sidecar is expected to run at ``http://127.0.0.1:5556`` and expose:

    - ``POST /xhs/detail`` — fetch note detail + download URLs
    - ``GET /``            — health check (returns 200 when ready)

    Usage::

        async with XHSAPIClient() as client:
            data = await client.get_note_detail("https://www.xiaohongshu.com/explore/abc")
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "XHSAPIClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_note_detail(self, url: str) -> dict[str, Any]:
        """Fetch full note detail from the XHS-Downloader sidecar.

        Args:
            url: A xiaohongshu.com or xhslink.com note URL.

        Returns:
            Raw JSON dict from the sidecar response.  Returns an empty dict
            if the sidecar responds with a non-200 status or an empty body.

        Raises:
            httpx.ConnectError: If the sidecar is not reachable.
            httpx.HTTPStatusError: If the sidecar returns a 4xx/5xx error.
        """
        payload = {
            "url": url,
            "download": False,
            "skip": False,
        }
        logger.debug("XHS API request: POST /xhs/detail url=%s", url)
        resp = await self._client.post(
            f"{self.base_url}/xhs/detail",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.debug("XHS API response: %s", data)
        return data if isinstance(data, dict) else {}

    async def is_available(self) -> bool:
        """Return True if the XHS-Downloader sidecar is reachable.

        Tries GET / first, then POST /xhs/detail as fallback.
        Accepts any successful connection (status < 500) as "available".
        """
        try:
            resp = await self._client.get(
                self.base_url,
                timeout=_HEALTH_TIMEOUT,
            )
            return resp.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            pass

        # Fallback: try the actual API endpoint
        try:
            resp = await self._client.post(
                f"{self.base_url}/xhs/detail",
                json={"url": "https://www.xiaohongshu.com/explore/test", "download": False},
                timeout=_HEALTH_TIMEOUT,
            )
            # Even if it returns error data, the server is running
            return resp.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.debug("XHS sidecar not available: %s", exc)
            return False
