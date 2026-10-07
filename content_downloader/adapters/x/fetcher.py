"""X (Twitter) content fetcher using yt-dlp as an external CLI tool."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_YTDLP_NOT_INSTALLED_MESSAGE = (
    "yt-dlp is not installed or not found in PATH.\n"
    "Install it with:\n"
    "  pip install yt-dlp\n"
    "or:\n"
    "  brew install yt-dlp"
)


def _find_info_json(directory: Path) -> "Path | None":
    """Return the first .info.json file found in *directory*, or None."""
    matches = list(directory.glob("*.info.json"))
    if not matches:
        return None
    return matches[0]


class XFetcher:
    """Fetch X/Twitter post metadata and media using yt-dlp as a subprocess.

    yt-dlp is invoked via asyncio.create_subprocess_exec (not shell) to
    avoid command-injection vulnerabilities. The URL is passed as a positional
    argument, never interpolated into a shell string.
    """

    async def is_available(self) -> bool:
        """Return True if yt-dlp is installed and accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "yt-dlp",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            return proc.returncode == 0
        except (FileNotFoundError, OSError):
            return False

    async def fetch_post(self, url: str, output_dir: Path) -> "dict[str, Any]":
        """Download a tweet's media and return its yt-dlp info dict.

        Two-phase approach:
        1. --dump-json to get metadata (works for all tweets, even text-only)
        2. If tweet has media, download it with --write-thumbnail

        Args:
            url: The tweet URL (x.com or twitter.com).
            output_dir: Directory where yt-dlp output files are written.

        Returns:
            Parsed dict from yt-dlp's JSON output.

        Raises:
            RuntimeError: If yt-dlp fails to fetch metadata.
        """
        media_dir = output_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Get metadata (works for all tweets)
        info = await self._fetch_metadata(url)

        # Phase 2: Download media if available
        has_media = info.get("url") or info.get("formats")
        if has_media:
            await self._download_media(url, media_dir)

        return info

    async def _fetch_metadata(self, url: str) -> "dict[str, Any]":
        """Use --dump-json to get tweet metadata without downloading."""
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--dump-json",
            "--no-playlist",
            "--no-warnings",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        # yt-dlp returns code 1 for text-only tweets — try to parse stderr
        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace") if stderr else ""
            # These errors mean the tweet has no native media — treat as text-only
            # - "No video could be found" = text-only tweet
            # - "Unsupported URL" = yt-dlp followed an embedded link in the tweet
            if "No video could be found" in stderr_text or "Unsupported URL" in stderr_text:
                logger.info("Text-only tweet (or embedded link only): %s", url)
                return self._build_text_only_info(url)
            raise RuntimeError(
                f"yt-dlp metadata fetch failed (code {proc.returncode}).\n"
                f"URL: {url!r}\nstderr: {stderr_text}"
            )

        stdout_text = stdout.decode(errors="replace").strip()
        if not stdout_text:
            return self._build_text_only_info(url)

        return json.loads(stdout_text)

    async def _download_media(self, url: str, media_dir: Path) -> None:
        """Download media files using yt-dlp."""
        output_template = str(media_dir / "%(id)s.%(ext)s")
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--write-thumbnail",
            "--no-playlist",
            "--no-warnings",
            "-o",
            output_template,
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if stdout:
            logger.debug("yt-dlp download stdout: %s", stdout.decode(errors="replace"))
        if stderr:
            logger.debug("yt-dlp download stderr: %s", stderr.decode(errors="replace"))

    @staticmethod
    def _build_text_only_info(url: str) -> "dict[str, Any]":
        """Build a minimal info dict for text-only tweets."""
        # Extract tweet ID from URL
        tweet_id = ""
        for part in url.rstrip("/").split("/"):
            if part.isdigit():
                tweet_id = part
                break
        return {
            "id": tweet_id,
            "title": "",
            "description": "",
            "uploader": "",
            "uploader_id": "",
            "timestamp": None,
            "like_count": None,
            "repost_count": None,
            "comment_count": None,
            "view_count": None,
            "webpage_url": url,
            "_text_only": True,
        }
