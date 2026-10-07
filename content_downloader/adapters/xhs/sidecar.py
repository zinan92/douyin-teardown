"""XHS-Downloader sidecar — auto-install, auto-start, health check."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from content_downloader.adapters.xhs.api_client import XHSAPIClient

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 5556
_STARTUP_TIMEOUT = 30  # seconds to wait for sidecar to become healthy
_HEALTH_POLL_INTERVAL = 1.0


class XHSSidecar:
    """Manages the XHS-Downloader sidecar process lifecycle.

    Automatically installs (pip) and starts XHS-Downloader if not running.
    """

    def __init__(self, base_url: str = f"http://127.0.0.1:{_DEFAULT_PORT}") -> None:
        self._base_url = base_url
        self._process: subprocess.Popen | None = None

    async def ensure_running(self) -> bool:
        """Ensure XHS-Downloader is running. Auto-start if needed.

        Returns True if sidecar is healthy.
        """
        # 1. Already running?
        if await self._check_health():
            logger.debug("XHS sidecar already running at %s", self._base_url)
            return True

        # 2. Is xhs-downloader installed?
        if not self._is_installed():
            logger.info("XHS-Downloader not found, installing via pip...")
            self._install()

        # 3. Start sidecar
        logger.info("Starting XHS-Downloader sidecar...")
        self._start()

        # 4. Wait for healthy
        return await self._wait_for_healthy()

    async def _check_health(self) -> bool:
        try:
            async with XHSAPIClient(base_url=self._base_url) as client:
                return await client.is_available()
        except Exception:
            return False

    def _get_install_dir(self) -> Path:
        return Path.home() / ".content-downloader" / "XHS-Downloader"

    def _is_installed(self) -> bool:
        """Check if XHS-Downloader is available (cloned repo with main.py)."""
        main_py = self._get_install_dir() / "main.py"
        return main_py.exists()

    def _install(self) -> None:
        """Clone XHS-Downloader from GitHub and install its dependencies."""
        install_dir = self._get_install_dir()
        if not install_dir.exists():
            logger.info("Cloning XHS-Downloader from GitHub...")
            install_dir.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/JoeanAmier/XHS-Downloader.git",
                 str(install_dir)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.error("git clone failed: %s", result.stderr)
                return

        # Install requirements
        req_file = install_dir / "requirements.txt"
        if req_file.exists():
            logger.info("Installing XHS-Downloader dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
                capture_output=True, text=True, timeout=120,
            )

    def _start(self) -> None:
        """Start XHS-Downloader in API mode as a background process."""
        repo_dir = self._get_install_dir()
        main_py = repo_dir / "main.py"
        if not main_py.exists():
            logger.error("Cannot start XHS-Downloader — main.py not found at %s", main_py)
            return

        logger.info("Starting XHS-Downloader API at %s ...", self._base_url)
        self._process = subprocess.Popen(
            [sys.executable, str(main_py), "api"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(repo_dir),
        )

    async def _wait_for_healthy(self) -> bool:
        """Poll until sidecar is healthy or timeout."""
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if await self._check_health():
                logger.info("XHS sidecar is ready at %s", self._base_url)
                return True
            await asyncio.sleep(_HEALTH_POLL_INTERVAL)

        logger.error("XHS sidecar failed to start within %ds", _STARTUP_TIMEOUT)
        return False

    def stop(self) -> None:
        """Stop the sidecar process if we started it."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
