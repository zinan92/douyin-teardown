"""Unit tests for XHSSidecar auto-install, auto-start, and health check."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from content_downloader.adapters.xhs.sidecar import XHSSidecar


@pytest.mark.asyncio
async def test_ensure_running_when_already_healthy():
    """If sidecar is already running, ensure_running returns True immediately."""
    sidecar = XHSSidecar()
    with patch.object(sidecar, "_check_health", new_callable=AsyncMock, return_value=True):
        result = await sidecar.ensure_running()
    assert result is True


@pytest.mark.asyncio
async def test_ensure_running_auto_starts():
    """If sidecar is not running, ensure_running installs and starts it."""
    sidecar = XHSSidecar()

    health_calls = [False, True]  # first check fails, then succeeds after start

    async def mock_health():
        return health_calls.pop(0) if health_calls else True

    with patch.object(sidecar, "_check_health", side_effect=mock_health), \
         patch.object(sidecar, "_is_installed", return_value=True), \
         patch.object(sidecar, "_start"), \
         patch.object(sidecar, "_wait_for_healthy", new_callable=AsyncMock, return_value=True):
        result = await sidecar.ensure_running()
    assert result is True


@pytest.mark.asyncio
async def test_ensure_running_installs_if_missing():
    """If XHS-Downloader is not installed, ensure_running installs it first."""
    sidecar = XHSSidecar()

    with patch.object(sidecar, "_check_health", new_callable=AsyncMock, return_value=False), \
         patch.object(sidecar, "_is_installed", return_value=False), \
         patch.object(sidecar, "_install") as mock_install, \
         patch.object(sidecar, "_start"), \
         patch.object(sidecar, "_wait_for_healthy", new_callable=AsyncMock, return_value=True):
        result = await sidecar.ensure_running()

    assert result is True
    mock_install.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_running_returns_false_on_timeout():
    """If sidecar fails to start, ensure_running returns False."""
    sidecar = XHSSidecar()

    with patch.object(sidecar, "_check_health", new_callable=AsyncMock, return_value=False), \
         patch.object(sidecar, "_is_installed", return_value=True), \
         patch.object(sidecar, "_start"), \
         patch.object(sidecar, "_wait_for_healthy", new_callable=AsyncMock, return_value=False):
        result = await sidecar.ensure_running()
    assert result is False


def test_stop_terminates_process():
    """stop() terminates the sidecar process if running."""
    sidecar = XHSSidecar()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    sidecar._process = mock_proc

    sidecar.stop()

    mock_proc.terminate.assert_called_once()
    assert sidecar._process is None


def test_stop_noop_when_no_process():
    """stop() does nothing if no process was started."""
    sidecar = XHSSidecar()
    sidecar.stop()  # should not raise
