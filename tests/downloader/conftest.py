"""Shared pytest fixtures for content-downloader tests."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory for tests."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def fixture_video_url() -> str:
    return "https://fixture.test/video/abc123"


@pytest.fixture
def fixture_profile_url() -> str:
    return "https://fixture.test/user/test-author"
