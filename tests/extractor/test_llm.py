"""Tests for LLM client infrastructure: token loading, client factory, error types."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from content_extractor.llm import (
    LLMAPIError,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    _load_token_from_cli_proxy,
    create_claude_client,
)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_llm_config_error_is_llm_error(self) -> None:
        assert issubclass(LLMConfigError, LLMError)

    def test_llm_rate_limit_error_is_llm_error(self) -> None:
        assert issubclass(LLMRateLimitError, LLMError)

    def test_llm_api_error_is_llm_error(self) -> None:
        assert issubclass(LLMAPIError, LLMError)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_token_file(
    directory: Path,
    *,
    email: str = "test@example.com",
    access_token: str = "sk-ant-oat01-test-token",
    disabled: bool = False,
    expired: str = "",
    token_type: str = "claude",
) -> Path:
    """Write a CLI Proxy API token file and return its path."""
    data = {
        "access_token": access_token,
        "disabled": disabled,
        "email": email,
        "expired": expired,
        "last_refresh": "",
        "refresh_token": "sk-ant-ort01-refresh",
        "type": token_type,
    }
    filename = f"claude-{email}.json"
    path = directory / filename
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# _load_token_from_cli_proxy
# ---------------------------------------------------------------------------


class TestLoadTokenFromCliProxy:
    def test_returns_access_token_from_valid_file(self, tmp_path: Path) -> None:
        _write_token_file(tmp_path, access_token="sk-ant-oat01-valid")
        result = _load_token_from_cli_proxy(config_dir=tmp_path)
        assert result == "sk-ant-oat01-valid"

    def test_skips_disabled_tokens(self, tmp_path: Path) -> None:
        _write_token_file(tmp_path, disabled=True)
        result = _load_token_from_cli_proxy(config_dir=tmp_path)
        assert result is None

    def test_skips_expired_tokens(self, tmp_path: Path) -> None:
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        _write_token_file(tmp_path, expired=past)
        result = _load_token_from_cli_proxy(config_dir=tmp_path)
        assert result is None

    def test_returns_none_when_no_files(self, tmp_path: Path) -> None:
        result = _load_token_from_cli_proxy(config_dir=tmp_path)
        assert result is None

    def test_skips_non_claude_type(self, tmp_path: Path) -> None:
        _write_token_file(tmp_path, token_type="codex", email="codex@example.com")
        result = _load_token_from_cli_proxy(config_dir=tmp_path)
        assert result is None

    def test_accepts_future_expiry(self, tmp_path: Path) -> None:
        future = (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat()
        _write_token_file(tmp_path, expired=future, access_token="sk-ant-oat01-future")
        result = _load_token_from_cli_proxy(config_dir=tmp_path)
        assert result == "sk-ant-oat01-future"


# ---------------------------------------------------------------------------
# create_claude_client
# ---------------------------------------------------------------------------


class TestCreateClaudeClient:
    @patch("content_extractor.llm.anthropic")
    def test_uses_cli_proxy_token(self, mock_anthropic: MagicMock, tmp_path: Path) -> None:
        _write_token_file(tmp_path, access_token="sk-ant-oat01-proxy")
        create_claude_client(config_dir=tmp_path)
        mock_anthropic.Anthropic.assert_called_once_with(
            api_key="sk-ant-oat01-proxy", max_retries=5
        )

    @patch("content_extractor.llm.anthropic")
    def test_falls_back_to_env_var(self, mock_anthropic: MagicMock, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            create_claude_client(config_dir=tmp_path)
        mock_anthropic.Anthropic.assert_called_once_with(
            api_key="sk-ant-env-key", max_retries=5
        )

    def test_raises_config_error_when_no_token(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Ensure ANTHROPIC_API_KEY is not set
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(LLMConfigError, match="No Claude API token found"):
                create_claude_client(config_dir=tmp_path)

    def test_raises_config_error_for_expired_token(self, tmp_path: Path) -> None:
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        _write_token_file(tmp_path, expired=past)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(LLMConfigError, match="expired"):
                create_claude_client(config_dir=tmp_path)

    @patch("content_extractor.llm.anthropic")
    def test_sets_max_retries(self, mock_anthropic: MagicMock, tmp_path: Path) -> None:
        _write_token_file(tmp_path)
        create_claude_client(config_dir=tmp_path)
        call_kwargs = mock_anthropic.Anthropic.call_args[1]
        assert call_kwargs["max_retries"] == 5
