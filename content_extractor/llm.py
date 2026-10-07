"""LLM client infrastructure for content-extractor.

Routes through CLI Proxy API (localhost:8317) which exposes an OpenAI-compatible
interface backed by your Claude Max subscription. Falls back to direct Anthropic
API if ANTHROPIC_API_KEY is set.

Usage:
    from content_extractor.llm import llm_chat
    text = llm_chat(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=100,
    )
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base exception for all LLM-related errors."""


class LLMConfigError(LLMError):
    """No valid token/client found."""


class LLMAPIError(LLMError):
    """API call failed."""


# ---------------------------------------------------------------------------
# CLI Proxy API config
# ---------------------------------------------------------------------------

_DEFAULT_PROXY_URL = "http://localhost:8317/v1"
_DEFAULT_CONFIG_DIR = Path.home() / ".cli-proxy-api"
_DEFAULT_CONFIG_FILE = Path("/opt/homebrew/etc/cliproxyapi.conf")


def _load_proxy_api_key() -> str | None:
    """Load the client API key from cliproxyapi.conf.

    The proxy uses a separate client key (e.g. 'sk-cliproxy-xxxx')
    for authenticating requests from your code to the proxy server.
    This is NOT the OAuth token — that's used internally by the proxy.
    """
    if not _DEFAULT_CONFIG_FILE.exists():
        return None

    try:
        import yaml  # noqa: F811
    except ImportError:
        # No yaml, parse manually
        content = _DEFAULT_CONFIG_FILE.read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith('- "sk-') or stripped.startswith("- 'sk-"):
                return stripped.strip('- "\'')
        return None

    try:
        raw = yaml.safe_load(_DEFAULT_CONFIG_FILE.read_text())
        keys = raw.get("api-keys", [])
        if keys:
            return keys[0]
    except Exception as exc:
        logger.warning("Failed to parse cliproxyapi.conf: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Public API — single function, no SDK confusion
# ---------------------------------------------------------------------------


def llm_chat(
    *,
    model: str = "claude-sonnet-4-20250514",
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system: str | None = None,
) -> str:
    """Send a chat completion request and return the response text.

    Tries in order:
    1. CLI Proxy API (localhost:8317, OpenAI-compatible) — uses your Claude Max sub
    2. Direct Anthropic API (if ANTHROPIC_API_KEY is set)

    Returns the assistant's response text.
    Raises LLMConfigError if no client is available, LLMAPIError on failure.
    """
    # Prepend system message if provided (OpenAI format)
    full_messages = list(messages)
    if system:
        full_messages = [{"role": "system", "content": system}] + full_messages

    # Try CLI Proxy API first
    proxy_key = _load_proxy_api_key()
    if proxy_key:
        try:
            return _call_openai_compat(
                base_url=_DEFAULT_PROXY_URL,
                api_key=proxy_key,
                model=model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning("CLI Proxy API failed: %s. Trying fallback.", exc)

    # Fallback: direct Anthropic API
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        try:
            return _call_anthropic_direct(
                api_key=env_key,
                model=model,
                messages=messages,  # Original messages without system prepended
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise LLMAPIError(f"Anthropic API failed: {exc}") from exc

    msg = (
        "No LLM client available. Checked:\n"
        "  1. CLI Proxy API (localhost:8317) — is cliproxyapi running?\n"
        "     Start: /opt/homebrew/bin/cliproxyapi\n"
        "     Login: /opt/homebrew/bin/cliproxyapi -claude-login\n"
        "  2. ANTHROPIC_API_KEY environment variable — not set\n"
    )
    raise LLMConfigError(msg)


def _call_openai_compat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    """Call an OpenAI-compatible API endpoint."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def create_claude_client(
    config: "ExtractorConfig | None" = None,
    *,
    config_dir: Path | None = None,
) -> object:
    """Legacy compatibility shim for vision.py and gallery.py.

    These modules use multimodal Anthropic-specific features (base64 images)
    that can't go through the OpenAI-compat proxy. Returns an Anthropic client
    if ANTHROPIC_API_KEY is available, otherwise raises LLMConfigError with
    instructions.
    """
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        import anthropic
        return anthropic.Anthropic(api_key=env_key, max_retries=5)

    # For vision/gallery, we can't use the OpenAI-compat proxy (no multimodal support)
    msg = (
        "Vision/gallery extraction requires ANTHROPIC_API_KEY (direct API access).\n"
        "The CLI Proxy API only supports text chat, not multimodal.\n"
        "Set: export ANTHROPIC_API_KEY=sk-ant-api03-..."
    )
    raise LLMConfigError(msg)


def _call_anthropic_direct(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    system: str | None,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call Anthropic Messages API directly."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, max_retries=5)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""
