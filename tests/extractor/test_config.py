"""Tests for ExtractorConfig frozen configuration model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from content_extractor.config import ExtractorConfig


class TestExtractorConfigDefaults:
    """ExtractorConfig() with no args has sensible defaults."""

    def test_default_whisper_model(self) -> None:
        config = ExtractorConfig()
        assert config.whisper_model == "turbo"

    def test_default_force_reprocess(self) -> None:
        config = ExtractorConfig()
        assert config.force_reprocess is False

    def test_default_output_dir(self) -> None:
        config = ExtractorConfig()
        assert config.output_dir is None


class TestExtractorConfigOverrides:
    """CLI-exposed settings can be overridden at construction time."""

    def test_override_whisper_model(self) -> None:
        config = ExtractorConfig(whisper_model="large-v3")
        assert config.whisper_model == "large-v3"

    def test_override_force_reprocess(self) -> None:
        config = ExtractorConfig(force_reprocess=True)
        assert config.force_reprocess is True


class TestExtractorConfigFrozen:
    """ExtractorConfig is frozen -- attribute assignment raises."""

    def test_frozen_whisper_model(self) -> None:
        config = ExtractorConfig()
        with pytest.raises(ValidationError):
            config.whisper_model = "large-v3"  # type: ignore[misc]

    def test_frozen_force_reprocess(self) -> None:
        config = ExtractorConfig()
        with pytest.raises(ValidationError):
            config.force_reprocess = True  # type: ignore[misc]


class TestExtractorConfigLLMDefaults:
    """LLM settings are hardcoded defaults, not CLI-exposed (per D-06)."""

    def test_claude_model_default(self) -> None:
        config = ExtractorConfig()
        assert config.claude_model == "claude-sonnet-4-20250514"

    def test_claude_max_tokens_default(self) -> None:
        config = ExtractorConfig()
        assert config.claude_max_tokens == 4096

    def test_claude_temperature_default(self) -> None:
        config = ExtractorConfig()
        assert config.claude_temperature == 0.0
