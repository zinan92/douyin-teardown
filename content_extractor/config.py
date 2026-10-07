"""Centralized configuration for content-extractor.

ExtractorConfig is a frozen Pydantic model holding all extraction settings.
CLI-exposed fields (whisper_model, force_reprocess, output_dir) follow D-05.
LLM fields are hardcoded defaults, not exposed to CLI (per D-06).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ExtractorConfig(BaseModel):
    """Frozen configuration for the extraction pipeline.

    CLI-exposed settings can be overridden at construction time.
    LLM settings are internal defaults -- not surfaced in the CLI.
    """

    model_config = ConfigDict(frozen=True)

    # CLI-exposed settings (per D-05)
    whisper_model: str = "turbo"
    force_reprocess: bool = False
    output_dir: Path | None = None

    # False = transcription only: skip the LLM restructuring and summary (douyin-teardown
    # does its own analysis and should not try, fail and warn on a second AI first).
    llm_enabled: bool = True

    # LLM settings -- hardcoded defaults, not exposed to CLI (per D-06)
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 4096
    claude_temperature: float = 0.0
