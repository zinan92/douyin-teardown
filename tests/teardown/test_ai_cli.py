from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from douyin_teardown import ai, cli, judge, pipeline


def test_unknown_provider_is_a_setup_error(monkeypatch) -> None:
    monkeypatch.setenv("TEARDOWN_AI", "gpt")
    with pytest.raises(ai.AILoginError, match="只能是"):
        ai.provider()


@pytest.mark.parametrize("name,key", [("deepseek", "TEARDOWN_DEEPSEEK_KEY"), ("anthropic", "ANTHROPIC_API_KEY")])
def test_api_provider_without_key_fails_fast(monkeypatch, name: str, key: str) -> None:
    monkeypatch.setenv("TEARDOWN_AI", name)
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(ai.AILoginError, match="key"):
        ai.ask("x")


def test_codex_runs_read_only_without_web(monkeypatch) -> None:
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        out = Path(argv[argv.index("-o") + 1])
        out.write_text('{"ok": 1}', encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setenv("TEARDOWN_AI", "codex")
    monkeypatch.setattr(ai.subprocess, "run", fake_run)
    assert ai.ask("prompt") == '{"ok": 1}'
    argv = seen["argv"]
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert 'web_search="disabled"' in argv
    assert argv[-1] == "-"


def test_claude_runs_with_no_tools(monkeypatch) -> None:
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "answer\n", "")

    monkeypatch.setenv("TEARDOWN_AI", "claude")
    monkeypatch.setattr(ai.subprocess, "run", fake_run)
    assert ai.ask("prompt") == "answer"
    argv = seen["argv"]
    assert argv[argv.index("--tools") + 1] == ""


def test_logged_out_cli_becomes_a_login_error_for_the_judge(monkeypatch) -> None:
    monkeypatch.setenv("TEARDOWN_AI", "claude")
    monkeypatch.delenv(judge.LLM_COMMAND_ENV, raising=False)
    monkeypatch.setattr(
        ai.subprocess, "run", lambda argv, **_k: subprocess.CompletedProcess(argv, 1, "", "Please log in")
    )
    with pytest.raises(judge.JudgeLoginError, match="登录"):
        judge.cli_judge("x")


def test_cookies_inside_the_repo_are_refused(tmp_path: Path) -> None:
    inside = pipeline.REPO_ROOT / "cookies.json"
    with pytest.raises(pipeline.PipelineError, match="仓库外"):
        pipeline._download_one("https://www.douyin.com/video/1", inside, tmp_path)


def test_bare_link_means_run(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}

    def fake_run_pipeline(urls, **kwargs):
        seen.update(urls=urls, **kwargs)
        return {"status": "ok", "reports": [{"status": "ok", "url": urls[0], "report_markdown": "r.md"}]}

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setenv("TEARDOWN_HOME", str(tmp_path))
    assert cli.main(["https://v.douyin.com/abc/", "--json"]) == 0
    assert seen["urls"] == ["https://v.douyin.com/abc/"]


def test_env_file_never_overrides_real_environment(monkeypatch, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("TEARDOWN_AI=deepseek\nTEARDOWN_X_TEST=from-file\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("TEARDOWN_AI", "claude")
    monkeypatch.delenv("TEARDOWN_X_TEST", raising=False)
    cli.load_env_file(env)
    assert ai.provider() == "claude"
    import os

    assert os.environ["TEARDOWN_X_TEST"] == "from-file"
    monkeypatch.delenv("TEARDOWN_X_TEST")


def test_glossary_ships_inside_the_package() -> None:
    raw = json.loads(pipeline.DEFAULT_GLOSSARY_PATH.read_text(encoding="utf-8"))
    assert raw["replacements"]


def test_transcription_only_never_calls_the_extractor_llm(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    import importlib

    ex = importlib.import_module("content_extractor.extract")
    from content_extractor.config import ExtractorConfig

    def boom(**_kwargs):
        raise AssertionError("LLM step must be skipped")

    written: dict = {}
    result = SimpleNamespace(content_id="1", content_type="video", raw_text="字")
    monkeypatch.setattr(ex, "is_extracted", lambda _d: False)
    monkeypatch.setattr(ex, "load_content_item", lambda _d: SimpleNamespace(content_type="video"))
    monkeypatch.setattr(ex, "get_extractor", lambda _t: SimpleNamespace(extract=lambda _d, _c: result))
    monkeypatch.setattr(ex, "restructure_transcript", boom)
    monkeypatch.setattr(ex, "analyze_content", boom)
    monkeypatch.setattr(ex, "write_extraction_output", lambda *a, **k: written.update(k))
    assert ex.extract_content(tmp_path, ExtractorConfig(llm_enabled=False)) is result
    assert written["analysis_degraded"] is True
