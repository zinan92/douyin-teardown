"""Tests for the CLI commands."""

import json
import pytest
from pathlib import Path
from click.testing import CliRunner

from content_downloader.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestMainGroup:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "download" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_no_args_shows_help(self, runner):
        result = runner.invoke(main, [])
        # Click groups exit with 0 or 2 when called with no subcommand
        assert result.exit_code in (0, 2)


class TestPlatformsCommand:
    def test_lists_all_platforms(self, runner):
        result = runner.invoke(main, ["platforms"])
        assert result.exit_code == 0
        # Check for key terms that appear in the descriptions
        for term in ("Douyin", "Xiaohongshu", "WeChat", "Twitter", "Fixture"):
            assert term in result.output

    def test_shows_fixture_as_ready(self, runner):
        result = runner.invoke(main, ["platforms"])
        assert "[ready]" in result.output


class TestDownloadCommand:
    def test_single_fixture_url_downloads(self, runner, tmp_path):
        result = runner.invoke(
            main,
            ["download", "https://fixture.test/video/abc123", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "abc123" in result.output
        assert (tmp_path / "fixture" / "test-author" / "abc123").is_dir()

    def test_creates_manifest(self, runner, tmp_path):
        runner.invoke(
            main,
            ["download", "https://fixture.test/video/abc123", "--output-dir", str(tmp_path)],
        )
        manifest = tmp_path / "manifest.jsonl"
        assert manifest.exists()
        data = json.loads(manifest.read_text().strip())
        assert data["content_id"] == "abc123"

    def test_dedup_skip_on_second_run(self, runner, tmp_path):
        url = "https://fixture.test/video/dup001"
        runner.invoke(main, ["download", url, "--output-dir", str(tmp_path)])
        result2 = runner.invoke(main, ["download", url, "--output-dir", str(tmp_path)])
        assert result2.exit_code == 0
        assert "Skipped" in result2.output

    def test_force_flag_redownloads(self, runner, tmp_path):
        url = "https://fixture.test/video/force001"
        runner.invoke(main, ["download", url, "--output-dir", str(tmp_path)])
        result2 = runner.invoke(
            main, ["download", url, "--output-dir", str(tmp_path), "--force"]
        )
        assert result2.exit_code == 0
        assert "Downloaded" in result2.output

    def test_profile_url_with_limit(self, runner, tmp_path):
        result = runner.invoke(
            main,
            [
                "download",
                "https://fixture.test/user/test-author",
                "--output-dir",
                str(tmp_path),
                "--limit",
                "3",
            ],
        )
        assert result.exit_code == 0
        assert "Downloaded" in result.output or "Total" in result.output

    def test_unsupported_url_exits_with_error(self, runner, tmp_path):
        result = runner.invoke(
            main,
            ["download", "https://unknown.example.com/video/abc", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_invalid_since_exits_with_error(self, runner, tmp_path):
        result = runner.invoke(
            main,
            [
                "download",
                "https://fixture.test/user/test-author",
                "--output-dir",
                str(tmp_path),
                "--since",
                "not-a-date",
            ],
        )
        assert result.exit_code != 0

    def test_unsupported_platform_exits_gracefully(self, runner, tmp_path):
        result = runner.invoke(
            main,
            ["download", "https://tiktok.com/@user/video/123", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_douyin_without_cookies_shows_login_instructions(
        self, runner, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        result = runner.invoke(
            main,
            [
                "download",
                "https://www.douyin.com/video/7658238400629083402",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        output = result.output.lower()
        assert "cookies" in output
        assert "login" in output or "登录" in result.output


class TestListCommand:
    def test_list_empty(self, runner, tmp_path):
        result = runner.invoke(main, ["list", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No items found" in result.output

    def test_list_after_download(self, runner, tmp_path):
        runner.invoke(
            main,
            ["download", "https://fixture.test/video/vid001", "--output-dir", str(tmp_path)],
        )
        result = runner.invoke(main, ["list", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "vid001" in result.output

    def test_list_platform_filter(self, runner, tmp_path):
        runner.invoke(
            main,
            ["download", "https://fixture.test/video/fv01", "--output-dir", str(tmp_path)],
        )
        result = runner.invoke(
            main, ["list", "--output-dir", str(tmp_path), "--platform", "fixture"]
        )
        assert result.exit_code == 0
        assert "fv01" in result.output

    def test_list_platform_filter_no_match(self, runner, tmp_path):
        runner.invoke(
            main,
            ["download", "https://fixture.test/video/fv01", "--output-dir", str(tmp_path)],
        )
        result = runner.invoke(
            main, ["list", "--output-dir", str(tmp_path), "--platform", "douyin"]
        )
        assert result.exit_code == 0
        assert "No items found" in result.output
