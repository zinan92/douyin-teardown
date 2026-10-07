"""Integration tests — full end-to-end download flow using the fixture adapter."""

import json
import pytest
from pathlib import Path
from click.testing import CliRunner

from content_downloader.cli import main
from content_downloader.manifest import ManifestManager
from content_downloader.models import ContentItem


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestScenario1_SingleUrl:
    """Single URL → download → verify directory structure + manifest."""

    def test_creates_full_directory_structure(self, runner, tmp_path):
        result = runner.invoke(
            main,
            ["download", "https://fixture.test/video/sc1-vid", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0

        content_dir = tmp_path / "fixture" / "test-author" / "sc1-vid"
        assert content_dir.is_dir(), "Content directory not created"
        assert (content_dir / "media").is_dir(), "media/ subdirectory not created"
        assert (content_dir / "media" / "video.mp4").exists(), "video.mp4 not created"
        assert (content_dir / "media" / "cover.jpg").exists(), "cover.jpg not created"
        assert (content_dir / "metadata.json").exists(), "metadata.json not created"
        assert (content_dir / "content_item.json").exists(), "content_item.json not created"

    def test_manifest_has_one_line(self, runner, tmp_path):
        runner.invoke(
            main,
            ["download", "https://fixture.test/video/sc1-vid", "--output-dir", str(tmp_path)],
        )
        manifest = tmp_path / "manifest.jsonl"
        assert manifest.exists()
        lines = [l for l in manifest.read_text().strip().split("\n") if l]
        assert len(lines) == 1

    def test_content_item_json_schema_valid(self, runner, tmp_path):
        runner.invoke(
            main,
            ["download", "https://fixture.test/video/sc1-vid", "--output-dir", str(tmp_path)],
        )
        item_path = tmp_path / "fixture" / "test-author" / "sc1-vid" / "content_item.json"
        data = json.loads(item_path.read_text())

        # Validate against Pydantic model
        item = ContentItem(**data)
        assert item.platform == "fixture"
        assert item.content_id == "sc1-vid"
        assert item.content_type == "video"
        assert item.author_id == "test-author"
        assert "media/video.mp4" in item.media_files
        assert item.metadata_file == "metadata.json"

    def test_metadata_json_is_valid(self, runner, tmp_path):
        runner.invoke(
            main,
            ["download", "https://fixture.test/video/sc1-vid", "--output-dir", str(tmp_path)],
        )
        metadata_path = tmp_path / "fixture" / "test-author" / "sc1-vid" / "metadata.json"
        data = json.loads(metadata_path.read_text())
        assert data["platform"] == "fixture"
        assert data["id"] == "sc1-vid"

    def test_manifest_content_id_correct(self, runner, tmp_path):
        runner.invoke(
            main,
            ["download", "https://fixture.test/video/sc1-vid", "--output-dir", str(tmp_path)],
        )
        manifest = tmp_path / "manifest.jsonl"
        record = json.loads(manifest.read_text().strip())
        assert record["content_id"] == "sc1-vid"
        assert record["platform"] == "fixture"


class TestScenario2_ProfileWithLimit:
    """Profile URL with --limit → download N items → verify count + manifest."""

    def test_downloads_exactly_n_items(self, runner, tmp_path):
        result = runner.invoke(
            main,
            [
                "download",
                "https://fixture.test/user/profile-user",
                "--output-dir",
                str(tmp_path),
                "--limit",
                "3",
            ],
        )
        assert result.exit_code == 0

        mgr = ManifestManager(tmp_path)
        records = mgr.all_records()
        assert len(records) == 3

    def test_each_item_has_separate_directory(self, runner, tmp_path):
        runner.invoke(
            main,
            [
                "download",
                "https://fixture.test/user/profile-user",
                "--output-dir",
                str(tmp_path),
                "--limit",
                "3",
            ],
        )
        mgr = ManifestManager(tmp_path)
        for record in mgr.all_records():
            content_dir = (
                tmp_path / "fixture" / record["author_id"] / record["content_id"]
            )
            assert content_dir.is_dir()

    def test_manifest_has_n_lines(self, runner, tmp_path):
        runner.invoke(
            main,
            [
                "download",
                "https://fixture.test/user/profile-user",
                "--output-dir",
                str(tmp_path),
                "--limit",
                "3",
            ],
        )
        manifest = tmp_path / "manifest.jsonl"
        lines = [l for l in manifest.read_text().strip().split("\n") if l]
        assert len(lines) == 3


class TestScenario3_Dedup:
    """Duplicate URL → skip + report."""

    def test_second_download_is_skipped(self, runner, tmp_path):
        url = "https://fixture.test/video/dedup-001"
        runner.invoke(main, ["download", url, "--output-dir", str(tmp_path)])
        result2 = runner.invoke(main, ["download", url, "--output-dir", str(tmp_path)])
        assert result2.exit_code == 0
        assert "Skipped" in result2.output

    def test_manifest_only_has_one_entry_after_duplicate(self, runner, tmp_path):
        url = "https://fixture.test/video/dedup-002"
        runner.invoke(main, ["download", url, "--output-dir", str(tmp_path)])
        runner.invoke(main, ["download", url, "--output-dir", str(tmp_path)])
        manifest = tmp_path / "manifest.jsonl"
        lines = [l for l in manifest.read_text().strip().split("\n") if l]
        assert len(lines) == 1


class TestScenario4_UnsupportedUrl:
    """Unsupported URL → clear error message."""

    def test_exits_with_nonzero_code(self, runner, tmp_path):
        result = runner.invoke(
            main,
            ["download", "https://tiktok.com/video/abc", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_error_message_mentions_supported_platforms(self, runner, tmp_path):
        result = runner.invoke(
            main,
            ["download", "https://tiktok.com/video/abc", "--output-dir", str(tmp_path)],
        )
        combined = (result.output or "") + (result.stderr if result.stderr else "")
        # Error goes to stderr via click.echo err=True, captured together
        assert any(
            p in combined for p in ("douyin", "fixture", "Supported")
        ), f"Expected platform list in output, got: {combined!r}"

    def test_no_manifest_created_on_failure(self, runner, tmp_path):
        runner.invoke(
            main,
            ["download", "https://tiktok.com/video/abc", "--output-dir", str(tmp_path)],
        )
        assert not (tmp_path / "manifest.jsonl").exists()


class TestScenario5_ForceRedownload:
    """--force flag → re-download even if exists."""

    def test_force_creates_new_content_item(self, runner, tmp_path):
        url = "https://fixture.test/video/force-test"
        runner.invoke(main, ["download", url, "--output-dir", str(tmp_path)])
        result2 = runner.invoke(
            main, ["download", url, "--output-dir", str(tmp_path), "--force"]
        )
        assert result2.exit_code == 0
        assert "Downloaded" in result2.output

    def test_force_appends_to_manifest(self, runner, tmp_path):
        """Force re-download appends another manifest line (current behavior).

        Note: Dedup check is bypassed by --force, so the manifest will have 2 lines.
        This is intentional — the force flag means "record this download regardless."
        """
        url = "https://fixture.test/video/force-manifest"
        runner.invoke(main, ["download", url, "--output-dir", str(tmp_path)])
        runner.invoke(main, ["download", url, "--output-dir", str(tmp_path), "--force"])
        manifest = tmp_path / "manifest.jsonl"
        lines = [l for l in manifest.read_text().strip().split("\n") if l]
        # Force bypasses dedup → 2 entries
        assert len(lines) == 2
