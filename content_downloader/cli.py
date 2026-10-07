"""CLI entry point for content-downloader."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from content_downloader.manifest import ManifestManager
from content_downloader.output import OutputManager
from content_downloader.router import (
    UnsupportedPlatformError,
    classify_url,
    get_adapter,
    list_supported_platforms,
)


DEFAULT_OUTPUT_DIR = Path("./output")


@click.group()
@click.version_option(package_name="douyin-teardown")
def main() -> None:
    """Unified content downloader — give a URL, get standardized local files."""


# ---------------------------------------------------------------------------
# download command
# ---------------------------------------------------------------------------


@main.command("download")
@click.argument("url")
@click.option(
    "--output-dir",
    "-o",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Root output directory.",
)
@click.option(
    "--limit",
    "-l",
    default=0,
    show_default=True,
    type=int,
    help="Max items for profile downloads. 0 = no limit.",
)
@click.option(
    "--since",
    default=None,
    type=str,
    help="Only download items published after this date (ISO 8601, profile mode).",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Re-download even if item already exists.",
)
@click.option(
    "--cookies",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to cookies JSON file (required for authenticated platforms like Douyin).",
)
def download_cmd(
    url: str,
    output_dir: Path,
    limit: int,
    since: str | None,
    force: bool,
    cookies: Path | None,
) -> None:
    """Download a single content item or an entire profile.

    URL can be a single content URL or a creator profile URL.
    The platform is detected automatically.
    """
    try:
        platform, url_type = classify_url(url)
    except UnsupportedPlatformError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    adapter = get_adapter(url, cookies_path=cookies)
    if platform == "douyin" and not adapter.has_cookies:
        click.echo(
            "Error: Douyin download requires login cookies. "
            "Log in to Douyin, export a cookies JSON file, and pass it with "
            "--cookies <path>.",
            err=True,
        )
        sys.exit(1)
    output_mgr = OutputManager(output_dir)
    manifest_mgr = ManifestManager(output_dir)

    since_dt = None
    if since:
        from datetime import datetime, timezone

        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            click.echo(f"Error: --since value {since!r} is not a valid ISO 8601 date.", err=True)
            sys.exit(1)

    asyncio.run(
        _run_download(
            url=url,
            url_type=url_type,
            platform=platform,
            adapter=adapter,
            output_dir=output_dir,
            output_mgr=output_mgr,
            manifest_mgr=manifest_mgr,
            limit=limit,
            since_dt=since_dt,
            force=force,
        )
    )


async def _run_download(
    url: str,
    url_type: str,
    platform: str,
    adapter,
    output_dir: Path,
    output_mgr: OutputManager,
    manifest_mgr: ManifestManager,
    limit: int,
    since_dt,
    force: bool,
) -> None:
    if url_type == "single":
        await _download_single(
            url=url,
            adapter=adapter,
            output_dir=output_dir,
            output_mgr=output_mgr,
            manifest_mgr=manifest_mgr,
            force=force,
        )
    else:
        await _download_profile(
            profile_url=url,
            adapter=adapter,
            output_dir=output_dir,
            manifest_mgr=manifest_mgr,
            limit=limit,
            since_dt=since_dt,
            force=force,
        )


async def _download_single(
    url: str,
    adapter,
    output_dir: Path,
    output_mgr: OutputManager,
    manifest_mgr: ManifestManager,
    force: bool,
) -> None:
    """Download a single content item."""
    try:
        # Pre-check: call adapter to get content_id for dedup check
        # We try to download first because we need the content_id from the network
        item = await adapter.download_single(url, output_dir)
    except NotImplementedError as exc:
        click.echo(f"Not implemented: {exc}", err=True)
        sys.exit(2)
    except Exception as exc:
        click.echo(f"Download failed: {exc}", err=True)
        sys.exit(1)

    if not force and manifest_mgr.contains(item.content_id):
        click.echo(f"Skipped (already downloaded): {item.content_id}")
        return

    # Write content_item.json (adapter already wrote media + metadata)
    output_mgr.write_content_item(item)
    manifest_mgr.append(item)

    content_dir = output_mgr.content_dir(item)
    click.echo(f"Downloaded: {item.content_id}")
    click.echo(f"  Platform : {item.platform}")
    click.echo(f"  Type     : {item.content_type}")
    click.echo(f"  Location : {content_dir}")


async def _download_profile(
    profile_url: str,
    adapter,
    output_dir: Path,
    manifest_mgr: ManifestManager,
    limit: int,
    since_dt,
    force: bool,
) -> None:
    """Download all items from a profile."""
    try:
        result = await adapter.download_profile(
            profile_url, output_dir, limit=limit, since=since_dt
        )
    except NotImplementedError as exc:
        click.echo(f"Not implemented: {exc}", err=True)
        sys.exit(2)
    except Exception as exc:
        click.echo(f"Profile download failed: {exc}", err=True)
        sys.exit(1)

    output_mgr = OutputManager(output_dir)
    skipped = 0
    appended = 0

    for item in result.items:
        if not force and manifest_mgr.contains(item.content_id):
            skipped += 1
            continue
        output_mgr.write_content_item(item)
        manifest_mgr.append(item)
        appended += 1

    click.echo(f"Profile download complete:")
    click.echo(f"  Downloaded : {appended}")
    click.echo(f"  Skipped    : {skipped} (already in manifest)")
    click.echo(f"  Failed     : {result.failed}")
    click.echo(f"  Total      : {result.total}")

    if result.errors:
        click.echo("Errors:")
        for err in result.errors:
            click.echo(f"  [{err.error_type}] {err.content_id}: {err.message}", err=True)


# ---------------------------------------------------------------------------
# fetch-cookies command
# ---------------------------------------------------------------------------


@main.command("fetch-cookies")
@click.option(
    "--headless",
    is_flag=True,
    default=False,
    help="Run browser in headless mode (not recommended — CAPTCHA may appear).",
)
@click.option(
    "--output",
    "-o",
    default="cookies.json",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output path for the captured cookies JSON file.",
)
@click.option(
    "--url",
    default="https://www.douyin.com",
    show_default=True,
    help="URL to open in the browser.",
)
@click.option(
    "--browser",
    default="chromium",
    show_default=True,
    type=click.Choice(["chromium", "firefox", "webkit"], case_sensitive=False),
    help="Playwright browser engine.",
)
@click.option(
    "--keep-all",
    is_flag=True,
    default=False,
    help="Save all cookies, not just the required Douyin set.",
)
def fetch_cookies_cmd(
    headless: bool,
    output: Path,
    url: str,
    browser: str,
    keep_all: bool,
) -> None:
    """Capture Douyin login cookies via a real browser session.

    Opens a browser window so you can log in manually, then saves the
    captured cookies to OUTPUT (default: cookies.json).  Pass the output
    file to --cookies when running the download command.

    Requires playwright:  pip install playwright && playwright install chromium
    """
    try:
        from content_downloader.tools.cookie_fetcher import capture_cookies
    except ImportError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    try:
        asyncio.run(
            capture_cookies(
                url=url,
                browser_type=browser,
                headless=headless,
                output_path=output,
                keep_all=keep_all,
            )
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


@main.command("list")
@click.option(
    "--output-dir",
    "-o",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Root output directory.",
)
@click.option(
    "--platform",
    "-p",
    default=None,
    type=str,
    help="Filter by platform name.",
)
def list_cmd(output_dir: Path, platform: str | None) -> None:
    """List downloaded content items from the manifest."""
    manifest_mgr = ManifestManager(output_dir)
    records = (
        manifest_mgr.filter_by_platform(platform)
        if platform
        else manifest_mgr.all_records()
    )

    if not records:
        click.echo("No items found.")
        return

    for record in records:
        click.echo(
            f"[{record.get('platform', '?')}] {record.get('content_id', '?')} "
            f"— {record.get('title', '')}"
        )
    click.echo(f"\nTotal: {len(records)} item(s)")


# ---------------------------------------------------------------------------
# platforms command
# ---------------------------------------------------------------------------


@main.command("platforms")
def platforms_cmd() -> None:
    """Show all supported platforms and their URL patterns."""
    click.echo("Supported platforms:\n")
    descriptions = {
        "douyin": "Douyin (抖音) — douyin.com/video/*, v.douyin.com/*, douyin.com/user/*",
        "xhs": "Xiaohongshu (小红书) — xiaohongshu.com/explore/*, xhslink.com/*, xiaohongshu.com/user/profile/*",
        "wechat_oa": "WeChat Official Account (公众号) — mp.weixin.qq.com/s/*",
        "x": "X / Twitter — x.com/*/status/*, twitter.com/*/status/*",
        "fixture": "Fixture (testing) — fixture.test/video/*, fixture.test/image/*, fixture.test/user/*",
    }
    for platform in list_supported_platforms():
        status = "[ready]"
        desc = descriptions.get(platform, platform)
        click.echo(f"  {status:20s} {desc}")
