"""Tests for the URL router."""

import pytest
from content_downloader.router import (
    classify_url,
    get_adapter,
    list_supported_platforms,
    UnsupportedPlatformError,
)
from content_downloader.adapters.fixture import FixtureAdapter
from content_downloader.adapters.douyin.adapter import DouyinAdapter
from content_downloader.adapters.stub import StubAdapter


class TestClassifyUrl:
    # Douyin single
    def test_douyin_video(self):
        assert classify_url("https://www.douyin.com/video/12345") == ("douyin", "single")

    def test_douyin_short_link(self):
        assert classify_url("https://v.douyin.com/abcde") == ("douyin", "single")

    # Douyin profile
    def test_douyin_profile(self):
        assert classify_url("https://www.douyin.com/user/somebody") == ("douyin", "profile")

    def test_douyin_profile_no_www(self):
        assert classify_url("https://douyin.com/user/somebody") == ("douyin", "profile")

    # XHS single
    def test_xhs_explore(self):
        assert classify_url("https://www.xiaohongshu.com/explore/abc") == ("xhs", "single")

    def test_xhs_discovery(self):
        assert classify_url("https://www.xiaohongshu.com/discovery/xyz") == ("xhs", "single")

    def test_xhs_short_link(self):
        assert classify_url("https://xhslink.com/abc123") == ("xhs", "single")

    # XHS profile
    def test_xhs_profile(self):
        assert classify_url("https://www.xiaohongshu.com/user/profile/uid123") == ("xhs", "profile")

    # WeChat OA
    def test_wechat_oa_article(self):
        assert classify_url("https://mp.weixin.qq.com/s/abc123") == ("wechat_oa", "single")

    # X / Twitter
    def test_x_status(self):
        assert classify_url("https://x.com/user123/status/999") == ("x", "single")

    def test_twitter_status(self):
        assert classify_url("https://twitter.com/user123/status/999") == ("x", "single")

    # Fixture
    def test_fixture_video(self):
        assert classify_url("https://fixture.test/video/abc123") == ("fixture", "single")

    def test_fixture_image(self):
        assert classify_url("https://fixture.test/image/img01") == ("fixture", "single")

    def test_fixture_profile(self):
        assert classify_url("https://fixture.test/user/alice") == ("fixture", "profile")

    # Unsupported
    def test_unsupported_raises(self):
        with pytest.raises(UnsupportedPlatformError):
            classify_url("https://example.com/some/path")

    def test_unsupported_error_contains_supported_list(self):
        with pytest.raises(UnsupportedPlatformError) as exc_info:
            classify_url("https://unknown.site/page")
        message = str(exc_info.value)
        assert "douyin" in message
        assert "fixture" in message

    def test_empty_url_raises(self):
        with pytest.raises(UnsupportedPlatformError):
            classify_url("")

    def test_non_http_raises(self):
        with pytest.raises(UnsupportedPlatformError):
            classify_url("ftp://fixture.test/video/abc")


class TestGetAdapter:
    def test_fixture_returns_fixture_adapter(self):
        adapter = get_adapter("https://fixture.test/video/abc123")
        assert isinstance(adapter, FixtureAdapter)

    def test_douyin_returns_douyin_adapter(self):
        adapter = get_adapter("https://www.douyin.com/video/12345")
        assert isinstance(adapter, DouyinAdapter)
        assert adapter.platform == "douyin"

    def test_xhs_returns_real_adapter(self):
        adapter = get_adapter("https://www.xiaohongshu.com/explore/abc")
        assert adapter.platform == "xhs"

    def test_unsupported_raises(self):
        with pytest.raises(UnsupportedPlatformError):
            get_adapter("https://tiktok.com/video/abc")


class TestListSupportedPlatforms:
    def test_returns_sorted_list(self):
        platforms = list_supported_platforms()
        assert platforms == sorted(platforms)

    def test_contains_expected_platforms(self):
        platforms = list_supported_platforms()
        for p in ("douyin", "xhs", "wechat_oa", "x", "fixture"):
            assert p in platforms
