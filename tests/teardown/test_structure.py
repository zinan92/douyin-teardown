from __future__ import annotations

from pathlib import Path

import pytest

from douyin_teardown.judge import JudgeError, parse_json_object
from douyin_teardown.structure import (
    ReportError,
    build_report,
    canonical_source_url,
    render_markdown,
)


SRC = Path(__file__).resolve().parents[2] / "douyin_teardown"


def _item(**overrides) -> dict:
    item = {
        "platform": "douyin",
        "content_id": "123",
        "title": "测试视频",
        "author_name": "作者",
        "source_url": "https://www.iesdouyin.com/share/video/123/?did=abc&iid=def&u_code=xyz",
        "likes": 2000,
        "comments": 50,
        "shares": 300,
        "collects": 1500,
        "views": 100000,
    }
    item.update(overrides)
    return item


TRANSCRIPT = {
    "language": "zh",
    "segments": [
        {"text": "主张一句话。", "start": 0.0, "end": 10.0},
        {"text": "论证第一部分。", "start": 10.0, "end": 60.0},
        {"text": "我顺便讲讲别的事情，还有 Kodak 这个工具。", "start": 60.0, "end": 120.0},
        {"text": "回到主张。", "start": 120.0, "end": 150.0},
    ],
}


def _judgement(**overrides) -> dict:
    value = {
        "thesis": {"text": "主线", "evidence_start": 12},
        "segments": [
            {"start": 0, "end": 10, "label": "钩子", "summary": "亮出主张", "serves_thesis": True, "reason": "直接给结论"},
            {"start": 10, "end": 60, "label": "论点", "summary": "论证", "serves_thesis": True, "reason": "支撑主线"},
            {"start": 60, "end": 120, "label": "跑题", "summary": "别的事", "serves_thesis": False, "reason": "与主线无关"},
            {"start": 120, "end": 150, "label": "收束", "summary": "回扣", "serves_thesis": True, "reason": "回到主张"},
        ],
        "why_boom": [{"text": "收藏/赞 75.0%，说明观众想留存", "evidence_start": 11}],
        "why_scatter": [{"text": "60 秒的插叙打断论证", "evidence_start": 65}],
        "opening": None,
    }
    value.update(overrides)
    return value


def test_facts_are_computed_from_the_content_item_and_baseline() -> None:
    report = build_report(
        _item(),
        TRANSCRIPT,
        judge_fn=lambda _prompt: _judgement(),
        baseline={"median_likes": 500, "post_count": 40},
    )
    facts = report["facts"]
    assert facts["likes"] == 2000
    assert facts["collect_per_like"] == 0.75
    assert facts["share_per_like"] == 0.15
    assert facts["multiple_of_median"] == 4.0
    markdown = render_markdown(report)
    assert "| 点赞是账号中位数的倍数 | 4.0× |" in markdown
    assert "| 收藏/赞 | 75.0% |" in markdown


def test_drift_comes_from_the_judgement_and_is_measured_in_code() -> None:
    report = build_report(_item(), TRANSCRIPT, judge_fn=lambda _prompt: _judgement())
    drift = [segment for segment in report["segments"] if segment["drift"]]
    assert [segment["start"] for segment in drift] == [60.0]
    assert drift[0]["reason"] == "与主线无关"
    assert report["drift"]["seconds"] == 60.0
    assert any("60 秒不服务主线" in item["text"] for item in report["why_scatter"])


def test_evidence_quotes_real_transcript_lines_not_model_text() -> None:
    report = build_report(_item(), TRANSCRIPT, judge_fn=lambda _prompt: _judgement())
    assert report["thesis"]["evidence"][0]["quote"] == "论证第一部分。"
    assert report["why_scatter"][0]["evidence"][0]["start"] == 60.0


def test_glossary_fixes_transcription_terms_without_code_changes() -> None:
    report = build_report(
        _item(),
        TRANSCRIPT,
        judge_fn=lambda _prompt: _judgement(),
        glossary={"Kodak": "Codex"},
    )
    assert "Codex 这个工具" in report["segments"][2]["text"]
    assert "Kodak" not in render_markdown(report)


def test_opening_window_only_for_videos_with_creator_metrics() -> None:
    prompts: list[str] = []

    def judge(prompt: str) -> dict:
        prompts.append(prompt)
        return _judgement(opening={"text": "前 25 秒没有说出主张"})

    own = build_report(_item(), TRANSCRIPT, judge_fn=judge, creator_metrics={"avg_view_second": 25.4, "bounce_rate_2s": 0.38})
    assert own["opening"]["seconds"] == 25.4
    assert own["opening"]["transcript"] == "主张一句话。 论证第一部分。"
    assert "观众平均只看 25 秒" in prompts[0]
    assert "后台 2 秒跳出率：38.0%" in prompts[0]
    assert "观众平均看到的前 25 秒" in render_markdown(own)

    benchmark = build_report(_item(), TRANSCRIPT, judge_fn=lambda _prompt: _judgement())
    assert benchmark["opening"] is None
    assert "观众平均看到的前" not in render_markdown(benchmark)


def test_invalid_judgement_is_retried_with_the_validation_error() -> None:
    prompts: list[str] = []
    answers = [
        _judgement(why_boom=[{"text": "内容优质", "evidence_start": 0}]),
        _judgement(),
    ]

    def judge(prompt: str) -> dict:
        prompts.append(prompt)
        return answers[len(prompts) - 1]

    report = build_report(_item(), TRANSCRIPT, judge_fn=judge)
    assert len(prompts) == 2
    assert "没有任何一条引用数据里的数字" in prompts[1]
    assert report["why_boom"][0]["text"].startswith("收藏/赞")


@pytest.mark.parametrize("baseline", [{"median_likes": 4, "post_count": 40}, {"median_likes": 500, "post_count": 11}])
def test_multiple_is_withheld_when_the_account_baseline_is_too_small(baseline: dict) -> None:
    report = build_report(_item(), TRANSCRIPT, judge_fn=lambda _prompt: _judgement(), baseline=baseline)
    assert "multiple_of_median" not in report["facts"]
    assert report["facts"]["baseline_too_small"] is True


def test_scatter_reasons_may_be_structural_without_numbers() -> None:
    report = build_report(
        _item(),
        TRANSCRIPT,
        judge_fn=lambda _prompt: _judgement(why_scatter=[{"text": "三条路径之间缺少清晰的过渡", "evidence_start": 60}]),
    )
    assert report["why_scatter"][0]["text"] == "三条路径之间缺少清晰的过渡"


def test_judgement_that_never_validates_raises() -> None:
    bad = _judgement(segments=[{"start": 0, "end": 150, "label": "钩子", "summary": "x", "serves_thesis": True, "reason": "x"}])
    with pytest.raises(ReportError):
        build_report(_item(), TRANSCRIPT, judge_fn=lambda _prompt: bad)


def test_conflicting_drift_signals_are_treated_as_drift() -> None:
    judgement = _judgement()
    judgement["segments"][1]["serves_thesis"] = False
    judgement["segments"][2]["serves_thesis"] = True
    report = build_report(_item(), TRANSCRIPT, judge_fn=lambda _prompt: judgement)
    assert [s["label"] for s in report["segments"]][1:3] == ["跑题", "跑题"]
    assert [s["drift"] for s in report["segments"]] == [False, True, True, False]


def test_source_url_drops_tracking_parameters() -> None:
    assert canonical_source_url(_item()) == "https://www.douyin.com/video/123"
    report = build_report(_item(), TRANSCRIPT, judge_fn=lambda _prompt: _judgement())
    assert "did=" not in render_markdown(report)


def test_cli_judge_explains_expired_login() -> None:
    from douyin_teardown.judge import cli_judge

    with pytest.raises(JudgeError, match="重新登录"):
        cli_judge("x", command="sh -c 'echo \"Failed to authenticate: OAuth session expired\"; exit 1'")


def test_cli_judge_surfaces_stdout_and_limits() -> None:
    from douyin_teardown.judge import cli_judge

    with pytest.raises(JudgeError, match="something broke"):
        cli_judge("x", command="sh -c 'echo something broke; exit 1'")
    with pytest.raises(JudgeError, match="额度"):
        cli_judge("x", command="sh -c 'echo Claude usage limit reached; exit 1'")


def test_expired_login_fails_fast_without_retries() -> None:
    from douyin_teardown.judge import JudgeLoginError

    calls = []

    def judge(_prompt):
        calls.append(1)
        raise JudgeLoginError("本机 Claude 命令行登录已过期")

    with pytest.raises(ReportError, match="登录已过期"):
        build_report(_item(), TRANSCRIPT, judge_fn=judge)
    assert len(calls) == 1


def test_parse_json_object_tolerates_fences_and_rejects_prose() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(JudgeError):
        parse_json_object("没有 JSON")


@pytest.mark.parametrize(
    "fragment",
    ["AI越强", "马斯克", "时薪300", "三大神级", "必装", "同花顺", "Codex Harness", "7658238400629083402", "7645203889066724646", "7683154176955731234"],
)
def test_source_code_contains_no_sample_specific_strings(fragment: str) -> None:
    for path in SRC.rglob("*.py"):
        assert fragment not in path.read_text(encoding="utf-8"), f"{fragment!r} found in {path.name}"
