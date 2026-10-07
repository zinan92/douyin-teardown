from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .judge import JudgeError, JudgeFn, JudgeLoginError, cli_judge


MIN_BASELINE_POSTS = 20
MIN_BASELINE_MEDIAN = 50
LABELS = ("钩子", "承诺", "论点", "案例", "跑题", "收束", "引导")
OPENING_FALLBACK_SECONDS = 15.0
HYPOTHESIS_NOTE = "结构标签和“为什么爆”是模型基于转写与数据给出的待验证假设，不是爆款判定规则。"


class ReportError(RuntimeError):
    """The structure judgement could not be turned into a grounded report."""


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------


def _timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def load_glossary(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    replacements = raw.get("replacements", {}) if isinstance(raw, dict) else {}
    return {str(k): str(v) for k, v in replacements.items() if k and not str(k).startswith("_")}


def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    for wrong in sorted(glossary, key=len, reverse=True):
        text = text.replace(wrong, glossary[wrong])
    return text


def transcript_lines(transcript: dict[str, Any], glossary: dict[str, str] | None = None) -> list[dict[str, Any]]:
    lines = []
    for line in transcript.get("segments", []):
        if not isinstance(line, dict):
            continue
        text = apply_glossary(str(line.get("text") or "").strip(), glossary or {})
        if not text:
            continue
        start = float(line.get("start") or 0)
        lines.append({"text": text, "start": start, "end": float(line.get("end") or start)})
    lines.sort(key=lambda line: line["start"])
    return lines


def _line_at(lines: list[dict[str, Any]], second: float) -> dict[str, Any]:
    """The transcript line covering (or nearest before) a timestamp."""
    chosen = lines[0]
    for line in lines:
        if line["start"] <= second + 0.5:
            chosen = line
        else:
            break
    return chosen


def _evidence(lines: list[dict[str, Any]], second: Any) -> list[dict[str, Any]]:
    try:
        value = float(second)
    except (TypeError, ValueError):
        return []
    line = _line_at(lines, value)
    return [{"start": line["start"], "end": line["end"], "quote": line["text"][:180]}]


def _text_between(lines: list[dict[str, Any]], start: float, end: float) -> str:
    return " ".join(line["text"] for line in lines if start - 0.5 <= line["start"] < end)


def canonical_source_url(content_item: dict[str, Any]) -> str | None:
    content_id = str(content_item.get("content_id") or "").strip()
    if content_item.get("platform") == "douyin" and content_id.isdigit():
        return f"https://www.douyin.com/video/{content_id}"
    url = content_item.get("source_url")
    return str(url).split("?", 1)[0] if url else None


# ---------------------------------------------------------------------------
# Numbers the conclusions must stand on (computed here, never by the model)
# ---------------------------------------------------------------------------


def _int(value: Any) -> int | None:
    if value in (None, "", "-"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_facts(
    content_item: dict[str, Any],
    baseline: dict[str, Any] | None,
    creator_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    likes = _int(content_item.get("likes"))
    # Douyin hides play counts on other creators' videos and reports 0.
    views = _int(content_item.get("views")) or None
    facts: dict[str, Any] = {
        "likes": likes,
        "views": views,
        "comments": _int(content_item.get("comments")),
        "shares": _int(content_item.get("shares")),
        "collects": _int(content_item.get("collects")),
    }
    if likes:
        for key, name in (("collects", "collect_per_like"), ("shares", "share_per_like"), ("comments", "comment_per_like")):
            if facts[key] is not None:
                facts[name] = round(facts[key] / likes, 3)
    if likes is not None and facts["views"]:
        facts["like_per_view"] = round(likes / facts["views"], 4)
    if baseline and baseline.get("median_likes"):
        facts["account_median_likes"] = baseline["median_likes"]
        facts["account_post_count"] = baseline.get("post_count")
        post_count = baseline.get("post_count")
        too_small = (post_count is not None and post_count < MIN_BASELINE_POSTS) or baseline["median_likes"] < MIN_BASELINE_MEDIAN
        if too_small:
            # A new or dormant account's median says nothing about breakout; a 1000× multiple would mislead.
            facts["baseline_too_small"] = True
        elif likes is not None:
            facts["multiple_of_median"] = round(likes / float(baseline["median_likes"]), 1)
    if creator_metrics:
        for key in (
            "view_count",
            "completion_rate",
            "completion_rate_5s",
            "bounce_rate_2s",
            "avg_view_second",
            "cover_click_rate",
            "homepage_visit_count",
            "fan_increment",
        ):
            if creator_metrics.get(key) is not None:
                facts[f"creator_{key}"] = creator_metrics[key]
    return facts


FACT_LABELS = {
    "likes": "点赞",
    "views": "播放",
    "comments": "评论",
    "shares": "转发",
    "collects": "收藏",
    "collect_per_like": "收藏/赞",
    "share_per_like": "转发/赞",
    "comment_per_like": "评论/赞",
    "like_per_view": "赞/播放",
    "account_median_likes": "该账号近期点赞中位数",
    "account_post_count": "中位数样本条数",
    "multiple_of_median": "点赞是账号中位数的倍数",
    "creator_view_count": "后台播放",
    "creator_completion_rate": "后台完播率",
    "creator_completion_rate_5s": "后台 5 秒完播率",
    "creator_bounce_rate_2s": "后台 2 秒跳出率",
    "creator_avg_view_second": "后台平均观看秒数",
    "creator_cover_click_rate": "后台封面点击率",
    "creator_homepage_visit_count": "后台主页访问",
    "creator_fan_increment": "后台涨粉",
}

_RATE_KEYS = {"collect_per_like", "share_per_like", "comment_per_like", "like_per_view"}
_PERCENT_KEYS = {"creator_completion_rate", "creator_completion_rate_5s", "creator_bounce_rate_2s", "creator_cover_click_rate"}


def format_fact(key: str, value: Any) -> str:
    if key in _RATE_KEYS or key in _PERCENT_KEYS:
        return f"{float(value) * 100:.1f}%"
    if key == "creator_avg_view_second":
        return f"{float(value):.0f} 秒"
    if key == "multiple_of_median":
        return f"{value}×"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, int) else str(value)


# ---------------------------------------------------------------------------
# Judgement prompt and validation
# ---------------------------------------------------------------------------


def build_prompt(
    title: str,
    lines: list[dict[str, Any]],
    facts: dict[str, Any],
    opening_seconds: float | None,
    error: str | None = None,
) -> str:
    fact_block = "\n".join(
        f"- {FACT_LABELS[key]}：{format_fact(key, value)}" for key, value in facts.items() if key in FACT_LABELS and value is not None
    )
    transcript_block = "\n".join(f"[{line['start']:.1f}] {line['text']}" for line in lines)
    duration = lines[-1]["end"] if lines else 0
    opening_rule = (
        f"- 这是创作者本人的视频，后台显示观众平均只看 {opening_seconds:.0f} 秒。必须填写 opening："
        f"只根据 0–{opening_seconds:.0f} 秒内的原文，判断观众在离开前听到了什么、有没有听到主线，写 2–4 句。"
        if opening_seconds
        else "- opening 填 null。"
    )
    retry = f"\n\n上一次输出没有通过校验：{error}\n请修正后重新输出完整 JSON。" if error else ""
    return f"""你是短视频内容结构分析师。下面是一条抖音视频的标题、数据和带时间戳（秒）的完整转写。
请只输出一个 JSON 对象，不要输出任何其他文字。字符串里需要引号时用「」，不要用英文双引号。

## 标题
{title}

## 数据（已经算好，结论引用数字时只能用这里的数）
{fact_block or "- 无"}

## 转写（[秒] 原文）
{transcript_block}

## 要求
1. thesis：通读**全部**转写后，用一句话写出这条视频真正想让观众带走的主线；不要只看开头或标题。evidence_start 填最能代表主线的那句原文的秒数。
2. segments：按**意思**切段（不是按固定时长），覆盖 0 到 {duration:.0f} 秒，首尾相接、不重叠，一般 5–15 段。
   每段字段：start、end（秒）、label（只能是 {"、".join(LABELS)} 之一）、summary（一句话概括这段在讲什么）、
   serves_thesis（这段是否在为主线服务，true/false）、reason（为什么服务或不服务主线，一句话）。
   label 为“跑题”当且仅当 serves_thesis 为 false。判断标准要严格：假设删掉这一段，主线的论证会不会明显变弱？
   不会变弱（只是间接相关、铺垫过长、绕远路、重复已说过的内容、推荐别人、展示与论点无直接关系的个人素材），就是 serves_thesis=false；
   会变弱（提供了主线必需的论据、例子或结论），才是 true。不要因为话题本身判断，要看这段对主线的必要性。
3. why_boom 与 why_scatter：各 2–4 条，每条都要落到转写里的具体做法。why_boom 至少有一条引用上面“数据”里的数字；why_scatter 能用数据支撑就引用，纯结构问题可以不带数字；evidence_start 填支撑它的原文秒数。
   text 里不要写任何时间点或秒数（报告会根据 evidence_start 自动附上原文和时间）。
   如果该视频相对账号中位数并不突出，why_boom 写“做对了什么”，不要硬说爆。why_scatter 写结构上让人听散、流失的原因。
   不要用空泛套话（如“内容优质”“节奏好”），不要把“5 类生态位”或“四步结构”当成判定规则。
{opening_rule}

## 输出格式
{{"thesis": {{"text": "...", "evidence_start": 0}},
 "segments": [{{"start": 0, "end": 30, "label": "钩子", "summary": "...", "serves_thesis": true, "reason": "..."}}],
 "why_boom": [{{"text": "...", "evidence_start": 0}}],
 "why_scatter": [{{"text": "...", "evidence_start": 0}}],
 "opening": null}}{retry}"""


def _has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def reconcile_drift(raw: dict[str, Any]) -> dict[str, Any]:
    """Either drift signal marks a segment as drift, so the two fields never disagree."""
    for segment in raw.get("segments") or []:
        if isinstance(segment, dict) and (segment.get("label") == "跑题" or segment.get("serves_thesis") is False):
            segment["label"] = "跑题"
            segment["serves_thesis"] = False
        elif isinstance(segment, dict):
            segment["serves_thesis"] = True
    return raw


def validate_judgement(raw: dict[str, Any], duration: float, needs_opening: bool) -> list[str]:
    problems: list[str] = []
    thesis = raw.get("thesis")
    if not isinstance(thesis, dict) or not str(thesis.get("text") or "").strip():
        problems.append("thesis.text 缺失")
    segments = raw.get("segments")
    if not isinstance(segments, list) or not segments:
        problems.append("segments 为空")
    else:
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                problems.append(f"segments[{index}] 不是对象")
                continue
            if segment.get("label") not in LABELS:
                problems.append(f"segments[{index}].label 不在允许列表")
            try:
                start, end = float(segment["start"]), float(segment["end"])
            except (KeyError, TypeError, ValueError):
                problems.append(f"segments[{index}] 缺少数值 start/end")
                continue
            if end <= start:
                problems.append(f"segments[{index}] end 必须大于 start")
            if not str(segment.get("reason") or "").strip():
                problems.append(f"segments[{index}].reason 缺失")
        if duration > 60 and len(segments) < 3:
            problems.append("segments 太少，没有按意思分段")
    for key in ("why_boom", "why_scatter"):
        items = raw.get(key)
        if not isinstance(items, list) or not items:
            problems.append(f"{key} 为空")
            continue
        texts = []
        for index, item in enumerate(items):
            text = str((item or {}).get("text") or "") if isinstance(item, dict) else ""
            if not text.strip():
                problems.append(f"{key}[{index}].text 缺失")
            texts.append(text)
        # Why-it-worked must stand on data; why-it-scattered is often purely structural.
        if key == "why_boom" and texts and not any(_has_number(text) for text in texts):
            problems.append(f"{key} 没有任何一条引用数据里的数字")
    if needs_opening:
        opening = raw.get("opening")
        if not isinstance(opening, (str, dict)) or not str(opening.get("text") if isinstance(opening, dict) else opening).strip():
            problems.append("opening 缺失")
    return problems


def judge_structure(
    title: str,
    lines: list[dict[str, Any]],
    facts: dict[str, Any],
    opening_seconds: float | None,
    judge_fn: JudgeFn,
    attempts: int = 4,
) -> dict[str, Any]:
    duration = lines[-1]["end"] if lines else 0.0
    error: str | None = None
    for _ in range(attempts):
        try:
            raw = judge_fn(build_prompt(title, lines, facts, opening_seconds, error))
        except JudgeLoginError as exc:
            raise ReportError(str(exc)) from exc
        except JudgeError as exc:
            error = str(exc)
            continue
        raw = reconcile_drift(raw)
        problems = validate_judgement(raw, duration, needs_opening=opening_seconds is not None)
        if not problems:
            return raw
        error = "；".join(problems[:8])
    raise ReportError(f"拆解结果连续 {attempts} 次没通过校验，可点重试：{error}")


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    content_item: dict[str, Any],
    transcript: dict[str, Any],
    *,
    judge_fn: JudgeFn = cli_judge,
    baseline: dict[str, Any] | None = None,
    creator_metrics: dict[str, Any] | None = None,
    glossary: dict[str, str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a timestamp-grounded structure report for any video."""
    title = str(content_item.get("title") or content_item.get("description") or "未命名视频").strip()
    lines = transcript_lines(transcript, glossary)
    if not lines:
        raise ReportError("transcript has no usable lines")
    duration = lines[-1]["end"]
    facts = compute_facts(content_item, baseline, creator_metrics)
    opening_seconds = None
    if creator_metrics and creator_metrics.get("avg_view_second"):
        opening_seconds = max(float(creator_metrics["avg_view_second"]), OPENING_FALLBACK_SECONDS)

    raw = judge_structure(title, lines, facts, opening_seconds, judge_fn)

    segments = []
    for index, segment in enumerate(sorted(raw["segments"], key=lambda s: float(s["start"]))):
        start = max(0.0, float(segment["start"]))
        end = min(duration, float(segment["end"]))
        drift = segment["label"] == "跑题"
        segments.append(
            {
                "index": index,
                "label": segment["label"],
                "drift": drift,
                "start": start,
                "end": end,
                "summary": str(segment.get("summary") or "").strip(),
                "reason": str(segment.get("reason") or "").strip(),
                "text": _text_between(lines, start, end),
                "evidence": _evidence(lines, start),
            }
        )

    def conclusions(key: str) -> list[dict[str, Any]]:
        return [
            {"text": str(item["text"]).strip(), "evidence": _evidence(lines, item.get("evidence_start"))}
            for item in raw[key]
        ]

    why_scatter = conclusions("why_scatter")
    drift_seconds = sum(segment["end"] - segment["start"] for segment in segments if segment["drift"])
    drift_share = drift_seconds / duration if duration else 0.0
    first_drift = next((segment for segment in segments if segment["drift"]), None)
    if first_drift:
        why_scatter.append(
            {
                "text": f"模型判定 {drift_seconds:.0f} 秒不服务主线，占全片 {drift_share:.0%}（逐段理由见时间轴）。",
                "evidence": first_drift["evidence"],
            }
        )

    opening = None
    if opening_seconds is not None:
        raw_opening = raw.get("opening")
        opening_text = raw_opening.get("text") if isinstance(raw_opening, dict) else raw_opening
        opening = {
            "seconds": opening_seconds,
            "analysis": str(opening_text).strip(),
            "transcript": _text_between(lines, 0.0, opening_seconds),
            "evidence": _evidence(lines, 0.0),
        }

    thesis = raw["thesis"]
    return {
        "schema_version": 2,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "content_id": str(content_item.get("content_id") or ""),
        "title": title,
        "author": content_item.get("author_name"),
        "source_url": canonical_source_url(content_item),
        "facts": facts,
        "transcript": {"language": transcript.get("language") or "zh", "duration_seconds": duration, "line_count": len(lines)},
        "thesis": {"text": str(thesis["text"]).strip(), "evidence": _evidence(lines, thesis.get("evidence_start"))},
        "opening": opening,
        "segments": segments,
        "drift": {"seconds": round(drift_seconds, 1), "share": round(drift_share, 3)},
        "why_boom": conclusions("why_boom"),
        "why_scatter": why_scatter,
        "hypothesis_note": HYPOTHESIS_NOTE,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a report with transcript excerpts and timestamp evidence."""

    def cite(evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return "（无时间点证据）"
        return "；".join(f"[{_timestamp(float(item['start']))}] {item['quote']}" for item in evidence)

    out = [
        f"# {report['title']}",
        "",
        f"- 作者：{report.get('author') or '未知'}",
        f"- 链接：{report.get('source_url') or '未知'}",
        f"- 生成时间：`{report['generated_at']}`",
        "",
        "## 数据",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
    ]
    for key, value in report["facts"].items():
        if key in FACT_LABELS and value is not None:
            out.append(f"| {FACT_LABELS[key]} | {format_fact(key, value)} |")
    out.extend(["", "## 主线一句话", "", f"> {report['thesis']['text']}", f"> 证据：{cite(report['thesis']['evidence'])}", ""])

    opening = report.get("opening")
    if opening:
        out.extend(
            [
                f"## 观众平均看到的前 {opening['seconds']:.0f} 秒",
                "",
                opening["analysis"],
                "",
                f"> 这段原文：{opening['transcript']}",
                "",
            ]
        )

    out.extend(
        [
            "## 分段时间轴",
            "",
            f"跑题合计 {report['drift']['seconds']:.0f} 秒，占全片 {report['drift']['share']:.0%}。",
            "",
            "| 时间 | 标签 | 服务主线 | 这段在讲什么 | 判断理由 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for segment in report["segments"]:
        cells = [segment["summary"], segment["reason"]]
        cells = [cell.replace("|", "\\|") for cell in cells]
        out.append(
            f"| {_timestamp(segment['start'])}–{_timestamp(segment['end'])} | {segment['label']} | "
            f"{'否' if segment['drift'] else '是'} | {cells[0]} | {cells[1]} |"
        )
    out.extend(["", "## 为什么爆", ""])
    out.extend(f"- {item['text']}（证据：{cite(item['evidence'])}）" for item in report["why_boom"])
    out.extend(["", "## 为什么散", ""])
    out.extend(f"- {item['text']}（证据：{cite(item['evidence'])}）" for item in report["why_scatter"])
    out.extend(["", "## 逐段原文", ""])
    for segment in report["segments"]:
        out.extend([f"### [{_timestamp(segment['start'])}–{_timestamp(segment['end'])}] {segment['label']}", "", segment["text"], ""])
    out.extend(["## 解释边界", "", f"> {report['hypothesis_note']}", ""])
    return "\n".join(out)
