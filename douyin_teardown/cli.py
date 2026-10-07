"""teardown <抖音链接> → 倍数 + 带时间戳的结构拆解；teardown doctor → 缺什么一次说清。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import sys

from . import ai
from .cookies import CookieFileError, load_cookie_file
from .pipeline import DEFAULT_GLOSSARY_PATH, REPO_ROOT, run_pipeline


HOME = Path(os.environ.get("TEARDOWN_HOME", "~/.douyin-teardown")).expanduser()


def load_env_file(path: Path) -> None:
    """KEY=VALUE lines from .env; real environment variables win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def cookie_path() -> Path:
    return Path(os.environ.get("TEARDOWN_COOKIES") or HOME / "cookies.json").expanduser()


def data_dir() -> Path:
    return Path(os.environ.get("TEARDOWN_DATA_DIR") or HOME / "data").expanduser()


def run_command(args: argparse.Namespace) -> int:
    receipt = run_pipeline(
        args.urls,
        cookie_path=args.cookies or cookie_path(),
        data_dir=args.data_dir or data_dir(),
        whisper_model=args.whisper_model,
        glossary_path=args.glossary,
    )
    if args.json:
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    else:
        for item in receipt["reports"]:
            if item["status"] == "ok":
                print(f"✓ {item['url']}\n  报告：{item['report_markdown']}")
            else:
                print(f"✗ {item['url']}\n  {item['error']}")
    return 0 if receipt["status"] == "ok" else 1


def _check(ok: bool, label: str, fix: str = "") -> bool:
    print(f"{'✓' if ok else '✗'} {label}" + ("" if ok or not fix else f"\n    → {fix}"))
    return ok


def doctor_command(_args: argparse.Namespace) -> int:
    results = [_check(sys.version_info >= (3, 11), f"Python {platform.python_version()}", "需要 Python 3.11 或更新")]
    results.append(_check(bool(shutil.which("ffmpeg") and shutil.which("ffprobe")), "ffmpeg / ffprobe", "brew install ffmpeg（Linux：apt install ffmpeg）"))

    mlx = importlib.util.find_spec("mlx_whisper") is not None
    faster = importlib.util.find_spec("faster_whisper") is not None
    apple = sys.platform == "darwin" and platform.machine() == "arm64"
    engine = "mlx-whisper（Apple 芯片 GPU）" if mlx and apple else "faster-whisper（CPU）" if faster else ""
    results.append(_check(bool(engine), f"转文字：{engine or '没装'}", "pip install -e .（Apple 芯片的 Mac 加 [mlx] 更快）"))

    path = cookie_path()
    try:
        inside = path.resolve().is_relative_to(REPO_ROOT)
        load_cookie_file(path)
        cookie_ok, cookie_fix = not inside, "cookies 挪到仓库外面，比如 ~/.douyin-teardown/cookies.json"
    except CookieFileError as exc:
        cookie_ok, cookie_fix = False, f"{exc}；格式见 cookies.json.example，导出后 chmod 600"
    results.append(_check(cookie_ok, f"抖音 cookies：{path}", cookie_fix))
    if path.is_file() and stat.S_IMODE(path.stat().st_mode) & 0o077:
        print(f"    → chmod 600 {path}")

    name = (os.environ.get("TEARDOWN_AI") or ai.DEFAULT_PROVIDER).strip().lower()
    if name in ("codex", "claude"):
        ai_ok = bool(shutil.which(os.environ.get(f"TEARDOWN_{name.upper()}_BIN") or name))
    elif name == "deepseek":
        ai_ok = bool(os.environ.get("TEARDOWN_DEEPSEEK_KEY"))
    elif name == "anthropic":
        ai_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
    else:
        ai_ok = False
    fix = ai.login_hint(name) if name in ai.PROVIDERS else f"TEARDOWN_AI 只能是 {' / '.join(ai.PROVIDERS)}"
    results.append(_check(ai_ok, f"AI：{name}", fix))

    print("\n全部就绪，可以跑：teardown <抖音链接>" if all(results) else "\n先把 ✗ 的几项补上再跑。")
    return 0 if all(results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="teardown", description="一条抖音视频为什么爆：倍数 + 带时间戳的结构拆解")
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run", help="拆一条或几条抖音链接（直接写 teardown <链接> 也行）")
    run.add_argument("urls", nargs="+", metavar="链接")
    run.add_argument("--cookies", type=Path, default=None, help="默认 ~/.douyin-teardown/cookies.json")
    run.add_argument("--data-dir", type=Path, default=None, help="默认 ~/.douyin-teardown/data")
    run.add_argument("--whisper-model", default="turbo")
    run.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY_PATH, help="转写纠错词表")
    run.add_argument("--json", action="store_true", help="输出完整回执 JSON")
    run.set_defaults(func=run_command)
    doctor = commands.add_parser("doctor", help="检查 Python、ffmpeg、转文字、cookies、AI 是否就绪")
    doctor.set_defaults(func=doctor_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file(REPO_ROOT / ".env")
    load_env_file(HOME / ".env")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("run", "doctor", "-h", "--help"):
        argv.insert(0, "run")
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        build_parser().print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
