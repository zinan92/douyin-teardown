"""Which AI reads the transcript: a logged-in CLI (codex / claude) or an API key (deepseek / anthropic).

Pick one with TEARDOWN_AI. Every provider only answers text: the CLIs run read-only, with no
tools and no web, from a temp directory so no project instructions get loaded.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import httpx


PROVIDERS = ("codex", "claude", "deepseek", "anthropic")
DEFAULT_PROVIDER = "codex"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 8000

LOGIN_PATTERN = re.compile(r"authenticat|log ?in|oauth|401 Unauthorized|invalid api key|x-api-key", re.IGNORECASE)
LIMIT_PATTERN = re.compile(r"usage limit|rate limit|limit reached|quota|insufficient balance|402", re.IGNORECASE)


class AIError(RuntimeError):
    """The AI call failed; retrying with a fixed prompt may help."""


class AILoginError(AIError):
    """Missing login / key or out of quota; retrying will not help until a person acts."""


def provider() -> str:
    name = (os.environ.get("TEARDOWN_AI") or DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise AILoginError(f"TEARDOWN_AI={name} 不认识，只能是 {' / '.join(PROVIDERS)}")
    return name


def login_hint(name: str | None = None) -> str:
    return {
        "codex": "在终端运行 codex login",
        "claude": "在终端运行 claude 并按提示登录",
        "deepseek": "在 .env 里填 TEARDOWN_DEEPSEEK_KEY",
        "anthropic": "在 .env 里填 ANTHROPIC_API_KEY",
    }[name or provider()]


def _cli_bin(name: str) -> str:
    configured = os.environ.get(f"TEARDOWN_{name.upper()}_BIN", "").strip()
    return configured or shutil.which(name) or name


def _run_cli(name: str, argv: list[str], prompt: str, timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv, input=prompt, capture_output=True, text=True, timeout=timeout, check=False, cwd=tempfile.gettempdir()
        )
    except FileNotFoundError as exc:
        raise AILoginError(f"没找到 {name} 命令行：先安装并登录（{login_hint(name)}），或换 TEARDOWN_AI") from exc
    except subprocess.TimeoutExpired as exc:
        raise AIError(f"{name} 超过 {timeout:.0f} 秒没回话") from exc


def _check_cli(name: str, done: subprocess.CompletedProcess) -> None:
    if done.returncode == 0:
        return
    detail = f"{done.stderr}\n{done.stdout}".strip()
    if LOGIN_PATTERN.search(detail):
        raise AILoginError(f"本机 {name} 没登录或登录过期：{login_hint(name)}")
    if LIMIT_PATTERN.search(detail):
        raise AILoginError(f"本机 {name} 额度用完或被限流，稍后再试：{detail[-120:]}")
    raise AIError(f"{name} 退出码 {done.returncode}：{detail[-300:] or '没有输出'}")


def _codex(prompt: str, timeout: float) -> str:
    with tempfile.TemporaryDirectory(prefix="teardown-codex-") as tmp:
        out_file = Path(tmp) / "answer.txt"
        argv = [_cli_bin("codex"), "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                "-c", "project_doc_max_bytes=0", "-c", 'web_search="disabled"',
                "-c", 'model_reasoning_effort="medium"', "-o", str(out_file)]
        model = os.environ.get("TEARDOWN_CODEX_MODEL", "").strip()
        if model:
            argv += ["-m", model]
        done = _run_cli("codex", [*argv, "-"], prompt, timeout)
        _check_cli("codex", done)
        try:
            answer = out_file.read_text(encoding="utf-8").strip()
        except OSError:
            answer = (done.stdout or "").strip()
    if not answer:
        raise AIError("codex 没有回话")
    return answer


def _claude(prompt: str, timeout: float) -> str:
    argv = [_cli_bin("claude"), "-p", "--output-format", "text", "--tools", "", "--no-session-persistence"]
    model = os.environ.get("TEARDOWN_CLAUDE_MODEL", "").strip()
    if model:
        argv += ["--model", model]
    done = _run_cli("claude", argv, prompt, timeout)
    _check_cli("claude", done)
    answer = (done.stdout or "").strip()
    if not answer:
        raise AIError("claude 没有回话")
    return answer


def _post(name: str, url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.HTTPError as exc:
        raise AIError(f"{name} 连不上：{exc}") from exc
    if response.status_code in (401, 403):
        raise AILoginError(f"{name} 的 key 不对或没权限：{login_hint(name)}")
    if response.status_code == 402 or (response.status_code == 429 and LIMIT_PATTERN.search(response.text)):
        raise AILoginError(f"{name} 余额不足或被限流：{response.text[:120]}")
    if response.status_code >= 400:
        raise AIError(f"{name} 返回 HTTP {response.status_code}：{response.text[:300]}")
    return response.json()


def _deepseek(prompt: str, timeout: float) -> str:
    key = os.environ.get("TEARDOWN_DEEPSEEK_KEY", "").strip()
    if not key:
        raise AILoginError(f"没有 DeepSeek key：{login_hint('deepseek')}")
    model = os.environ.get("TEARDOWN_DEEPSEEK_MODEL", "").strip() or DEEPSEEK_MODEL
    payload: dict = {"model": model, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}
    if "v4" in model.lower():
        payload["thinking"] = {"type": "disabled"}  # V4 thinks by default; this is a fixed-shape JSON task
    data = _post("deepseek", os.environ.get("TEARDOWN_DEEPSEEK_ENDPOINT", "").strip() or DEEPSEEK_ENDPOINT,
                 {"Authorization": f"Bearer {key}"}, payload, timeout)
    choices = data.get("choices") or []
    answer = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    if not answer:
        raise AIError("deepseek 没有回话")
    return answer


def _anthropic(prompt: str, timeout: float) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise AILoginError(f"没有 Anthropic key：{login_hint('anthropic')}")
    model = os.environ.get("TEARDOWN_ANTHROPIC_MODEL", "").strip() or ANTHROPIC_MODEL
    data = _post("anthropic", ANTHROPIC_ENDPOINT,
                 {"x-api-key": key, "anthropic-version": "2023-06-01"},
                 {"model": model, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}, timeout)
    answer = "".join(block.get("text", "") for block in data.get("content") or [] if block.get("type") == "text").strip()
    if not answer:
        raise AIError("anthropic 没有回话")
    return answer


def ask(prompt: str, *, timeout: float = 600.0) -> str:
    """Send one prompt to the configured provider and return its text answer."""
    return {"codex": _codex, "claude": _claude, "deepseek": _deepseek, "anthropic": _anthropic}[provider()](prompt, timeout)
