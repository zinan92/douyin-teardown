from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from typing import Callable

from . import ai


JudgeFn = Callable[[str], dict]

# The default judge asks the provider picked in TEARDOWN_AI (see ai.py). TEARDOWN_LLM_CMD
# overrides it with any shell command that reads the prompt on stdin and prints the answer.
LLM_COMMAND_ENV = "TEARDOWN_LLM_CMD"


class JudgeError(RuntimeError):
    """The structure judge did not return a usable JSON object."""


class JudgeLoginError(JudgeError):
    """The local LLM CLI is logged out; retrying will not help until someone logs in."""


def parse_json_object(text: str) -> dict:
    """Parse the first JSON object in model output, tolerating code fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise JudgeError("model output contains no JSON object")
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise JudgeError("model output JSON must be an object")
    return value


def cli_judge(prompt: str, *, command: str | None = None, timeout: float = 600.0) -> dict:
    """Ask the AI for the structure judgement and parse its JSON answer."""
    command = command or os.environ.get(LLM_COMMAND_ENV)
    if not command:
        try:
            return parse_json_object(ai.ask(prompt, timeout=timeout))
        except ai.AILoginError as exc:
            raise JudgeLoginError(str(exc)) from exc
        except ai.AIError as exc:
            raise JudgeError(str(exc)) from exc
    try:
        completed = subprocess.run(
            shlex.split(command), input=prompt, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise JudgeError(f"LLM command failed to run: {exc}") from exc
    if completed.returncode != 0:
        detail = f"{completed.stderr}\n{completed.stdout}"
        if re.search(r"authenticat|log ?in|oauth", detail, re.IGNORECASE):
            raise JudgeLoginError("AI 命令行没登录或登录过期：重新登录后再跑")
        detail = (completed.stderr.strip() or completed.stdout.strip())[-300:] or "没有输出"
        if re.search(r"usage limit|rate limit|limit reached|quota", detail, re.IGNORECASE):
            raise JudgeLoginError(f"AI 额度用完或被限流，稍后再跑：{detail[:120]}")
        raise JudgeError(f"LLM command exited with {completed.returncode}: {detail}")
    return parse_json_object(completed.stdout)
