<div align="center">

# douyin-teardown · 抖音爆款拆解

**一条抖音视频，到底爆没爆、为什么爆、哪里让人听散了。**

[![CI](https://github.com/zinan92/douyin-teardown/actions/workflows/ci.yml/badge.svg)](https://github.com/zinan92/douyin-teardown/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

```
in   一条（或几条）抖音视频链接 + 你自己导出的抖音 cookies
out  report.md：点赞是作者中位数的几倍 + 一句话主线 + 按意思切的段落（钩子 / 承诺 / 论点 / 案例 / 跑题 / 收束 / 引导）
     + 为什么爆 / 为什么听散，每一条都附原话和秒数

fail cookies 缺失、过期或放在仓库里 → 直接说清楚，不发请求
fail 抖音弹验证页 / 反作弊          → 立刻停，不绕过
fail AI 没登录 / 没 key / 额度用完   → 说清楚去哪登录，不空转重试
fail AI 输出不合格                  → 带着错误重试，还不行就报错
```

## 它怎么判断

1. **下载**（`content_downloader`）：视频、标题、点赞、收藏、转发、评论、作者。
2. **转文字**（`content_extractor`）：带时间戳的完整文字稿。Apple 芯片的 Mac 用 mlx-whisper（GPU），其他电脑用 faster-whisper（CPU）。
3. **算倍数**：这条的点赞 ÷ 作者最近 60 条（不算置顶）的点赞中位数。作者不到 20 条、或中位数不到 50 赞，就不给倍数——新号的中位数说明不了爆没爆。
4. **结构拆解**：AI 通读全文，写出主线，按**意思**切段，判断每段是不是在为主线服务，再给「为什么爆」「为什么听散」——引用的原话和秒数由程序从文字稿里取，不让 AI 自己编。

> 结构标签和「为什么爆」是基于文字稿和数据的**待验证假设**，不是爆款判定规则。报告里也会写这一句。

## 装

需要：Python 3.11+、ffmpeg、一个能用的 AI（下面四选一）、你自己的抖音 cookies。

```bash
git clone https://github.com/zinan92/douyin-teardown.git
cd douyin-teardown
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[mlx]'        # Apple 芯片的 Mac；其他电脑：pip install -e .
brew install ffmpeg            # Linux：apt install ffmpeg
cp .env.example .env           # 按里面的说明填
teardown doctor                # 缺什么一次说清
```

### 抖音 cookies

登录 douyin.com → 浏览器开发者工具 → Application → Cookies，按 [`cookies.json.example`](cookies.json.example) 的格式存成 JSON：

```bash
mkdir -p ~/.douyin-teardown
# 把导出的 JSON 存成 ~/.douyin-teardown/cookies.json，然后：
chmod 600 ~/.douyin-teardown/cookies.json
```

cookies 只在你本机用，**不能放进这个仓库目录**（程序会拒绝）。拆解会用你的登录去拉作者最近的作品，别一次拆太多条。

### AI（`.env` 里 `TEARDOWN_AI`）

| 填 | 要先做 |
|---|---|
| `codex`（默认） | 装 [Codex CLI](https://github.com/openai/codex)，`codex login` |
| `claude` | 装 [Claude Code](https://docs.anthropic.com/en/docs/claude-code)，在终端登录 |
| `deepseek` | 填 `TEARDOWN_DEEPSEEK_KEY`（任何 OpenAI 兼容接口都行） |
| `anthropic` | 填 `ANTHROPIC_API_KEY` |

命令行版本都是只读、不联网、不给工具，只让它读文字稿、回 JSON。

## 用

```bash
teardown https://v.douyin.com/xxxxxxx/
teardown 链接1 链接2 链接3      # 一条一条串着跑
teardown 链接 --json            # 完整回执
```

报告在 `~/.douyin-teardown/data/reports/<视频 id>/report.md`（同目录还有 `report.json`）。转完文字后视频会删掉，只留文字稿和数据，省空间；同一条再拆不会重新下载。

转写把专有名词听错了（比如 Codex 听成 Kodak），在 [`douyin_teardown/glossary.json`](douyin_teardown/glossary.json) 里加一行，不用改代码。

### 在 Claude Code / Codex 里用

先照上面装好（仓库里要有 `.venv`），再把仓库链接成 skill：

```bash
mkdir -p ~/.claude/skills ~/.codex/skills
ln -s "$PWD" ~/.claude/skills/douyin-teardown   # Claude Code
ln -s "$PWD" ~/.codex/skills/douyin-teardown    # Codex
```

然后直接说「拆一下这条：<链接>」。skill 的说明在 [`SKILL.md`](SKILL.md)。

## 也能单独用的两个工具

这个仓库合并了原来的 [content-downloader](https://github.com/zinan92/content-downloader) 和 [content-extractor](https://github.com/zinan92/content-extractor)（两个都已归档，以后只在这里更新）。装好后它们的命令照样能用：

```bash
content-downloader download <抖音 / 小红书 / 公众号 / X 链接> --cookies ~/.douyin-teardown/cookies.json
content-extractor <下载出来的目录>
```

## 已知

- 来自 content-extractor 的 9 个测试在原仓库里就已经和代码对不上（它的 AI 摘要那部分改写过），这里标成 xfail，原因写在 [`tests/conftest.py`](tests/conftest.py)。拆解只用它的转文字，不受影响。
- 拆解调用转文字时关掉了 content-extractor 自带的 AI 摘要（`ExtractorConfig(llm_enabled=False)`），只用一个 AI；单独跑 `content-extractor` 时照旧会做摘要，它要 `ANTHROPIC_API_KEY`。
- 抖音站内搜索不做：它的搜索接口会触发反作弊。

## 版权

MIT，见 [LICENSE](LICENSE)。抖音签名算法的两个文件来自 Apache-2.0 项目，保留原版权头，出处见 [NOTICE](NOTICE)。
