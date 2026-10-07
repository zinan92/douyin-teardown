"""Typer CLI for content-extractor.

Provides ``extract`` (single item) and ``extract-batch`` (directory scan)
commands with Rich progress bars and error summary tables.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from content_extractor.config import ExtractorConfig
from content_extractor.extract import extract_content

app = typer.Typer(
    name="content-extractor",
    help="Turn raw multimedia content into structured text.",
)

_console = Console()
_err_console = Console(stderr=True)


_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m4v"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def _wrap_bare_file(file_path: Path) -> Path:
    """Create a temporary ContentItem directory for a bare media file.

    Returns a directory that looks like a content-downloader output so the
    rest of the extraction pipeline works unchanged.
    """
    import json
    import shutil
    from datetime import datetime, timezone

    stem = file_path.stem
    tmp_dir = file_path.parent / f".extract-tmp-{stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    media_dir = tmp_dir / "media"
    media_dir.mkdir(exist_ok=True)

    dest = media_dir / file_path.name
    if not dest.exists():
        shutil.copy2(file_path, dest)

    suffix = file_path.suffix.lower()
    if suffix in _VIDEO_EXTS:
        content_type = "video"
    elif suffix in _AUDIO_EXTS:
        content_type = "audio"
    else:
        content_type = "video"

    item = {
        "platform": "local",
        "content_id": stem,
        "content_type": content_type,
        "title": stem,
        "description": "",
        "author_id": "local",
        "author_name": "local",
        "publish_time": datetime.now(tz=timezone.utc).isoformat(),
        "source_url": str(file_path.resolve()),
        "media_files": [f"media/{file_path.name}"],
        "downloaded_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    (tmp_dir / "content_item.json").write_text(
        json.dumps(item, ensure_ascii=False, indent=2)
    )

    return tmp_dir


@app.command()
def extract(
    path: Path = typer.Argument(
        ...,
        help="Path to a ContentItem directory or a single video/audio file.",
    ),
    whisper_model: str = typer.Option(
        "turbo",
        "--whisper-model",
        help="Whisper model name for audio transcription.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Reprocess even if already extracted.",
    ),
) -> None:
    """Extract content from a single ContentItem directory or media file."""
    if not path.exists():
        _err_console.print(f"[red]Error:[/red] Path does not exist: {path}")
        raise typer.Exit(code=1)

    # If user passed a bare media file, wrap it into a ContentItem directory
    wrapped = False
    if path.is_file() and path.suffix.lower() in (_VIDEO_EXTS | _AUDIO_EXTS):
        _console.print(
            f"[dim]Detected bare media file. Wrapping as ContentItem...[/dim]"
        )
        path = _wrap_bare_file(path)
        wrapped = True

    if not path.is_dir():
        _err_console.print(
            f"[red]Error:[/red] Expected a directory or media file (.mp4/.mp3/etc), got: {path}"
        )
        raise typer.Exit(code=1)

    config = ExtractorConfig(
        whisper_model=whisper_model,
        force_reprocess=force,
    )

    try:
        result = extract_content(path, config)
    except Exception as exc:  # noqa: BLE001
        _err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    if result is None:
        _console.print(
            "[yellow]Already extracted. Use --force to reprocess.[/yellow]"
        )
        return

    _console.print(f"[green]Extracted:[/green] {result.content_id}")
    _console.print(f"  Type: {result.content_type}")
    _console.print(f"  Words: {result.quality.word_count}")
    _console.print(
        f"  Time: {result.quality.processing_time_seconds:.1f}s"
    )


@app.command("extract-batch")
def extract_batch_cmd(
    path: Path = typer.Argument(
        ...,
        help="Parent directory to scan for ContentItem directories.",
    ),
    whisper_model: str = typer.Option(
        "turbo",
        "--whisper-model",
        help="Whisper model name for audio transcription.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Reprocess even if already extracted.",
    ),
) -> None:
    """Scan a directory and extract all ContentItems with a progress bar."""
    if not path.exists() or not path.is_dir():
        _err_console.print(
            f"[red]Error:[/red] Path does not exist or is not a directory: {path}"
        )
        raise typer.Exit(code=1)

    config = ExtractorConfig(
        whisper_model=whisper_model,
        force_reprocess=force,
    )

    # Scan for content_item.json directories (same logic as extract.py)
    content_dirs = sorted(
        p.parent for p in path.rglob("content_item.json")
    )

    if not content_dirs:
        _console.print("[yellow]No content items found.[/yellow]")
        return

    succeeded = 0
    skipped = 0
    errors: list[tuple[str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=_console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(content_dirs))

        for content_dir in content_dirs:
            progress.update(task, description=f"[cyan]{content_dir.name}[/cyan]")
            try:
                result = extract_content(content_dir, config)
                if result is not None:
                    succeeded += 1
                else:
                    skipped += 1
            except Exception as exc:  # noqa: BLE001
                errors.append((str(content_dir), str(exc)))
            progress.advance(task)

    total = len(content_dirs)
    _console.print(
        f"\nProcessed {total} items: "
        f"{succeeded} succeeded, {skipped} skipped, {len(errors)} failed"
    )

    if errors:
        table = Table(title="Errors", show_header=True)
        table.add_column("Content Directory", style="cyan")
        table.add_column("Error", style="red")
        for dir_path, error_msg in errors:
            table.add_row(dir_path, error_msg)
        _err_console.print(table)
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the ``content-extractor`` console script."""
    app()
