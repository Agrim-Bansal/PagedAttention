#!/usr/bin/env python3
"""Concatenate every rendered beat into one MP4 for YouTube.

Reads slides/<Class>.json in talk order, stitches the clips with ffmpeg, and
writes dist/pagedattention.mp4. A silent AAC track is muxed so players and
YouTube see an audio stream. Live-talk pause holds are not included.

Run after ``make render`` (or ``make render-low``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parent.parent
SLIDES_DIR = ROOT / "slides"
DIST = ROOT / "dist"
DEFAULT_OUT = DIST / "pagedattention.mp4"

# Talk order — keep in sync with Makefile SCENE_CLASSES.
SCENE_CLASSES = [
    "S0Title",
    "S1GPU",
    "S2Transformers",
    "S3KVCache",
    "S4Problem",
    "S5PagedAttention",
    "S6OSAndWhyHard",
    "S7Sharing",
    "S8Scheduling",
    "S9Results",
    "S10Ablations",
    "S11Takeaways",
]


def run(
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def fail(message: str) -> None:
    sys.exit(message)


def clip_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_clips() -> list[Path]:
    clips: list[Path] = []
    for cls in SCENE_CLASSES:
        index = SLIDES_DIR / f"{cls}.json"
        if not index.is_file():
            fail(f"Missing {index.relative_to(ROOT)}. Run: make render")
        data = json.loads(index.read_text())
        slides = data.get("slides")
        if not slides:
            fail(f"{index.relative_to(ROOT)} has no slides")
        for i, slide in enumerate(slides):
            raw = slide.get("file")
            if not raw:
                fail(f"{index.name} slide {i} has no file")
            path = clip_path(raw)
            if not path.is_file():
                fail(f"Missing clip {path} (from {index.name} slide {i})")
            clips.append(path)
    return clips


def concat_line(path: Path) -> str:
    posix = path.as_posix().replace("'", r"'\''")
    return f"file '{posix}'"


def ffmpeg_concat(
    list_path: Path, dest: Path, *, reencode: bool
) -> subprocess.CompletedProcess[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    common = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-movflags",
        "+faststart",
    ]
    if reencode:
        cmd = common + [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(dest),
        ]
    else:
        cmd = common + [
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(dest),
        ]
    return run(cmd, check=False)


def probe_duration(path: Path) -> str | None:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
    )
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    try:
        seconds = float(text)
    except ValueError:
        return text or None
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:04.1f}s"


def build(*, dest: Path, reencode: bool) -> None:
    clips = load_clips()
    if not clips:
        fail("No clips found. Run: make render")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="pagedattention-concat-",
        delete=False,
        encoding="utf-8",
    ) as handle:
        list_path = Path(handle.name)
        handle.write("\n".join(concat_line(p) for p in clips) + "\n")

    try:
        result = ffmpeg_concat(list_path, dest, reencode=reencode)
        if result.returncode != 0 and not reencode:
            print("Stream copy failed; re-encoding with libx264")
            result = ffmpeg_concat(list_path, dest, reencode=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            fail(detail or f"ffmpeg failed with exit {result.returncode}")
    finally:
        list_path.unlink(missing_ok=True)

    rel = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
    duration = probe_duration(dest)
    extra = f", {duration}" if duration else ""
    print(f"Wrote {rel} ({len(clips)} clips{extra})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output MP4 (default: {DEFAULT_OUT.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode H.264 instead of stream-copying the clips",
    )
    args = parser.parse_args()
    dest = args.output
    if not dest.is_absolute():
        dest = ROOT / dest
    if which("ffmpeg") is None:
        fail("Missing ffmpeg. macOS: brew install ffmpeg")
    build(dest=dest, reencode=args.reencode)


if __name__ == "__main__":
    main()
