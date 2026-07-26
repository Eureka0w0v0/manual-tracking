"""End-to-end: video in → hand track → vector overlay → video out."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2

from .paths import ensure_model
from .renderer import VectorOverlayRenderer
from .tracker import HandTracker


def export_landmarks_json(
    video_path: str | Path,
    out_json: str | Path,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Dump per-frame landmarks for AM / After Effects import."""
    model = ensure_model(model_path)
    frames: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}

    with HandTracker(model) as tracker:
        for frame, hands, fps in tracker.iter_video(video_path):
            if not meta:
                h, w = frame.shape[:2]
                meta = {"width": w, "height": h, "fps": fps, "video": str(video_path)}
            entry = {
                "frame": hands.index,
                "hands": [
                    {
                        "handedness": hp.handedness,
                        "score": hp.score,
                        "points": hp.points.tolist(),
                    }
                    for hp in hands.hands
                ],
            }
            frames.append(entry)

    payload = {**meta, "frame_count": len(frames), "frames": frames}
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path | None = None,
    style: str = "mirror",
    show_source: bool = True,
    source_dim: float = 0.35,
    filter_on: bool = True,
    max_frames: int | None = None,
    progress: bool = True,
) -> Path:
    """
    Render overlay video.

    Returns path to written mp4.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = ensure_model(model_path)

    renderer = VectorOverlayRenderer(
        style=style,
        show_source=show_source,
        source_dim=source_dim,
    )

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    fps_out = 30.0

    with HandTracker(model, filter_on=filter_on) as tracker:
        for frame, hands, fps in tracker.iter_video(input_path):
            fps_out = fps
            rendered = renderer.render(frame, hands)
            if writer is None:
                h, w = rendered.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps_out, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open VideoWriter for {output_path}")
            writer.write(rendered)
            frame_count += 1
            if progress and frame_count % 15 == 0:
                print(f"\r  frames: {frame_count}", end="", file=sys.stderr, flush=True)
            if max_frames is not None and frame_count >= max_frames:
                break

    if progress:
        print(f"\r  frames: {frame_count} done", file=sys.stderr)

    if writer is not None:
        writer.release()

    if frame_count == 0:
        raise RuntimeError(f"no frames read from {input_path}")

    # Remux with ffmpeg for broader player compatibility when available
    remuxed = _try_ffmpeg_remux(output_path, fps_out)
    return remuxed or output_path


def _try_ffmpeg_remux(path: Path, fps: float) -> Path | None:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return None
    tmp = path.with_suffix(".h264.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(path)
        return path
    except (subprocess.CalledProcessError, OSError):
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None
