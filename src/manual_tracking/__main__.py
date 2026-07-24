"""python -m manual_tracking ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import default_model_path, export_landmarks_json, process_video
from .live import run_live


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manual-tracking",
        description="MediaPipe hand tracking + vector overlay (manualtracking / AM style)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Track hands and render overlay video")
    run.add_argument("-i", "--input", required=True, help="Input video path")
    run.add_argument("-o", "--output", required=True, help="Output mp4 path")
    run.add_argument("--model", default=None, help="Path to hand_landmarker.task")
    run.add_argument(
        "--style",
        choices=("mirror", "screen", "wire", "fabric", "track", "outline"),
        default="mirror",
        help="Overlay style (default: mirror; fabric/track/outline are legacy aliases)",
    )
    run.add_argument("--no-source", action="store_true", help="Black bg, no source video")
    run.add_argument("--source-dim", type=float, default=0.35, help="Source dim factor 0-1")
    run.add_argument("--smooth", type=float, default=0.55, help="Landmark temporal smooth 0-0.95")
    run.add_argument("--max-frames", type=int, default=None, help="Debug: only first N frames")

    dump = sub.add_parser("dump", help="Export per-frame landmarks JSON")
    dump.add_argument("-i", "--input", required=True)
    dump.add_argument("-o", "--output", required=True)
    dump.add_argument("--model", default=None)

    live = sub.add_parser("live", help="Realtime webcam: YOU get the effect")
    live.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    live.add_argument("--model", default=None, help="Path to hand_landmarker.task")
    live.add_argument(
        "--style",
        choices=("mirror", "screen", "wire", "fabric", "track", "outline"),
        default="mirror",
        help="Overlay style (default: mirror; fabric/track/outline are legacy aliases)",
    )
    live.add_argument("--no-source", action="store_true", help="Black bg, hide camera image")
    live.add_argument("--source-dim", type=float, default=0.55, help="Camera dim 0-1")
    live.add_argument("--smooth", type=float, default=0.45, help="Landmark smooth 0-0.95")
    live.add_argument("--no-mirror", action="store_true", help="Disable mirror")
    live.add_argument("--width", type=int, default=1280, help="Capture width (default 1280)")
    live.add_argument("--height", type=int, default=720, help="Capture height (default 720)")
    live.add_argument(
        "--infer-size",
        type=int,
        default=480,
        help="Max side for MediaPipe inference (default 480, lower=faster)",
    )
    live.add_argument("--record", default=None, help="Optional output path to start recording")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = args.model or str(default_model_path())

    if args.cmd == "run":
        out = process_video(
            args.input,
            args.output,
            model_path=model,
            style=args.style,
            show_source=not args.no_source,
            source_dim=args.source_dim,
            smooth=args.smooth,
            max_frames=args.max_frames,
        )
        print(f"wrote {out}")
        return 0

    if args.cmd == "dump":
        payload = export_landmarks_json(args.input, args.output, model_path=model)
        print(f"wrote {args.output} ({payload['frame_count']} frames)")
        return 0

    if args.cmd == "live":
        run_live(
            camera=args.camera,
            model_path=model,
            style=args.style,
            show_source=not args.no_source,
            source_dim=args.source_dim,
            smooth=args.smooth,
            mirror=not args.no_mirror,
            width=args.width,
            height=args.height,
            infer_size=args.infer_size,
            record=args.record,
        )
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
