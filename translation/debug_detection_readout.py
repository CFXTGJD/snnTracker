"""Debug the task-level bbox readout layer with synthetic activity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translation.detection_readout import (  # noqa: E402
    BBoxDetection,
    evaluate_iou_matches,
    readout_bboxes,
    save_detections_json,
    save_mot_txt,
)


def make_two_object_activity(frames: int, height: int, width: int) -> tuple[np.ndarray, list[BBoxDetection]]:
    activity = np.zeros((frames, height, width), dtype=np.float32)
    gt: list[BBoxDetection] = []
    box_h = max(3, height // 7)
    box_w = max(3, width // 7)
    for t in range(frames):
        x0 = int(round(t * max(width // 2 - box_w - 1, 1) / max(frames - 1, 1)))
        y0 = int(round(t * max(height - box_h - 1, 1) / max(frames - 1, 1)))
        x1 = width - box_w - x0 - 1
        y1 = height // 4
        activity[t, y0 : y0 + box_h, x0 : x0 + box_w] += 2.0
        activity[t, y1 : y1 + box_h, x1 : x1 + box_w] += 1.5
    return activity, gt


def save_preview(frames: np.ndarray, output_path: Path) -> None:
    cols = min(4, frames.shape[0])
    rows = int(np.ceil(frames.shape[0] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    vmax = max(float(frames.max()), 1.0)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= frames.shape[0]:
            continue
        ax.imshow(frames[idx], cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"window {idx}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--closing-radius", type=int, default=0)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=Path("translation/debug_outputs/detection_readout"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, _ = make_two_object_activity(args.frames, args.height, args.width)
    np.save(args.output_dir / "synthetic_activity.npy", frames)
    activity_windows, detections = readout_bboxes(
        frames,
        window=args.window,
        stride=args.stride,
        threshold=args.threshold,
        min_area=args.min_area,
        closing_radius=args.closing_radius,
        nms_iou=args.nms_iou,
    )
    det_json = save_detections_json(detections, args.output_dir / "detections.json")
    mot_txt = save_mot_txt(detections, args.output_dir / "detections_mot.txt")
    preview = args.output_dir / "activity_windows_preview.png"
    save_preview(activity_windows, preview)

    report = {
        "frames_shape": list(frames.shape),
        "activity_windows_shape": list(activity_windows.shape),
        "num_detections": len(detections),
        "detections_json": str(det_json),
        "detections_mot": str(mot_txt),
        "preview": str(preview),
        "self_eval": evaluate_iou_matches(detections, detections),
    }
    report_path = args.output_dir / "detection_readout_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if detections else 1


if __name__ == "__main__":
    raise SystemExit(main())

