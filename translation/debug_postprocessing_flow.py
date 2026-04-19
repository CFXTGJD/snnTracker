"""Debug SpikeNet post-processing with synthetic or provided out.h5 files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translation.spikenet_postprocessing import (
    process_out_h5_to_detections,
    read_spikenet_out_h5,
    save_detections_json,
    save_mot_txt,
)


def make_synthetic_spikenet_out(out_path: Path, config_path: Path, *, frames: int, height: int, width: int) -> tuple[Path, Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    grid = np.zeros((height, width), dtype=np.int8)
    grid[1::2, 1::2] = 1
    n = np.asarray([int(np.sum(grid == 0)), int(np.sum(grid == 1))], dtype=np.int64)
    pop0_map = np.full((height, width), -1, dtype=np.int64)
    flat = np.flatnonzero(grid.ravel(order="F") == 0)
    rows, cols = np.unravel_index(flat, grid.shape, order="F")
    pop0_map[rows, cols] = np.arange(flat.size, dtype=np.int64)

    with h5py.File(config_path, "w") as cfg:
        cfg.create_dataset("/config/Net/INIT001/N", data=n)
        cfg.create_dataset("/config/Net/INIT002/dt", data=1.0)
        cfg.create_dataset("/config/Net/INIT002/step_tot", data=frames)

    spike_hist: list[int] = []
    counts: list[int] = []
    box_h = max(3, height // 5)
    box_w = max(3, width // 5)
    for t in range(frames):
        y0 = int(round(t * max(height - box_h - 1, 1) / max(frames - 1, 1)))
        x0 = int(round(t * max(width - box_w - 1, 1) / max(frames - 1, 1)))
        patch = pop0_map[y0 : y0 + box_h, x0 : x0 + box_w]
        ids = patch[patch >= 0].astype(np.int64)
        spike_hist.extend(ids.tolist())
        counts.append(int(ids.size))

    with h5py.File(out_path, "w") as out:
        dtype = h5py.string_dtype(encoding="utf-8")
        out.create_dataset("/config_filename/config_filename", data=np.asarray([str(config_path)], dtype=dtype))
        out.create_dataset("/pop_result_0/spike_hist_tot", data=np.asarray(spike_hist, dtype=np.int32))
        out.create_dataset("/pop_result_0/num_spikes_pop", data=np.asarray(counts, dtype=np.int32))
        out.create_dataset("/pop_result_0/num_ref_pop", data=np.zeros(frames, dtype=np.int32))

    return out_path, config_path


def save_activity_preview(activity: np.ndarray, output_path: Path) -> None:
    if activity.size == 0:
        return
    cols = min(4, activity.shape[0])
    rows = int(np.ceil(activity.shape[0] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    vmax = max(float(activity.max()), 1.0)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= activity.shape[0]:
            continue
        ax.imshow(activity[idx], cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"window {idx}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-h5", type=Path)
    parser.add_argument("--config-h5", type=Path)
    parser.add_argument("--make-synthetic", action="store_true")
    parser.add_argument("--synthetic-frames", type=int, default=20)
    parser.add_argument("--synthetic-height", type=int, default=24)
    parser.add_argument("--synthetic-width", type=int, default=32)
    parser.add_argument("--grid-shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("translation/debug_outputs/postprocessing"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_h5 = args.out_h5
    config_h5 = args.config_h5
    if args.make_synthetic:
        out_h5 = args.output_dir / "synthetic_out.h5"
        config_h5 = args.output_dir / "synthetic_in.h5"
        make_synthetic_spikenet_out(
            out_h5,
            config_h5,
            frames=args.synthetic_frames,
            height=args.synthetic_height,
            width=args.synthetic_width,
        )
    if out_h5 is None:
        raise SystemExit("Provide --out-h5 or --make-synthetic")

    grid_shape = None if args.grid_shape is None else (int(args.grid_shape[0]), int(args.grid_shape[1]))
    if grid_shape is None and args.make_synthetic:
        grid_shape = (args.synthetic_height, args.synthetic_width)

    result = read_spikenet_out_h5(out_h5, config_path=config_h5, grid_shape=grid_shape)
    activity, detections = process_out_h5_to_detections(
        out_h5,
        config_path=config_h5,
        grid_shape=grid_shape,
        window=args.window,
        stride=args.stride,
        threshold=args.threshold,
        min_area=args.min_area,
    )

    det_json = save_detections_json(detections, args.output_dir / "detections.json")
    mot_txt = save_mot_txt(detections, args.output_dir / "detections_mot.txt")
    preview = args.output_dir / "activity_windows_preview.png"
    save_activity_preview(activity, preview)
    report = {
        "out_h5": str(out_h5),
        "config_h5": str(config_h5) if config_h5 is not None else None,
        "dt": result.dt,
        "step_tot": result.step_tot,
        "N": result.n,
        "activity_shape": list(activity.shape),
        "num_detections": len(detections),
        "detections_json": str(det_json),
        "detections_mot": str(mot_txt),
        "preview": str(preview),
    }
    report_path = args.output_dir / "postprocessing_debug_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if len(detections) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
