import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spkData.load_dat import SpikeStream


def load_scene_config(scene_dir: Path) -> dict:
    config_path = scene_dir.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found near scene directory: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def pick_frame_indices(total_frames: int, num_samples: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    num_samples = max(1, min(num_samples, total_frames))
    return np.linspace(0, total_frames - 1, num=num_samples, dtype=int)


def save_montage(spikes: np.ndarray, indices: np.ndarray, output_path: Path) -> None:
    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, indices):
        ax.imshow(spikes[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"frame {idx}")
        ax.axis("off")

    fig.suptitle("Sample Spike Frames", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_activity_curve(spikes: np.ndarray, output_path: Path) -> None:
    activity = spikes.reshape(spikes.shape[0], -1).sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(activity, linewidth=1.2)
    ax.set_title("Spike Activity per Frame")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Active Pixels")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_spatial_heatmap(spikes: np.ndarray, output_path: Path) -> None:
    heatmap = spikes.sum(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heatmap, cmap="hot")
    ax.set_title("Spatial Spike Accumulation")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Spike Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_temporal_projection(spikes: np.ndarray, output_path: Path) -> None:
    x_time = spikes.sum(axis=1)
    y_time = spikes.sum(axis=2)

    vmax = max(float(np.percentile(x_time, 99)), float(np.percentile(y_time, 99)), 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    im0 = axes[0].imshow(
        x_time.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )
    axes[0].set_title("X-Time Spike Projection")
    axes[0].set_ylabel("Width")
    fig.colorbar(im0, ax=axes[0], shrink=0.9, label="Spike Count")

    im1 = axes[1].imshow(
        y_time.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )
    axes[1].set_title("Y-Time Spike Projection")
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylabel("Height")
    fig.colorbar(im1, ax=axes[1], shrink=0.9, label="Spike Count")

    fig.suptitle("Temporal Spike Projections", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_gt_boxes(gt_path: Path) -> dict:
    boxes_by_frame = {}
    if not gt_path.exists():
        return boxes_by_frame

    with open(gt_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) == 1:
                row = row[0].replace(",", " ").split()
            if len(row) < 6:
                continue

            try:
                frame_id = int(float(row[0]))
                track_id = int(float(row[1]))
                x, y, w, h = (float(row[i]) for i in range(2, 6))
            except ValueError:
                continue

            boxes_by_frame.setdefault(frame_id, []).append((track_id, x, y, w, h))

    return boxes_by_frame


def save_gt_overlay(
    spikes: np.ndarray,
    gt_boxes: dict,
    indices: np.ndarray,
    begin_idx: int,
    output_path: Path,
) -> None:
    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, indices):
        frame_id = begin_idx + int(idx)
        ax.imshow(spikes[idx], cmap="gray", vmin=0, vmax=1)

        for track_id, x, y, w, h in gt_boxes.get(frame_id, []):
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1.5,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x,
                max(0, y - 2),
                str(track_id),
                color="yellow",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "edgecolor": "none"},
            )

        ax.set_title(f"frame {frame_id}")
        ax.axis("off")

    fig.suptitle("Spike Frames with GT Boxes", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize spike dataset scenes.")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to a scene directory, e.g. ./motVidarReal2025/spike59")
    parser.add_argument("--begin_idx", type=int, default=0, help="Starting frame index")
    parser.add_argument("--num_frames", type=int, default=300, help="Number of frames to load")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of frames shown in the montage")
    parser.add_argument("--output_dir", type=str, default="results/spike_dataset_viz", help="Directory to save figures")
    parser.add_argument("--gt_path", type=str, default=None, help="Optional GT file path. Defaults to scene_dir/spikes_gt.txt when it exists")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene directory not found: {scene_dir}")

    scene_name = scene_dir.name
    output_dir = Path(args.output_dir).resolve() / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_scene_config(scene_dir)
    spike_path = scene_dir / "spikes.dat"
    if not spike_path.exists():
        raise FileNotFoundError(f"spikes.dat not found: {spike_path}")

    stream = SpikeStream(
        filepath=str(spike_path),
        spike_h=config["spike_h"],
        spike_w=config["spike_w"],
        print_dat_detail=False,
    )
    spikes = stream.get_block_spikes(begin_idx=args.begin_idx, block_len=args.num_frames)

    sample_indices = pick_frame_indices(spikes.shape[0], args.num_samples)
    save_montage(spikes, sample_indices, output_dir / f"{scene_name}_montage.png")
    save_activity_curve(spikes, output_dir / f"{scene_name}_activity_curve.png")
    save_spatial_heatmap(spikes, output_dir / f"{scene_name}_spatial_heatmap.png")
    save_temporal_projection(spikes, output_dir / f"{scene_name}_temporal_projection.png")

    gt_path = Path(args.gt_path).resolve() if args.gt_path else scene_dir / "spikes_gt.txt"
    gt_boxes = load_gt_boxes(gt_path)
    if gt_boxes:
        save_gt_overlay(
            spikes=spikes,
            gt_boxes=gt_boxes,
            indices=sample_indices,
            begin_idx=args.begin_idx,
            output_path=output_dir / f"{scene_name}_gt_overlay.png",
        )

    print("scene_dir:", scene_dir)
    print("loaded_spikes_shape:", spikes.shape)
    print("output_dir:", output_dir)
    print("gt_path:", gt_path if gt_boxes else "not found or empty")


if __name__ == "__main__":
    main()
