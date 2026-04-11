import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize spike dataset scenes.")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to a scene directory, e.g. ./motVidarReal2025/spike59")
    parser.add_argument("--begin_idx", type=int, default=0, help="Starting frame index")
    parser.add_argument("--num_frames", type=int, default=300, help="Number of frames to load")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of frames shown in the montage")
    parser.add_argument("--output_dir", type=str, default="results/spike_dataset_viz", help="Directory to save figures")
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

    print("scene_dir:", scene_dir)
    print("loaded_spikes_shape:", spikes.shape)
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
