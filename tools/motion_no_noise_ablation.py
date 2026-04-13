import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spkProc.motion.motion_detection import motion_estimation


class DummyLogger:
    def add_image(self, *args, **kwargs):
        return None


def make_output_dir(path: str) -> Path:
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def dominant_nonzero(values: np.ndarray) -> int:
    values = values[values > 0]
    if values.size == 0:
        return 0
    ids, counts = np.unique(values, return_counts=True)
    return int(ids[np.argmax(counts)])


def run_direction_sweep(device: torch.device, output_dir: Path) -> dict:
    directions = {
        "right": (0, 1),
        "left": (0, -1),
        "down": (1, 0),
        "up": (-1, 0),
        "down_right": (1, 1),
        "down_left": (1, -1),
        "up_right": (-1, 1),
        "up_left": (-1, -1),
    }

    mapping = {}
    for name, (dy, dx) in directions.items():
        estimator = motion_estimation(dvs_h=48, dvs_w=64, device=device, logger=DummyLogger())
        spikes = np.zeros((18, 48, 64), dtype=np.uint8)
        y0, x0 = 20, 20

        for t in range(spikes.shape[0]):
            y = min(max(2, y0 + dy * t), 48 - 6)
            x = min(max(2, x0 + dx * t), 64 - 6)
            spikes[t, y:y + 4, x:x + 4] = 1

        dominant_dirs = []
        for t in range(spikes.shape[0]):
            frame = torch.from_numpy(spikes[t]).float().to(device)
            estimator.stdp_tracking(frame)
            motion_id, _, _ = estimator.local_wta(frame, timestamp=t, visualize=False)
            active = motion_id > 0
            if active.any():
                values, counts = torch.unique(motion_id[active], return_counts=True)
                dominant_dirs.append(int(values[counts.argmax()].item()))

        mapping[name] = dominant_dirs[-1] if dominant_dirs else 0

    names = list(directions.keys())
    ids = [mapping[name] for name in names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(names)), ids, color="tab:blue")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Recovered motion ID")
    ax.set_title("No-Noise Ablation: Direction to Motion ID Mapping")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "motion_no_noise_direction_mapping.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return mapping


def run_no_noise_ablation(device: torch.device, output_dir: Path, direction_mapping: dict) -> dict:
    directions = {
        "right": (0, 1),
        "left": (0, -1),
        "down": (1, 0),
        "up": (-1, 0),
        "down_right": (1, 1),
        "down_left": (1, -1),
        "up_right": (-1, 1),
        "up_left": (-1, -1),
    }
    speeds = [1, 2]
    warmup = 6

    correct_rates = np.zeros((len(directions), len(speeds)), dtype=np.float32)
    hit_rates = np.zeros_like(correct_rates)
    stable_ids = np.zeros_like(correct_rates, dtype=np.int32)
    expected_ids = np.zeros(len(directions), dtype=np.int32)
    case_summary = {}

    for row, (name, (dy, dx)) in enumerate(directions.items()):
        expected_id = int(direction_mapping.get(name, 0))
        expected_ids[row] = expected_id
        case_summary[name] = {}

        for col, speed in enumerate(speeds):
            estimator = motion_estimation(dvs_h=72, dvs_w=96, device=device, logger=DummyLogger())
            spikes = np.zeros((30, 72, 96), dtype=np.uint8)
            masks = np.zeros_like(spikes, dtype=bool)
            y0, x0 = 34, 46
            box_h, box_w = 5, 5

            for t in range(spikes.shape[0]):
                y = min(max(4, y0 + dy * speed * t), 72 - box_h - 4)
                x = min(max(4, x0 + dx * speed * t), 96 - box_w - 4)
                masks[t, y:y + box_h, x:x + box_w] = True
                spikes[t, masks[t]] = 1

            dominant_dirs = []
            hit_trace = []
            for t in range(spikes.shape[0]):
                frame = torch.from_numpy(spikes[t]).float().to(device)
                estimator.stdp_tracking(frame)
                motion_id, _, _ = estimator.local_wta(frame, timestamp=t, visualize=False)
                motion_id_np = motion_id.detach().cpu().numpy()
                obj_values = motion_id_np[masks[t]]
                dominant_dirs.append(dominant_nonzero(obj_values))
                hit_trace.append(float((obj_values > 0).mean()))

            valid_dirs = dominant_dirs[warmup:]
            valid_hits = hit_trace[warmup:]
            correct_rate = float(np.mean([d == expected_id for d in valid_dirs]))
            hit_rate = float(np.mean(valid_hits))
            stable_id = dominant_nonzero(np.array(valid_dirs, dtype=np.int64))

            correct_rates[row, col] = correct_rate
            hit_rates[row, col] = hit_rate
            stable_ids[row, col] = stable_id
            case_summary[name][f"speed_{speed}"] = {
                "expected_id": expected_id,
                "stable_id": stable_id,
                "correct_rate": correct_rate,
                "hit_rate": hit_rate,
            }

    direction_names = list(directions.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(correct_rates, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("No-Noise Correct Direction Rate")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(hit_rates, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("No-Noise Hit Rate")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].imshow(stable_ids, cmap="tab20", vmin=0, vmax=8)
    axes[2].set_title("No-Noise Stable Motion ID")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    for ax, data in zip(axes, [correct_rates, hit_rates, stable_ids]):
        ax.set_xticks(np.arange(len(speeds)))
        ax.set_xticklabels([f"speed {s}" for s in speeds])
        ax.set_yticks(np.arange(len(direction_names)))
        ax.set_yticklabels(direction_names)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                label = f"{data[row, col]:.2f}" if data.dtype.kind == "f" else str(int(data[row, col]))
                ax.text(col, row, label, ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "motion_no_noise_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    passed = bool(
        np.all(expected_ids > 0)
        and np.all(correct_rates >= 0.70)
        and np.all(hit_rates >= 0.70)
    )
    return {
        "mapping": direction_mapping,
        "cases": case_summary,
        "min_correct_rate": float(correct_rates.min()),
        "min_hit_rate": float(hit_rates.min()),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-noise ablation for motion estimation toy validation.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/motion_no_noise_ablation",
        help="Directory to save no-noise ablation plots and summary.",
    )
    args = parser.parse_args()

    output_dir = make_output_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = run_direction_sweep(device, output_dir)
    result = run_no_noise_ablation(device, output_dir, mapping)

    report = output_dir / "summary.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"device: {device}\n")
        f.write(f"mapping: {result['mapping']}\n")
        f.write(f"cases: {result['cases']}\n")
        f.write(f"min_correct_rate: {result['min_correct_rate']:.4f}\n")
        f.write(f"min_hit_rate: {result['min_hit_rate']:.4f}\n")
        f.write(f"passed: {result['passed']}\n")

    print(f"device: {device}")
    print(f"mapping: {result['mapping']}")
    print(f"min_correct_rate: {result['min_correct_rate']:.4f}")
    print(f"min_hit_rate: {result['min_hit_rate']:.4f}")
    print(f"passed: {result['passed']}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
