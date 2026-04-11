import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spkProc.filters.stp_filters_torch import STPFilter
from spkProc.motion.motion_detection import motion_estimation


class DummyLogger:
    def add_image(self, *args, **kwargs):
        return None


def make_output_dir(path: str) -> Path:
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def generate_stp_toy_sequence(num_frames: int = 80, h: int = 48, w: int = 64) -> np.ndarray:
    spikes = np.zeros((num_frames, h, w), dtype=np.uint8)
    static_y, static_x = 12, 14
    moving_y0, moving_x0 = 30, 6

    for t in range(num_frames):
        # High-frequency static flicker: should be suppressed after adaptation settles.
        spikes[t, static_y, static_x] = 1

        # Delayed moving target: should remain salient after it appears.
        if t >= 20:
            x = min(w - 3, moving_x0 + (t - 20))
            spikes[t, moving_y0:moving_y0 + 2, x:x + 2] = 1

    return spikes


def run_stp_toy(device: torch.device, output_dir: Path) -> dict:
    spikes = generate_stp_toy_sequence()
    stp = STPFilter(
        spike_h=spikes.shape[1],
        spike_w=spikes.shape[2],
        device=device,
        filterThr=0.25,
        voltageMin=-8,
        lifThr=2,
    )

    static_pass = []
    moving_pass = []
    threshold_trace = []

    static_y, static_x = 12, 14
    moving_y = 30

    for t in range(spikes.shape[0]):
        frame = torch.from_numpy(spikes[t]).to(device)
        stp.update_dynamics(t, frame)

        filtered = stp.filter_spk.detach().cpu().numpy()
        adjusted_thr = stp.adjusted_threshold.detach().cpu().numpy()

        static_pass.append(float(filtered[static_y, static_x]))
        moving_pass.append(float(filtered[moving_y:moving_y + 2].max()))
        threshold_trace.append(float(adjusted_thr[static_y, static_x]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(spikes[0], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("STP Toy Input Frame 0")
    axes[0, 1].imshow(spikes[35], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("STP Toy Input Frame 35")

    axes[1, 0].plot(static_pass, label="static pixel survives")
    axes[1, 0].plot(moving_pass, label="moving object survives")
    axes[1, 0].set_title("Filtered Spike Response")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Binary pass")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(threshold_trace, color="tab:red")
    axes[1, 1].set_title("Adaptive Threshold at Static Pixel")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Threshold")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "stp_toy_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    static_late_mean = float(np.mean(static_pass[20:]))
    moving_late_mean = float(np.mean(moving_pass[25:]))

    return {
        "static_late_mean": static_late_mean,
        "moving_late_mean": moving_late_mean,
        "passed": moving_late_mean > static_late_mean + 0.4,
    }


def generate_motion_toy_sequence(num_frames: int = 18, h: int = 48, w: int = 64) -> np.ndarray:
    spikes = np.zeros((num_frames, h, w), dtype=np.uint8)
    y0, x0 = 20, 8

    for t in range(num_frames):
        x = min(w - 5, x0 + t)
        spikes[t, y0:y0 + 4, x:x + 4] = 1

    return spikes


def run_motion_toy(device: torch.device, output_dir: Path) -> dict:
    spikes = generate_motion_toy_sequence()
    estimator = motion_estimation(
        dvs_h=spikes.shape[1],
        dvs_w=spikes.shape[2],
        device=device,
        logger=DummyLogger(),
    )

    dominant_dirs = []
    mean_dx = []
    mean_dy = []
    mask_sizes = []

    for t in range(spikes.shape[0]):
        frame = torch.from_numpy(spikes[t]).float().to(device)
        estimator.stdp_tracking(frame)
        motion_id, motion_vector_max, _ = estimator.local_wta(frame, timestamp=t, visualize=False)

        motion_id_np = motion_id.detach().cpu().numpy()
        mv_np = motion_vector_max.detach().cpu().numpy()
        active = motion_id_np > 0
        mask_sizes.append(int(active.sum()))

        if active.any():
            values, counts = np.unique(motion_id_np[active], return_counts=True)
            dominant_dirs.append(int(values[np.argmax(counts)]))
            mean_dx.append(float(mv_np[:, :, 0][active].mean()))
            mean_dy.append(float(mv_np[:, :, 1][active].mean()))
        else:
            dominant_dirs.append(0)
            mean_dx.append(0.0)
            mean_dy.append(0.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(spikes[0], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Motion Toy Frame 0")
    axes[0, 1].imshow(spikes[-1], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Motion Toy Last Frame")

    axes[1, 0].plot(dominant_dirs, marker="o")
    axes[1, 0].set_title("Dominant Motion ID per Frame")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Motion ID (1-8)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(mean_dx, label="mean dx")
    axes[1, 1].plot(mean_dy, label="mean dy")
    axes[1, 1].plot(mask_sizes, label="active motion pixels")
    axes[1, 1].set_title("Recovered Motion Statistics")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "motion_toy_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    valid_dirs = [d for d, n in zip(dominant_dirs[2:], mask_sizes[2:]) if n > 0 and d > 0]
    mean_dx_valid = float(np.mean(mean_dx[2:])) if len(mean_dx) > 2 else 0.0
    mean_dy_valid = float(np.mean(mean_dy[2:])) if len(mean_dy) > 2 else 0.0
    stable_dir = int(max(set(valid_dirs), key=valid_dirs.count)) if valid_dirs else 0

    return {
        "dominant_dirs": valid_dirs,
        "stable_dir": stable_dir,
        "mean_dx": mean_dx_valid,
        "mean_dy": mean_dy_valid,
        "passed": len(valid_dirs) > 0 and len(set(valid_dirs)) == 1,
    }


def run_motion_direction_sweep(device: torch.device, output_dir: Path) -> dict:
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

        for t in range(18):
            y = min(max(2, y0 + dy * t), 48 - 6)
            x = min(max(2, x0 + dx * t), 64 - 6)
            spikes[t, y:y + 4, x:x + 4] = 1

        dominant_dirs = []
        for t in range(18):
            frame = torch.from_numpy(spikes[t]).float().to(device)
            estimator.stdp_tracking(frame)
            motion_id, _, _ = estimator.local_wta(frame, timestamp=t, visualize=False)
            active = motion_id > 0
            if active.any():
                values, counts = torch.unique(motion_id[active], return_counts=True)
                dominant_dirs.append(int(values[counts.argmax()].item()))

        mapping[name] = dominant_dirs[-1] if dominant_dirs else 0

    ordered_names = list(directions.keys())
    ordered_ids = [mapping[name] for name in ordered_names]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(ordered_names)), ordered_ids, color="tab:blue")
    ax.set_xticks(np.arange(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=30, ha="right")
    ax.set_ylabel("Recovered motion ID")
    ax.set_title("Direction Sweep: Input Motion vs Recovered ID")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "motion_direction_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    unique_nonzero = len({v for v in ordered_ids if v > 0})
    return {
        "mapping": mapping,
        "unique_nonzero": unique_nonzero,
        "passed": unique_nonzero == len(ordered_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy validations for STP adaptation and motion estimation.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/toy_module_validation",
        help="Directory to save plots and logs.",
    )
    args = parser.parse_args()

    output_dir = make_output_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stp_res = run_stp_toy(device, output_dir)
    motion_res = run_motion_toy(device, output_dir)
    motion_sweep_res = run_motion_direction_sweep(device, output_dir)

    report = output_dir / "summary.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"device: {device}\n")
        f.write("stp_toy:\n")
        f.write(f"  static_late_mean: {stp_res['static_late_mean']:.4f}\n")
        f.write(f"  moving_late_mean: {stp_res['moving_late_mean']:.4f}\n")
        f.write(f"  passed: {stp_res['passed']}\n")
        f.write("motion_toy:\n")
        f.write(f"  stable_dir: {motion_res['stable_dir']}\n")
        f.write(f"  mean_dx: {motion_res['mean_dx']:.4f}\n")
        f.write(f"  mean_dy: {motion_res['mean_dy']:.4f}\n")
        f.write(f"  dominant_dirs: {motion_res['dominant_dirs']}\n")
        f.write(f"  passed: {motion_res['passed']}\n")
        f.write("motion_direction_sweep:\n")
        f.write(f"  mapping: {motion_sweep_res['mapping']}\n")
        f.write(f"  unique_nonzero: {motion_sweep_res['unique_nonzero']}\n")
        f.write(f"  passed: {motion_sweep_res['passed']}\n")

    print(f"device: {device}")
    print(f"stp_toy: {stp_res}")
    print(f"motion_toy: {motion_res}")
    print(f"motion_direction_sweep: {motion_sweep_res}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
