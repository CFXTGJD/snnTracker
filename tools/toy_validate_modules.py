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


def generate_stp_complex_sequence(num_frames: int = 120, h: int = 64, w: int = 80) -> tuple:
    rng = np.random.default_rng(7)
    spikes = np.zeros((num_frames, h, w), dtype=np.uint8)

    static_mask = np.zeros((h, w), dtype=bool)
    static_mask[8:12, 10:50] = True
    static_mask[22:50:4, 62] = True

    burst_mask = np.zeros((h, w), dtype=bool)
    burst_mask[42:50, 18:28] = True

    moving_masks = np.zeros((num_frames, h, w), dtype=bool)
    moving2_masks = np.zeros((num_frames, h, w), dtype=bool)
    noise_masks = np.zeros((num_frames, h, w), dtype=bool)

    for t in range(num_frames):
        # Persistent structured flicker that should be suppressed after adaptation.
        spikes[t, static_mask] = 1

        # Sparse uncorrelated noise that should not dominate the filtered output.
        noise = rng.random((h, w)) < 0.003
        noise[static_mask] = False
        noise_masks[t] = noise
        spikes[t, noise] = 1

        # First moving target: diagonal/right motion, delayed onset.
        if 18 <= t < 95:
            y = min(h - 8, 18 + (t - 18) // 3)
            x = min(w - 8, 5 + (t - 18))
            moving_masks[t, y:y + 4, x:x + 4] = True
            spikes[t, moving_masks[t]] = 1

        # Second moving target: vertical motion with a different onset.
        if 45 <= t < 112:
            y = min(h - 7, 4 + (t - 45))
            x = 54
            moving2_masks[t, y:y + 5, x:x + 3] = True
            spikes[t, moving2_masks[t]] = 1

        # Short transient object: should create a strong response when it appears.
        if 70 <= t < 75:
            spikes[t, burst_mask] = 1

    masks = {
        "static": static_mask,
        "moving1": moving_masks,
        "moving2": moving2_masks,
        "noise": noise_masks,
        "burst": burst_mask,
    }
    return spikes, masks


def run_stp_complex_toy(device: torch.device, output_dir: Path) -> dict:
    spikes, masks = generate_stp_complex_sequence()
    stp = STPFilter(
        spike_h=spikes.shape[1],
        spike_w=spikes.shape[2],
        device=device,
        filterThr=0.25,
        voltageMin=-8,
        lifThr=2,
    )

    static_pass = []
    noise_pass = []
    moving1_hit = []
    moving2_hit = []
    burst_pass = []
    input_count = []
    filtered_count = []
    static_thr = []

    static_mask = masks["static"]
    burst_mask = masks["burst"]

    for t in range(spikes.shape[0]):
        frame = torch.from_numpy(spikes[t]).to(device)
        stp.update_dynamics(t, frame)
        filtered = stp.filter_spk.detach().cpu().numpy().astype(bool)
        adjusted_thr = stp.adjusted_threshold.detach().cpu().numpy()

        moving1_mask = masks["moving1"][t]
        moving2_mask = masks["moving2"][t]
        noise_mask = masks["noise"][t]

        static_pass.append(float(filtered[static_mask].mean()))
        noise_pass.append(float(filtered[noise_mask].mean()) if noise_mask.any() else 0.0)
        moving1_hit.append(float(filtered[moving1_mask].any()) if moving1_mask.any() else 0.0)
        moving2_hit.append(float(filtered[moving2_mask].any()) if moving2_mask.any() else 0.0)
        burst_pass.append(float(filtered[burst_mask].mean()) if 70 <= t < 75 else 0.0)
        input_count.append(float(spikes[t].sum()))
        filtered_count.append(float(filtered.sum()))
        static_thr.append(float(adjusted_thr[static_mask].mean()))

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, frame_idx in zip(axes[0], [0, 50]):
        ax.imshow(spikes[frame_idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Complex STP Input Frame {frame_idx}")
        ax.axis("off")

    axes[1, 0].plot(static_pass, label="static structure pass")
    axes[1, 0].plot(noise_pass, label="random noise pass", alpha=0.8)
    axes[1, 0].plot(moving1_hit, label="moving target 1 hit")
    axes[1, 0].plot(moving2_hit, label="moving target 2 hit")
    axes[1, 0].set_title("Filtered Response by Region")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Pass / hit rate")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(static_thr, color="tab:red")
    axes[1, 1].set_title("Mean Adaptive Threshold on Static Structure")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Threshold")
    axes[1, 1].grid(alpha=0.3)

    axes[2, 0].plot(input_count, label="input spikes")
    axes[2, 0].plot(filtered_count, label="filtered spikes")
    axes[2, 0].set_title("Input vs Filtered Spike Count")
    axes[2, 0].set_xlabel("Frame")
    axes[2, 0].set_ylabel("Pixels")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].plot(burst_pass, color="tab:purple")
    axes[2, 1].axvspan(70, 74, color="tab:purple", alpha=0.12, label="burst window")
    axes[2, 1].set_title("Transient Burst Response")
    axes[2, 1].set_xlabel("Frame")
    axes[2, 1].set_ylabel("Burst mask pass rate")
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "stp_complex_toy_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    static_late_mean = float(np.mean(static_pass[40:]))
    noise_mean = float(np.mean(noise_pass))
    moving1_mean = float(np.mean(moving1_hit[25:95]))
    moving2_mean = float(np.mean(moving2_hit[52:112]))
    burst_peak = float(np.max(burst_pass[70:75]))
    suppression_ratio = float(np.mean(filtered_count[40:]) / max(np.mean(input_count[40:]), 1.0))

    return {
        "static_late_mean": static_late_mean,
        "noise_mean": noise_mean,
        "moving1_mean": moving1_mean,
        "moving2_mean": moving2_mean,
        "burst_peak": burst_peak,
        "suppression_ratio": suppression_ratio,
        "passed": (
            static_late_mean < 0.15
            and noise_mean < 0.35
            and moving1_mean > 0.55
            and moving2_mean > 0.55
            and burst_peak > 0.5
            and suppression_ratio < 0.8
        ),
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


def _dominant_nonzero(values: np.ndarray) -> int:
    values = values[values > 0]
    if values.size == 0:
        return 0
    ids, counts = np.unique(values, return_counts=True)
    return int(ids[np.argmax(counts)])


def run_motion_complex_toy(device: torch.device, output_dir: Path, direction_mapping: dict) -> dict:
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
    rng = np.random.default_rng(23)
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

                # Sparse distractor noise tests whether the learned dominant direction survives clutter.
                noise = rng.random((72, 96)) < 0.001
                noise[masks[t]] = False
                spikes[t, noise] = 1

            dominant_dirs = []
            hit_trace = []
            for t in range(spikes.shape[0]):
                frame = torch.from_numpy(spikes[t]).float().to(device)
                estimator.stdp_tracking(frame)
                motion_id, _, _ = estimator.local_wta(frame, timestamp=t, visualize=False)
                motion_id_np = motion_id.detach().cpu().numpy()
                obj_values = motion_id_np[masks[t]]
                dominant_dirs.append(_dominant_nonzero(obj_values))
                hit_trace.append(float((obj_values > 0).mean()))

            valid_dirs = dominant_dirs[warmup:]
            valid_hits = hit_trace[warmup:]
            correct_rate = float(np.mean([d == expected_id for d in valid_dirs]))
            hit_rate = float(np.mean(valid_hits))
            stable_id = _dominant_nonzero(np.array(valid_dirs, dtype=np.int64))

            correct_rates[row, col] = correct_rate
            hit_rates[row, col] = hit_rate
            stable_ids[row, col] = stable_id
            case_summary[name][f"speed_{speed}"] = {
                "expected_id": expected_id,
                "stable_id": stable_id,
                "correct_rate": correct_rate,
                "hit_rate": hit_rate,
            }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    direction_names = list(directions.keys())

    im0 = axes[0].imshow(correct_rates, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Correct Direction Rate")
    axes[0].set_xticks(np.arange(len(speeds)))
    axes[0].set_xticklabels([f"speed {s}" for s in speeds])
    axes[0].set_yticks(np.arange(len(direction_names)))
    axes[0].set_yticklabels(direction_names)
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(hit_rates, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Object Motion Pixel Hit Rate")
    axes[1].set_xticks(np.arange(len(speeds)))
    axes[1].set_xticklabels([f"speed {s}" for s in speeds])
    axes[1].set_yticks(np.arange(len(direction_names)))
    axes[1].set_yticklabels(direction_names)
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].imshow(stable_ids, cmap="tab20", vmin=0, vmax=8)
    axes[2].set_title("Recovered Stable Motion ID")
    axes[2].set_xticks(np.arange(len(speeds)))
    axes[2].set_xticklabels([f"speed {s}" for s in speeds])
    axes[2].set_yticks(np.arange(len(direction_names)))
    axes[2].set_yticklabels(direction_names)
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    for ax, data in zip(axes, [correct_rates, hit_rates, stable_ids]):
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data.dtype.kind in {"f"}:
                    label = f"{data[row, col]:.2f}"
                else:
                    label = str(int(data[row, col]))
                ax.text(col, row, label, ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "motion_complex_toy_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    passed = bool(
        np.all(expected_ids > 0)
        and np.all(correct_rates >= 0.70)
        and np.all(hit_rates >= 0.70)
    )

    return {
        "cases": case_summary,
        "min_correct_rate": float(correct_rates.min()),
        "min_hit_rate": float(hit_rates.min()),
        "passed": passed,
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
    stp_complex_res = run_stp_complex_toy(device, output_dir)
    motion_res = run_motion_toy(device, output_dir)
    motion_sweep_res = run_motion_direction_sweep(device, output_dir)
    motion_complex_res = run_motion_complex_toy(device, output_dir, motion_sweep_res["mapping"])

    report = output_dir / "summary.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"device: {device}\n")
        f.write("stp_toy:\n")
        f.write(f"  static_late_mean: {stp_res['static_late_mean']:.4f}\n")
        f.write(f"  moving_late_mean: {stp_res['moving_late_mean']:.4f}\n")
        f.write(f"  passed: {stp_res['passed']}\n")
        f.write("stp_complex_toy:\n")
        f.write(f"  static_late_mean: {stp_complex_res['static_late_mean']:.4f}\n")
        f.write(f"  noise_mean: {stp_complex_res['noise_mean']:.4f}\n")
        f.write(f"  moving1_mean: {stp_complex_res['moving1_mean']:.4f}\n")
        f.write(f"  moving2_mean: {stp_complex_res['moving2_mean']:.4f}\n")
        f.write(f"  burst_peak: {stp_complex_res['burst_peak']:.4f}\n")
        f.write(f"  suppression_ratio: {stp_complex_res['suppression_ratio']:.4f}\n")
        f.write(f"  passed: {stp_complex_res['passed']}\n")
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
        f.write("motion_complex_toy:\n")
        f.write(f"  cases: {motion_complex_res['cases']}\n")
        f.write(f"  min_correct_rate: {motion_complex_res['min_correct_rate']:.4f}\n")
        f.write(f"  min_hit_rate: {motion_complex_res['min_hit_rate']:.4f}\n")
        f.write(f"  passed: {motion_complex_res['passed']}\n")

    print(f"device: {device}")
    print(f"stp_toy: {stp_res}")
    print(f"stp_complex_toy: {stp_complex_res}")
    print(f"motion_toy: {motion_res}")
    print(f"motion_direction_sweep: {motion_sweep_res}")
    print(f"motion_complex_toy: {motion_complex_res}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
