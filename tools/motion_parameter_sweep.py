import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spkProc.motion.motion_detection import motion_estimation


DIRECTIONS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
    "down_right": (1, 1),
    "down_left": (1, -1),
    "up_right": (-1, 1),
    "up_left": (-1, -1),
}


class DummyLogger:
    def add_image(self, *args, **kwargs):
        return None


def parse_number_list(value: str, cast):
    return [cast(item) for item in value.replace(",", " ").split()]


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


def get_direction_mapping(device: torch.device) -> dict:
    mapping = {}
    for name, (dy, dx) in DIRECTIONS.items():
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
    return mapping


def run_case(
    device: torch.device,
    direction_name: str,
    speed: int,
    box_size: int,
    noise_prob: float,
    num_frames: int,
    height: int,
    width: int,
    warmup: int,
    expected_id: int,
    seed: int,
) -> dict:
    dy, dx = DIRECTIONS[direction_name]
    rng = np.random.default_rng(seed)
    estimator = motion_estimation(dvs_h=height, dvs_w=width, device=device, logger=DummyLogger())

    spikes = np.zeros((num_frames, height, width), dtype=np.uint8)
    masks = np.zeros_like(spikes, dtype=bool)
    y0, x0 = height // 2, width // 2

    margin = max(4, box_size + 1)
    for t in range(num_frames):
        y = min(max(margin, y0 + dy * speed * t), height - box_size - margin)
        x = min(max(margin, x0 + dx * speed * t), width - box_size - margin)
        masks[t, y:y + box_size, x:x + box_size] = True
        spikes[t, masks[t]] = 1

        if noise_prob > 0:
            noise = rng.random((height, width)) < noise_prob
            noise[masks[t]] = False
            spikes[t, noise] = 1

    dominant_dirs = []
    hit_trace = []
    for t in range(num_frames):
        frame = torch.from_numpy(spikes[t]).float().to(device)
        estimator.stdp_tracking(frame)
        motion_id, _, _ = estimator.local_wta(frame, timestamp=t, visualize=False)
        motion_id_np = motion_id.detach().cpu().numpy()
        obj_values = motion_id_np[masks[t]]
        dominant_dirs.append(dominant_nonzero(obj_values))
        hit_trace.append(float((obj_values > 0).mean()))

    valid_dirs = dominant_dirs[warmup:]
    valid_hits = hit_trace[warmup:]
    correct_rate = float(np.mean([d == expected_id for d in valid_dirs])) if valid_dirs else 0.0
    hit_rate = float(np.mean(valid_hits)) if valid_hits else 0.0
    stable_id = dominant_nonzero(np.array(valid_dirs, dtype=np.int64))

    return {
        "direction": direction_name,
        "speed": speed,
        "box_size": box_size,
        "noise_prob": noise_prob,
        "expected_id": expected_id,
        "stable_id": stable_id,
        "correct_rate": correct_rate,
        "hit_rate": hit_rate,
    }


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(output_dir: Path, aggregate_rows: list, speeds: list, box_sizes: list, noise_probs: list) -> None:
    for noise_prob in noise_probs:
        subset = [row for row in aggregate_rows if row["noise_prob"] == noise_prob]
        min_correct = np.zeros((len(box_sizes), len(speeds)), dtype=np.float32)
        min_hit = np.zeros_like(min_correct)
        passed = np.zeros_like(min_correct)

        for row in subset:
            bi = box_sizes.index(row["box_size"])
            si = speeds.index(row["speed"])
            min_correct[bi, si] = row["min_correct_rate"]
            min_hit[bi, si] = row["min_hit_rate"]
            passed[bi, si] = 1.0 if row["passed"] else 0.0

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        data_items = [
            ("Min Correct Rate", min_correct, 0, 1),
            ("Min Hit Rate", min_hit, 0, 1),
            ("Pass Mask", passed, 0, 1),
        ]
        for ax, (title, data, vmin, vmax) in zip(axes, data_items):
            im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}, noise={noise_prob:g}")
            ax.set_xlabel("Speed")
            ax.set_ylabel("Box Size")
            ax.set_xticks(np.arange(len(speeds)))
            ax.set_xticklabels([str(s) for s in speeds])
            ax.set_yticks(np.arange(len(box_sizes)))
            ax.set_yticklabels([str(b) for b in box_sizes])
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    label = f"{data[y, x]:.2f}"
                    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(output_dir / f"motion_sweep_noise_{noise_prob:g}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sweep for motion direction failure boundaries.")
    parser.add_argument("--output_dir", type=str, default="results/motion_parameter_sweep")
    parser.add_argument("--speeds", type=str, default="1 2 3")
    parser.add_argument("--box_sizes", type=str, default="3 5 7")
    parser.add_argument("--noise_probs", type=str, default="0 0.001 0.003")
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--correct_thr", type=float, default=0.70)
    parser.add_argument("--hit_thr", type=float, default=0.70)
    args = parser.parse_args()

    output_dir = make_output_dir(args.output_dir)
    speeds = parse_number_list(args.speeds, int)
    box_sizes = parse_number_list(args.box_sizes, int)
    noise_probs = parse_number_list(args.noise_probs, float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mapping = get_direction_mapping(device)
    case_rows = []
    aggregate_rows = []

    total = len(noise_probs) * len(box_sizes) * len(speeds)
    done = 0
    for noise_prob in noise_probs:
        for box_size in box_sizes:
            for speed in speeds:
                done += 1
                print(f"[{done}/{total}] speed={speed} box_size={box_size} noise={noise_prob:g}", flush=True)
                combo_rows = []
                for direction_index, direction_name in enumerate(DIRECTIONS):
                    row = run_case(
                        device=device,
                        direction_name=direction_name,
                        speed=speed,
                        box_size=box_size,
                        noise_prob=noise_prob,
                        num_frames=args.num_frames,
                        height=args.height,
                        width=args.width,
                        warmup=args.warmup,
                        expected_id=int(mapping[direction_name]),
                        seed=1000 + direction_index + speed * 100 + box_size * 10 + int(noise_prob * 1_000_000),
                    )
                    row["passed"] = row["correct_rate"] >= args.correct_thr and row["hit_rate"] >= args.hit_thr
                    case_rows.append(row)
                    combo_rows.append(row)

                min_correct = min(row["correct_rate"] for row in combo_rows)
                mean_correct = float(np.mean([row["correct_rate"] for row in combo_rows]))
                min_hit = min(row["hit_rate"] for row in combo_rows)
                mean_hit = float(np.mean([row["hit_rate"] for row in combo_rows]))
                failed_dirs = [
                    row["direction"]
                    for row in combo_rows
                    if row["correct_rate"] < args.correct_thr or row["hit_rate"] < args.hit_thr
                ]
                aggregate_rows.append(
                    {
                        "speed": speed,
                        "box_size": box_size,
                        "noise_prob": noise_prob,
                        "min_correct_rate": min_correct,
                        "mean_correct_rate": mean_correct,
                        "min_hit_rate": min_hit,
                        "mean_hit_rate": mean_hit,
                        "failed_directions": " ".join(failed_dirs),
                        "passed": len(failed_dirs) == 0,
                    }
                )

    write_csv(
        output_dir / "motion_parameter_sweep_cases.csv",
        case_rows,
        [
            "direction",
            "speed",
            "box_size",
            "noise_prob",
            "expected_id",
            "stable_id",
            "correct_rate",
            "hit_rate",
            "passed",
        ],
    )
    write_csv(
        output_dir / "motion_parameter_sweep_aggregate.csv",
        aggregate_rows,
        [
            "speed",
            "box_size",
            "noise_prob",
            "min_correct_rate",
            "mean_correct_rate",
            "min_hit_rate",
            "mean_hit_rate",
            "failed_directions",
            "passed",
        ],
    )
    plot_heatmaps(output_dir, aggregate_rows, speeds, box_sizes, noise_probs)

    passing = [row for row in aggregate_rows if row["passed"]]
    failing = [row for row in aggregate_rows if not row["passed"]]
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"device: {device}\n")
        f.write(f"mapping: {mapping}\n")
        f.write(f"speeds: {speeds}\n")
        f.write(f"box_sizes: {box_sizes}\n")
        f.write(f"noise_probs: {noise_probs}\n")
        f.write(f"num_frames: {args.num_frames}\n")
        f.write(f"warmup: {args.warmup}\n")
        f.write(f"correct_thr: {args.correct_thr}\n")
        f.write(f"hit_thr: {args.hit_thr}\n")
        f.write(f"passing_combinations: {len(passing)}\n")
        f.write(f"failing_combinations: {len(failing)}\n")
        f.write("passing:\n")
        for row in passing:
            f.write(f"  {row}\n")
        f.write("failing:\n")
        for row in failing:
            f.write(f"  {row}\n")

    print(f"mapping: {mapping}")
    print(f"passing_combinations: {len(passing)}")
    print(f"failing_combinations: {len(failing)}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
