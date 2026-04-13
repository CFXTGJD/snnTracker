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

OBJECTS = {
    "obj_right": {"direction": "right", "start": (12, 8)},
    "obj_left": {"direction": "left", "start": (12, 82)},
    "obj_down_right": {"direction": "down_right", "start": (40, 8)},
    "obj_up_left": {"direction": "up_left", "start": (62, 82)},
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


def build_sequence(speed: int, box_size: int, noise_prob: float, num_frames: int, height: int, width: int, seed: int):
    rng = np.random.default_rng(seed)
    spikes = np.zeros((num_frames, height, width), dtype=np.uint8)
    masks = {name: np.zeros_like(spikes, dtype=bool) for name in OBJECTS}
    margin = max(3, box_size + 1)

    for t in range(num_frames):
        occupied = np.zeros((height, width), dtype=bool)
        for obj_name, cfg in OBJECTS.items():
            dy, dx = DIRECTIONS[cfg["direction"]]
            y0, x0 = cfg["start"]
            y = min(max(margin, y0 + dy * speed * t), height - box_size - margin)
            x = min(max(margin, x0 + dx * speed * t), width - box_size - margin)
            masks[obj_name][t, y:y + box_size, x:x + box_size] = True
            occupied |= masks[obj_name][t]
            spikes[t, masks[obj_name][t]] = 1
        if noise_prob > 0:
            noise = rng.random((height, width)) < noise_prob
            noise[occupied] = False
            spikes[t, noise] = 1
    return spikes, masks


def run_case(device, mapping, speed, box_size, noise_prob, num_frames, height, width, warmup, seed):
    estimator = motion_estimation(dvs_h=height, dvs_w=width, device=device, logger=DummyLogger())
    spikes, masks = build_sequence(speed, box_size, noise_prob, num_frames, height, width, seed)
    traces = {name: [] for name in OBJECTS}
    hits = {name: [] for name in OBJECTS}

    for t in range(num_frames):
        frame = torch.from_numpy(spikes[t]).float().to(device)
        estimator.stdp_tracking(frame)
        motion_id, _, _ = estimator.local_wta(frame, timestamp=t, visualize=False)
        motion_id_np = motion_id.detach().cpu().numpy()
        for obj_name, cfg in OBJECTS.items():
            values = motion_id_np[masks[obj_name][t]]
            traces[obj_name].append(dominant_nonzero(values))
            hits[obj_name].append(float((values > 0).mean()))

    rows = []
    for obj_name, cfg in OBJECTS.items():
        direction = cfg["direction"]
        expected_id = int(mapping[direction])
        valid_dirs = traces[obj_name][warmup:]
        valid_hits = hits[obj_name][warmup:]
        correct_rate = float(np.mean([item == expected_id for item in valid_dirs]))
        hit_rate = float(np.mean(valid_hits))
        rows.append(
            {
                "object": obj_name,
                "direction": direction,
                "speed": speed,
                "box_size": box_size,
                "noise_prob": noise_prob,
                "expected_id": expected_id,
                "stable_id": dominant_nonzero(np.array(valid_dirs, dtype=np.int64)),
                "correct_rate": correct_rate,
                "hit_rate": hit_rate,
            }
        )
    return rows


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output_dir: Path, aggregate_rows: list, speeds: list, box_sizes: list, noise_probs: list) -> None:
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
        for ax, title, data in zip(
            axes,
            ["Min Correct Rate", "Min Hit Rate", "Pass Mask"],
            [min_correct, min_hit, passed],
        ):
            im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"No-Overlap {title}, noise={noise_prob:g}")
            ax.set_xlabel("Speed")
            ax.set_ylabel("Box Size")
            ax.set_xticks(np.arange(len(speeds)))
            ax.set_xticklabels([str(item) for item in speeds])
            ax.set_yticks(np.arange(len(box_sizes)))
            ax.set_yticklabels([str(item) for item in box_sizes])
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    ax.text(x, y, f"{data[y, x]:.2f}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(output_dir / f"multi_no_overlap_noise_{noise_prob:g}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-object no-overlap motion direction sweep.")
    parser.add_argument("--output_dir", type=str, default="results/motion_multi_object_no_overlap_sweep")
    parser.add_argument("--speeds", type=str, default="1 2")
    parser.add_argument("--box_sizes", type=str, default="3 5 7")
    parser.add_argument("--noise_probs", type=str, default="0 0.00025 0.0005")
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=100)
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
    total = len(speeds) * len(box_sizes) * len(noise_probs)
    done = 0
    for noise_prob in noise_probs:
        for box_size in box_sizes:
            for speed in speeds:
                done += 1
                print(f"[{done}/{total}] speed={speed} box_size={box_size} noise={noise_prob:g}", flush=True)
                rows = run_case(
                    device,
                    mapping,
                    speed,
                    box_size,
                    noise_prob,
                    args.num_frames,
                    args.height,
                    args.width,
                    args.warmup,
                    seed=4000 + int(noise_prob * 1_000_000) + box_size * 10 + speed,
                )
                for row in rows:
                    row["passed"] = row["correct_rate"] >= args.correct_thr and row["hit_rate"] >= args.hit_thr
                case_rows.extend(rows)
                failed = [row["object"] for row in rows if not row["passed"]]
                aggregate_rows.append(
                    {
                        "speed": speed,
                        "box_size": box_size,
                        "noise_prob": noise_prob,
                        "min_correct_rate": min(row["correct_rate"] for row in rows),
                        "mean_correct_rate": float(np.mean([row["correct_rate"] for row in rows])),
                        "min_hit_rate": min(row["hit_rate"] for row in rows),
                        "mean_hit_rate": float(np.mean([row["hit_rate"] for row in rows])),
                        "failed_objects": " ".join(failed),
                        "passed": len(failed) == 0,
                    }
                )

    write_csv(
        output_dir / "multi_no_overlap_cases.csv",
        case_rows,
        ["object", "direction", "speed", "box_size", "noise_prob", "expected_id", "stable_id", "correct_rate", "hit_rate", "passed"],
    )
    write_csv(
        output_dir / "multi_no_overlap_aggregate.csv",
        aggregate_rows,
        ["speed", "box_size", "noise_prob", "min_correct_rate", "mean_correct_rate", "min_hit_rate", "mean_hit_rate", "failed_objects", "passed"],
    )
    plot_summary(output_dir, aggregate_rows, speeds, box_sizes, noise_probs)

    passing = [row for row in aggregate_rows if row["passed"]]
    failing = [row for row in aggregate_rows if not row["passed"]]
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"device: {device}\n")
        f.write(f"mapping: {mapping}\n")
        f.write(f"objects: {OBJECTS}\n")
        f.write(f"speeds: {speeds}\n")
        f.write(f"box_sizes: {box_sizes}\n")
        f.write(f"noise_probs: {noise_probs}\n")
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
