"""Validate btorch-style preprocessing and postprocessing contracts.

This script does not build or run a simulator. It checks that strict GU
preprocessing produces the input object a btorch/python simulator should
consume, then feeds a simulator-output-shaped spike tensor into postprocessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spkData.export_hdf5_events import write_event_hdf5  # noqa: E402
from translation.btorch_interface import (  # noqa: E402
    build_gu_2018_strict_btorch_bundle,
    postprocess_btorch_simulator_output,
    summarize_bundle,
)
from translation.detection_readout import save_detections_json  # noqa: E402
from translation.spikenet_preprocessing import GU2018Config  # noqa: E402


def make_moving_box(frames: int, height: int, width: int) -> np.ndarray:
    data = np.zeros((frames, height, width), dtype=np.uint8)
    box_h = max(2, height // 5)
    box_w = max(2, width // 5)
    for t in range(frames):
        y0 = int(round(t * max(height - box_h, 1) / max(frames - 1, 1)))
        x0 = int(round(t * max(width - box_w, 1) / max(frames - 1, 1)))
        data[t, y0 : y0 + box_h, x0 : x0 + box_w] = 1
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("translation/debug_outputs/btorch_interface"))
    parser.add_argument("--lattice-shape", nargs=2, type=int, default=(8, 10), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--n-i", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--p-scale", type=float, default=0.2)
    parser.add_argument("--common-neighbor-iterations", type=int, default=2)
    parser.add_argument("--connection-device", default="auto")
    parser.add_argument("--current-scale", type=float, default=20.0)
    parser.add_argument("--simulator-output-npy", type=Path)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--min-area", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = int(args.lattice_shape[0]), int(args.lattice_shape[1])

    event_h5 = out_dir / f"synthetic_{height}x{width}_events.h5"
    write_event_hdf5(event_h5, make_moving_box(args.frames, height, width), dt=1, start_time=0)

    p_base = np.asarray([[0.16, 0.2], [0.2, 0.4]], dtype=float) * float(args.p_scale)
    cfg = GU2018Config(
        lattice_shape=(height, width),
        n_i=args.n_i,
        step_tot=args.frames,
        p_mat=((float(p_base[0, 0]), float(p_base[0, 1])), (float(p_base[1, 0]), float(p_base[1, 1]))),
        common_neighbor_iterations=args.common_neighbor_iterations,
        connection_device=args.connection_device,
    )
    bundle = build_gu_2018_strict_btorch_bundle(
        event_h5,
        config=cfg,
        input_pop=0,
        mapping="gu_column_major",
        current_scale=args.current_scale,
        batch_size=1,
        seed=args.seed,
    )

    if args.simulator_output_npy:
        simulator_output = np.load(args.simulator_output_npy)
    else:
        simulator_output = (bundle.input_current > 0).astype(np.float32)

    activity, detections = postprocess_btorch_simulator_output(
        simulator_output,
        bundle,
        grid_shape=(height, width),
        pop_index=0,
        mapping="gu_column_major",
        window=args.window,
        stride=args.stride,
        threshold=args.threshold,
        min_area=args.min_area,
    )
    det_json = save_detections_json(detections, out_dir / "btorch_detections.json")

    summary = summarize_bundle(bundle)
    summary.update(
        {
            "event_h5": str(event_h5),
            "simulator_input_keys": sorted(bundle.simulator_kwargs().keys()),
            "simulator_output_shape": list(np.asarray(simulator_output).shape),
            "simulator_output_nonzero": int(np.count_nonzero(simulator_output)),
            "activity_shape": list(activity.shape),
            "detections_json": str(det_json),
            "num_detections": len(detections),
        }
    )
    summary_path = out_dir / "debug_btorch_interface_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
