"""Debug MOT Vidar data through the pre-attention SNNTracker path.

This reuses the same loading and STP/LIF pre-processing used by
``test_snntracker.py`` before the attention layer:

    SpikeStream.get_block_spikes
    -> STPFilter.update_dynamics
    -> STPFilter.local_connect
    -> lif_spk  # input to SaccadeInput/update_dnf/get_attention_location

The resulting ``lif_spk`` frames are exported with ``spkData/export_hdf5_events.py``
and then checked by ``translation/debug_preprocessing_flow.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import path as path_utils  # noqa: E402
from spkData.export_hdf5_events import write_event_hdf5  # noqa: E402
from spkData.load_dat import SpikeStream, data_parameter_dict  # noqa: E402
from spkProc.detection.attention_select import SaccadeInput  # noqa: E402
from spkProc.filters.stp_filters_torch import STPFilter  # noqa: E402
from translation.debug_preprocessing_flow import (  # noqa: E402
    Check,
    Report,
    check_population_match,
    inspect_event_file,
    inspect_in_file,
    save_event_preview,
    h5_tree,
    write_json_report,
)
from translation.spikenet_preprocessing import ChenGong2019Config, build_chen_gong_2019_input  # noqa: E402


SCENES = [
    "spike59",
    "rotTrans",
    "cplCam",
    "cpl1",
    "badminton",
    "ball",
    "badminton-l1",
    "badminton-l2",
    "pingpong",
]


def downscale_input(spikes: np.ndarray, scale_w: int, scale_h: int) -> np.ndarray:
    if spikes.ndim != 3:
        raise ValueError(f"Expected spikes [T,H,W], got {spikes.shape}")
    return spikes[:, ::scale_h, ::scale_w]


def resize_binary_frames(frames: np.ndarray, size: int) -> np.ndarray:
    resized = np.zeros((frames.shape[0], size, size), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        resized[idx] = cv2.resize(frame.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return resized


def collect_pre_attention_frames(
    spikes: np.ndarray,
    *,
    spike_h: int,
    spike_w: int,
    calibration_time: int,
    export_frames: int,
    attention_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    stp_filter = STPFilter(spike_h, spike_w, device)
    saccade = SaccadeInput(spike_h, spike_w, box_size=attention_size, device=device)

    total_needed = calibration_time + export_frames
    if spikes.shape[0] < total_needed:
        raise ValueError(f"Need {total_needed} frames, got {spikes.shape[0]}")

    for t in range(calibration_time):
        input_spk = torch.from_numpy(spikes[t]).to(device)
        stp_filter.update_dynamics(t, input_spk)

    lif_frames = np.zeros((export_frames, spike_h, spike_w), dtype=np.uint8)
    attention_box_counts: list[int] = []
    lif_counts: list[int] = []
    raw_counts: list[int] = []

    for out_idx, t in enumerate(range(calibration_time, calibration_time + export_frames)):
        input_spk = torch.from_numpy(spikes[t]).to(device)
        stp_filter.update_dynamics(t, input_spk)
        stp_filter.local_connect(stp_filter.filter_spk)
        lif_spk = stp_filter.lif_spk.detach().cpu().numpy().astype(np.uint8)
        lif_frames[out_idx] = lif_spk
        lif_counts.append(int(lif_spk.sum()))
        raw_counts.append(int(spikes[t].sum()))

        # This is only for diagnostics. The exported replacement input is still
        # the pre-attention lif_spk tensor.
        saccade.update_dnf(stp_filter.lif_spk)
        attention_box, _ = saccade.get_attention_location(stp_filter.lif_spk)
        attention_box_counts.append(int(attention_box.shape[0]))

    stats = {
        "raw_spike_count_min": int(np.min(raw_counts)) if raw_counts else 0,
        "raw_spike_count_max": int(np.max(raw_counts)) if raw_counts else 0,
        "raw_spike_count_mean": float(np.mean(raw_counts)) if raw_counts else 0.0,
        "lif_spike_count_min": int(np.min(lif_counts)) if lif_counts else 0,
        "lif_spike_count_max": int(np.max(lif_counts)) if lif_counts else 0,
        "lif_spike_count_mean": float(np.mean(lif_counts)) if lif_counts else 0.0,
        "attention_box_count_min": int(np.min(attention_box_counts)) if attention_box_counts else 0,
        "attention_box_count_max": int(np.max(attention_box_counts)) if attention_box_counts else 0,
        "attention_box_count_mean": float(np.mean(attention_box_counts)) if attention_box_counts else 0.0,
    }
    return lif_frames, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("motVidarReal2025"))
    parser.add_argument("--scene-idx", type=int, default=0)
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--label-type", default="tracking")
    parser.add_argument("--block-len", type=int, default=220)
    parser.add_argument("--calibration-time", type=int, default=150)
    parser.add_argument("--export-frames", type=int, default=40)
    parser.add_argument("--attention-size", type=int, default=15)
    parser.add_argument("--scale-w", type=int, default=1)
    parser.add_argument("--scale-h", type=int, default=1)
    parser.add_argument("--spikenet-size", type=int, default=32)
    parser.add_argument("--square-debug", action="store_true", help="Also build a nearest-neighbor square-grid diagnostic input.")
    parser.add_argument("--max-build-neurons", type=int, default=5000, help="Skip full in.h5 connectivity build above this grid area.")
    parser.add_argument("--step-tot", type=int, default=25000)
    parser.add_argument("--connection-backend", choices=("auto", "torch", "numpy"), default="auto")
    parser.add_argument("--connection-device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--output-dir", type=Path, default=Path("translation/debug_outputs/mot_attention"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene = args.scene or SCENES[args.scene_idx]
    data_filename = args.data_path / scene
    para_dict = data_parameter_dict(str(data_filename), args.label_type)
    stream = SpikeStream(**para_dict)

    block_len = max(args.block_len, args.calibration_time + args.export_frames)
    spikes = stream.get_block_spikes(begin_idx=0, block_len=block_len)
    spikes = downscale_input(spikes, args.scale_w, args.scale_h)
    spike_h = int(spikes.shape[1])
    spike_w = int(spikes.shape[2])
    report.pass_("dataset loaded", f"{scene}: spikes shape={spikes.shape}, HxW={spike_h}x{spike_w}")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    lif_frames, pre_attention_stats = collect_pre_attention_frames(
        spikes,
        spike_h=spike_h,
        spike_w=spike_w,
        calibration_time=args.calibration_time,
        export_frames=args.export_frames,
        attention_size=args.attention_size,
        device=device,
    )
    report.pass_("pre-attention frames", f"lif_spk shape={lif_frames.shape}")
    report.pass_("pre-attention stats", json.dumps(pre_attention_stats, ensure_ascii=False))

    native_event = args.output_dir / f"{scene}_pre_attention_native.h5"
    write_event_hdf5(native_event, lif_frames, dt=1, start_time=args.calibration_time, threshold=0, pol_value=1)
    native_summary = inspect_event_file(native_event, report)
    native_preview = args.output_dir / f"{scene}_pre_attention_native_preview.png"
    save_event_preview(native_event, native_preview)
    report.pass_("native pre-attention preview", str(native_preview))

    if native_summary is not None and native_summary["num_events"] > 0:
        x_ok_native = 0 <= native_summary["x_min"] and native_summary["x_max"] < spike_w
        y_ok_native = 0 <= native_summary["y_min"] and native_summary["y_max"] < spike_h
        if x_ok_native and y_ok_native:
            report.pass_("native event/native frame match", f"x within [0,{spike_w - 1}], y within [0,{spike_h - 1}]")
        else:
            report.fail(
                "native event/native frame match",
                f"x=[{native_summary['x_min']},{native_summary['x_max']}], "
                f"y=[{native_summary['y_min']},{native_summary['y_max']}], frame={spike_h}x{spike_w}",
            )
        check_population_match(native_summary, (spike_h, spike_w), report)
    elif native_summary is not None:
        report.warn("native event/native frame match", "zero-event file cannot validate coordinate coverage")
        check_population_match(native_summary, (spike_h, spike_w), report)

    in_summary = None
    if spike_h * spike_w <= args.max_build_neurons:
        in_path = args.output_dir / f"{scene}_{spike_h}x{spike_w}_debug_in.h5"
        cfg = ChenGong2019Config(
            grid_shape=(spike_h, spike_w),
            step_tot=args.step_tot,
            connection_backend=args.connection_backend,
            connection_device=args.connection_device,
        )
        build_chen_gong_2019_input(in_path, native_event, native_event, config=cfg, loop_num=1, seed=1)
        report.pass_("translation preprocessing build", str(in_path))
        in_summary = inspect_in_file(in_path, native_event, native_event, report)
        tree_path = args.output_dir / f"{scene}_{spike_h}x{spike_w}_debug_in_tree.txt"
        tree_path.write_text("\n".join(h5_tree(in_path)), encoding="utf-8")
        report.pass_("translation in.h5 tree", str(tree_path))
    else:
        report.warn(
            "translation preprocessing build",
            f"skipped full connectivity build for {spike_h}x{spike_w}={spike_h * spike_w} grid cells; "
            f"use --scale-w/--scale-h or raise --max-build-neurons for a heavier run",
        )

    square_summary = None
    if args.square_debug:
        square_frames = resize_binary_frames(lif_frames, args.spikenet_size)
        square_event = args.output_dir / f"{scene}_pre_attention_square{args.spikenet_size}.h5"
        write_event_hdf5(square_event, square_frames, dt=1, start_time=args.calibration_time, threshold=0, pol_value=1)
        square_summary = inspect_event_file(square_event, report)
        square_preview = args.output_dir / f"{scene}_pre_attention_square{args.spikenet_size}_preview.png"
        save_event_preview(square_event, square_preview)
        report.pass_("square pre-attention preview", str(square_preview))
        if square_summary is not None:
            check_population_match(square_summary, (args.spikenet_size, args.spikenet_size), report)

        square_in_path = args.output_dir / f"{scene}_square{args.spikenet_size}_debug_in.h5"
        square_cfg = ChenGong2019Config(
            gsize=args.spikenet_size,
            step_tot=args.step_tot,
            connection_backend=args.connection_backend,
            connection_device=args.connection_device,
        )
        build_chen_gong_2019_input(square_in_path, square_event, square_event, config=square_cfg, loop_num=1, seed=1)
        report.pass_("square translation preprocessing build", str(square_in_path))

    payload = {
        "scene": scene,
        "data_filename": str(data_filename),
        "para_dict": {k: str(v) if isinstance(v, Path) else v for k, v in para_dict.items()},
        "pre_attention_stats": pre_attention_stats,
        "native_event": native_summary,
        "square_event": square_summary,
        "in_h5": in_summary,
        "checks": [check.__dict__ for check in report.checks],
    }
    report_path = args.output_dir / f"{scene}_mot_attention_preprocessing_report.json"
    write_json_report(report_path, payload)
    report.pass_("json report", str(report_path))
    report.print()
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
