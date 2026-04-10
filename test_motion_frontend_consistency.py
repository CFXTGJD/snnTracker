# -*- coding: utf-8 -*-
"""Consistency test between MotionFrontend and explicit STP+motion pipeline.

This script follows the same data-loading and time scheduling style as
test_snntracker.py and validates that MotionFrontend produces the same
frontend/motion outputs as manually chaining STPFilter + motion_estimation.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from pprint import pprint

from spkData.load_dat import data_parameter_dict, SpikeStream
from spkProc.filters.stp_filters_torch import STPFilter
from spkProc.motion.motion_detection import motion_estimation
from motion_frontend.motion_frontend import MotionFrontend


class _NullLogger:
    """Logger stub to keep motion_estimation API unchanged in test mode."""

    def add_image(self, *args, **kwargs):
        return


def downscale_input(spikes, scale_w, scale_h):
    """Simple spatial downsampling used in test_snntracker.py."""
    if len(spikes.shape) == 3:
        return spikes[:, ::scale_h, ::scale_w]
    if len(spikes.shape) == 4:
        return spikes[:, ::scale_h, ::scale_w, :]
    raise ValueError(f"Unsupported spikes shape: {spikes.shape}")


def to_binary_float_tensor(frame_np, device):
    """Convert one frame to device tensor in {0,1} float format."""
    frame = torch.from_numpy(frame_np).to(device)
    if not torch.is_floating_point(frame):
        frame = frame.float()
    return (frame > 0).float()


def update_stats(stats, ref_tensor, test_tensor, atol):
    """Accumulate mismatch counters and absolute error metrics."""
    diff = (ref_tensor - test_tensor).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    neq = int((diff > atol).sum().item())

    stats["numel"] += int(diff.numel())
    stats["mismatch"] += neq
    stats["sum_abs"] += float(diff.sum().item())
    stats["max_abs"] = max(stats["max_abs"], max_abs)
    stats["max_mean_abs_per_frame"] = max(stats["max_mean_abs_per_frame"], mean_abs)


def init_stats():
    return {
        "numel": 0,
        "mismatch": 0,
        "sum_abs": 0.0,
        "max_abs": 0.0,
        "max_mean_abs_per_frame": 0.0,
    }


def summarize_stats(name, stats):
    numel = max(stats["numel"], 1)
    mismatch_ratio = stats["mismatch"] / numel
    mean_abs_global = stats["sum_abs"] / numel
    print(f"[{name}] numel={stats['numel']}, mismatch={stats['mismatch']}, mismatch_ratio={mismatch_ratio:.6e}")
    print(f"[{name}] max_abs={stats['max_abs']:.6e}, mean_abs_global={mean_abs_global:.6e}, max_mean_abs_per_frame={stats['max_mean_abs_per_frame']:.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_idx", "-s", type=int, default=0)
    parser.add_argument("--data_path", "-d", type=str, default="/home/wangxukang/motVidarReal2025/")
    parser.add_argument("--label_type", "-l", type=str, default="tracking")
    parser.add_argument("--block_len", type=int, default=1000)
    parser.add_argument("--calibration_time", type=int, default=150)
    parser.add_argument("--scale_w", type=int, default=1)
    parser.add_argument("--scale_h", type=int, default=1)
    parser.add_argument("--max_eval_frames", type=int, default=0,
                        help="0 means use all frames after calibration")
    parser.add_argument("--atol_frontend", type=float, default=0.0)
    parser.add_argument("--atol_motion", type=float, default=1e-6)
    args = parser.parse_args()

    test_scene = [
        "spike59", "rotTrans", "cplCam", "cpl1", "badminton",
        "ball", "badminton-l1", "badminton-l2", "pingpong"
    ]
    data_name = test_scene[args.scene_idx]
    data_filename = os.path.join(args.data_path, data_name)

    para_dict = data_parameter_dict(data_filename, args.label_type)
    pprint(para_dict)
    vidar_spikes = SpikeStream(**para_dict)
    spikes = vidar_spikes.get_block_spikes(begin_idx=0, block_len=args.block_len)

    spikes = downscale_input(spikes, args.scale_w, args.scale_h)
    spike_h = int(para_dict.get("spike_h") // args.scale_h)
    spike_w = int(para_dict.get("spike_w") // args.scale_w)

    calibration_time = int(args.calibration_time)
    if calibration_time < 0 or calibration_time > spikes.shape[0]:
        raise ValueError("calibration_time is out of valid range")

    available_eval = int(spikes.shape[0] - calibration_time)
    eval_frames = available_eval if args.max_eval_frames <= 0 else min(available_eval, int(args.max_eval_frames))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    print("DATA:", data_filename)
    print("SPIKES SHAPE:", spikes.shape)
    print("CALIBRATION:", calibration_time)
    print("EVAL FRAMES:", eval_frames)

    # Baseline: explicit STPFilter + motion_estimation.
    baseline_stp = STPFilter(spike_h, spike_w, device, diff_time=1)
    baseline_motion = motion_estimation(spike_h, spike_w, device, logger=_NullLogger())

    # Frontend under test.
    frontend = MotionFrontend(
        spike_h=spike_h,
        spike_w=spike_w,
        device=device,
        diff_time=1,
        enable_stp=True,
        enable_motion=True,
    )

    # Calibration uses the same schedule as SNNTracker.calibrate_motion:
    # update_dynamics only, without local_connect and motion calls.
    print("Begin calibration...")
    with torch.no_grad():
        for t in range(calibration_time):
            frame = to_binary_float_tensor(spikes[t], device)
            baseline_stp.update_dynamics(t, frame)
            frontend.stp_filter.update_dynamics(t, frame)

    # Keep frontend internal timestamp aligned with baseline timeline.
    frontend.timestamp = calibration_time

    stats_frontend = init_stats()
    stats_motion_id = init_stats()
    stats_motion_vec = init_stats()
    stats_motion_vec_l1 = init_stats()

    print("Begin consistency comparison...")
    with torch.no_grad():
        for i in range(eval_frames):
            ts = calibration_time + i
            frame = to_binary_float_tensor(spikes[ts], device)

            # Explicit baseline path.
            baseline_stp.update_dynamics(ts, frame)
            baseline_stp.local_connect(baseline_stp.filter_spk)

            # Keep scheduling consistent with the original tracker path:
            # motion_estimation consumes the same reusable lif_spk tensor.
            baseline_motion_input = baseline_stp.lif_spk
            ref_frontend_spikes = baseline_motion_input.detach().clone()

            baseline_motion.stdp_tracking(baseline_motion_input)
            ref_motion_id, ref_motion_vec, ref_motion_vec_l1 = baseline_motion.local_wta(
                baseline_motion_input, ts, visualize=False
            )

            # MotionFrontend path.
            out = frontend.process(
                frame,
                start_timestamp=None,
                visualize=False,
                update_stdp=True,
                return_intermediate=True,
                as_numpy=False,
            )

            test_frontend_spikes = out["frontend_spikes"][0]
            test_motion_id = out["motion_id"][0]
            test_motion_vec = out["motion_vector"][0]
            test_motion_vec_l1 = out["motion_vector_layer1"][0]

            update_stats(stats_frontend, ref_frontend_spikes, test_frontend_spikes, args.atol_frontend)
            update_stats(stats_motion_id, ref_motion_id.float(), test_motion_id.float(), args.atol_frontend)
            update_stats(stats_motion_vec, ref_motion_vec, test_motion_vec, args.atol_motion)
            update_stats(stats_motion_vec_l1, ref_motion_vec_l1, test_motion_vec_l1, args.atol_motion)

            if (i + 1) % 100 == 0 or (i + 1) == eval_frames:
                print(f"Compared {i + 1}/{eval_frames} frames")

    print("\n=== Consistency Summary ===")
    summarize_stats("frontend_spikes", stats_frontend)
    summarize_stats("motion_id", stats_motion_id)
    summarize_stats("motion_vector", stats_motion_vec)
    summarize_stats("motion_vector_layer1", stats_motion_vec_l1)

    passed = (
        stats_frontend["mismatch"] == 0
        and stats_motion_id["mismatch"] == 0
        and stats_motion_vec["mismatch"] == 0
        and stats_motion_vec_l1["mismatch"] == 0
    )

    print("\nCONSISTENCY RESULT:", "PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
