# -*- coding: utf-8 -*-
"""Consistency and shape tests for MotionFrontend.

Coverage:
- Data loading from the current repository-local dataset root.
- Full pipeline and partial ablations (disable STP and/or motion stage).
- Explicit shape checks for all supported input formats:
    [T,H,W], [H,W,T], and [H,W].
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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "motVidarReal2025"


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


def parse_speed_list(speed_list_str):
    values = [v.strip() for v in speed_list_str.split(",") if v.strip()]
    if not values:
        raise ValueError("--speed_list must contain at least one positive integer")
    speeds = [int(v) for v in values]
    if any(v <= 0 for v in speeds):
        raise ValueError("--speed_list values must be positive")
    return speeds


def resolve_data_root(path_str):
    data_root = Path(path_str).expanduser()
    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root does not exist: {data_root}. "
            f"Expected location is {DEFAULT_DATA_ROOT}"
        )
    return data_root


def assert_shape(name, got, expected):
    if tuple(got) != tuple(expected):
        raise AssertionError(f"{name} shape mismatch: got={tuple(got)}, expected={tuple(expected)}")


def assert_io_shapes(frontend_kwargs, spikes_t_h_w, device, enable_motion, speed_list):
    """Check documented MotionFrontend input/output shapes explicitly."""
    t, h, w = spikes_t_h_w.shape
    num_channels = 8 * len(speed_list)

    # [T,H,W] input
    frontend = MotionFrontend(**frontend_kwargs)
    out_t = frontend.process(torch.from_numpy(spikes_t_h_w).to(device), return_intermediate=False, update_stdp=False)
    expected_t = (t, h, w, num_channels) if enable_motion else (t, h, w)
    assert_shape("default output for [T,H,W]", out_t.shape, expected_t)

    # [H,W,T] input
    frontend = MotionFrontend(**frontend_kwargs)
    spikes_h_w_t = np.transpose(spikes_t_h_w, (1, 2, 0))
    out_hwt = frontend.process(spikes_h_w_t, return_intermediate=False, update_stdp=False)
    expected_hwt = (t, h, w, num_channels) if enable_motion else (t, h, w)
    assert_shape("default output for [H,W,T]", out_hwt.shape, expected_hwt)

    # [H,W] single-frame input
    frontend = MotionFrontend(**frontend_kwargs)
    spikes_h_w = spikes_t_h_w[0]
    out_hw = frontend.process(spikes_h_w, return_intermediate=False, update_stdp=False)
    expected_hw = (1, h, w, num_channels) if enable_motion else (1, h, w)
    assert_shape("default output for [H,W]", out_hw.shape, expected_hw)

    # Intermediate output checks on [T,H,W]
    frontend = MotionFrontend(**frontend_kwargs)
    out_i = frontend.process(spikes_t_h_w, return_intermediate=True, update_stdp=False)
    assert_shape("intermediate frontend_spikes", out_i["frontend_spikes"].shape, (t, h, w))

    if enable_motion:
        assert_shape("intermediate motion_pattern", out_i["motion_pattern"].shape, (t, h, w, num_channels))
        assert_shape("intermediate motion_vector", out_i["motion_vector"].shape, (t, h, w, 2))
        assert_shape("intermediate motion_id", out_i["motion_id"].shape, (t, h, w))
        assert_shape("intermediate motion_vector_layer1", out_i["motion_vector_layer1"].shape, (t, h, w, 2))
    else:
        if out_i["motion_pattern"] is not None:
            raise AssertionError("motion_pattern must be None when motion stage is disabled")
        if out_i["motion_vector"] is not None:
            raise AssertionError("motion_vector must be None when motion stage is disabled")
        if out_i["motion_id"] is not None:
            raise AssertionError("motion_id must be None when motion stage is disabled")
        if out_i["motion_vector_layer1"] is not None:
            raise AssertionError("motion_vector_layer1 must be None when motion stage is disabled")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_idx", "-s", type=int, default=0)
    parser.add_argument("--data_path", "-d", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--label_type", "-l", type=str, default="tracking")
    parser.add_argument("--block_len", type=int, default=1000)
    parser.add_argument("--calibration_time", type=int, default=150)
    parser.add_argument("--scale_w", type=int, default=1)
    parser.add_argument("--scale_h", type=int, default=1)
    parser.add_argument("--disable_stp", action="store_true", help="Disable STPFilter stage")
    parser.add_argument("--disable_motion", action="store_true", help="Disable motion_estimation stage")
    parser.add_argument("--no_update_stdp", action="store_true", help="Disable STDP weight update during evaluation")
    parser.add_argument("--speed_list", type=str, default="1,2", help="Comma-separated positive speeds, e.g. 1,2,3")
    parser.add_argument("--max_eval_frames", type=int, default=0,
                        help="0 means use all frames after calibration")
    parser.add_argument("--atol_frontend", type=float, default=0.0)
    parser.add_argument("--atol_motion", type=float, default=1e-6)
    args = parser.parse_args()

    enable_stp = not args.disable_stp
    enable_motion = not args.disable_motion
    update_stdp = not args.no_update_stdp
    speed_list = parse_speed_list(args.speed_list)

    test_scene = [
        "spike59", "rotTrans", "cplCam", "cpl1", "badminton",
        "ball", "badminton-l1", "badminton-l2", "pingpong"
    ]
    data_name = test_scene[args.scene_idx]
    data_root = resolve_data_root(args.data_path)
    data_filename = os.path.join(data_root.as_posix(), data_name)

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
    print("ENABLE_STP:", enable_stp)
    print("ENABLE_MOTION:", enable_motion)
    print("UPDATE_STDP:", update_stdp)
    print("SPEED_LIST:", speed_list)
    print("CALIBRATION:", calibration_time)
    print("EVAL FRAMES:", eval_frames)

    frontend_kwargs = {
        "spike_h": spike_h,
        "spike_w": spike_w,
        "device": device,
        "diff_time": 1,
        "enable_stp": enable_stp,
        "enable_motion": enable_motion,
        "speed_list": speed_list,
    }

    # Explicitly validate the documented IO shape contract before value consistency checks.
    shape_frames = min(max(eval_frames, 1), 4)
    shape_start = calibration_time if calibration_time + shape_frames <= spikes.shape[0] else 0
    shape_slice = spikes[shape_start:shape_start + shape_frames]
    assert_io_shapes(
        frontend_kwargs=frontend_kwargs,
        spikes_t_h_w=shape_slice,
        device=device,
        enable_motion=enable_motion,
        speed_list=speed_list,
    )
    print("Shape checks: PASS")

    # Baseline modules built only for enabled stages.
    baseline_stp = STPFilter(spike_h, spike_w, device, diff_time=1) if enable_stp else None
    baseline_motion = (
        motion_estimation(spike_h, spike_w, device, logger=_NullLogger(), speed_list=speed_list)
        if enable_motion
        else None
    )

    # Frontend under test.
    frontend = MotionFrontend(**frontend_kwargs)

    # Calibration is meaningful only when STP stage is enabled.
    if enable_stp:
        print("Begin calibration...")
        with torch.no_grad():
            for t in range(calibration_time):
                frame = to_binary_float_tensor(spikes[t], device)
                baseline_stp.update_dynamics(t, frame)
                frontend.stp_filter.update_dynamics(t, frame)

    # Keep frontend internal timestamp aligned with baseline timeline.
    frontend.timestamp = calibration_time

    stats_frontend = init_stats()
    stats_default = init_stats()
    stats_motion_id = init_stats() if enable_motion else None
    stats_motion_vec = init_stats() if enable_motion else None
    stats_motion_vec_l1 = init_stats() if enable_motion else None
    stats_motion_pattern = init_stats() if enable_motion else None

    print("Begin consistency comparison...")
    with torch.no_grad():
        for i in range(eval_frames):
            ts = calibration_time + i
            frame = to_binary_float_tensor(spikes[ts], device)

            # Explicit baseline path for enabled stages.
            baseline_stage_spikes = frame
            if enable_stp:
                baseline_stp.update_dynamics(ts, frame)
                baseline_stp.local_connect(baseline_stp.filter_spk)
                baseline_stage_spikes = baseline_stp.lif_spk

            ref_frontend_spikes = baseline_stage_spikes.detach().clone()

            if enable_motion:
                if update_stdp:
                    baseline_motion.stdp_tracking(baseline_stage_spikes)
                ref_motion_id, ref_motion_vec, ref_motion_vec_l1, ref_motion_pattern = baseline_motion.local_wta(
                    baseline_stage_spikes,
                    ts,
                    visualize=False,
                    return_pattern_map=True,
                )

            # MotionFrontend path.
            out = frontend.process(
                frame,
                start_timestamp=None,
                visualize=False,
                update_stdp=update_stdp,
                return_intermediate=True,
                as_numpy=False,
            )

            test_frontend_spikes = out["frontend_spikes"][0]

            update_stats(stats_frontend, ref_frontend_spikes, test_frontend_spikes, args.atol_frontend)

            if enable_motion:
                test_motion_id = out["motion_id"][0]
                test_motion_vec = out["motion_vector"][0]
                test_motion_vec_l1 = out["motion_vector_layer1"][0]
                test_motion_pattern = out["motion_pattern"][0]

                update_stats(stats_motion_id, ref_motion_id.float(), test_motion_id.float(), args.atol_frontend)
                update_stats(stats_motion_vec, ref_motion_vec, test_motion_vec, args.atol_motion)
                update_stats(stats_motion_vec_l1, ref_motion_vec_l1, test_motion_vec_l1, args.atol_motion)
                update_stats(stats_motion_pattern, ref_motion_pattern, test_motion_pattern, args.atol_frontend)

                # Default output for motion-enabled mode should be pattern map.
                update_stats(stats_default, ref_motion_pattern, test_motion_pattern, args.atol_frontend)
            else:
                # Default output for motion-disabled mode should be frontend spikes.
                update_stats(stats_default, ref_frontend_spikes, test_frontend_spikes, args.atol_frontend)

            if (i + 1) % 100 == 0 or (i + 1) == eval_frames:
                print(f"Compared {i + 1}/{eval_frames} frames")

    print("\n=== Consistency Summary ===")
    summarize_stats("frontend_spikes", stats_frontend)
    summarize_stats("default_output", stats_default)
    if enable_motion:
        summarize_stats("motion_id", stats_motion_id)
        summarize_stats("motion_vector", stats_motion_vec)
        summarize_stats("motion_vector_layer1", stats_motion_vec_l1)
        summarize_stats("motion_pattern", stats_motion_pattern)

    passed = stats_frontend["mismatch"] == 0 and stats_default["mismatch"] == 0
    if enable_motion:
        passed = (
            passed
            and stats_motion_id["mismatch"] == 0
            and stats_motion_vec["mismatch"] == 0
            and stats_motion_vec_l1["mismatch"] == 0
            and stats_motion_pattern["mismatch"] == 0
        )

    print("\nCONSISTENCY RESULT:", "PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
