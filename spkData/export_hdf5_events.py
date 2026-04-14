import argparse
from pathlib import Path

import h5py
import numpy as np


def _to_numpy(frames):
    if hasattr(frames, "detach"):
        frames = frames.detach().cpu().numpy()
    return np.asarray(frames)


def spike_frames_to_events(
    frames,
    *,
    dt=1,
    start_time=0,
    threshold=0,
    pol_value=1,
    flipud=False,
    one_based_xy=False,
):
    """Convert binary spike frames into DVS-style event lists.

    Args:
        frames: Spike frame array with shape [H, W] or [T, H, W].
        dt: Timestamp interval between adjacent frames.
        start_time: Timestamp assigned to frame 0.
        threshold: Pixels with values greater than this value become events.
        pol_value: Polarity value written for every event. Current SNNTracker
            spike frames do not retain ON/OFF polarity, so this is constant.
        flipud: If true, export y as H - 1 - row.
        one_based_xy: If true, export x/y as 1-based coordinates.

    Returns:
        Tuple (x, y, t, pol), each a 1-D numpy array.
    """
    frames_np = _to_numpy(frames)
    if frames_np.ndim == 2:
        frames_np = frames_np[np.newaxis, :, :]
    if frames_np.ndim != 3:
        raise ValueError(f"Expected frames with shape [H, W] or [T, H, W], got {frames_np.shape}")

    _, height, _ = frames_np.shape
    frame_idx, row_idx, col_idx = np.nonzero(frames_np > threshold)

    x = col_idx.astype(np.int64, copy=False)
    y = row_idx.astype(np.int64, copy=False)
    if flipud:
        y = (height - 1 - y).astype(np.int64, copy=False)
    if one_based_xy:
        x = x + 1
        y = y + 1

    t = start_time + frame_idx * dt
    if float(dt).is_integer() and float(start_time).is_integer():
        t = t.astype(np.int64, copy=False)
    else:
        t = t.astype(np.float64, copy=False)

    pol = np.full(x.shape, pol_value, dtype=np.int8)
    return x, y, t, pol


def write_event_hdf5(
    output_path,
    frames,
    *,
    dt=1,
    start_time=0,
    threshold=0,
    pol_value=1,
    flipud=False,
    one_based_xy=False,
    compression=None,
):
    """Write spike frames as an HDF5 event file with x/y/t/pol datasets."""
    frames_np = _to_numpy(frames)
    if frames_np.ndim == 2:
        shape = (1,) + frames_np.shape
    elif frames_np.ndim == 3:
        shape = frames_np.shape
    else:
        raise ValueError(f"Expected frames with shape [H, W] or [T, H, W], got {frames_np.shape}")

    x, y, t, pol = spike_frames_to_events(
        frames_np,
        dt=dt,
        start_time=start_time,
        threshold=threshold,
        pol_value=pol_value,
        flipud=flipud,
        one_based_xy=one_based_xy,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("x", data=x, compression=compression)
        h5f.create_dataset("y", data=y, compression=compression)
        h5f.create_dataset("t", data=t, compression=compression)
        h5f.create_dataset("pol", data=pol, compression=compression)
        h5f.attrs["source_format"] = "dense_spike_frames"
        h5f.attrs["frames_shape"] = shape
        h5f.attrs["dt"] = dt
        h5f.attrs["start_time"] = start_time
        h5f.attrs["threshold"] = threshold
        h5f.attrs["pol_value"] = pol_value
        h5f.attrs["flipud"] = int(flipud)
        h5f.attrs["one_based_xy"] = int(one_based_xy)

    return {
        "output_path": str(output_path),
        "num_events": int(x.shape[0]),
        "frames_shape": tuple(int(v) for v in shape),
    }


def _main():
    parser = argparse.ArgumentParser(
        description="Export dense spike frames [T,H,W] or [H,W] to HDF5 x/y/t/pol event lists."
    )
    parser.add_argument("input_npy", help="Input .npy file containing [T,H,W] or [H,W] spike frames.")
    parser.add_argument("output_h5", help="Output HDF5 file.")
    parser.add_argument("--dt", type=float, default=1, help="Timestamp interval between adjacent frames.")
    parser.add_argument("--start-time", type=float, default=0, help="Timestamp assigned to frame 0.")
    parser.add_argument("--threshold", type=float, default=0, help="Export pixels with value > threshold.")
    parser.add_argument("--pol-value", type=int, default=1, help="Constant polarity value to write.")
    parser.add_argument("--flipud", action="store_true", help="Export y as H - 1 - row.")
    parser.add_argument("--one-based-xy", action="store_true", help="Export x/y as 1-based coordinates.")
    parser.add_argument("--gzip", action="store_true", help="Use gzip compression for datasets.")
    args = parser.parse_args()

    frames = np.load(args.input_npy)
    summary = write_event_hdf5(
        args.output_h5,
        frames,
        dt=args.dt,
        start_time=args.start_time,
        threshold=args.threshold,
        pol_value=args.pol_value,
        flipud=args.flipud,
        one_based_xy=args.one_based_xy,
        compression="gzip" if args.gzip else None,
    )
    print(summary)


if __name__ == "__main__":
    _main()
