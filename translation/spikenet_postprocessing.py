"""Post-processing utilities for SpikeNet-style simulator outputs.

This module implements the Python-side equivalent of the minimal
``ReadH5/ReadYG + AnalyseYG + SaveRYG`` path needed by ``translation/agent.md``:

* read SpikeNet ``*_out.h5`` population spike outputs;
* reconstruct dense spike/activity frames on a rectangular grid;
* integrate activity in time windows;
* extract connected-component detections as bounding boxes;
* save JSON/NPZ/MOT-style outputs for debugging and evaluation.

Inverse-pool is intentionally not used here. It belongs to preprocessing
weight assignment before ``writeChemicalConnectionHDF5`` writes ``in.h5``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translation.detection_readout import (  # noqa: E402
    BBoxDetection as Detection,
    detect_bboxes_from_windows,
    integrate_activity_windows,
    save_detections_json as _save_detections_json,
    save_mot_txt as _save_mot_txt,
)


def _read_optional(h5f: h5py.File, path: str, default=None):
    return h5f[path][()] if path in h5f else default


def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar(value[()])
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"} and value.size == 1:
        return _decode_scalar(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_1d(value, dtype=None) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _find_first_dataset(h5f: h5py.File, candidates: Iterable[str]) -> str | None:
    for path in candidates:
        if path in h5f:
            return path
    return None


def _list_datasets(h5f: h5py.File) -> list[str]:
    names: list[str] = []

    def visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            names.append("/" + name)

    h5f.visititems(visitor)
    return names


@dataclass
class PopulationOutput:
    pop_index: int
    n_neurons: int
    spike_hist_tot: np.ndarray
    num_spikes_pop: np.ndarray
    dense_spike_hist: np.ndarray | None = None
    rate_hz: np.ndarray | None = None


@dataclass
class SpikeNetOutput:
    out_path: str
    config_path: str | None
    dt: float | None
    step_tot: int
    n: list[int]
    populations: list[PopulationOutput]
    available_datasets: list[str]


def make_population_index_maps(grid_shape: tuple[int, int]) -> list[np.ndarray]:
    """Return grid-shaped maps from y/x coordinate to pop-local neuron index."""
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    grid = np.zeros((grid_h, grid_w), dtype=np.int8)
    grid[1::2, 1::2] = 1
    maps: list[np.ndarray] = []
    for pop_zero in (0, 1):
        pop_map = np.full((grid_h, grid_w), -1, dtype=np.int64)
        flat = np.flatnonzero(grid.ravel(order="F") == pop_zero)
        rows, cols = np.unravel_index(flat, grid.shape, order="F")
        pop_map[rows, cols] = np.arange(flat.size, dtype=np.int64)
        maps.append(pop_map)
    return maps


def pop_local_to_yx(pop_index: int, neuron_ids: np.ndarray, grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    maps = make_population_index_maps(grid_shape)
    if pop_index >= len(maps):
        raise ValueError(f"Only pop0/pop1 grid maps are supported, got pop{pop_index}")
    pop_map = maps[pop_index]
    coords = np.argwhere(pop_map >= 0)
    local_ids = pop_map[coords[:, 0], coords[:, 1]]
    order = np.argsort(local_ids)
    coords = coords[order]
    ids = _as_1d(neuron_ids, dtype=np.int64)
    valid = (ids >= 0) & (ids < coords.shape[0])
    y = np.full(ids.shape, -1, dtype=np.int64)
    x = np.full(ids.shape, -1, dtype=np.int64)
    y[valid] = coords[ids[valid], 0]
    x[valid] = coords[ids[valid], 1]
    return y, x


def compressed_spikes_to_dense(
    spike_hist_tot: np.ndarray,
    num_spikes_pop: np.ndarray,
    n_neurons: int,
    *,
    step_tot: int | None = None,
) -> np.ndarray:
    """Reconstruct dense ``[T, N]`` spike history from 0-based compressed output."""
    counts = _as_1d(num_spikes_pop, dtype=np.int64)
    total_steps = int(step_tot if step_tot is not None else counts.size)
    if counts.size < total_steps:
        counts = np.pad(counts, (0, total_steps - counts.size))
    elif counts.size > total_steps:
        counts = counts[:total_steps]

    ids = _as_1d(spike_hist_tot, dtype=np.int64)
    dense = np.zeros((total_steps, int(n_neurons)), dtype=np.uint8)
    offset = 0
    for t, count in enumerate(counts):
        count_i = int(count)
        if count_i <= 0:
            continue
        step_ids = ids[offset : offset + count_i]
        step_ids = step_ids[(step_ids >= 0) & (step_ids < n_neurons)]
        dense[t, step_ids] = 1
        offset += count_i
    return dense


def dense_pop_to_grid_frames(
    dense_spike_hist: np.ndarray,
    *,
    pop_index: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Map dense ``[T, N_pop]`` pop-local spikes back to ``[T, H, W]`` frames."""
    frames = np.zeros((dense_spike_hist.shape[0], int(grid_shape[0]), int(grid_shape[1])), dtype=np.uint8)
    pop_map = make_population_index_maps(grid_shape)[pop_index]
    valid = pop_map >= 0
    frames[:, valid] = dense_spike_hist[:, pop_map[valid]]
    return frames


def read_spikenet_out_h5(
    out_path: str | Path,
    *,
    config_path: str | Path | None = None,
    grid_shape: tuple[int, int] | None = None,
    reconstruct_dense: bool = True,
) -> SpikeNetOutput:
    """Read the useful subset of SpikeNet ``*_out.h5``.

    This follows ``ReadH5.m`` for canonical paths:
    ``/pop_result_{i}/spike_hist_tot`` and ``/pop_result_{i}/num_spikes_pop``.
    """
    out_path = Path(out_path)
    with h5py.File(out_path, "r") as out_h5:
        datasets = _list_datasets(out_h5)
        stored_config = _read_optional(out_h5, "/config_filename/config_filename")
        stored_config_path = None if stored_config is None else _decode_scalar(stored_config)
        config_candidate = Path(config_path) if config_path is not None else (Path(stored_config_path) if stored_config_path else None)

        n = None
        dt = None
        step_tot = None
        if config_candidate is not None and config_candidate.exists():
            with h5py.File(config_candidate, "r") as cfg_h5:
                n = _read_optional(cfg_h5, "/config/Net/INIT001/N")
                dt = _read_optional(cfg_h5, "/config/Net/INIT002/dt")
                step_tot = _read_optional(cfg_h5, "/config/Net/INIT002/step_tot")

        if n is None:
            # Fallback for dense/btorch outputs that store config in out.h5.
            n = _read_optional(out_h5, "/config/Net/INIT001/N")
        if dt is None:
            dt = _read_optional(out_h5, "/config/Net/INIT002/dt")
        if step_tot is None:
            step_tot = _read_optional(out_h5, "/config/Net/INIT002/step_tot")

        pop_indices = sorted(
            int(name.split("/")[1].replace("pop_result_", ""))
            for name in datasets
            if name.startswith("/pop_result_") and len(name.split("/")) > 2
        )
        pop_indices = sorted(set(pop_indices))

        if n is None:
            n_arr = []
            for pop in pop_indices:
                ids = _read_optional(out_h5, f"/pop_result_{pop}/spike_hist_tot", np.asarray([], dtype=np.int64))
                n_arr.append(int(np.max(ids)) + 1 if np.asarray(ids).size else 0)
            n = np.asarray(n_arr, dtype=np.int64)
        n_arr = _as_1d(n, dtype=np.int64)
        if not pop_indices:
            pop_indices = list(range(len(n_arr)))
        step_total = int(_decode_scalar(step_tot)) if step_tot is not None else 0

        populations: list[PopulationOutput] = []
        for pop in pop_indices:
            spike_hist = _read_optional(out_h5, f"/pop_result_{pop}/spike_hist_tot", np.asarray([], dtype=np.int64))
            counts = _read_optional(out_h5, f"/pop_result_{pop}/num_spikes_pop", None)
            if counts is None:
                counts = np.asarray([], dtype=np.int64)
            counts_arr = _as_1d(counts, dtype=np.int64)
            n_neurons = int(n_arr[pop]) if pop < len(n_arr) else (int(np.max(spike_hist)) + 1 if np.asarray(spike_hist).size else 0)
            if step_total == 0:
                step_total = int(counts_arr.size)
            dense = None
            rate = None
            if reconstruct_dense:
                dense = compressed_spikes_to_dense(spike_hist, counts_arr, n_neurons, step_tot=step_total)
                duration_s = max((float(_decode_scalar(dt)) if dt is not None else 1.0) * step_total / 1000.0, 1e-12)
                rate = dense.sum(axis=0) / duration_s
            populations.append(
                PopulationOutput(
                    pop_index=pop,
                    n_neurons=n_neurons,
                    spike_hist_tot=_as_1d(spike_hist, dtype=np.int64),
                    num_spikes_pop=counts_arr,
                    dense_spike_hist=dense,
                    rate_hz=rate,
                )
            )

    return SpikeNetOutput(
        out_path=str(out_path),
        config_path=str(config_candidate) if config_candidate is not None else stored_config_path,
        dt=None if dt is None else float(_decode_scalar(dt)),
        step_tot=step_total,
        n=[int(v) for v in n_arr],
        populations=populations,
        available_datasets=datasets,
    )


def read_activity_frames(
    out_path: str | Path,
    *,
    config_path: str | Path | None = None,
    grid_shape: tuple[int, int] | None = None,
    pop_index: int = 0,
) -> np.ndarray:
    """Read dense activity frames from common Python or SpikeNet output schemas."""
    out_path = Path(out_path)
    with h5py.File(out_path, "r") as h5f:
        direct = _find_first_dataset(
            h5f,
            [
                "/activity",
                "/spikes",
                "/spike_frames",
                "/output/activity",
                "/output/spikes",
                f"/pop_result_{pop_index}/activity",
                f"/pop_result_{pop_index}/spike_frames",
            ],
        )
        if direct is not None:
            arr = np.asarray(h5f[direct][()])
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            if arr.ndim != 3:
                raise ValueError(f"{direct} must be [T,H,W] or [H,W], got {arr.shape}")
            return arr.astype(np.float32, copy=False)

    if grid_shape is None:
        raise ValueError("grid_shape is required when reconstructing frames from compressed SpikeNet output")
    result = read_spikenet_out_h5(out_path, config_path=config_path, grid_shape=grid_shape, reconstruct_dense=True)
    pop = next((p for p in result.populations if p.pop_index == pop_index), None)
    if pop is None or pop.dense_spike_hist is None:
        raise ValueError(f"pop_result_{pop_index} is missing or cannot be reconstructed")
    return dense_pop_to_grid_frames(pop.dense_spike_hist, pop_index=pop_index, grid_shape=grid_shape).astype(np.float32)


def integrate_windows(frames: np.ndarray, *, window: int, stride: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    return integrate_activity_windows(frames, window=window, stride=stride)


def detect_connected_components(
    activity_windows: np.ndarray,
    ranges: list[tuple[int, int]],
    *,
    threshold: float = 1.0,
    min_area: int = 4,
) -> list[Detection]:
    return detect_bboxes_from_windows(activity_windows, ranges, threshold=threshold, min_area=min_area)


def save_detections_json(detections: list[Detection], output_path: str | Path) -> Path:
    return _save_detections_json(detections, output_path)


def save_mot_txt(detections: list[Detection], output_path: str | Path) -> Path:
    return _save_mot_txt(detections, output_path)


def process_out_h5_to_detections(
    out_path: str | Path,
    *,
    config_path: str | Path | None = None,
    grid_shape: tuple[int, int] | None = None,
    pop_index: int = 0,
    window: int = 5,
    stride: int = 5,
    threshold: float = 1.0,
    min_area: int = 4,
) -> tuple[np.ndarray, list[Detection]]:
    frames = read_activity_frames(out_path, config_path=config_path, grid_shape=grid_shape, pop_index=pop_index)
    activity, ranges = integrate_windows(frames, window=window, stride=stride)
    detections = detect_connected_components(activity, ranges, threshold=threshold, min_area=min_area)
    return activity, detections


def _parse_grid_shape(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--grid-shape requires HEIGHT WIDTH")
    return int(values[0]), int(values[1])


def _main() -> None:
    parser = argparse.ArgumentParser(description="Read SpikeNet out.h5 and extract connected-component detections.")
    parser.add_argument("out_h5")
    parser.add_argument("--config-h5")
    parser.add_argument("--grid-shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--pop-index", type=int, default=0)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-mot")
    args = parser.parse_args()

    activity, detections = process_out_h5_to_detections(
        args.out_h5,
        config_path=args.config_h5,
        grid_shape=_parse_grid_shape(args.grid_shape),
        pop_index=args.pop_index,
        window=args.window,
        stride=args.stride,
        threshold=args.threshold,
        min_area=args.min_area,
    )
    save_detections_json(detections, args.output_json)
    if args.output_mot:
        save_mot_txt(detections, args.output_mot)
    print(json.dumps({"activity_shape": list(activity.shape), "num_detections": len(detections)}, indent=2))


if __name__ == "__main__":
    _main()
