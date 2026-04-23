"""Adapters between strict GU preprocessing and btorch-style tensors.

The primary interface builds a strict GU network directly into a
``BtorchInputBundle`` without writing ``*_in.h5``. A secondary from-HDF5
interface remains for debugging/replay when a SpikeNet-style config file is
useful.

The btorch recurrent connection layer, ``btorch.models.linear.SparseConn``,
expects a scipy sparse matrix with shape ``(num_src, num_dst)`` and computes
``x @ conn`` for input tensors whose last axis is ``num_src``.

This module provides only adapter contracts:

* preprocessing -> btorch/python-simulator input;
* btorch/python-simulator output -> postprocessing input.

It does not build or run a simulator. The simulator should live outside
``translation`` and consume ``BtorchInputBundle``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import h5py
import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ConnectivitySignRule = Literal["spikenet_type", "pre_pop1", "raw"]
EventMapping = Literal["gu_column_major", "row_major", "checkerboard"]


@dataclass(frozen=True)
class BtorchSynapseBlock:
    syn_index: int
    syn_type: int
    pop_pre: int
    pop_post: int
    pre_local: np.ndarray
    post_local: np.ndarray
    pre_global: np.ndarray
    post_global: np.ndarray
    weights_raw: np.ndarray
    weights_signed: np.ndarray
    delays: np.ndarray


@dataclass(frozen=True)
class BtorchInputBundle:
    in_h5: str | None
    n_by_pop: np.ndarray
    offsets: np.ndarray
    dt: float
    step_tot: int
    synapses: tuple[BtorchSynapseBlock, ...]
    connectivity: sparse.coo_array
    input_current: np.ndarray | None
    input_event_files: tuple[str, ...]
    grid_shape: tuple[int, int] | None
    mapping: EventMapping

    @property
    def n_total(self) -> int:
        return int(self.n_by_pop.sum())

    def simulator_kwargs(self) -> dict[str, Any]:
        """Return the minimal kwargs expected by a btorch-style simulator.

        ``connectivity`` is kept as scipy COO because btorch ``SparseConn``
        accepts scipy sparse matrices directly. ``input_current`` uses the
        btorch recurrent convention ``[T, batch, N_total]``.
        """
        return {
            "connectivity": self.connectivity,
            "input_current": self.input_current,
            "dt": self.dt,
            "step_tot": self.step_tot,
            "n_by_pop": self.n_by_pop,
            "offsets": self.offsets,
            "grid_shape": self.grid_shape,
            "mapping": self.mapping,
        }


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar(value[()])
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"} and value.size == 1:
        return _decode_scalar(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_1d(value: Any, dtype=None) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _read_required(h5f: h5py.File, path: str):
    if path not in h5f:
        raise KeyError(f"Missing required dataset: {path}")
    return h5f[path][()]


def _population_offsets(n_by_pop: np.ndarray) -> np.ndarray:
    n = _as_1d(n_by_pop, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(n[:-1])]).astype(np.int64)


def read_spikenet_basic_config(in_h5: str | Path) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Read population sizes, global offsets, dt, and step count."""
    with h5py.File(in_h5, "r") as h5f:
        n = _as_1d(_read_required(h5f, "/config/Net/INIT001/N"), dtype=np.int64)
        dt = float(_decode_scalar(_read_required(h5f, "/config/Net/INIT002/dt")))
        step_tot = int(_decode_scalar(_read_required(h5f, "/config/Net/INIT002/step_tot")))
    return n, _population_offsets(n), dt, step_tot


def _signed_weights(
    weights: np.ndarray,
    *,
    syn_type: int,
    pop_pre: int,
    sign_rule: ConnectivitySignRule,
) -> np.ndarray:
    raw = np.asarray(weights, dtype=np.float32)
    if sign_rule == "raw":
        return raw
    inhibitory = (syn_type == 1) if sign_rule == "spikenet_type" else (pop_pre == 1)
    return -np.abs(raw) if inhibitory else np.abs(raw)


def read_spikenet_synapses_for_btorch(
    in_h5: str | Path,
    *,
    sign_rule: ConnectivitySignRule = "spikenet_type",
) -> tuple[np.ndarray, np.ndarray, float, int, tuple[BtorchSynapseBlock, ...]]:
    """Read SpikeNet synapse blocks and convert local ids to global ids.

    Returned synapse weights are signed for btorch current-based layers. The
    translated SpikeNet HDF5 stores inhibitory conductance magnitudes as
    positive ``K`` values, so the default maps ``type == 1`` to negative
    btorch weights.
    """
    n_by_pop, offsets, dt, step_tot = read_spikenet_basic_config(in_h5)
    blocks: list[BtorchSynapseBlock] = []
    with h5py.File(in_h5, "r") as h5f:
        n_syns = int(_decode_scalar(_read_required(h5f, "/config/syns/n_syns")))
        for syn_index in range(n_syns):
            base = f"/config/syns/syn{syn_index}/INIT006"
            syn_type = int(_decode_scalar(_read_required(h5f, f"{base}/type")))
            pop_pre = int(_decode_scalar(_read_required(h5f, f"{base}/i_pre")))
            pop_post = int(_decode_scalar(_read_required(h5f, f"{base}/j_post")))
            pre_local = _as_1d(_read_required(h5f, f"{base}/I"), dtype=np.int64)
            post_local = _as_1d(_read_required(h5f, f"{base}/J"), dtype=np.int64)
            weights = _as_1d(_read_required(h5f, f"{base}/K"), dtype=np.float32)
            delays = _as_1d(_read_required(h5f, f"{base}/D"), dtype=np.float32)
            if not (pre_local.shape == post_local.shape == weights.shape == delays.shape):
                raise ValueError(f"syn{syn_index} I/J/K/D shape mismatch")
            if pre_local.size:
                if pre_local.min() < 0 or pre_local.max() >= n_by_pop[pop_pre]:
                    raise ValueError(f"syn{syn_index} pre ids out of bounds for pop{pop_pre}")
                if post_local.min() < 0 or post_local.max() >= n_by_pop[pop_post]:
                    raise ValueError(f"syn{syn_index} post ids out of bounds for pop{pop_post}")
            signed = _signed_weights(weights, syn_type=syn_type, pop_pre=pop_pre, sign_rule=sign_rule)
            blocks.append(
                BtorchSynapseBlock(
                    syn_index=syn_index,
                    syn_type=syn_type,
                    pop_pre=pop_pre,
                    pop_post=pop_post,
                    pre_local=pre_local,
                    post_local=post_local,
                    pre_global=pre_local + offsets[pop_pre],
                    post_global=post_local + offsets[pop_post],
                    weights_raw=weights,
                    weights_signed=signed,
                    delays=delays,
                )
            )
    return n_by_pop, offsets, dt, step_tot, tuple(blocks)


def build_btorch_connectivity(
    in_h5: str | Path,
    *,
    sign_rule: ConnectivitySignRule = "spikenet_type",
) -> tuple[sparse.coo_array, tuple[BtorchSynapseBlock, ...]]:
    """Build btorch ``SparseConn`` connectivity in ``(pre, post)`` orientation."""
    n_by_pop, _, _, _, blocks = read_spikenet_synapses_for_btorch(in_h5, sign_rule=sign_rule)
    n_total = int(n_by_pop.sum())
    rows = [block.pre_global for block in blocks if block.pre_global.size]
    cols = [block.post_global for block in blocks if block.post_global.size]
    data = [block.weights_signed for block in blocks if block.weights_signed.size]
    if not rows:
        conn = sparse.coo_array((n_total, n_total), dtype=np.float32)
    else:
        conn = sparse.coo_array(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_total, n_total),
            dtype=np.float32,
        )
        conn.sum_duplicates()
    return conn, blocks


def build_connectivity_from_synapse_blocks(
    n_by_pop: Iterable[int],
    blocks: Iterable[BtorchSynapseBlock],
) -> sparse.coo_array:
    """Build btorch connectivity from already converted synapse blocks."""
    n_arr = _as_1d(n_by_pop, dtype=np.int64)
    n_total = int(n_arr.sum())
    block_tuple = tuple(blocks)
    rows = [block.pre_global for block in block_tuple if block.pre_global.size]
    cols = [block.post_global for block in block_tuple if block.post_global.size]
    data = [block.weights_signed for block in block_tuple if block.weights_signed.size]
    if not rows:
        return sparse.coo_array((n_total, n_total), dtype=np.float32)
    conn = sparse.coo_array(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_total, n_total),
        dtype=np.float32,
    )
    conn.sum_duplicates()
    return conn


def btorch_blocks_from_gu_network(
    network,
    *,
    sign_rule: ConnectivitySignRule = "spikenet_type",
) -> tuple[BtorchSynapseBlock, ...]:
    """Convert an in-memory strict GU network into btorch synapse blocks."""
    n_by_pop = _as_1d(network.n_by_pop, dtype=np.int64)
    offsets = _population_offsets(n_by_pop)
    out: list[BtorchSynapseBlock] = []
    for syn_index, block in enumerate(network.synapses):
        pre_local = _as_1d(block.i_pre, dtype=np.int64)
        post_local = _as_1d(block.j_post, dtype=np.int64)
        weights = _as_1d(block.weights, dtype=np.float32)
        delays = _as_1d(block.delays, dtype=np.float32)
        signed = _signed_weights(weights, syn_type=block.syn_type, pop_pre=block.pop_pre, sign_rule=sign_rule)
        out.append(
            BtorchSynapseBlock(
                syn_index=syn_index,
                syn_type=int(block.syn_type),
                pop_pre=int(block.pop_pre),
                pop_post=int(block.pop_post),
                pre_local=pre_local,
                post_local=post_local,
                pre_global=pre_local + offsets[block.pop_pre],
                post_global=post_local + offsets[block.pop_post],
                weights_raw=weights,
                weights_signed=signed,
                delays=delays,
            )
        )
    return tuple(out)


def read_pop_input_filenames(in_h5: str | Path) -> list[str | None]:
    with h5py.File(in_h5, "r") as h5f:
        n = _as_1d(_read_required(h5f, "/config/Net/INIT001/N"), dtype=np.int64)
        out: list[str | None] = []
        for pop in range(len(n)):
            path = f"/config/pops/pop{pop}/file_current_input/fname"
            out.append(None if path not in h5f else str(_decode_scalar(h5f[path][()][0])))
    return out


def read_event_h5(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5f:
        payload = {
            "x": _as_1d(_read_required(h5f, "x"), dtype=np.int64),
            "y": _as_1d(_read_required(h5f, "y"), dtype=np.int64),
            "t": _as_1d(_read_required(h5f, "t")),
            "pol": _as_1d(_read_required(h5f, "pol"), dtype=np.float32),
            "frames_shape": tuple(int(v) for v in np.asarray(h5f.attrs.get("frames_shape", ())).reshape(-1)),
            "dt": float(h5f.attrs.get("dt", 1.0)),
            "start_time": float(h5f.attrs.get("start_time", 0.0)),
        }
    if len(payload["frames_shape"]) != 3:
        raise ValueError(f"{path} is missing attrs['frames_shape'] = (T,H,W)")
    return payload


def _checkerboard_pop_map(grid_shape: tuple[int, int], pop: int) -> np.ndarray:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    grid = np.zeros((height, width), dtype=np.int8)
    grid[1::2, 1::2] = 1
    pop_map = np.full((height, width), -1, dtype=np.int64)
    flat = np.flatnonzero(grid.ravel(order="F") == int(pop))
    rows, cols = np.unravel_index(flat, grid.shape, order="F")
    pop_map[rows, cols] = np.arange(flat.size, dtype=np.int64)
    return pop_map


def xy_to_pop_local_ids(
    x: np.ndarray,
    y: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    pop_size: int,
    pop_index: int = 0,
    mapping: EventMapping = "gu_column_major",
) -> tuple[np.ndarray, np.ndarray]:
    """Map event x/y coordinates to population-local ids.

    ``gu_column_major`` is the GU lattice convention used by
    ``rectangular_lattice_coordinates``: ``local_id = y + x * height``.
    """
    height, width = int(grid_shape[0]), int(grid_shape[1])
    x = _as_1d(x, dtype=np.int64)
    y = _as_1d(y, dtype=np.int64)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    if mapping == "gu_column_major":
        local = y + x * height
    elif mapping == "row_major":
        local = y * width + x
    elif mapping == "checkerboard":
        pop_map = _checkerboard_pop_map((height, width), pop_index)
        local = np.full(x.shape, -1, dtype=np.int64)
        local[in_bounds] = pop_map[y[in_bounds], x[in_bounds]]
    else:
        raise ValueError(f"unknown event mapping: {mapping}")

    valid = in_bounds & (local >= 0) & (local < int(pop_size))
    return local[valid].astype(np.int64, copy=False), valid


def event_h5_to_btorch_current(
    event_h5: str | Path,
    *,
    n_by_pop: Iterable[int],
    offsets: Iterable[int],
    pop_index: int = 0,
    step_tot: int | None = None,
    grid_shape: tuple[int, int] | None = None,
    mapping: EventMapping = "gu_column_major",
    batch_size: int = 1,
    current_scale: float = 1.0,
    use_polarity: bool = False,
) -> np.ndarray:
    """Rasterize event HDF5 into btorch external current ``[T,B,N_total]``."""
    event = read_event_h5(event_h5)
    event_t, height, width = event["frames_shape"]
    grid_shape = (height, width) if grid_shape is None else grid_shape
    n_arr = _as_1d(n_by_pop, dtype=np.int64)
    offsets_arr = _as_1d(offsets, dtype=np.int64)
    total = int(n_arr.sum())
    total_steps = int(step_tot if step_tot is not None else event_t)
    current = np.zeros((total_steps, int(batch_size), total), dtype=np.float32)

    local_ids, valid_xy = xy_to_pop_local_ids(
        event["x"],
        event["y"],
        grid_shape=grid_shape,
        pop_size=int(n_arr[pop_index]),
        pop_index=pop_index,
        mapping=mapping,
    )
    t = event["t"][valid_xy].astype(np.float64, copy=False)
    bins = np.rint((t - event["start_time"]) / max(float(event["dt"]), 1e-12)).astype(np.int64)
    valid_t = (bins >= 0) & (bins < total_steps)
    if not np.any(valid_t):
        return current
    global_ids = local_ids[valid_t] + int(offsets_arr[pop_index])
    values = event["pol"][valid_xy][valid_t] if use_polarity else 1.0
    values = np.asarray(values, dtype=np.float32) * float(current_scale)
    for batch in range(int(batch_size)):
        np.add.at(current, (bins[valid_t], np.full(global_ids.shape, batch), global_ids), values)
    return current


def load_btorch_input_bundle(
    in_h5: str | Path,
    *,
    input_pop: int = 0,
    event_h5: str | Path | None = None,
    grid_shape: tuple[int, int] | None = None,
    mapping: EventMapping = "gu_column_major",
    sign_rule: ConnectivitySignRule = "spikenet_type",
    batch_size: int = 1,
    current_scale: float = 1.0,
    use_polarity: bool = False,
    load_input_current: bool = True,
) -> BtorchInputBundle:
    """Read SpikeNet preprocessing output as btorch-ready arrays."""
    in_h5 = Path(in_h5)
    n_by_pop, offsets, dt, step_tot, blocks = read_spikenet_synapses_for_btorch(in_h5, sign_rule=sign_rule)
    conn, _ = build_btorch_connectivity(in_h5, sign_rule=sign_rule)
    pop_files = read_pop_input_filenames(in_h5)
    event_path = Path(event_h5) if event_h5 is not None else (Path(pop_files[input_pop]) if pop_files[input_pop] else None)
    current = None
    event_files: tuple[str, ...] = tuple(path for path in pop_files if path)
    if load_input_current:
        if event_path is None:
            raise ValueError(f"No event HDF5 path found for pop{input_pop}")
        current = event_h5_to_btorch_current(
            event_path,
            n_by_pop=n_by_pop,
            offsets=offsets,
            pop_index=input_pop,
            step_tot=step_tot,
            grid_shape=grid_shape,
            mapping=mapping,
            batch_size=batch_size,
            current_scale=current_scale,
            use_polarity=use_polarity,
        )
    resolved_shape = None
    if event_path is not None:
        event = read_event_h5(event_path)
        resolved_shape = (event["frames_shape"][1], event["frames_shape"][2]) if grid_shape is None else grid_shape
    return BtorchInputBundle(
        in_h5=str(in_h5),
        n_by_pop=n_by_pop,
        offsets=offsets,
        dt=dt,
        step_tot=step_tot,
        synapses=blocks,
        connectivity=conn,
        input_current=current,
        input_event_files=event_files,
        grid_shape=resolved_shape,
        mapping=mapping,
    )


def build_gu_2018_strict_btorch_bundle(
    event_h5: str | Path,
    *,
    event_h5_i: str | Path | None = None,
    config=None,
    input_pop: int = 0,
    grid_shape: tuple[int, int] | None = None,
    mapping: EventMapping = "gu_column_major",
    sign_rule: ConnectivitySignRule = "spikenet_type",
    batch_size: int = 1,
    current_scale: float = 1.0,
    use_polarity: bool = False,
    load_input_current: bool = True,
    seed: int | None = 1,
) -> BtorchInputBundle:
    """Direct strict GU -> btorch path that does not write ``*_in.h5``.

    This is the primary in-memory interface for btorch runners. Use
    ``build_gu_2018_strict_input`` + ``load_btorch_input_bundle`` only when a
    persistent SpikeNet-style HDF5 artifact is useful for debugging or replay.
    """
    from translation.spikenet_preprocessing import GU2018Config, build_gu_2018_strict_network

    cfg = GU2018Config() if config is None else config
    network = build_gu_2018_strict_network(event_h5, event_h5_i, config=cfg, seed=seed)
    n_by_pop = _as_1d(network.n_by_pop, dtype=np.int64)
    offsets = _population_offsets(n_by_pop)
    blocks = btorch_blocks_from_gu_network(network, sign_rule=sign_rule)
    conn = build_connectivity_from_synapse_blocks(n_by_pop, blocks)

    current = None
    if load_input_current:
        current = event_h5_to_btorch_current(
            event_h5,
            n_by_pop=n_by_pop,
            offsets=offsets,
            pop_index=input_pop,
            step_tot=cfg.step_tot,
            grid_shape=grid_shape,
            mapping=mapping,
            batch_size=batch_size,
            current_scale=current_scale,
            use_polarity=use_polarity,
        )
    event = read_event_h5(event_h5)
    resolved_shape = (event["frames_shape"][1], event["frames_shape"][2]) if grid_shape is None else grid_shape
    return BtorchInputBundle(
        in_h5=None,
        n_by_pop=n_by_pop,
        offsets=offsets,
        dt=float(cfg.dt),
        step_tot=int(cfg.step_tot),
        synapses=blocks,
        connectivity=conn,
        input_current=current,
        input_event_files=network.input_event_files,
        grid_shape=resolved_shape,
        mapping=mapping,
    )


def save_btorch_bundle_npz(bundle: BtorchInputBundle, output_npz: str | Path) -> Path:
    """Save arrays for quick inspection or handoff to a btorch runner."""
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    conn = bundle.connectivity.tocoo()
    np.savez_compressed(
        output_npz,
        n_by_pop=bundle.n_by_pop,
        offsets=bundle.offsets,
        dt=np.asarray(bundle.dt, dtype=np.float32),
        step_tot=np.asarray(bundle.step_tot, dtype=np.int64),
        conn_row=conn.row.astype(np.int64),
        conn_col=conn.col.astype(np.int64),
        conn_data=conn.data.astype(np.float32),
        conn_shape=np.asarray(conn.shape, dtype=np.int64),
        input_current=np.asarray([]) if bundle.input_current is None else bundle.input_current,
    )
    return output_npz


def spikes_to_grid_activity(
    spikes: Any,
    *,
    n_by_pop: Iterable[int],
    offsets: Iterable[int],
    grid_shape: tuple[int, int],
    pop_index: int = 0,
    mapping: EventMapping = "gu_column_major",
    batch_index: int = 0,
) -> np.ndarray:
    """Convert btorch spike tensor ``[T,N]`` or ``[T,B,N]`` to ``[T,H,W]``."""
    arr = spikes.detach().cpu().numpy() if hasattr(spikes, "detach") else np.asarray(spikes)
    if arr.ndim == 3:
        arr = arr[:, int(batch_index), :]
    if arr.ndim != 2:
        raise ValueError(f"spikes must be [T,N] or [T,B,N], got {arr.shape}")
    n_arr = _as_1d(n_by_pop, dtype=np.int64)
    offsets_arr = _as_1d(offsets, dtype=np.int64)
    start = int(offsets_arr[pop_index])
    stop = start + int(n_arr[pop_index])
    pop_spikes = arr[:, start:stop]
    height, width = int(grid_shape[0]), int(grid_shape[1])
    frames = np.zeros((pop_spikes.shape[0], height, width), dtype=np.float32)
    if mapping == "gu_column_major":
        usable = min(pop_spikes.shape[1], height * width)
        rows, cols = np.unravel_index(np.arange(usable), (height, width), order="F")
        frames[:, rows, cols] = pop_spikes[:, :usable]
    elif mapping == "row_major":
        usable = min(pop_spikes.shape[1], height * width)
        frames[:, :, :] = pop_spikes[:, :usable].reshape(pop_spikes.shape[0], height, width)
    elif mapping == "checkerboard":
        pop_map = _checkerboard_pop_map((height, width), pop_index)
        valid = pop_map >= 0
        frames[:, valid] = pop_spikes[:, pop_map[valid]]
    else:
        raise ValueError(f"unknown event mapping: {mapping}")
    return frames


def extract_spikes_from_simulator_output(simulator_output: Any) -> Any:
    """Extract a spike tensor from common btorch-style simulator outputs.

    Accepted forms:

    * spike tensor/ndarray directly;
    * ``(spikes, states, ...)`` tuple/list;
    * dict with ``spikes``, ``z``, ``output``, or ``activity``.
    """
    if isinstance(simulator_output, dict):
        for key in ("spikes", "z", "output", "activity"):
            if key in simulator_output:
                return simulator_output[key]
        raise KeyError("simulator output dict must contain one of: spikes, z, output, activity")
    if isinstance(simulator_output, (tuple, list)):
        if not simulator_output:
            raise ValueError("simulator output tuple/list is empty")
        return simulator_output[0]
    return simulator_output


def simulator_output_to_activity(
    simulator_output: Any,
    bundle: BtorchInputBundle,
    *,
    grid_shape: tuple[int, int] | None = None,
    pop_index: int = 0,
    mapping: EventMapping | None = None,
    batch_index: int = 0,
) -> np.ndarray:
    """Convert python-simulator output into postprocessing ``[T,H,W]`` frames."""
    resolved_shape = grid_shape or bundle.grid_shape
    if resolved_shape is None:
        raise ValueError("grid_shape is required when bundle.grid_shape is unavailable")
    return spikes_to_grid_activity(
        extract_spikes_from_simulator_output(simulator_output),
        n_by_pop=bundle.n_by_pop,
        offsets=bundle.offsets,
        grid_shape=resolved_shape,
        pop_index=pop_index,
        mapping=mapping or bundle.mapping,
        batch_index=batch_index,
    )


def postprocess_btorch_simulator_output(
    simulator_output: Any,
    bundle: BtorchInputBundle,
    *,
    grid_shape: tuple[int, int] | None = None,
    pop_index: int = 0,
    mapping: EventMapping | None = None,
    batch_index: int = 0,
    window: int = 5,
    stride: int = 5,
    threshold: float | None = 1.0,
    threshold_quantile: float | None = None,
    min_area: int = 4,
    closing_radius: int = 0,
    nms_iou: float = 0.0,
):
    """Postprocess btorch/python-simulator output directly.

    The simulator output is expected to be btorch-style spikes with shape
    ``[T, N]`` or ``[T, batch, N]``. No HDF5 is required.
    """
    from translation.detection_readout import readout_bboxes

    frames = simulator_output_to_activity(
        simulator_output,
        bundle,
        grid_shape=grid_shape,
        pop_index=pop_index,
        mapping=mapping,
        batch_index=batch_index,
    )
    activity, detections = readout_bboxes(
        frames,
        window=window,
        stride=stride,
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        min_area=min_area,
        closing_radius=closing_radius,
        nms_iou=nms_iou,
    )
    return activity, detections


def write_btorch_output_h5(
    output_h5: str | Path,
    spikes: Any,
    *,
    in_h5: str | Path,
    grid_shape: tuple[int, int],
    pop_index: int = 0,
    mapping: EventMapping = "gu_column_major",
    batch_index: int = 0,
) -> Path:
    """Write btorch spikes to HDF5 compatible with current postprocessing."""
    n_by_pop, offsets, dt, step_tot = read_spikenet_basic_config(in_h5)
    frames = spikes_to_grid_activity(
        spikes,
        n_by_pop=n_by_pop,
        offsets=offsets,
        grid_shape=grid_shape,
        pop_index=pop_index,
        mapping=mapping,
        batch_index=batch_index,
    )
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as h5f:
        h5f.create_dataset("/activity", data=frames, compression="gzip")
        h5f.create_dataset(f"/pop_result_{pop_index}/activity", data=frames, compression="gzip")
        h5f.create_dataset("/config_filename/config_filename", data=np.asarray([str(in_h5)], dtype=h5py.string_dtype("utf-8")))
        h5f.create_dataset("/config/Net/INIT001/N", data=n_by_pop)
        h5f.create_dataset("/config/Net/INIT002/dt", data=float(dt))
        h5f.create_dataset("/config/Net/INIT002/step_tot", data=int(step_tot))
        h5f.attrs["producer"] = "translation.btorch_interface.write_btorch_output_h5"
        h5f.attrs["grid_shape"] = np.asarray(grid_shape, dtype=np.int64)
        h5f.attrs["mapping"] = mapping
    return output_h5


def write_btorch_output_h5_from_bundle(
    output_h5: str | Path,
    spikes: Any,
    bundle: BtorchInputBundle,
    *,
    grid_shape: tuple[int, int] | None = None,
    pop_index: int = 0,
    mapping: EventMapping | None = None,
    batch_index: int = 0,
) -> Path:
    """Write btorch spikes using bundle metadata, including direct bundles."""
    resolved_shape = grid_shape or bundle.grid_shape
    if resolved_shape is None:
        raise ValueError("grid_shape is required when bundle.grid_shape is unavailable")
    resolved_mapping = mapping or bundle.mapping
    frames = spikes_to_grid_activity(
        spikes,
        n_by_pop=bundle.n_by_pop,
        offsets=bundle.offsets,
        grid_shape=resolved_shape,
        pop_index=pop_index,
        mapping=resolved_mapping,
        batch_index=batch_index,
    )
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as h5f:
        h5f.create_dataset("/activity", data=frames, compression="gzip")
        h5f.create_dataset(f"/pop_result_{pop_index}/activity", data=frames, compression="gzip")
        if bundle.in_h5 is not None:
            h5f.create_dataset("/config_filename/config_filename", data=np.asarray([bundle.in_h5], dtype=h5py.string_dtype("utf-8")))
        h5f.create_dataset("/config/Net/INIT001/N", data=bundle.n_by_pop)
        h5f.create_dataset("/config/Net/INIT002/dt", data=float(bundle.dt))
        h5f.create_dataset("/config/Net/INIT002/step_tot", data=int(bundle.step_tot))
        h5f.attrs["producer"] = "translation.btorch_interface.write_btorch_output_h5_from_bundle"
        h5f.attrs["grid_shape"] = np.asarray(resolved_shape, dtype=np.int64)
        h5f.attrs["mapping"] = resolved_mapping
    return output_h5


def summarize_bundle(bundle: BtorchInputBundle) -> dict[str, Any]:
    conn = bundle.connectivity
    weights = conn.data
    return {
        "in_h5": bundle.in_h5,
        "n_by_pop": bundle.n_by_pop.astype(int).tolist(),
        "n_total": bundle.n_total,
        "dt": bundle.dt,
        "step_tot": bundle.step_tot,
        "grid_shape": None if bundle.grid_shape is None else list(bundle.grid_shape),
        "mapping": bundle.mapping,
        "input_event_files": list(bundle.input_event_files),
        "input_current_shape": None if bundle.input_current is None else list(bundle.input_current.shape),
        "input_current_nonzero": None if bundle.input_current is None else int(np.count_nonzero(bundle.input_current)),
        "connectivity_shape": list(conn.shape),
        "connectivity_nnz": int(conn.nnz),
        "weight_min": None if weights.size == 0 else float(np.min(weights)),
        "weight_max": None if weights.size == 0 else float(np.max(weights)),
        "synapses": [
            {
                "syn_index": block.syn_index,
                "type": block.syn_type,
                "pop_pre": block.pop_pre,
                "pop_post": block.pop_post,
                "count": int(block.pre_local.size),
                "signed_min": None if block.weights_signed.size == 0 else float(block.weights_signed.min()),
                "signed_max": None if block.weights_signed.size == 0 else float(block.weights_signed.max()),
            }
            for block in bundle.synapses
        ],
    }


def _parse_grid_shape(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--grid-shape requires HEIGHT WIDTH")
    return int(values[0]), int(values[1])


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build btorch-ready arrays from strict GU inputs.")
    subparsers = parser.add_subparsers(dest="interface", required=True)

    from_h5 = subparsers.add_parser("from-h5", help="Debug/replay path: *_in.h5 -> BtorchInputBundle.")
    from_h5.add_argument("in_h5")
    from_h5.add_argument("--event-h5", type=Path)
    from_h5.add_argument("--input-pop", type=int, default=0)
    from_h5.add_argument("--grid-shape", nargs=2, type=int)
    from_h5.add_argument("--mapping", choices=("gu_column_major", "row_major", "checkerboard"), default="gu_column_major")
    from_h5.add_argument("--sign-rule", choices=("spikenet_type", "pre_pop1", "raw"), default="spikenet_type")
    from_h5.add_argument("--current-scale", type=float, default=1.0)
    from_h5.add_argument("--batch-size", type=int, default=1)
    from_h5.add_argument("--no-input-current", action="store_true")
    from_h5.add_argument("--output-npz", type=Path)
    from_h5.add_argument("--summary-json", type=Path)

    direct = subparsers.add_parser("direct-gu", help="Primary path: strict GU -> BtorchInputBundle without *_in.h5.")
    direct.add_argument("event_h5", type=Path)
    direct.add_argument("--event-h5-i", type=Path)
    direct.add_argument("--input-pop", type=int, default=0)
    direct.add_argument("--lattice-shape", nargs=2, type=int)
    direct.add_argument("--n-i", type=int)
    direct.add_argument("--step-tot", type=int)
    direct.add_argument("--grid-shape", nargs=2, type=int)
    direct.add_argument("--mapping", choices=("gu_column_major", "row_major", "checkerboard"), default="gu_column_major")
    direct.add_argument("--sign-rule", choices=("spikenet_type", "pre_pop1", "raw"), default="spikenet_type")
    direct.add_argument("--current-scale", type=float, default=1.0)
    direct.add_argument("--batch-size", type=int, default=1)
    direct.add_argument("--connection-device", default="auto")
    direct.add_argument("--common-neighbor-iterations", type=int, default=10)
    direct.add_argument("--seed", type=int, default=1)
    direct.add_argument("--no-input-current", action="store_true")
    direct.add_argument("--output-npz", type=Path)
    direct.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    if args.interface == "from-h5":
        bundle = load_btorch_input_bundle(
            args.in_h5,
            input_pop=args.input_pop,
            event_h5=args.event_h5,
            grid_shape=_parse_grid_shape(args.grid_shape),
            mapping=args.mapping,
            sign_rule=args.sign_rule,
            batch_size=args.batch_size,
            current_scale=args.current_scale,
            load_input_current=not args.no_input_current,
        )
    else:
        from translation.spikenet_preprocessing import GU2018Config

        bundle = build_gu_2018_strict_btorch_bundle(
            args.event_h5,
            event_h5_i=args.event_h5_i,
            config=GU2018Config(
                lattice_shape=(63, 63) if args.lattice_shape is None else (int(args.lattice_shape[0]), int(args.lattice_shape[1])),
                n_i=args.n_i,
                step_tot=100000 if args.step_tot is None else int(args.step_tot),
                connection_device=args.connection_device,
                common_neighbor_iterations=args.common_neighbor_iterations,
            ),
            input_pop=args.input_pop,
            grid_shape=_parse_grid_shape(args.grid_shape),
            mapping=args.mapping,
            sign_rule=args.sign_rule,
            batch_size=args.batch_size,
            current_scale=args.current_scale,
            load_input_current=not args.no_input_current,
            seed=args.seed,
        )
    summary = summarize_bundle(bundle)
    print(json.dumps(summary, indent=2))
    if args.output_npz:
        save_btorch_bundle_npz(bundle, args.output_npz)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    _main()
