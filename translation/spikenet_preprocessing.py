"""SpikeNet pre-processing translated to Python.

The generated HDF5 layout follows SpikeNet's C++ simulator interface. Unlike
the original Matlab helpers, public Python functions use 0-based population,
neuron, synapse-type, and coordinate indices throughout.
"""

from __future__ import annotations

import argparse
import types
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


SPIKENET_ROOT = Path(__file__).resolve().parents[2] / "SpikeNet"
NETWORK_GENERATOR_ROOT = Path(__file__).resolve().parents[2] / "network_generator"


def _as_1d_array(values, *, dtype=None) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _require_nonnegative(values: np.ndarray, name: str) -> None:
    if values.size and np.min(values) < 0:
        raise ValueError(f"{name} must use Python-style 0-based nonnegative indices")


def _require_index_bounds(values: np.ndarray, name: str, size: int) -> None:
    _require_nonnegative(values, name)
    if values.size and np.max(values) >= int(size):
        raise ValueError(f"{name} contains index {int(np.max(values))}, but size is {int(size)}")


def _delete_if_exists(h5f: h5py.File, path: str) -> None:
    if path in h5f:
        del h5f[path]


def _write_dataset(h5f: h5py.File, path: str, data, *, dtype=None) -> None:
    _delete_if_exists(h5f, path)
    parent = str(Path(path).parent).replace("\\", "/")
    if parent and parent != ".":
        h5f.require_group(parent)
    h5f.create_dataset(path, data=np.asarray(data, dtype=dtype) if dtype else data)


def _write_ascii_codes(h5f: h5py.File, path: str, text: str) -> None:
    _write_dataset(h5f, path, np.frombuffer(text.encode("ascii"), dtype=np.uint8).astype(np.float64))


def _parameter_string(pairs: dict[str, object] | Sequence[tuple[str, object]]) -> str:
    items = pairs.items() if isinstance(pairs, dict) else pairs
    return "".join(f"{name},{value}," for name, value in items)


def _open_h5(file_or_path, mode: str = "a"):
    if isinstance(file_or_path, h5py.File):
        return file_or_path, False
    path = Path(file_or_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(path, mode), True


def new_ygin_file(
    loop_num: int,
    *,
    output_dir: str | Path = ".",
    filename: str | None = None,
    seed: int | None = None,
) -> tuple[Path, np.random.Generator]:
    """Create a SpikeNet ``*_in.h5`` file and seed a NumPy RNG.

    Equivalent to ``new_ygin_files_and_randseedHDF5.m``. When ``filename`` is
    omitted, the Matlab timestamp naming convention is used.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    date_part = now.strftime("%Y%m%d%H%M")
    sec_ms = f"{now.second:02d}{now.microsecond // 1000:03d}"
    generated_name = f"{loop_num:04g}-{date_part}-{sec_ms}_in.h5"
    path = output_dir / (filename or generated_name)
    rand_seed = int(seed if seed is not None else loop_num * 100000 + int(sec_ms))

    with h5py.File(path, "w") as h5f:
        _write_dataset(h5f, "Original_filename", np.bytes_(path.name))
        _write_dataset(h5f, "/MATLAB/rng_seed", rand_seed)
        _write_dataset(h5f, "/MATLAB/rng_alg", np.bytes_("twister"))

    return path, np.random.default_rng(rand_seed)


def write_basic_para(file_or_path, dt: float, step_tot: int, n: Sequence[int]) -> None:
    with _managed_h5(file_or_path) as h5f:
        _write_dataset(h5f, "/config/Net/INIT001/N", _as_1d_array(n, dtype=np.int64))
        _write_dataset(h5f, "/config/Net/INIT002/dt", float(dt))
        _write_dataset(h5f, "/config/Net/INIT002/step_tot", int(step_tot))


def write_pop_para(file_or_path, pop_ind: int, **params) -> None:
    _require_nonnegative(np.asarray([pop_ind]), "pop_ind")
    para_str = _parameter_string(params)
    with _managed_h5(file_or_path) as h5f:
        _write_ascii_codes(
            h5f,
            f"/config/pops/pop{pop_ind}/PARA001/para_str_ascii",
            para_str,
        )


def write_syn_para(file_or_path, **params) -> None:
    with _managed_h5(file_or_path) as h5f:
        _write_ascii_codes(h5f, "/config/syns/PARA002/para_str_ascii", _parameter_string(params))


def write_synapse_model_choice(file_or_path, model_choice: int) -> None:
    if model_choice < 0 or int(model_choice) != model_choice:
        raise ValueError("model_choice must be a nonnegative integer")
    if model_choice == 0:
        return
    with _managed_h5(file_or_path) as h5f:
        _write_dataset(h5f, "/config/syns/INIT013/model_choice", int(model_choice), dtype=np.int32)


def write_elif_neuron_model(file_or_path, pop_ind: int, elif_vt: float, elif_delt: float) -> None:
    _require_nonnegative(np.asarray([pop_ind]), "pop_ind")
    with _managed_h5(file_or_path) as h5f:
        _write_dataset(h5f, f"/config/pops/pop{pop_ind}/neuron_model", 1, dtype=np.int32)
        _write_dataset(h5f, f"/config/pops/pop{pop_ind}/ELIF/ELIF_VT", float(elif_vt))
        _write_dataset(h5f, f"/config/pops/pop{pop_ind}/ELIF/ELIF_delT", float(elif_delt))


def write_init_cond(file_or_path, r_v0: Sequence[float], p_fire: Sequence[float]) -> None:
    r_v0_arr = _as_1d_array(r_v0, dtype=float)
    p_fire_arr = _as_1d_array(p_fire, dtype=float)
    if r_v0_arr.shape != p_fire_arr.shape:
        raise ValueError("r_v0 and p_fire must have the same length")
    if np.any((r_v0_arr < 0) | (r_v0_arr > 1)) or np.any((p_fire_arr < 0) | (p_fire_arr > 1)):
        raise ValueError("r_v0 and p_fire must be within [0, 1]")
    with _managed_h5(file_or_path) as h5f:
        for i, (rv, pf) in enumerate(zip(r_v0_arr, p_fire_arr)):
            _write_dataset(h5f, f"/config/pops/pop{i}/INIT011/r_V0", float(rv))
            _write_dataset(h5f, f"/config/pops/pop{i}/INIT011/p_fire", float(pf))


def write_ext_conductance_settings(file_or_path, pop_ind: int, mean, std) -> None:
    mean_arr = _as_1d_array(mean, dtype=float)
    std_arr = _as_1d_array(std, dtype=float)
    _require_nonnegative(np.asarray([pop_ind]), "pop_ind")
    with _managed_h5(file_or_path) as h5f:
        _write_dataset(h5f, f"/config/pops/pop{pop_ind}/INIT012/mean", mean_arr)
        _write_dataset(h5f, f"/config/pops/pop{pop_ind}/INIT012/std", std_arr)


def write_ext_current_pop(file_or_path, fname: str | Path, pop_ind: int) -> None:
    _require_nonnegative(np.asarray([pop_ind]), "pop_ind")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with _managed_h5(file_or_path) as h5f:
        _write_dataset(
            h5f,
            f"/config/pops/pop{pop_ind}/file_current_input/fname",
            np.asarray([str(fname)], dtype=string_dtype),
        )


def write_chemical_connection(
    file_or_path,
    syn_type: int,
    pop_pre: int,
    pop_post: int,
    i_pre,
    j_post,
    weights,
    delays=None,
) -> int | None:
    i_arr = _as_1d_array(i_pre, dtype=np.int64)
    j_arr = _as_1d_array(j_post, dtype=np.int64)
    k_arr = _as_1d_array(weights, dtype=float)
    if not (i_arr.shape == j_arr.shape == k_arr.shape):
        raise ValueError("i_pre, j_post, and weights must have the same length")
    if i_arr.size == 0:
        return None
    _require_nonnegative(i_arr, "i_pre")
    _require_nonnegative(j_arr, "j_post")
    _require_nonnegative(np.asarray([syn_type, pop_pre, pop_post]), "syn_type/pop indices")
    d_arr = np.zeros_like(k_arr) if delays is None else _as_1d_array(delays, dtype=float)
    if d_arr.shape != k_arr.shape:
        raise ValueError("delays must have the same length as weights")

    with _managed_h5(file_or_path) as h5f:
        n_syns = int(h5f["/config/syns/n_syns"][()]) if "/config/syns/n_syns" in h5f else 0
        syn_index = n_syns
        _write_dataset(h5f, "/config/syns/n_syns", n_syns + 1)
        base = f"/config/syns/syn{syn_index}/INIT006"
        _write_dataset(h5f, f"{base}/type", int(syn_type), dtype=np.int32)
        _write_dataset(h5f, f"{base}/i_pre", int(pop_pre), dtype=np.int32)
        _write_dataset(h5f, f"{base}/j_post", int(pop_post), dtype=np.int32)
        _write_dataset(h5f, f"{base}/I", i_arr, dtype=np.int64)
        _write_dataset(h5f, f"{base}/J", j_arr, dtype=np.int64)
        _write_dataset(h5f, f"{base}/K", k_arr)
        _write_dataset(h5f, f"{base}/D", d_arr)
    return syn_index


def write_neuron_sampling(file_or_path, pop_ind: int, data_type, sample_ind, time_index) -> None:
    data = _as_1d_array(data_type, dtype=np.int8)
    if data.size < 8:
        raise ValueError("data_type must have length at least 8")
    if np.any((data != 0) & (data != 1)):
        raise ValueError("data_type must be logical")
    neurons = _as_1d_array(sample_ind, dtype=np.int64)
    _require_nonnegative(neurons, "sample_ind")
    times = _as_1d_array(time_index, dtype=np.int8)
    if np.any((times != 0) & (times != 1)):
        raise ValueError("time_index must be logical")

    _require_nonnegative(np.asarray([pop_ind]), "pop_ind")
    names = ["V", "I_leak", "I_AMPA", "I_GABA", "I_NMDA", "I_GJ", "I_ext", "I_K", "rhat"]
    with _managed_h5(file_or_path) as h5f:
        for idx, name in enumerate(names[: data.size]):
            _write_dataset(h5f, f"/config/pops/pop{pop_ind}/SAMP001/data_type/{name}", int(data[idx]), dtype=np.int8)
        _write_dataset(h5f, f"/config/pops/pop{pop_ind}/SAMP001/neurons", neurons, dtype=np.int64)
        _write_dataset(h5f, f"/config/pops/pop{pop_ind}/SAMP001/time_points", times, dtype=np.int8)


def write_expl_var(file_or_path, **variables) -> None:
    with _managed_h5(file_or_path) as h5f:
        _write_dataset(h5f, "/config/explanatory_variables", _parameter_string(variables))


def append_config_text(file_or_path, text: str) -> None:
    with _managed_h5(file_or_path) as h5f:
        _write_dataset(h5f, "/config/MATLAB/config.m", text)


class _managed_h5:
    def __init__(self, file_or_path):
        self.file_or_path = file_or_path
        self.h5f = None
        self.should_close = False

    def __enter__(self):
        self.h5f, self.should_close = _open_h5(self.file_or_path, "a")
        return self.h5f

    def __exit__(self, exc_type, exc, tb):
        if self.should_close:
            self.h5f.close()
        return False


def population_coordinates(grid: np.ndarray, pop_ind: int) -> np.ndarray:
    """Return 0-based ``[row, col]`` coordinates in column-major neuron order."""
    _require_nonnegative(np.asarray([pop_ind]), "pop_ind")
    flat = np.flatnonzero(np.asarray(grid).ravel(order="F") == int(pop_ind))
    rows, cols = np.unravel_index(flat, grid.shape, order="F")
    return np.column_stack((rows, cols)).astype(np.float64)


def periodic_distance_matrix(
    pre_coords: np.ndarray,
    post_coords: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    grid_h, grid_w = grid_shape
    dx = np.abs(pre_coords[:, None, 0] - post_coords[None, :, 0])
    dy = np.abs(pre_coords[:, None, 1] - post_coords[None, :, 1])
    dx = np.minimum(dx, grid_h - dx)
    dy = np.minimum(dy, grid_w - dy)
    return np.sqrt(dx * dx + dy * dy)


def _resolve_torch_device(device: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for torch connection generation") from exc

    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = device
    return torch, torch.device(resolved)


def generate_chen_gong_connections_torch(
    grid: np.ndarray,
    pop_pre: int,
    pop_post: int,
    weight: float,
    drange: float,
    sigma: float,
    *,
    inhibitory_pre: bool,
    pbc: bool = True,
    delay_max: float = 0.0,
    rng: np.random.Generator | None = None,
    post_chunk_size: int = 512,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one population-pair connectivity block with PyTorch.

    The returned ``I`` and ``J`` arrays are pop-local, 0-based neuron indices.
    Distance computation and thresholding run on ``device``; final arrays are
    returned on CPU because HDF5 writing is CPU-bound.
    """
    torch, torch_device = _resolve_torch_device(device)
    rng = rng or np.random.default_rng()
    pre_coords_np = population_coordinates(grid, pop_pre).astype(np.float32, copy=False)
    post_coords_np = population_coordinates(grid, pop_post).astype(np.float32, copy=False)
    grid_h, grid_w = int(grid.shape[0]), int(grid.shape[1])

    pre_coords = torch.as_tensor(pre_coords_np, dtype=torch.float32, device=torch_device)
    drange2 = float(drange) * float(drange)
    sigma_f = float(sigma)

    all_i: list[np.ndarray] = []
    all_j: list[np.ndarray] = []
    all_k: list[np.ndarray] = []
    all_d: list[np.ndarray] = []

    for start in range(0, len(post_coords_np), post_chunk_size):
        stop = min(start + post_chunk_size, len(post_coords_np))
        chunk = torch.as_tensor(post_coords_np[start:stop], dtype=torch.float32, device=torch_device)
        drow = torch.abs(pre_coords[:, None, 0] - chunk[None, :, 0])
        dcol = torch.abs(pre_coords[:, None, 1] - chunk[None, :, 1])
        if pbc:
            drow = torch.minimum(drow, torch.as_tensor(float(grid_h), device=torch_device) - drow)
            dcol = torch.minimum(dcol, torch.as_tensor(float(grid_w), device=torch_device) - dcol)
        dist2 = drow.square() + dcol.square()
        pre_idx, post_local = torch.nonzero(dist2 <= drange2, as_tuple=True)
        if pre_idx.numel() == 0:
            continue

        post_idx = post_local + start
        if pop_pre == pop_post:
            keep = pre_idx != post_idx
            if not bool(torch.any(keep)):
                continue
            pre_idx = pre_idx[keep]
            post_idx = post_idx[keep]
            post_local = post_local[keep]

        dist2_vals = dist2[pre_idx, post_local]
        if inhibitory_pre:
            weights = torch.full((pre_idx.numel(),), float(weight), dtype=torch.float32, device=torch_device)
        else:
            weights = float(weight) * torch.exp(-dist2_vals / sigma_f)
        delays = rng.random(int(pre_idx.numel())) * delay_max if delay_max else np.zeros(int(pre_idx.numel()), dtype=float)

        all_i.append(pre_idx.detach().cpu().numpy().astype(np.int64, copy=False))
        all_j.append(post_idx.detach().cpu().numpy().astype(np.int64, copy=False))
        all_k.append(weights.detach().cpu().numpy().astype(float, copy=False))
        all_d.append(np.asarray(delays, dtype=float))

        del chunk, drow, dcol, dist2, pre_idx, post_local, post_idx, weights
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_i:
        empty_i = np.asarray([], dtype=np.int64)
        empty_f = np.asarray([], dtype=float)
        return empty_i, empty_i.copy(), empty_f, empty_f.copy()
    return tuple(np.concatenate(parts) for parts in (all_i, all_j, all_k, all_d))


def generate_chen_gong_connections(
    grid: np.ndarray,
    pop_pre: int,
    pop_post: int,
    weight: float,
    drange: float,
    sigma: float,
    *,
    inhibitory_pre: bool,
    pbc: bool = True,
    delay_max: float = 0.0,
    rng: np.random.Generator | None = None,
    post_chunk_size: int = 512,
    backend: str = "auto",
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one Chen/Gong 2019 population-pair connectivity block.

    ``backend="auto"`` uses PyTorch when available and falls back to NumPy.
    All returned indices are 0-based and ready to write to C++ HDF5 config.
    """
    if backend not in {"auto", "torch", "numpy"}:
        raise ValueError("backend must be one of: auto, torch, numpy")
    if backend in {"auto", "torch"}:
        try:
            return generate_chen_gong_connections_torch(
                grid,
                pop_pre,
                pop_post,
                weight,
                drange,
                sigma,
                inhibitory_pre=inhibitory_pre,
                pbc=pbc,
                delay_max=delay_max,
                rng=rng,
                post_chunk_size=post_chunk_size,
                device=device,
            )
        except RuntimeError:
            if backend == "torch":
                raise

    rng = rng or np.random.default_rng()
    pre_coords = population_coordinates(grid, pop_pre)
    post_coords = population_coordinates(grid, pop_post)
    grid_shape = (int(grid.shape[0]), int(grid.shape[1]))

    all_i: list[np.ndarray] = []
    all_j: list[np.ndarray] = []
    all_k: list[np.ndarray] = []
    all_d: list[np.ndarray] = []
    for start in range(0, len(post_coords), post_chunk_size):
        stop = min(start + post_chunk_size, len(post_coords))
        chunk = post_coords[start:stop]
        if pbc:
            dist = periodic_distance_matrix(pre_coords, chunk, grid_shape)
        else:
            delta = pre_coords[:, None, :] - chunk[None, :, :]
            dist = np.sqrt(np.sum(delta * delta, axis=2))

        mask = dist <= drange
        if pop_pre == pop_post:
            pre_idx, post_local = np.nonzero(mask)
            post_idx = post_local + start
            keep = pre_idx != post_idx
            pre_idx = pre_idx[keep]
            post_idx = post_idx[keep]
            dist_vals = dist[pre_idx, post_local[keep]]
        else:
            pre_idx, post_local = np.nonzero(mask)
            post_idx = post_local + start
            dist_vals = dist[pre_idx, post_local]

        if pre_idx.size == 0:
            continue
        if inhibitory_pre:
            weights = np.full(pre_idx.size, weight, dtype=float)
        else:
            weights = weight * np.exp(-(dist_vals * dist_vals) / sigma)
        delays = rng.random(pre_idx.size) * delay_max if delay_max else np.zeros(pre_idx.size)
        all_i.append(pre_idx.astype(np.int64))
        all_j.append(post_idx.astype(np.int64))
        all_k.append(weights.astype(float))
        all_d.append(delays.astype(float))

    if not all_i:
        empty_i = np.asarray([], dtype=np.int64)
        empty_f = np.asarray([], dtype=float)
        return empty_i, empty_i.copy(), empty_f, empty_f.copy()
    return tuple(np.concatenate(parts) for parts in (all_i, all_j, all_k, all_d))


@dataclass(frozen=True)
class ChenGong2019Config:
    gsize: int = 250
    grid_shape: tuple[int, int] | None = None
    dt: float = 0.1
    step_tot: int = 100000
    alpha: float = 1.65
    f_external: float = 1e-2
    sample_start: int = 15999
    sample_stop: int = 23999
    sample_stride: int = 10
    post_chunk_size: int = 512
    connection_backend: str = "auto"
    connection_device: str = "auto"

    @property
    def resolved_grid_shape(self) -> tuple[int, int]:
        if self.grid_shape is not None:
            if len(self.grid_shape) != 2:
                raise ValueError("grid_shape must be (height, width)")
            grid_h, grid_w = int(self.grid_shape[0]), int(self.grid_shape[1])
        else:
            grid_h = grid_w = int(self.gsize)
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("grid dimensions must be positive")
        return grid_h, grid_w


def build_chen_gong_2019_input(
    output_path: str | Path,
    ext_current_e: str | Path,
    ext_current_i: str | Path,
    *,
    config: ChenGong2019Config = ChenGong2019Config(),
    loop_num: int = 1,
    seed: int | None = 1,
) -> Path:
    """Translate ``models/main_Chen_and_Gong_2019.m`` into a Python builder."""
    output_path = Path(output_path)
    path, rng = new_ygin_file(loop_num, output_dir=output_path.parent, filename=output_path.name, seed=seed)

    grid_h, grid_w = config.resolved_grid_shape
    grid = np.zeros((grid_h, grid_w), dtype=np.int8)
    grid[1::2, 1::2] = 1
    n = np.bincount(grid.ravel(), minlength=2).astype(np.int64)

    write_synapse_model_choice(path, 1)
    write_basic_para(path, config.dt, config.step_tot, n)

    pop_params = {
        "Cm": 1,
        "tau_ref": 5.0,
        "V_rt": -75.6250,
        "V_lk": -70,
        "V_th": -40,
        "g_lk": 0.050,
    }
    for pop in range(2):
        write_pop_para(path, pop, **pop_params)
        write_elif_neuron_model(path, pop, -60.6250, 6.5625)

    f_ext = config.f_external
    ext_mean_std = [np.full(pop_n, f_ext, dtype=float) for pop_n in n]
    for pop, values in enumerate(ext_mean_std):
        write_ext_conductance_settings(path, pop, values, values)
    write_init_cond(path, np.ones(2), np.zeros(2))
    write_ext_current_pop(path, ext_current_e, 0)
    write_ext_current_pop(path, ext_current_i, 1)

    drange = np.asarray([[45, 45], [45, 45]], dtype=float)
    sigma = np.asarray([[18, 18], [9e9, 9e9]], dtype=float)
    weight_e = 0.13 * config.alpha
    weight_i = 0.035 * config.alpha
    weights = np.asarray([[weight_e, weight_e], [weight_i, weight_i]], dtype=float)
    synapse_type = np.asarray([[0, 0], [1, 1]], dtype=int)

    for pop_pre in range(2):
        for pop_post in range(2):
            i, j, k, d = generate_chen_gong_connections(
                grid,
                pop_pre,
                pop_post,
                weights[pop_pre, pop_post],
                drange[pop_pre, pop_post],
                sigma[pop_pre, pop_post],
                inhibitory_pre=(pop_pre == 1),
                rng=rng,
                post_chunk_size=config.post_chunk_size,
                backend=config.connection_backend,
                device=config.connection_device,
            )
            _require_index_bounds(i, f"syn pop{pop_pre}->pop{pop_post} I", n[pop_pre])
            _require_index_bounds(j, f"syn pop{pop_pre}->pop{pop_post} J", n[pop_post])
            write_chemical_connection(path, synapse_type[pop_pre, pop_post], pop_pre, pop_post, i, j, k, d)

    write_syn_para(
        path,
        tau_decay_GABA=3,
        Dt_trans_AMPA=0.3,
        tau_decay_AMPA=2.0,
        Dt_trans_GABA=0.3,
        V_ex=0,
        V_in=-80,
    )

    sample_steps = np.zeros(config.step_tot, dtype=np.int8)
    sample_stop = min(config.sample_stop, config.step_tot - 1)
    if config.sample_start <= sample_stop:
        sample_steps[np.arange(config.sample_start, sample_stop + 1, config.sample_stride)] = 1
    for pop, pop_n in enumerate(n):
        write_neuron_sampling(path, pop, [1, 0, 0, 0, 0, 0, 0, 0], np.arange(pop_n, dtype=np.int64), sample_steps)

    write_expl_var(path, discard_transient=0, loop_num=loop_num, F=f_ext)
    append_config_text(path, "Python translation of SpikeNet/models/main_Chen_and_Gong_2019.m")
    return path


def assign_inverse_pool_weights_from_repo_b(
    k_in: Sequence[int],
    *,
    k_scale: Sequence[float] | None = None,
    weight_mean: float,
    weight_std: float,
    max_weight: float | None = None,
    random_seed: int | None = None,
) -> list[np.ndarray]:
    """Reuse repository B's inverse-pool implementation when it is available."""
    if str(NETWORK_GENERATOR_ROOT) not in sys.path:
        sys.path.insert(0, str(NETWORK_GENERATOR_ROOT))
    # Repository B imports ipdb at module scope for debugging. Keep this adapter
    # dependency-free without editing B by supplying an inert module when ipdb is
    # not installed in the active environment.
    sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
    from network_generator.sampling.weight_assign.inversepool import InversePoolWeightAssign

    assigner = InversePoolWeightAssign(
        weight_mean=weight_mean,
        weight_std=weight_std,
        max_weight=max_weight,
        random_seed=random_seed,
    )
    return assigner.assign_weights(
        _as_1d_array(k_in, dtype=np.int64),
        None if k_scale is None else _as_1d_array(k_scale, dtype=float),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build a SpikeNet *_in.h5 file from Python.")
    parser.add_argument("output_h5")
    parser.add_argument("--ext-current-e", required=True)
    parser.add_argument("--ext-current-i", required=True)
    parser.add_argument("--gsize", type=int, default=250, help="Square grid size. Ignored when --grid-shape is provided.")
    parser.add_argument("--grid-shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--step-tot", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--connection-backend", choices=("auto", "torch", "numpy"), default="auto")
    parser.add_argument("--connection-device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    cfg = ChenGong2019Config(
        gsize=args.gsize,
        grid_shape=None if args.grid_shape is None else tuple(args.grid_shape),
        step_tot=args.step_tot,
        connection_backend=args.connection_backend,
        connection_device=args.connection_device,
    )
    path = build_chen_gong_2019_input(
        args.output_h5,
        args.ext_current_e,
        args.ext_current_i,
        config=cfg,
        seed=args.seed,
    )
    print(path)


if __name__ == "__main__":
    _main()
