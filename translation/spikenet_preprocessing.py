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
from scipy import stats
from scipy import sparse


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


def write_spike_freq_adpt(file_or_path, pop_ind: int, dg_k: float = 0.01) -> None:
    """Enable SpikeNet spike-frequency adaptation for one 0-based population."""
    _require_nonnegative(np.asarray([pop_ind]), "pop_ind")
    with _managed_h5(file_or_path) as h5f:
        _write_dataset(h5f, f"/config/pops/pop{pop_ind}/INIT010/spike_freq_adpt", 1, dtype=np.int8)
        _write_dataset(h5f, f"/config/pops/pop{pop_ind}/INIT010/dg_K", float(dg_k))


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


def rectangular_lattice_coordinates(shape: tuple[int, int]) -> np.ndarray:
    """Return centered 0-based neuron coordinates in column-major id order."""
    height, width = int(shape[0]), int(shape[1])
    rows, cols = np.unravel_index(np.arange(height * width), (height, width), order="F")
    coords = np.column_stack((rows, cols)).astype(np.float32)
    coords[:, 0] -= (height - 1) / 2.0
    coords[:, 1] -= (width - 1) / 2.0
    return coords


def quasi_lattice_coordinates(
    n_points: int,
    shape: tuple[int, int],
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate GU-style random inhibitory coordinates inside a rectangular lattice."""
    rng = rng or np.random.default_rng()
    height, width = int(shape[0]), int(shape[1])
    coords = np.column_stack((rng.random(int(n_points)) * height, rng.random(int(n_points)) * width)).astype(np.float32)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    coords = coords[order]
    coords[:, 0] -= height / 2.0
    coords[:, 1] -= width / 2.0
    return coords


def _pairwise_periodic_dist_torch(torch, pre_coords, post_coords, shape, device):
    height, width = float(shape[0]), float(shape[1])
    drow = torch.abs(pre_coords[:, None, 0] - post_coords[None, :, 0])
    dcol = torch.abs(pre_coords[:, None, 1] - post_coords[None, :, 1])
    drow = torch.minimum(drow, torch.as_tensor(height, dtype=torch.float32, device=device) - drow)
    dcol = torch.minimum(dcol, torch.as_tensor(width, dtype=torch.float32, device=device) - dcol)
    return torch.sqrt(drow.square() + dcol.square())


def generate_gu_lattice_connections(
    pre_coords_np: np.ndarray,
    post_coords_np: np.ndarray,
    lattice_shape: tuple[int, int],
    *,
    p0: float,
    tau_c: float,
    rng: np.random.Generator | None = None,
    post_chunk_size: int = 512,
    device: str = "auto",
    allow_self: bool = True,
    max_probability: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate GU-style distance-dependent lattice connections.

    This is a vectorized Python/PyTorch adaptation of the GU lattice idea:
    nearby post neurons get larger sampling probability, and each pre neuron
    has expected out-degree ``p0 * N_post``. It intentionally keeps the HDF5
    interface simple by returning pop-local 0-based ``I/J`` arrays.
    """
    torch, torch_device = _resolve_torch_device(device)
    rng = rng or np.random.default_rng()
    pre_coords = torch.as_tensor(pre_coords_np, dtype=torch.float32, device=torch_device)
    post_coords_all = np.asarray(post_coords_np, dtype=np.float32)
    n_pre = int(pre_coords_np.shape[0])
    n_post = int(post_coords_np.shape[0])
    if n_pre == 0 or n_post == 0 or p0 <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    factor_sum = torch.zeros(n_pre, dtype=torch.float32, device=torch_device)
    for start in range(0, n_post, post_chunk_size):
        stop = min(start + post_chunk_size, n_post)
        post_coords = torch.as_tensor(post_coords_all[start:stop], dtype=torch.float32, device=torch_device)
        dist = _pairwise_periodic_dist_torch(torch, pre_coords, post_coords, lattice_shape, torch_device)
        factor = torch.exp(-dist / float(tau_c))
        if not allow_self and n_pre == n_post:
            diag = torch.arange(start, stop, dtype=torch.long, device=torch_device)
            valid_diag = diag < n_pre
            if bool(torch.any(valid_diag)):
                factor[diag[valid_diag], diag[valid_diag] - start] = 0.0
        factor_sum += factor.sum(dim=1)

    factor_sum = torch.clamp(factor_sum, min=1e-12)
    expected_out = float(p0) * float(n_post)

    all_i: list[np.ndarray] = []
    all_j: list[np.ndarray] = []
    for start in range(0, n_post, post_chunk_size):
        stop = min(start + post_chunk_size, n_post)
        post_coords = torch.as_tensor(post_coords_all[start:stop], dtype=torch.float32, device=torch_device)
        dist = _pairwise_periodic_dist_torch(torch, pre_coords, post_coords, lattice_shape, torch_device)
        factor = torch.exp(-dist / float(tau_c))
        if not allow_self and n_pre == n_post:
            diag = torch.arange(start, stop, dtype=torch.long, device=torch_device)
            valid_diag = diag < n_pre
            if bool(torch.any(valid_diag)):
                factor[diag[valid_diag], diag[valid_diag] - start] = 0.0
        prob = torch.clamp(expected_out * factor / factor_sum[:, None], min=0.0, max=float(max_probability))
        mask = torch.rand(prob.shape, dtype=torch.float32, device=torch_device) < prob
        pre_idx, post_local = torch.nonzero(mask, as_tuple=True)
        if pre_idx.numel():
            all_i.append(pre_idx.detach().cpu().numpy().astype(np.int64, copy=False))
            all_j.append((post_local + start).detach().cpu().numpy().astype(np.int64, copy=False))
        del post_coords, dist, factor, prob, mask
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_i:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    return np.concatenate(all_i), np.concatenate(all_j)


def assign_gu_ee_weights(
    j_post: np.ndarray,
    n_post: int,
    *,
    weight_mean: float,
    weight_std: float,
    rng: np.random.Generator | None = None,
    use_inverse_pool: bool = True,
    random_seed: int | None = None,
) -> np.ndarray:
    """Assign GU-style E/E weights with inverse-pool fallback semantics."""
    rng = rng or np.random.default_rng(random_seed)
    j_arr = _as_1d_array(j_post, dtype=np.int64)
    in_degree = np.bincount(j_arr, minlength=int(n_post)).astype(np.int64)
    if j_arr.size == 0:
        return np.asarray([], dtype=float)

    if use_inverse_pool and np.any(in_degree > 0):
        try:
            k_cell = assign_inverse_pool_weights_from_repo_b(
                in_degree,
                k_scale=np.sqrt(in_degree),
                weight_mean=weight_mean,
                weight_std=weight_std,
                random_seed=random_seed,
            )
            weights = np.zeros(j_arr.size, dtype=float)
            for post in range(int(n_post)):
                mask = j_arr == post
                count = int(np.sum(mask))
                if count == 0:
                    continue
                vals = np.asarray(k_cell[post], dtype=float).reshape(-1)
                if vals.size != count:
                    vals = np.resize(vals, count)
                weights[mask] = vals
            return weights
        except Exception:
            pass

    var = float(weight_std) ** 2
    mean_sq = float(weight_mean) ** 2
    sigma = np.sqrt(np.log(var / mean_sq + 1.0)) if var > 0 else 0.0
    mu = np.log(mean_sq / np.sqrt(var + mean_sq)) if var > 0 else np.log(float(weight_mean))
    weights = rng.lognormal(mu, sigma, j_arr.size)
    scale = np.sqrt(in_degree.astype(float))
    total = float(np.sum(weights))
    target = np.zeros_like(scale, dtype=float)
    if np.sum(scale) > 0:
        target = total * scale / np.sum(scale)
    for post in range(int(n_post)):
        mask = j_arr == post
        subtotal = float(np.sum(weights[mask]))
        if subtotal > 0 and target[post] > 0:
            weights[mask] *= target[post] / subtotal
    return weights


def correlated_lognormal_degree(
    n: int,
    deg_mean: float,
    deg_std: float,
    corr: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Translate ``my_logn_rand(..., 'mu_sigma_log')`` for a two-column degree pair."""
    mu = np.asarray([deg_mean, deg_mean], dtype=float)
    sigma = np.asarray([deg_std, deg_std], dtype=float)
    mu_norm = np.log((mu * mu) / np.sqrt(sigma * sigma + mu * mu))
    sigma_norm = np.sqrt(np.log((sigma * sigma) / (mu * mu) + 1.0))
    sigma_down = np.tile(sigma_norm.reshape(1, -1), (2, 1))
    sigma_across = np.tile(sigma_norm.reshape(-1, 1), (1, 2))
    corr_mat = np.asarray([[1.0, corr], [corr, 1.0]], dtype=float)
    cov = np.log(corr_mat * np.sqrt(np.exp(sigma_down**2) - 1.0) * np.sqrt(np.exp(sigma_across**2) - 1.0) + 1.0)
    return np.exp(rng.multivariate_normal(mu_norm, cov, int(n)))


def correlated_poisson_degree(
    n: int,
    deg_mean: float,
    corr: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Translate the Gaussian-copula part of ``poissrnd_2D``."""
    cov = np.asarray([[1.0, corr], [corr, 1.0]], dtype=float)
    normal = rng.multivariate_normal(np.zeros(2), cov, int(n))
    p = stats.norm.cdf(normal)
    return stats.poisson.ppf(p, deg_mean)


def hybrid_degree_strict(
    n: int,
    deg_mean: float,
    deg_std: float,
    corr: float,
    hybrid: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate GU ``hybrid_degree`` into Python with 0-based arrays."""
    mismatch_tol = float(deg_mean) / 2.0
    mismatch = mismatch_tol + 1.0
    degree_pair = None
    while abs(mismatch) > mismatch_tol:
        deg_logn = np.ceil(correlated_lognormal_degree(n, deg_mean, deg_std, corr, rng=rng))
        deg_pois = correlated_poisson_degree(n, deg_mean, corr, rng=rng)
        degree_pair = np.asarray(deg_pois, dtype=float)
        n_logn = int(round(int(n) * float(hybrid)))
        if n_logn > 0:
            logn_ind = rng.permutation(int(n))[:n_logn]
            degree_pair[logn_ind, :] = deg_logn[logn_ind, :]
        degree_pair = np.nan_to_num(degree_pair, nan=0.0, posinf=0.0, neginf=0.0)
        degree_pair = np.maximum(np.ceil(degree_pair), 0)
        degree_pair = np.minimum(degree_pair, max(int(n) - 1, 0))
        mismatch = int(np.sum(degree_pair[:, 0]) - np.sum(degree_pair[:, 1]))

    assert degree_pair is not None
    mismatch_i = int(mismatch)
    if mismatch_i != 0:
        adjust_ind = rng.permutation(int(n))[: abs(mismatch_i)]
        if mismatch_i > 0:
            degree_pair[adjust_ind, 1] += 1
        else:
            degree_pair[adjust_ind, 0] += 1
    return degree_pair[:, 0].astype(np.int64), degree_pair[:, 1].astype(np.int64)


def _normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0.0
    norm = np.linalg.norm(arr)
    if norm <= 0:
        return arr
    return arr / norm


def compute_common_neighbor_torch(
    i_pre: np.ndarray,
    j_post: np.ndarray,
    n: int,
    *,
    device: str = "auto",
) -> np.ndarray:
    """Compute ``cn = A.T @ A`` for GU E/E iteration using torch when possible."""
    torch, torch_device = _resolve_torch_device(device)
    dense = torch.zeros((int(n), int(n)), dtype=torch.float32, device=torch_device)
    if len(i_pre):
        dense[
            torch.as_tensor(i_pre, dtype=torch.long, device=torch_device),
            torch.as_tensor(j_post, dtype=torch.long, device=torch_device),
        ] = 1.0
    cn = dense.transpose(0, 1).matmul(dense)
    cn.fill_diagonal_(0.0)
    result = cn.detach().cpu().numpy()
    del dense, cn
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def generate_ij_2d_strict(
    degree_in_0: np.ndarray,
    degree_out_0: np.ndarray,
    tau_d: float,
    cn_scale_wire: float,
    iter_num: int,
    *,
    lattice_shape: tuple[int, int],
    rng: np.random.Generator,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strict Python translation of GU ``generate_IJ_2D`` core logic.

    The implementation keeps the Matlab control flow: each iteration rebuilds
    ``I/J`` from scratch, then computes ``cn=A.T@A`` for the next iteration.
    Indices returned are 0-based.
    """
    n = int(len(degree_in_0))
    coords = rectangular_lattice_coordinates(lattice_shape)
    if coords.shape[0] != n:
        raise ValueError(f"lattice_shape has {coords.shape[0]} nodes but degree arrays have {n}")
    torch, torch_device = _resolve_torch_device(device)
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=torch_device)
    dist_full = _pairwise_periodic_dist_torch(torch, coords_t, coords_t, lattice_shape, torch_device).detach().cpu().numpy()

    cn = None
    cn_min = 0.0
    cn_span = 1.0
    out_degree = np.minimum(_as_1d_array(degree_out_0, dtype=np.int64), n - 1)
    in_degree_base = np.minimum(_as_1d_array(degree_in_0, dtype=np.float64), n - 1)

    final_i = np.asarray([], dtype=np.int64)
    final_j = np.asarray([], dtype=np.int64)
    final_dist = np.asarray([], dtype=float)
    for _ in range(int(iter_num)):
        all_i: list[np.ndarray] = []
        all_j: list[np.ndarray] = []
        all_dist: list[np.ndarray] = []
        degree_in_left = in_degree_base.copy()
        order = rng.permutation(n)
        for pre in order:
            need = int(out_degree[pre])
            if need <= 0:
                continue
            dist_factor = np.exp(-dist_full[:, pre] / float(tau_d))
            dist_factor[pre] = 0.0
            if cn is None:
                cn_factor = np.ones(n, dtype=float)
            else:
                cn_factor = 1.0 + (cn[:, pre] - cn_min) / cn_span * (float(cn_scale_wire) - 1.0)
                cn_factor = np.nan_to_num(cn_factor, nan=1.0, posinf=1.0, neginf=1.0)
            degree_factor = degree_in_left.copy()
            low = degree_factor <= 4
            degree_factor[low] = np.power(np.maximum(degree_factor[low], 0.0), 6) / (4.0**6)
            joint = _normalize_nonnegative(dist_factor * degree_factor * cn_factor)
            valid = np.isfinite(joint) & (joint > 0)
            if not np.any(valid):
                continue
            scores = np.full(n, np.inf, dtype=float)
            scores[valid] = rng.random(int(np.sum(valid))) / joint[valid]
            chosen = np.argsort(scores)[: min(need, int(np.sum(valid)))]
            chosen = chosen[np.isfinite(scores[chosen])]
            if chosen.size == 0:
                continue
            degree_in_left[chosen] -= 1
            degree_in_left[degree_in_left == 0] = np.nan
            all_i.append(np.full(chosen.size, pre, dtype=np.int64))
            all_j.append(chosen.astype(np.int64))
            all_dist.append(dist_full[chosen, pre].astype(float))

        if all_i:
            final_i = np.concatenate(all_i)
            final_j = np.concatenate(all_j)
            final_dist = np.concatenate(all_dist)
        else:
            final_i = np.asarray([], dtype=np.int64)
            final_j = np.asarray([], dtype=np.int64)
            final_dist = np.asarray([], dtype=float)

        cn = compute_common_neighbor_torch(final_i, final_j, n, device=device)
        upper = cn[np.triu_indices(n, 1)]
        cn_min = float(np.min(upper)) if upper.size else 0.0
        cn_max = float(np.max(upper)) if upper.size else cn_min
        cn_span = max(cn_max - cn_min, 1.0)

    return final_i, final_j, final_dist


def lattice_to_lattice_strict(
    pre_coords: np.ndarray,
    post_coords: np.ndarray,
    lattice_shape: tuple[int, int],
    tau_c: float,
    p0: float,
    *,
    rng: np.random.Generator,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Translate GU ``Lattice2Lattice`` sampling with 0-based indices."""
    torch, torch_device = _resolve_torch_device(device)
    pre_t = torch.as_tensor(np.asarray(pre_coords, dtype=np.float32), dtype=torch.float32, device=torch_device)
    post_t = torch.as_tensor(np.asarray(post_coords, dtype=np.float32), dtype=torch.float32, device=torch_device)
    dist = _pairwise_periodic_dist_torch(torch, pre_t, post_t, lattice_shape, torch_device).detach().cpu().numpy()
    n_pre, n_post = dist.shape
    all_i: list[np.ndarray] = []
    all_j: list[np.ndarray] = []
    for pre in range(n_pre):
        count = int(rng.poisson(float(n_post) * float(p0)))
        count = min(count, n_post)
        if count <= 0:
            continue
        dist_factor = np.exp(-dist[pre] / float(tau_c))
        valid = dist_factor > 0
        scores = np.full(n_post, np.inf, dtype=float)
        scores[valid] = rng.random(int(np.sum(valid))) / dist_factor[valid]
        chosen = np.argsort(scores)[:count]
        all_i.append(np.full(chosen.size, pre, dtype=np.int64))
        all_j.append(chosen.astype(np.int64))
    if not all_i:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    return np.concatenate(all_i), np.concatenate(all_j)


def g_to_epsp_linear(g: float) -> float:
    """Approximate SpikeNet ``g_EPSP_conversion`` fit on the default 0.001..0.010 range."""
    g_values = np.linspace(0.001, 0.010, 10)
    epsp = []
    dt = 0.1
    cm = 0.25
    v_lk = -70.0
    g_lk = 0.0167
    v_rev = 0.0
    tau_r = 1.0
    tau_d = 5.0
    v_th = -50.0
    for gv in g_values:
        v_old = v_lk
        peaked = False
        tau_r_step_left = round(tau_r / dt)
        s = 0.0
        while (not peaked) and v_old < v_th:
            if tau_r_step_left > 0:
                s = s + 1.0 / round(tau_r / dt) * (1.0 - s)
                tau_r_step_left -= 1
            current = -gv * s * (v_old - v_rev)
            i_lk = -g_lk * (v_old - v_lk)
            v_new = v_old + ((current + i_lk) / cm) * dt
            peaked = v_new < v_old
            v_old = v_new
            s = s * np.exp(-dt / tau_d)
            if v_old >= v_th:
                v_old = np.nan
                break
        epsp.append(v_old - v_lk)
    coef = np.polyfit(g_values, np.asarray(epsp, dtype=float), 1)
    return float(np.polyval(coef, float(g)))


@dataclass(frozen=True)
class GU2018Config:
    lattice_shape: tuple[int, int] = (63, 63)
    n_i: int | None = 1000
    dt: float = 0.1
    step_tot: int = 100000
    tau_ref: float = 4.0
    delay_max: float = 4.0
    dg_k: float = 0.01
    p_mat: tuple[tuple[float, float], tuple[float, float]] = ((0.16, 0.2), (0.2, 0.4))
    zeta: float = 27 / 8
    g_ee_mu: float = 4e-3
    g_ee_std: float = 1e-3
    g_ie: float = 5e-3
    g_ii: float = 25e-3
    tau_c_ee: float = 8.0
    tau_c_ie: float = 10.0
    tau_c_i: float = 20.0
    post_chunk_size: int = 512
    connection_device: str = "auto"
    max_connection_probability: float = 1.0
    degree_hybrid: float = 0.4
    degree_cv: float = 0.2
    in_out_corr: float = 0.13
    common_neighbor_scale: float = 2.0
    common_neighbor_iterations: int = 10

    @property
    def resolved_lattice_shape(self) -> tuple[int, int]:
        if len(self.lattice_shape) != 2:
            raise ValueError("lattice_shape must be (height, width)")
        height, width = int(self.lattice_shape[0]), int(self.lattice_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("lattice dimensions must be positive")
        return height, width

    @property
    def n_e(self) -> int:
        height, width = self.resolved_lattice_shape
        return int(height * width)

    @property
    def resolved_n_i(self) -> int:
        if self.n_i is not None:
            return int(self.n_i)
        return max(1, int(round(self.n_e / 4)))


@dataclass(frozen=True)
class GU2018SynapseBlock:
    syn_type: int
    pop_pre: int
    pop_post: int
    i_pre: np.ndarray
    j_post: np.ndarray
    weights: np.ndarray
    delays: np.ndarray


@dataclass(frozen=True)
class GU2018Network:
    config: GU2018Config
    n_by_pop: np.ndarray
    lattice_shape: tuple[int, int]
    input_event_files: tuple[str, str]
    synapses: tuple[GU2018SynapseBlock, ...]
    sample_e: np.ndarray
    sample_i: np.ndarray


def build_gu_2018_strict_network(
    ext_input_e: str | Path,
    ext_input_i: str | Path | None = None,
    *,
    config: GU2018Config = GU2018Config(),
    seed: int | None = 1,
) -> GU2018Network:
    """Build the strict GU network in memory without writing ``*_in.h5``."""
    rng = np.random.default_rng(seed)
    lattice_shape = config.resolved_lattice_shape
    n_e = config.n_e
    n_i = config.resolved_n_i
    n = np.asarray([n_e, n_i], dtype=np.int64)
    p_mat = np.asarray(config.p_mat, dtype=float)

    deg_mean = n_e * p_mat[0, 0]
    deg_std = config.degree_cv * deg_mean
    deg_in_0, deg_out_0 = hybrid_degree_strict(
        n_e,
        deg_mean,
        deg_std,
        config.in_out_corr,
        config.degree_hybrid,
        rng=rng,
    )
    i_ee, j_ee, _ = generate_ij_2d_strict(
        deg_in_0,
        deg_out_0,
        config.tau_c_ee,
        config.common_neighbor_scale,
        config.common_neighbor_iterations,
        lattice_shape=lattice_shape,
        rng=rng,
        device=config.connection_device,
    )

    epsp_mu = g_to_epsp_linear(config.g_ee_mu)
    epsp_sigma = 1.0
    mu_p = np.log((epsp_mu**2) / np.sqrt(epsp_sigma**2 + epsp_mu**2))
    s_p = np.sqrt(np.log(epsp_sigma**2 / (epsp_mu**2) + 1.0))
    mu_p = mu_p + s_p**2
    epsp_mu_norm = mu_p - s_p**2
    epsp_sigma_norm = s_p
    epsp_pool = rng.lognormal(epsp_mu_norm, epsp_sigma_norm, max(i_ee.size, 1))
    epsp_pool = epsp_pool[epsp_pool <= 20.0]
    while epsp_pool.size < i_ee.size:
        extra = rng.lognormal(epsp_mu_norm, epsp_sigma_norm, i_ee.size - epsp_pool.size)
        epsp_pool = np.concatenate([epsp_pool, extra[extra <= 20.0]])
    k_ee = assign_gu_ee_weights(
        j_ee,
        n_e,
        weight_mean=config.g_ee_mu,
        weight_std=config.g_ee_std,
        rng=rng,
        use_inverse_pool=True,
        random_seed=seed,
    )
    d_ee = rng.random(i_ee.size) * config.delay_max

    coords_e = rectangular_lattice_coordinates(lattice_shape)
    coords_i = quasi_lattice_coordinates(n_i, lattice_shape, rng=rng)

    i_ie, j_ie = lattice_to_lattice_strict(
        coords_i,
        coords_e,
        lattice_shape,
        config.tau_c_i,
        p_mat[1, 0],
        rng=rng,
        device=config.connection_device,
    )
    in_weight_ee = np.bincount(j_ee, weights=k_ee, minlength=n_e)
    count_ie = np.bincount(j_ie, minlength=n_e)
    g_ei_mu = config.g_ee_mu * config.zeta
    mu_ie = np.zeros(n_e, dtype=float)
    valid_ie = count_ie > 0
    mu_ie[valid_ie] = (in_weight_ee[valid_ie] / count_ie[valid_ie]) * (g_ei_mu / config.g_ee_mu)
    k_ie = np.abs(rng.normal(mu_ie[j_ie], np.maximum(mu_ie[j_ie] * 0.25, 1e-12))) if j_ie.size else np.asarray([], dtype=float)
    d_ie = rng.random(i_ie.size) * config.delay_max

    i_ei, j_ei = lattice_to_lattice_strict(
        coords_e,
        coords_i,
        lattice_shape,
        config.tau_c_ie,
        p_mat[0, 1],
        rng=rng,
        device=config.connection_device,
    )
    d_ei = rng.random(i_ei.size) * config.delay_max

    i_ii, j_ii = lattice_to_lattice_strict(
        coords_i,
        coords_i,
        lattice_shape,
        config.tau_c_i,
        p_mat[1, 1],
        rng=rng,
        device=config.connection_device,
    )
    keep = i_ii != j_ii
    i_ii = i_ii[keep]
    j_ii = j_ii[keep]
    d_ii = rng.random(i_ii.size) * config.delay_max

    sample_e = rng.choice(n_e, size=min(20, n_e), replace=False).astype(np.int64)
    sample_i = np.arange(min(2, n_i), dtype=np.int64)
    return GU2018Network(
        config=config,
        n_by_pop=n,
        lattice_shape=lattice_shape,
        input_event_files=(str(ext_input_e), str(ext_input_i if ext_input_i is not None else ext_input_e)),
        synapses=(
            GU2018SynapseBlock(0, 0, 0, i_ee, j_ee, k_ee, d_ee),
            GU2018SynapseBlock(1, 1, 0, i_ie, j_ie, k_ie, d_ie),
            GU2018SynapseBlock(0, 0, 1, i_ei, j_ei, np.full(i_ei.size, config.g_ie, dtype=float), d_ei),
            GU2018SynapseBlock(1, 1, 1, i_ii, j_ii, np.full(i_ii.size, config.g_ii, dtype=float), d_ii),
        ),
        sample_e=sample_e,
        sample_i=sample_i,
    )


def write_gu_2018_network_h5(
    output_path: str | Path,
    network: GU2018Network,
    *,
    loop_num: int = 1,
    seed: int | None = 1,
) -> Path:
    """Write an in-memory strict GU network to SpikeNet-style ``*_in.h5``."""
    output_path = Path(output_path)
    path, _ = new_ygin_file(loop_num, output_dir=output_path.parent, filename=output_path.name, seed=seed)
    config = network.config
    write_basic_para(path, config.dt, config.step_tot, network.n_by_pop)
    for pop in range(2):
        write_pop_para(path, pop, tau_ref=config.tau_ref)
    write_spike_freq_adpt(path, 0, config.dg_k)
    write_ext_current_pop(path, network.input_event_files[0], 0)
    write_ext_current_pop(path, network.input_event_files[1], 1)
    write_syn_para(path, tau_decay_GABA=3)
    write_init_cond(path, [0.1, 0.0], [0.0, 0.0])

    for block in network.synapses:
        write_chemical_connection(
            path,
            block.syn_type,
            block.pop_pre,
            block.pop_post,
            block.i_pre,
            block.j_post,
            block.weights,
            block.delays,
        )

    sample_steps = np.ones(config.step_tot, dtype=np.int8)
    write_neuron_sampling(path, 1, [1, 1, 1, 1, 0, 0, 1, 0], network.sample_i, sample_steps)
    write_neuron_sampling(path, 0, [1, 1, 1, 1, 0, 0, 1, 1], network.sample_e, sample_steps)
    write_expl_var(
        path,
        discard_transient=0,
        loop_num=loop_num,
        gu_lattice_shape=f"{network.lattice_shape[0]}x{network.lattice_shape[1]}",
        gu_n_i=int(network.n_by_pop[1]),
        gu_mode="strict",
    )
    append_config_text(path, "Strict Python translation of SpikeNet/models/main_GU_et_al_2018.m")
    return path


def build_gu_2018_strict_input(
    output_path: str | Path,
    ext_input_e: str | Path,
    ext_input_i: str | Path | None = None,
    *,
    config: GU2018Config = GU2018Config(),
    loop_num: int = 1,
    seed: int | None = 1,
) -> Path:
    """Strict GU 2018 preprocessing translation with common-neighbor E/E iteration."""
    network = build_gu_2018_strict_network(ext_input_e, ext_input_i, config=config, seed=seed)
    return write_gu_2018_network_h5(output_path, network, loop_num=loop_num, seed=seed)


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
    parser = argparse.ArgumentParser(description="Build a GU-style SpikeNet *_in.h5 file from Python.")
    parser.add_argument("output_h5")
    parser.add_argument("--event-e", required=True)
    parser.add_argument("--event-i")
    parser.add_argument("--lattice-shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=(63, 63))
    parser.add_argument("--n-i", type=int)
    parser.add_argument("--step-tot", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--connection-device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--post-chunk-size", type=int, default=512)
    parser.add_argument("--common-neighbor-iterations", type=int, default=10)
    args = parser.parse_args()

    cfg = GU2018Config(
        lattice_shape=(int(args.lattice_shape[0]), int(args.lattice_shape[1])),
        n_i=args.n_i,
        step_tot=args.step_tot,
        post_chunk_size=args.post_chunk_size,
        connection_device=args.connection_device,
        common_neighbor_iterations=args.common_neighbor_iterations,
    )
    path = build_gu_2018_strict_input(
        args.output_h5,
        args.event_e,
        args.event_i,
        config=cfg,
        seed=args.seed,
    )
    print(path)


if __name__ == "__main__":
    _main()
