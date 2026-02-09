"""Dynamic Mode Decomposition with control (DMDc) utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import h5py
from pydmd import DMDc
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
from control import ss
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.lib.format import open_memmap



# Progress bar (Necessary reassurance considering the long execution time)
def _update_progress(step: int, total: int, prefix: str = "") -> None:
    if total <= 0:
        return
    bar_len = 30
    step = min(step, total)
    filled_len = int(bar_len * step // total)
    bar = "█" * filled_len + "─" * (bar_len - filled_len)
    percent = 100.0 * step / total
    print(f"\r{prefix} [{bar}] {percent:5.1f}% ({step}/{total})", end="", flush=True)
    if step == total:
        print()



# Data loading, POD construction, and reconstruction helpers; the high level routines (apart from mode plots) are near bottom.
def _load_grid_and_area(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x/y grid files, trim ghosts, and compute cell areas."""
    x_path = data_dir / "x.dat"
    y_path = data_dir / "y.dat"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"Missing x.dat or y.dat in {data_dir}")

    xg = np.loadtxt(x_path).ravel()
    yg = np.loadtxt(y_path).ravel()

    # Trim ghost nodes: 3 at start, 4 at end for x; 3 at start, 3 at end for y
    xg = xg[3:-4]
    yg = np.flip(yg[3:-3])

    dx = np.diff(xg, append=xg[-1])
    dy = np.abs(np.diff(yg, append=yg[-1]))

    # Grid is ordered (nx, ny) in span_averages
    area = np.outer(dx, dy)
    return xg, yg, area


def _load_rho(data_dir: Path, span_ds: h5py.Dataset, batch_time: int = 16) -> np.memmap:
    """Load (or create) rho.npy from the span-averaged dataset."""

    rho_path = data_dir / "rho.npy"
    T, n_meas, nx, ny = span_ds.shape

    if n_meas < 1:
        raise ValueError("span_average dataset must contain at least one component for rho.")

    if not rho_path.is_file():
        rho_mm = np.lib.format.open_memmap(
            rho_path, mode="w+", dtype=span_ds.dtype, shape=(T, nx, ny)
        )
        print("Creating rho.npy from span_average dataset...")
        for t0 in range(0, T, batch_time):
            t1 = min(t0 + batch_time, T)
            rho_mm[t0:t1] = np.asarray(span_ds[t0:t1, 0, :, :], dtype=span_ds.dtype)
            _update_progress(t1, T, prefix="rho.npy progress")
        rho_mm.flush()
        del rho_mm

    print("Loading rho data...")
    rho_data = np.load(rho_path, mmap_mode="r")
    return rho_data


def _compute_favre_mean_and_weights(
    span_ds: h5py.Dataset,
    rho_data: np.memmap,
    velocity_components: Sequence[int],
    area_flat: np.ndarray,
    batch_time: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Favre means and compressible, non-uniform grid weights."""

    T, _, nx, ny = span_ds.shape
    nF = nx * ny
    n_comp = len(velocity_components)

    rho_sum_flat = np.zeros(nF, dtype=np.float64)
    rho_phi_sum_flat = np.zeros((n_comp, nF), dtype=np.float64)

    for t0 in range(0, T, batch_time):
        t1 = min(t0 + batch_time, T)
        rho_blk = np.asarray(rho_data[t0:t1], dtype=np.float64)
        phi_blk = np.asarray(span_ds[t0:t1, velocity_components, :, :], dtype=np.float64)

        rho_sum_flat += rho_blk.sum(axis=0, dtype=np.float64).ravel(order="C")
        for ci in range(n_comp):
            rho_phi_sum_flat[ci] += (
                rho_blk * phi_blk[:, ci]
            ).sum(axis=0, dtype=np.float64).ravel(order="C")
        _update_progress(t1, T, prefix="Favre mean computation")

    eps = 1e-30
    den = np.where(rho_sum_flat > eps, rho_sum_flat, 1.0)
    favre_mean_flat = rho_phi_sum_flat / den

    rho_mean_flat = rho_sum_flat / T
    w_flat_base = rho_mean_flat * area_flat
    w_sum = float(w_flat_base.sum())
    if w_sum <= 0.0:
        raise ValueError("Non-positive total weight; check rho / area inputs.")
    w_flat_base /= w_sum

    sqrt_w_base = np.sqrt(w_flat_base).astype(np.float64)
    sqrt_w_safe_base = np.where(sqrt_w_base > 0.0, sqrt_w_base, 1.0)
    zero_w_mask_base = sqrt_w_base == 0.0

    return favre_mean_flat, w_flat_base, sqrt_w_safe_base, zero_w_mask_base


def _energy_rank_from_eigs(eigvals: np.ndarray, energy_pct: float) -> int:
    if not (0 < energy_pct <= 100.0):
        raise ValueError("energy_pct must be in the interval (0, 100].")
    energies = np.maximum(eigvals, 0.0)
    cumulative = np.cumsum(energies)
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("Total POD energy is non-positive; check input data.")
    energy_ratio = cumulative / total
    rank = int(np.searchsorted(energy_ratio, energy_pct / 100.0) + 1)
    return rank


def _build_pod_basis(
    span_ds: h5py.Dataset,
    favre_mean_flat: np.ndarray,
    w_flat_base: np.ndarray,
    sqrt_w_safe_base: np.ndarray,
    zero_w_mask_base: np.ndarray,
    velocity_components: Sequence[int],
    energy_pct: float,
    basis_path: Path,
    block_rows: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-core Favre-weighted POD returning spatial basis and singular values."""

    T, _, nx, ny = span_ds.shape
    nF = nx * ny
    n_comp = len(velocity_components)
    n_features_total = n_comp * nF

    C = np.zeros((T, T), dtype=np.float64)

    for comp_idx, comp in enumerate(velocity_components):
        for x0 in range(0, nx, block_rows):
            x1 = min(x0 + block_rows, nx)
            block_len = (x1 - x0) * ny
            flat_slice = slice(x0 * ny, x1 * ny)

            X_block = np.asarray(span_ds[:, comp, x0:x1, :], dtype=np.float64).reshape(
                T, block_len, order="C"
            )
            mean_block = favre_mean_flat[comp_idx, flat_slice]
            sqrt_w_block = np.sqrt(w_flat_base[flat_slice])

            Xbw = (X_block - mean_block[None, :]) * sqrt_w_block[None, :]
            C += Xbw @ Xbw.T
            _update_progress(x1, nx, prefix=f"Covariance build (comp {comp})")

    eigvals, U = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    U = U[:, idx]

    rank = _energy_rank_from_eigs(eigvals, energy_pct)
    if rank > T:
        raise ValueError(f"Requested rank {rank} exceeds snapshot count {T}.")

    U_r = U[:, :rank].astype(np.float64, copy=False)
    sigma = np.sqrt(np.maximum(eigvals[:rank], 0.0))
    inv_sig = np.where(sigma > 0, 1.0 / sigma, 0.0)[None, :]

    V_r_mm = np.memmap(basis_path, mode="w+", dtype=np.float32, shape=(n_features_total, rank))

    for comp_idx, comp in enumerate(velocity_components):
        offset = comp_idx * nF
        for x0 in range(0, nx, block_rows):
            x1 = min(x0 + block_rows, nx)
            block_len = (x1 - x0) * ny
            flat_slice = slice(x0 * ny, x1 * ny)
            sl_total = slice(offset + x0 * ny, offset + x1 * ny)

            X_block = np.asarray(span_ds[:, comp, x0:x1, :], dtype=np.float64).reshape(
                T, block_len, order="C"
            )
            mean_block = favre_mean_flat[comp_idx, flat_slice]
            sqrt_w_block = np.sqrt(w_flat_base[flat_slice])
            sqrt_w_safe_block = sqrt_w_safe_base[flat_slice]
            zero_mask_block = zero_w_mask_base[flat_slice]

            Xbw = (X_block - mean_block[None, :]) * sqrt_w_block[None, :]
            Vw_block = (Xbw.T @ U_r) * inv_sig
            V_block = Vw_block / sqrt_w_safe_block[:, None]
            if np.any(zero_mask_block):
                V_block[zero_mask_block, :] = 0.0

            V_r_mm[sl_total, :] = V_block.astype(np.float32, copy=False)
            _update_progress(x1, nx, prefix=f"Basis build (comp {comp})")

    V_r_mm.flush()
    del V_r_mm

    V_r = np.memmap(basis_path, mode="r", dtype=np.float32, shape=(n_features_total, rank))
    V_r = np.array(V_r, copy=True, order="C")
    return V_r, sigma


def _project_snapshots_to_pod(
    span_ds: h5py.Dataset,
    V_r: np.ndarray,
    favre_mean_flat: np.ndarray,
    w_flat_base: np.ndarray,
    velocity_components: Sequence[int],
    time_batch: int = 16,
    block_rows: int = 16,
) -> np.ndarray:
    """Project full snapshots into POD coordinates using weighted inner product."""

    T, _, nx, ny = span_ds.shape
    nF = nx * ny
    n_comp = len(velocity_components)
    r = V_r.shape[1]

    w_flat_full = np.tile(w_flat_base, n_comp)
    favre_mean_full = favre_mean_flat.reshape(n_comp * nF, order="C")

    A_red = np.zeros((r, T), dtype=np.float64)

    for t0 in range(0, T, time_batch):
        t1 = min(t0 + time_batch, T)
        batch_len = t1 - t0
        A_batch = np.zeros((r, batch_len), dtype=np.float64)

        for comp_idx, comp in enumerate(velocity_components):
            offset = comp_idx * nF
            for x0 in range(0, nx, block_rows):
                x1 = min(x0 + block_rows, nx)
                block_len = (x1 - x0) * ny
                sl_total = slice(offset + x0 * ny, offset + x1 * ny)

                X_block = np.asarray(span_ds[t0:t1, comp, x0:x1, :], dtype=np.float64).reshape(
                    batch_len, block_len, order="C"
                )
                mean_block = favre_mean_full[sl_total]
                w_block = w_flat_full[sl_total]
                V_block = V_r[sl_total, :]

                Xc = X_block - mean_block[None, :]
                A_batch += V_block.T @ (Xc * w_block[None, :]).T

        A_red[:, t0:t1] = A_batch
        _update_progress(t1, T, prefix="Projection to POD space")

    return A_red


def _load_controls(traj_path: Path, n_steps: int) -> tuple[np.ndarray, float]:
    with h5py.File(traj_path, "r") as f:
        if "jet_amplitude" not in f:
            raise KeyError("trajectories.h5 is missing the 'jet_amplitude' dataset")
        U_full = np.asarray(f["jet_amplitude"][:], dtype=np.float64)
        if "dt" in f:
            dt_arr = np.asarray(f["dt"][:], dtype=np.float64).ravel()
            dt = float(dt_arr[0]) if dt_arr.size else 1.0
        else:
            dt = 1.0

    if U_full.ndim != 1:
        U_full = U_full.ravel()

    if U_full.size < n_steps:
        raise ValueError(
            f"Control history too short: need {n_steps} entries but found {U_full.size}."
        )

    U = U_full[:n_steps][np.newaxis, :]
    return U, dt


def init_snapshot_red(
    V_r: np.ndarray,
    favre_mean_flat: np.ndarray,
    w_flat_base: np.ndarray,
    span_ds: h5py.Dataset,
    velocity_components: Sequence[int],
) -> np.ndarray:
    _, _, nx, ny = span_ds.shape
    nF = nx * ny
    n_comp = len(velocity_components)
    w_flat_full = np.tile(w_flat_base, n_comp)
    favre_mean_full = favre_mean_flat.reshape(n_comp * nF, order="C")

    snapshot0 = np.asarray(span_ds[0, velocity_components, :, :], dtype=np.float64).reshape(
        n_comp * nF, order="C"
    )
    a0 = V_r.T @ ((snapshot0 - favre_mean_full) * w_flat_full)
    return a0


def dmdc_sim(
    A: np.ndarray,
    B: np.ndarray,
    U: np.ndarray,
    dt: float,
    x0: np.ndarray,
    V_r: np.ndarray,
    favre_mean_flat: np.ndarray,
    output_h5: Path,
    output_shape: tuple[int, int, int],
    write_batch: int = 16,
) -> Path:
    """Simulate the DMDc ROM in reduced space and reconstruct to disk."""

    n_states = A.shape[0]
    n_steps = U.shape[1]
    nx, ny, n_comp = output_shape
    nF = nx * ny

    X = np.zeros((n_states, n_steps + 1), dtype=np.float64)
    X[:, 0] = x0
    for k in range(n_steps):
        X[:, k + 1] = A.dot(X[:, k]) + B.dot(U[:, k])
        _update_progress(k + 1, n_steps, prefix="Simulating DMDc")

    favre_mean_full = favre_mean_flat.reshape(n_comp * nF, order="C")

    with h5py.File(output_h5, "w") as f:
        ds = f.create_dataset(
            "span_average",
            shape=(n_steps + 1, n_comp, nx, ny),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset("dt", data=dt)

        for t0 in range(0, n_steps + 1, write_batch):
            t1 = min(t0 + write_batch, n_steps + 1)
            coeffs = X[:, t0:t1]
            recon_flat = V_r @ coeffs
            recon_flat += favre_mean_full[:, None]
            recon = recon_flat.T.reshape((t1 - t0, n_comp, nx, ny), order="C")
            ds[t0:t1] = recon.astype(np.float32, copy=False)
            _update_progress(t1, n_steps + 1, prefix="Reconstructing snapshots")

    return output_h5



# Error calculations
def _weighted_error_accumulators(
    fom_ds: h5py.Dataset,
    rom_ds: h5py.Dataset,
    w_flat_base: np.ndarray,
    velocity_components: Sequence[int],
    time_batch: int = 16,
) -> tuple[float, float, float]:
    """Accumulate weighted squared norms for error metrics."""

    T, _, nx, ny = fom_ds.shape
    nF = nx * ny
    n_comp = len(velocity_components)
    w_flat_full = np.tile(w_flat_base, n_comp)

    sum_diff_sq = 0.0
    sum_fom_sq = 0.0
    total_weight = float(T) * float(w_flat_full.sum())

    for t0 in range(0, T, time_batch):
        t1 = min(t0 + time_batch, T)
        fom_block = np.asarray(fom_ds[t0:t1, velocity_components, :, :], dtype=np.float64).reshape(
            t1 - t0, n_comp * nF, order="C"
        )
        rom_block = np.asarray(rom_ds[t0:t1, :, :, :], dtype=np.float64).reshape(
            t1 - t0, n_comp * nF, order="C"
        )

        diff = rom_block - fom_block
        sum_diff_sq += float((w_flat_full[None, :] * diff * diff).sum())
        sum_fom_sq += float((w_flat_full[None, :] * fom_block * fom_block).sum())
        _update_progress(t1, T, prefix="Error accumulation")

    if total_weight <= 0:
        raise ValueError("Non-positive total weight when computing errors.")

    return sum_diff_sq, sum_fom_sq, total_weight


def Erms(sum_diff_sq: float, total_weight: float) -> float:
    return float(np.sqrt(sum_diff_sq / total_weight))


def NRMSE(sum_diff_sq: float, sum_fom_sq: float) -> float:
    denom = float(np.sqrt(sum_fom_sq))
    return float(np.sqrt(sum_diff_sq) / denom) if denom > 0 else float("nan")


def Efrobenius(sum_diff_sq: float, sum_fom_sq: float) -> float:
    denom = float(np.sqrt(sum_fom_sq))
    return float(np.sqrt(sum_diff_sq) / denom) if denom > 0 else float("nan")


def weighted_error_metrics(
    span_avg_file: Path,
    rom_file: Path,
    w_flat_base: np.ndarray,
    velocity_components: Sequence[int],
    time_batch: int = 16,
) -> tuple[float, float, float]:
    with h5py.File(span_avg_file, "r") as f_fom, h5py.File(rom_file, "r") as f_rom:
        fom_ds = f_fom["span_average"]
        rom_ds = f_rom["span_average"]
        sum_diff_sq, sum_fom_sq, total_weight = _weighted_error_accumulators(
            fom_ds, rom_ds, w_flat_base, velocity_components, time_batch
        )

    e_rms = Erms(sum_diff_sq, total_weight)
    nrmse_val = NRMSE(sum_diff_sq, sum_fom_sq)
    efro = Efrobenius(sum_diff_sq, sum_fom_sq)
    return e_rms, nrmse_val, efro

## Too RAM intensive
#def Model_Red_Error(span_avg_file: Path, rom_file: Path) -> np.ndarray:
#    with h5py.File(span_avg_file, "r") as f:
#        FOMdata = f["span_average"][:, 1:4, :, :]
#    with h5py.File(rom_file, "r") as f:
#        ROMdata = f["span_average"][:]
#    RSEdata = np.abs(FOMdata - ROMdata) / np.abs(FOMdata)
#    np.save(rom_file.parent / "SquaredError.npy", RSEdata)
#    return RSEdata

def Model_Red_Error(
    span_avg_file: Path | str,
    rom_file: Path | str,
    time_batch: int = 16,
) -> Path:
    """
    Compute pointwise squared error (FOM - ROM)^2 in a streaming, low-memory
    way.

    The result is stored on disk as an HDF5 file called 'SquaredError.h5'
    next to `rom_file`, with dataset 'SquaredError' of shape
    (T, n_comp, nx, ny). Returns the Path to this HDF5 file.
    """
    span_avg_file = Path(span_avg_file)
    rom_file = Path(rom_file)

    with h5py.File(span_avg_file, "r") as f_fom, h5py.File(rom_file, "r") as f_rom:
        fom_ds = f_fom["span_average"]       # (T_fom, n_meas, nx, ny)
        rom_ds = f_rom["span_average"]       # (T_rom, n_comp, nx, ny)

        T_fom, n_meas, nx, ny = fom_ds.shape
        T_rom, n_comp, nx_rom, ny_rom = rom_ds.shape

        if T_fom != T_rom or nx != nx_rom or ny != ny_rom:
            raise ValueError(
                f"Shape mismatch between FOM {fom_ds.shape} and ROM {rom_ds.shape}"
            )

        # Original code compared ROM against FOM[:, 1:4, ...]
        # Generalize to: use FOM indices 1 .. 1 + n_comp
        start_comp = 1
        end_comp = 1 + n_comp
        if end_comp > n_meas:
            raise ValueError(
                f"ROM has {n_comp} components but FOM only has {n_meas}; "
                f"cannot slice span_average[:, 1:{end_comp}, ...]."
            )

        # Create an HDF5 dataset on disk (float32 to save space)
        rse_path = rom_file.parent / "SquaredError.h5"
        with h5py.File(rse_path, "w") as f_err:
            ds = f_err.create_dataset(
                "SquaredError",
                shape=(T_fom, n_comp, nx, ny),
                dtype="f4",
                # Chunked along time for streaming writes
                chunks=(min(time_batch, T_fom), n_comp, nx, ny),
                compression="gzip",
                compression_opts=4,
            )

            for t0 in range(0, T_fom, time_batch):
                t1 = min(t0 + time_batch, T_fom)

                # Load small time-chunk of FOM and ROM
                fom_block = np.asarray(
                    fom_ds[t0:t1, start_comp:end_comp, :, :],
                    dtype=np.float32,
                )
                rom_block = np.asarray(
                    rom_ds[t0:t1, :, :, :],
                    dtype=np.float32,
                )

                with np.errstate(invalid="ignore"):
                    block_err = (fom_block - rom_block) ** 2

                # Write chunk directly into HDF5 on disk
                ds[t0:t1, :, :, :] = block_err

                _update_progress(t1, T_fom, prefix="Pointwise error")

        print()  # newline after progress bar
        return rse_path

# Plot modal energy spectrum and cumulative energy for POD singular values.
# sigma_path: Path to ``Sigma_<label>.npy`` containing the singular values.
# energy_pct: Cumulative energy percentage to annotate on the plot (0, 100].
def plot_pod_energy_spectrum(
    sigma_path: str | Path,
    max_modes: int | None = None,
    energy_pct: float = 90.0,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:

    sigma_path = Path(sigma_path)
    if not sigma_path.is_file():
        raise FileNotFoundError(f"Singular value file not found: {sigma_path}")

    if not (0 < energy_pct <= 100.0):
        raise ValueError("energy_pct must be in the interval (0, 100].")

    sigma = np.asarray(np.load(sigma_path)).squeeze()
    if sigma.ndim != 1:
        raise ValueError("Singular values must be a 1D array.")

    n_modes = sigma.shape[0]
    if max_modes is None:
        max_modes = n_modes
    if max_modes <= 0:
        raise ValueError("max_modes must be positive.")
    if max_modes > n_modes:
        raise ValueError(
            f"Requested max_modes={max_modes} exceeds available modes ({n_modes})."
        )

    sigma_trunc = sigma[:max_modes]
    energies = sigma_trunc ** 2
    cumulative = np.cumsum(energies)
    total_energy = float(cumulative[-1])
    if total_energy <= 0:
        raise ValueError("Total modal energy is non-positive; check singular values.")
    cumulative_frac = cumulative / total_energy

    mode_indices = np.arange(1, max_modes + 1)
    energy_frac = energy_pct / 100.0
    threshold_idx = int(np.searchsorted(cumulative_frac, energy_frac))
    threshold_mode = min(threshold_idx + 1, max_modes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    energy_line, = ax.plot(mode_indices, energies, marker="o", label="Energy")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Energy")

    ax2 = ax.twinx()
    cumulative_line, = ax2.plot(
        mode_indices,
        cumulative_frac,
        color="tab:orange",
        linestyle="--",
        marker="x",
        label="Cumulative energy",
    )
    ax2.set_ylabel("Cumulative Energy")

    annotation_y = energy_frac
    ax.axvline(threshold_mode, color="k", linestyle=":", alpha=0.7)
    ax2.axhline(energy_frac, color="gray", linestyle="--", alpha=0.5)
    ax2.annotate(
        f"{energy_pct:g}% energy",
        xy=(threshold_mode, annotation_y),
        xytext=(10, 10),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "gray"},
        ha="left",
        va="bottom",
    )

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(
        handles + handles2,
        labels + labels2,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.8),
        bbox_transform=ax.transAxes,
    )
    legend.set_zorder(max(energy_line.get_zorder(), cumulative_line.get_zorder()) + 1)

    fig.tight_layout()
    return fig, ax

# Plot spatial POD modes
# Note: basis_path, sigma_path are paths to DMDcBasis_<label>.npy and Sigma_<label>.npy saved by run_dmdc.
# n_modes is the number of leading modes to visualize (ordered by singular value).
def plot_pod_modes(
    basis_path: str | Path,
    sigma_path: str | Path,
    n_modes: int,
    nx: int,
    ny: int,
    n_comp: int,
    components: Sequence[int] | Sequence[str] | None = None,
    output_path: str | Path,
    nx: int,
    ny: int,
    cmap: str = "viridis",
):

    basis_path = Path(basis_path)
    sigma_path = Path(sigma_path)

    V_r = np.load(basis_path)
    sigma = np.load(sigma_path)

    if V_r.ndim != 2:
        raise ValueError(f"DMDc basis must be 2D (features x modes); got shape {V_r.shape}.")
    if sigma.ndim != 1:
        raise ValueError(f"Sigma must be 1D; got shape {sigma.shape}.")

    n_features, available_modes = V_r.shape
    n_comp = len(velocity_components)
    expected_features = n_comp * nx * ny
    if n_features != expected_features:
        raise ValueError(
            f"Basis shape {V_r.shape} incompatible with nx*ny*n_comp={expected_features}."
        )
    if sigma.shape[0] != available_modes:
        raise ValueError(
            f"Sigma length {sigma.shape[0]} does not match basis mode count {available_modes}."
        )
    if n_modes < 1:
        raise ValueError("n_modes must be positive.")
    if n_modes > available_modes:
        raise ValueError(f"Requested n_modes={n_modes} exceeds available rank {available_modes}.")

    sigma_sq = sigma.astype(float) ** 2
    total_energy = sigma_sq.sum()
    if total_energy <= 0:
        raise ValueError("Total singular value energy must be positive.")
    energy_frac = sigma_sq / total_energy
    cumulative_energy = np.cumsum(energy_frac)

    # Determine component indices and names
    if components is None:
        comp_indices = list(range(n_comp))
        comp_names = [f"component {i}" for i in comp_indices]
    elif all(isinstance(c, int) for c in components):
        comp_indices = list(components)
        if any(c < 0 or c >= n_comp for c in comp_indices):
            raise ValueError(f"Component indices must be in [0, {n_comp}); got {components}.")
        comp_names = [f"component {c}" for c in comp_indices]
    else:
        if len(components) != n_comp:
            raise ValueError("Component name list must have length n_comp when providing strings.")
        comp_indices = list(range(n_comp))
        comp_names = list(components)

    n_rows = len(comp_indices)
    fig, axes = plt.subplots(n_rows, n_modes, figsize=(4 * n_modes, 3 * n_rows), squeeze=False)

    # Load grid
    xg, yg, _ = _load_grid_and_area(output_path / "distribute_save")
    if xg.shape[0] != nx or yg.shape[0] != ny:
        raise ValueError("Grid vectors must have lengths matching nx and ny.")
    Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
    plot_fn = lambda ax, data: ax.pcolormesh(Xg, Yg, data, shading="auto", cmap=cmap)
    x_extent = float(np.ptp(xg)) if xg.size > 0 else float(nx)
    y_extent = float(np.ptp(yg)) if yg.size > 0 else float(ny)

    aspect_ratio = y_extent / x_extent if x_extent != 0 else 1.0

    for j in range(n_modes):
        mode_energy_pct = energy_frac[j] * 100.0
        cum_energy_pct = cumulative_energy[j] * 100.0
        mode = V_r[:, j].reshape(n_comp, nx, ny, order="C")
        for i, (comp_idx, comp_name) in enumerate(zip(comp_indices, comp_names)):
            ax = axes[i, j]
            im = plot_fn(ax, mode[comp_idx])
            ax.set_title(
                f"Mode {j} ({comp_name}) – {mode_energy_pct:.2f}% (cum {cum_energy_pct:.2f}%)"
            )
            fig.colorbar(im, ax=ax)
            if grid is not None:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            ax.set_aspect(aspect_ratio)

    fig.tight_layout()
    return fig, axes

# Plot spatial DMD modes obtained from the reduced A matrix. eigendecomposes A_red_matrix to obtain DMDc eigenvectors, 
# lifts them back to physical space via the POD basis V_r, and plots the real part of the resulting spatial modes.
def plot_dmd_modes(
    A_path: str | Path,
    basis_path: str | Path,
    n_modes: int,
    velocity_components: Sequence[int] | None = None,
    output_path: str | Path,
    nx: int,
    ny: int,
    cmap: str = "viridis",
):

    A_path = Path(A_path)
    basis_path = Path(basis_path)

    A = np.load(A_path)
    V_r = np.load(basis_path)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square 2D matrix; got shape {A.shape}.")
    if V_r.ndim != 2:
        raise ValueError(f"POD basis must be 2D (features x modes); got shape {V_r.shape}.")

    n_features, r = V_r.shape
    n_comp = len(velocity_components)
    expected_features = n_comp * nx * ny
    if n_features != expected_features:
        raise ValueError(
            f"Basis shape {V_r.shape} incompatible with nx*ny*n_comp={expected_features}."
        )
    if A.shape[0] != r:
        raise ValueError(
            f"A shape {A.shape} incompatible with basis rank {r}; expected ({r}, {r})."
        )
    if n_modes < 1:
        raise ValueError("n_modes must be positive.")
    if n_modes > r:
        raise ValueError(f"Requested n_modes={n_modes} exceeds available rank {r}.")

    eigvals, eigvecs = np.linalg.eig(A)
    order = np.argsort(np.abs(eigvals))[::-1]

    imag_tol = 1e-10
    filtered_indices: list[int] = []
    for idx in order:
        eigval = eigvals[idx]
        if np.isclose(eigval.imag, 0.0, atol=imag_tol):
            filtered_indices.append(idx)
        elif eigval.imag >= -imag_tol:
            filtered_indices.append(idx)

    if not filtered_indices:
        raise ValueError("No eigenvalues available after filtering for plotting.")

    n_modes = min(n_modes, len(filtered_indices))

    # Determine component indices and names
    if velocity_components is None:
        comp_indices = list(range(n_comp))
        comp_names = [f"component {i}" for i in comp_indices]
    elif all(isinstance(c, int) for c in velocity_components):
        comp_indices = list(velocity_components)
        if any(c < 0 or c >= n_comp for c in comp_indices):
            raise ValueError(f"Component indices must be in [0, {n_comp}); got {velocity_components}.")
        comp_names = [f"component {c}" for c in comp_indices]
    else:
        if len(velocity_components) != n_comp:
            raise ValueError("Component name list must have length n_comp when providing strings.")
        comp_indices = list(range(n_comp))
        comp_names = list(velocity_components)

    selected_indices = filtered_indices[:n_modes]

    n_rows = len(comp_indices)
    fig, axes = plt.subplots(n_rows, n_modes, figsize=(4 * n_modes, 3 * n_rows), squeeze=False)

    # Load grid
    xg, yg, _ = _load_grid_and_area(output_path / "distribute_save")
    if xg.shape[0] != nx or yg.shape[0] != ny:
        raise ValueError("Grid vectors must have lengths matching nx and ny.")
    Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
    plot_fn = lambda ax, data: ax.pcolormesh(Xg, Yg, data, shading="auto", cmap=cmap)
    x_extent = float(np.ptp(xg)) if xg.size > 0 else float(nx)
    y_extent = float(np.ptp(yg)) if yg.size > 0 else float(ny)

    aspect_ratio = y_extent / x_extent if x_extent != 0 else 1.0

    Phi = V_r @ eigvecs

    for j, mode_idx in enumerate(selected_indices):
        eig = eigvals[mode_idx]
        mode = Phi[:, mode_idx].reshape(n_comp, nx, ny, order="C")
        if np.isclose(eig.imag, 0):
            eig_str = f"λ={eig.real:.3f}{eig.imag:+.3f}i |λ|={abs(eig):.3f}"
        else:
            eig_str = f"λ={eig.real:.3f}±{abs(eig.imag):.3f}i |λ|={abs(eig):.3f}"
        for i, (comp_idx, comp_name) in enumerate(zip(comp_indices, comp_names)):
            ax = axes[i, j]
            im = plot_fn(ax, np.real(mode[comp_idx]))
            ax.set_title(f"Mode {j} ({comp_name}) – {eig_str}")
            fig.colorbar(im, ax=ax)
            if grid is not None:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            ax.set_aspect(aspect_ratio)

    fig.tight_layout()
    return fig, axes



# H2 and Hinf norms
def load_rom_ss(rom_dir: Path, label: str):
    A = np.load(rom_dir / f"A_red_matrix_{label}.npy")
    B = np.load(rom_dir / f"B_red_matrix_{label}.npy")
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    C = np.eye(n_states)
    D = np.zeros((n_states, n_inputs))
    return ss(A, B, C, D)


#def h2_norm_via_lyap(sys):
#    A, B, C, D = sys.A, sys.B, sys.C, sys.D
#    X = solve_continuous_lyapunov(A, -B @ B.T)
#    return float(np.sqrt(np.trace(C @ X @ C.T + D @ D.T)))


#def hinf_norm_approx(sys, wmin=1e-3, wmax=1e3, npts=500):
#    A, B, C, D = sys.A, sys.B, sys.C, sys.D
#    freqs = np.logspace(np.log10(wmin), np.log10(wmax), npts)
#    max_gain = 0.0
#    w_at_max = freqs[0]
#    I = np.eye(A.shape[0])
#    for w in freqs:
#        G = C @ np.linalg.inv(1j * w * I - A) @ B + D
#        sigma = np.linalg.svd(G, compute_uv=False)[0]
#        if sigma > max_gain:
#            max_gain = float(sigma)
#            w_at_max = float(w)
#    return max_gain, w_at_max

def h2_norm_via_lyap(sys):
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    Q = B @ B.T
    X = solve_discrete_lyapunov(A, Q)
    return float(np.sqrt(np.trace(C @ X @ C.T + D @ D.T)))


def hinf_norm_approx(sys, wmin=0.0, wmax=np.pi, npts=500):
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    freqs = np.linspace(wmin, wmax, npts)
    max_gain = 0.0
    w_at_max = freqs[0]
    I = np.eye(A.shape[0])

    for w in freqs:
        z = np.exp(1j * w)             # z lies on the unit circle
        G = C @ np.linalg.inv(z * I - A) @ B + D
        sigma = np.linalg.svd(G, compute_uv=False)[0]
        if sigma > max_gain:
            max_gain = float(sigma)
            w_at_max = float(w)

    return max_gain, w_at_max


#def H2_Hinf_Analysis(rom_dir: Path):
#    energy_labels = ["99pct", "96pct", "93pct", "90pct"]
#    for label in energy_labels:
#        sysr = load_rom_ss(rom_dir, label)
#        h2_val = h2_norm_via_lyap(sysr)
#        hinf_val, w_peak = hinf_norm_approx(sysr, wmin=1e-2, wmax=1e2, npts=800)
#        print(f"[{label}]  ‖G‖₂ ≈ {h2_val:.3e}  ;  ‖G‖∞ ≈ {hinf_val:.3e}  @ ω={w_peak:.2f} rad/s")

def H2_Hinf_Analysis(rom_dir: Path):
    energy_labels = ["99pct", "96pct", "93pct", "90pct"]
    for label in energy_labels:
        sysr = load_rom_ss(rom_dir, label)
        h2_val = h2_norm_via_lyap(sysr)
        hinf_val, w_peak = hinf_norm_approx(sysr, wmin=0.0, wmax=np.pi, npts=800)
        print(f"[{label}]  ‖G‖₂ ≈ {h2_val:.3e}  ;  ‖G‖∞ ≈ {hinf_val:.3e}  @ ω={w_peak:.2f} rad/sample")


# HIGH-LEVEL ROUTINES

# Compute a DMDc ROM using Favre-weighted POD on span-averaged data.
def run_dmdc(
    sa_path: str,
    traj_path: str,
    output_dir: str,
    energy_pct: float,
    velocity_components: Sequence[int] = (1, 2, 3),
    block_rows: int = 16,
    time_batch: int = 16,
) -> Path:

    sa_path = Path(sa_path)
    traj_path = Path(traj_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting DMDc pipeline...")
    print("Loading span-averaged data and preparing grids...")
    with h5py.File(sa_path, "r") as sa_file:
        if "span_average" not in sa_file:
            raise KeyError("span_averages.h5 missing 'span_average' dataset")
        span_ds = sa_file["span_average"]
        T, n_meas, nx, ny = span_ds.shape

        if any(c >= n_meas for c in velocity_components):
            raise ValueError(
                f"Requested velocity components {velocity_components} exceed available measurements ({n_meas})."
            )

        data_dir = sa_path.parent
        xg, yg, area = _load_grid_and_area(data_dir)
        if len(xg) != nx or len(yg) != ny:
            raise ValueError(
                f"Grid length mismatch: x={len(xg)} vs nx={nx}, y={len(yg)} vs ny={ny}"
            )

        print("Building rho field and Favre-weighted statistics...")
        rho_data = _load_rho(data_dir, span_ds, batch_time=time_batch)
        if rho_data.shape != (T, nx, ny):
            raise ValueError(
                f"rho.npy shape {rho_data.shape} incompatible with span_average {(T, nx, ny)}"
            )

        area_flat = area.ravel(order="C")
        favre_mean_flat, w_flat_base, sqrt_w_safe_base, zero_w_mask_base = _compute_favre_mean_and_weights(
            span_ds, rho_data, velocity_components, area_flat
        )

        basis_path = output_dir / "V_r.memmap"
        print("Constructing POD basis (this may take a while)...")
        V_r, sigma = _build_pod_basis(
            span_ds,
            favre_mean_flat,
            w_flat_base,
            sqrt_w_safe_base,
            zero_w_mask_base,
            velocity_components,
            energy_pct,
            basis_path=basis_path,
            block_rows=block_rows,
        )

        print("Projecting snapshots to reduced coordinates...")
        A_red = _project_snapshots_to_pod(
            span_ds,
            V_r,
            favre_mean_flat,
            w_flat_base,
            velocity_components,
            time_batch=time_batch,
            block_rows=block_rows,
        )

    n_steps = A_red.shape[1] - 1
    print("Loading control inputs and fitting DMDc model...")
    U, dt = _load_controls(traj_path, n_steps)

    dmdc = DMDc(svd_rank=V_r.shape[1])
    dmdc.fit(A_red, U)

    A = getattr(dmdc, "A", dmdc.operator.as_numpy_array)
    B = dmdc.B

    label = f"{int(energy_pct)}pct"
    red_dir = output_dir / "ReducedModels"
    red_dir.mkdir(exist_ok=True)

    np.save(red_dir / f"A_red_matrix_{label}.npy", A)
    np.save(red_dir / f"B_red_matrix_{label}.npy", B)
    np.save(red_dir / f"DMDcBasis_{label}.npy", V_r)
    np.save(red_dir / f"Favre_mean_{label}.npy", favre_mean_flat)
    np.save(red_dir / f"Weights_{label}.npy", w_flat_base)
    np.save(red_dir / f"Sigma_{label}.npy", sigma)

    x0 = A_red[:, 0]
    np.save(output_dir / "Init_snap_red.npy", x0)

    print("Simulating reduced-order model and reconstructing snapshots...")
    rom_file = output_dir / "dmdc_span_averages.h5"
    dmdc_sim(
        A,
        B,
        U,
        dt,
        x0,
        V_r,
        favre_mean_flat,
        rom_file,
        (nx, ny, len(velocity_components)),
        write_batch=time_batch,
    )

    print("Computing weighted error metrics...")
    e_rms, nrmse_val, efro = weighted_error_metrics(
        sa_path, rom_file, w_flat_base, velocity_components, time_batch=time_batch
    )

    print(f"Erms (weighted): {e_rms}")
    print(f"NRMSE (weighted): {nrmse_val}")
    print(f"Efrobenius (weighted): {efro}")

    print(f"DMDc results written to {output_dir}")
    return rom_file

# Validate an existing DMDc ROM on a new actuation regime
# The sa_path and traj_path are the paths to the actuation regime, the other parameters are covered earlier
def validate_rom_on_case(
    sa_path: str | Path,
    traj_path: str | Path,
    dmdc_output_dir: str | Path,
    energy_pct: float,
    case_label: str = "case",
    velocity_components: Sequence[int] = (1, 2, 3),
    time_batch: int = 16,
    compute_pointwise_error: bool = False,
) -> tuple[Path, float, float, float]:

    sa_path = Path(sa_path)
    traj_path = Path(traj_path)
    dmdc_output_dir = Path(dmdc_output_dir)

    if not sa_path.is_file():
        raise FileNotFoundError(f"Span-average file not found: {sa_path}")
    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectories file not found: {traj_path}")

    label = f"{int(energy_pct)}pct"
    red_dir = dmdc_output_dir / "ReducedModels"
    if not red_dir.is_dir():
        raise FileNotFoundError(f"ReducedModels directory not found: {red_dir}")

    # --- Load ROM pieces built by run_dmdc ---
    A = np.load(red_dir / f"A_red_matrix_{label}.npy")
    B = np.load(red_dir / f"B_red_matrix_{label}.npy")
    V_r = np.load(red_dir / f"DMDcBasis_{label}.npy")
    favre_mean_flat = np.load(red_dir / f"Favre_mean_{label}.npy")
    w_flat_base = np.load(red_dir / f"Weights_{label}.npy")

    # --- Inspect new FOM data ---
    with h5py.File(sa_path, "r") as sa_file:
        if "span_average" not in sa_file:
            raise KeyError("span_averages.h5 missing 'span_average' dataset")

        span_ds = sa_file["span_average"]
        T, n_meas, nx, ny = span_ds.shape

        if any(c < 0 or c >= n_meas for c in velocity_components):
            raise ValueError(
                f"velocity_components {velocity_components} incompatible with "
                f"span_average shape {span_ds.shape}"
            )

        n_comp = len(velocity_components)
        n_steps = T - 1

        # Consistency check between weights and grid
        nF = nx * ny
        if w_flat_base.size != nF:
            raise ValueError(
                f"Weight vector length {w_flat_base.size} incompatible with "
                f"grid size {nx} x {ny} (= {nF})"
            )

        # --- Initial reduced state x0 ---
        x0 = init_snapshot_red(
            V_r=V_r,
            favre_mean_flat=favre_mean_flat,
            w_flat_base=w_flat_base,
            span_ds=span_ds,
            velocity_components=velocity_components,
        )

    # --- Load control history for the new case ---
    U, dt = _load_controls(traj_path, n_steps)

    # --- Simulate ROM and reconstruct snapshots ---
    case_label_clean = case_label.replace(" ", "_")
    rom_file = dmdc_output_dir / f"dmdc_span_averages_{case_label_clean}_{label}.h5"

    print(f"Simulating ROM for validation case '{case_label_clean}'...")
    dmdc_sim(
        A=A,
        B=B,
        U=U,
        dt=dt,
        x0=x0,
        V_r=V_r,
        favre_mean_flat=favre_mean_flat,
        output_h5=rom_file,
        output_shape=(nx, ny, n_comp),
        write_batch=time_batch,
    )

    # --- Compute global weighted error metrics ---
    print("Computing weighted error metrics for validation case...")
    e_rms, nrmse_val, efro = weighted_error_metrics(
        span_avg_file=sa_path,
        rom_file=rom_file,
        w_flat_base=w_flat_base,
        velocity_components=velocity_components,
        time_batch=time_batch,
    )

    print(f"[{case_label_clean}] Erms (weighted): {e_rms}")
    print(f"[{case_label_clean}] NRMSE (weighted): {nrmse_val}")
    print(f"[{case_label_clean}] Efrobenius (weighted): {efro}")

    # --- Optional: pointwise relative error field ---
    if compute_pointwise_error:
        print("Computing pointwise relative error field (Model_Red_Error)...")
        Model_Red_Error(sa_path, rom_file, time_batch=time_batch)


    return rom_file, e_rms, nrmse_val, efro
