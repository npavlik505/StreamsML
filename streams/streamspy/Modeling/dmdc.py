"""Dynamic Mode Decomposition with control (DMDc) utilities."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py
from pydmd import DMDc
from scipy.linalg import solve_continuous_lyapunov
from control import ss
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Data loading and reconstruction helpers
# ---------------------------------------------------------------------------

def load_h5_data(span_avg_file: Path, traj_file: Path, output_dir: Path, velocity_components: list[int] | tuple[int, ...] = (1, 2, 3)):
    """Load span averaged snapshots and actuation history.
    
    The function also stores the initial snapshot, subsampled control input and
    time step in ``output_dir`` for later use.
    """
    with h5py.File(span_avg_file, "r") as f:
        data = f["span_average"][:]  # (m, comps, nx, ny)
    with h5py.File(traj_file, "r") as f:
        u_full = f["jet_amplitude"][:]
        dt_full = f["dt"][:]
        
    m_sa = data.shape[0]
    sa_int = max(len(u_full) // m_sa, 1) if m_sa else 1
    
    snapshots0 = data[0, velocity_components, :, :]
    np.save(output_dir / "init_orig_state.npy", snapshots0)
    
    snapshots = data[:, velocity_components, :, :]
    snapshots_flat = snapshots.reshape(m_sa, -1).T

    # Subsample the control history to align with available snapshots
    u_sub = u_full[::sa_int][:m_sa]
    dt_sub = dt_full[::sa_int][:m_sa]
    u = u_sub[:-1][np.newaxis, :]
    # Ensure snapshots and control inputs have consistent lengths
    snapshots_flat = snapshots_flat[:, : u.shape[1] + 1]
    
    np.save(output_dir / "U.npy", u)
    np.save(output_dir / "dt.npy", dt_sub[0])
    return snapshots_flat, u, dt_sub[0]


def init_snapshot_red(basis: np.ndarray, output_dir: Path) -> np.ndarray:
    init_full = np.load(output_dir / "init_orig_state.npy")
    init_snap_full_flat = init_full.ravel()
    init_snap_red = basis.T @ init_snap_full_flat
    np.save(output_dir / "Init_snap_red.npy", init_snap_red)
    return init_snap_red


def dmdc_sim(A: np.ndarray, B: np.ndarray, U: np.ndarray, dt: float, x0: np.ndarray, basis: np.ndarray, output_h5: Path) -> Path:
    """Simulate the DMDc ROM and write output in span-average format."""
    n_states = A.shape[0]
    n_steps = U.shape[1]
    X = np.zeros((n_states, n_steps + 1))
    X[:, 0] = x0
    for k in range(n_steps):
        X[:, k + 1] = A.dot(X[:, k]) + B.dot(U[:, k])

    x_full_flat = basis.dot(X)
    init_orig_state = np.load(output_h5.parent / "init_orig_state.npy")
    if init_orig_state.ndim == 2:
        n_comp = 1
        nx, ny = init_orig_state.shape
    else:
        n_comp, nx, ny = init_orig_state.shape

    T = n_steps + 1
    data = x_full_flat.T.reshape((T, n_comp, nx, ny), order="C")

    with h5py.File(output_h5, "w") as f:
        f.create_dataset("span_average", data=data)
        f.create_dataset("dt", data=dt)
    return output_h5


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def Erms(snapshots_flat_FOM: np.ndarray, snapshots_flat_ROM: np.ndarray) -> float:
    diff = snapshots_flat_ROM - snapshots_flat_FOM
    mse = np.mean(diff ** 2)
    return float(np.sqrt(mse))


def NRMSE(snapshots_flat_FOM: np.ndarray, snapshots_flat_ROM: np.ndarray) -> float:
    E_rms = Erms(snapshots_flat_FOM, snapshots_flat_ROM)
    denom = np.sqrt(np.mean(snapshots_flat_FOM ** 2))
    return float(E_rms / denom) if denom != 0 else float("nan")


def Efrobenius(snapshots_flat_FOM: np.ndarray, snapshots_flat_ROM: np.ndarray) -> float:
    num = np.linalg.norm(snapshots_flat_ROM - snapshots_flat_FOM, ord="fro")
    den = np.linalg.norm(snapshots_flat_FOM, ord="fro")
    return float(num / den) if den != 0 else float("nan")


def Model_Red_Error(span_avg_file: Path, rom_file: Path) -> np.ndarray:
    with h5py.File(span_avg_file, "r") as f:
        FOMdata = f["span_average"][:, 1:4, :, :]
    with h5py.File(rom_file, "r") as f:
        ROMdata = f["span_average"][:]
    RSEdata = np.abs(FOMdata - ROMdata) / np.abs(FOMdata)
    np.save(rom_file.parent / "SquaredError.npy", RSEdata)
    return RSEdata


def animate_error(err: np.ndarray, comp: int = 0, interval: int = 100, cmap: str = "viridis"):
    nt, ncomp, nx, ny = err.shape
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(err[0, comp].T, origin="lower", cmap=cmap, vmin=0, vmax=np.nanmax(err))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative error")
    title = ax.set_title("t = 0 (frame 0)")

    def update(i):
        im.set_data(err[i, comp])
        title.set_text(f"t = {i} (frame {i})")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=nt, interval=interval, blit=True)
    return ani


# ---------------------------------------------------------------------------
# H2 and Hinf norms
# ---------------------------------------------------------------------------

def load_rom_ss(rom_dir: Path, label: str):
    A = np.load(rom_dir / f"A_red_matrix_{label}.npy")
    B = np.load(rom_dir / f"B_red_matrix_{label}.npy")
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    C = np.eye(n_states)
    D = np.zeros((n_states, n_inputs))
    return ss(A, B, C, D)


def h2_norm_via_lyap(sys):
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    X = solve_continuous_lyapunov(A, -B @ B.T)
    return float(np.sqrt(np.trace(C @ X @ C.T + D @ D.T)))


def hinf_norm_approx(sys, wmin=1e-3, wmax=1e3, npts=500):
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    freqs = np.logspace(np.log10(wmin), np.log10(wmax), npts)
    max_gain = 0.0
    w_at_max = freqs[0]
    I = np.eye(A.shape[0])
    for w in freqs:
        G = C @ np.linalg.inv(1j * w * I - A) @ B + D
        sigma = np.linalg.svd(G, compute_uv=False)[0]
        if sigma > max_gain:
            max_gain = float(sigma)
            w_at_max = float(w)
    return max_gain, w_at_max


def H2_Hinf_Analysis(rom_dir: Path):
    energy_labels = ["99pct", "96pct", "93pct", "90pct"]
    for label in energy_labels:
        sysr = load_rom_ss(rom_dir, label)
        h2_val = h2_norm_via_lyap(sysr)
        hinf_val, w_peak = hinf_norm_approx(sysr, wmin=1e-2, wmax=1e2, npts=800)
        print(f"[{label}]  ‖G‖₂ ≈ {h2_val:.3e}  ;  ‖G‖∞ ≈ {hinf_val:.3e}  @ ω={w_peak:.2f} rad/s")


# ---------------------------------------------------------------------------
# High-level routine
# ---------------------------------------------------------------------------

def _energy_rank(X: np.ndarray, energy_pct: float) -> int:
    """Return the reduced order based on requested energy percentage."""
    if not (0 < energy_pct <= 100.0):
        raise ValueError("energy_pct must be in the interval (0, 100].")
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    cumulative = np.cumsum(s ** 2)
    total = cumulative[-1]
    energy_ratio = cumulative / total
    rank = int(np.searchsorted(energy_ratio, energy_pct / 100.0) + 1)
    return rank

def run_dmdc(sa_path: Path, traj_path: Path, output_dir: Path, energy_pct: float) -> Path:
    """Compute a DMDc ROM from solver output and write results to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_flat, U, dt = load_h5_data(sa_path, traj_path, output_dir)

    rank = _energy_rank(snapshots_flat[:, :-1], energy_pct)
    dmdc = DMDc(svd_rank=rank)
    dmdc.fit(snapshots_flat, U)
    
    # A = dmdc.A
    # New version syntax
    A = getattr(dmdc, "A", dmdc.operator.as_numpy_array)
    basis = dmdc.basis
    B = (
        basis.T @ dmdc.B
        if basis.shape[0] == dmdc.B.shape[0]
        else np.linalg.pinv(basis) @ dmdc.B
    )

    label = f"{int(energy_pct)}pct"
    red_dir = output_dir / "ReducedModels"
    red_dir.mkdir(exist_ok=True)
    np.save(red_dir / f"A_red_matrix_{label}.npy", A)
    np.save(red_dir / f"B_red_matrix_{label}.npy", B)
    np.save(red_dir / f"DMDcBasis_{label}.npy", basis)

    x0 = init_snapshot_red(basis, output_dir)
    rom_file = output_dir / "dmdc_span_averages.h5"
    dmdc_sim(A, B, U, dt, x0, basis, rom_file)
    
    snapshots_flat_ROM, _, _ = load_h5_data(rom_file, traj_path, output_dir,
                                            velocity_components=(0, 1, 2))
    print(f"Erms: {Erms(snapshots_flat, snapshots_flat_ROM)}")
    print(f"NRMSE: {NRMSE(snapshots_flat, snapshots_flat_ROM)}")
    print(f"Efrobenius: {Efrobenius(snapshots_flat, snapshots_flat_ROM)}")
    
    print(f"DMDc results written to {output_dir}")
    return rom_file
