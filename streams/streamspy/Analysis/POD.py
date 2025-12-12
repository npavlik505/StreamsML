"""Proper Orthogonal Decomposition utilities."""

from pathlib import Path
import numpy as np
import h5py

# Compute POD modes from ``span_averages.h5`` and write pod_results.h5 to ``output_dir``.
# This iteration now accounts for the non-uniform grid and compressible nature of STREAmS data by using a weighted inner product and Favre averaging. 
# The weights are given by the product of the local cell area and time-averaged density, so the resulting modes are orthonormal with respect to this weighting
def run_pod(sa_path: Path, dxdy_path: Path, output_dir: Path) -> Path:

    with h5py.File(sa_path, "r") as f:
        sa = f["span_average"][:]
        time = f["time"][:]

    # sa layout: (num_snapshots, 5, nx, ny)
    rho = sa[:, 0, :, :]
    u = sa[:, 1, :, :]
    v = sa[:, 2, :, :]

    num_snaps, nx, ny = u.shape
    
    
    # grid weights
    dxg = np.loadtxt(dxdy_path / "dxg.dat")
    dyg = np.loadtxt(dxdy_path / "dyg.dat")

    xg = dxg[:, 0]
    yg = np.flip(dyg[:, 0])

    dx = np.diff(xg)
    dx = np.append(dx, dx[-1])
    dy = np.abs(np.diff(yg))
    dy = np.append(dy, dy[-1])

    area = np.outer(dx, dy)  # shape (nx, ny)

    # Time-averaged density for weighting
    rho_bar = np.mean(rho, axis=0)
    weights = (rho_bar * area).reshape(-1)
    weights_full = np.concatenate([weights, weights])

    # Favre mean subtraction
    rho_sum = np.sum(rho, axis=0)
    # Prevent division by zero
    rho_sum = np.where(rho_sum == 0.0, 1.0, rho_sum)
    u_favre = np.sum(rho * u, axis=0) / rho_sum
    v_favre = np.sum(rho * v, axis=0) / rho_sum

    u_fluc = u - u_favre
    v_fluc = v - v_favre
    
    data = np.concatenate(
        [u_fluc.reshape(num_snaps, -1), v_fluc.reshape(num_snaps, -1)], axis=1
    ).T

    # Weighted SVD
    sqrt_w = np.sqrt(weights_full)
    weighted_data = sqrt_w[:, None] * data

    modes_w, svals, vt = np.linalg.svd(weighted_data, full_matrices=False)
    modes = modes_w / sqrt_w[:, None]
    coeffs = vt.T * svals
    
    mean_field = np.concatenate([u_favre.reshape(-1), v_favre.reshape(-1)])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "pod_results.h5"
    with h5py.File(out_file, "w") as f:
        f.create_dataset("mean_field", data=mean_field)
        f.create_dataset("modes", data=modes)
        f.create_dataset("singular_values", data=svals)
        f.create_dataset("coefficients", data=coeffs)
        f.create_dataset("time", data=time)

    print(f"POD results written to {out_file}")
    return out_file
    


