# TODO: Does not yet account for non-uniform grid and compressible flow. Reference POD code.  

"""Dynamic Mode Decomposition utilities."""

from pathlib import Path
import numpy as np
import h5py

# DMD modes from span_averages.h5 saved in dmd_results.h5
# TODO Update with local system's DMD script 
def run_dmd(sa_path: Path, output_dir: Path) -> Path:

    with h5py.File(sa_path, "r") as f:
        sa = f["span_average"][:]  # shape (num_snaps, 5, nx, ny)
        time = f["time"][:]

    num_snaps = sa.shape[0]

    # flatten all flowfield components for each snapshot
    data = sa.reshape(num_snaps, -1).T  # shape (state_dim, num_snaps)

    if num_snaps < 2:
        raise ValueError("Need at least two snapshots for DMD")

    X1 = data[:, :-1]
    X2 = data[:, 1:]

    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    A_tilde = U.T.conj() @ X2 @ Vh.T.conj() / s
    eigvals, W = np.linalg.eig(A_tilde)
    Phi = X2 @ Vh.T.conj() / s @ W

    b = np.linalg.lstsq(Phi, data[:, 0], rcond=None)[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "dmd_results.h5"
    with h5py.File(out_file, "w") as f:
        f.create_dataset("modes", data=Phi)
        f.create_dataset("eigenvalues", data=eigvals)
        f.create_dataset("amplitudes", data=b)
        f.create_dataset("time", data=time)

    print(f"DMD results written to {out_file}")
    return out_file
