"""Utilities for generating a single snapshot from span-averaged data."""

from pathlib import Path
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

VARIABLE_MAP = {"rho": 0, "u": 1, "v": 2, "w": 3, "E": 4}


def run_snapshot(sa_path: Path, output_dir: Path, variable: str, snapshot: int) -> None:
    """Create a contour plot for ``variable`` at ``snapshot``.

    Parameters
    ----------
    sa_path:
        Path to ``span_averages.h5`` file.
    output_dir:
        Directory where the figure should be written.
    variable:
        Field to visualise (``rho``, ``u``, ``v``, ``w`` or ``E``).
    snapshot:
        Index of the snapshot to plot.
    """

    variable = variable.strip()
    if variable not in VARIABLE_MAP:
        raise ValueError(
            f"Unknown variable '{variable}'. Expected one of {list(VARIABLE_MAP)}."
        )

    with h5py.File(sa_path, "r") as sa:
        data = sa["span_average"]
        if snapshot < 0 or snapshot >= data.shape[0]:
            raise ValueError(
                f"Snapshot index out of range. Must be between 0 and {data.shape[0] - 1}."
            )
        field = data[snapshot, VARIABLE_MAP[variable], :, :]

    mesh_path = sa_path.parent / "mesh.h5"
    with h5py.File(mesh_path, "r") as mesh:
        x = mesh["x_grid"][0, :]
        y = mesh["y_grid"][0, :]
        X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    cf = ax.contourf(X, Y, field.T, levels=40, cmap="viridis")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax)

    os.makedirs(output_dir, exist_ok=True)
    fname = output_dir / f"snap_{variable}_{snapshot}.png"
    fig.savefig(fname)
    plt.close(fig)
