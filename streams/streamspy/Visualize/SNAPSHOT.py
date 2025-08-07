"""Utilities for generating a single snapshot from span-averaged data."""

from pathlib import Path
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
        frames = data[:, VARIABLE_MAP[variable], :, :]
        if snapshot < 0 or snapshot >= frames.shape[0]:
            raise ValueError(f"Snapshot index out of range. Must be between 0 and {frames.shape[0] - 1}.")
        field = frames[snapshot]
        colorbarmax = float(np.nanmax(frames))

    mesh_path = sa_path.parent / "mesh.h5"
    with h5py.File(mesh_path, "r") as mesh:
        x = mesh["x_grid"][0, :]
        y = mesh["y_grid"][0, :]
        X, Y = np.meshgrid(x, y)

    norm = mpl.colors.Normalize(vmin = 0.0, vmax = colorbarmax)
    levels = np.linspace(0.0, colorbarmax, 40)
    fig, ax = plt.subplots()
    cf = ax.contourf(X, Y, field.T, levels=levels, cmap="viridis", norm=norm, extend = 'max')
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax)

    os.makedirs(output_dir, exist_ok=True)
    fname = output_dir / f"snap_{variable}_{snapshot}.png"
    fig.savefig(fname)
    plt.close(fig)
