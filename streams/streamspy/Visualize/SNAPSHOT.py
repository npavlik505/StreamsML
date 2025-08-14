"""Utilities for generating a single snapshot from span-averaged data."""

from pathlib import Path
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

VARIABLE_MAP = {"rho": 0, "u": 1, "v": 2, "w": 3, "E": 4}


def run_snapshot(sa_path: Path, output_dir: Path, variable: str, snapshot: int, domain: tuple[float, float]) -> None:
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
        colorbarmin = float(np.nanmin(frames))
        colorbarmax = float(np.nanmax(frames))

    mesh_path = sa_path.parent / "mesh.h5"
    with h5py.File(mesh_path, "r") as mesh:
        x = mesh["x_grid"][0, :]
        y = mesh["y_grid"][0, :]
        lx, ly = domain # TEST
        x_end = np.searchsorted(x, lx, side="right") # TEST
        y_end = np.searchsorted(y, ly, side="right") # TEST
        x = x[:x_end] # TEST
        y = y[:y_end] # TEST
        field = field[:x_end, :y_end] # TEST
        X, Y = np.meshgrid(x, y)

    norm = mpl.colors.Normalize(vmin = colorbarmin, vmax = colorbarmax)
    levels = np.linspace(colorbarmin, colorbarmax, 40)
    fig, ax = plt.subplots()
    cf = ax.contourf(X, Y, field.T, levels=levels, cmap="viridis", norm=norm, extend = 'max')
    ax.axis("scaled")
    fig.colorbar(cf, ax=ax, shrink=0.4, fraction=0.1)
    
    
    os.makedirs(output_dir, exist_ok=True)
    fname = output_dir / f"snap_{variable}_{snapshot}.png"
    fig.savefig(fname)
    plt.close(fig)
