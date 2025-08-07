"""Utilities for animating span-averaged data over time."""

from pathlib import Path
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter

VARIABLE_MAP = {"rho": 0, "u": 1, "v": 2, "w": 3, "E": 4}


def run_animation(sa_path: Path, output_dir: Path, variable: str) -> None:
    """Create an animation for ``variable`` across all snapshots."""

    variable = variable.strip()
    if variable not in VARIABLE_MAP:
        raise ValueError(
            f"Unknown variable '{variable}'. Expected one of {list(VARIABLE_MAP)}."
        )

    with h5py.File(sa_path, "r") as sa:
        data = sa["span_average"]
        frames = data[:, VARIABLE_MAP[variable], :, :]
        colorbarmax = float(np.nanmax(frames))

    mesh_path = sa_path.parent / "mesh.h5"
    with h5py.File(mesh_path, "r") as mesh:
        x = mesh["x_grid"][0, :]
        y = mesh["y_grid"][0, :]
        X, Y = np.meshgrid(x, y)

    norm = mpl.colors.Normalize(vmin = 0.0, vmax = colorbarmax)
    levels = np.linspace(0.0, colorbarmax, 40)  
    fig, ax = plt.subplots()

    cf = ax.contourf(X, Y, frames[0].T, levels=levels, cmap="viridis", norm=norm, extend='max')
    ax.set_aspect("equal")
    cbar = fig.colorbar(cf, ax=ax)

    def update(i):
        nonlocal cf 
        for coll in cf.collections:
            coll.remove()
        # redraw with same norm/levels
        cf = ax.contourf(X, Y, frames[i].T, levels=levels, cmap="viridis", norm=norm, extend='max')
        # keep colorbar scale fixed, just relink to new mappable
        cbar.update_normal(cf)
        return cf.collections

    ani = FuncAnimation(fig, update, frames=range(frames.shape[0]), blit=False)
    os.makedirs(output_dir, exist_ok=True)
    fname = output_dir / f"anim_{variable}.gif"
    ani.save(fname, writer=PillowWriter(fps=60))
    plt.close(fig)
