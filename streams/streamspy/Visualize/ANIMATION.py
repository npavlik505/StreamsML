"""Utilities for animating span-averaged data over time."""

from pathlib import Path
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter

VARIABLE_MAP = {"rho": 0, "u": 1, "v": 2, "w": 3, "E": 4}


def run_animation(sa_path: Path, output_dir: Path, variable: str, domain: tuple[float, float]) -> None:
    """Create an animation for ``variable`` across all snapshots."""

    variable = variable.strip()
    if variable not in VARIABLE_MAP:
        raise ValueError(
            f"Unknown variable '{variable}'. Expected one of {list(VARIABLE_MAP)}."
        )

    with h5py.File(sa_path, "r") as sa:
        data = sa["span_average"]
        frames = data[:, VARIABLE_MAP[variable], :, :]
        colorbarmin = float(np.nanmin(frames))
        colorbarmax = float(np.nanmax(frames))

    mesh_path = sa_path.parent / "mesh.h5"
    with h5py.File(mesh_path, "r") as mesh:
        x = mesh["x_grid"][0, :]
        y = mesh["y_grid"][0, :]
        lx, ly = domain
        x_end = np.searchsorted(x, lx, side="right")
        y_end = np.searchsorted(y, ly, side="right")
        x = x[:x_end]
        y = y[:y_end]
        frames = frames[:, :x_end, :y_end]
        X, Y = np.meshgrid(x, y)

    norm = mpl.colors.Normalize(vmin = colorbarmin, vmax = colorbarmax)
    levels = np.linspace(colorbarmin, colorbarmax, 40)  
    fig, ax = plt.subplots()

    cf = ax.contourf(X, Y, frames[0].T, levels=levels, cmap="viridis", norm=norm, extend='max')
    ax.set_aspect("equal")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.4, fraction=0.1)

    def update(i):
        nonlocal cf
        cf.remove()                                       
        cf = ax.contourf(X, Y, frames[i].T,               
                         levels=levels, cmap="viridis", norm=norm, extend='max')
        cbar.update_normal(cf)                             
        return [cf]                                       


    ani = FuncAnimation(fig, update, frames=range(frames.shape[0]), blit=False)
    os.makedirs(output_dir, exist_ok=True)
    fname = output_dir / f"anim_{variable}.gif"
    ani.save(fname, writer=PillowWriter(fps=60))
    plt.close(fig)
