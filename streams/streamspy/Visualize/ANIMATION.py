"""Utilities for animating span-averaged data over time."""

from pathlib import Path
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
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

    mesh_path = sa_path.parent / "mesh.h5"
    with h5py.File(mesh_path, "r") as mesh:
        x = mesh["x_grid"][0, :]
        y = mesh["y_grid"][0, :]
        X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()

    def update(i):
        ax.clear()
        cf = ax.contourf(X, Y, frames[i].T, levels=40, cmap="viridis")
        ax.set_aspect("equal")
        return cf.collections

    ani = FuncAnimation(fig, update, frames=range(frames.shape[0]), blit=False)
    os.makedirs(output_dir, exist_ok=True)
    fname = output_dir / f"anim_{variable}.gif"
    ani.save(fname, writer=PillowWriter(fps=10))
    plt.close(fig)
