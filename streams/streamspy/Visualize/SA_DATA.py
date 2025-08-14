"""Utilities for plotting span-averaged data."""

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def run_sadata(sa_path: Path, output_dir: Path) -> None:
    """Plot shear stress, energy and dissipation rate against time.

    Parameters
    ----------
    sa_path:
        Path to ``span_averages.h5`` file.
    output_dir:
        Directory where the figures should be written.
    """

    with h5py.File(sa_path, "r") as sa:
        time = sa["time"][:]
        shear_stress = sa["shear_stress"][:]
        energy = sa["energy"][:]
        dissipation_rate = sa["dissipation_rate"][:]

    # Average the shear stress across the spatial dimension to obtain a
    # single value per time step.
    shear_stress_mean = np.mean(shear_stress, axis=1)

    # Create a simple three panel figure for the requested quantities.
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    axes[0].plot(time, shear_stress_mean)
    axes[0].set_ylabel("Shear Stress")

    axes[1].plot(time, energy)
    axes[1].set_ylabel("Energy")

    axes[2].plot(time, dissipation_rate)
    axes[2].set_ylabel("Dissipation Rate")
    axes[2].set_xlabel("Time")

    fig.tight_layout()

    out_file = output_dir / "sa_data.png"
    fig.savefig(out_file)
    plt.close(fig)

        
  
