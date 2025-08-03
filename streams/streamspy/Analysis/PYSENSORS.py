"""Utilities for Sparse Sensor Placement Optimization using PySensors."""
import numpy as np
import argparse
import json
import sys
import h5py
from pathlib import Path

# Script imports
from config import Config

def run_pysensors(sa_path: Path, output_dir: Path, num_sensors: int = 10) -> Path:
    """Apply PySensors SSPOR on velocity fields in ``sa_path``.

    Parameters
    ----------
    sa_path : Path
        Path to ``span_averages.h5``.
    output_dir : Path
        Directory where the sensing results will be stored.
    num_sensors : int, optional
        Number of sensors to select, by default ``10``.

    Returns
    -------
    Path
        Path to the created ``sensors.h5`` file.
    """

    try:
        from pysensors.optimization import approximate_sspor
    except Exception as exc:  # pragma: no cover - package may not be installed
        raise ImportError(
            "The `pysensors` package is required to run Sparse Sensor Placement "
            "Optimization. Install it with `pip install pysensors`."
        ) from exc

    with h5py.File(sa_path, "r") as f:
        sa = f["span_average"][:]
        time = f["time"][:]

    # Extract u and v components and form data matrix
    u = sa[:, 1, :, :]
    v = sa[:, 2, :, :]

    num_snaps, nx, ny = u.shape
    data = np.concatenate(
        [u.reshape(num_snaps, -1), v.reshape(num_snaps, -1)], axis=1
    ).T

    sensor_indices, _ = approximate_sspor(data, num_sensors)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "sensors.h5"
    with h5py.File(out_file, "w") as f:
        f.create_dataset("indices", data=sensor_indices)
        f.create_dataset("time", data=time)

    print(f"Sensing results written to {out_file}")
    return out_file





