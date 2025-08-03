"""Utilities for Sparse Sensor Placement Optimization using PySensors."""

from pathlib import Path

import h5py
import numpy as np


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
    if num_sensors <= 0:
        raise ValueError("num_sensors must be positive")

    try:
        # from pysensors.optimizers import QR
        from pysensors import SSPOR
        
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
    data = np.concatenate([u.reshape(num_snaps, -1), v.reshape(num_snaps, -1)], axis=1)

    # https://python-sensors.readthedocs.io/en/latest/api/pysensors.reconstruction.html
    model = SSPOR(n_sensors = num_sensors)
    model.fit(data)
    sensor_indices = model.selected_sensors

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "sensors.h5"
    with h5py.File(out_file, "w") as f:
        f.create_dataset("indices", data=sensor_indices)
        f.create_dataset("time", data=time)

    print(f"Sensing results written to {out_file}")
    return out_file





