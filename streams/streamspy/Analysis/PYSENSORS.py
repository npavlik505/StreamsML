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

    with h5py.File(sa_path.with_name("mesh.h5"), "r") as m:
        xg = np.asarray(m["x_grid"][:]).squeeze()
        yg = np.asarray(m["y_grid"][:]).squeeze()


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

    offset = nx * ny
    records = []  # will hold (index, b'U'/b'V', ix, iy, x, y)

    for i in map(int, sensor_indices):
        if i < offset:
            ch, k = b"U", i
        else:
            ch, k = b"V", i - offset
        ix = k // ny
        iy = k % ny
        records.append((i, ch, int(ix), int(iy), float(xg[ix]), float(yg[iy])))

    # Pretty print
    print(" index  var  ix  iy      x_phys         y_phys")
    for i, ch, ix, iy, x, y in records:
        print(f"{i:6d}  {ch.decode():>1s}  {ix:3d} {iy:3d}  {x:12.6g}  {y:12.6g}")

    # Save to HDF5
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "sensors.h5"

    # Compound dtype: fixed-length 1-byte string for var, plus ints/floats
    table_dt = np.dtype([
        ("index", "<i4"),
        ("var",   "S1"),   # b"U" or b"V"
        ("ix",    "<i4"),
        ("iy",    "<i4"),
        ("x",     "<f8"),
        ("y",     "<f8"),
    ])
    table = np.array(records, dtype=table_dt)

    with h5py.File(out_file, "w") as f:
        # raw feature indices exactly as returned by PySensors
        f.create_dataset("indices_raw", data=np.asarray(sensor_indices, dtype=np.int32))

        # convenient, self-describing table
        f.create_dataset("sensors_table", data=table)

        # metadata for reproducibility
        f.attrs["nx"] = int(nx)
        f.attrs["ny"] = int(ny)
        f.attrs["offset"] = int(offset)       # nx*ny
        f.attrs["flatten_order"] = "C"        # how you flattened (row-major)
        f.attrs["channel_legend"] = "var: b'U'->U, b'V'->V"

    print(f"Sensing results written to {out_file}")
    return out_file
